"""One explicit training loop, shared by baseline, resume, and freeze branches."""

import hashlib
import json
import platform
import subprocess
import time
import tomllib
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import latest_checkpoint, load_checkpoint, save_checkpoint
from .model import ModelConfig, Transformer
from .processes import canonical_process


def read_config(path: Path) -> dict:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def choose_device(device: str = "auto") -> torch.device:
    # CPU is the reference path on Mac; CUDA is selected on a GPU host.
    return torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )


class WindowedPlateau:
    """Source recipe: compare the last ten losses every hundred updates."""

    def __init__(self, optimizer):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.7,
            patience=100,
            threshold=1e-4,
            cooldown=200,
            min_lr=1e-7,
        )
        self.losses = deque(maxlen=10)
        self.steps = 0

    def step(self, loss: float) -> None:
        self.losses.append(loss)
        self.steps += 1
        if len(self.losses) == 10 and self.steps % 100 == 0:
            self.scheduler.step(sum(self.losses) / 10)

    def state_dict(self) -> dict:
        return dict(
            losses=list(self.losses), steps=self.steps, scheduler=self.scheduler.state_dict()
        )

    def load_state_dict(self, state: dict) -> None:
        self.losses = deque(state["losses"], maxlen=10)
        self.steps = state["steps"]
        self.scheduler.load_state_dict(state["scheduler"])


def frozen_parameter(name: str, freeze: str) -> bool:
    return (
        (freeze == "attention" and ".attn." in name)
        or (freeze == "mlp" and ".mlp." in name)
        or (freeze == "embeddings" and name in ("embed.W_E", "pos_embed.W_pos"))
    )


def _validate(config: dict) -> dict:
    config = dict(config)
    required = {
        "name",
        "steps",
        "batch_size",
        "sequence_length",
        "model_seed",
        "data_seed",
        "learning_rate",
        "scheduler",
        "checkpoint_every",
        "log_every",
    }
    if required - config.keys() or config.keys() - required - {"model"}:
        raise ValueError(f"Config keys: required {sorted(required)}, optional 'model'")
    for key in ("steps", "batch_size", "sequence_length", "checkpoint_every", "log_every"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["scheduler"] not in ("constant", "windowed_plateau"):
        raise ValueError("scheduler must be constant or windowed_plateau")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    model = ModelConfig(**config.get("model", {}))
    if model.d_vocab != 433 or config["sequence_length"] > model.n_ctx:
        raise ValueError("Canonical generator requires d_vocab=433 and sequence_length <= n_ctx")
    config["model"] = asdict(model)
    return config


def _metadata(device: torch.device, freeze: str, parent: Path | None) -> dict:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        digest.update(str(source.relative_to(package)).encode())
        digest.update(source.read_bytes())
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True
        ).stdout.strip()
        or None
    )
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        device=str(device),
        cuda=torch.version.cuda,
        source_sha256=digest.hexdigest(),
        git_commit=commit,
        freeze=freeze,
        parent_checkpoint=str(parent) if parent else None,
        generator="independent 3 Mess3 + 2 Tom Quantum; NumPy stream, not upstream JAX",
        data_seed_rule="SeedSequence([data_seed, global_update]).generate_state(1)[0]",
    )


def train(
    config: dict,
    run_dir: Path,
    device: str = "auto",
    resume: bool = False,
    until: int | None = None,
    parent_checkpoint: Path | None = None,
    freeze: str = "none",
) -> Path:
    """Train to config.steps; until stops early for a resumable partial run.

    A branch loads its parent's weights, Adam state, and scheduler by default.
    Batch/length/seed/scheduler/LR/model must agree with the parent. Branches
    change only which weights may adapt; frozen operations still execute.
    """
    config = _validate(config)
    run_dir, device = Path(run_dir), choose_device(device)
    if freeze not in ("none", "attention", "mlp", "embeddings"):
        raise ValueError("Unknown freeze family")
    if resume and parent_checkpoint:
        raise ValueError("Choose resume or a new parent branch")
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {run_dir}; use --resume or a new path")
    if resume and not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"No run to resume: {run_dir}")
    if resume and json.loads((run_dir / "config.json").read_text()) != config:
        raise ValueError("Resume requires the saved config; use a separate branch for changes")
    saved = (
        load_checkpoint(latest_checkpoint(run_dir))
        if resume
        else (load_checkpoint(parent_checkpoint) if parent_checkpoint else None)
    )
    if saved:
        for key in config.keys() - {"steps", "name", "checkpoint_every", "log_every"}:
            if saved["config"][key] != config[key]:
                raise ValueError(f"Parent differs in {key}; this branch only changes plasticity")
        if resume:
            freeze = saved["freeze"]
    model = Transformer(ModelConfig(**config["model"]), config["model_seed"]).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not frozen_parameter(name, freeze))
    # Keep every parameter in the group so parent Adam state loads unchanged.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    scheduler = WindowedPlateau(optimizer) if config["scheduler"] == "windowed_plateau" else None
    start = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler:
            scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start = saved["step"]
    stop = min(config["steps"], until if until is not None else config["steps"])
    if stop < start:
        raise ValueError("Requested stopping update precedes the saved checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "metadata.json", _metadata(device, freeze, parent_checkpoint))
    process = canonical_process()

    def checkpoint(step):
        path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
        save_checkpoint(
            path,
            dict(
                format_version=1,
                config=config,
                step=step,
                examples_seen=step * config["batch_size"],
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler else None,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                freeze=freeze,
            ),
        )
        return path

    if not resume:
        checkpoint(start)
    started = time.monotonic()
    model.train()
    for step in range(start + 1, stop + 1):
        data_seed = int(np.random.SeedSequence([config["data_seed"], step]).generate_state(1)[0])
        batch = process.sample(config["batch_size"], config["sequence_length"], data_seed, False)
        inputs = torch.as_tensor(batch.tokens, dtype=torch.long, device=device)
        targets = torch.as_tensor(batch.targets, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {step}")
        loss.backward()
        optimizer.step()
        value = loss.item()
        if scheduler:
            scheduler.step(value)
        if step % config["log_every"] == 0 or step == stop:
            row = dict(
                step=step,
                examples_seen=step * config["batch_size"],
                loss=value,
                learning_rate=optimizer.param_groups[0]["lr"],
                invocation_seconds=time.monotonic() - started,
            )
            with (run_dir / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        early = step <= 10 or (step <= 500 and step % 10 == 0) or (step <= 1000 and step % 100 == 0)
        if early or step % config["checkpoint_every"] == 0 or step == stop:
            checkpoint(step)
    return run_dir

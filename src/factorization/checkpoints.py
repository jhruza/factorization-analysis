"""Atomic, portable training snapshots; no external tracking service."""

from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    """Replace only after the complete file has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def latest_checkpoint(run_dir: Path) -> Path:
    paths = sorted((Path(run_dir) / "checkpoints").glob("step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return paths[-1]


def load_checkpoint(path: Path) -> dict:
    """Load this project's own trusted checkpoints, initially onto CPU."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("Not a version-1 factorization-lab training checkpoint")
    return state

"""Descriptive vary-one PCA following factored-reps' orthogonality analysis.

Each factor varies in natural trajectories while the other factors repeat a
fixed background. Centering is across samples separately at each position and
background. Samples use this project's NumPy stream, not the source JAX stream.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ..checkpoints import load_checkpoint
from ..model import ModelConfig, Transformer
from ..processes import canonical_process
from ..train import choose_device, write_json
from .development import capture


def center_positions(activations):
    """Drop BOS and center each position across a batch, returning rows."""
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim != 3 or a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError("activations need at least two samples and one non-BOS position")
    a = a[:, 1:]
    return (a - a.mean(axis=0, keepdims=True)).reshape(-1, a.shape[-1])


def overlap_curve(first, second, max_components):
    """Paper's normalized squared Frobenius overlap of row-wise PCA bases."""
    a, b = np.asarray(first), np.asarray(second)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or max_components < 1:
        raise ValueError("incompatible bases or max_components")
    return [
        float(np.sum((a[:k] @ b[:k].T) ** 2) / k)
        for k in range(1, min(max_components, len(a), len(b)) + 1)
    ]


def _pca(rows):
    _, s, components = np.linalg.svd(rows, full_matrices=False)
    energy = s * s
    if energy.sum() == 0:
        return None, None, np.empty((0, rows.shape[1]))
    cev = np.cumsum(energy) / energy.sum()
    return cev.tolist(), int(np.searchsorted(cev, 0.95) + 1), components


def _vary_tokens(process, seq_len, factor, background, count):
    # Input samples already start with BOS; only ordinary positions are decoded.
    frozen = process.sample(1, seq_len, 730000 + factor * 100 + background, False).tokens
    varied = process.sample(count, seq_len, 731000 + factor * 100 + background, False).tokens
    fixed_factors = process.decode(frozen[:, 1:])
    varying_factors = process.decode(varied[:, 1:])
    factors = np.repeat(fixed_factors, count, axis=0)
    factors[..., factor] = varying_factors[..., factor]
    return np.concatenate([varied[:, :1], process.encode(factors)], axis=1)


def analyze_paper(
    run_dir,
    checkpoints=None,
    n_sequences=1024,
    backgrounds=4,
    batch_per_background=128,
    max_components=10,
    batch_size=128,
    device="auto",
    interface="L4_mlp",
    output=None,
):
    """Compute descriptive PCA for fixed samples at each saved checkpoint."""
    if min(n_sequences, backgrounds, batch_per_background, max_components, batch_size) < 1:
        raise ValueError("sample counts and max_components must be positive")
    if n_sequences < 2 or batch_per_background < 2:
        raise ValueError("centering needs at least two sequences per group")
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / "analyses" / "paper"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Analysis output already exists: {output}")
    config = json.loads((run_dir / "config.json").read_text())
    length = config["model"]["n_ctx"]
    if length < 2:
        raise ValueError("analysis requires at least one non-BOS position")
    model = Transformer(ModelConfig(**config["model"]), config["model_seed"]).to(
        choose_device(device)
    )
    valid = {"input", "final"} | {
        f"L{i}_{kind}" for i in range(1, model.config.n_layers + 1) for kind in ("attn", "mlp")
    }
    if interface not in valid:
        raise ValueError(f"Unknown interface: {interface}")
    paths = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if checkpoints is not None:
        by_step = {int(p.stem.split("_")[1]): p for p in paths}
        missing = set(checkpoints) - by_step.keys()
        if missing:
            raise FileNotFoundError(f"Missing checkpoint updates: {sorted(missing)}")
        paths = [by_step[s] for s in sorted(set(checkpoints))]
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    process = canonical_process()
    combined = process.sample(n_sequences, length, 720001, False).tokens
    varying = [
        [
            _vary_tokens(process, length, factor, bg, batch_per_background)
            for bg in range(backgrounds)
        ]
        for factor in range(process.num_factors)
    ]
    result = {
        "format_version": 1,
        "run": str(run_dir),
        "protocol": {
            "interface": interface,
            "input_length": length,
            "n_sequences": n_sequences,
            "backgrounds": backgrounds,
            "batch_per_background": batch_per_background,
            "max_components": max_components,
            "combined_seed": 720001,
            "background_seed_rule": "730000 + factor * 100 + background",
            "vary_seed_rule": "731000 + factor * 100 + background",
            "sampling": "independent NumPy process; not the original JAX random stream",
            "centering": "exclude BOS; center each position separately within each background",
            "interpretation": "descriptive PCA, one trained model; backgrounds are not training replications",
        },
        "rows": [],
    }
    for path in paths:
        saved = load_checkpoint(path)
        model.load_state_dict(saved["model"])
        combined_rows = center_positions(capture(model, combined, batch_size)[interface])
        combined_cev, k95, _ = _pca(combined_rows)
        factors = [
            np.concatenate(
                [center_positions(capture(model, group, batch_size)[interface]) for group in groups]
            )
            for groups in varying
        ]
        pc = [_pca(factor) for factor in factors]
        union_cev, _, _ = _pca(np.concatenate(factors))
        pairs = {
            f"F{i},F{j}": overlap_curve(pc[i][2], pc[j][2], max_components)
            for i, j in combinations(range(process.num_factors), 2)
        }
        result["rows"].append(
            {
                "step": saved["step"],
                "combined_cev": combined_cev,
                "k95": k95,
                "factor_cevs": [x[0] for x in pc],
                "factor_k95": [x[1] for x in pc],
                "union_cev": union_cev,
                "pair_overlap": pairs,
            }
        )
    write_json(output / "results.json", result)
    _plot(result, output / "figures")
    return result


def _plot(result, directory):
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in rows:
        if row["combined_cev"] is not None:
            axes[0].plot(
                np.arange(1, len(row["combined_cev"]) + 1),
                row["combined_cev"],
                label=str(row["step"]),
            )
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel="Principal components", ylabel="Cumulative explained variance", title="Combined PCA"
    )
    axes[0].legend(title="Update")
    axes[1].plot([r["step"] for r in rows], [r["k95"] for r in rows], marker="o")
    axes[1].set(xlabel="Update", ylabel="Components for 95% variance", title="Dimensionality")
    fig.tight_layout()
    fig.savefig(directory / "paper_pca.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        curves = list(row["pair_overlap"].values())
        if curves and all(curves):
            mean = np.mean(curves, axis=0)
            ax.plot(np.arange(1, len(mean) + 1), mean, label=str(row["step"]))
    ax.set(xlabel="Components per factor", ylabel="Mean normalized overlap", title="Vary-one PCA")
    ax.legend(title="Update")
    fig.tight_layout()
    fig.savefig(directory / "paper_overlap.png", dpi=150)
    plt.close(fig)

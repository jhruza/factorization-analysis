"""Small, descriptive plots for saved developmental analysis results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _step_ticks(steps):
    if len(steps) <= 7:
        return list(range(len(steps))), steps
    indices = np.unique(np.linspace(0, len(steps) - 1, 7, dtype=int))
    return indices.tolist(), [steps[i] for i in indices]


def _plot_trajectory(rows, interfaces, steps, out):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, interface in enumerate(interfaces):
        series = [r for r in rows if r["interface"] == interface]
        color = colors(index % 10)
        axes[0].plot(
            [r["step"] for r in series],
            [np.nan if r["pca"]["k95"] is None else r["pca"]["k95"] for r in series],
            marker=".",
            label=interface,
            color=color,
        )
        axes[1].plot(
            [r["step"] for r in series],
            [
                np.nan if r["natural_encoding_r2"] is None else r["natural_encoding_r2"]
                for r in series
            ],
            marker=".",
            label=interface,
            color=color,
        )
    axes[0].set_ylabel("PCA dimensions for 95% variance")
    axes[1].set_ylabel("Natural feature encoding $R^2$")
    axes[1].set_xlabel("Training update")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Representation measures over training (one training seed)")
    fig.savefig(out / "trajectory.png", dpi=150)
    plt.close(fig)


def _plot_cev(rows, final_interface, steps, out):
    selected = [r for r in rows if r["interface"] == final_interface]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    colors = plt.get_cmap("viridis")
    vmax = max(steps) if steps and max(steps) else 1
    for row in selected:
        cumulative = row["pca"].get("cumulative_variance")
        if cumulative is not None:
            x = np.arange(1, len(cumulative) + 1)
            ax.plot(x, cumulative, color=colors(row["step"] / vmax), alpha=0.7, linewidth=1)
    ax.axhline(0.95, color="0.35", linestyle="--", linewidth=1, label="95%")
    ax.set(
        xlabel="Principal component",
        ylabel="Cumulative explained variance",
        ylim=(0, 1.02),
        title=f"PCA spectrum at {final_interface} (one training seed)",
    )
    if selected:
        scalar = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(0, vmax))
        scalar.set_array([])
        fig.colorbar(scalar, ax=ax, label="Training update")
    ax.legend(frameon=False)
    fig.savefig(out / "cev.png", dpi=150)
    plt.close(fig)


def _plot_development(rows, interfaces, steps, scope, out):
    groups = (("Factors 0–2", (0, 1, 2)), ("Factors 3–4", (3, 4)))
    fig, axes = plt.subplots(
        1, 2, figsize=(10, max(3.5, 0.38 * len(interfaces))), sharey=True, constrained_layout=True
    )
    mode = "non_bos" if scope == "non_bos" else "last"
    measured = False
    for ax, (title, factors) in zip(axes, groups):
        matrix = np.full((len(interfaces), len(steps)), np.nan)
        for yi, interface in enumerate(interfaces):
            for xi, step in enumerate(steps):
                row = next(
                    (r for r in rows if r["interface"] == interface and r["step"] == step), None
                )
                if row:
                    ratios = [
                        row.get("invariance", [])[i].get(mode, {}).get("ratio")
                        for i in factors
                        if i < len(row.get("invariance", []))
                    ]
                    finite = [x for x in ratios if x is not None and np.isfinite(x)]
                    if finite:
                        matrix[yi, xi] = float(np.mean(finite))
                        measured = True
        image = ax.imshow(
            matrix,
            vmin=0,
            vmax=2,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            origin="upper",
        )
        tick_positions, tick_labels = _step_ticks(steps)
        ax.set(
            title=title,
            xlabel="Training update",
            xticks=tick_positions,
            xticklabels=tick_labels,
            yticks=range(len(interfaces)),
            yticklabels=interfaces,
        )
        ax.tick_params(axis="x", labelrotation=30)
    if not measured:
        axes[0].text(
            0.5,
            0.5,
            "Consistency measurements unavailable\n(run with interventions enabled)",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            color="white",
        )
    axes[0].set_ylabel(f"Model interface · {mode} positions")
    fig.colorbar(image, ax=axes, label="Context dependence (0 = invariant)")
    fig.suptitle("Factor-change consistency across contexts (one training seed)")
    fig.savefig(out / "development.png", dpi=150)
    plt.close(fig)


def _plot_edit_recovery(rows, final_interface, steps, out):
    selected = [r for r in rows if r["interface"] == final_interface]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    measured = False
    for factor in range(5):
        xs, ys = [], []
        for row in selected:
            edits = row.get("interventions") or []
            if factor < len(edits) and edits[factor].get("recovery") is not None:
                xs.append(row["step"])
                ys.append(edits[factor]["recovery"])
                measured = True
        if xs:
            ax.plot(xs, ys, marker=".", color=colors(factor), label=f"Factor {factor}")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set(
        xlabel="Training update",
        ylabel="Whole-history edit recovery",
        title=f"Factor edit recovery at {final_interface} (one training seed)",
    )
    if not measured:
        ax.text(
            0.5,
            0.5,
            "Edit measurements unavailable\n(run with interventions enabled)",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out / "edit_recovery.png", dpi=150)
    plt.close(fig)


def _plot_geometry(rows, interfaces, final_interface, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for interface in interfaces:
        selected = [row for row in rows if row["interface"] == interface]
        medians = []
        for row in selected:
            values = [
                row["overlap"][i][j]
                for i in range(5)
                for j in range(i + 1, 5)
                if row["overlap"][i][j] is not None
            ]
            medians.append(np.median(values) if values else np.nan)
        axes[0].plot([row["step"] for row in selected], medians, label=interface)
    axes[0].set(
        xlabel="Training update",
        ylabel="Median pairwise maximum principal cosine",
        ylim=(0, 1.03),
        title="Factor subspace overlap",
    )
    axes[0].legend(fontsize=7, ncol=2)
    latest = max((r for r in rows if r["interface"] == final_interface), key=lambda r: r["step"])
    matrix = np.array([[np.nan if x is None else x for x in row] for row in latest["overlap"]])
    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axes[1].set(
        xticks=range(5),
        yticks=range(5),
        xlabel="Factor",
        ylabel="Factor",
        title=f"{final_interface}, update {latest['step']}",
    )
    fig.colorbar(image, ax=axes[1], label="Maximum principal cosine")
    fig.suptitle("Fitted factor geometry (0 = orthogonal; one training seed)")
    fig.savefig(out / "geometry.png", dpi=150)
    plt.close(fig)


def plot_results(results: dict, output: Path) -> list[Path]:
    """Save trajectory, PCA CEV, factor consistency, and edit plots.

    Rows and measurements are passed through as recorded. Undefined values are
    left as gaps; the plots do not impute them or add inferential summaries.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = results.get("rows", [])
    if not rows:
        raise ValueError("results contain no analysis rows")
    interfaces = list(dict.fromkeys(row["interface"] for row in rows))
    steps = sorted({int(row["step"]) for row in rows})
    final_interface = interfaces[-1]
    _plot_trajectory(rows, interfaces, steps, output)
    _plot_cev(rows, final_interface, steps, output)
    _plot_development(
        rows, interfaces, steps, results.get("protocol", {}).get("scope", "last"), output
    )
    _plot_edit_recovery(rows, final_interface, steps, output)
    _plot_geometry(rows, interfaces, final_interface, output)
    return [
        output / name
        for name in (
            "trajectory.png",
            "cev.png",
            "development.png",
            "edit_recovery.png",
            "geometry.png",
        )
    ]

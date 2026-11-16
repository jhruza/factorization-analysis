# Factorization Lab

A small, standalone PyTorch and NumPy codebase for studying how transformers represent independent latent factors. It provides a canonical five-factor data process, a compact causal transformer, reproducible training runs, and held-out representation analyses. Motivated and forked from the finding of https://arxiv.org/abs/2602.02385.

## Quick start

Install [Pixi](https://pixi.sh), then run:

```sh
pixi install
pixi run train --config configs/smoke.toml --output runs/smoke
pixi run analyze --run runs/smoke --n-fit 32 --n-test 32 --pairs 8
pixi run paper --run runs/smoke --steps 0 8 --n-sequences 32 --backgrounds 2 --batch-per-background 16 --max-components 6 --interface L4_mlp --output runs/smoke/analyses/paper
```

The eight-update smoke preset checks the pipeline only; it is not evidence of factorization. Use `local.toml` for a longer exploratory run and `paper.toml` for the paper-scale budget. See [Setup](docs/setup.md) and [Experiments](docs/experiments.md) for interpretation and limits.

## Folder structure

```text
factorization-lab/
├── README.md
├── pixi.toml                  # Dependencies and runnable tasks
├── pixi.lock                  # Locked dependency versions for Mac and Linux
├── pyproject.toml             # Python package and tool settings
├── configs/
│   ├── smoke.toml             # Quick pipeline check
│   ├── local.toml             # Reduced training budget
│   └── paper.toml             # Original reproduction training settings
├── src/factorization/
│   ├── processes.py           # Five-factor generator and exact state updates
│   ├── data.py                # Sample and oracle data containers
│   ├── model.py               # Transformer and activation interfaces
│   ├── train.py               # Training, resuming, and freezing parameters
│   ├── checkpoints.py         # Saving and loading training snapshots
│   └── analysis/
│       ├── paper.py           # Paper-style PCA and vary-one factor overlap
│       ├── development.py     # Measurements across layers and checkpoints
│       ├── factors.py         # Probes, factor encoding, and subspace geometry
│       ├── interventions.py   # Natural hybrids and selective activation edits
│       └── plotting.py        # Figures from saved measurements
├── experiments/
│   ├── reproduce_independent.py  # Train the independent-factor setting
│   ├── analyze_paper.py          # Run the paper-style measurements
│   ├── analyze_development.py    # Run our developmental measurements
│   └── train_freeze_branches.py  # Compare parameter-freeze training branches
├── notebooks/                 # Short walkthroughs using the same package
│   ├── 00_generator.ipynb
│   ├── 01_paper_figures.ipynb
│   └── 02_development_interventions.ipynb
├── tests/                     # Generator, model, analysis, and resume checks
├── scripts/                   # Reference comparisons and validation records
├── docs/
│   ├── setup.md               # Local and RunPod workflow
│   ├── experiments.md         # Protocols and measurement definitions
│   └── validation.md          # Completed checks and their limits
└── runs/                      # Generated checkpoints, metrics, and figures
```

`src/` contains reusable calculations; `experiments/` contains the recipes that call them. Notebooks explain and explore those same functions. Training saves checkpoints once, so different analyses can reuse them. Generated `runs/` outputs are excluded from Git.

## What the experiment measures

The canonical process combines five independent factors with alphabet sizes `(3, 3, 3, 4, 4)`. Their fused vocabulary has 432 ordinary tokens plus BOS (ID 432). The model is a four-layer causal transformer with three heads per layer, width 120, and a width-480 ReLU MLP. Its context capacity is 9; training inputs contain 8 positions (BOS plus seven observations), with next-token targets.

The analyses measure factor information, additivity, subspace overlap, and edits at a specified model interface. They are distinct measurements; a positive result on one does not establish the others. The developmental `geometry.png` summarizes overlap between factor-effect subspaces fitted from natural predictive features. The separate `pixi run paper` analysis uses direct vary-one-factor PCA for the paper-style orthogonality diagnostic; these measure different things.

## Runs and notebooks

Training is a script-driven, resumable job. Runs save the resolved configuration, metadata, JSONL metrics, and checkpoints including optimizer, scheduler, RNG, and data provenance. Analysis is a separate held-out job and writes results and figures below the run directory. Notebooks are short entry points for inspecting the generator, running analyses, and exploring development/intervention output; they reuse the package and stored artifacts rather than reimplementing its math.

For a local machine or a RunPod instance, follow the same commands. On RunPod, keep `runs/` on a persistent network volume (for example `/workspace/runs`) so outputs survive instance shutdown. No GPU execution is claimed by this repository's documentation; select the device explicitly if needed.

See [Setup](docs/setup.md) for installation, storage, and commands, and [Experiments](docs/experiments.md) for protocol, interpretation, and limitations.

To follow the code, start with `src/factorization/processes.py`, then `model.py`, `train.py`, and `analysis/`. The implementation is self-contained and does not import the earlier reproduction repository.

## Attribution

The canonical process and architecture are informed by Shai et al., [“Transformers Learn Factored Representations”](https://proceedings.mlr.press/v306/shai26a.html). This repository is an independent, focused implementation and does not claim a complete replication of that work.

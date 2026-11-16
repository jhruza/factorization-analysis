# Setup and runs

## Install

Install Pixi, then from the repository root:

```sh
pixi install
pixi run --help
```

Pixi installs the project and its Python dependencies into a local environment. The lockfile targets Apple silicon macOS and x86-64 Linux; it resolves a CPU PyTorch wheel on macOS and a CUDA 12.8 PyTorch wheel on Linux. On a GPU host, confirm that its NVIDIA driver supports the bundled CUDA runtime:

```sh
pixi run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then pass `--device cuda` to a run command. A CUDA-enabled wheel does not guarantee that the host driver or GPU is compatible.

## Train

```sh
pixi run train --config configs/smoke.toml --output runs/smoke
pixi run train --config configs/local.toml --output runs/local
pixi run train --config configs/paper.toml --output runs/paper
```

The presets retain the canonical model and process while changing compute budget: smoke uses 8 updates and batch size 16; local uses 6,000 updates and batch size 128; paper uses 500,000 updates and batch size 25,000. The smoke run is a pipeline check only, not factorization evidence. The paper preset follows the source's plateau learning-rate schedule; the shorter presets use a constant rate.

Resume a run using the same output directory:

```sh
pixi run train --config configs/local.toml --output runs/local --resume
```

Start four paired freeze branches from a saved local checkpoint, for example update 1,000, and train each arm for 1,000 additional updates:

```sh
pixi run branches --checkpoint runs/local/checkpoints/step_001000.pt --output runs/local/branches/step-1000 --steps 1000
```

Branches inherit the parent's batch size, optimizer state, and scheduler state; the CLI has no batch-size override. Tests check exact CPU state equality for restart and unfrozen-branch examples; replay across different devices is not claimed.

Runs contain the resolved `config.json`, `metadata.json`, append-only `metrics.jsonl`, and `checkpoints/step_*.pt`. Checkpoints include model, optimizer, scheduler, RNG state, and data provenance. Do not reuse an output directory for an unrelated run.

## Analyze and plot

Analysis uses separate fitting and test sequences. Each analysis output directory must be fresh; choose a new `--output` path to keep multiple analyses. For a quick workflow check:

```sh
pixi run analyze --run runs/smoke --n-fit 32 --n-test 32 --pairs 8
```

Select checkpoints or interfaces explicitly when useful, for example `--steps 0 1000 3000 6000 --scope non_bos --observational-only --output runs/local/analyses/development`. This is the natural-feature factor diagnostic; it does not compute the paper-style vary-one PCA overlap. `--observational-only` skips quartet and edit measurements.

For the paper-style orthogonality analysis, run direct vary-one-factor PCA on selected checkpoints:

```sh
pixi run paper --run runs/local --steps 0 1000 3000 6000 --n-sequences 1024 --backgrounds 4 --batch-per-background 128 --max-components 10 --interface L4_mlp --output runs/local/analyses/paper
```

This saves `paper_pca.png` and `paper_overlap.png` under `figures/`. Use a fresh output path for each analysis.

The paper analysis uses the model's full 9-position input capacity (BOS plus 8 observations), even though included training presets use 8 input positions. Its default sample is smaller than the source orthogonality protocol (4 backgrounds × 128 varying sequences and up to 10 components, versus 10 × 200 and 20 components). These choices are recorded in `results.json`.

For a more stable analysis, use the default sample sizes or larger ones. Results are saved in `results.json`, including checkpoint-level model-oracle KL and row-level representation diagnostics. Figures are written to `figures/` and include trajectory, PCA cumulative explained variance, factor-change consistency, edit recovery, and factor-subspace geometry. Fitting activations/probes and test evaluation are kept separate; do not tune choices against the test split.

## Local machine and RunPod

Use the same commands in either environment. On RunPod, attach a persistent network volume and set the output path on that volume, such as `/workspace/runs/five-factor`. Start training in a persistent terminal session, and reconnect to inspect metrics or resume after an interruption. Keep the config and the complete run directory together when transferring artifacts.

## Notebooks

Launch Jupyter with `pixi run notebook`. Start with `notebooks/00_generator.ipynb`, then use `01_paper_figures.ipynb` to inspect saved analysis and `02_development_interventions.ipynb` for short development/intervention explorations. Long training remains a CLI job; notebooks should import package code and inspect saved outputs.

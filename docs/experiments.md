# Experimental protocol

## Canonical task

The process has five independent latent factors with alphabet sizes `(3, 3, 3, 4, 4)`. Their Cartesian product yields 432 ordinary fused-token IDs; BOS has ID 432. Model context capacity is 9. Training examples contain 8 input positions: BOS followed by seven generated observations, with next-token targets. The canonical model has four causal transformer layers, three attention heads per layer, residual width 120, and a width-480 ReLU MLP.

The three presets change only budget and learning-rate schedule:

| Preset | Updates | Batch size | Schedule |
| --- | ---: | ---: | --- |
| `smoke.toml` | 8 | 16 | Constant |
| `local.toml` | 6,000 | 128 | Constant |
| `paper.toml` | 500,000 | 25,000 | Source plateau schedule |

The paper-scale budget follows the motivating source protocol; it is not evidence that an unrun job has been reproduced. The 8-update smoke preset is a pipeline check, not evidence of factorization. Each training run uses one seed, so repeat seeds and report uncertainty before drawing broader conclusions.

## Run, analysis, and artifacts

Training runs are independent jobs that can be resumed from their checkpoints. Analysis is a later job over selected checkpoints. It fits probes and factor-effect estimates on fitting sequences and reports performance on independently generated test sequences. Keep fitting and test samples distinct and freeze checkpoint selection before using test results.

Each run stores a resolved configuration and metadata, metrics, and checkpoints. Checkpoints retain model, optimizer, scheduler, RNG, and data provenance. Analyses write `results.json` and plots under `analyses/`; `results.json` includes checkpoint-level model-oracle KL. Record the repository revision and any changes to data generation, sampling, or analysis when sharing results.

Probe and factor-effect fitting uses separate fitting sequences; test sequences are generated independently and should not guide checkpoint selection. Natural factor features (log probabilities) are computed in float64, floored at `1e-10`, and linear rank decisions use relative cutoff `1e-8`. Factor-effect bases retain 95% of fitted effect variance. When intervention measurements are enabled, factor-effect maps are always fitted on non-BOS positions, even when observational readability uses `--scope last`.

The notebooks are for quick inspection: generator sanity checks, plots from a completed run, and short development/intervention views. They call the package or CLI and do not replace long-running training scripts.

## What the measurements mean

- **Factor information** asks whether activations predict each factor's belief or predictive vector.
- **Additivity** asks whether combined factor effects explain the code or output without large interaction terms.
- **Factor-effect geometry** in the developmental analysis measures overlap between subspaces fitted from natural predictive features. `geometry.png` plots the median pairwise maximum principal cosine by interface and the overlap matrix at the latest checkpoint.
- **Paper-style orthogonality** uses direct vary-one-factor PCA: each factor varies over natural trajectories while the other factors are held at sampled backgrounds. It centers activations per position within each background and measures pairwise PCA-basis overlap. `paper_overlap.png` reports mean normalized squared Frobenius overlap. This differs from the natural-feature fitted geometry above.
- **Editability** asks whether changing one factor-associated component changes the intended prediction while preserving others.

These claims are related but not interchangeable. The developmental geometry figure reports maximum principal cosines between fitted factor-effect subspaces (0 indicates orthogonality); the paper-style diagnostic instead compares direct PCA bases. Neither establishes independent computation. Information in an activation does not prove that the network uses it.

The paper-style analysis uses all nine model input positions (BOS plus eight observations), although the included training presets use eight positions. Its defaults (4 backgrounds × 128 varying sequences and up to 10 components) are smaller than the source orthogonality protocol (10 × 200 and 20 components). Samples use this repository's NumPy random stream, not the source experiment's JAX stream; backgrounds are not independent training replications.

Whole-history edits patch every sequence position at an interface. The native whole-history hybrid is a positive control. The native query-only patch changes only the final position, so earlier attention still reflects the recipient history; this structural difference can create a plateau even when a whole-history edit works. Orthogonal projection and two random projections of matching rank are edit controls. Counterfactual targets are the model's own predictions on natural hybrid sequences, not oracle correctness. Recovery is `1 − edited_KL / no_patch_KL`; it can be negative. The weak-baseline flag is set below `1e-5`; recovery is undefined below `1e-10`.

For the invariance ratio, let `δ0` and `δ1` be a factor's activation changes in two contexts. The ratio is `mean(||δ0 − δ1||²) / (mean(||δ0||²) + mean(||δ1||²))`; zero means the changes match across contexts.

## Scope and limitations

This is a focused, independent PyTorch/NumPy implementation inspired by Shai et al.; it does not reproduce every model, condition, analysis, or result in the paper. Each run uses one training seed. Synthetic data, one canonical architecture per run, finite sequence lengths, and linear probes/interventions limit what can be inferred about natural language models or general transformer learning.

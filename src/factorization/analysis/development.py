"""Apply paper diagnostics and developmental interventions to saved checkpoints."""

import json
from pathlib import Path

import numpy as np
import torch

from ..checkpoints import load_checkpoint
from ..model import ModelConfig, Transformer
from ..processes import canonical_process
from ..train import choose_device, write_json
from .factors import context_dependence, factor_metrics, oblique_maps, pca_metrics
from .interventions import factor_edit, make_quartets, ordinary_marginals, predict


def capture(model: Transformer, tokens: np.ndarray, batch_size: int) -> dict:
    """Evaluate in chunks; keep CPU arrays, not an autograd graph."""
    device = next(model.parameters()).device
    collected = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            _, states = model.capture(
                torch.as_tensor(tokens[start : start + batch_size], dtype=torch.long, device=device)
            )
            for name, value in states.items():
                collected.setdefault(name, []).append(value.cpu().numpy())
    return {name: np.concatenate(values) for name, values in collected.items()}


def select_scope(array: np.ndarray, scope: str) -> np.ndarray:
    if scope == "last":
        return array[:, -1].astype(np.float64)
    if scope == "non_bos":
        return array[:, 1:].reshape(-1, array.shape[-1]).astype(np.float64)
    raise ValueError("scope must be last or non_bos")


def analyze(
    run_dir: Path,
    checkpoints: list[int] | None = None,
    n_fit: int = 1024,
    n_test: int = 1024,
    pairs: int = 64,
    batch_size: int = 128,
    device: str = "auto",
    scope: str = "last",
    interfaces: list[str] | None = None,
    interventions: bool = True,
    output: Path | None = None,
) -> dict:
    """Reuse fixed samples across checkpoints. All fitted quantities use fit data.

    scope chooses observational rows. Intervention bases always use non-BOS
    fitting positions because a whole-history patch acts at every position.
    Output must be a fresh directory, keeping previous measurement sets intact.
    """
    if min(n_fit, n_test, batch_size) < 1 or (interventions and not 1 <= pairs <= n_test):
        raise ValueError("Positive sample/chunk counts required; pairs must be in [1,n_test]")
    select_scope(np.zeros((1, 2, 1)), scope)  # Validate before starting model evaluation.
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / "analyses" / "development"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Analysis output already exists: {output}; choose --output")
    config = json.loads((run_dir / "config.json").read_text())
    if interventions and config["sequence_length"] < 2:
        raise ValueError("Whole-history interventions require a non-BOS position")
    model = Transformer(ModelConfig(**config["model"]), config["model_seed"]).to(
        choose_device(device)
    )
    process = canonical_process()
    fit = process.sample(n_fit, config["sequence_length"], 710001)
    test = process.sample(n_test, config["sequence_length"], 710002)
    if scope == "non_bos" and config["sequence_length"] < 2:
        raise ValueError("Non-BOS analysis requires sequence_length >= 2")
    paths = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if checkpoints is not None:
        by_step = {int(path.stem.split("_")[1]): path for path in paths}
        missing = set(checkpoints) - by_step.keys()
        if missing:
            raise FileNotFoundError(f"Missing checkpoint updates: {sorted(missing)}")
        paths = [by_step[step] for step in sorted(set(checkpoints))]
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    quartets = make_quartets(process, test.tokens, pairs) if interventions else None
    hybrid_states = None
    result = dict(
        format_version=1,
        run=str(run_dir),
        config=config,
        protocol=dict(
            fit_seed=710001,
            test_seed=710002,
            pair_seed=710003,
            n_fit=n_fit,
            n_test=n_test,
            pairs=pairs if interventions else 0,
            scope=scope,
            map_fit_scope="non_bos",
            log_probability_floor=1e-10,
            map_variance_cutoff=0.95,
            interventions=interventions,
            counterfactual_target="model prediction on the natural hybrid, not oracle correctness",
            training_replications=1,
        ),
        factor_sizes=list(process.sizes),
        checkpoints=[],
        rows=[],
    )
    if quartets:
        result["pairs"] = {key: quartets[key].tolist() for key in ("recipients", "donors")}
    for path in paths:
        saved = load_checkpoint(path)
        model.load_state_dict(saved["model"])
        prediction = predict(model, test.tokens, batch_size=batch_size)
        predicted_factors = ordinary_marginals(prediction, process.sizes)
        true_factors = [p[:, -1] for p in test.factor_probs]

        def mean_kl(p, q):
            return float(
                np.mean(
                    np.sum(
                        p * (np.log(np.maximum(p, 1e-100)) - np.log(np.maximum(q, 1e-100))), axis=-1
                    )
                )
            )

        joint = true_factors[0]
        for probabilities in true_factors[1:]:
            joint = (joint[:, :, None] * probabilities[:, None, :]).reshape(n_test, -1)
        ordinary = prediction[:, : process.vocab_size].copy()
        ordinary /= ordinary.sum(axis=1, keepdims=True)
        result["checkpoints"].append(
            dict(
                step=saved["step"],
                oracle_conditional_kl=mean_kl(joint, ordinary),
                factor_oracle_kl=[mean_kl(p, q) for p, q in zip(true_factors, predicted_factors)],
                mean_bos_probability=float(prediction[:, process.bos_token].mean()),
            )
        )
        h_fit, h_test = (
            capture(model, fit.tokens, batch_size),
            capture(model, test.tokens, batch_size),
        )
        chosen = interfaces if interfaces is not None else list(h_fit)
        if set(chosen) - h_fit.keys():
            raise ValueError(f"Unknown interfaces: {set(chosen) - h_fit.keys()}")
        if quartets:
            combined = np.concatenate([*quartets["x10"], *quartets["x01"]])
            hybrid_states = capture(model, combined, batch_size)
        for interface in chosen:
            fitted, bases = factor_metrics(
                select_scope(h_fit[interface], scope),
                select_scope(h_test[interface], scope),
                [select_scope(a, scope) for a in fit.beliefs],
                [select_scope(a, scope) for a in test.beliefs],
                [select_scope(a, scope) for a in fit.factor_probs],
                [select_scope(a, scope) for a in test.factor_probs],
            )
            row = dict(
                step=saved["step"],
                interface=interface,
                **fitted,
                pca=pca_metrics(
                    select_scope(h_fit[interface], scope), select_scope(h_test[interface], scope)
                ),
            )
            if quartets:
                # Whole-history maps are fitted from all non-BOS fitting positions.
                if scope != "non_bos":
                    map_metrics, bases = factor_metrics(
                        select_scope(h_fit[interface], "non_bos"),
                        select_scope(h_test[interface], "non_bos"),
                        [select_scope(a, "non_bos") for a in fit.beliefs],
                        [select_scope(a, "non_bos") for a in test.beliefs],
                        [select_scope(a, "non_bos") for a in fit.factor_probs],
                        [select_scope(a, "non_bos") for a in test.factor_probs],
                    )
                else:
                    map_metrics = fitted
                row["map_condition"] = map_metrics["condition"]
                try:
                    maps = oblique_maps(bases)
                except ValueError as error:
                    maps = None
                    row["map_error"] = str(error)
                rec, don = quartets["recipients"], quartets["donors"]
                h00, h11 = h_test[interface][rec], h_test[interface][don]
                row["invariance"], row["interventions"] = [], []
                factors = len(process.sizes)
                for factor in range(factors):
                    h10 = hybrid_states[interface][factor * pairs : (factor + 1) * pairs]
                    h01 = hybrid_states[interface][
                        (factor + factors) * pairs : (factor + factors + 1) * pairs
                    ]
                    row["invariance"].append(
                        {
                            mode: context_dependence(
                                *[select_scope(h, mode) for h in (h00, h10, h01, h11)]
                            )
                            for mode in ("last", "non_bos")
                        }
                    )
                    if maps is not None:
                        edit = factor_edit(
                            model,
                            quartets["x00"],
                            quartets["x10"][factor],
                            h00,
                            h11,
                            h10,
                            maps[factor],
                            interface,
                            factor,
                            process.sizes,
                            batch_size,
                        )
                        row["interventions"].append(edit)
            result["rows"].append(row)
            print(
                json.dumps(dict(step=row["step"], interface=interface, k95=row["pca"]["k95"])),
                flush=True,
            )
        write_json(output / "results.json", result)
    # Plot only saved numerical results; replotting never needs a GPU.
    from .plotting import plot_results

    plot_results(result, output / "figures")
    return result

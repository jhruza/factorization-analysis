"""Check retained reference values, or regenerate them against factored-reps.

Without --source, this needs only the new project's NumPy environment. To compare
directly, run it with the reference project's Python and pass --source /path/to/
factored-reps. JAX/Hydra are never runtime dependencies of factorization.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorization.processes import canonical_process  # noqa: E402


def compare(fixture: dict) -> dict[str, float]:
    process = canonical_process()
    truth = process.evaluate(np.asarray(fixture["tokens"], dtype=np.int64), joint="all")
    actual = {
        "transition_matrices": process.transition_matrices,
        "initial_states": process.initial_states,
        "normalizing_eigenvectors": process.normalizing_eigenvectors,
        "beliefs": truth.beliefs,
        "factor_probs": truth.factor_probs,
        "joint_probs": [truth.joint_probs],
    }
    errors = {}
    for name, arrays in actual.items():
        errors[name] = max(
            float(np.max(np.abs(a - np.asarray(b))))
            for a, b in zip(arrays, fixture[name], strict=True)
        )
        for a, b in zip(arrays, fixture[name], strict=True):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6, err_msg=name)
    return errors


def reference_fixture(source: Path) -> dict:
    sys.path.insert(0, str(source.resolve()))
    import jax.numpy as jnp
    from fwh_core.generative_processes.builder import build_factored_process_from_spec
    from fwh_core.utils.factoring_utils import compute_obs_dist_for_variant

    spec = [
        {
            "component_type": "hmm",
            "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
        }
        for _ in range(3)
    ] + [
        {
            "component_type": "ghmm",
            "variants": [
                {"process_name": "tom_quantum", "process_params": {"alpha": 1.0, "beta": 3.0}}
            ],
        }
        for _ in range(2)
    ]
    reference = build_factored_process_from_spec("independent", spec)
    tokens = [[432, 0, 431, 216, 53, 120, 378, 64], [432, 127, 333, 42, 201, 17, 413, 288]]
    beliefs = [[] for _ in range(5)]
    factor_probs = [[] for _ in range(5)]
    joint_probs = []
    for row in tokens:
        state = reference.initial_state
        row_beliefs = [[] for _ in range(5)]
        row_probs = [[] for _ in range(5)]
        row_joint = []
        for token in row:
            state = (
                reference.initial_state
                if token == 432
                else reference.transition_states(state, jnp.asarray(token))
            )
            for i in range(5):
                row_beliefs[i].append(np.asarray(state[i]).tolist())
                p = compute_obs_dist_for_variant(
                    reference.component_types[i],
                    state[i],
                    reference.transition_matrices[i][0],
                    reference.normalizing_eigenvectors[i][0],
                )
                row_probs[i].append(np.asarray(p).tolist())
            row_joint.append(
                np.asarray(reference.observation_probability_distribution(state)).tolist()
            )
        for i in range(5):
            beliefs[i].append(row_beliefs[i])
            factor_probs[i].append(row_probs[i])
        joint_probs.append(row_joint)
    return {
        "tokens": tokens,
        "transition_matrices": [
            np.asarray(t[0]).real.tolist() for t in reference.transition_matrices
        ],
        "initial_states": [np.asarray(s).tolist() for s in reference.initial_states],
        "normalizing_eigenvectors": [
            np.asarray(n[0]).tolist() for n in reference.normalizing_eigenvectors
        ],
        "beliefs": beliefs,
        "factor_probs": factor_probs,
        "joint_probs": [joint_probs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional original factored-reps checkout")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "scripts/reference_process_check.json"
    )
    args = parser.parse_args()
    if args.source:
        fixture = reference_fixture(args.source)
        commit = subprocess.run(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        report = {
            "reference": "factored-reps: independent 3mess3_2tom_quantum",
            "source_commit": commit,
            "parameters": {
                "mess3": {"x": 0.15, "a": 0.6, "count": 3},
                "tom_quantum": {"alpha": 1.0, "beta": 3.0, "count": 2},
            },
            "tolerance": {"atol": 2e-6, "rtol": 2e-6},
            "note": "NumPy float64 vs source JAX float32; distribution/oracle parity, not identical random streams. BOS uses raw source initial states.",
            "fixture": fixture,
        }
    else:
        report = json.loads(args.output.read_text())
    errors = compare(report["fixture"])
    if args.source:
        report["max_absolute_errors"] = errors
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "max_absolute_errors": errors}, indent=2))


if __name__ == "__main__":
    main()

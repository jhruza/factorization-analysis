"""Exercise saved-model analysis, including its state-complete controls."""

import numpy as np
import pytest
import torch

from factorization.analysis import analyze
from factorization.train import train


def test_training_to_analysis_pipeline(tmp_path):
    torch.set_num_threads(1)
    config = dict(
        name="integration",
        steps=2,
        batch_size=4,
        sequence_length=4,
        model_seed=7,
        data_seed=77,
        learning_rate=0.0005,
        scheduler="constant",
        checkpoint_every=2,
        log_every=1,
        model=dict(d_model=24, d_head=8, n_heads=3, n_layers=1, n_ctx=9, d_mlp=48, d_vocab=433),
    )
    run = train(config, tmp_path / "run", device="cpu")
    result = analyze(
        run,
        checkpoints=[0, 2],
        n_fit=32,
        n_test=32,
        pairs=4,
        interfaces=["input", "L1_mlp", "final"],
        device="cpu",
    )
    assert len(result["rows"]) == 6
    assert len(result["checkpoints"]) == 2
    assert all(np.isfinite(c["oracle_conditional_kl"]) for c in result["checkpoints"])
    assert result["protocol"]["scope"] == "last"
    assert result["protocol"]["map_fit_scope"] == "non_bos"
    for row in result["rows"]:
        assert len(row["invariance"]) == 5
        for edit in row["interventions"]:
            assert edit["control_max_error"] < 2e-5
            assert edit["native_target_kl"] < 1e-10
            if row["interface"] in ("L1_mlp", "final"):
                assert edit["native_query_only"]["target_kl"] < 1e-10
    output = run / "analyses" / "development"
    assert (output / "results.json").exists()
    assert len(list((output / "figures").glob("*.png"))) >= 2
    with pytest.raises(FileExistsError):
        analyze(run, checkpoints=[0], n_fit=8, n_test=8, pairs=2)

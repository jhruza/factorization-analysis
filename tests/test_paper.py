import json

import numpy as np
import pytest
import torch

from factorization.analysis.paper import (
    _vary_tokens,
    analyze_paper,
    center_positions,
    overlap_curve,
)
from factorization.processes import canonical_process
from factorization.train import train


def test_position_centering_and_overlap():
    rng = np.random.default_rng(3)
    activations = rng.normal(size=(8, 4, 5))
    offsets = rng.normal(size=(1, 4, 5))
    np.testing.assert_allclose(
        center_positions(activations), center_positions(activations + offsets), atol=1e-14
    )
    centered = center_positions(activations).reshape(8, 3, 5)
    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-14)
    eye = np.eye(4)
    np.testing.assert_allclose(overlap_curve(eye[:2], eye[2:], 3), [0, 0])
    np.testing.assert_allclose(overlap_curve(eye[:2], eye[:2], 3), [1, 1])
    np.testing.assert_allclose(overlap_curve(eye[:2], eye[[0, 2]], 2), [1, 0.5])


def test_vary_one_freezes_other_histories():
    process = canonical_process()
    tokens = _vary_tokens(process, 9, 2, 0, 30)
    assert np.all(tokens[:, 0] == process.bos_token)
    factors = process.decode(tokens[:, 1:])
    for i in [0, 1, 3, 4]:
        assert np.all(factors[..., i] == factors[0, :, i])
    assert np.any(factors[..., 2] != factors[0, :, 2])


def test_end_to_end_two_checkpoints(tmp_path):
    config = dict(
        name="paper-test",
        steps=1,
        batch_size=4,
        sequence_length=4,
        model_seed=7,
        data_seed=123,
        learning_rate=0.001,
        scheduler="constant",
        checkpoint_every=1,
        log_every=1,
        model=dict(d_model=12, d_head=4, n_heads=3, n_layers=1, n_ctx=9, d_mlp=24, d_vocab=433),
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run = tmp_path / "run"
        train(config, run, device="cpu")
        result = analyze_paper(
            run,
            n_sequences=16,
            backgrounds=2,
            batch_per_background=8,
            max_components=3,
            batch_size=8,
            device="cpu",
            interface="L1_mlp",
        )
    finally:
        torch.set_num_threads(previous)
    assert result["protocol"]["input_length"] == 9
    assert [r["step"] for r in result["rows"]] == [0, 1]
    for row in result["rows"]:
        assert len(row["combined_cev"]) == 12
        assert len(row["factor_cevs"]) == len(row["factor_k95"]) == 5
        assert len(row["pair_overlap"]) == 10
        assert all(len(curve) == 3 for curve in row["pair_overlap"].values())
        assert all(
            0 <= value <= 1 + 1e-12 for curve in row["pair_overlap"].values() for value in curve
        )
    output = run / "analyses" / "paper"
    assert json.loads((output / "results.json").read_text()) == result
    assert (output / "figures" / "paper_pca.png").exists()
    assert (output / "figures" / "paper_overlap.png").exists()
    with pytest.raises(FileExistsError):
        analyze_paper(run, n_sequences=2, backgrounds=1, batch_per_background=2, interface="L1_mlp")

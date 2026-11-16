import json
from pathlib import Path

import numpy as np
import pytest

from factorization.processes import Process, canonical_process


def test_retained_source_operators_states_and_oracle():
    fixture = json.loads(
        (Path(__file__).parents[1] / "scripts/reference_process_check.json").read_text()
    )["fixture"]
    process = canonical_process()
    truth = process.evaluate(np.asarray(fixture["tokens"]), joint="all")
    for name, actual in (
        ("transition_matrices", process.transition_matrices),
        ("initial_states", process.initial_states),
        ("normalizing_eigenvectors", process.normalizing_eigenvectors),
        ("beliefs", truth.beliefs),
        ("factor_probs", truth.factor_probs),
        ("joint_probs", [truth.joint_probs]),
    ):
        for a, expected in zip(actual, fixture[name], strict=True):
            np.testing.assert_allclose(a, expected, atol=2e-6, rtol=2e-6)


def test_encoding_roundtrip_and_arbitrary_factor_count():
    process = canonical_process()
    assert process.sizes == (3, 3, 3, 4, 4)
    assert process.vocab_size == process.bos_token == 432
    tokens = np.arange(432).reshape(2, 216)
    np.testing.assert_array_equal(process.encode(process.decode(tokens)), tokens)
    np.testing.assert_array_equal(process.decode(np.array([0, 431])), [[0] * 5, [2, 2, 2, 3, 3]])
    small = Process([np.ones((v, 1, 1)) / v for v in (2, 5)], [np.ones(1)] * 2, [np.ones(1)] * 2)
    np.testing.assert_array_equal(small.encode(small.decode(np.arange(10))), np.arange(10))
    for invalid in (np.array([432]), np.array([-1]), np.array([1.5])):
        with pytest.raises(ValueError):
            process.decode(invalid)


def test_sample_alignment_reproducibility_and_exact_truth():
    process = canonical_process()
    batch = process.sample(12, 8, seed=73)
    assert batch.tokens.shape == batch.targets.shape == (12, 8)
    assert np.all(batch.tokens[:, 0] == 432)
    np.testing.assert_array_equal(batch.tokens[:, 1:], batch.targets[:, :-1])
    np.testing.assert_array_equal(batch.tokens, process.sample(12, 8, seed=73).tokens)
    truth = process.evaluate(batch.tokens, joint="last")
    for actual, expected in zip(batch.beliefs, truth.beliefs, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    for actual, expected in zip(batch.factor_probs, truth.factor_probs, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert truth.joint_probs.shape == (12, 432)
    np.testing.assert_allclose(truth.joint_probs.sum(axis=-1), 1)
    no_truth = process.sample(12, 8, seed=73, with_truth=False)
    np.testing.assert_array_equal(no_truth.tokens, batch.tokens)
    assert no_truth.beliefs is no_truth.factor_probs is None


def test_signed_quantum_filter_stays_finite_and_physical():
    process = canonical_process()
    batch = process.sample(32, 256, seed=91)
    for i in (3, 4):
        states = batch.beliefs[i]
        assert np.isfinite(states).all()
        assert np.any(states[..., 1:] < 0)
        np.testing.assert_allclose(states[:, 0], np.tile([1, 0, 0], (32, 1)))
        np.testing.assert_allclose(states[:, 1:, 0], 1 / 3)
        bloch = states[..., 1:] / states[..., :1]
        assert np.max(np.sum(bloch**2, axis=-1)) <= 1 + 1e-12
        assert np.min(batch.factor_probs[i]) >= 0
        np.testing.assert_allclose(batch.factor_probs[i].sum(axis=-1), 1)


def test_sampling_matches_conditional_oracle():
    process = canonical_process()
    batch = process.sample(50_000, 2, seed=18)
    symbols = process.decode(batch.targets)
    for factor, size in enumerate(process.sizes):
        for previous in range(size):
            mask = symbols[:, 0, factor] == previous
            n = mask.sum()
            actual = np.bincount(symbols[mask, 1, factor], minlength=size) / n
            expected = batch.factor_probs[factor][mask, 1].mean(axis=0)
            tolerance = 6 * np.sqrt(expected * (1 - expected) / n) + 0.001
            assert np.all(np.abs(actual - expected) < tolerance)


def test_hybrid_histories_and_bos_reset():
    process = canonical_process()
    tokens = process.sample(2, 8, seed=39).tokens
    base = process.evaluate(tokens)
    factors = process.decode(tokens[:, 1:])
    factors[0, :, 0] = factors[1, :, 0]
    hybrid = np.concatenate((tokens[:, :1], process.encode(factors)), axis=1)
    truth = process.evaluate(hybrid)
    np.testing.assert_allclose(truth.beliefs[0][0], base.beliefs[0][1])
    for i in range(1, 5):
        np.testing.assert_allclose(truth.beliefs[i][0], base.beliefs[i][0])
    tokens[:, 4] = process.bos_token
    reset = process.evaluate(tokens)
    for i in range(5):
        np.testing.assert_allclose(
            reset.beliefs[i][:, 4], np.tile(process.initial_states[i], (2, 1))
        )
        suffix = process.evaluate(tokens[:, 4:])
        np.testing.assert_allclose(reset.beliefs[i][:, 4:], suffix.beliefs[i])

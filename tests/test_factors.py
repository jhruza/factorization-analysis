import numpy as np
import pytest

from factorization.analysis.factors import (
    _natural,
    context_dependence,
    factor_metrics,
    oblique_maps,
    pca_metrics,
)


def example(seed=3, n=400):
    rng = np.random.default_rng(seed)
    logits = [rng.normal(size=(n, 3)), rng.normal(size=(n, 3))]
    probabilities = [np.exp(x - x.max(axis=1, keepdims=True)) for x in logits]
    probabilities = [p / p.sum(axis=1, keepdims=True) for p in probabilities]
    beliefs = [p.copy() for p in probabilities]
    features = _natural(probabilities)
    # Two independent nonorthogonal factor planes in four hidden dimensions.
    b1 = np.array([[1, 0], [0, 1], [0, 0], [0, 0.0]])
    b2 = np.array([[0.5, 0], [0, 0.4], [1, 0], [0, 1.0]])
    hidden = features[0][:, :2] @ b1.T + features[1][:, :2] @ b2.T
    return hidden, beliefs, probabilities, b1, b2


def test_known_oblique_code_and_distinct_probe_encoding():
    h, beliefs, probabilities, b1, b2 = example()
    metrics, bases = factor_metrics(
        h[:300],
        h[300:],
        [x[:300] for x in beliefs],
        [x[300:] for x in beliefs],
        [x[:300] for x in probabilities],
        [x[300:] for x in probabilities],
        variance_cutoff=1.0,
    )
    assert metrics["natural_encoding_r2"] > 0.999999
    assert all(0.7 < score < 0.99 for score in metrics["belief_probe_r2"])
    assert metrics["factor_ranks"] == [2, 2]
    assert 0.3 < metrics["overlap"][0][1] < 0.6
    assert metrics["condition"] > 1
    q1, q2 = oblique_maps(bases)
    np.testing.assert_allclose(q1 @ b1, b1, atol=1e-10)
    np.testing.assert_allclose(q1 @ b2, 0, atol=1e-10)
    np.testing.assert_allclose(q2 @ b2, b2, atol=1e-10)
    np.testing.assert_allclose(q2 @ b1, 0, atol=1e-10)


def test_zero_variance_and_bad_bases():
    h = np.ones((12, 4))
    b = [np.tile([0.2, 0.3, 0.5], (12, 1))]
    metrics, bases = factor_metrics(h[:8], h[8:], [b[0][:8]], [b[0][8:]], [b[0][:8]], [b[0][8:]])
    assert metrics["belief_probe_r2"] == [None]
    assert metrics["natural_encoding_r2"] is None
    assert metrics["factor_ranks"] == [0]
    assert metrics["overlap"] == [[None]] and metrics["condition"] is None
    with pytest.raises(ValueError):
        oblique_maps(bases)
    with pytest.raises(ValueError):
        oblique_maps([np.array([[1.0], [0.0]]), np.array([[1.0], [0.0]])])
    pca = pca_metrics(h[:8], h[8:])
    assert pca["k95"] is None and pca["cumulative_variance"] is None


def test_natural_features_use_float64_and_have_12_independent_dimensions():
    rng = np.random.default_rng(8)
    sizes = [3, 3, 3, 4, 4]
    probs = []
    for size in sizes:
        p = rng.uniform(0.001, 1, size=(200, size)).astype(np.float32)
        probs.append(p / p.sum(axis=1, keepdims=True))
    features = _natural(probs)
    x = np.concatenate(features, axis=1)
    assert x.dtype == np.float64
    assert all(np.max(np.abs(a.sum(axis=1))) < 1e-13 for a in features)
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    assert np.sum(singular > singular[0] * 1e-8) == 12
    assert np.isfinite(_natural([np.array([[0.0, 0.5, 0.5]])])[0]).all()


def test_pca_is_fit_only_and_context_ratio():
    h, _, _, _, _ = example()
    original = pca_metrics(h[:200], h[200:])
    assert original == pca_metrics(h[:200], h[200:] * 200)
    assert original["k95"] >= 1
    z = np.zeros((4, 2))
    d = np.ones((4, 2))
    no_context = context_dependence(z, d, z, d)
    assert no_context["ratio"] == 0 and no_context["effect_energy_0"] == 2
    assert context_dependence(z, z, z, z)["ratio"] is None
    changed = context_dependence(z, d, z, -d)
    assert changed["ratio"] == 2

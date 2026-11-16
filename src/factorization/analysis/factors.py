"""Fit-only linear diagnostics for factor representations.

Natural features are log probabilities, clipped below at 1e-10 in float64,
then centered across symbols within each factor. Basis vectors describe the
independently fitted factor effects on hidden states, not readout directions.
"""

import numpy as np

RCOND = 1e-8
PROBABILITY_FLOOR = 1e-10


def _matrix(x, name):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty, finite 2D array")
    return a


def _r2(target, prediction):
    total = np.sum((target - target.mean(axis=0)) ** 2)
    if total == 0:
        return None
    return float(1 - np.sum((target - prediction) ** 2) / total)


def _affine_fit(x, y):
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    return xm, ym, np.linalg.lstsq(x - xm, y - ym, rcond=RCOND)[0]


def _natural(probabilities):
    result = []
    for p in probabilities:
        p = _matrix(p, "probabilities")
        if np.any(p < 0):
            raise ValueError("probabilities must be nonnegative")
        logp = np.log(np.maximum(p, PROBABILITY_FLOOR))
        result.append(logp - logp.mean(axis=1, keepdims=True))
    return result


def factor_metrics(
    h_fit, h_test, belief_fit, belief_test, prob_fit, prob_test, variance_cutoff=0.95
):
    """Return held-out scores and training-only factor-effect bases.

    ``belief_probe_r2`` decodes beliefs from the hidden state; the distinct
    ``natural_encoding_r2`` predicts the hidden state from all natural features.
    Overlap is the largest principal cosine per pair. Undefined R², overlaps,
    and joint condition numbers are reported as None, never NaN or infinity.
    """
    h_fit, h_test = _matrix(h_fit, "h_fit"), _matrix(h_test, "h_test")
    if h_fit.shape[1] != h_test.shape[1]:
        raise ValueError("hidden dimensions must match")
    if not 0 < variance_cutoff <= 1:
        raise ValueError("variance_cutoff must be in (0, 1]")
    count = len(belief_fit)
    if count == 0 or any(len(xs) != count for xs in (belief_test, prob_fit, prob_test)):
        raise ValueError("belief and probability lists must have the same nonzero length")
    bf = [_matrix(b, "belief_fit") for b in belief_fit]
    bt = [_matrix(b, "belief_test") for b in belief_test]
    nf, nt = _natural(prob_fit), _natural(prob_test)
    for fit, test, natural_fit, natural_test in zip(bf, bt, nf, nt):
        if (
            fit.shape != (len(h_fit), 3)
            or test.shape != (len(h_test), 3)
            or natural_fit.shape[0] != len(h_fit)
            or natural_test.shape[0] != len(h_test)
            or natural_fit.shape[1] != natural_test.shape[1]
        ):
            raise ValueError("factor samples or feature dimensions do not match")

    beliefs_fit, beliefs_test = np.concatenate(bf, axis=1), np.concatenate(bt, axis=1)
    hm, bm, probe = _affine_fit(h_fit, beliefs_fit)
    pred = bm + (h_test - hm) @ probe
    belief_scores = [
        _r2(beliefs_test[:, 3 * i : 3 * i + 3], pred[:, 3 * i : 3 * i + 3]) for i in range(count)
    ]

    xfit, xtest = np.concatenate(nf, axis=1), np.concatenate(nt, axis=1)
    xm, hm, encoding = _affine_fit(xfit, h_fit)
    score = _r2(h_test, hm + (xtest - xm) @ encoding)
    centered = xfit - xm
    bases, ranks = [], []
    start = 0
    for features in nf:
        end = start + features.shape[1]
        effect = centered[:, start:end] @ encoding[start:end]
        _, singular, vh = np.linalg.svd(effect, full_matrices=False)
        # A vanishing effect has no direction; don't create arbitrary bases.
        if len(singular) == 0 or singular[0] <= np.finfo(float).eps:
            rank = 0
        else:
            available = int(np.sum(singular > singular[0] * RCOND))
            energy = np.cumsum(singular[:available] ** 2)
            rank = min(available, int(np.searchsorted(energy / energy[-1], variance_cutoff) + 1))
        bases.append(vh[:rank].T.copy())
        ranks.append(rank)
        start = end

    overlap = [[None] * count for _ in range(count)]
    for i, a in enumerate(bases):
        for j, b in enumerate(bases):
            if a.shape[1] and b.shape[1]:
                overlap[i][j] = float(np.linalg.svd(a.T @ b, compute_uv=False)[0])
    joint = np.concatenate(bases, axis=1)
    condition = None
    if all(ranks) and joint.shape[1] <= joint.shape[0]:
        s = np.linalg.svd(joint, compute_uv=False)
        if s[-1] > s[0] * RCOND:
            condition = float(s[0] / s[-1])
    return {
        "belief_probe_r2": belief_scores,
        "natural_encoding_r2": score,
        "factor_ranks": ranks,
        "overlap": overlap,
        "condition": condition,
    }, bases


def pca_metrics(h_fit, h_test):
    """Fit centered PCA on the fit split; test data never affects the spectrum."""
    fit, test = _matrix(h_fit, "h_fit"), _matrix(h_test, "h_test")
    if fit.shape[1] != test.shape[1]:
        raise ValueError("hidden dimensions must match")
    centered = fit - fit.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(fit))[::-1], 0)
    total = eigenvalues.sum()
    if total == 0:
        return {"eigenvalues": eigenvalues.tolist(), "cumulative_variance": None, "k95": None}
    cumulative = np.cumsum(eigenvalues) / total
    return {
        "eigenvalues": eigenvalues.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "k95": int(np.searchsorted(cumulative, 0.95) + 1),
    }


def oblique_maps(bases):
    """Maps acting on column vectors, preserving their factor and removing others."""
    if not bases:
        raise ValueError("at least one nonempty basis is required")
    bs = [_matrix(b, "basis") for b in bases]
    if any(b.shape[0] != bs[0].shape[0] for b in bs):
        raise ValueError("basis dimensions must match")
    joint = np.concatenate(bs, axis=1)
    singular = np.linalg.svd(joint, compute_uv=False)
    if joint.shape[1] > joint.shape[0] or singular[-1] <= singular[0] * RCOND:
        raise ValueError("joint basis is rank deficient at relative tolerance 1e-8")
    inverse = np.linalg.pinv(joint, rcond=RCOND)
    maps, start = [], 0
    for b in bs:
        end = start + b.shape[1]
        maps.append(b @ inverse[start:end])
        start = end
    return maps


def context_dependence(h00, h10, h01, h11):
    """Compare the first factor's effect across the two contexts in a quartet."""
    arrays = [_matrix(x, "quartet state") for x in (h00, h10, h01, h11)]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("quartet states must have matching shapes")
    a, b, c, d = arrays
    delta0, delta1 = b - a, d - c
    e0 = float(np.mean(np.sum(delta0**2, axis=1)))
    e1 = float(np.mean(np.sum(delta1**2, axis=1)))
    difference = float(np.mean(np.sum((delta0 - delta1) ** 2, axis=1)))
    return {
        "ratio": difference / (e0 + e1) if e0 + e1 else None,
        "difference_energy": difference,
        "effect_energy_0": e0,
        "effect_energy_1": e1,
    }

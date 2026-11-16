"""Natural-history quartets and full residual-state interchange interventions.

Edits replace all positions at one interface. Query-only edits are reported
separately: earlier attention layers can still read the recipient's old history.
The native hybrid state is a positive control for each continuation interface.
"""

import numpy as np
import torch


def make_quartets(process, tokens, pairs, seed=710003):
    """Choose donors differing in every factor history and build natural hybrids.

    Returns recipient/donor row indices, x00/x11, and tuples x10/x01. In x10[f]
    only factor f comes from the donor; x01[f] uses the complementary history.
    BOS positions must agree across rows and are preserved in every hybrid.
    Pairs may repeat; draws are reproducible and never pair a row with itself.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or len(tokens) < 2 or tokens.shape[1] == 0 or pairs < 1:
        raise ValueError(
            "Quartets require at least two nonempty histories and a positive pair count"
        )
    bos = tokens == process.bos_token
    if not np.all(bos == bos[:1]):
        raise ValueError("Quartet histories must have matching BOS positions")
    factors = process.decode(np.where(bos, 0, tokens))
    rng = np.random.default_rng(seed)
    recipients, donors = [], []
    for _ in range(pairs):
        for recipient in rng.permutation(len(tokens)):
            different = np.any(factors != factors[recipient], axis=1).all(axis=-1)
            candidates = np.flatnonzero(different)
            if candidates.size:
                recipients.append(int(recipient))
                donors.append(int(rng.choice(candidates)))
                break
        else:
            raise ValueError(
                "No donor differs in every factor history; provide more diverse histories"
            )
    recipients, donors = np.asarray(recipients), np.asarray(donors)
    x00, x11 = tokens[recipients].copy(), tokens[donors].copy()
    rec, don = factors[recipients], factors[donors]
    x10, x01 = [], []
    for factor in range(process.num_factors):
        one, complement = rec.copy(), don.copy()
        one[..., factor] = don[..., factor]
        complement[..., factor] = rec[..., factor]
        x10.append(np.where(bos[recipients], process.bos_token, process.encode(one)))
        x01.append(np.where(bos[recipients], process.bos_token, process.encode(complement)))
    return {
        "recipients": recipients,
        "donors": donors,
        "x00": x00,
        "x11": x11,
        "x10": tuple(x10),
        "x01": tuple(x01),
    }


def predict(model, tokens, interface=None, states=None, batch_size=128):
    """Return final-position probabilities [batch,vocab], including BOS.

    Supplying both interface and a complete [batch,time,d_model] state replaces
    that interface. The model's original training/eval mode is restored.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or not len(tokens) or not tokens.shape[1] or batch_size < 1:
        raise ValueError("Expected nonempty [batch,time] tokens and positive batch_size")
    if (interface is None) != (states is None):
        raise ValueError("Supply both interface and states, or neither")
    if states is not None:
        states = np.asarray(states)
        if states.ndim != 3 or states.shape[:2] != tokens.shape or not np.isfinite(states).all():
            raise ValueError("Patched states must be finite [batch,time,d_model] arrays")
    parameter = next(model.parameters())
    was_training = model.training
    result = []
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(tokens), batch_size):
                end = start + batch_size
                t = torch.as_tensor(tokens[start:end], dtype=torch.long, device=parameter.device)
                patches = (
                    None
                    if states is None
                    else {
                        interface: torch.as_tensor(
                            states[start:end], dtype=parameter.dtype, device=parameter.device
                        )
                    }
                )
                result.append(model(t, patches=patches)[:, -1].softmax(-1).cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(result).astype(np.float64)


def ordinary_marginals(probabilities, sizes):
    """Condition on an ordinary observation, then marginalize radix axes."""
    size = int(np.prod(sizes))
    probabilities = np.asarray(probabilities, dtype=np.float64)
    p = probabilities[:, :size]
    if probabilities.shape[1] not in (size, size + 1) or np.any(p.sum(axis=1) <= 0):
        raise ValueError("Model vocabulary must equal ordinary vocabulary with optional BOS")
    p = (p / p.sum(axis=1, keepdims=True)).reshape(len(p), *sizes)
    return [
        p.sum(axis=tuple(j + 1 for j in range(len(sizes)) if j != i)) for i in range(len(sizes))
    ]


def _kl(target, candidate):
    target, candidate = np.maximum(target, 1e-15), np.maximum(candidate, 1e-15)
    # Tiny negative roundoff should not produce a negative divergence.
    return float(max(0.0, np.mean(np.sum(target * (np.log(target) - np.log(candidate)), axis=-1))))


def factor_edit(
    model,
    recipient_tokens,
    hybrid_tokens,
    h_rec,
    h_donor,
    h_hybrid,
    projector,
    interface,
    factor,
    sizes,
    batch_size=128,
):
    """Interchange one factor using h_rec + (h_donor-h_rec) @ projector.T.

    Targets are the model's ordinary conditional marginals on natural hybrid
    histories, not the source oracle. Recovery is 1-edited_KL/no_patch_KL; it is
    None below 1e-10 and flagged weak below 1e-5. Other-factor drift compares
    edited marginals with the recipient model. Query variants modify only the
    final position. Orthogonal and two seeded rank-matched random controls use
    the same full donor delta; random directions are independent of the edit.
    """
    arrays = [np.asarray(h, dtype=np.float64) for h in (h_rec, h_donor, h_hybrid)]
    h_rec, h_donor, h_hybrid = arrays
    if (
        h_rec.ndim != 3
        or any(h.shape != h_rec.shape or not np.isfinite(h).all() for h in arrays)
        or h_rec.shape[:2] != np.shape(recipient_tokens)
        or np.shape(hybrid_tokens) != np.shape(recipient_tokens)
    ):
        raise ValueError("States must match each other and the [batch,time] token histories")
    projector = np.asarray(projector, dtype=np.float64)
    if projector.shape != (h_rec.shape[-1], h_rec.shape[-1]) or not np.isfinite(projector).all():
        raise ValueError("projector must be a finite [d_model,d_model] map")
    if not 0 <= factor < len(sizes):
        raise ValueError("factor index is outside sizes")

    def run(states=None, tokens=recipient_tokens):
        return predict(model, tokens, interface if states is not None else None, states, batch_size)

    baseline, target = run(), run(tokens=hybrid_tokens)
    native = run(h_hybrid)
    control_error = float(np.max(np.abs(native - target)))
    if control_error > 2e-5:
        raise ValueError(
            f"Native whole-history control failed at {interface}: max probability error {control_error:.3g}; check cached states and interface"
        )
    rec_probs, target_probs = ordinary_marginals(baseline, sizes), ordinary_marginals(target, sizes)
    baseline_kl = _kl(target_probs[factor], rec_probs[factor])

    def scores(probabilities):
        marginals = ordinary_marginals(probabilities, sizes)
        target_kl = _kl(target_probs[factor], marginals[factor])
        return {
            "target_kl": target_kl,
            "recovery": None if baseline_kl < 1e-10 else 1 - target_kl / baseline_kl,
            "other_factor_drift": [
                None if i == factor else _kl(rec_probs[i], marginals[i]) for i in range(len(sizes))
            ],
        }

    delta = h_donor - h_rec
    edited_delta = delta @ projector.T
    edited = scores(run(h_rec + edited_delta))
    query = h_rec.copy()
    query[:, -1] += edited_delta[:, -1]
    native_query = h_rec.copy()
    native_query[:, -1] = h_hybrid[:, -1]
    left, singular, _ = np.linalg.svd(projector, full_matrices=False)
    rank = int(np.sum(singular > (singular[0] * 1e-8 if len(singular) else 0)))
    span = left[:, :rank]
    controls = {"orthogonal": scores(run(h_rec + delta @ (span @ span.T)))}
    rng = np.random.default_rng(710003 + factor)
    for draw in range(2):
        basis = np.linalg.qr(rng.normal(size=(h_rec.shape[-1], rank)), mode="reduced")[0]
        controls[f"random_{draw}"] = scores(run(h_rec + delta @ (basis @ basis.T)))
    return {
        "no_patch_target_kl": baseline_kl,
        "native_target_kl": scores(native)["target_kl"],
        "edited_target_kl": edited["target_kl"],
        "recovery": edited["recovery"],
        "weak_baseline": baseline_kl < 1e-5,
        "other_factor_drift": edited["other_factor_drift"],
        "query_only": scores(run(query)),
        "native_query_only": scores(run(native_query)),
        "control_max_error": control_error,
        "projector_rank": rank,
        "controls": controls,
    }

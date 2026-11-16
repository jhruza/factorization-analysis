import numpy as np
import pytest
import torch

from factorization.analysis.interventions import (
    factor_edit,
    make_quartets,
    ordinary_marginals,
    predict,
)
from factorization.model import ModelConfig, Transformer
from factorization.processes import Process, canonical_process


def test_natural_quartets_are_reproducible_and_change_only_selected_history():
    process = canonical_process()
    tokens = process.sample(64, 8, seed=301, with_truth=False).tokens
    quartet = make_quartets(process, tokens, 20)
    again = make_quartets(process, tokens, 20)
    np.testing.assert_array_equal(quartet["recipients"], again["recipients"])
    np.testing.assert_array_equal(quartet["donors"], again["donors"])
    assert np.all(quartet["recipients"] != quartet["donors"])
    rec, don = process.decode(quartet["x00"][:, 1:]), process.decode(quartet["x11"][:, 1:])
    assert np.any(rec != don, axis=1).all()
    for factor in range(5):
        one = process.decode(quartet["x10"][factor][:, 1:])
        complement = process.decode(quartet["x01"][factor][:, 1:])
        assert np.all(quartet["x10"][factor][:, 0] == process.bos_token)
        for j in range(5):
            np.testing.assert_array_equal(one[..., j], (don if j == factor else rec)[..., j])
            np.testing.assert_array_equal(complement[..., j], (rec if j == factor else don)[..., j])
    with pytest.raises(ValueError, match="No donor differs"):
        make_quartets(process, np.tile(tokens[0], (4, 1)), 3)


class ModularModel(torch.nn.Module):
    """Two independent factors with additive complete history states."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def states(self, tokens):
        first = torch.nn.functional.one_hot((tokens // 3).clamp(max=1), 2)
        second = torch.nn.functional.one_hot(tokens % 3, 3)
        features = torch.cat((first, second), dim=-1) * (tokens != 6)[..., None]
        return features.float().cumsum(dim=1)

    def forward(self, tokens, patches=None):
        state = self.states(tokens) if patches is None else patches["input"]
        logits = (state[..., :2, None] + state[..., None, 2:]).flatten(-2)
        return torch.cat((logits * self.weight, torch.full_like(logits[..., :1], -100.0)), dim=-1)


def test_selective_edit_recovers_modular_target_without_other_factor_drift():
    process = Process([np.ones((v, 1, 1)) / v for v in (2, 3)], [np.ones(1)] * 2, [np.ones(1)] * 2)
    tokens = process.sample(24, 5, 21, with_truth=False).tokens
    quartet = make_quartets(process, tokens, 10)
    model = ModularModel()
    model.train()
    rec, don, hybrid = quartet["x00"], quartet["x11"], quartet["x10"][0]
    states = [model.states(torch.as_tensor(t)).numpy() for t in (rec, don, hybrid)]
    snapshots = [s.copy() for s in states]
    q = np.diag([1, 1, 0, 0, 0])
    result = factor_edit(model, rec, hybrid, *states, q, "input", 0, process.sizes, batch_size=3)
    assert model.training
    assert result["no_patch_target_kl"] > 1e-5
    assert result["edited_target_kl"] < 1e-12
    assert result["recovery"] == pytest.approx(1)
    assert result["other_factor_drift"][1] < 1e-12
    assert result["query_only"]["target_kl"] < 1e-12
    assert result["control_max_error"] < 1e-7
    assert result["projector_rank"] == 2
    assert result["controls"]["orthogonal"]["target_kl"] < 1e-12
    for name in ("orthogonal", "random_0", "random_1"):
        assert result["controls"][name]["target_kl"] >= 0
    for before, after in zip(snapshots, states, strict=True):
        np.testing.assert_array_equal(before, after)


def test_no_signal_recovery_is_guarded():
    model = ModularModel()
    tokens = np.array([[6, 0, 1], [6, 4, 5]])
    state = model.states(torch.as_tensor(tokens)).numpy()
    result = factor_edit(model, tokens, tokens, state, state, state, np.eye(5), "input", 0, (2, 3))
    assert result["weak_baseline"]
    assert result["recovery"] is None
    assert result["edited_target_kl"] == 0


def test_ordinary_marginals_condition_out_bos_with_radix_order():
    probabilities = np.array([[0.02, 0.04, 0.04, 0.06, 0.08, 0.16, 0.60]])
    first, second = ordinary_marginals(probabilities, (2, 3))
    np.testing.assert_allclose(first, [[0.25, 0.75]])
    np.testing.assert_allclose(second, [[0.20, 0.30, 0.50]])
    np.testing.assert_allclose(probabilities.sum(), 1)


def test_whole_state_control_at_every_transformer_interface_and_last_mlp_query():
    model = Transformer(ModelConfig(d_model=12, d_head=4, n_heads=3, n_layers=2, d_mlp=24, n_ctx=5))
    process = canonical_process()
    quartet = make_quartets(process, process.sample(10, 5, 62, with_truth=False).tokens, 3)
    rec, hybrid = quartet["x00"], quartet["x10"][0]
    with torch.inference_mode():
        _, cached = model.capture(torch.as_tensor(rec))
        _, natural = model.capture(torch.as_tensor(hybrid))
    target = predict(model, hybrid, batch_size=2)
    for interface in cached:
        patched = predict(model, rec, interface, natural[interface].numpy(), batch_size=2)
        np.testing.assert_allclose(patched, target, atol=1e-7)
    rec_state, hybrid_state = cached["L2_mlp"].numpy(), natural["L2_mlp"].numpy()
    result = factor_edit(
        model,
        rec,
        hybrid,
        rec_state,
        hybrid_state,
        hybrid_state,
        np.eye(12),
        "L2_mlp",
        0,
        process.sizes,
        batch_size=2,
    )
    assert result["native_query_only"]["target_kl"] < 1e-12
    assert result["query_only"]["target_kl"] < 1e-12
    with pytest.raises(ValueError, match="Native whole-history control failed"):
        factor_edit(
            model,
            rec,
            hybrid,
            rec_state,
            hybrid_state,
            rec_state,
            np.eye(12),
            "L2_mlp",
            0,
            process.sizes,
            batch_size=2,
        )

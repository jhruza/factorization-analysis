import torch

from factorization.model import ModelConfig, Transformer


def small_model():
    return Transformer(
        ModelConfig(d_model=12, d_head=4, n_heads=3, n_layers=2, n_ctx=6, d_mlp=24, d_vocab=17),
        seed=7,
    )


def test_causal_attention_and_capture():
    model = small_model()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 6, 7]])
    logits, states = model.capture(first)
    changed_logits, changed_states = model.capture(changed_future)
    assert list(states) == ["input", "L1_attn", "L1_mlp", "L2_attn", "L2_mlp", "final"]
    assert logits.shape == (1, 5, 17)
    torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])
    for name in states:
        torch.testing.assert_close(states[name][:, :3], changed_states[name][:, :3])


def test_patching_restores_downstream_states_and_supports_gradients():
    model = small_model()
    source = torch.tensor([[1, 2, 3, 4]])
    target = torch.tensor([[5, 6, 7, 8]])
    expected, states = model.capture(source)
    replacement = states["L1_mlp"].detach().clone().requires_grad_()
    actual, patched_states = model.capture(target, {"L1_mlp": replacement})
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(patched_states["L1_mlp"], replacement)
    actual.sum().backward()
    assert replacement.grad is not None and replacement.grad.abs().sum() > 0
    torch.testing.assert_close(model(target, {"final": states["final"]}), expected)


def test_seed_and_checkpoint_round_trip():
    a, b = small_model(), small_model()
    assert all(torch.equal(value, b.state_dict()[name]) for name, value in a.state_dict().items())
    tokens = torch.tensor([[1, 2, 3]])
    b.load_reference_state(a.state_dict())
    torch.testing.assert_close(a(tokens), b(tokens))


def test_invalid_patches_rejected():
    model = small_model()
    tokens = torch.tensor([[1, 2, 3]])
    for patches in ({"unknown": torch.zeros(1)}, {"input": torch.zeros(1)}):
        try:
            model(tokens, patches)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid patch accepted")

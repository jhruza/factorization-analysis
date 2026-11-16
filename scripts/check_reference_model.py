"""Compare the native model with the source checkpoint (requires TransformerLens)."""

import json
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from factorization.model import ModelConfig, Transformer  # noqa: E402


def main():
    torch.set_num_threads(1)
    c = ModelConfig()
    cfg = HookedTransformerConfig(
        d_model=c.d_model,
        d_head=c.d_head,
        n_heads=c.n_heads,
        n_layers=c.n_layers,
        n_ctx=c.n_ctx,
        d_mlp=c.d_mlp,
        d_vocab=c.d_vocab,
        act_fn="relu",
        normalization_type="LN",
        device="cpu",
        seed=7,
    )
    reference = HookedTransformer(cfg)
    native = Transformer(c, seed=7)
    initial_errors = {
        k: (v - native.state_dict()[k]).abs().max().item()
        for k, v in reference.state_dict().items()
        if v.is_floating_point() and torch.isfinite(v).all()
    }
    checkpoint = (
        ROOT.parent
        / "preliminary-results/2026-11-05-global-pca-belief-reconstruction/checkpoints/step_6000.pt"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference.load_state_dict(state, strict=True)
    native.load_reference_state(state)
    generator = torch.Generator().manual_seed(991)
    tokens = torch.randint(c.d_vocab, (4, c.n_ctx), generator=generator)
    with torch.no_grad():
        expected, cache = reference.run_with_cache(tokens)
        actual, states = native.capture(tokens)
    names = {"input": "blocks.0.hook_resid_pre", "final": "ln_final.hook_normalized"}
    for i in range(1, c.n_layers + 1):
        names[f"L{i}_attn"] = f"blocks.{i - 1}.hook_resid_mid"
        names[f"L{i}_mlp"] = f"blocks.{i - 1}.hook_resid_post"
    errors = {name: (states[name] - cache[key]).abs().max().item() for name, key in names.items()}
    errors["logits"] = (actual - expected).abs().max().item()
    result = {
        "checkpoint": str(checkpoint.relative_to(ROOT.parent)),
        "seed": 7,
        "token_seed": 991,
        "batch_shape": list(tokens.shape),
        "initialization_max_abs_error": max(initial_errors.values()),
        "checkpoint_max_abs_errors": errors,
        "tolerance": 1e-4,
        "passed": max(errors.values()) < 1e-4,
    }
    output = ROOT / "scripts/reference_model_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Checkpoint parity failed")


if __name__ == "__main__":
    main()

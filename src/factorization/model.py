"""A small decoder matching the canonical factored-reps TransformerLens model.

Architecture and parameter conventions follow TransformerLens HookedTransformer,
used by factored-reps (https://github.com/TransformerLensOrg/TransformerLens).
This implementation does not require TransformerLens at runtime.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 120
    d_head: int = 40
    n_heads: int = 3
    n_layers: int = 4
    n_ctx: int = 9
    d_mlp: int = 480
    d_vocab: int = 433
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        dimensions = (
            self.d_model,
            self.d_head,
            self.n_heads,
            self.n_layers,
            self.n_ctx,
            self.d_mlp,
            self.d_vocab,
        )
        if any(not isinstance(size, int) or size < 1 for size in dimensions):
            raise ValueError("All model dimensions must be positive integers")
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be finite and positive")


class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w = nn.Parameter(torch.ones(config.d_model))
        self.b = nn.Parameter(torch.zeros(config.d_model))
        self.eps = config.layer_norm_eps

    def forward(self, x: Tensor) -> Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt() * self.w + self.b


class Embedding(nn.Module):
    def __init__(self, shape: tuple[int, int], name: str):
        super().__init__()
        self.register_parameter(name, nn.Parameter(torch.empty(shape)))


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.W_Q = nn.Parameter(torch.empty(c.n_heads, c.d_model, c.d_head))
        self.W_O = nn.Parameter(torch.empty(c.n_heads, c.d_head, c.d_model))
        self.b_Q = nn.Parameter(torch.zeros(c.n_heads, c.d_head))
        self.b_O = nn.Parameter(torch.zeros(c.d_model))
        self.W_K = nn.Parameter(torch.empty(c.n_heads, c.d_model, c.d_head))
        self.W_V = nn.Parameter(torch.empty(c.n_heads, c.d_model, c.d_head))
        self.b_K = nn.Parameter(torch.zeros(c.n_heads, c.d_head))
        self.b_V = nn.Parameter(torch.zeros(c.n_heads, c.d_head))
        self.register_buffer("mask", torch.ones(c.n_ctx, c.n_ctx, dtype=torch.bool).tril())
        self.register_buffer("IGNORE", torch.tensor(-torch.inf))
        self.scale = math.sqrt(c.d_head)
        self.n_heads, self.d_head = c.n_heads, c.d_head

    def forward(self, x: Tensor) -> Tensor:
        def project(weight: Tensor, bias: Tensor) -> Tensor:
            return F.linear(
                x, weight.permute(0, 2, 1).reshape(-1, x.shape[-1]), bias.reshape(-1)
            ).reshape(*x.shape[:2], self.n_heads, self.d_head)

        q, k, v = (
            project(self.W_Q, self.b_Q),
            project(self.W_K, self.b_K),
            project(self.W_V, self.b_V),
        )
        scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1) / self.scale
        length = x.shape[1]
        pattern = scores.masked_fill(~self.mask[:length, :length], -torch.inf).softmax(dim=-1)
        z = (pattern @ v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return F.linear(z.reshape(*x.shape[:2], -1), self.W_O.reshape(-1, x.shape[-1]).T, self.b_O)


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(c.d_model, c.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(c.d_mlp))
        self.W_out = nn.Parameter(torch.empty(c.d_mlp, c.d_model))
        self.b_out = nn.Parameter(torch.zeros(c.d_model))

    def forward(self, x: Tensor) -> Tensor:
        hidden = torch.addmm(self.b_in, x.reshape(-1, x.shape[-1]), self.W_in).relu()
        return torch.addmm(self.b_out, hidden, self.W_out).reshape(*x.shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(c)
        self.ln2 = LayerNorm(c)
        self.attn = Attention(c)
        self.mlp = MLP(c)


class Transformer(nn.Module):
    """Causal transformer with replaceable residual-stream interfaces.

    Interfaces are full [batch, position, d_model] tensors. ``input`` is the
    embedding sum, Lx_attn and Lx_mlp are post-residual-addition states, and
    ``final`` is the output of final layer normalization. Replacements must
    have the same shape, device and dtype as the state they replace.
    """

    def __init__(self, config: ModelConfig = ModelConfig(), seed: int = 7):
        super().__init__()
        self.config = config
        self.embed = Embedding((config.d_vocab, config.d_model), "W_E")
        self.pos_embed = Embedding((config.n_ctx, config.d_model), "W_pos")
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_final = LayerNorm(config)
        self.unembed = Embedding((config.d_model, config.d_vocab), "W_U")
        self.unembed.b_U = nn.Parameter(torch.zeros(config.d_vocab))
        # TransformerLens seeds only its initialization, in named parameter order.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for name, parameter in self.named_parameters():
                if "W_" in name:
                    nn.init.normal_(parameter, std=0.8 / math.sqrt(config.d_model))

    def _run(self, tokens: Tensor, patches: dict[str, Tensor] | None, capture: bool):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.n_ctx:
            raise ValueError("tokens must have shape [batch, position] with 1 <= position <= n_ctx")
        patches = {} if patches is None else patches
        valid = {"input", "final"} | {
            f"L{i}_{part}" for i in range(1, self.config.n_layers + 1) for part in ("attn", "mlp")
        }
        if patches.keys() - valid:
            raise ValueError(f"unknown patch interfaces: {sorted(patches.keys() - valid)}")
        states = {}

        def interface(name: str, value: Tensor) -> Tensor:
            if name in patches:
                replacement = patches[name]
                if (
                    replacement.shape != value.shape
                    or replacement.device != value.device
                    or replacement.dtype != value.dtype
                ):
                    raise ValueError(
                        f"patch {name} must match the shape, device, and dtype of its interface"
                    )
                value = replacement
            if capture:
                states[name] = value
            return value

        x = interface("input", self.embed.W_E[tokens] + self.pos_embed.W_pos[: tokens.shape[1]])
        for i, block in enumerate(self.blocks, 1):
            x = interface(f"L{i}_attn", x + block.attn(block.ln1(x)))
            x = interface(f"L{i}_mlp", x + block.mlp(block.ln2(x)))
        x = interface("final", self.ln_final(x))
        return x @ self.unembed.W_U + self.unembed.b_U, states

    def forward(self, tokens: Tensor, patches: dict[str, Tensor] | None = None) -> Tensor:
        return self._run(tokens, patches, False)[0]

    def capture(
        self, tokens: Tensor, patches: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self._run(tokens, patches, True)

    def load_reference_state(self, state_dict: dict[str, Tensor]):
        """Load a canonical TransformerLens checkpoint, checking every key."""
        return self.load_state_dict(state_dict, strict=True)

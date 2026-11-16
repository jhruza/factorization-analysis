"""Small NumPy implementation of the independent five-factor source process.

The Mess3 and Tom Quantum mathematical definitions and state conventions follow
factored-reps/fwh_core/generative_processes/{transition_matrices,builder,
generalized_hidden_markov_model}.py and utils/factoring_utils.py. This module is
an independent implementation; it does not import the reference project.

"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .data import Batch, Truth


def mess3(x: float = 0.15, a: float = 0.6) -> np.ndarray:
    """Return the three observation operators, indexed [symbol,from,to]."""
    if not (0 <= x <= 0.5 and 0 <= a <= 1):
        raise ValueError("Mess3 requires 0 <= x <= .5 and 0 <= a <= 1")
    transition = np.full((3, 3), x, dtype=np.float64)
    np.fill_diagonal(transition, 1 - 2 * x)
    emissions = np.full((3, 3), (1 - a) / 2, dtype=np.float64)
    np.fill_diagonal(emissions, a)
    return transition[None, :, :] * emissions[:, None, :]


def tom_quantum(alpha: float = 1.0, beta: float = 3.0) -> np.ndarray:
    """Four signed GHMM operators in the reference's three-coordinate basis."""
    if alpha * alpha + beta * beta == 0:
        raise ValueError("alpha and beta cannot both be zero")
    scale = 1 / (4 * (alpha * alpha + beta * beta))
    middle = (alpha * alpha - beta * beta) * scale
    off = 2 * alpha * beta * scale
    operators = np.tile(np.diag([0.25, middle, middle]), (4, 1, 1))
    for symbol, (axis, sign) in enumerate(((2, 1), (2, -1), (1, 1), (1, -1))):
        operators[symbol, axis, axis] = 0.25
        operators[symbol, 0, axis] = sign * off
        operators[symbol, axis, 0] = sign * off
    return operators


class Process:
    """Independent factors with row-state update s' = s T_o / (s T_o n).

    HMM factors use n=ones. Quantum factors use n=[3,0,0], preserving
    negative coordinates. The raw source prior [1,0,0] is deliberately retained
    at BOS; after the first observation its first coordinate is 1/3.
    Randomness is local to sample(seed), using NumPy rather than JAX's stream.
    """

    def __init__(
        self,
        transition_matrices: Sequence[np.ndarray],
        initial_states: Sequence[np.ndarray],
        normalizing_eigenvectors: Sequence[np.ndarray],
    ) -> None:
        self.transition_matrices = tuple(
            np.asarray(t, dtype=np.float64) for t in transition_matrices
        )
        self.initial_states = tuple(np.asarray(s, dtype=np.float64) for s in initial_states)
        self.normalizing_eigenvectors = tuple(
            np.asarray(n, dtype=np.float64) for n in normalizing_eigenvectors
        )
        if not self.transition_matrices or not (
            len(self.transition_matrices)
            == len(self.initial_states)
            == len(self.normalizing_eigenvectors)
        ):
            raise ValueError("Supply one operator tensor, initial state, and normalizer per factor")
        for t, s, n in zip(
            self.transition_matrices, self.initial_states, self.normalizing_eigenvectors
        ):
            if t.ndim != 3 or t.shape[1:] != (s.size, s.size) or s.shape != n.shape or s.ndim != 1:
                raise ValueError("Incompatible transition/state/normalizer shapes")
            if s @ n <= 0:
                raise ValueError("Initial state must have positive normalization")
        self.sizes = tuple(t.shape[0] for t in self.transition_matrices)
        self.num_factors = len(self.sizes)
        self.vocab_size = int(np.prod(self.sizes))
        self.bos_token = self.vocab_size
        self.radix = np.array([int(np.prod(self.sizes[i + 1 :])) for i in range(self.num_factors)])

    def encode(self, factors: np.ndarray) -> np.ndarray:
        """Encode [...,factor] symbols as [...]; factor zero is most significant."""
        values = np.asarray(factors)
        if values.ndim < 1 or values.shape[-1] != self.num_factors:
            raise ValueError("Expected a final axis containing one symbol per factor")
        if (
            not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
            or np.any(values >= self.sizes)
        ):
            raise ValueError("Factor symbols must be integers in their vocabulary ranges")
        return (values * self.radix).sum(axis=-1, dtype=np.int64)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode observation tokens into [...,factor]; BOS has no factor tuple."""
        values = self._tokens(tokens, allow_bos=False)
        return (values[..., None] // self.radix) % self.sizes

    def _tokens(self, tokens: np.ndarray, *, allow_bos: bool = True) -> np.ndarray:
        values = np.asarray(tokens)
        limit = self.vocab_size + int(allow_bos)
        if (
            not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
            or np.any(values >= limit)
        ):
            raise ValueError(f"Tokens must be integers in [0,{limit})")
        return values.astype(np.int64, copy=False)

    def _prior(self, batch_size: int) -> list[np.ndarray]:
        return [np.broadcast_to(s, (batch_size, s.size)).copy() for s in self.initial_states]

    def _probs(self, state: np.ndarray, i: int) -> np.ndarray:
        n = self.normalizing_eigenvectors[i]
        result = (
            np.einsum("bi,vij,j->bv", state, self.transition_matrices[i], n) / (state @ n)[:, None]
        )
        # Match upstream's numerical guard, without clipping signed state vectors.
        return np.maximum(result, 0)

    def _step(self, state: np.ndarray, symbols: np.ndarray, i: int) -> np.ndarray:
        result = np.einsum("bi,bij->bj", state, self.transition_matrices[i][symbols])
        return result / (result @ self.normalizing_eigenvectors[i])[:, None]

    def evaluate(self, tokens: np.ndarray, joint: Literal[False, "last", "all"] = False) -> Truth:
        """Filter [batch,time] input tokens, returning truth after each token.

        BOS resets that row to the source prior, wherever it appears. With no
        BOS, filtering starts from the prior and consumes the first observation.
        Joint probabilities use radix token ordering and can be memory-heavy.
        """
        values = self._tokens(tokens)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("Expected tokens shaped [batch,positive time]")
        if joint not in (False, "last", "all"):
            raise ValueError("joint must be False, 'last', or 'all'")
        batch, time = values.shape
        states = self._prior(batch)
        beliefs = tuple(np.empty((batch, time, s.size)) for s in self.initial_states)
        probs = tuple(np.empty((batch, time, v)) for v in self.sizes)
        for t in range(time):
            reset = values[:, t] == self.bos_token
            symbols = self.decode(np.where(reset, 0, values[:, t]))
            for i in range(self.num_factors):
                states[i] = self._step(states[i], symbols[:, i], i)
                states[i][reset] = self.initial_states[i]
                beliefs[i][:, t] = states[i]
                probs[i][:, t] = self._probs(states[i], i)
        joint_probs = None
        if joint:
            distributions = [p[:, -1] if joint == "last" else p for p in probs]
            joint_probs = distributions[0]
            for p in distributions[1:]:
                joint_probs = (joint_probs[..., :, None] * p[..., None, :]).reshape(
                    *p.shape[:-1], -1
                )
        return Truth(beliefs, probs, joint_probs)

    def sample(self, batch_size: int, seq_len: int, seed: int, with_truth: bool = True) -> Batch:
        """Draw independent histories; inputs and next-token labels have equal length."""
        if batch_size < 1 or seq_len < 1:
            raise ValueError("batch_size and seq_len must be positive")
        rng = np.random.default_rng(seed)
        states = self._prior(batch_size)
        observations = np.empty((batch_size, seq_len), dtype=np.int64)
        beliefs = (
            tuple(np.empty((batch_size, seq_len, s.size)) for s in self.initial_states)
            if with_truth
            else None
        )
        probs = (
            tuple(np.empty((batch_size, seq_len, v)) for v in self.sizes) if with_truth else None
        )
        for t in range(seq_len):
            symbols = np.empty((batch_size, self.num_factors), dtype=np.int64)
            for i in range(self.num_factors):
                p = self._probs(states[i], i)
                if with_truth:
                    beliefs[i][:, t] = states[i]
                    probs[i][:, t] = p
                # Inverse CDF avoids a Python loop over examples. Renormalization
                # only removes roundoff, as all process probabilities sum to 1.
                cdf = np.cumsum(p / p.sum(axis=-1, keepdims=True), axis=-1)
                symbols[:, i] = np.minimum(
                    (rng.random(batch_size)[:, None] > cdf).sum(axis=-1), self.sizes[i] - 1
                )
                states[i] = self._step(states[i], symbols[:, i], i)
            observations[:, t] = self.encode(symbols)
        tokens = np.concatenate(
            (np.full((batch_size, 1), self.bos_token), observations[:, :-1]), axis=1
        )
        return Batch(tokens, observations, beliefs, probs)


def canonical_process() -> Process:
    """Three Mess3(.15,.6) factors followed by two Tom Quantum(1,3) factors."""
    return Process(
        [mess3() for _ in range(3)] + [tom_quantum() for _ in range(2)],
        [np.full(3, 1 / 3) for _ in range(3)] + [np.array([1.0, 0.0, 0.0]) for _ in range(2)],
        [np.ones(3) for _ in range(3)] + [np.array([3.0, 0.0, 0.0]) for _ in range(2)],
    )

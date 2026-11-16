"""NumPy data containers shared by generation, training, and analysis."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Truth:
    """Exact state and next-symbol distribution at each input position.

    Each tuple contains one array per factor, shaped [batch, time, state/vocab].
    Quantum states are signed linear states, not categorical probabilities.
    """

    beliefs: tuple[np.ndarray, ...]
    factor_probs: tuple[np.ndarray, ...]
    joint_probs: np.ndarray | None = None


@dataclass
class Batch:
    """Inputs BOS,o0,... and targets o0,o1,..., both shaped [batch,time]."""

    tokens: np.ndarray
    targets: np.ndarray
    beliefs: tuple[np.ndarray, ...] | None = None
    factor_probs: tuple[np.ndarray, ...] | None = None

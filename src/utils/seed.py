"""Reproducibility utilities for Python, NumPy, PyTorch, and Qiskit."""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

try:
    import torch
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    torch = None  # type: ignore[assignment]

try:
    from qiskit.utils import algorithm_globals
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    algorithm_globals = None  # type: ignore[assignment]

log = get_logger(__name__)


def set_global_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """Set global random seeds across supported libraries.

    Parameters
    ----------
    seed : int, default=42
        Seed value applied to Python, NumPy, PyTorch, and Qiskit.
    deterministic_torch : bool, default=True
        If ``True``, enables deterministic CuDNN behavior where available.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if algorithm_globals is not None:
        algorithm_globals.random_seed = seed

    log.info(f"Global random seed set to {seed}.")

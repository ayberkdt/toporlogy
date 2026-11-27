"""Small argument checks shared by the time-dependent mesh solvers (cure_slab, nonlinear_heat)."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

Scalar = float | Callable[[float], float]


def positive(x: Any, what: str, allow_zero: bool = False) -> float:
    """``x`` as a finite float > 0 (or >= 0 with ``allow_zero``); ValueError naming ``what`` otherwise."""
    v = float(x)
    if not math.isfinite(v) or v < 0.0 or (v == 0.0 and not allow_zero):
        raise ValueError(f"{what} must be finite and {'>= 0' if allow_zero else '> 0'}, got {x!r}")
    return v


def time_function(v: Scalar, what: str) -> Callable[[float], float]:
    """A callable of time t: ``v`` itself if callable, else the constant ``float(v)`` (finite)."""
    if callable(v):
        return v
    c = float(v)
    if not math.isfinite(c):
        raise ValueError(f"{what} must be finite, got {v!r}")
    return lambda t: c


def max_iterations(n: Any) -> int:
    """``n`` as an int >= 1 (bool refused); ValueError otherwise."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"max_iter must be an integer >= 1, got {n!r}")
    return int(n)

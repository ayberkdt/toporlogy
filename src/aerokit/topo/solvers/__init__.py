"""Linear solvers: sparse direct (PARDISO / SuperLU) and matrix-free geometric multigrid (CUDA / CPU).

The multigrid names are imported lazily: :mod:`aerokit.topo.solvers.gmg` imports torch, and this package is imported
by ``aerokit.topo`` before MKL PARDISO is loaded (torch first would abort the process, see ``aerokit/topo/__init__``).
"""
from typing import Any

from aerokit.topo.solvers.direct import DirectSolver

__all__ = ["DirectSolver", "GMGSolver", "GMGConvergenceError"]


def __getattr__(name: str) -> Any:
    if name in ("GMGSolver", "GMGConvergenceError"):
        from aerokit.topo.solvers import gmg
        return getattr(gmg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Linear solvers: sparse direct (PARDISO / SuperLU) and matrix-free geometric multigrid (CUDA / CPU).

The multigrid names are imported lazily: :mod:`toporlogy.solvers.gmg` imports torch, and this package is imported
by ``toporlogy`` before MKL PARDISO is loaded (torch first would abort the process, see ``toporlogy/__init__``).
"""
from typing import Any

from toporlogy.solvers.direct import DirectSolver

__all__ = ["DirectSolver", "GMGSolver", "GMGConvergenceError"]


def __getattr__(name: str) -> Any:
    if name in ("GMGSolver", "GMGConvergenceError"):
        from toporlogy.solvers import gmg
        return getattr(gmg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

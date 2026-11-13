"""Locate and load Intel MKL PARDISO (pypardiso) without depending on a fixed install path.

The MKL / Intel OpenMP DLLs are looked for in, in this order: ``$AEROKIT_MKL_BIN``, then a ``mkl_bin`` folder next to
any ``sys.path`` entry that contains the ``pypardiso`` package. Load-order constraint (measured 13 Nov 2026 on
Windows): torch and MKL each ship ``libiomp5md.dll``; loading MKL after torch aborts the process ("OMP: Error #15"),
loading it first works. :func:`load` therefore refuses to load MKL once torch is imported.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp

__all__ = ["load", "available", "state"]

_STATE: dict[str, Any] = {"module": None, "tried": False, "reason": ""}


def _dll_folders() -> list[str]:
    out = []
    env = os.environ.get("AEROKIT_MKL_BIN")
    if env:
        out.append(env)
    for p in sys.path:
        if p and os.path.isdir(os.path.join(p, "pypardiso")):
            cand = os.path.join(p, "mkl_bin")
            if os.path.isdir(cand):
                out.append(cand)
    return out


def load() -> Any:
    """The pypardiso module with the MKL DLLs loadable, or None (reason in :func:`state`)."""
    if _STATE["tried"]:
        return _STATE["module"]
    if "torch" in sys.modules:
        _STATE.update(tried=False, reason="torch already imported: MKL would abort the process; import "
                      "aerokit.topo before torch")
        return None
    _STATE["tried"] = True
    try:
        for folder in _dll_folders():
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(folder)
            rt = [f for f in os.listdir(folder) if f.lower().startswith("mkl_rt")]
            if rt:
                os.environ.setdefault("PYPARDISO_MKL_RT", os.path.join(folder, rt[0]))
        import pypardiso
        A = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 4.0]]))
        pypardiso.PyPardisoSolver(mtype=2).solve(sp.triu(A, format="csr"), np.ones(2))
        _STATE.update(module=pypardiso, reason="ok")
    except Exception as exc:  # noqa: BLE001 - optional accelerator
        _STATE.update(module=None, reason=f"{type(exc).__name__}: {exc}")
    return _STATE["module"]


def available() -> bool:
    return load() is not None


def state() -> dict[str, Any]:
    return {"loaded": _STATE["module"] is not None, "tried": _STATE["tried"], "reason": _STATE["reason"]}

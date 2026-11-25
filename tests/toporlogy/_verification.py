"""What a test run did and did not verify: the environment, the optional backends and the baseline comparison mode.

Filled during the run (conftest hooks, test_baseline) and reported at its end (terminal section "verification
status"; JSON to the file named by the environment variable TOPORLOGY_ENV_REPORT). A backend whose tests were all
skipped or not selected is NOT VERIFIED; it is never reported as passed.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

BACKENDS = {
    "pardiso": "PARDISO direct solver (pypardiso + MKL)",
    "gmsh": "gmsh tetrahedral meshing and T10 re-analysis",
    "cuda": "CUDA (GPU multigrid)",
}
OUTCOMES: dict[str, list[tuple[str, str, str]]] = {b: [] for b in BACKENDS}     # backend -> (test, outcome, reason)
BASELINE: dict[str, Any] = {}                                                    # filled by test_baseline


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    """Versions and machine facts that decide whether a numerical result can be reproduced bit for bit."""
    import numpy as np
    import scipy
    import torch

    from toporlogy.solvers import pardiso
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    blas: Any = None
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas")
    except Exception:  # noqa: BLE001 - older numpy without mode="dicts": recorded as unknown
        blas = None
    return {
        "git_sha": git.stdout.strip() or None,
        "python": platform.python_version(), "system": platform.system(), "arch": platform.machine(),
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()), "blas": blas,
        "pardiso": pardiso.state(),
        "optional": {d: _version(d) for d in ("pypardiso", "gmsh", "pyvista", "vtk", "numba")},
    }


def backend_status() -> dict[str, dict[str, Any]]:
    out = {}
    for b, rows in OUTCOMES.items():
        passed = [t for t, o, _ in rows if o == "passed"]
        failed = [t for t, o, _ in rows if o == "failed"]
        skipped = [(t, r) for t, o, r in rows if o == "skipped"]
        if failed:
            status = "FAILED"
        elif passed:
            status = "VERIFIED"
        else:
            status = "NOT VERIFIED"
        reason = "" if passed or failed else (skipped[0][1] if skipped else "no test of this backend was selected")
        out[b] = {"status": status, "passed": len(passed), "failed": len(failed), "skipped": len(skipped),
                  "reason": reason, "what": BACKENDS[b]}
    return out


# ---------------------------------------------------------------------------------------------- collected items
# Every collected item with its xfail / skipif markers, BEFORE the -m / -k deselection (conftest hook, tryfirst).
# Read by tests/toporlogy/test_catalog_ledger.py (the strict-xfail ledger docs/vv_kutugu.json must match it).
COLLECTED: list[dict[str, Any]] = []


def item_record(item: Any) -> dict[str, Any]:
    strict_default = bool(item.config.getini("xfail_strict"))
    xf = [{"strict": bool(m.kwargs.get("strict", strict_default)), "reason": str(m.kwargs.get("reason", ""))}
          for m in item.iter_markers("xfail")]
    sk = [str(m.kwargs.get("reason", "")) for m in item.iter_markers("skipif")] + \
         [str(m.kwargs.get("reason", "")) for m in item.iter_markers("skip")]
    return {"nodeid": item.nodeid, "xfail": xf, "skip_markers": sk,
            "slow": item.get_closest_marker("slow") is not None}

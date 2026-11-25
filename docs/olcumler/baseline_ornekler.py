"""Baseline record of the example studies: environment, objective/constraints, residuals, time and peak memory.

00_MASTER "Başlangıç denetimi": at least seven existing examples with objective/constraint, design, solution
residual, time, memory and environment. Every example runs its configured ``run.max_iter`` (not the 10-iteration
cut of ``tests/toporlogy/test_baseline.py``) in a fresh process with torch imported before toporlogy, so the direct
solver is SuperLU on every machine (PARDISO is never loaded; see ``toporlogy/__init__.py``).

    python docs/olcumler/baseline_ornekler.py docs/olcumler/baseline_ornekler.json            # all examples
    python docs/olcumler/baseline_ornekler.py out.json --only mbb2d inverter2d                   # a subset

What is recorded per example (``examples[name]``):

* ``status``, ``termination``, ``iterations``, ``objective``, ``constraints`` (value, upper, satisfied),
  ``max_violation``, ``kkt_residual``, ``stationarity``, ``grey_fraction`` from ``summary.json``;
* ``state_residual``: ``physics.last_residual`` after the run, the worst load case ‖K u − f‖ / ‖f‖ of the last
  linear solve (``None`` when the physics does not record it);
* ``design``: SHA-256 of ``xphys`` (dtype, shape and bytes), its mean and the number of design variables;
* ``wall_seconds`` (whole child process body: build + run + post), ``optimizer_seconds`` (``summary.seconds``),
  ``peak_memory_bytes`` of the child process and how it was measured.

The times are ONE run each on a machine that may be loaded by other work: they are a record of this run, not a
performance benchmark (08 §2: no repeats, no idle machine, no warm-up).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((REPO / "examples").glob("*.yaml")) + sorted((REPO / "examples" / "research").glob("*.yaml"))
TIMEOUT = 3600

CHILD = r'''
import json, sys, time
t0 = time.perf_counter()
import torch                                            # before toporlogy: SuperLU, never PARDISO
import hashlib
import numpy as np
from pathlib import Path
from toporlogy.configuration.build import build_setup
from toporlogy.configuration.resolve import read_config
from toporlogy.execution.runner import run_setup

src, folder, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
setup = build_setup(read_config(src))
sm = run_setup(setup, folder, source=src)
st = setup.study
z = np.load(folder / "result.npz")
xphys = np.ascontiguousarray(z["xphys"])
sha = hashlib.sha256(f"{xphys.dtype.str}{xphys.shape}".encode() + xphys.tobytes()).hexdigest()
res = getattr(st.physics, "last_residual", None)
wall = time.perf_counter() - t0
peak, how = None, "not available"
try:
    import psutil
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):
        peak, how = int(mi.peak_wset), "psutil peak_wset (Windows peak working set)"
except ImportError:
    pass
if peak is None:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(r) * (1 if sys.platform == "darwin" else 1024)
        how = "resource.getrusage ru_maxrss"
    except ImportError:
        pass
row = {k: sm.get(k) for k in ("status", "termination", "iterations", "max_iter", "converged", "objective_name",
                              "objective", "max_violation", "kkt_residual", "stationarity", "grey_fraction", "verdict",
                              "flags")}
row["constraints"] = [{k: c.get(k) for k in ("name", "value", "upper", "verdict")} for c in sm["constraints"]]
row["post"] = {k: (v.get("passed") if isinstance(v, dict) else None) for k, v in sm.get("post", {}).items()}
row["physics"] = type(st.physics).__name__
row["optimizer"] = type(st.optimizer).__name__
row["n_active"] = int(st.domain.n_active)
row["ndof"] = int(st.domain.ndof)
row["state_residual"] = None if res is None else float(res)
row["design"] = {"xphys_sha256": sha, "xphys_mean": float(xphys.mean()), "n_design": int(xphys.size)}
row["optimizer_seconds"] = sm.get("seconds")
row["wall_seconds"] = wall
row["peak_memory_bytes"] = peak
row["peak_memory_method"] = how
Path(out).write_text(json.dumps(row, default=str), encoding="utf-8")
'''


def _git(*args: str) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return p.stdout.strip()


def environment() -> dict[str, Any]:
    """Versions, BLAS, threads and machine facts (the reproducibility record of 08 §2)."""
    import importlib.metadata as md

    import numpy as np
    import scipy
    import torch

    def ver(d: str) -> str | None:
        try:
            return md.version(d)
        except md.PackageNotFoundError:
            return None

    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas")
        lapack = cfg.get("Build Dependencies", {}).get("lapack")
    except Exception:  # noqa: BLE001 - recorded as unknown
        blas = lapack = None
    mem = None
    try:
        import psutil
        mem = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    return {
        "git_sha": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "platform": platform.platform(), "arch": platform.machine(),
        "processor": platform.processor(), "cpu_count_logical": os.cpu_count(), "memory_total_bytes": mem,
        "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_blas": blas, "numpy_lapack": lapack,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_parallel_info": torch.__config__.parallel_info().splitlines()[:12],
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "NUMEXPR_NUM_THREADS", "TOPORLOGY_PYLIB")},
        "packages": {d: ver(d) for d in ("pyyaml", "pypardiso", "mkl", "gmsh", "pyvista", "vtk", "numba",
                                         "scikit-fem", "matplotlib", "pytest", "hypothesis", "mypy", "ruff", "psutil")},
        "solver": "SuperLU (torch imported before toporlogy; PARDISO not loaded)",
        "float_dtype": "float64",
    }


def run_one(path: Path, workdir: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    out = workdir / f"{path.stem}.row.json"
    t = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", CHILD, str(path), str(workdir / path.stem), str(out)], env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)
    elapsed = time.perf_counter() - t
    if p.returncode != 0 or not out.is_file():
        return {"error": f"exit {p.returncode}", "stderr_tail": p.stderr[-3000:], "process_seconds": elapsed}
    row: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    row["process_seconds"] = elapsed
    row["source"] = path.relative_to(REPO).as_posix()
    row["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="JSON file to write")
    ap.add_argument("--only", nargs="*", default=None, help="example stems to run (default: all)")
    a = ap.parse_args(argv)
    chosen = [p for p in EXAMPLES if a.only is None or p.stem in a.only]
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = environment()                 # before the runs: git state as it was when they started
    rows: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as d:
        for ex in chosen:
            rows[ex.stem] = run_one(ex, Path(d))
            r = rows[ex.stem]
            print(f"{ex.stem:24s} {r.get('status', r.get('error')):16s} {r.get('process_seconds', 0):7.1f} s",
                  flush=True)
    data = {"meta": {"started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "command": "python docs/olcumler/baseline_ornekler.py " + " ".join(argv or sys.argv[1:]),
                     "repeats": 1, "machine_load": "not controlled (other sessions may run)"},
            "environment": env, "examples": rows}
    Path(a.out).write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0 if all("error" not in r for r in rows.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

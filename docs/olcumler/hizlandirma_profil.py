"""Per-component time of one optimisation iteration (docs/olcumler/hizlandirma_onkayit.md, hizlandirma.md).

Each example (or a resized copy, ``--set domain.shape=[...]``) is built from its YAML as the CLI builds it, run for
``--warmup`` iterations (one-time costs: support check, elimination order, filter matrix), then ``--repeats`` timed
runs of ``--iters`` iterations each. Every timed run starts from the same start design, so the runs do the same work.
An iteration is one ``Study.evaluate`` with gradients plus one optimizer step (the final re-evaluation of the
returned design after the loop is not counted).

Components are inclusive wall times of wrapped functions, summed per iteration:

``assembly``      Elasticity.matrix (and the other physics' matrix assembly that goes through it)
``factor``        DirectSolver.factor (symmetrise + permute + SuperLU / PARDISO)
``solve``         DirectSolver.solve (forward and adjoint back-substitutions)
``residual``      checked_residual + matrix_norm1 (residual and backward-error check of every solve)
``sensitivity``   Elasticity.element_energy (λᵀ KE u per element)
``filter``        DensityFilter / HelmholtzFilter forward and backward (sparse products or SuperLU solves)
``optimizer``     Optimizer.step without the study evaluations it makes (``optimizer_incl`` = the whole step)
``opt_evals``     study evaluations made inside the step: Study.constraint_values (OC bisection, one chain forward
                  per bisection step) and Study.evaluate (GCMMA inner iterations, SiMPL line search); the components
                  inside the step are also reported alone as ``<component>@opt`` (not in the top-level components)
``other``         iteration wall time minus the top-level components, ``optimizer`` and ``opt_evals`` (autograd
                  graph, interpolation, responses, torch ↔ NumPy copies, Python)

The machine is not idle when other work runs: the CPU load (psutil, whole machine, sampled every 0.5 s) is recorded
with every row, and so are the thread settings. Output: one JSON with environment, rows and per-iteration samples.

    python docs/olcumler/hizlandirma_profil.py out.json                          # every example, own size
    python docs/olcumler/hizlandirma_profil.py out.json --only mbb2d --set "domain.shape=[300,100]" --tag mbb300
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch  # noqa: F401  (before toporlogy: SuperLU, never PARDISO; as test_baseline.py)
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((REPO / "examples").glob("*.yaml")) + sorted((REPO / "examples" / "research").glob("*.yaml"))

ACC: dict[str, float] = {}
_DEPTH: dict[str, int] = {}


def _wrap(owner: Any, attr: str, key: str) -> None:
    """Replace ``owner.attr`` by a timed version; nested calls of the same key count once (outermost)."""
    f = getattr(owner, attr)
    if getattr(f, "_timed", False):
        return

    @functools.wraps(f)
    def g(*a: Any, **k: Any) -> Any:
        d = _DEPTH.get(key, 0)
        _DEPTH[key] = d + 1
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            _DEPTH[key] = d
            if d == 0:
                # inside the optimizer step (OC bisection, GCMMA / SiMPL inner evaluations) a component is booked as
                # "<key>@opt"
                k2 = key if key == "optimizer_incl" or not _DEPTH.get("optimizer_incl") else key + "@opt"
                ACC[k2] = ACC.get(k2, 0.0) + time.perf_counter() - t
    g._timed = True  # type: ignore[attr-defined]
    setattr(owner, attr, g)


def instrument() -> None:
    from toporlogy.design import chain as ch
    from toporlogy.design import pde_filter as pf
    from toporlogy.fem import elastic as el
    from toporlogy.solvers import direct as dr
    _wrap(el.Elasticity, "matrix", "assembly")
    _wrap(dr.DirectSolver, "factor", "factor")
    _wrap(dr.DirectSolver, "solve", "solve")
    _wrap(el, "checked_residual", "residual")
    _wrap(el.Elasticity, "element_energy", "sensitivity")
    _wrap(ch._SparseApply, "forward", "filter")
    _wrap(ch._SparseApply, "backward", "filter")
    _wrap(pf.HelmholtzFilter, "apply_numpy", "filter")
    # matrix_norm1 is called by name inside elastic.py (factor and checked_residual): patch the module global
    _wrap(el, "matrix_norm1", "residual")


class Load:
    """Whole-machine CPU percentage sampled in a thread (psutil); None without psutil."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        try:
            import psutil
            self._ps: Any = psutil
            psutil.cpu_percent(None)
        except ImportError:
            self._ps = None
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.samples.append(float(self._ps.cpu_percent(None)))

    def __enter__(self) -> Load:
        if self._ps is not None:
            self._t.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._ps is not None:
            self._t.join()

    def summary(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        s = sorted(self.samples)
        return {"median": statistics.median(s), "min": s[0], "max": s[-1], "n": len(s)}


def environment() -> dict[str, Any]:
    import scipy
    env: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "torch": torch.__version__, "machine": platform.platform(),
                           "processor": platform.processor(), "cpu_count": os.cpu_count(),
                           "torch_threads": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                           "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                           "OPENBLAS_NUM_THREADS")}}
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src"], capture_output=True,
                           text=True)
    env["git"] = git.stdout.strip()
    env["git_dirty_src"] = bool(dirty.stdout.strip())
    try:
        import psutil
        env["ram_total_gib"] = psutil.virtual_memory().total / 2 ** 30
        env["ram_available_gib_at_start"] = psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        pass
    from toporlogy.solvers.direct import DirectSolver
    env["direct_solver"] = DirectSolver().method
    return env


def profile(path: Path, overrides: list[str], iters: int, warmup: int, repeats: int) -> dict[str, Any]:
    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.resolve import read_config
    from toporlogy.optimization import study as stm

    t_build = time.perf_counter()
    setup = build_setup(read_config(str(path)), overrides)
    t_build = time.perf_counter() - t_build
    st = setup.study
    opt = st.optimizer
    # study evaluations made by the optimizer (OC bisection, GCMMA / SiMPL) are timed separately from the step
    _wrap(st, "constraint_values", "constraint_values")
    _wrap(st, "evaluate", "evaluate")
    _wrap(type(opt), "step", "optimizer_incl")
    samples: list[dict[str, float]] = []
    marks: dict[str, Any] = {}

    def cb(it: int, ev: Any, change: float) -> None:
        now = time.perf_counter()
        row = {k: ACC.get(k, 0.0) - marks["acc"].get(k, 0.0) for k in set(ACC) | set(marks["acc"])}
        row["iteration"] = now - marks["t"]
        samples.append(row)
        marks["t"], marks["acc"] = time.perf_counter(), dict(ACC)

    t_warm = time.perf_counter()
    if warmup:
        st.run(max_iter=warmup, tol=0.0)
    t_warm = time.perf_counter() - t_warm
    runs = []
    with Load() as load:
        for _ in range(repeats):
            ACC.clear()
            samples.clear()
            marks["t"], marks["acc"] = time.perf_counter(), {}
            t = time.perf_counter()
            res = st.run(max_iter=iters, tol=0.0, callback=cb)
            runs.append({"wall": time.perf_counter() - t, "iterations": res.iterations, "objective": res.objective,
                         "samples": [dict(s) for s in samples]})
    comps = ("assembly", "factor", "solve", "residual", "sensitivity", "filter")
    extra = ("optimizer_incl", "optimizer", "opt_evals", "other", "iteration", "evaluate")
    per_iter: dict[str, list[float]] = {k: [] for k in comps + extra + tuple(c + "@opt" for c in comps)}
    for r in runs:
        for s in r["samples"]:
            for k in comps:
                per_iter[k].append(s.get(k, 0.0))
                per_iter[k + "@opt"].append(s.get(k + "@opt", 0.0))
            inc = s.get("optimizer_incl", 0.0)
            ev_in = s.get("constraint_values@opt", 0.0) + s.get("evaluate@opt", 0.0)
            per_iter["optimizer_incl"].append(inc)
            per_iter["opt_evals"].append(ev_in)
            per_iter["optimizer"].append(inc - ev_in)
            per_iter["evaluate"].append(s.get("evaluate", 0.0))
            per_iter["other"].append(s["iteration"] - inc - sum(s.get(k, 0.0) for k in comps))
            per_iter["iteration"].append(s["iteration"])
    med = {k: statistics.median(v) for k, v in per_iter.items() if v}
    ph = st.physics
    d = st.domain
    info = {"example": path.stem, "overrides": overrides, "shape": list(d.shape), "dim": d.dim,
            "n_active": int(d.n_active), "n_design": int(d.n_design),
            "n_free": int(len(getattr(ph, "free", []))) if hasattr(ph, "free") else None,
            "n_cases": int(st.F.shape[1]), "physics": type(ph).__name__, "optimizer": opt.name,
            "constraints": [c.key for c in st.constraints], "objective": st.objective.name,
            "build_s": t_build, "warmup_s": t_warm, "warmup_iters": warmup, "iters": iters, "repeats": repeats,
            "median_per_iteration_s": med, "load_cpu_percent": load.summary(),
            "objectives": [r["objective"] for r in runs], "run_wall_s": [r["wall"] for r in runs],
            "runs": runs}
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--set", action="append", default=[], help="override passed to build_setup (as the CLI --set)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    instrument()
    out = Path(a.out)
    data: dict[str, Any] = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"rows": []}
    data["environment"] = environment()
    data.setdefault("argv_log", []).append(sys.argv)
    for ex in EXAMPLES:
        if a.only and ex.stem not in a.only:
            continue
        t = time.perf_counter()
        try:
            row = profile(ex, a.set, a.iters, a.warmup, a.repeats)
        except Exception as e:  # noqa: BLE001 - recorded, the other examples still run
            row = {"example": ex.stem, "overrides": a.set, "error": f"{type(e).__name__}: {e}"}
        row["tag"] = a.tag or ex.stem
        row["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        row["argv"] = sys.argv
        row["git"] = data["environment"]["git"]
        data["rows"].append(row)
        m = row.get("median_per_iteration_s", {})
        parts = " ".join(f"{k}={1e3 * m[k]:.1f}" for k in ("iteration", "assembly", "factor", "solve", "residual",
                                                              "sensitivity", "filter", "optimizer", "opt_evals",
                                                              "other") if k in m)
        print(f"{row['tag']:28s} {time.perf_counter() - t:6.1f} s  ms/it: {parts} {row.get('error', '')}",
              flush=True)
        out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

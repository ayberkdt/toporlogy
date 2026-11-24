"""Running a built study into a run folder: ``run_setup``.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from toporlogy.configuration.build import Setup
from toporlogy.core.verdict import RETURNED_DESIGN, constraint_entry
from toporlogy.execution.runs import flags_of, status_of, verdict_of, write_resolved


def run_setup(setup: Setup, folder: str | Path, progress: Callable[[int, dict[str, Any]], None] | None = None,
              source: str | None = None, overrides: Sequence[str] = (),
              post_progress: Callable[[str, dict[str, Any] | None, float], None] | None = None) -> dict[str, Any]:
    """Run a built study into ``folder``; returns the summary (also written to ``summary.json``).

    Folder content: ``config.resolved.yaml``, ``setup.json``, ``history.jsonl``, ``result.npz``, ``result.json``
    (:class:`~toporlogy.execution.record.Recorder`), ``snapshots.npz`` (if ``output.snapshot_every`` > 0),
    ``summary.json``, and what the post steps write (STL, reports, ``result_repaired.npz``).
    ``progress(it, row)`` gets ``row = {objective, <constraint names>, change, <schedule state>, seconds}``;
    ``post_progress(name, out, seconds)`` is called before each post step (``out`` None) and after it. A post step
    that raises is recorded as ``{"passed": false, "error": ...}`` and the remaining steps still run.
    """
    import torch

    from toporlogy.optimization.diagnostics import grey_fraction
    from toporlogy.execution.record import Recorder

    folder = Path(folder)
    r, st = setup.config, setup.study
    write_resolved(r, folder, source, overrides)
    run = r["run"]
    np.random.seed(int(run["seed"]))                 # global seeds for user components that draw random numbers
    torch.manual_seed(int(run["seed"]))
    st.recorder = Recorder(folder, r["output"]["snapshot_every"], meta={"name": r["name"], "source": source})
    t0 = time.perf_counter()

    def cb(it: int, ev: Any, change: float) -> None:
        if progress is None:
            return
        row = {"objective": ev.f0, **{n: ev.values[n] for n in setup.constraint_names}, "change": change,
               **st.schedule.state(), "seconds": time.perf_counter() - t0}
        progress(it, row)

    res = st.run(max_iter=int(run["max_iter"]), tol=float(run["tol"]), callback=cb,
                 feasibility_tol=float(run["feasibility_tol"]))
    ftol = float(run["feasibility_tol"])
    final = res.constraint_values               # returned design, as res.objective (history[-1] is the step before)
    cons = [constraint_entry(n, float(final[c.key]), c.upper, ftol, RETURNED_DESIGN)
            for c, n in zip(st.constraints, setup.constraint_names, strict=True)]
    summary: dict[str, Any] = {
        "name": r["name"], "objective_name": st.objective.name, "objective": res.objective,
        "objective_source": RETURNED_DESIGN, "constraints": cons, "iterations": res.iterations,
        "max_iter": int(run["max_iter"]), "converged": bool(res.converged), "termination": res.termination,
        "max_violation": res.max_violation, "kkt_residual": res.kkt_residual, "seconds": res.seconds,
        "grey_fraction": grey_fraction(res.xphys, st.domain.active), "tol": float(run["tol"]),
        "config": "config.resolved.yaml", "source": source, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post": {}}
    rep = setup.bc_report
    if rep is not None:
        summary["bc_check"] = {"errors": len(rep.errors), "warnings": len(rep.warnings),
                               "issues": [str(i) for i in rep.issues]}
    for name, step in setup.post:
        key, k = name, 2
        while key in summary["post"]:
            key, k = f"{name}_{k}", k + 1
        if post_progress is not None:
            post_progress(key, None, 0.0)
        tp = time.perf_counter()
        try:
            out = step(res, st, folder)
        except Exception as e:  # noqa: BLE001 - a failing post step must not lose the optimised design
            out = {"passed": False, "error": f"{type(e).__name__}: {e}", "summary": f"ERROR {type(e).__name__}: {e}"}
        if out:
            summary["post"][key] = out
        if post_progress is not None:
            post_progress(key, out or {}, time.perf_counter() - tp)
    summary["verdict"] = verdict_of(summary)
    summary["status"] = status_of(summary)
    summary["flags"] = flags_of(summary)
    (folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary

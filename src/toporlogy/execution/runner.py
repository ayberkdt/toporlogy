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
from toporlogy.configuration.history_build import HistorySetup
from toporlogy.core.verdict import RETURNED_DESIGN, constraint_entry
from toporlogy.execution.runs import flags_of, status_of, verdict_of, write_resolved


def run_setup(setup: Setup, folder: str | Path, progress: Callable[[int, dict[str, Any]], None] | None = None,
              source: str | None = None, overrides: Sequence[str] = (),
              post_progress: Callable[[str, dict[str, Any] | None, float], None] | None = None,
              resume: bool = False) -> dict[str, Any]:
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

    every = int(run.get("checkpoint_every") or 0)
    if every < 0:
        raise ValueError("run.checkpoint_every must be >= 0")
    ck = folder / "checkpoint.ckpt"
    if resume and not ck.is_file():
        raise FileNotFoundError(f"{folder} has no checkpoint.ckpt to resume from (run.checkpoint_every > 0 writes it)")
    x0 = st.perturbed_start(run["start"], int(run["seed"])) if run.get("start") else None
    res = st.run(x0, max_iter=int(run["max_iter"]), tol=float(run["tol"]), callback=cb,
                 feasibility_tol=float(run["feasibility_tol"]), checkpoint=str(ck) if every > 0 else None,
                 checkpoint_every=every, resume=str(ck) if resume else None)
    ftol = float(run["feasibility_tol"])
    phases = None
    if int(getattr(st.interp, "n_fields", 1)) > 1:      # several phases: grey share and volumes from the fractions
        from toporlogy.design.phases import phase_report
        phases = phase_report(st.interp, res.xphys, st.domain)
    final = res.constraint_values               # returned design, as res.objective (history[-1] is the step before)
    cons = [constraint_entry(n, float(final[c.key]), c.upper, ftol, RETURNED_DESIGN)
            for c, n in zip(st.constraints, setup.constraint_names, strict=True)]
    summary: dict[str, Any] = {
        "name": r["name"], "objective_name": st.objective.name, "objective": res.objective,
        "objective_source": RETURNED_DESIGN, "constraints": cons, "iterations": res.iterations,
        "max_iter": int(run["max_iter"]), "converged": bool(res.converged), "termination": res.termination,
        "max_violation": res.max_violation, "kkt_residual": res.kkt_residual, "stationarity": res.stationarity,
        "seconds": res.seconds,
        "grey_fraction": grey_fraction(res.xphys, st.domain.active) if phases is None else phases["grey_fraction"],
        "tol": float(run["tol"]),
        "config": "config.resolved.yaml", "source": source, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post": {}}
    if phases is not None:
        summary["phases"] = phases
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


def run_history_setup(setup: HistorySetup, folder: str | Path,
                      progress: Callable[[int, dict[str, Any]], None] | None = None, source: str | None = None,
                      overrides: Sequence[str] = (),
                      post_progress: Callable[[str, dict[str, Any] | None, float], None] | None = None
                      ) -> dict[str, Any]:
    """Run a built history study (``analysis: history``) into ``folder``; returns the summary (also ``summary.json``).

    Folder content: ``config.resolved.yaml``, ``history.jsonl`` (``it``, ``f0`` = the objective, ``volume``,
    ``change``), ``result.npz`` (``x``, ``xphys``, ``times``, ``compliance`` of the returned design at every step time),
    ``summary.json`` (the keys of :func:`run_setup`, so ``show`` and ``compare`` read it) and what the post steps write
    (``vtu``, ``summary``). ``optimize_viscoelastic`` has no per-iteration callback, so ``progress(it, row)`` is called
    for every iteration after the optimisation; ``post_progress`` as in :func:`run_setup`. A post step that raises is
    recorded as ``{"passed": false, "error": ...}``."""
    import torch

    from toporlogy.core.verdict import constraint_entry
    from toporlogy.optimization.diagnostics import grey_fraction

    folder = Path(folder)
    r = setup.config
    write_resolved(r, folder, source, overrides)
    t0 = time.perf_counter()
    res = setup.optimize()
    seconds = time.perf_counter() - t0
    with torch.no_grad():
        final = setup.structure.response(setup.weights(res.xphys))
    n = len(res.history["f0"])
    rows = [{"it": i + 1, "f0": res.history["f0"][i], "volume": res.history["volume"][i],
             "change": res.history["change"][i] if i < len(res.history["change"]) else 0.0, "t": None}
            for i in range(n)]
    (folder / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if progress is not None:
        for row in rows:
            progress(row["it"], {"objective": row["f0"], "volume": row["volume"], "change": row["change"],
                                 "seconds": None})
    comp = final.compliance.detach().numpy()
    np.savez(folder / "result.npz", x=res.x, xphys=res.xphys, times=setup.times, compliance=comp)
    upper = float(setup.options["volume"])
    ftol = 1e-3
    cons = [constraint_entry("volume", float(res.volume), upper, ftol, RETURNED_DESIGN)]
    summary: dict[str, Any] = {
        "name": r["name"], "analysis": "history", "model": setup.kind, "objective_name": setup.objective_name,
        "objective": res.objective, "objective_source": RETURNED_DESIGN, "constraints": cons,
        "iterations": res.iterations, "max_iter": int(setup.options["max_iter"]), "converged": bool(res.converged),
        "termination": res.termination, "max_violation": max(0.0, float(res.volume) - upper),
        "kkt_residual": res.kkt_residual, "stationarity": None, "seconds": seconds,
        "grey_fraction": grey_fraction(res.xphys), "tol": float(setup.options["tol"]),
        "objective_steps": list(setup.steps), "objective_times_s": [float(setup.times[k]) for k in setup.steps],
        "objective_values": [float(v) for v in res.objectives] if res.objectives is not None else None,
        "config": "config.resolved.yaml", "source": source, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post": {}}
    for name, step in setup.post:
        key, k = name, 2
        while key in summary["post"]:
            key, k = f"{name}_{k}", k + 1
        if post_progress is not None:
            post_progress(key, None, 0.0)
        tp = time.perf_counter()
        try:
            out = step(res, setup, folder, final)
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

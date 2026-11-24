"""Run folders: resolved config, run status, sweep CSV, reading a finished run.
"""
from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporlogy.configuration.schema import STATUS_OK
from toporlogy.core.verdict import (FAIL, LAST_ITERATE, MISSING, PASS, RETURNED_DESIGN, UNKNOWN, UNVERIFIED, combine,
                                    constraint_entry)


# ============================================================================================== running
def write_resolved(resolved: dict[str, Any], folder: str | Path, source: str | None = None,
                   overrides: Sequence[str] = ()) -> Path:
    """Write ``config.resolved.yaml`` with a provenance block; returns its path."""
    import toporlogy as _t
    from toporlogy.execution.record import git_revision

    out = {k: v for k, v in resolved.items() if k != "provenance"}
    out["provenance"] = {"git": git_revision(Path(_t.__file__).parent), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "source": source, "overrides": list(overrides)}
    p = Path(folder) / "config.resolved.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def verdict_of(summary: dict[str, Any]) -> str:
    """Overall verdict of a run summary: FAIL if a constraint FAILs, else UNVERIFIED if a constraint is UNVERIFIED
    or the objective does not belong to the returned design, else PASS (also with no constraints)."""
    objective = PASS if summary.get("objective_source") == RETURNED_DESIGN else UNVERIFIED
    return combine([*_verdicts(summary), objective])


def _verdicts(summary: dict[str, Any]) -> list[str]:
    """Constraint verdicts; a constraint without one is UNVERIFIED."""
    return [c.get("verdict", UNVERIFIED) for c in summary.get("constraints", [])]


def status_of(summary: dict[str, Any]) -> str:
    """"ok", or the problems joined by ", ": "not converged", "constraint violated", "constraint unverified",
    "objective unverified" (a value that does not belong to the returned design), "<post step> failed" (a post
    step whose result has ``passed: false``, e.g. connectivity verdict FAIL, failed manufacturing checks, an error)."""
    flags = []
    if not summary.get("converged", False):
        flags.append("not converged")
    if FAIL in _verdicts(summary):
        flags.append("constraint violated")
    if UNVERIFIED in _verdicts(summary):
        flags.append("constraint unverified")
    if summary.get("objective_source") != RETURNED_DESIGN:
        flags.append("objective unverified")
    for name, out in (summary.get("post") or {}).items():
        if isinstance(out, dict) and out.get("passed") is False:
            flags.append(f"{name} failed")
    return ", ".join(flags) or STATUS_OK


def flags_of(summary: dict[str, Any]) -> list[str]:
    """Machine-readable findings of a run summary, one code each (empty list: nothing to report). The status line
    says the same in words; the flags also name what was NOT verified (docs/gelistirme_plani.md G7):

    ``not_converged``, ``schedule_step_at_max_iter``, ``constraint_violated``, ``constraint_unverified``,
    ``objective_unverified``, ``<post step>_failed``,
    ``repair_unverified`` (a connectivity repair whose constraints could not all be evaluated),
    ``bc_mapping_approximate`` / ``bc_mapping_unsupported`` (a re-analysis that did not solve the optimisation's
    exact supports and loads)."""
    flags: list[str] = []
    if not summary.get("converged", False):
        flags.append("not_converged")
    if summary.get("termination") == "schedule_step_at_max_iter":
        flags.append("schedule_step_at_max_iter")
    if FAIL in _verdicts(summary):
        flags.append("constraint_violated")
    if UNVERIFIED in _verdicts(summary):
        flags.append("constraint_unverified")
    if summary.get("objective_source") != RETURNED_DESIGN:
        flags.append("objective_unverified")
    for name, out in (summary.get("post") or {}).items():
        if not isinstance(out, dict):
            continue
        if out.get("passed") is False:
            flags.append(f"{name}_failed")
        rd = out.get("repaired_design")
        if isinstance(rd, dict) and rd.get("verdict") == UNVERIFIED:
            flags.append("repair_unverified")
        fid = out.get("bc_fidelity")
        if fid in ("approximate", "unsupported"):
            flags.append(f"bc_mapping_{fid}")
    return list(dict.fromkeys(flags))


def write_sweep_csv(path: str | Path, rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> Path:
    """``sweep.csv``: index, folder, swept keys, status, objective, constraint values, iterations, seconds, grey."""
    cnames: list[str] = []
    for row in rows:
        for c in row.get("constraints", []):
            if c["name"] not in cnames:
                cnames.append(c["name"])
    header = ["index", "folder", *keys, "status", "objective", *cnames, "iterations", "converged", "seconds",
              "grey_fraction"]
    p = Path(path)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            cv = {c["name"]: c["value"] for c in row.get("constraints", [])}
            w.writerow([row["index"], row["folder"], *[json.dumps(row["values"][k]) for k in keys], row["status"],
                        row.get("objective"), *[cv.get(n, "") for n in cnames], row.get("iterations"),
                        row.get("converged"), row.get("seconds"), row.get("grey_fraction")])
    return p


# ============================================================================================== reading run folders
def read_run(folder: str | Path) -> dict[str, Any]:
    """Summary of a finished run: ``summary.json`` (CLI runs) or rebuilt from ``result.json`` + ``history.jsonl``
    (runs recorded from Python with :class:`~toporlogy.execution.record.Recorder`). Adds ``history`` (list of rows).

    Both paths give the same contract: per constraint ``verdict`` and ``source``
    (:func:`toporlogy.core.verdict.constraint_entry`), ``objective_source``, the overall ``verdict``, and ``status``
    / ``flags`` recomputed from them. Values that may not belong to the returned design are UNVERIFIED: a
    result.json written before 14 Nov 2026 (no "constraints"; values and objective of the last iterate) and a
    summary.json written before 18 Nov 2026 (no "flags"; before 17 Nov run_setup took the constraint values from the
    last iterate)."""
    from toporlogy.optimization.diagnostics import grey_fraction

    f = Path(folder)
    if not f.is_dir():
        raise FileNotFoundError(f"run folder not found: {f}")
    hist_path = f / "history.jsonl"
    history = [json.loads(line) for line in hist_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if hist_path.is_file() else []
    if (f / "summary.json").is_file():
        s = json.loads((f / "summary.json").read_text(encoding="utf-8"))
        if "objective_source" not in s:              # written before the verdict contract
            # "flags" came with G7 (a7a5631, 18 Nov 2026), after run_setup switched to the returned design's
            # constraint values (42f593c, 17 Nov); without it the values may be those of the last iterate.
            src = RETURNED_DESIGN if "flags" in s else UNKNOWN
            s["constraints"] = [{"name": c["name"], "value": c.get("value"), "upper": c.get("upper"),
                                 "verdict": UNVERIFIED if src != RETURNED_DESIGN or c.get("satisfied") is None
                                 else PASS if c["satisfied"] else FAIL, "source": src}
                                for c in s.get("constraints", [])]   # judged with the run's own feasibility_tol
            s["objective_source"] = src
            s.pop("feasible", None)                  # the overall verdict below says it
    elif (f / "result.json").is_file():
        res = json.loads((f / "result.json").read_text(encoding="utf-8"))
        settings = res.get("settings", {})
        # The returned design's constraint values are in result.json (since 99d8d67, 14 Nov 2026). Before, result.json
        # had none and its objective was history f0[-1]: the last iterate, the design BEFORE the final update.
        if isinstance(res.get("constraints"), dict):
            final, source = res["constraints"], RETURNED_DESIGN
        else:
            final, source = (history[-1] if history else {}), LAST_ITERATE
        tol = float(res.get("feasibility_tol") or 1e-3)   # not recorded before PR-08: Study.run's default
        cons = [constraint_entry(n, final.get(n), u, tol, source if n in final else MISSING)
                for n, u in settings.get("constraints", [])]
        grey = None
        if (f / "result.npz").is_file():
            with np.load(f / "result.npz") as z:
                grey = grey_fraction(z["xphys"])
        s = {"name": f.name, "objective_name": settings.get("objective", "objective"),
             "objective": res.get("objective"), "objective_source": source, "constraints": cons,
             "iterations": res.get("iterations"), "max_iter": None, "converged": res.get("converged"),
             "seconds": res.get("seconds"), "grey_fraction": grey, "termination": res.get("termination"),
             "max_violation": res.get("max_violation"), "kkt_residual": res.get("kkt_residual")}
    else:
        raise FileNotFoundError(f"{f} has neither summary.json nor result.json (is the run finished?)")
    s["verdict"] = verdict_of(s)
    s["status"] = status_of(s)
    s["flags"] = flags_of(s)
    s["history"] = history
    s["folder"] = str(f)
    setup = f / "setup.json"
    if setup.is_file():
        meta = json.loads(setup.read_text(encoding="utf-8"))
        s.setdefault("git", meta.get("git"))
        s.setdefault("created", meta.get("created"))
    return s

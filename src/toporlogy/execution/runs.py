"""Run folders: resolved config, run status, sweep CSV, reading a finished run (moved from toporlogy.config).
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


def status_of(summary: dict[str, Any]) -> str:
    """"ok", or the problems joined by ", ": "not converged", "constraint violated", "<post step> failed" (a post
    step whose result has ``passed: false``, e.g. connectivity verdict FAIL, failed manufacturing checks, an error)."""
    flags = []
    if not summary.get("converged", False):
        flags.append("not converged")
    if any(not c.get("satisfied", True) for c in summary.get("constraints", [])):
        flags.append("constraint violated")
    for name, out in (summary.get("post") or {}).items():
        if isinstance(out, dict) and out.get("passed") is False:
            flags.append(f"{name} failed")
    return ", ".join(flags) or STATUS_OK


def flags_of(summary: dict[str, Any]) -> list[str]:
    """Machine-readable findings of a run summary, one code each (empty list: nothing to report). The status line
    says the same in words; the flags also name what was NOT verified (docs/gelistirme_plani.md G7):

    ``not_converged``, ``schedule_step_at_max_iter``, ``constraint_violated``, ``<post step>_failed``,
    ``repair_unverified`` (a connectivity repair whose constraints could not all be evaluated),
    ``bc_mapping_approximate`` / ``bc_mapping_unsupported`` (a re-analysis that did not solve the optimisation's
    exact supports and loads)."""
    flags: list[str] = []
    if not summary.get("converged", False):
        flags.append("not_converged")
    if summary.get("termination") == "schedule_step_at_max_iter":
        flags.append("schedule_step_at_max_iter")
    if any(not c.get("satisfied", True) for c in summary.get("constraints", [])):
        flags.append("constraint_violated")
    for name, out in (summary.get("post") or {}).items():
        if not isinstance(out, dict):
            continue
        if out.get("passed") is False:
            flags.append(f"{name}_failed")
        rd = out.get("repaired_design")
        if isinstance(rd, dict) and rd.get("feasible") is None:
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
    (runs recorded from Python with :class:`~toporlogy.record.Recorder`). Adds ``history`` (list of rows)."""
    from toporlogy.optimization.checks import grey_fraction

    f = Path(folder)
    if not f.is_dir():
        raise FileNotFoundError(f"run folder not found: {f}")
    hist_path = f / "history.jsonl"
    history = [json.loads(line) for line in hist_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if hist_path.is_file() else []
    if (f / "summary.json").is_file():
        s = json.loads((f / "summary.json").read_text(encoding="utf-8"))
    elif (f / "result.json").is_file():
        res = json.loads((f / "result.json").read_text(encoding="utf-8"))
        settings = res.get("settings", {})
        last = history[-1] if history else {}
        cons = [{"name": n, "value": last.get(n, float("nan")), "upper": u,
                 "satisfied": bool(last.get(n, float("inf")) <= u + 1e-3 * max(abs(u), 1e-12))}
                for n, u in settings.get("constraints", [])]
        grey = None
        if (f / "result.npz").is_file():
            with np.load(f / "result.npz") as z:
                grey = grey_fraction(z["xphys"])
        s = {"name": f.name, "objective_name": settings.get("objective", "objective"),
             "objective": res.get("objective"), "constraints": cons, "iterations": res.get("iterations"),
             "max_iter": None, "converged": res.get("converged"), "seconds": res.get("seconds"),
             "grey_fraction": grey}
        s["status"] = status_of(s)
    else:
        raise FileNotFoundError(f"{f} has neither summary.json nor result.json (is the run finished?)")
    s["history"] = history
    s["folder"] = str(f)
    setup = f / "setup.json"
    if setup.is_file():
        meta = json.loads(setup.read_text(encoding="utf-8"))
        s.setdefault("git", meta.get("git"))
        s.setdefault("created", meta.get("created"))
    return s

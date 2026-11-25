"""Command line: ``python -m toporlogy <command>``.

Commands: ``run``, ``validate``, ``show``, ``compare``, ``list`` (``python -m toporlogy <command> -h``).
Exit codes: 0 = ok; 1 = run finished but not converged, a constraint violated or unverified, an objective unverified
or a post step failed (connectivity verdict FAIL, manufacturing checks); 2 = config or input error (incl.
boundary-condition check errors).
Output is plain ASCII (safe on any console code page); errors go to stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from toporlogy.configuration.build import Setup, build_setup
from toporlogy.configuration.registry import KINDS, REGISTRY, ConfigError, suggest
from toporlogy.configuration.resolve import (apply_overrides, combination_label, parse_sweep, read_config, resolve,
                                             sweep_combinations)
from toporlogy.configuration.schema import STATUS_OK
from toporlogy.execution.runner import run_setup
from toporlogy.core.verdict import PASS, RETURNED_DESIGN, UNVERIFIED
from toporlogy.execution.runs import read_run, write_sweep_csv

__all__ = ["main"]

PROG = "python -m toporlogy"
COMMANDS = ("run", "validate", "show", "compare", "list")
EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2
LABEL = 14                       # width of the label column of key-value output (fits "CONFIG ERROR")


# ============================================================================================== formatting
def fobj(v: Any) -> str:
    """Objective values: 5 significant digits in scientific notation."""
    return "-" if v is None else f"{float(v):.5e}"


def fval(v: Any) -> str:
    """Constraint values and parameters: 4 significant digits."""
    return "-" if v is None else f"{float(v):.4g}"


def fsec(v: Any) -> str:
    return "-" if v is None else f"{float(v):.1f}"


def fpct(v: Any) -> str:
    return "-" if v is None else f"{100 * float(v):.1f}%"


def kv(label: str, text: str) -> str:
    return f"{label:<{LABEL}}{text}"


def out(line: str = "") -> None:
    print(line, flush=True)


def err(line: str = "") -> None:
    print(line, file=sys.stderr, flush=True)


def table(header: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool]) -> list[str]:
    """Aligned columns: text left, numbers right, two spaces between columns."""
    w = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i]) for i in range(len(header))]

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(c.rjust(w[i]) if numeric[i] else c.ljust(w[i]) for i, c in enumerate(cells)).rstrip()

    return [fmt(header)] + [fmt(r) for r in rows]


def component_str(kind: str, spec: dict[str, Any] | None) -> str:
    """``density_filter(rmin=2)``: the type plus the parameters that differ from their defaults."""
    if spec is None:
        return "none"
    try:
        params = REGISTRY.get(kind, spec["type"]).params
    except ConfigError:
        params = {}
    parts = []
    for k, v in spec.items():
        if k == "type":
            continue
        p = params.get(k)
        if p is not None and not p.required and (v == p.default or (p.nested and isinstance(v, dict)
                                                                     and v.get("type") == p.default)):
            continue
        if isinstance(v, dict) and "type" in v:
            v = component_str(p.nested if p is not None and p.nested else kind, v)
        elif isinstance(v, float):
            v = fval(v)
        parts.append(f"{k}={v}")
    return spec["type"] + (f"({', '.join(parts)})" if parts else "")


def vec(v: Sequence[float]) -> str:
    """Components with 4 significant digits; round-off below 1e-12 of the largest one shown as 0."""
    big = max((abs(float(x)) for x in v), default=0.0)
    return "(" + ", ".join(fval(0.0 if abs(float(x)) <= 1e-12 * big else x) for x in v) + ")"


def report_config_error(e: ConfigError, source: str, command: str) -> int:
    n = len(e.errors)
    err(kv("CONFIG ERROR", f"{source}  ({n} problem{'s' if n != 1 else ''})"))
    w = min(max((len(p) for p, _ in e.errors), default=0), 40)
    for p, m in e.errors:
        err(f"  {p:<{w}}  {m}" if p else f"  {m}")
    if any("unknown" in m for _, m in e.errors):
        err(kv("NEXT", f"{PROG} list   (registered components and their parameters)"))
    else:
        err(kv("NEXT", f"fix the keys above, then: {PROG} {command} {source}"))
    return EXIT_CONFIG


# ============================================================================================== run
MAX_COLUMNS = 7                  # columns of any table (Miller); extra constraint / state columns are dropped


def column_budget(constraints: Sequence[str], state: Sequence[str]) -> tuple[list[str], list[str], int]:
    """Constraint and schedule-state columns that fit next to it, objective, change and s (7 in all).

    The first schedule-state value (beta) is kept; constraints fill the rest. Returns (constraints, state, hidden).
    """
    room = MAX_COLUMNS - 4
    st = list(state)[:1] if room > 1 else []
    cs = list(constraints)[:room - len(st)]
    hidden = len(constraints) - len(cs) + len(state) - len(st)
    return cs, st, hidden


class Progress:
    """One aligned line per iteration: it, objective, constraints, change, schedule state (beta), seconds."""

    def __init__(self, objective: str, constraints: Sequence[str]) -> None:
        self.objective = objective
        self.all_constraints = list(constraints)
        self.cols: list[tuple[str, str, int]] | None = None

    def __call__(self, it: int, row: dict[str, Any]) -> None:
        if self.cols is None:
            state = [k for k in row if k not in ("objective", "change", "seconds") and k not in self.all_constraints]
            cs, st, hidden = column_budget(self.all_constraints, state)
            self.cols = ([("objective", self.objective[:11], 11)] + [(c, c[:10], 10) for c in cs] +
                         [("change", "change", 7)] + [(k, k[:6], 6) for k in st] + [("seconds", "s", 7)])
            head = f"{'it':>5}  " + "  ".join(f"{lab:>{w}}" for _, lab, w in self.cols)
            if hidden:
                head += f"   (+{hidden} more in history.jsonl)"
            out(head)
        cells = []
        for key, _, w in self.cols:
            v = row.get(key)
            s = fobj(v) if key == "objective" else f"{v:.4f}" if key == "change" else fsec(v) if key == "seconds" \
                else fval(v)
            cells.append(f"{s:>{w}}")
        out(f"{it:>5}  " + "  ".join(cells))


def _default_folder(cfg: dict[str, Any], suffix: str = "") -> Path:
    name = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in cfg["name"]) or "study"
    base = Path(cfg["output"]["folder"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}{suffix}"
    p, k = base, 2
    while p.exists():
        p = base.with_name(f"{base.name}-{k}")
        k += 1
    return p


def _finished(folder: Path) -> bool:
    return (folder / "summary.json").is_file() or (folder / "result.json").is_file()


def _groups(ph: Any) -> list[Any]:
    """The physics objects behind ``ph`` (one, or one per support set for bc.PerCaseSupports)."""
    return [g for g, _ in ph.groups] if hasattr(ph, "groups") else [ph]


def _free_dofs(ph: Any) -> str:
    return " / ".join(str(len(g.free)) for g in _groups(ph))


def _problem_line(s: Setup) -> str:
    st, d = s.study, s.study.domain
    grid = " x ".join(str(n) for n in d.shape)
    el = getattr(_groups(st.physics)[0], "element", "?")
    nc = len(s.cases)
    cases = f"{nc} load case{'s' if nc != 1 else ''}" if nc else "modal"
    return (f"{grid} {el} | {d.n_design} design elements | {_free_dofs(st.physics)} free DOFs | "
            f"{cases} | {st.optimizer.name} | "
            f"{len(st.constraints)} constraint{'s' if len(st.constraints) != 1 else ''} | "
            f"max {s.config['run']['max_iter']} it")


def _result_line(sm: dict[str, Any]) -> str:
    parts = [sm["status"], f"{sm['objective_name']} {fobj(sm['objective'])}"]
    parts += [f"{c['name']} {fval(c['value'])} <= {fval(c['upper'])}" for c in sm["constraints"]]
    it = f"{sm['iterations']}/{sm['max_iter']} it" if sm.get("max_iter") else f"{sm['iterations']} it"
    parts += [it, f"{fsec(sm['seconds'])} s", f"grey {fpct(sm['grey_fraction'])}"]
    return " | ".join(parts)


def _next_after_run(sm: dict[str, Any], folder: Path, source: str) -> str:
    if "constraint violated" in sm["status"]:
        return f"{PROG} show {folder}"
    if "not converged" in sm["status"]:
        return f"{PROG} run {source} --set run.max_iter={2 * int(sm.get('max_iter') or 100)}"
    return f"{PROG} show {folder}"


def cmd_run(a: argparse.Namespace) -> int:
    src = a.config
    try:
        raw = read_config(src)
        if a.sweep:
            return _run_sweep(a, raw)
        setup = build_setup(raw, a.set)
    except ConfigError as e:
        return report_config_error(e, src, "validate")
    folder = Path(a.out) if a.out else _default_folder(setup.config)
    if _finished(folder):
        err(kv("ERROR", f"{folder} already holds a finished run; it is not overwritten"))
        err(kv("NEXT", f"{PROG} run {src} --out <new folder>"))
        return EXIT_CONFIG
    out(kv("RUN", f"{setup.config['name']} -> {folder}"))
    out(kv("problem", _problem_line(setup)))
    for w in setup.warnings:
        out(kv("warning", w))
    prog = None if a.quiet else Progress(setup.study.objective.name, setup.constraint_names)
    sm = run_setup(setup, folder, progress=prog, source=str(src), overrides=a.set, post_progress=_post_line)
    out(kv("RESULT", _result_line(sm)))
    out(kv("NEXT", _next_after_run(sm, folder, src)))
    return EXIT_OK if sm["status"] == STATUS_OK else EXIT_FAIL


def _post_line(name: str, result: dict[str, Any] | None, seconds: float) -> None:
    """One line per post step: the label is printed when the step starts (some take minutes), the result after."""
    if result is None:
        print(kv("post", name), end="", flush=True)
    else:
        print(f"  {result.get('summary', 'done')} | {fsec(seconds)} s", flush=True)


def _run_sweep(a: argparse.Namespace, raw: dict[str, Any]) -> int:
    sweep = parse_sweep(a.sweep)
    combos = sweep_combinations(sweep)
    keys = [k for k, _ in sweep]
    base = apply_overrides(raw, a.set)
    # validate every combination before any solve (a typo in the last value must not cost the first runs)
    problems: list[tuple[str, str]] = []
    for combo in combos:
        try:
            resolve(apply_overrides(base, combo))
        except ConfigError as e:
            label = combination_label(combo)
            problems += [(f"[{label}] {p}", m) for p, m in e.errors]
    if problems:
        raise ConfigError(problems)
    first = resolve(apply_overrides(base, combos[0]))
    root = Path(a.out) if a.out else _default_folder(first, "_sweep")
    if root.is_dir() and any(_finished(x) for x in root.iterdir() if x.is_dir()):
        err(kv("ERROR", f"{root} already holds sweep runs; they are not overwritten"))
        err(kv("NEXT", f"{PROG} run {a.config} --sweep ... --out <new folder>"))
        return EXIT_CONFIG
    n = len(combos)
    out(kv("SWEEP", f"{n} runs over {' x '.join(f'{k} ({len(v)})' for k, v in sweep)} -> {root}"))
    rows: list[dict[str, Any]] = []
    worst = EXIT_OK
    width = len(str(n))
    for i, combo in enumerate(combos, 1):
        label = combination_label(combo)
        sub = root / f"{i:0{width}d}_{label}"
        head = f"[{i:>{width}}/{n}] " + "  ".join(f"{k}={v}" for k, v in combo)
        print(head, end="", flush=True)
        try:
            setup = build_setup(base, combo)
        except ConfigError as e:
            print(f"  config error: {e}", flush=True)
            rows.append({"index": i, "folder": sub.name, "values": dict(combo), "status": "config error"})
            worst = EXIT_CONFIG
            continue
        sm = run_setup(setup, sub, source=str(a.config), overrides=[*a.set, *(f"{k}={v}" for k, v in combo)])
        print(f"  {sm['status']}  {sm['objective_name']} {fobj(sm['objective'])}  {sm['iterations']} it  "
              f"{fsec(sm['seconds'])} s", flush=True)
        rows.append({"index": i, "folder": sub.name, "values": dict(combo), **sm})
        if sm["status"] != STATUS_OK and worst == EXIT_OK:
            worst = EXIT_FAIL
    csv_path = write_sweep_csv(root / "sweep.csv", rows, keys)
    n_ok = sum(r["status"] == STATUS_OK for r in rows)
    ok_rows = [r for r in rows if r["status"] == STATUS_OK]
    best = ""
    if ok_rows:
        b = min(ok_rows, key=lambda r: r["objective"])
        best = f" | best [{b['index']}] {b['objective_name']} {fobj(b['objective'])}"
    out(kv("RESULT", f"{n_ok}/{n} ok{best} | {csv_path}"))
    out(kv("NEXT", f"{PROG} compare {root}"))
    return worst


# ============================================================================================== validate
def cmd_validate(a: argparse.Namespace) -> int:
    src = a.config
    try:
        s = build_setup(read_config(src), a.set)
    except ConfigError as e:
        return report_config_error(e, src, "validate")
    r, st = s.config, s.study
    d = st.domain
    ph = _groups(st.physics)[0]
    size = " x ".join(fval(n * d.h) for n in d.shape)
    out(kv("CONFIG", f"{src}  valid"))
    out(kv("grid", f"{' x '.join(str(n) for n in d.shape)} elements ({d.dim}-D, {getattr(ph, 'element', '?')}), "
                   f"h = {fval(d.h)}, box {size}"))
    out(kv("elements", f"{d.n_active} active | {d.n_design} design | {int(d.keep.sum())} keep | "
                       f"{int(d.void.sum())} void"))
    n_act = len(d.active_dofs)
    out(kv("DOFs", f"{n_act} in the part | {_free_dofs(st.physics)} free | {len(s.fixed)} fixed "
                   f"({len(r['supports'])} support{'s' if len(r['supports']) != 1 else ''})"))
    out(kv("load cases", str(len(s.cases)) if s.cases else "none (modal analysis)"))
    for i, c in enumerate(s.cases, 1):
        out(f"  {i}  {c['name']:<10} weight {fval(c['weight'])}  resultant {vec(c['resultant'])}  "
            f"on {c['n_nodes']} node{'s' if c['n_nodes'] != 1 else ''}"
            + ("  (combination)" if c.get("combination") else ""))
    _print_bc(s)
    _print_features(s)
    out(kv("chain", " -> ".join(component_str("link", lk) for lk in r["chain"])))
    out(kv("interp", component_str("interpolation", r["interpolation"])))
    out(kv("objective", component_str("response", r["objective"])))
    if st.constraints:
        out(kv("constraints", "; ".join(f"{n} <= {fval(c.upper)}" for n, c in zip(s.constraint_names,
                                                                                 st.constraints, strict=True))))
    else:
        out(kv("constraints", "none"))
    out(kv("optimizer", f"{component_str('optimizer', r['optimizer'])} | schedule "
                        f"{component_str('schedule', r['schedule'])}"))
    out(kv("run", f"max_iter {r['run']['max_iter']} | tol {fval(r['run']['tol'])} | output {r['output']['folder']}"))
    if r["post"]:
        out(kv("post", " -> ".join(p["type"] for p in r["post"])))
    for w in s.warnings:
        out(kv("warning", w))
    out(kv("NEXT", f"{PROG} run {src}"))
    return EXIT_OK


def _print_bc(s: Setup) -> None:
    """Load / support tables of the boundary conditions and the result of their check (errors never get here)."""
    rep = s.bc_report
    if rep is None:
        return
    n_w = len(rep.warnings)
    out(kv("bc check", f"OK | {len(rep.errors)} errors | {n_w} warning{'s' if n_w != 1 else ''}"))
    for line in s.bc_summary().rstrip().splitlines():
        out(("  " + line).rstrip())
    for it in rep.issues:
        out(f"  {it}")


def _print_features(s: Setup) -> None:
    info = s.info
    if info.get("bores"):
        out(kv("bores", info["bores"]))
    rb = info.get("robust")
    if rb:
        size = (f"min solid {fval(rb['solid_size'])} / void {fval(rb['void_size'])} -> rmin {fval(rb['rmin'])} el | "
                if "rmin" in rb else "")
        out(kv("robust", f"{size}eta_e {fval(rb['eta_e'])}, eta_d {fval(rb['eta_d'])}"))
    cb = s.config.get("connectivity")
    if cb is not None:
        ties = (f"ties {info['ties']} cells + {info.get('frozen_interface_cells', 0)} interface cells frozen"
                if "ties" in info else "no ties")
        con = "no constraint" if cb["constraint"] is None else f"{cb['constraint']} constraint"
        gate = "gate" + (f" + {cb['repair']} repair" if cb["repair"] != "none" else "") if cb["gate"] else "no gate"
        out(kv("connectivity", f"{len(s.interfaces)} interfaces | {ties} | {con} | {gate}"))
        out("  " + ", ".join(f"{k} ({v} el)" for k, v in s.interfaces.items()))
    for n in info.get("notes", []):
        out(kv("note", n))


# ============================================================================================== show / compare
def _load_runs(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Run folders; a folder without a run but with run subfolders (a sweep) expands to them."""
    runs = []
    for p in paths:
        f = Path(p)
        if f.is_dir() and not _finished(f):
            subs = sorted(x for x in f.iterdir() if x.is_dir() and _finished(x))
            if subs:
                runs += [read_run(x) for x in subs]
                continue
        runs.append(read_run(f))
    return runs


def cmd_show(a: argparse.Namespace) -> int:
    try:
        s = read_run(a.run_dir)
    except FileNotFoundError as e:
        err(kv("ERROR", str(e)))
        err(kv("NEXT", f"{PROG} run <config>   (creates a run folder)"))
        return EXIT_CONFIG
    it = s.get("iterations")
    mx = s.get("max_iter")
    out(kv("RUN", s["folder"]))
    out(kv("status", f"{s['status']} ({it}{f' of {mx}' if mx else ''} iterations)"))
    osrc = s.get("objective_source")
    out(kv("objective", f"{s['objective_name']} {fobj(s['objective'])}"
                        + ("" if osrc == RETURNED_DESIGN else f"  {UNVERIFIED} ({osrc})")))
    for c in s.get("constraints", []):
        out(kv("constraint", f"{c['name']} {fval(c['value'])} <= {fval(c['upper'])}  {c['verdict']}"
                             + ("" if c["source"] == RETURNED_DESIGN else f" ({c['source']})")))
    out(kv("time", f"{fsec(s.get('seconds'))} s | grey {fpct(s.get('grey_fraction'))}"))
    out(kv("created", f"{s.get('created') or '-'} | git {s.get('git') or 'none'}"))
    hist = s.get("history", [])
    if hist:
        idx = sorted({0, len(hist) // 4, len(hist) // 2, (3 * len(hist)) // 4, len(hist) - 1})
        all_c = [c["name"] for c in s.get("constraints", [])]
        state = [k for k in hist[-1] if k not in ("it", "t", "f0", "change", s["objective_name"], *all_c)]
        cnames, state, _ = column_budget(all_c, state)
        header = ["it", s["objective_name"], *cnames, "change", *state, "s"]
        rows = []
        for i in idx:
            h = hist[i]
            rows.append([str(h.get("it")), fobj(h.get("f0")), *[fval(h.get(n)) for n in cnames],
                         f"{h.get('change', 0):.4f}", *[fval(h.get(k)) for k in state], fsec(h.get("t"))])
        out(kv("history", f"{len(idx)} of {len(hist)} iterations"))
        for line in table(header, rows, [True] * len(header)):
            out("  " + line)
    out(kv("NEXT", f"{PROG} compare {s['folder']} <other run folder>"))
    return EXIT_OK if s["status"] == STATUS_OK else EXIT_FAIL


def cmd_compare(a: argparse.Namespace) -> int:
    try:
        runs = _load_runs(a.run_dirs)
    except FileNotFoundError as e:
        err(kv("ERROR", str(e)))
        err(kv("NEXT", f"{PROG} show <run folder>   (check that each folder is a finished run)"))
        return EXIT_CONFIG
    onames = {r["objective_name"] for r in runs}
    oname = onames.pop() if len(onames) == 1 else "objective"
    cnames: list[str] = []
    for r in runs:
        for c in r.get("constraints", []):
            if c["name"] not in cnames:
                cnames.append(c["name"])
    feasible = [r for r in runs if r["verdict"] == PASS and r.get("objective") is not None]
    best = min(feasible, key=lambda r: r["objective"]) if feasible else None
    header = ["  run", "status", oname, *cnames, "it", "s", "grey"]
    rows = []
    for r in runs:
        cv = {c["name"]: c["value"] for c in r.get("constraints", [])}
        rows.append([("* " if r is best else "  ") + Path(r["folder"]).name, r["status"], fobj(r.get("objective")),
                     *[fval(cv.get(n)) for n in cnames], str(r.get("iterations")), fsec(r.get("seconds")),
                     fpct(r.get("grey_fraction"))])
    numeric = [False, False, True, *[True] * len(cnames), True, True, True]
    for line in table(header, rows, numeric):
        out(line)
    if best is not None:
        out(f"* lowest {oname} with every constraint {PASS} on the returned design")
    out(kv("NEXT", f"{PROG} show {best['folder'] if best else runs[0]['folder']}"))
    return EXIT_OK


# ============================================================================================== list
def _wrap(prefix: str, items: Sequence[str], width: int = 100) -> list[str]:
    lines, cur = [], prefix
    for it in items:
        if len(cur) + len(it) + 2 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = " " * len(prefix)
        cur += it + "  "
    lines.append(cur.rstrip())
    return lines


def cmd_list(a: argparse.Namespace) -> int:
    kinds = REGISTRY.kinds()
    if a.kind:
        k = a.kind.strip().lower()
        if k.endswith("s") and k[:-1] in kinds:
            k = k[:-1]
        if k not in kinds:
            err(kv("ERROR", f'unknown kind "{a.kind}"{suggest(k, kinds)}'))
            err(kv("NEXT", f"{PROG} list   (kinds: {', '.join(kinds)})"))
            return EXIT_CONFIG
        kinds = [k]
    for kind in kinds:
        nm = REGISTRY.names(kind)
        out(f"{kind}  - {KINDS[kind]} ({len(nm)})")
        w = max((len(n) for n in nm), default=0)
        for n in nm:
            e = REGISTRY.get(kind, n)
            ps = []
            for p in e.params.values():
                dv = "?" if p.required else json.dumps(p.default) if not isinstance(p.default, str) else p.default
                ps.append(f"{p.name}={dv}" + (f" [{p.unit}]" if p.unit else ""))
            for line in _wrap(f"  {n:<{w}}  ", ps or ["(no parameters)"]):
                out(line)
        out()
    out("? = required.  Units: lengths in the unit of h, forces in force units, E0 in force/length^2.")
    out(kv("NEXT", f"{PROG} validate <config>"))
    return EXIT_OK


# ============================================================================================== parser
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description="Density-based topology optimisation from a YAML/JSON "
                                "config.", epilog=f"example: {PROG} run study.yaml --set run.max_iter=50   "
                                "| exit codes: 0 ok, 1 not converged / constraint violated or unverified / "
                                "objective unverified / post check failed, "
                                "2 config error")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")
    r = sub.add_parser("run", help="run a config (one run folder; with --sweep one subfolder per combination)")
    r.add_argument("config", help="YAML or JSON config file")
    r.add_argument("--out", metavar="DIR", help="run folder (default: <output.folder>/<date-time>_<name>)")
    r.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE",
                   help="override a config value, e.g. --set chain.1.rmin=2.0 (repeatable)")
    r.add_argument("--sweep", action="append", default=[], metavar="KEY.PATH=V1,V2",
                   help="run every combination of the listed values (repeatable: cartesian product)")
    r.add_argument("-q", "--quiet", action="store_true", help="no per-iteration lines")
    v = sub.add_parser("validate", help="check a config and print the problem summary (no solve)")
    v.add_argument("config", help="YAML or JSON config file")
    v.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE", help="override (repeatable)")
    s = sub.add_parser("show", help="summary of a finished run")
    s.add_argument("run_dir", help="run folder")
    c = sub.add_parser("compare", help="table of several runs (a sweep folder expands to its runs)")
    c.add_argument("run_dirs", nargs="+", help="run folders or sweep folders")
    li = sub.add_parser("list", help="registered components with their parameters")
    li.add_argument("kind", nargs="?", help=f"one kind: {', '.join(KINDS)}")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0].lower() in COMMANDS:          # commands are case-insensitive
        args[0] = args[0].lower()
    a = parser().parse_args(args)
    handlers = {"run": cmd_run, "validate": cmd_validate, "show": cmd_show, "compare": cmd_compare,
                "list": cmd_list}
    try:
        return handlers[a.command](a)
    except ConfigError as e:
        return report_config_error(e, getattr(a, "config", ""), "validate")

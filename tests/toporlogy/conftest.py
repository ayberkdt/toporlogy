"""tests/toporlogy: import toporlogy before anything imports torch.

MKL PARDISO must be loaded before torch, otherwise two libiomp5md.dll copies abort the process (see
toporlogy/__init__.py). Optional native packages (pypardiso + MKL, gmsh) may live in a separate folder; set
``TOPORLOGY_PYLIB`` to it (several folders separated by ``os.pathsep``). Without them the tests fall back to SuperLU
and skip the gmsh-based checks.
"""
import os
import sys

for _p in os.environ.get("TOPORLOGY_PYLIB", "").split(os.pathsep):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# frozen copies of earlier drivers used as regression oracles (tests/toporlogy/topo_oracles)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import toporlogy  # noqa: E402,F401  (side effect: PARDISO before torch)


# ---------------------------------------------------------------------------------------------- verification report
import json  # noqa: E402

import pytest  # noqa: E402

import _verification as V  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    for b, what in V.BACKENDS.items():
        config.addinivalue_line("markers", f"{b}: needs the optional backend: {what}")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    V.COLLECTED[:] = [V.item_record(i) for i in items]          # before -m / -k deselection (strict-xfail ledger)


def pytest_collection_finish(session: pytest.Session) -> None:
    path = os.environ.get("TOPORLOGY_COLLECT_DUMP")             # test_catalog_ledger.py: collection in a subprocess
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(V.COLLECTED, fh)


def pytest_report_header(config: pytest.Config) -> list[str]:
    env = V.environment()
    opt = ", ".join(f"{k} {v or '-'}" for k, v in env["optional"].items())
    return [f"toporlogy git {env['git_sha']} | python {env['python']} | numpy {env['numpy']} | scipy {env['scipy']} | "
            f"torch {env['torch']} (cuda {env['cuda_available']}) | {env['system']} {env['arch']}",
            f"optional: {opt} | PARDISO: {env['pardiso'].get('loaded')} ({env['pardiso'].get('reason')})"]


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call" or (report.when == "setup" and report.skipped):
        for b in V.BACKENDS:
            if b in report.keywords:
                reason = ""
                if report.skipped and isinstance(report.longrepr, tuple):
                    reason = str(report.longrepr[2]).removeprefix("Skipped: ")
                V.OUTCOMES[b].append((report.nodeid, report.outcome, reason))


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    status = V.backend_status()
    terminalreporter.write_sep("-", "verification status")
    for b, s in status.items():
        detail = f"{s['passed']} passed" if s["status"] != "NOT VERIFIED" else s["reason"]
        if s["failed"]:
            detail += f", {s['failed']} FAILED"
        terminalreporter.write_line(f"{b:8s} {s['status']:13s} {detail}")
    if V.BASELINE:
        terminalreporter.write_line(f"baseline {V.BASELINE['mode']:13s} {V.BASELINE['note']}")
        tol = V.BASELINE.get("rel_tol")
        for name, row in V.BASELINE.get("examples", {}).items():
            line = f"max relative difference {row['max_rel']:.2e}" + (f" (tolerance {tol:g})" if tol else "")
            if row["differs"]:
                line += f"; differs: {', '.join(row['differs'])}"
            terminalreporter.write_line(f"  {name:22s} {line}")
            if os.environ.get("GITHUB_ACTIONS") == "true":        # public annotation: readable without logs access
                level = "error" if row["differs"] else "notice"
                terminalreporter.write_line(f"::{level} title=baseline {name}::{line}")
    path = os.environ.get("TOPORLOGY_ENV_REPORT")
    if path:
        stats = {k: len(v) for k, v in terminalreporter.stats.items() if k in ("passed", "failed", "skipped", "error")}
        report = {"environment": V.environment(), "backends": status, "baseline": V.BASELINE or None,
                  "results": stats}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)

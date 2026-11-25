"""Runs the multi-phase checks of tests/toporlogy/test_multi_material.py (slow V5 included) and writes their printed
values to docs/olcumler/cok_malzeme.json.

    PYTHONPATH=src python docs/olcumler/cok_malzeme.py
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import toporlogy  # noqa: F401  (PARDISO-before-torch import order)
    import torch
    test_file = ROOT / "tests/toporlogy/test_multi_material.py"
    spec = importlib.util.spec_from_file_location("test_multi_material", test_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    status: dict[str, str] = {}
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        params = getattr(fn, "pytestmark", [])
        cases = [()]
        for m in params:
            if m.name == "parametrize":
                cases = list(m.args[1])
        for case in cases:
            key = name + (f"[{'-'.join(map(str, case))}]" if case else "")
            t0 = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    kw = {"tmp_path": Path(tmp)} if "tmp_path" in inspect.signature(fn).parameters else {}
                    fn(*case, **kw)
                status[key] = f"PASS ({time.perf_counter() - t0:.1f} s)"
            except AssertionError as e:
                status[key] = f"FAIL: {e}"
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S607
                         cwd=ROOT).stdout.strip()
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_base": rev, "python": platform.python_version(),
           "torch": torch.__version__, "status": status, "results": mod.RESULTS}
    path = ROOT / "docs/olcumler/cok_malzeme.json"
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()

"""G0 baseline: every example YAML, 10 iterations, in a fresh process on CPU + SuperLU; outputs compared bitwise.

docs/gelistirme_plani.md §5.2 (G0) and §5.3: the seven examples gave bitwise identical x, xphys, objective,
constraints and STL in two separate processes (17 Nov 2026), so a move-only commit must reproduce
``baseline/examples.json`` exactly. A commit that changes numbers on purpose rewrites the file and says why.

    pytest tests/toporlogy/test_baseline.py -m baseline        # compare (~3 min)
    python tests/toporlogy/test_baseline.py --write            # rewrite after an intended change

The runner imports torch before toporlogy: PARDISO is then never loaded (solvers/pardiso.py) and the direct
solver is SuperLU on every machine. PARDISO and CUDA results are not part of the baseline (not measured).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.baseline, pytest.mark.slow]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLES = REPO / "examples"
FILE = HERE / "baseline" / "examples.json"
MAX_ITER = 10
RUNNER = "import sys, torch; from toporlogy.cli import main; raise SystemExit(main(sys.argv[1:]))"
GEOMETRY_FILES = (".stl", ".ply", ".obj")


def _sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()


def needs_post(path: Path) -> bool:
    return "\npost:" in path.read_text(encoding="utf-8")


def run_example(path: Path, out: Path) -> dict[str, Any]:
    """One example in a fresh process; the numbers and hashes that must not change."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    p = subprocess.run([sys.executable, "-c", RUNNER, "run", str(path), "--out", str(out), "-q",
                        "--set", f"run.max_iter={MAX_ITER}"], env=env, capture_output=True, text=True,
                       timeout=1800)
    if not (out / "summary.json").is_file():
        raise RuntimeError(f"{path.name}: no summary.json (exit {p.returncode})\n{p.stderr[-2000:]}")
    sm = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    z = np.load(out / "result.npz")
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(out.iterdir()) if f.suffix in GEOMETRY_FILES}
    return {"returncode": p.returncode, "status": sm["status"], "iterations": sm["iterations"],
            "objective": sm["objective"], "constraints": {c["name"]: c["value"] for c in sm["constraints"]},
            "x": _sha(z["x"]), "xphys": _sha(z["xphys"]), "post_steps": sorted(sm.get("post", {})),
            "post_errors": sorted(k for k, v in sm.get("post", {}).items() if isinstance(v, dict) and "error" in v),
            "geometry_files": files}


def examples() -> list[Path]:
    return sorted(EXAMPLES.glob("*.yaml"))


def write() -> None:
    import torch
    rows: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as d:
        for ex in examples():
            t = time.perf_counter()
            rows[ex.stem] = run_example(ex, Path(d) / ex.stem)
            print(f"{ex.stem:24s} {rows[ex.stem]['status']:14s} {time.perf_counter() - t:6.1f} s", flush=True)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git.stdout.strip(), "max_iter": MAX_ITER,
            "solver": "SuperLU (torch imported first)", "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "scipy": importlib.import_module("scipy").__version__,
            "machine": platform.platform()}
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps({"meta": meta, "examples": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FILE}")


def _stored() -> dict[str, Any]:
    # the baseline is tracked in git: a missing file is a deleted safety net, not a reason to skip
    assert FILE.is_file(), f"baseline {FILE} is missing: restore it from git or run python {Path(__file__).name} --write"
    return json.loads(FILE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", examples(), ids=lambda p: p.stem)
def test_example_matches_baseline(path: Path, tmp_path: Path) -> None:
    stored = _stored()["examples"]
    assert path.stem in stored, f"{path.stem} has no baseline entry: rewrite the baseline"
    if needs_post(path) and importlib.util.find_spec("pyvista") is None:
        pytest.skip("post steps need the post extra (pyvista)")
    got = run_example(path, tmp_path / path.stem)
    want = stored[path.stem]
    diff = {k: (want.get(k), got.get(k)) for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)}
    assert not diff, f"{path.stem} differs from the baseline (stored, now): {json.dumps(diff, indent=1)}"


def test_every_baseline_entry_has_an_example() -> None:
    stored = _stored()["examples"]
    assert sorted(stored) == [p.stem for p in examples()]


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit(f"usage: python {Path(__file__).name} --write")
    write()

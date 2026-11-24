"""CLI ``python -m toporlogy``: run / validate / show / compare / list, sweeps, exit codes.

Most commands are driven in-process through :func:`toporlogy.execution.cli.main` (same argv, same exit codes, captured
stdout/stderr): every fresh interpreter pays the torch + SciPy import (measured 26-53 s on 13 Nov 2026 with the CPU
shared by other jobs). One test runs the real ``python -m toporlogy`` entry point in a subprocess.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from toporlogy.execution.cli import main

ROOT = Path(__file__).resolve().parents[2]


def tiny(max_iter: int = 4, tol: float = 0.01, vf: float = 0.5) -> dict:
    return {
        "name": "tiny",
        "domain": {"type": "box", "shape": [12, 4]},
        "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}, "dofs": "x"},
                     {"type": "fixed", "where": {"type": "point", "at": ["max", "min"]}, "dofs": "y"}],
        "load_cases": [{"name": "centre", "loads": [{"type": "force", "where": {"type": "point", "at": ["min", "max"]},
                                                     "force": [0, -1]}]}],
        "chain": [{"type": "density_filter", "rmin": 1.5}],
        "constraints": [{"response": "volume", "upper": vf}],
        "run": {"max_iter": max_iter, "tol": tol},
    }


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(tiny()), encoding="utf-8")
    return p


def run_cli(capsys, *argv) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    o = capsys.readouterr()
    return code, o.out, o.err


# ---------------------------------------------------------------------------------------------- validate / list
def test_validate_summary(capsys, cfg):
    code, out, _ = run_cli(capsys, "validate", cfg)
    assert code == 0
    assert "12 x 4 elements (2-D, Q4)" in out
    assert "48 active | 48 design" in out
    assert "130 in the part | 124 free | 6 fixed (2 supports)" in out  # 13*5 nodes * 2; 5 + 1 fixed
    assert "resultant (0, -1)  on 1 node" in out
    assert "embed -> density_filter(rmin=1.5) -> passive" in out
    assert "volume <= 0.5" in out
    assert out.strip().splitlines()[-1].startswith("NEXT")


def test_validate_config_error_exit_2(capsys, tmp_path):
    raw = tiny()
    raw["chain"][0]["type"] = "density_filtr"
    raw["material"] = {"E0": "stiff"}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    code, out, err = run_cli(capsys, "VALIDATE", p)                    # commands are case-insensitive
    assert code == 2 and out == ""
    assert "CONFIG ERROR" in err and "(2 problems)" in err
    assert 'chain.0.type  unknown link "density_filtr"; did you mean "density_filter"?' in err
    assert "material.E0" in err
    assert err.strip().splitlines()[-1].startswith("NEXT")
    code, _, err = run_cli(capsys, "validate", tmp_path / "missing.yaml")
    assert code == 2 and "config file not found" in err


def test_list(capsys):
    code, out, _ = run_cli(capsys, "list")
    assert code == 0
    for kind in ("domain", "selector", "support", "load", "link", "interpolation", "response", "optimizer",
                 "schedule", "post"):
        assert f"\n{kind}  - " in "\n" + out
    assert "density_filter        rmin=? [elements]" in out
    code, out, _ = run_cli(capsys, "list", "links")
    assert code == 0 and out.startswith("link  - ")
    code, _, err = run_cli(capsys, "list", "optimiser")
    assert code == 2 and 'did you mean "optimizer"' in err


# ---------------------------------------------------------------------------------------------- run / show / compare
def test_run_show_compare(capsys, cfg, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    code, out, _ = run_cli(capsys, "run", cfg, "--out", a)
    assert code == 1                                                   # 4 iterations: not converged
    lines = out.strip().splitlines()
    assert lines[0].startswith("RUN") and lines[1].startswith("problem")
    assert lines[2].split() == ["it", "compliance", "volume", "change", "s"]
    assert [ln.split()[0] for ln in lines[3:7]] == ["1", "2", "3", "4"]
    assert lines[-2].split()[:5] == ["RESULT", "not", "converged", "|", "compliance"]
    assert "--set run.max_iter=8" in lines[-1]
    assert (a / "config.resolved.yaml").is_file() and (a / "summary.json").is_file()

    code, out, _ = run_cli(capsys, "run", cfg, "--out", b, "--set", "run.max_iter=40", "--set", "run.tol=0.05",
                           "-q")
    assert code == 0, out
    assert out.splitlines()[-2].split()[:3] == ["RESULT", "ok", "|"]
    assert len(out.strip().splitlines()) == 4                         # quiet: RUN, problem, RESULT, NEXT
    assert "run.tol=0.05" in (b / "config.resolved.yaml").read_text(encoding="utf-8")

    code, out, _ = run_cli(capsys, "show", b)
    assert code == 0
    assert out.splitlines()[1].split()[:2] == ["status", "ok"]
    assert "volume" in out and "PASS" in out and "history" in out

    code, out, _ = run_cli(capsys, "compare", a, b)
    assert code == 0
    rows = out.strip().splitlines()
    assert rows[0].split() == ["run", "status", "compliance", "volume", "it", "s", "grey"]
    assert [r[2:].split()[0] for r in rows[1:3]] == ["a", "b"]
    assert [r[0] for r in rows[1:3]].count("*") == 1                  # best feasible run marked once

    code, _, err = run_cli(capsys, "run", cfg, "--out", b)              # finished run is never overwritten
    assert code == 2 and "not overwritten" in err
    code, _, err = run_cli(capsys, "show", tmp_path / "nope")
    assert code == 2 and "not found" in err


def test_sweep_2x2(capsys, cfg, tmp_path):
    root = tmp_path / "sw"
    code, out, _ = run_cli(capsys, "run", cfg, "--out", root, "--set", "run.max_iter=2",
                           "--sweep", "chain.0.rmin=1.5,2.0", "--sweep", "constraints.0.upper=0.4,0.5")
    assert code == 1                                                   # 2 iterations: not converged
    subs = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert subs == ["1_chain.0.rmin=1.5__constraints.0.upper=0.4", "2_chain.0.rmin=1.5__constraints.0.upper=0.5",
                    "3_chain.0.rmin=2.0__constraints.0.upper=0.4", "4_chain.0.rmin=2.0__constraints.0.upper=0.5"]
    for s in subs:
        r = yaml.safe_load((root / s / "config.resolved.yaml").read_text(encoding="utf-8"))
        assert r["run"]["max_iter"] == 2
    with open(root / "sweep.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert [r["chain.0.rmin"] for r in rows] == ["1.5", "1.5", "2.0", "2.0"]
    assert [r["constraints.0.upper"] for r in rows] == ["0.4", "0.5", "0.4", "0.5"]
    assert all(r["iterations"] == "2" and r["status"] == "not converged" for r in rows)
    assert "[4/4]" in out
    assert out.strip().splitlines()[-1].split()[:5] == ["NEXT", "python", "-m", "toporlogy", "compare"]

    code, out, _ = run_cli(capsys, "compare", root)                   # a sweep folder expands to its runs
    assert code == 0
    assert [ln[2:].split()[0] for ln in out.splitlines()[1:5]] == subs


def test_sweep_validates_every_combination_first(capsys, cfg, tmp_path):
    code, _, err = run_cli(capsys, "run", cfg, f"--out={tmp_path / 'sw'}", "--sweep=chain.0.rmin=1.5,abc")
    assert code == 2 and "[chain.0.rmin=abc] chain.0.rmin" in err
    assert not (tmp_path / "sw").exists()                              # nothing was run


# ---------------------------------------------------------------------------------------------- real entry point
def test_module_entry_point_subprocess(cfg, tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), *filter(None, os.environ.get("TOPORLOGY_PYLIB", "").split(os.pathsep))]))
    p = subprocess.run([sys.executable, "-m", "toporlogy", "run", str(cfg), "--out", str(tmp_path / "r"),
                        "--set", "run.max_iter=2", "-q"], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 1, p.stderr                                 # not converged after 2 iterations
    assert [ln.split()[:3] for ln in p.stdout.splitlines()][-2] == ["RESULT", "not", "converged"]
    assert p.stdout.isascii()


def test_table_columns_capped_at_seven():
    from toporlogy.execution.cli import MAX_COLUMNS, column_budget

    cs, st, hidden = column_budget(["c1", "c2", "c3", "c4", "c5"], ["beta"])
    assert (cs, st, hidden) == (["c1", "c2"], ["beta"], 3)
    assert 4 + len(cs) + len(st) == MAX_COLUMNS                         # it, objective, change, s + the rest
    assert column_budget(["volume"], []) == (["volume"], [], 0)

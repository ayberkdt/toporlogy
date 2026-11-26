"""Resumable sweeps (docs/olcumler/tarama_surdurme_onkayit.md, R1-R6, pre-registered before the code)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

import toporlogy.execution.cli as cli
from toporlogy.execution.cli import main

CFG = {"domain": {"type": "box", "shape": [12, 6], "h": 1.0},
       "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
       "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                  "force": [0, -1]}]}],
       "chain": [{"type": "density_filter", "rmin": 1.5}],
       "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma", "run": {"max_iter": 3}}
SWEEP = ["--sweep", "constraints.0.upper=0.3,0.4,0.5"]
# 3 iterations do not converge, so a finished sweep exits 1 (a failed check), not 0; 2 is a configuration error


def _cfg(tmp: Path) -> Path:
    p = tmp / "c.yaml"
    p.write_text(yaml.safe_dump(CFG), encoding="utf-8")
    return p


def _rows(folder: Path) -> list[dict[str, str]]:
    with open(folder / "sweep.csv", encoding="utf-8", newline="") as f:
        return [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(f)]


def test_r1_to_r3_interrupt_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _cfg(tmp_path)
    full = tmp_path / "full"
    assert main(["run", str(p), "--out", str(full), "-q", *SWEEP]) != 2
    part = tmp_path / "part"
    real = cli.run_setup
    calls = {"n": 0}

    def stop_after_two(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            m = json.loads((part / "sweep_manifest.json").read_text(encoding="utf-8"))     # R1: readable, current
            assert [r["status"] for r in m["runs"]] == ["done", "done", "pending"]
            raise KeyboardInterrupt
        return real(*args, **kw)
    monkeypatch.setattr(cli, "run_setup", stop_after_two)
    with pytest.raises(KeyboardInterrupt):
        main(["run", str(p), "--out", str(part), "-q", *SWEEP])
    monkeypatch.setattr(cli, "run_setup", real)
    runs = sorted(x for x in part.iterdir() if x.is_dir())
    before = {r.name: (r / "result.npz").read_bytes() for r in runs if (r / "summary.json").is_file()}
    assert len(before) == 2
    assert main(["run", str(p), "--out", str(part), "-q", "--resume", *SWEEP]) != 2
    for name, data in before.items():                                                    # R2: not re-run
        assert (part / name / "result.npz").read_bytes() == data
    assert _rows(part) == _rows(full)                                                    # R3
    m = json.loads((part / "sweep_manifest.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in m["runs"]] == ["done", "done", "done"]


def test_r4_resume_with_another_request_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    p = _cfg(tmp_path)
    out = tmp_path / "s"
    assert main(["run", str(p), "--out", str(out), "-q", "--sweep", "constraints.0.upper=0.3,0.4"]) != 2
    assert main(["run", str(p), "--out", str(out), "-q", "--resume", "--sweep", "constraints.0.upper=0.3,0.5"]) == 2
    assert "sweep" in capsys.readouterr().err
    assert main(["run", str(p), "--out", str(out), "-q", "--resume", "--set", "run.max_iter=4",
                 "--sweep", "constraints.0.upper=0.3,0.4"]) == 2


def test_r5_without_resume_a_full_folder_is_refused(tmp_path: Path) -> None:
    p = _cfg(tmp_path)
    out = tmp_path / "s"
    assert main(["run", str(p), "--out", str(out), "-q", "--sweep", "constraints.0.upper=0.3,0.4"]) != 2
    assert main(["run", str(p), "--out", str(out), "-q", "--sweep", "constraints.0.upper=0.3,0.4"]) == 2


def test_r6_manifest_holds_environment_and_resources(tmp_path: Path) -> None:
    p = _cfg(tmp_path)
    out = tmp_path / "s"
    assert main(["run", str(p), "--out", str(out), "-q", "--sweep", "constraints.0.upper=0.3,0.4"]) != 2
    m = json.loads((out / "sweep_manifest.json").read_text(encoding="utf-8"))
    assert {"python", "numpy", "torch", "git"} <= set(m["environment"])
    assert all(r["seconds"] > 0 for r in m["runs"])
    try:
        import psutil  # noqa: F401
        assert all(r["peak_memory_bytes"] > 0 for r in m["runs"])
    except ImportError:
        assert all(r["peak_memory_bytes"] is None for r in m["runs"])

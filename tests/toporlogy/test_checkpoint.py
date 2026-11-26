"""Optimisation checkpoint and restart (docs/olcumler/checkpoint_onkayit.md, CK1-CK5, pre-registered before the code)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from toporlogy.configuration.build import build_setup

CFG = {
    "domain": {"type": "box", "shape": [30, 10], "h": 1.0},
    "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}, "dofs": "x"},
                 {"type": "fixed", "where": {"type": "point", "at": ["max", "min"]}, "dofs": "y"}],
    "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["min", "max"]}, "force": [0, -1]}]}],
    "chain": [{"type": "density_filter", "rmin": 1.5}, {"type": "heaviside_projection", "beta": 1.0}],
    "constraints": [{"response": "volume", "upper": 0.5}],
    "schedule": {"type": "beta_continuation", "beta_max": 8.0, "every": 6},
}
N_TOTAL, N_CUT = 24, 10


def _study(optimizer: str):
    c = copy.deepcopy(CFG)
    c["optimizer"] = optimizer
    if optimizer == "oc":
        c["chain"] = [{"type": "density_filter", "rmin": 1.5}]
        del c["schedule"]
    return build_setup(c).study


class _Stop(Exception):
    pass


@pytest.mark.parametrize("optimizer", ["mma", "gcmma", "oc"])
def test_ck1_ck2_resume_reproduces_the_uninterrupted_run(optimizer: str, tmp_path: Path) -> None:
    full = _study(optimizer).run(max_iter=N_TOTAL, tol=1e-12)
    ck = tmp_path / "run.ckpt"
    first = _study(optimizer)

    def stop(it, ev, change):
        if it == N_CUT:
            raise _Stop
    with pytest.raises(_Stop):
        first.run(max_iter=N_TOTAL, tol=1e-12, callback=stop, checkpoint=str(ck), checkpoint_every=1)
    # the checkpoint was written after iteration N_CUT - 1 (the callback of N_CUT raised before its save)
    resumed = _study(optimizer).run(max_iter=N_TOTAL, tol=1e-12, resume=str(ck))
    assert resumed.iterations == full.iterations
    assert np.array_equal(resumed.x, full.x)
    assert resumed.objective == full.objective
    for k in full.history:
        if k == "seconds":          # wall time: not reproducible even between two uninterrupted runs (record)
            continue
        assert np.array_equal(np.asarray(resumed.history[k]), np.asarray(full.history[k])), k


def test_ck3_another_problem_is_refused(tmp_path: Path) -> None:
    ck = tmp_path / "a.ckpt"
    _study("mma").run(max_iter=3, checkpoint=str(ck), checkpoint_every=1)
    other = copy.deepcopy(CFG)
    other["domain"]["shape"] = [20, 10]
    other["optimizer"] = "mma"
    with pytest.raises(ValueError, match="another problem"):
        build_setup(other).study.run(max_iter=5, resume=str(ck))
    gc = _study("gcmma")
    with pytest.raises(ValueError, match="optimizer"):
        gc.run(max_iter=5, resume=str(ck))


def test_ck4_atomic_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    ck = tmp_path / "b.ckpt"
    _study("mma").run(max_iter=4, checkpoint=str(ck), checkpoint_every=2)
    assert ck.is_file()
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_ck6_ck7_cli_checkpoint_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import yaml

    from toporlogy.execution.cli import main
    from toporlogy.optimization.study import Study
    c = copy.deepcopy(CFG)
    c["optimizer"] = "mma"
    c["run"] = {"max_iter": 12, "tol": 1e-12, "checkpoint_every": 1}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    full, part = tmp_path / "full", tmp_path / "part"
    main(["run", str(p), "--out", str(full), "-q"])
    real = Study.evaluate
    count = {"n": 0}

    def interrupted(self, x, gradients=True):
        count["n"] += 1
        if count["n"] == 6:
            raise KeyboardInterrupt
        return real(self, x, gradients)
    monkeypatch.setattr(Study, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main(["run", str(p), "--out", str(part), "-q"])
    monkeypatch.setattr(Study, "evaluate", real)
    assert (part / "checkpoint.ckpt").is_file() and not (part / "summary.json").is_file()
    main(["run", str(p), "--out", str(part), "-q", "--resume"])
    assert np.array_equal(np.load(full / "result.npz")["x"], np.load(part / "result.npz")["x"])     # CK6
    assert main(["run", str(p), "--out", str(tmp_path / "none"), "-q", "--resume"]) == 2               # CK7

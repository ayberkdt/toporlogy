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

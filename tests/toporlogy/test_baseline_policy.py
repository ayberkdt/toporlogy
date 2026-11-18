"""PR-04: when the baseline comparison is bitwise and what the numerical comparison accepts (fast; the baseline run
itself is in test_baseline.py, marked slow)."""
from __future__ import annotations

import platform

import numpy as np
import scipy
import torch
from test_baseline import REL_TOL, compare, environment_mismatch


def _meta(**over):
    m = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
         "torch": torch.__version__, "system": platform.system(), "arch": platform.machine()}
    m.update(over)
    return m


def test_bitwise_only_in_the_reference_environment():
    assert environment_mismatch(_meta()) == {}
    assert environment_mismatch(_meta(numpy="0.0.1")) == {"numpy": ("0.0.1", np.__version__)}
    old = _meta(machine=f"{platform.system()}-11-10.0.26200-SP0")                # files written before PR-04
    del old["system"], old["arch"]
    assert environment_mismatch(old) == {}
    other = dict(old, machine="Linux-6.8.0-x86_64-with-glibc2.39")
    assert set(environment_mismatch(other)) == ({"system"} if platform.system() != "Linux" else set())


def _row(**over):
    r = {"returncode": 1, "status": "not converged", "iterations": 10, "objective": 286.35382321673757,
         "constraints": {"volume": 0.4999302124524384}, "x": "a" * 64, "xphys": "b" * 64, "post_steps": [],
         "post_errors": [], "geometry_files": {"part.stl": "c" * 64}}
    r.update(over)
    return r


def test_numerical_comparison_ignores_hashes_but_not_numbers_or_outcomes():
    want = _row()
    assert compare(want, _row(x="d" * 64), bitwise=True) == {"x": ("a" * 64, "d" * 64)}
    assert compare(want, _row(x="d" * 64, geometry_files={}), bitwise=False) == {}
    near = want["objective"] * (1 + 0.5 * REL_TOL)
    far = want["objective"] * (1 + 10 * REL_TOL)
    assert compare(want, _row(objective=near), bitwise=False) == {}
    assert set(compare(want, _row(objective=far), bitwise=False)) == {"objective"}
    assert set(compare(want, _row(constraints={"volume": 0.5}), bitwise=False)) == {"constraints.volume"}
    assert set(compare(want, _row(constraints={}), bitwise=False)) == {"constraints.volume"}
    assert set(compare(want, _row(iterations=9, status="converged"), bitwise=False)) == {"iterations", "status"}

"""Optimizer reset on a model epoch change (docs/olcumler/model_donemi_onkayit.md, ME1-ME4, pre-registered before the
code)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from toporlogy.fem.nonlinear import NonlinearConvergenceError

from toporlogy.configuration.build import build_setup

CFG = {
    "domain": {"type": "box", "shape": [16, 6], "h": 1.0},
    "material": {"E0": 1.0, "nu": 0.3},
    "physics": {"type": "nonlinear_elasticity", "material": "svk"},
    "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
    "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                              "force": [0, -0.05]}]}],
    "chain": [{"type": "density_filter", "rmin": 1.5}],
    "objective": "end_compliance",
    "constraints": [{"response": "volume", "upper": 0.5}],
    "optimizer": {"type": "mma", "move": 0.2},
}


def _run(reset: bool, physics: dict | None = None, max_iter: int = 30, load: float | None = None):
    cfg = copy.deepcopy(CFG)
    if load is not None:
        cfg["load_cases"][0]["loads"][0]["force"] = [0, -load]
    if physics:
        cfg["physics"].update(physics)
    st = build_setup(cfg).study
    st.reset_on_model_change = reset
    calls = []
    real = st.optimizer.reset

    def counted() -> None:
        calls.append(len(calls))
        real()
    st.optimizer.reset = counted                                  # type: ignore[method-assign]
    res = st.run(max_iter=max_iter, tol=1e-9)
    return st, res, len(calls)


@pytest.mark.xfail(strict=True, reason="pre-registered ME1 FAILED as written on the first run: optimizer.reset was called "
                   "6 times for 5 epoch changes; the extra call is the start-of-run restart() (Optimizer.restart "
                   "calls reset by default). The hook's own count equals the epoch changes (post hoc test below)")
def test_me1_one_reset_per_epoch_change() -> None:
    st, res, n_calls = _run(True)
    ep = np.asarray(res.history["model_epoch"])
    changes = int(np.count_nonzero(np.diff(ep)))
    print(f"\n  ME1: epochs {ep[0]:.0f} -> {ep[-1]:.0f}, changes {changes}, resets {st.n_model_resets}")
    assert st.n_model_resets == changes == n_calls
    assert changes >= 1                                           # the problem exercises the hook


def test_me1_posthoc_resets_after_the_start_equal_the_epoch_changes() -> None:
    """Post hoc (after ME1 failed): the reset calls after the start-of-run restart, and the hook's counter, equal the
    number of epoch changes."""
    st, res, n_calls = _run(True)
    changes = int(np.count_nonzero(np.diff(np.asarray(res.history["model_epoch"]))))
    assert st.n_model_resets == changes == n_calls - 1 and changes >= 1


@pytest.mark.xfail(strict=True, raises=NonlinearConvergenceError,
                   reason="pre-registered ME3 FAILED on the first run: with the fixed cutoff 0.1 and the tip load "
                   "0.05 Newton stalls at load factor 0.58 of a design in both settings, so the comparison cannot "
                   "be made (post hoc test below with the load 0.01 of the repository's non-linear YAML tests)")
def test_me3_fixed_cutoff_gives_the_same_run_with_and_without_reset() -> None:
    _, a, na = _run(True, {"cutoff_step": 0.0}, 15)
    _, b, nb = _run(False, {"cutoff_step": 0.0}, 15)
    assert na == nb == 0
    assert np.array_equal(a.x, b.x) and a.objective == b.objective


def test_me4_measured_effect() -> None:
    """Measurement only (no threshold)."""
    st_t, a, _ = _run(True)
    st_f, b, _ = _run(False)
    print(f"\n  ME4: reset objective {a.objective:.6g} ({st_t.n_model_resets} resets), no reset {b.objective:.6g}")
    assert np.isfinite(a.objective) and np.isfinite(b.objective)


def test_me3_posthoc_fixed_cutoff_same_run_at_a_smaller_load() -> None:
    """Post hoc (after ME3 failed): the same comparison at the tip load 0.01."""
    _, a, na = _run(True, {"cutoff_step": 0.0}, 15, load=0.01)
    _, b, nb = _run(False, {"cutoff_step": 0.0}, 15, load=0.01)
    assert na == nb == 1                                          # the start-of-run restart only
    assert np.array_equal(a.x, b.x) and a.objective == b.objective

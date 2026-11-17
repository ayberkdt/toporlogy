"""Study options: move-limit shrink at β steps (``move_beta_ref``).

Measured (14 Nov 2026, scratch run): the 3-D robust cantilever 20×8×5 (b = 2, OC move 0.1, β → 32) collapses without
the shrink (eroded / intermediate / dilated compliance 1.2e7 / 7.9e6 / 5.8e6 after 100 iterations: load path lost)
and converges with ``move_beta_ref = 2`` (76 iterations, 666 / 133 / 63).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.robust import DilatedVolume, RobustCompliance, RobustContinuation, RobustSet, robust_chain_settings


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _cantilever_2d(nx=24, ny=8):
    d = T.Domain((nx, ny))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    F = d.load_vector()
    F[d.dofs(d.nodes_where(lambda X: (X[:, 0] > nx - 1e-9) & (X[:, 1] < 1e-9)), 1), 0] = -1.0
    return d, fixed, F


def test_move_shrinks_with_beta_and_is_restored():
    d, fixed, F = _cantilever_2d()
    chain = T.standard_chain(d, 1.5, beta=1.0)
    opt = T.OC(move=0.2, dv_scale=d.n_active)
    sch = T.BetaContinuation(chain, 1.0, 16.0, every=3)
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(), [T.Constraint(T.Volume(), 0.4,
                 scale=1.0)], opt, sch, move_beta_ref=2.0)
    seen = []
    st.run(max_iter=16, callback=lambda it, ev, ch: seen.append((sch.beta, opt.move)))
    for beta, move in seen:
        assert move == pytest.approx(0.2 * min(1.0, 2.0 / beta)) or beta == 1.0
    assert any(b >= 8 for b, _ in seen)
    assert opt.move == 0.2                                   # restored after the run


def test_robust_3d_does_not_collapse_with_move_shrink():
    nx, ny, nz = 20, 8, 5
    keep = np.zeros((nx, ny, nz), bool)
    keep[-2:, :, :2] = True
    d = T.Domain((nx, ny, nz), 1.0, keep=keep)
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-6))
    F = d.load_vector()
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - nx) < 1e-6) & (X[:, 2] < 1e-6))
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    s = robust_chain_settings(2.0, 2.0, 1.0, 3)
    rs = RobustSet.from_settings(s)
    chain = T.standard_chain(d, s["rmin"], projection=True, beta=1.0)
    obj = RobustCompliance(rs)
    con = T.Constraint(DilatedVolume(rs), 0.3)
    sch = RobustContinuation(chain, rs, 1.0, 32.0, every=15, tol=0.01, volume=con, target=0.3, volume_every=10,
                             rescale=[obj])
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), obj, [con], T.OC(move=0.1, dv_scale=d.n_active),
                 sch, move_beta_ref=2.0)
    res = st.run(max_iter=100, tol=0.01)
    lv = obj.last_values
    print(f"\n  {res.iterations} it, realisations {lv}, grey {T.grey_fraction(res.xphys, d.active):.3f}")
    assert sch.beta == 32.0
    assert lv["eroded"] >= lv["intermediate"] >= lv["dilated"]
    assert lv["eroded"] < 20.0 * lv["dilated"]               # without the shrink: ~2 (all realisations ~1e7)
    assert lv["intermediate"] < 1e3


@pytest.mark.parametrize("opt_name,iters", [("OC", 1), ("OC", 5), ("MMA", 2), ("MMA", 10)])
def test_final_state_belongs_to_the_returned_design(opt_name, iters):
    """Result.objective / constraint values / U / xphys are re-evaluated at Result.x (not taken from the last loop
    iteration, which analysed the design before the final update)."""
    d, fixed, F = _cantilever_2d()
    chain = T.standard_chain(d, 1.5, beta=1.0)
    opt = T.OC(move=0.2, dv_scale=d.n_active) if opt_name == "OC" else T.MMA()
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(),
                 [T.Constraint(T.Volume(), 0.4)], opt, T.BetaContinuation(chain, 1.0, 8.0, every=3))
    res = st.run(max_iter=iters)
    ev = st.evaluate(res.x, gradients=False)
    assert res.objective == pytest.approx(ev.f0, rel=1e-12)
    assert res.constraint_values["volume"] == pytest.approx(ev.values["volume"], rel=1e-12)
    assert np.allclose(res.U, ev.U, rtol=1e-12, atol=0)
    assert np.array_equal(res.xphys, ev.xphys)
    # the loop history is the design BEFORE the last update: generally different from the returned one
    assert res.history["f0"][-1] != pytest.approx(res.objective, rel=1e-9)


@pytest.mark.parametrize("optimizer", ["oc", "mma"])
def test_second_run_of_a_study_repeats_the_first(optimizer):
    """B-03: run() restarts the continuation schedule (β back to β0, counter 0) and the optimizer (MMA asymptotes,
    iterates, objective scale), so two runs of one Study give the same design and history. Before, the second run
    started at the β the first one ended with (and with the first run's MMA state)."""
    d, fixed, F = _cantilever_2d()
    chain = T.standard_chain(d, 1.5, beta=1.0)
    opt = T.OC(move=0.2, dv_scale=d.n_active) if optimizer == "oc" else T.MMA(move=0.2)
    sch = T.BetaContinuation(chain, 1.0, 8.0, every=3)
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), T.Compliance(), [T.Constraint(T.Volume(), 0.4,
                 scale=1.0)], opt, sch)
    r1 = st.run(max_iter=12, tol=0.0)
    assert sch.beta > 1.0                                    # the first run did step β
    n1 = getattr(opt, "n_resets", None)
    r2 = st.run(max_iter=12, tol=0.0)
    assert r2.history["beta"][0] == 1.0
    assert np.array_equal(r1.x, r2.x) and r1.history["f0"] == r2.history["f0"]
    assert getattr(opt, "n_resets", None) == n1             # MMA counts the resets of the current run only


def test_robust_continuation_restarts_beta_and_volume_bound():
    """B-03 for RobustContinuation: β and the updated dilated-volume bound go back to their initial values."""
    d, fixed, F = _cantilever_2d(30, 10)
    s = robust_chain_settings(2.0, 2.0, 1.0, 2)
    rs = RobustSet.from_settings(s)
    chain = T.standard_chain(d, s["rmin"], projection=True, beta=1.0)
    obj = RobustCompliance(rs)
    vol = T.Constraint(DilatedVolume(rs), 0.5)
    sch = RobustContinuation(chain, rs, 1.0, 8.0, every=3, volume=vol, target=0.4, volume_every=2, rescale=[obj])
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), obj, [vol], T.MMA(move=0.2), sch)
    r1 = st.run(max_iter=8, tol=0.0)
    assert sch.beta > 1.0 and vol.upper != 0.5
    r2 = st.run(max_iter=8, tol=0.0)
    assert np.array_equal(r1.x, r2.x) and r1.history["f0"] == r2.history["f0"]

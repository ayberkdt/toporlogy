"""Study options: move-limit shrink at β steps (``move_beta_ref``).

Measured (14 Nov 2026, scratch run): the 3-D robust cantilever 20×8×5 (b = 2, OC move 0.1, β → 32) collapses without
the shrink (eroded / intermediate / dilated compliance 1.2e7 / 7.9e6 / 5.8e6 after 100 iterations: load path lost)
and converges with ``move_beta_ref = 2`` (76 iterations, 666 / 133 / 63).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import aerokit.topo as T
from aerokit.topo.robust import DilatedVolume, RobustCompliance, RobustContinuation, RobustSet, robust_chain_settings


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

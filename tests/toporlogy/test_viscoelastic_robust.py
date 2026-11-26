"""Min-max over service times for viscoelastic topology optimisation (docs/foundations/prereg/VR_viscoelastic_robust.md,
VR1-VR4)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure, optimize_viscoelastic

from test_viscoelastic_topology import TIMES, _cantilever, _general, _nonprop, _schedule, _weights


def _mbb():
    nx, ny = 30, 10
    m = rectangle(nx, ny, float(nx), float(ny))
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    corner = int(np.flatnonzero(np.isclose(P[:, 0], nx) & np.isclose(P[:, 1], 0.0))[0])
    fixed = np.concatenate([2 * left, [2 * corner + 1]])
    tl = int(np.flatnonzero(np.isclose(P[:, 0], 0.0) & np.isclose(P[:, 1], ny))[0])
    times = np.concatenate([[0.0], np.logspace(-1, 2, 13)])                 # 0+, then 0.1 ... 100 s (10 s on the grid)
    loads = np.zeros((len(times), m.n_nodes * 2))
    loads[:, 2 * tl + 1] = -1.0
    return ViscoelasticStructure(m, _nonprop(), times, loads, fixed), times


def _step_of(times, t):
    return int(np.argmin(np.abs(times - t)))


@pytest.fixture(scope="module")
def vr_runs():
    st, times = _mbb()
    ks = [0, _step_of(times, 10.0), len(times) - 1]
    assert abs(times[ks[1]] - 10.0) <= 1e-9 * 10.0 and times[ks[2]] == 100.0
    mm = optimize_viscoelastic(st, 0.5, 1.5, steps=ks, max_iter=300, tol=1e-3)
    singles = [optimize_viscoelastic(st, 0.5, 1.5, step_index=k, max_iter=300, tol=1e-3) for k in ks]
    with torch.no_grad():
        def all_J(xp):
            c = st.response(_weights(torch.as_tensor(xp))).compliance
            return np.array([float(c[k]) for k in ks])
        worst_single = [float(all_J(r.xphys).max()) for r in singles]
        worst_mm = float(all_J(mm.xphys).max())
    print(f"VR1 worst-case min-max {worst_mm:.6e} vs single-time designs {worst_single}; "
          f"iterations mm {mm.iterations} (converged {mm.converged}), singles "
          f"{[(r.iterations, r.converged) for r in singles]}; volumes {[mm.volume] + [r.volume for r in singles]}")
    return st, ks, mm, singles, worst_mm, worst_single


@pytest.mark.xfail(strict=True, reason="pre-registered VR1 FAILED on the first valid run: worst case of the min-max "
                   "design 1.21480e-4 vs 1.21135e-4 for the single t = 100 s design (+0.28 %); under a constant load the "
                   "worst case is always the last time, the min-max reduces to J(100 s) and the bound formulation "
                   "stopped at a worse point (95 vs 134 iterations, tol 1e-3)")
def test_vr1_minmax_worst_case_not_worse(vr_runs) -> None:
    _st, _ks, _mm, _singles, worst_mm, worst_single = vr_runs
    assert worst_mm <= (1.0 + 1e-3) * min(worst_single)


def test_vr2_bound_active(vr_runs) -> None:
    _st, _ks, mm, _singles, _w, _ws = vr_runs
    z, cref = mm.history["z"][0], mm.history["c_ref"][0]
    assert abs(z - mm.objectives.max() / cref) <= 1e-3 * z


def test_vr3_one_forward_many_gradients() -> None:
    m, fixed, tip = _cantilever()
    st = ViscoelasticStructure(m, _general(), TIMES, _schedule(m.n_nodes * 2, tip, TIMES, 2.0), fixed)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    ks = [0, 5, len(TIMES) - 1]
    w = torch.as_tensor(_weights(x)).requires_grad_(True)
    comp = st.response(w).compliance
    joint = [torch.autograd.grad(comp[k], w, retain_graph=True)[0].numpy() for k in ks]
    for k, gj in zip(ks, joint, strict=True):
        w2 = torch.as_tensor(_weights(x)).requires_grad_(True)
        st.response(w2).compliance[k].backward()
        g2 = w2.grad.numpy()
        assert np.linalg.norm(gj - g2) <= 1e-12 * np.linalg.norm(g2)


def test_vr4_convergence_flag() -> None:
    st, _times = _mbb()
    r = optimize_viscoelastic(st, 0.5, 1.5, step_index=-1, max_iter=3, tol=0.0)
    assert not r.converged and r.iterations == 3
    r2 = optimize_viscoelastic(st, 0.5, 1.5, step_index=-1, max_iter=300, tol=1e-3)
    if r2.converged:
        assert r2.history["change"][-1] < 1e-3 and r2.iterations <= 300

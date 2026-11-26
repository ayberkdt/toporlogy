"""Viscoelastic topology optimisation with the exact history adjoint (docs/foundations/prereg/VO_viscoelastic_topology.md,
VO1-VO6)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.constitutive import TimeStep
from toporlogy.core.linear_elastic import isotropic_moduli
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries
from toporlogy.fem.mesh.history import HistoryProblem
from toporlogy.fem.mesh.mesh import perturb, rectangle
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure, optimize_viscoelastic

MPA = 1e6


def _nonprop() -> GeneralizedMaxwell:
    """VO4 set: shear does not relax, bulk does (K = 0.5 + 4.5 e^{-t/10 s} MPa); synthetic."""
    return GeneralizedMaxwell(PronySeries(1.0 * MPA, [], []), PronySeries(0.5 * MPA, [4.5 * MPA], [10.0]))


def _general() -> GeneralizedMaxwell:
    return GeneralizedMaxwell(PronySeries(1.0 * MPA, [2.0 * MPA, 3.0 * MPA], [1.0, 100.0]),
                              PronySeries(5.0 * MPA, [4.0 * MPA], [10.0]))


def _proportional() -> GeneralizedMaxwell:
    g = np.array([2.0, 3.0])
    return GeneralizedMaxwell(PronySeries(1.0 * MPA, g * MPA, [1.0, 100.0]),
                              PronySeries(2.5 * MPA, 2.5 * g * MPA, [1.0, 100.0]))


def _cantilever(nx=8, ny=4, seed=3, distort=True):
    m = rectangle(nx, ny, 2.0, 1.0)
    if distort:
        m = perturb(m, 0.06, seed=seed)
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    fixed = np.concatenate([2 * left, 2 * left + 1])
    tip = int(np.flatnonzero(np.isclose(P[:, 0], 2.0) & np.isclose(P[:, 1], 0.0))[0])
    return m, fixed, tip


def _schedule(ndof, tip, times, ramp_end):
    loads = np.zeros((len(times), ndof))
    for k, t in enumerate(times):
        loads[k, 2 * tip + 1] = -1e3 * min(t / ramp_end, 1.0)
    return loads


TIMES = np.array([0.5, 1.0, 1.7, 2.0, 3.5, 6.0, 9.0, 15.0, 30.0, 60.0, 120.0, 300.0])


@pytest.mark.parametrize("c", [1.0, 0.37])
def test_vo1_forward_matches_history_fem(c) -> None:
    mat = _general()
    m, fixed, tip = _cantilever()
    loads = _schedule(m.n_nodes * 2, tip, TIMES, 2.0)
    st = ViscoelasticStructure(m, mat, TIMES, loads, fixed)
    u = st.response(torch.full((m.n_cells,), c, dtype=torch.float64)).u.detach().numpy()
    scaled = GeneralizedMaxwell(PronySeries(c * mat.shear.inf, c * mat.shear.amplitudes, mat.shear.taus),
                                PronySeries(c * mat.bulk.inf, c * mat.bulk.amplitudes, mat.bulk.taus))
    hp = HistoryProblem(m, scaled)
    tp = 0.0
    err = 0.0
    for k, t in enumerate(TIMES):
        r = hp.step(TimeStep(tp, t - tp), f_ext=loads[k], prescribed=(fixed, np.zeros(len(fixed))), tol=1e-13)
        tp = t
        err = max(err, float(np.abs(r.u - u[k]).max() / np.abs(r.u).max()))
    assert err <= 1e-10, err


def _weights(x, p=3.0, emin=1e-9):
    return emin + x ** p * (1.0 - emin)


def test_vo2_gradient_matches_fd() -> None:
    mat = _general()
    m, fixed, tip = _cantilever()
    st = ViscoelasticStructure(m, mat, TIMES, _schedule(m.n_nodes * 2, tip, TIMES, 2.0), fixed)
    rng = np.random.default_rng(4)
    x = 0.3 + 0.6 * rng.random(m.n_cells)
    d = rng.normal(size=m.n_cells)
    xt = torch.as_tensor(x).requires_grad_(True)
    J = st.response(_weights(xt)).compliance[-1]
    J.backward()
    g = float(xt.grad.numpy() @ d)
    errs = []
    with torch.no_grad():
        for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            jp = float(st.response(_weights(torch.as_tensor(x + h * d))).compliance[-1])
            jm = float(st.response(_weights(torch.as_tensor(x - h * d))).compliance[-1])
            errs.append(abs((jp - jm) / (2 * h) - g) / abs(g))
    assert min(errs) <= 1e-6, errs


def _elastic_compliance_grad(m, mat, fixed, f, w):
    """Independent elastic solve with the instantaneous moduli (fem.mesh.ElasticProblem), dc/dw_e = -u_e^T k_e u_e."""
    from toporlogy.core.linear_elastic import isotropic_plane_strain
    from toporlogy.fem.mesh import ElasticProblem
    from toporlogy.fem.mesh.linear import Factorization
    G0, K0 = mat.shear.instantaneous, mat.bulk.instantaneous
    iso = isotropic_moduli(K=K0, G=G0)
    prob = ElasticProblem(m, isotropic_plane_strain(iso.E, iso.nu))
    ke = np.asarray(prob.element_stiffness())
    K = prob.pattern.assemble(w[:, None, None] * ke).tocsr()
    free = np.setdiff1d(np.unique(prob.cell_dofs), fixed)
    u = np.zeros(prob.ndof)
    u[free] = Factorization(K[free][:, free], "spd").solve(f[free])
    ue = u[prob.cell_dofs]
    return float(f @ u), -np.einsum("ea,eab,eb->e", ue, ke, ue)


def _ratio_test(mat):
    m, fixed, tip = _cantilever()
    loads = _schedule(m.n_nodes * 2, tip, TIMES, 2.0)
    st = ViscoelasticStructure(m, mat, TIMES, loads, fixed)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    w = torch.as_tensor(_weights(x)).requires_grad_(True)
    J = st.response(w).compliance[-1]
    J.backward()
    gJ = w.grad.numpy()
    c_el, g_el = _elastic_compliance_grad(m, mat, fixed, loads[-1], _weights(x))
    return float(np.linalg.norm(gJ - (float(J) / c_el) * g_el) / np.linalg.norm(gJ))


def test_vo3_proportional_spectra_gradient_is_scaled_elastic() -> None:
    assert _ratio_test(_proportional()) <= 1e-10


def test_vo4_nonproportional_spectra_break_the_scaling() -> None:
    assert _ratio_test(_nonprop()) >= 1e-3


def test_vo5_detached_history_changes_the_gradient() -> None:
    mat = _general()
    m, fixed, tip = _cantilever()
    st = ViscoelasticStructure(m, mat, TIMES, _schedule(m.n_nodes * 2, tip, TIMES, 2.0), fixed)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    grads = []
    for det in (False, True):
        xt = torch.as_tensor(x).requires_grad_(True)
        st.response(_weights(xt), detach_history=det).compliance[-1].backward()
        grads.append(xt.grad.numpy().copy())
    assert np.linalg.norm(grads[1] - grads[0]) / np.linalg.norm(grads[0]) >= 1e-3


def test_vo6_service_time_design_differs_from_instantaneous() -> None:
    nx, ny = 30, 10
    m = rectangle(nx, ny, float(nx), float(ny))
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    corner = int(np.flatnonzero(np.isclose(P[:, 0], nx) & np.isclose(P[:, 1], 0.0))[0])
    fixed = np.concatenate([2 * left, [2 * corner + 1]])
    top_left = int(np.flatnonzero(np.isclose(P[:, 0], 0.0) & np.isclose(P[:, 1], ny))[0])
    times = np.concatenate([[0.0], np.logspace(-1, 2, 12)])                  # instantaneous step, then to 100 s
    loads = np.zeros((len(times), m.n_nodes * 2))
    loads[:, 2 * top_left + 1] = -1.0
    st = ViscoelasticStructure(m, _nonprop(), times, loads, fixed)
    r_s = optimize_viscoelastic(st, 0.5, 1.5, step_index=-1, max_iter=40)
    r_0 = optimize_viscoelastic(st, 0.5, 1.5, step_index=0, max_iter=40)
    diff = float(np.abs(r_s.xphys - r_0.xphys).mean())
    print(f"VO6 mean |dxphys| = {diff:.4e}; objectives at their own steps {r_s.objective:.6e}, {r_0.objective:.6e}; "
          f"volumes {r_s.volume:.5f} {r_0.volume:.5f}")
    assert r_s.volume <= 0.5 + 1e-3 and r_0.volume <= 0.5 + 1e-3
    assert diff >= 0.01

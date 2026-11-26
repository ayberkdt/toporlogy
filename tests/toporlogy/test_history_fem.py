"""History-dependent FE on a general mesh (docs/foundations/prereg/H_history.md, H0-H5)."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core.constitutive import J2Plasticity, LinearElastic, NeoHooke, TimeStep, drive_material_point
from toporlogy.core.linear_elastic import isotropic_plane_strain, isotropic_stiffness
from toporlogy.fem.mesh.history import HistoryProblem, StepRejected
from toporlogy.fem.mesh.mesh import box, perturb, rectangle


def _bar_supports(m, ndim=3):
    """Roller supports of a box: x = 0 face u_x = 0; nodes on y = 0 u_y = 0; nodes on z = 0 u_z = 0."""
    P = m.points
    dofs = [3 * n for n in np.flatnonzero(np.isclose(P[:, 0], 0.0))]
    dofs += [3 * n + 1 for n in np.flatnonzero(np.isclose(P[:, 1], 0.0))]
    dofs += [3 * n + 2 for n in np.flatnonzero(np.isclose(P[:, 2], 0.0))]
    return np.array(sorted(dofs))


def test_h0_affine_patch_linear() -> None:
    E, nu = 2.0e9, 0.3
    m = perturb(rectangle(4, 3, 2.0, 1.0), 0.08, seed=4)
    D = isotropic_plane_strain(E, nu)
    hp = HistoryProblem(m, LinearElastic(D))
    A = np.array([[1e-3, 2e-4], [-3e-4, 5e-4]])
    exact = m.points @ A.T
    bnd = m.boundary_nodes()
    dofs = np.concatenate([2 * bnd, 2 * bnd + 1])
    vals = np.concatenate([exact[bnd, 0], exact[bnd, 1]])
    r = hp.step(TimeStep(0.0, 1.0), prescribed=(dofs, vals))
    u = r.u.reshape(-1, 2)
    assert np.abs(u - exact).max() <= 1e-10 * np.abs(exact).max()
    eps = np.array([A[0, 0], A[1, 1], A[0, 1] + A[1, 0]])
    sx, sy, sxy = D @ eps
    sig = np.array([[sx, sxy], [sxy, sy]])
    # independent reference: div sigma = 0, so the reaction (f_int) at boundary node a is the boundary integral of
    # N_a sigma.n; on straight boundary edges with linear N each end node gets sigma.n L / 2 of every edge it is on
    P = m.points
    ref = np.zeros((m.n_nodes, 2))
    sides = [(lambda q: np.isclose(q[:, 0], 0.0), np.array([-1.0, 0.0]), 1),
             (lambda q: np.isclose(q[:, 0], 2.0), np.array([1.0, 0.0]), 1),
             (lambda q: np.isclose(q[:, 1], 0.0), np.array([0.0, -1.0]), 0),
             (lambda q: np.isclose(q[:, 1], 1.0), np.array([0.0, 1.0]), 0)]
    for on, nrm, coord in sides:
        nodes = np.flatnonzero(on(P))
        nodes = nodes[np.argsort(P[nodes, coord])]
        t = sig @ nrm
        for a, b in zip(nodes[:-1], nodes[1:]):
            L = float(np.linalg.norm(P[b] - P[a]))
            ref[a] += 0.5 * L * t
            ref[b] += 0.5 * L * t
    got = np.zeros((m.n_nodes, 2))
    got[bnd, 0] = r.reactions[:len(bnd)]
    got[bnd, 1] = r.reactions[len(bnd):]
    assert np.abs(got - ref).max() <= 1e-10 * np.abs(ref).max()


def _j2_axial_history():
    return [2e-4, 6e-4, 1.2e-3, 2.0e-3, 2.0e-3, 1.0e-3, 0.0]


def test_h1_homogeneous_bar_equals_material_point() -> None:
    m = box(2, 1, 1, 2.0, 1.0, 1.0)
    model = J2Plasticity(0.3, 1e-3, hardening=0.05)
    hp = HistoryProblem(m, model)
    sup = _bar_supports(m)
    right = np.flatnonzero(np.isclose(m.points[:, 0], 2.0))
    hist = _j2_axial_history()
    ctrl = np.array([True, False, False, False, False, False])
    strain = np.zeros((len(hist), 6))
    strain[:, 0] = hist
    mp = drive_material_point(model, strain, np.zeros((len(hist), 6)), ctrl)
    for k, e in enumerate(hist):
        dofs = np.concatenate([sup, 3 * right])
        vals = np.concatenate([np.zeros(len(sup)), np.full(len(right), 2.0 * e)])
        r = _step_presc(hp, k, dofs, vals)
        s = r.stress.reshape(-1, 6)
        assert np.abs(s - mp.stress[k]).max() <= 1e-9 * 1e-3, (k, np.abs(s - mp.stress[k]).max())


def _step_presc(hp, t, dofs, vals, **kw):
    order = np.argsort(dofs)
    return hp.step(TimeStep(float(t), 1.0), prescribed=(np.asarray(dofs)[order], np.asarray(vals)[order]), **kw)


class _TwoMaterials:
    """Cell-wise J2 with two yield stresses (cell 0: sy1, cell 1: sy2), 8 quadrature points per H8 cell."""

    name = "two_j2"
    kinematics = "small"
    conservative = False
    state_fields = ("plastic_strain", "alpha")

    def __init__(self, sy1, sy2, nq=8):
        self.a, self.b, self.nq = J2Plasticity(0.3, sy1, 0.05), J2Plasticity(0.3, sy2, 0.05), nq

    def initial_state(self, n):
        return self.a.initial_state(n)

    def evaluate(self, kin, params, committed, request):
        from toporlogy.core.constitutive import Kinematics, MaterialResponse
        n = kin.n
        ia = np.arange(n) < self.nq
        out = []
        for model, sel in ((self.a, ia), (self.b, ~ia)):
            cs = {k: np.asarray(v)[sel] for k, v in committed.items()}
            out.append((sel, model.evaluate(Kinematics(kin.grad[sel], "small"), None, cs, request)))
        S = np.zeros((n, 6))
        C = np.zeros((n, 6, 6))
        trial = {k: np.zeros_like(np.asarray(v)) for k, v in committed.items()}
        for sel, r in out:
            S[sel], C[sel] = r.stress, r.tangent
            for k in trial:
                trial[k][sel] = r.trial_state[k]
        return MaterialResponse(S, C, None, trial, None, {})


@pytest.mark.xfail(strict=True, reason="pre-registered H2 FAILED on the first run: end displacement 2.777e-3 vs the "
                   "sum of uniaxial-stress material points 4.200e-3 at F = 1.1e-3; the hypothesis is wrong for a 3-D "
                   "bar: the two cells contract laterally differently and the shared interface nodes constrain it, so "
                   "the stress is not uniaxial")
def test_h2_series_bar_two_materials() -> None:
    m = box(2, 1, 1, 2.0, 1.0, 1.0)
    assert np.all(m.cell_centroids()[:, 0] == [0.5, 1.5])
    model = _TwoMaterials(1e-3, 2e-3)
    hp = HistoryProblem(m, model)
    sup = _bar_supports(m)
    right = np.flatnonzero(np.isclose(m.points[:, 0], 2.0))
    forces = [0.5e-3, 1.1e-3, 1.5e-3, 1.8e-3]                   # area 1: sigma_xx = F
    ctrl = np.zeros(6, bool)
    stress = np.zeros((len(forces), 6))
    stress[:, 0] = forces
    e1 = drive_material_point(model.a, np.zeros((len(forces), 6)), stress, ctrl).strain[:, 0]
    e2 = drive_material_point(model.b, np.zeros((len(forces), 6)), stress, ctrl).strain[:, 0]
    for k, F in enumerate(forces):
        f = np.zeros(hp.ndof)
        f[3 * right] = F / len(right)
        r = hp.step(TimeStep(float(k), 1.0), f_ext=f, prescribed=(sup, np.zeros(len(sup))), tol=1e-12)
        ux = r.u[3 * right].mean()
        ref = e1[k] + e2[k]
        assert abs(ux - ref) <= 1e-8 * abs(ref), (k, ux, ref)


def test_h3_constrained_thermal_bar() -> None:
    from toporlogy.core.properties import Constant, IsotropicModuliOfT, Polynomial, ThermoElasticPoint
    Tref, T1 = 300.0, 340.0
    Ef = Polynomial([3e9, -1e7], T_ref=300.0, domain=(250.0, 400.0), units="Pa", source="synthetic")
    nuf = Constant(0.35, units="1", source="synthetic", domain=(250.0, 400.0))
    alpha = Polynomial([6e-5, 2e-7], T_ref=Tref, domain=(250.0, 400.0), units="1/K", source="synthetic")
    model = ThermoElasticPoint(IsotropicModuliOfT(E=Ef, nu=nuf), alpha, Tref)
    m = box(2, 1, 1, 2.0, 1.0, 1.0)
    hp = HistoryProblem(m, model)
    P = m.points
    ends = np.flatnonzero(np.isclose(P[:, 0], 0.0) | np.isclose(P[:, 0], 2.0))
    dofs = sorted(set([3 * n for n in ends] + [3 * n + 1 for n in np.flatnonzero(np.isclose(P[:, 1], 0.0))]
                      + [3 * n + 2 for n in np.flatnonzero(np.isclose(P[:, 2], 0.0))]))
    r = hp.step(TimeStep(0.0, 1.0, T1, Tref), prescribed=(np.array(dofs), np.zeros(len(dofs))), tol=1e-12)
    dT = T1 - Tref
    eth = 6e-5 * dT + 0.5 * 2e-7 * dT ** 2
    E1 = 3e9 - 1e7 * dT
    ref = -E1 * eth
    s = r.stress.reshape(-1, 6)
    assert np.abs(s[:, 0] - ref).max() <= 1e-10 * abs(ref)
    left = [i for i, d in enumerate(dofs) if d % 3 == 0 and np.isclose(P[d // 3, 0], 0.0)]
    # support reaction on the x = 0 face = traction sigma.n A with n = -e_x, i.e. -sigma_xx A (A = 1)
    assert abs(r.reactions[left].sum() - (-ref)) <= 1e-10 * abs(ref)


def test_h4_rollback_and_retry() -> None:
    m = box(2, 1, 1, 2.0, 1.0, 1.0)
    model = J2Plasticity(0.3, 1e-3, hardening=0.05)
    sup = _bar_supports(m)
    right = np.flatnonzero(np.isclose(m.points[:, 0], 2.0))
    dofs = np.concatenate([sup, 3 * right])

    def presc(e):
        return np.concatenate([np.zeros(len(sup)), np.full(len(right), 2.0 * e)])
    ref = HistoryProblem(m, model)
    _step_presc(ref, 0, dofs, presc(6e-4))
    r_ref = _step_presc(ref, 1, dofs, presc(2e-3))
    hp = HistoryProblem(m, model)
    _step_presc(hp, 0, dofs, presc(6e-4))
    before = {k: v.copy() for k, v in hp.store.committed.items()}
    u_before = hp.u.copy()
    with pytest.raises(StepRejected):
        _step_presc(hp, 1, dofs, presc(2e-3), max_iter=0)
    assert all(np.array_equal(hp.store.committed[k], before[k]) for k in before)
    assert np.array_equal(hp.u, u_before)
    r = _step_presc(hp, 1, dofs, presc(2e-3))
    assert np.array_equal(r.u, r_ref.u) and np.array_equal(r.stress, r_ref.stress)
    assert all(np.array_equal(hp.store.committed[k], ref.store.committed[k]) for k in before)


def test_h5_refusals() -> None:
    m2 = rectangle(2, 2)
    with pytest.raises(ValueError, match="small-strain"):
        HistoryProblem(m2, NeoHooke(0.3))
    with pytest.raises(ValueError, match="plane strain only"):
        HistoryProblem(m2, LinearElastic(isotropic_plane_strain(1.0, 0.3)), plane="stress")
    with pytest.raises(ValueError, match="contract"):
        HistoryProblem(m2, type("X", (), {"kinematics": "small"})())
    hp = HistoryProblem(box(1, 1, 1), LinearElastic(isotropic_stiffness(1.0, 0.3)))
    with pytest.raises(ValueError, match="prescribed"):
        hp.step(TimeStep(0.0, 1.0), prescribed=([10 ** 6], [0.0]))


def test_h2b_series_bar_lateral_dofs_held_posthoc() -> None:
    """POST HOC (added after H2 failed): all lateral DOFs held, so every point is in uniaxial strain and the series
    reference is exact: end displacement = sum of the axial strains of material points with zero lateral strains
    and sigma_xx = F."""
    m = box(2, 1, 1, 2.0, 1.0, 1.0)
    model = _TwoMaterials(1e-3, 2e-3)
    hp = HistoryProblem(m, model)
    P = m.points
    sup = np.array(sorted([3 * n for n in np.flatnonzero(np.isclose(P[:, 0], 0.0))]
                          + [3 * n + c for n in range(m.n_nodes) for c in (1, 2)]))
    right = np.flatnonzero(np.isclose(P[:, 0], 2.0))
    forces = [0.5e-3, 1.1e-3, 1.5e-3, 1.8e-3]
    ctrl = np.array([False, True, True, False, False, False])
    stress = np.zeros((len(forces), 6))
    stress[:, 0] = forces
    zero = np.zeros((len(forces), 6))
    e1 = drive_material_point(model.a, zero, stress, ctrl).strain[:, 0]
    e2 = drive_material_point(model.b, zero, stress, ctrl).strain[:, 0]
    for k, F in enumerate(forces):
        f = np.zeros(hp.ndof)
        f[3 * right] = F / len(right)
        r = hp.step(TimeStep(float(k), 1.0), f_ext=f, prescribed=(sup, np.zeros(len(sup))), tol=1e-12)
        ux = r.u[3 * right].mean()
        assert abs(ux - (e1[k] + e2[k])) <= 1e-8 * abs(e1[k] + e2[k]), (k, ux, e1[k] + e2[k])

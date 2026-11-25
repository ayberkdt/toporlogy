"""Geometrically non-linear SVK with a per-element elasticity (orientation field): the checks pre-registered in
docs/olcumler/geometrik_dogrusal_olmama.md §8 (2026-11-25 19:53 +0300, before the first run). Thresholds are those of
the original V1, V3 and V6 of tests/toporlogy/test_nonlinear.py.
"""
from __future__ import annotations

import numpy as np

from toporlogy.core.domain import Domain
from toporlogy.core.material import Orthotropic, rotations_about_z, unit_elasticity_field
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearElasticity
from toporlogy.optimization.nonlinear import EndCompliance
from toporlogy.optimization.study import Study
from toporlogy.optimizers.mma import MMA

MAT = Orthotropic(40.0, 9.0, 9.0, 4.0, 4.0, 3.2, 0.28, 0.28, 0.4)


def _field(d: Domain, base: float = 20.0, slope: float = 9.0) -> np.ndarray:
    """(n_active, 3, 3) unit D of MAT rotated by an angle that varies along x and y."""
    ix, iy = np.unravel_index(d.active_elements, d.shape)
    return unit_elasticity_field(MAT, 2, "stress", rotations_about_z(base + slope * ix - 0.5 * slope * iy))


def _cantilever(shape=(8, 4)):
    d = Domain(shape, h=0.5, thickness=0.8)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - X[:, 0].max()) < 1e-9)
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -1.0 / len(tip)
    return d, fixed, F


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _deformed(nl: NonlinearElasticity, d: Domain, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = d.node_coords()
    th = 0.5
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    G = R @ (np.eye(2) + 0.2 * np.diag(rng.random(2))) - np.eye(2)
    u = (X @ G.T).ravel() + 0.08 * d.h * rng.standard_normal(d.ndof)
    E = np.exp(rng.uniform(np.log(2e-4), 0.0, nl.n_elements))
    return u, E


def test_n1_uniform_field_equals_single_D():
    d, fixed, _ = _cantilever()
    one = unit_elasticity_field(MAT, 2, "stress", rotations_about_z(np.full(d.n_active, 30.0)))
    a = NonlinearElasticity(d, fixed, elasticity=one[0], energy_interpolation=EnergyInterpolation(E0=1.0))
    b = NonlinearElasticity(d, fixed, elasticity=one, energy_interpolation=EnergyInterpolation(E0=1.0))
    u, E = _deformed(a, d)
    sa, sb = a.element_state(u, E), b.element_state(u, E)
    errs = {k: _rel(sb[k], sa[k]) for k in ("f", "K")}
    errs["energy"] = abs(b.strain_energy(u, E) - a.strain_energy(u, E)) / abs(a.strain_energy(u, E))
    print(f"\n  N1: {errs}")
    assert max(errs.values()) <= 1e-13


def test_n2_small_load_equals_linear_with_the_same_field():
    d, fixed, F = _cantilever()
    D = _field(d)
    lin = Elasticity(d, fixed, elasticity=D)
    E = 0.05 + 0.95 * np.random.default_rng(1).random(lin.n_elements)
    UL = lin.solve_numpy(E, F)[:, 0]
    scale = 1e-8 / (np.abs(UL).max() / d.h)
    nl = NonlinearElasticity(d, fixed, elasticity=D, rtol=1e-12, energy_interpolation=EnergyInterpolation(E0=1.0))
    UN = nl.solve_numpy(E, scale * F)[:, 0] / scale
    rel = np.linalg.norm(UN - UL) / np.linalg.norm(UL)
    print(f"\n  N2: {rel:.2e}")
    assert rel <= 1e-6


def test_n3_consistent_tangent_with_a_field():
    d, fixed, _ = _cantilever((4, 3))
    nl = NonlinearElasticity(d, fixed, elasticity=_field(d), energy_interpolation=EnergyInterpolation(E0=1.0))
    u, E = _deformed(nl, d)
    du = np.zeros_like(u)
    du[nl.free] = np.random.default_rng(11).standard_normal(len(nl.free))
    du /= np.abs(du).max()
    eps = 1e-6
    Kd = nl.tangent(u, E) @ du[nl.free]
    fd = (nl.internal_force(u + eps * du, E) - nl.internal_force(u - eps * du, E))[nl.free] / (2 * eps)
    rel = np.linalg.norm(Kd - fd) / np.linalg.norm(Kd)
    print(f"\n  N3: {rel:.2e}")
    assert rel <= 1e-6


def test_n4_end_compliance_gradient_with_a_field():
    d = Domain((16, 6), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero((np.abs(X[:, 0] - X[:, 0].max()) < 1e-9) & (X[:, 1] < 1e-9))
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -0.02 / len(tip)
    nl = NonlinearElasticity(d, fixed, elasticity=_field(d, slope=4.0), rtol=1e-12,
                             energy_interpolation=EnergyInterpolation(E0=1.0))
    chain = Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)])
    st = Study(d, nl, F, chain, SIMP(d, E0=1.0, penal=3.0, emin_ratio=1e-6), EndCompliance(), [], MMA())
    x = 0.3 + 0.6 * np.random.default_rng(4).random(d.n_design)
    ev = st.evaluate(x)
    idx = np.random.default_rng(5).choice(d.n_design, 6, replace=False)
    fd = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-5
        xm[i] -= 1e-5
        fd.append((st.evaluate(xp, gradients=False).f0 - st.evaluate(xm, gradients=False).f0) / 2e-5)
    rel = np.abs(ev.df0[idx] - np.asarray(fd)).max() / np.abs(fd).max()
    umax = np.abs(ev.U).max() / X[:, 0].max()
    print(f"\n  N4: rel {rel:.2e}, max|u|/span {umax:.3f}")
    assert rel <= 1e-5


def test_n5_yaml_angle_field_with_nonlinear_physics(tmp_path):
    from toporlogy.configuration.build import study_from_dict
    f = tmp_path / "angle.npy"
    ix, iy = np.meshgrid(np.arange(16), np.arange(6), indexing="ij")
    np.save(f, 15.0 + 5.0 * ix - 3.0 * iy)
    cfg = {"domain": {"type": "box", "shape": [16, 6], "h": 1.0},
           "material": {"E0": 40.0, "nu": 0.3, "orthotropic": {"E1": 40.0, "E2": 9.0, "E3": 9.0, "G12": 4.0,
                                                               "G13": 4.0, "G23": 3.2, "nu12": 0.28, "nu13": 0.28,
                                                               "nu23": 0.4, "angle_field": str(f)}},
           "physics": {"type": "nonlinear_elasticity"},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                                     "force": [0, -0.4]}]}],
           "objective": "end_compliance", "constraints": [{"response": "volume", "upper": 0.5}],
           "optimizer": {"type": "mma", "move": 0.1}, "run": {"max_iter": 2}}
    st = study_from_dict(cfg)
    assert st.physics.D.shape == (st.domain.n_active, 3, 3)
    ev = st.evaluate(st.default_start())
    assert np.isfinite(ev.f0) and np.all(np.isfinite(ev.df0))

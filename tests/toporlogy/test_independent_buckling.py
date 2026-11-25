"""Linear buckling against a separately written dense implementation (docs/olcumler/dogrulama_turu.md B1-B6,
pre-registered 2026-11-25 21:05 +0300 before the first run).

K and K_G are built in ``_independent_fe`` from the definitions (own shape functions, Gauss rule, materials,
rotations and assembly loop); the eigenproblem −K_G φ = μ K φ is solved densely with ``scipy.linalg.eigh``. The
library under test is ``Elasticity`` (reference solve) and ``LinearBuckling`` (K_G, shift-invert Lanczos, Rayleigh–Ritz).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import _independent_fe as ind
from toporlogy.core.domain import Domain
from toporlogy.core.material import Orthotropic, rotations_about_z, unit_elasticity_field
from toporlogy.fem.buckling import LinearBuckling
from toporlogy.fem.elastic import Elasticity

ENG = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)
MAT = Orthotropic(**ENG)
NU = 0.3
TOL_MU = 1e-9          # pre-registered B1-B4, B6
TOL_RES = 1e-7         # pre-registered B5
TOL_NORM = 1e-9


def _grid_conventions(d: Domain) -> None:
    """The conventions ``_independent_fe`` relies on, asserted against the library's grid."""
    X = d.node_coords()
    node_shape = np.array(d.shape) + 1
    idx = np.array(np.unravel_index(np.arange(len(X)), node_shape)).T
    assert np.allclose(X, idx * d.h)
    assert np.array_equal(d.active_elements, np.arange(int(np.prod(d.shape))))
    assert np.array_equal(d.dofs(np.arange(len(X)), 0), np.arange(len(X)) * d.dim)
    # every library element touches the same node set as the independent element with the same number
    lib = np.sort(d.edof, axis=1)
    own = np.sort(ind.element_dofs(d.shape), axis=1)
    assert np.array_equal(lib, own)


def _problem(dim: int, field: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    if dim == 2:
        d = Domain((12, 4), h=0.5, thickness=0.8)
    else:
        d = Domain((6, 3, 3), h=0.5)
    _grid_conventions(d)
    X = d.node_coords()
    left = np.flatnonzero(X[:, 0] < 1e-9)
    right = np.flatnonzero(X[:, 0] > X[:, 0].max() - 1e-9)
    fixed = d.dofs(left)
    F = np.zeros(d.ndof)
    F[d.dim * right] = -1.0 / len(right)                     # compression along x
    F[d.dim * right[:1] + 1] = 0.02                          # small lateral load
    n = d.n_active
    E = 0.2 + 0.8 * rng.random(n)
    if field:
        ang = rng.uniform(-90.0, 90.0, n)
        D_lib = unit_elasticity_field(MAT, dim, "stress", rotations_about_z(ang))
        C3 = ind.orthotropic_3d(**ENG) / ENG["E1"]
        Ds = np.array([ind.rotate(C3, ind.rot_z(a)) for a in ang])
        if dim == 2:
            Ds = np.array([ind.plane_stress(c) for c in Ds])
        phys = Elasticity(d, fixed, elasticity=D_lib, element="Q4" if dim == 2 else "H8")
    else:
        D1 = ind.plane_stress_isotropic(1.0, NU) if dim == 2 else ind.isotropic_3d(1.0, NU)
        Ds = np.broadcast_to(D1, (n,) + D1.shape)
        phys = Elasticity(d, fixed, nu=NU, element="Q4" if dim == 2 else "H8")
    free = np.setdiff1d(np.arange(d.ndof), fixed)
    return d, phys, F, E, Ds, free


def _dense(d, E, Ds, F, free, F0=None):
    t = d.thickness if d.dim == 2 else 1.0
    K = ind.stiffness(d.shape, d.h, t, Ds, E)
    Kf = K[np.ix_(free, free)]
    u = np.zeros(d.ndof)
    u[free] = np.linalg.solve(Kf, F[free])
    KG = ind.geometric_stiffness(d.shape, d.h, t, Ds, E, u)[np.ix_(free, free)]
    M = Kf
    if F0 is not None:
        u0 = np.zeros(d.ndof)
        u0[free] = np.linalg.solve(Kf, F0[free])
        M = Kf + ind.geometric_stiffness(d.shape, d.h, t, Ds, E, u0)[np.ix_(free, free)]
    mu = sla.eigh(-KG, M, eigvals_only=True)
    return np.sort(mu)[::-1], Kf, KG, u


CASES = [(2, False), (3, False), (2, True), (3, True)]
IDS = ["B1-Q4", "B2-H8", "B3-Q4-field", "B4-H8-field"]


@pytest.mark.parametrize("dim,field", CASES, ids=IDS)
def test_load_factors_equal_the_dense_independent_solution(dim, field):
    d, phys, F, E, Ds, free = _problem(dim, field)
    mu_ref, Kf, _, u_ref = _dense(d, E, Ds, F, free)
    u = phys.solve_numpy(E, F[:, None])[:, 0]
    assert np.abs(u - u_ref).max() <= 1e-10 * np.abs(u_ref).max()
    mu, _ = LinearBuckling(phys).modes(E, E, u, k=6)
    err = np.abs(mu - mu_ref[:6]).max() / abs(mu_ref[0])
    print(f"\n  {dim}-D field={field}: mu {mu_ref[:6]}, max |dmu|/|mu1| = {err:.2e}")
    assert err <= TOL_MU


@pytest.mark.parametrize("dim,field", CASES, ids=["B5-" + i[3:] for i in IDS])
def test_library_modes_solve_the_independent_eigenproblem(dim, field):
    d, phys, F, E, Ds, free = _problem(dim, field, seed=1)
    _, Kf, KG, _ = _dense(d, E, Ds, F, free)
    u = phys.solve_numpy(E, F[:, None])[:, 0]
    mu, modes = LinearBuckling(phys).modes(E, E, u, k=6)
    res, nrm = 0.0, 0.0
    for j in range(6):
        p = modes[free, j]
        Kp = Kf @ p
        res = max(res, float(np.linalg.norm(-KG @ p - mu[j] * Kp) / (abs(mu[j]) * np.linalg.norm(Kp))))
        nrm = max(nrm, abs(float(p @ Kp) - 1.0))
    print(f"\n  {dim}-D field={field}: residual {res:.2e}, |phi K phi - 1| {nrm:.2e}")
    assert res <= TOL_RES and nrm <= TOL_NORM


@pytest.mark.parametrize("dim,field", CASES, ids=["B6-" + i[3:] for i in IDS])
def test_preload_equals_the_dense_independent_solution(dim, field):
    d, phys, F, E, Ds, free = _problem(dim, field, seed=2)
    X = d.node_coords()
    top = np.flatnonzero(X[:, 1] > X[:, 1].max() - 1e-9)
    F0 = np.zeros(d.ndof)
    F0[d.dim * top + 1] = -1.0 / len(top)                     # held load: pressure on the top face
    # scaled so that the preload alone has mu0 = 0.4 (it must not buckle by itself; K_G is linear in the load).
    # First run: a fixed -0.3 gave mu0 = 4.3 in 2-D, i.e. the preload alone buckled (test set-up error).
    t = d.thickness if d.dim == 2 else 1.0
    K = ind.stiffness(d.shape, d.h, t, Ds, E)[np.ix_(free, free)]
    u0 = np.zeros(d.ndof)
    u0[free] = np.linalg.solve(K, F0[free])
    KG0d = ind.geometric_stiffness(d.shape, d.h, t, Ds, E, u0)[np.ix_(free, free)]
    F0 *= 0.4 / sla.eigh(-KG0d, K, eigvals_only=True).max()
    mu_ref, *_ = _dense(d, E, Ds, F, free, F0=F0)
    U = phys.solve_numpy(E, np.column_stack([F, F0]))
    lb = LinearBuckling(phys)
    KG0 = lb.geometric_matrix(E, U[:, 1])
    mu, _ = lb.modes(E, E, U[:, 0], k=6, KG0=KG0)
    err = np.abs(mu - mu_ref[:6]).max() / abs(mu_ref[0])
    print(f"\n  {dim}-D field={field} preload: mu {mu_ref[:3]}, max |dmu|/|mu1| = {err:.2e}")
    assert err <= TOL_MU

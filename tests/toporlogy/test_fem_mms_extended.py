"""Manufactured solutions for the newer physics (docs/olcumler/dogrulama_turu.md M1-M4, pre-registered
2026-11-25 21:05 +0300 before the first run).

Same method as ``test_fem_mms.py``: a smooth field that vanishes on the boundary, the source it needs written here,
consistent nodal loads integrated here (multilinear hats, 3-point Gauss per axis), all boundary nodes fixed, and the
nodal error max|U − u| / max|u| on three grids. For a variable coefficient (orientation field) and for the
non-linear laws the source b = −Div P is formed by central differences (δ = 1e-5) of the analytic P(X) written here;
that error (~1e-10) is far below the measured FE errors. Nothing of the library is used except ``Domain`` and the
physics under test (``Elasticity`` with ``elasticity=``, ``HeatConduction``, ``NonlinearElasticity``) and, for the FE
input only, the library's orientation field of the element centres.
"""
from __future__ import annotations

import numpy as np
import pytest

import _independent_fe as ind
from test_fem_mms import solve as solve_linear_mms
from toporlogy.core.domain import Domain
from toporlogy.core.material import Orthotropic, rotations_about_z, unit_elasticity_field
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.heat import HeatConduction
from toporlogy.fem.nonlinear import NonlinearElasticity

G3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5, 8, 5]) / 9.0)
ENG = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)
MAT = Orthotropic(**ENG)
DELTA = 1e-5
RATE = (1.8, 2.25)                                         # pre-registered


def _rates(e):
    return [float(np.log2(e[i] / e[i + 1])) for i in range(len(e) - 1)]


# ------------------------------------------------------------------------------------------ manufactured field
def _modes(dim):
    if dim == 2:
        return [(1.0, (1, 1)), (0.5, (2, 1))]
    return [(1.0, (1, 1, 1)), (0.5, (2, 1, 1)), (0.3, (1, 2, 1))]


def _L(dim):
    return (2.0, 1.0) if dim == 2 else (1.0, 1.0, 1.0)


def u_field(X, dim, amp=1.0):
    k = [np.pi / v for v in _L(dim)]
    return amp * np.stack([a * np.prod([np.sin(m * kk * X[:, i]) for i, (m, kk) in enumerate(zip(mm, k))], 0)
                           for a, mm in _modes(dim)], 1)


def grad_u(X, dim, amp=1.0):
    """(n, dim, dim) ∂u_c/∂X_j."""
    k = [np.pi / v for v in _L(dim)]
    G = np.empty((len(X), dim, dim))
    for c, (a, mm) in enumerate(_modes(dim)):
        for j in range(dim):
            out = np.full(len(X), amp * a)
            for q in range(dim):
                w = mm[q] * k[q]
                out = out * (w * np.cos(w * X[:, q]) if q == j else np.sin(w * X[:, q]))
            G[:, c, j] = out
    return G


def consistent_loads(d, f_fn, ncomp):
    """Σ_cells ∫ f N_node dV, f_fn(X (n, dim)) -> (n, ncomp); DOF = node · ncomp + component."""
    dim, h, shape = d.dim, d.h, d.shape
    g, w = G3
    pts = np.array(np.meshgrid(*[g] * dim, indexing="ij")).reshape(dim, -1).T
    wts = np.prod(np.array(np.meshgrid(*[w] * dim, indexing="ij")).reshape(dim, -1), 0)
    cells = np.indices(shape).reshape(dim, -1).T
    Xg = (cells[:, None, :] + (pts[None] + 1) / 2) * h
    fg = f_fn(Xg.reshape(-1, dim)).reshape(len(cells), len(pts), ncomp)
    jac = (h / 2) ** dim * (1.0 if dim == 3 else d.thickness)
    node_shape = np.array(shape) + 1
    F = np.zeros(int(np.prod(node_shape)) * ncomp)
    for cn in ind.corners(dim):
        N = np.prod(np.where(cn[None, :] == 1, (1 + pts) / 2, (1 - pts) / 2), 1)
        nodes = np.ravel_multi_index((cells + cn).T, node_shape)
        load = np.einsum("g,egc,g->ec", N, fg, wts) * jac
        for c in range(ncomp):
            np.add.at(F, nodes * ncomp + c, load[:, c])
    return F


def divergence_source(P_fn, dim):
    """X -> −Div P with P_fn(X) -> (n, dim, dim) (first index: component, second: derivative direction)."""
    def f(X):
        out = np.zeros((len(X), dim))
        for j in range(dim):
            e = np.zeros(dim)
            e[j] = DELTA
            out -= (P_fn(X + e)[:, :, j] - P_fn(X - e)[:, :, j]) / (2 * DELTA)
        return out
    return f


def _grid(n, dim, thickness=1.0):
    h = 1.0 / n
    d = Domain(tuple(int(round(v / h)) for v in _L(dim)), h, thickness=thickness)
    X = d.node_coords()
    bnd = np.flatnonzero(np.any((X < 1e-9) | (np.array(_L(dim)) - 1e-9 < X), axis=1))
    return d, X, bnd


def _error(U, ue):
    return float(np.max(np.abs(U - ue)) / np.max(np.abs(ue)))


# ================================================================================================= M1 Q4IM
def test_m1_q4im_converges_at_second_order():
    e = [solve_linear_mms(n, 2, "Q4IM") for n in (8, 16, 32)]
    r = _rates(e)
    print(f"\n  M1 Q4IM: errors {e}, rates {r}")
    assert all(RATE[0] <= v <= RATE[1] for v in r) and e[-1] <= 1e-3


# ===================================================================================== M2 orientation field
def theta(X):
    if X.shape[1] == 2:
        return 30.0 + 40.0 * X[:, 0] / 2.0 + 20.0 * X[:, 1]
    return 30.0 + 40.0 * X[:, 0] + 20.0 * X[:, 1] + 15.0 * X[:, 2]


def _strain_transform(R):
    """(n, 6, 6) T with ε_material (Voigt, engineering) = T ε_global for material axes R (n, 3, 3)."""
    T = np.empty((len(R), 6, 6))
    for b, (i, j) in enumerate(ind.VOIGT[3]):
        eg = np.zeros((3, 3))
        if i == j:
            eg[i, i] = 1.0
        else:
            eg[i, j] = eg[j, i] = 0.5                       # unit engineering shear strain
        em = np.einsum("nai,ab,nbj->nij", R, eg, R)          # Rᵀ ε R
        for a, (p, q) in enumerate(ind.VOIGT[3]):
            T[:, a, b] = em[:, p, q] * (1.0 if p == q else 2.0)
    return T


C_MAT = ind.orthotropic_3d(**ENG)


def D_at(X):
    """Physical stiffness (n, nv, nv) in global axes at points X (plane stress in 2-D)."""
    t = np.deg2rad(theta(X))
    c, s = np.cos(t), np.sin(t)
    R = np.zeros((len(X), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    T = _strain_transform(R)
    Cg = np.einsum("nab,bc,ncd->nad", T.transpose(0, 2, 1), C_MAT, T)
    if X.shape[1] == 3:
        return Cg
    ip = [0, 1, 5]
    return np.linalg.inv(np.linalg.inv(Cg)[:, ip][:, :, ip])


def test_m2_transform_agrees_with_the_tensor_rotation():
    """Sanity of this file's energy-based rotation against ``_independent_fe.rotate`` (tensor rotation)."""
    X = np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.9]])
    for x, D in zip(X, D_at(X), strict=True):
        assert np.allclose(D, ind.rotate(C_MAT, ind.rot_z(float(theta(x[None])[0]))), rtol=1e-13, atol=1e-12)


def _stress_tensor(X, D, dim, amp=1.0):
    G = grad_u(X, dim, amp)
    eps = np.stack([G[:, i, i] if i == j else G[:, i, j] + G[:, j, i] for i, j in ind.VOIGT[dim]], 1)
    sv = np.einsum("nab,nb->na", D, eps)
    S = np.zeros((len(X), dim, dim))
    for v, (i, j) in enumerate(ind.VOIGT[dim]):
        S[:, i, j] = S[:, j, i] = sv[:, v]
    return S


def solve_orientation(n, dim, element, constant=False):
    d, X, bnd = _grid(n, dim)
    F = consistent_loads(d, divergence_source(lambda Y: _stress_tensor(Y, D_at(Y), dim), dim), dim)
    ix = np.array(np.unravel_index(d.active_elements, d.shape)).T
    Xc = (ix + 0.5) * d.h
    ang = np.full(len(Xc), float(theta(np.array([[v / 2 for v in _L(dim)]]))[0])) if constant else theta(Xc)
    D = unit_elasticity_field(MAT, dim, "stress", rotations_about_z(ang))
    phys = Elasticity(d, d.dofs(bnd), elasticity=D, element=element)
    U = phys.solve_numpy(np.full(d.n_active, ENG["E1"]), F[:, None])[:, 0].reshape(-1, dim)
    return _error(U, u_field(X, dim))


@pytest.mark.parametrize("dim,element,ns,finest", [(2, "Q4", (8, 16, 32), 5e-3), (2, "Q4IM", (8, 16, 32), 5e-3),
                                                   (3, "H8", (4, 8, 16), 5e-2)], ids=["M2a", "M2b", "M2c"])
def test_m2_orientation_field_converges_at_second_order(dim, element, ns, finest):
    e = [solve_orientation(n, dim, element) for n in ns]
    r = _rates(e)
    print(f"\n  {element} orientation field: errors {e}, rates {r}")
    assert all(RATE[0] <= v <= RATE[1] for v in r) and e[-1] <= finest


def test_m2d_constant_orientation_does_not_converge():
    e = [solve_orientation(n, 2, "Q4", constant=True) for n in (8, 16, 32)]
    r = _rates(e)
    print(f"\n  M2d constant angle: errors {e}, rates {r}")
    assert r[-1] < 1.0


# ========================================================================================== M3 heat conduction
def _temperature(X):
    return u_field(X, X.shape[1])[:, 0]


def _heat_source(X):
    dim = X.shape[1]
    k = [np.pi / v for v in _L(dim)]
    a, mm = _modes(dim)[0]
    return (sum((m * kk) ** 2 for m, kk in zip(mm, k)) * _temperature(X))[:, None]   # −lap T


def solve_heat(n, dim, k_fe=1.0):
    d, X, bnd = _grid(n, dim, thickness=0.7)
    Q = consistent_loads(d, _heat_source, 1)
    T = HeatConduction(d, bnd).solve_numpy(np.full(d.n_active, k_fe), Q)[:, 0]
    return _error(T, _temperature(X))


@pytest.mark.parametrize("dim,ns,finest", [(2, (8, 16, 32), 1e-3), (3, (4, 8, 16), 1e-2)], ids=["M3a", "M3b"])
def test_m3_heat_conduction_converges_at_second_order(dim, ns, finest):
    e = [solve_heat(n, dim) for n in ns]
    r = _rates(e)
    print(f"\n  M3 heat {dim}-D: errors {e}, rates {r}")
    assert all(RATE[0] <= v <= RATE[1] for v in r) and e[-1] <= finest


def test_m3c_wrong_conductivity_does_not_converge():
    e = [solve_heat(n, 2, k_fe=1.1) for n in (8, 16, 32)]
    r = _rates(e)
    print(f"\n  M3c k x 1.1: errors {e}, rates {r}")
    assert r[-1] < 1.0


# ================================================================================ M4 geometric non-linearity
NU = 0.3


def _amp(dim):
    """Amplitude giving max |∇u| = 0.2 (sampled on a fine grid)."""
    s = np.linspace(0, 1, 41)
    X = np.array(np.meshgrid(*[s * v for v in _L(dim)], indexing="ij")).reshape(dim, -1).T
    return 0.2 / np.abs(grad_u(X, dim)).max()


def P_svk(X, dim, amp):
    Fd = np.eye(dim)[None] + grad_u(X, dim, amp)
    Eg = 0.5 * (np.einsum("nki,nkj->nij", Fd, Fd) - np.eye(dim)[None])
    ev = np.stack([Eg[:, i, i] if i == j else 2 * Eg[:, i, j] for i, j in ind.VOIGT[dim]], 1)
    D = ind.plane_stress_isotropic(1.0, NU) if dim == 2 else ind.isotropic_3d(1.0, NU)
    sv = ev @ D.T
    S = np.zeros((len(X), dim, dim))
    for v, (i, j) in enumerate(ind.VOIGT[dim]):
        S[:, i, j] = S[:, j, i] = sv[:, v]
    return np.einsum("nik,nkj->nij", Fd, S)


def _psi_nh(F3):
    kap, mu = 1.0 / (3 * (1 - 2 * NU)), 1.0 / (2 * (1 + NU))
    J = np.linalg.det(F3)
    I1 = np.einsum("nij,nij->n", F3, F3)
    return 0.5 * kap * (J - 1) ** 2 + 0.5 * mu * (J ** (-2 / 3) * I1 - 3)


def _P_nh3(F3):
    kap, mu = 1.0 / (3 * (1 - 2 * NU)), 1.0 / (2 * (1 + NU))
    J = np.linalg.det(F3)
    FinvT = np.linalg.inv(F3).transpose(0, 2, 1)
    I1 = np.einsum("nij,nij->n", F3, F3)
    return ((kap * (J - 1) * J)[:, None, None] * FinvT
            + (mu * J ** (-2 / 3))[:, None, None] * (F3 - (I1 / 3)[:, None, None] * FinvT))


def P_nh(X, dim, amp):
    Fd = np.eye(dim)[None] + grad_u(X, dim, amp)
    if dim == 3:
        return _P_nh3(Fd)
    F3 = np.zeros((len(X), 3, 3))
    F3[:, :2, :2], F3[:, 2, 2] = Fd, 1.0                       # plane strain
    return _P_nh3(F3)[:, :2, :2]


def test_m4_neo_hooke_stress_is_the_energy_derivative():
    """Sanity of this file's P = ∂ψ/∂F (central differences of ψ)."""
    F = np.eye(3)[None] + 0.2 * np.random.default_rng(0).standard_normal((3, 3, 3))
    P = _P_nh3(F)
    for i in range(3):
        for j in range(3):
            E = np.zeros((3, 3))
            E[i, j] = 1e-6
            fd = (_psi_nh(F + E) - _psi_nh(F - E)) / 2e-6
            assert np.allclose(fd, P[:, i, j], rtol=1e-7, atol=1e-9)


def solve_nonlinear(n, dim, material, linear=False):
    amp = _amp(dim)
    d, X, bnd = _grid(n, dim)
    Pf = (lambda Y: P_svk(Y, dim, amp)) if material == "svk" else (lambda Y: P_nh(Y, dim, amp))
    F = consistent_loads(d, divergence_source(Pf, dim), dim)
    fixed = d.dofs(bnd)
    if linear:
        ip = [0, 1, 5]
        D = ind.plane_stress_isotropic(1.0, NU) if material == "svk" else ind.isotropic_3d(1.0, NU)[np.ix_(ip, ip)]
        U = Elasticity(d, fixed, elasticity=D).solve_numpy(np.ones(d.n_active), F[:, None])
    else:
        plane = "stress" if (material == "svk" or dim == 3) else "strain"
        phys = NonlinearElasticity(d, fixed, nu=NU, material=material, plane=plane, rtol=1e-11)
        U = phys.solve_numpy(np.ones(d.n_active), F[:, None])
    return _error(U[:, 0].reshape(-1, dim), u_field(X, dim, amp))


_SLOW = pytest.mark.slow                                    # ~60 s each (3-D, 14 k DOF, Newton per load step)
M4 = [pytest.param(2, "svk", (8, 16, 32), id="M4a"), pytest.param(3, "svk", (4, 8, 16), id="M4b", marks=_SLOW),
      pytest.param(2, "neo_hooke", (8, 16, 32), id="M4c"),
      pytest.param(3, "neo_hooke", (4, 8, 16), id="M4d", marks=_SLOW)]


@pytest.mark.parametrize("dim,material,ns", M4)
def test_m4_nonlinear_manufactured_solution_converges_at_second_order(dim, material, ns):
    e = [solve_nonlinear(n, dim, material) for n in ns]
    r = _rates(e)
    print(f"\n  M4 {material} {dim}-D: errors {e}, rates {r}")
    assert all(RATE[0] <= v <= RATE[1] for v in r)


@pytest.mark.parametrize("material", ["svk", "neo_hooke"])
def test_m4e_linear_solver_misses_the_nonlinear_solution(material):
    nl = solve_nonlinear(32, 2, material)
    lin = solve_nonlinear(32, 2, material, linear=True)
    print(f"\n  M4e {material}: non-linear {nl:.2e}, linear {lin:.2e}")
    assert lin >= 10 * nl

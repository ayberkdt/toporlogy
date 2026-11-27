"""F3 / F4 assembly totals, reactions and the general linear constraint transform (docs/foundations/prereg/
F34_assembly.md, preregistered 2026-11-27 09:38 +0300, amended 09:43 before the first run).

References are closed forms, tensor Gauss-Legendre rules written here, scipy quad, a dense KKT solve and an explicit
Lame D; only CT1 compares with the production ``solve`` and is labelled a regression check.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.integrate import quad

from toporlogy.fem.mechanisms import MechanismError
from toporlogy.fem.mesh import (Dirichlet, ElasticProblem, HeatProblem, Mesh, box, get, perturb, rectangle)
from toporlogy.fem.mesh.constraints import (ConstraintError, ContradictoryConstraints, LinearConstraints,
                                            RedundantConstraints, macro_jump, periodic, periodic_lattice,
                                            solve_constrained, solve_lagrange, solve_penalty)
from toporlogy.fem.mesh.linear import Factorization, condition_1norm

SOLID = ("Q4", "Q9", "T3", "T6", "H8", "T4", "T10")
QUAD2 = ("T6", "Q9", "T10")                         # degree-2 elements that can be curved
SIDES = {2: (("left", 0, -1), ("right", 0, 1), ("bottom", 1, -1), ("top", 1, 1)),
         3: (("left", 0, -1), ("right", 0, 1), ("bottom", 1, -1), ("top", 1, 1), ("back", 2, -1), ("front", 2, 1))}
L2D, L3D = (1.2, 0.9), (1.0, 0.8, 0.7)
TH2D = 0.7


def dims(name):
    return L2D if get(name).dim == 2 else L3D


def thick(name):
    return TH2D if get(name).dim == 2 else 1.0


def mesh_of(name, curved=False, n=(4, 3, 2), amp=0.2, seed=3):
    d = get(name).dim
    L = dims(name)
    m = rectangle(n[0], n[1], lx=L[0], ly=L[1], element=name) if d == 2 else \
        box(n[0], n[1], n[2], lx=L[0], ly=L[1], lz=L[2], element=name)
    h = min(l / k for l, k in zip(L, n, strict=False))
    return perturb(m, amp * h, seed=seed, curved=(0.1 * h if curved else 0.0))


def lame_D(dim, E=1.7, nu=0.3):
    """Explicit isotropic D (Voigt, engineering shear); 2-D plane strain."""
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    if dim == 2:
        return np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
    D = np.full((6, 6), 0.0)
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] = lam + 2 * mu
    D[3:, 3:] = mu * np.eye(3)
    return D


def sigma_of(D, H):
    """Cauchy stress tensor of the small strain sym(H)."""
    dim = len(H)
    if dim == 2:
        e = np.array([H[0, 0], H[1, 1], H[0, 1] + H[1, 0]])
        s = D @ e
        return np.array([[s[0], s[2]], [s[2], s[1]]])
    e = np.array([H[0, 0], H[1, 1], H[2, 2], H[1, 2] + H[2, 1], H[0, 2] + H[2, 0], H[0, 1] + H[1, 0]])
    s = D @ e
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


def gauss_box(lo, hi, f, npts=4):
    """Integral of f(X (n, dim)) -> (n, k) over the box [lo, hi] by a tensor Gauss-Legendre rule (npts per axis)."""
    x, w = np.polynomial.legendre.leggauss(npts)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    axes = [0.5 * (h - l) * x + 0.5 * (h + l) for l, h in zip(lo, hi, strict=True)]
    wts = [0.5 * (h - l) * w for l, h in zip(lo, hi, strict=True)]
    G = np.meshgrid(*axes, indexing="ij")
    W = np.meshgrid(*wts, indexing="ij")
    X = np.stack([g.ravel() for g in G], axis=1)
    Wt = np.prod(np.stack([g.ravel() for g in W], axis=1), axis=1)
    return np.einsum("n,nk->k", Wt, np.asarray(f(X), float))


def cross_rows(a, b):
    """Row-wise 'moment' a x b: 3-D vector, 2-D scalar (z)."""
    if a.shape[1] == 2:
        return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])[:, None]
    return np.cross(a, b)


def nodal_moment(X, F, p):
    return cross_rows(X - p, F).sum(axis=0)


def cell_volumes(m):
    """Independent volumes of straight-sided cells (shoelace, determinant, own trilinear Gauss rule)."""
    P = m.points[m.cells]
    nm = m.element
    if nm in ("T3", "T6"):
        a, b, c = P[:, 0], P[:, 1], P[:, 2]
        return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    if nm in ("Q4", "Q9"):
        x, y = P[:, :4, 0], P[:, :4, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
    if nm in ("T4", "T10"):
        M = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], axis=1)
        return np.abs(np.linalg.det(M)) / 6.0
    g, w = np.polynomial.legendre.leggauss(2)
    s = get("H8").nodes                                  # reference corner signs (data, not a kernel)
    vol = np.zeros(len(P))
    for xi in np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T:
        wq = np.prod([w[list(g).index(c)] for c in xi])
        dN = np.stack([0.125 * s[:, i] * np.prod([1 + s[:, j] * xi[j] for j in range(3) if j != i], axis=0)
                       for i in range(3)], axis=1)       # (8, 3)
        J = np.einsum("eni,nj->eij", P, dN)
        vol += wq * np.linalg.det(J)
    return vol


def box_volume(name):
    L = dims(name)
    return float(np.prod(L)) * thick(name)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ---------------------------------------------------------------------------------------------- FA1 - FA3
@pytest.mark.parametrize("name", SOLID)
def test_fa1_mass_components_and_per_cell_density(name):
    m = mesh_of(name)
    dim = m.dim
    P = ElasticProblem(m, lame_D(dim), thickness=thick(name))
    V = cell_volumes(m) * thick(name)
    assert abs(V.sum() - box_volume(name)) <= 1e-13 * box_volume(name)        # the reference volumes themselves
    rho = 2.3
    M = P.mass(rho)
    for a in range(dim):
        for b in range(dim):
            blk = M[a::dim, b::dim].sum()
            assert abs(blk - (rho * box_volume(name) if a == b else 0.0)) <= 1e-12 * rho * box_volume(name)
    rho_e = 1.0 + 0.5 * (np.arange(m.n_cells) % 3)
    Me = P.mass(rho_e)
    exp = float((rho_e * V).sum())
    for a in range(dim):
        assert abs(Me[a::dim, a::dim].sum() - exp) <= 1e-12 * exp
    Ml = P.mass(rho_e, lumped=True)
    d = Ml.diagonal()[P.active]
    assert (d > 0).all()
    assert abs(Ml.sum() / dim - exp) <= 1e-12 * exp


@pytest.mark.parametrize("name", SOLID)
def test_fa2_mass_first_moment_distorted_exact(name):
    m = mesh_of(name)
    P = ElasticProblem(m, lame_D(m.dim), thickness=thick(name))
    rho = 2.3
    M = P.mass(rho)
    L = dims(name)
    for c in range(m.dim):
        row = np.asarray(M[c::m.dim, c::m.dim].sum(axis=1)).ravel()
        mom = (m.points[:, c] * row).sum()
        exp = rho * box_volume(name) * L[c] / 2
        assert abs(mom - exp) <= 1e-12 * exp


@pytest.mark.parametrize("name", QUAD2)
def test_fa2_mass_first_moment_curved(name):
    m = mesh_of(name, curved=True)
    P = ElasticProblem(m, lame_D(m.dim), thickness=thick(name))
    M = P.mass(1.0)
    L = dims(name)
    errs = []
    for c in range(m.dim):
        row = np.asarray(M[c::m.dim, c::m.dim].sum(axis=1)).ravel()
        errs.append(abs((m.points[:, c] * row).sum() - box_volume(name) * L[c] / 2) / (box_volume(name) * L[c] / 2))
    err = max(errs)
    print(f"\n  FA2 curved {name}: {err:.2e}")
    if name == "T10":
        assert err > 1e-12                           # degree(x detJ) = 5 > mass rule 4
    else:
        assert err <= 1e-12


@pytest.mark.parametrize("name", SOLID)
def test_fa3_heat_capacity_totals(name):
    m = mesh_of(name)
    H = HeatProblem(m, kappa=1.0, thickness=thick(name))
    V = cell_volumes(m) * thick(name)
    rc = 2.3e3 * 1.7                                  # rho * c_p, synthetic
    C = H.capacity(rc)
    assert abs(C.sum() - rc * box_volume(name)) <= 1e-12 * rc * box_volume(name)
    rce = 1.0e3 * (1.0 + 0.5 * (np.arange(m.n_cells) % 3))
    exp = float((rce * V).sum())
    assert abs(H.capacity(rce).sum() - exp) <= 1e-12 * exp
    Cl = H.capacity(rce, lumped=True)
    assert (Cl.diagonal()[H.active] > 0).all()
    assert abs(Cl.sum() - exp) <= 1e-12 * exp
    assert abs(C - C.T).max() <= 1e-14 * abs(C).max()


CASES_CURVED = [(n, False) for n in SOLID] + [(n, True) for n in SOLID if n in QUAD2]


# ---------------------------------------------------------------------------------------------- FA4
@pytest.mark.parametrize(("name", "curved"), CASES_CURVED)  # curved variants of the degree-2 elements only
def test_fa4_conduction_of_a_linear_field_gives_the_exact_flux(name, curved):
    m = mesh_of(name, curved=curved)
    dim = m.dim
    rng = np.random.default_rng(5)
    A = rng.normal(size=(dim, dim))
    kappa = A @ A.T + np.eye(dim)
    g = rng.normal(size=dim)
    T = 3.0 + m.points @ g
    H = HeatProblem(m, kappa=kappa, thickness=thick(name))
    L = dims(name)
    f = np.zeros(H.ndof)
    for side, ax, sg in SIDES[dim]:
        nvec = np.zeros(dim)
        nvec[ax] = sg
        q_in = float(kappa @ g @ nvec)                # derived in the prereg: q_in = kappa g . n_out
        fs = H.flux(m.facet_sets[side], q_in)
        area = np.prod([L[i] for i in range(dim) if i != ax]) * thick(name)
        assert abs(fs.sum() - q_in * area) <= 1e-12 * abs(q_in * area) + 1e-15
        f += fs
    KT = H.K @ T
    err = np.abs(KT - f).max() / np.abs(f).max()
    print(f"\n  FA4 {name} curved={curved}: {err:.1e}")
    assert err <= 1e-12


# ---------------------------------------------------------------------------------------------- FA5
def _body_case(name, curved, b_kind, degree=None):
    m = mesh_of(name, curved=curved)
    dim = m.dim
    P = ElasticProblem(m, lame_D(dim), thickness=thick(name))
    L = dims(name)
    th = thick(name)
    p = np.array([0.3, -0.2, 0.1][:dim])
    rng = np.random.default_rng(11)
    b0 = rng.normal(size=dim)
    Bm = rng.normal(size=(dim, dim)) if b_kind == "linear" else np.zeros((dim, dim))

    def b(X):
        return b0 + X @ Bm.T
    f = P.body_force(b if b_kind == "linear" else b0, degree=degree).reshape(-1, dim)
    force = f.sum(axis=0)
    mom = nodal_moment(m.points, f, p)
    lo, hi = np.zeros(dim), np.array(L)
    Fx = gauss_box(lo, hi, b) * th
    Mx = gauss_box(lo, hi, lambda X: cross_rows(X - p, b(X))) * th
    return rel(force, Fx), rel(mom, Mx)


@pytest.mark.parametrize("name", SOLID)
def test_fa5a_constant_body_force_distorted_exact(name):
    ef, em = _body_case(name, False, "const")
    print(f"\n  FA5a {name}: force {ef:.1e}, moment {em:.1e}")
    assert ef <= 1e-12 and em <= 1e-12


@pytest.mark.parametrize("name", QUAD2)
def test_fa5b_constant_body_force_curved(name):
    ef, em = _body_case(name, True, "const")
    ef8, em8 = _body_case(name, True, "const", degree=8)
    print(f"\n  FA5b {name}: force {ef:.1e}, moment default {em:.1e}, degree 8 {em8:.1e}")
    assert ef <= 1e-12 and ef8 <= 1e-12 and em8 <= 1e-12
    if name == "Q9":
        assert em <= 1e-12
    else:
        assert em > 1e-12


@pytest.mark.parametrize("name", SOLID)
def test_fa5c_linear_body_force_distorted(name):
    ef, em = _body_case(name, False, "linear")
    ef8, em8 = _body_case(name, False, "linear", degree=8)
    print(f"\n  FA5c {name}: force {ef:.1e}, moment default {em:.1e}, degree 8 {em8:.1e}")
    assert ef <= 1e-12 and ef8 <= 1e-12 and em8 <= 1e-12
    if name in ("T3", "T4", "H8"):
        assert em > 1e-12
    else:
        assert em <= 1e-12


@pytest.mark.parametrize("name", QUAD2)
def test_fa5d_linear_body_force_curved_high_rule(name):
    ef, em = _body_case(name, True, "linear", degree=10)
    print(f"\n  FA5d {name}: force {ef:.1e}, moment {em:.1e}")
    assert ef <= 1e-12 and em <= 1e-12


# ---------------------------------------------------------------------------------------------- FA6
def _face_poly(dim, seed=2):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(dim, 6))

    def t(X):
        y = X[:, 1]
        z = X[:, 2] if dim == 3 else np.zeros(len(X))
        mono = np.stack([np.ones(len(X)), y, y * y, z, z * z, y * z], axis=1)
        return mono @ c.T
    return t


def _face_integral(name, f):
    """Integral of f(X) over the face x = Lx (2-D: times the thickness)."""
    L = dims(name)
    dim = len(L)
    lo = np.array([L[0]] + [0.0] * (dim - 1))
    hi = np.array([L[0]] + list(L[1:]))
    # integrate over the free axes only: collapse the x axis by evaluating at x = Lx with the face measure
    x, w = np.polynomial.legendre.leggauss(4)
    axes = [0.5 * L[i] * (x + 1) for i in range(1, dim)]
    wts = [0.5 * L[i] * w for i in range(1, dim)]
    G = np.meshgrid(*axes, indexing="ij")
    W = np.meshgrid(*wts, indexing="ij")
    Y = np.stack([g.ravel() for g in G], axis=1)
    Wt = np.prod(np.stack([g.ravel() for g in W], axis=1), axis=1)
    X = np.column_stack([np.full(len(Y), L[0]), Y])
    assert lo[0] == hi[0]
    return np.einsum("n,nk->k", Wt, np.asarray(f(X), float)) * thick(name)


@pytest.mark.parametrize(("name", "curved"), CASES_CURVED)  # curved variants of the degree-2 elements only
def test_fa6a_traction_force_and_moment_on_a_face(name, curved):
    m = mesh_of(name, curved=curved)
    dim = m.dim
    P = ElasticProblem(m, lame_D(dim), thickness=thick(name))
    t = _face_poly(dim)
    p = np.array([0.3, -0.2, 0.1][:dim])
    f = P.traction(m.facet_sets["right"], t).reshape(-1, dim)
    Fx = _face_integral(name, t)
    Mx = _face_integral(name, lambda X: cross_rows(X - p, t(X)))
    ef, em = rel(f.sum(axis=0), Fx), rel(nodal_moment(m.points, f, p), Mx)
    print(f"\n  FA6a {name} curved={curved}: force {ef:.1e}, moment {em:.1e}")
    assert ef <= 1e-12 and em <= 1e-12


def _bulged(name, a=0.2):
    m = rectangle(1, 1, element=name)
    f = m.facet_sets["right"]
    fn = m.facet_nodes(f)[0]
    assert np.allclose(m.points[fn[2]], [1.0, 0.5])    # mid node of the right facet before the bulge
    m.points[fn[2], 0] += a
    return m, f, a


@pytest.mark.parametrize("name", ["T6", "Q9"])
def test_fa6b_pressure_on_a_curved_facet_default_rule(name):
    m, f, a = _bulged(name)
    th, pr = TH2D, 2.5
    P = ElasticProblem(m, lame_D(2), thickness=th)
    fl = P.pressure(f, pr).reshape(-1, 2)
    p0 = np.array([0.3, -0.2])
    # facet: x(s) = 1 + 4 a s (1 - s), y(s) = s, s in [0, 1]; n dS = (y', -x') ds, outward (+x)
    F_exp = -pr * th * np.array([1.0, 0.0])
    xs = lambda s: 1.0 + 4 * a * s * (1 - s)
    dx = lambda s: 4 * a * (1 - 2 * s)
    Mz = -pr * th * quad(lambda s: (xs(s) - p0[0]) * (-dx(s)) - (s - p0[1]) * 1.0, 0, 1, epsabs=1e-14,
                         epsrel=1e-14)[0]
    ef = rel(fl.sum(axis=0), F_exp)
    em = abs(nodal_moment(m.points, fl, p0)[0] - Mz) / abs(Mz)
    print(f"\n  FA6b {name}: force {ef:.1e}, moment {em:.1e}")
    assert ef <= 1e-12 and em <= 1e-12


@pytest.mark.parametrize("name", ["T6", "Q9"])
def test_fa6c_constant_traction_on_a_curved_facet_high_rule(name):
    m, f, a = _bulged(name)
    th = TH2D
    P = ElasticProblem(m, lame_D(2), thickness=th)
    tv = np.array([0.3, -0.8])
    fl = P.traction(f, tv, degree=40).reshape(-1, 2)
    p0 = np.array([0.3, -0.2])
    ds = lambda s: np.hypot(4 * a * (1 - 2 * s), 1.0)
    xs = lambda s: 1.0 + 4 * a * s * (1 - s)
    Ls = quad(ds, 0, 1, epsabs=1e-14, epsrel=1e-14)[0]
    cx = quad(lambda s: (xs(s) - p0[0]) * ds(s), 0, 1, epsabs=1e-14, epsrel=1e-14)[0]
    cy = quad(lambda s: (s - p0[1]) * ds(s), 0, 1, epsabs=1e-14, epsrel=1e-14)[0]
    F_exp = th * Ls * tv
    Mz = th * (cx * tv[1] - cy * tv[0])
    ef = rel(fl.sum(axis=0), F_exp)
    em = abs(nodal_moment(m.points, fl, p0)[0] - Mz) / abs(Mz)
    print(f"\n  FA6c {name}: force {ef:.1e}, moment {em:.1e}")
    assert ef <= 1e-12 and em <= 1e-12


# ---------------------------------------------------------------------------------------------- FA7
def _affine_case(name, support, seed=0):
    """Affine field on a distorted mesh: the support face holds u = A x + a, the other faces carry sigma n_out."""
    m = mesh_of(name)
    dim = m.dim
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) * 1e-2
    a = rng.normal(size=dim) * 1e-2
    D = lame_D(dim)
    sig = sigma_of(D, A)
    P = ElasticProblem(m, D, thickness=thick(name))
    f = np.zeros(P.ndof)
    n_of = {}
    for side, ax, sg in SIDES[dim]:
        nv = np.zeros(dim)
        nv[ax] = sg
        n_of[side] = nv
        if side != support:
            f += P.traction(m.facet_sets[side], sig @ nv)
    return m, P, A, a, sig, f, n_of


def _expected_reaction(name, support, sig, n_of):
    L = dims(name)
    dim = len(L)
    ax = [s for s in SIDES[dim] if s[0] == support][0]
    area = np.prod([L[i] for i in range(dim) if i != ax[1]]) * thick(name)
    tau = sig @ n_of[support]
    c = np.array([L[i] / 2 for i in range(dim)])
    c[ax[1]] = 0.0 if ax[2] < 0 else L[ax[1]]
    return area * tau, area * cross_rows(c[None], tau[None])[0], area * np.linalg.norm(c) * np.linalg.norm(tau)


@pytest.mark.parametrize("name", SOLID)
def test_fa7_nonzero_dirichlet_reaction_is_the_analytic_support_force(name):
    # two supports on one problem: left, then bottom (different support sets, the problem object is reused)
    errs = []
    for support in ("left", "bottom"):
        m, P, A, a, sig, f, n_of = _affine_case(name, support)
        dim = m.dim
        nodes = m.node_sets[support]
        val = m.points[nodes] @ A.T + a
        bc = Dirichlet([(m.dofs(nodes, [c]), val[:, c]) for c in range(dim)])
        s = P.solve(f, bc)
        used = m.used_nodes()
        ue = m.points[used] @ A.T + a
        uh = s.u[:, 0].reshape(-1, dim)[used]
        eu = np.abs(uh - ue).max() / np.abs(ue).max()
        R = s.reactions[:, 0].reshape(-1, dim)
        F_exp, M_exp, mscale = _expected_reaction(name, support, sig, n_of)
        ef = rel(R[nodes].sum(axis=0), F_exp)
        em = float(np.abs(nodal_moment(m.points[nodes], R[nodes], np.zeros(dim)) - M_exp).max() / mscale)
        errs.append((support, eu, ef, em))
        assert eu <= 1e-10 and ef <= 1e-10 and em <= 1e-10
        assert np.abs(np.delete(R, nodes, axis=0)).max() == 0.0     # reactions only at constrained DOFs
    print(f"\n  FA7 {name}: " + "; ".join(f"{s}: u {a:.1e}, F {b:.1e}, M {c:.1e}" for s, a, b, c in errs))


@pytest.mark.parametrize("name", ["T6", "H8"])
def test_fa7_two_load_cases_with_their_own_values_one_call(name):
    m = mesh_of(name)
    dim = m.dim
    D = lame_D(dim)
    P = ElasticProblem(m, D, thickness=thick(name))
    nodes = m.node_sets["left"]
    cases = []
    fcols, vals = [], []
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(dim, dim)) * 1e-2
        a = rng.normal(size=dim) * 1e-2
        sig = sigma_of(D, A)
        f = sum(P.traction(m.facet_sets[s], sig @ np.eye(dim)[ax] * sg)
                for s, ax, sg in SIDES[dim] if s != "left")
        fcols.append(f)
        vals.append(m.points[nodes] @ A.T + a)
        cases.append((A, a, sig))
    bc = Dirichlet([(m.dofs(nodes, [c]), np.column_stack([vals[0][:, c], vals[1][:, c]])) for c in range(dim)],
                   n_cases=2)
    s = P.solve(np.column_stack(fcols), bc)
    assert P.n_factorizations == 1
    for k, (A, a, sig) in enumerate(cases):
        R = s.reactions[:, k].reshape(-1, dim)
        nv = -np.eye(dim)[0]
        L = dims(name)
        area = np.prod(L[1:]) * thick(name)
        assert rel(R[nodes].sum(axis=0), area * sig @ nv) <= 1e-10


# ---------------------------------------------------------------------------------------------- CT helpers
def _t6_problem():
    m = mesh_of("T6", n=(3, 2))
    P = ElasticProblem(m, lame_D(2), thickness=TH2D)
    return m, P


@pytest.mark.parametrize("name", ["T6", "H8"])
def test_ct1_dirichlet_rows_reproduce_the_elimination_path_bitwise_regression(name):
    """REGRESSION check: the reference is the production ``solve`` itself (not independent)."""
    m = mesh_of(name, n=(3, 2, 2))
    dim = m.dim
    P = ElasticProblem(m, lame_D(dim), thickness=thick(name))
    f = np.column_stack([P.body_force(np.arange(1, dim + 1) * 0.3), P.traction(m.facet_sets["right"], np.ones(dim))])
    left, bottom = m.node_sets["left"], m.node_sets["bottom"]
    rng = np.random.default_rng(4)
    pairs = [(m.dofs(left), rng.normal(size=(len(left) * dim, 2)) * 1e-2),
             (m.dofs(np.setdiff1d(bottom, left), [1]), rng.normal(size=(len(np.setdiff1d(bottom, left)), 2)) * 1e-2)]
    bc = Dirichlet(pairs, n_cases=2)
    ref = P.solve(f, bc)
    sol = solve_constrained(P, f, LinearConstraints.from_dirichlet(bc, P.ndof))
    c = bc.dofs
    assert np.array_equal(sol.u, ref.u)
    assert np.array_equal(sol.reactions[c], ref.reactions[c])
    assert sol.reduction.pure_selection and len(sol.reduction.masters) == len(P.active) - len(c)
    # the general triple product agrees to rounding
    red = sol.reduction
    K = P.K
    Kg = red.reduce_matrix(K, general=True)
    Ks = red.reduce_matrix(K)
    assert abs(Kg - Ks).max() <= 1e-14 * abs(Ks).max()
    assert np.abs(red.reduce_load(K, f, general=True) - red.reduce_load(K, f)).max() <= 1e-14 * np.abs(f).max()


def _kkt_problem():
    m, P = _t6_problem()
    n = P.ndof
    right = m.node_sets["right"]
    a, b = right[0], right[-1]
    interior = np.setdiff1d(m.used_nodes(), m.boundary_nodes())
    c, d, e = interior[:3]
    cons = (LinearConstraints.fix(n, m.dofs(m.node_sets["left"]), 0.0)
            + LinearConstraints.tie(n, [2 * a], [2 * b], 0.01)
            + LinearConstraints.mpc(n, [2 * c + 1, 2 * d + 1, 2 * e + 1], [1.0, 0.5, -1.5], 0.002))
    f = P.body_force([0.2, -0.5]) + P.traction(m.facet_sets["top"], [0.1, -0.3])
    return m, P, cons, f


def test_ct2_mpc_holds_and_matches_a_dense_kkt_solve():
    m, P, cons, f = _kkt_problem()
    sol = solve_constrained(P, f, cons)
    u = sol.u[:, 0]
    C = cons.C.toarray()
    d = cons.d[:, 0]
    assert np.abs(C @ u - d).max() <= 1e-12 * np.abs(u).max()                # constraints hold in the solution
    K = P.K.toarray()
    mrows = C.shape[0]
    Akkt = np.block([[K, C.T], [C, np.zeros((mrows, mrows))]])
    X = np.linalg.solve(Akkt, np.concatenate([f, d]))
    eu = np.abs(u - X[:P.ndof]).max() / np.abs(X[:P.ndof]).max()
    mu = X[P.ndof:]
    lam = sol.multipliers[:, 0]
    # kept rows are the independent rows: all of them here
    assert sol.reduction.rows == list(range(mrows)) and not sol.reduction.dropped
    el = np.abs(lam + mu).max() / np.abs(mu).max()
    print(f"\n  CT2: u {eu:.1e}, lambda + mu {el:.1e}, multiplier residual {sol.multiplier_residual:.1e}")
    assert eu <= 1e-9 and el <= 1e-8 and sol.multiplier_residual <= 1e-9


def test_ct3a_periodic_skew_cell_reproduces_the_homogeneous_solution_2d():
    s_skew = 0.35
    H = np.array([[0.004, -0.0025], [0.0035, -0.002]])
    D = lame_D(2)
    sig = sigma_of(D, H)
    for name in ("Q4", "T6", "Q9", "T3"):
        m0 = mesh_of(name, n=(4, 3))
        Aff = np.array([[1.0, s_skew], [0.0, 1.0]])
        m = replace(m0, points=m0.points @ Aff.T)
        L = dims(name)
        a1, a2 = np.array([L[0], 0.0]), Aff @ np.array([0.0, L[1]])
        P = ElasticProblem(m, D, thickness=TH2D)
        n = P.ndof
        pin = int(np.argmin(np.linalg.norm(m.points, axis=1)))
        cons = (periodic(m, m.node_sets["left"], m.node_sets["right"], a1, macro_jump(H, a1))
                + periodic(m, m.node_sets["bottom"], m.node_sets["top"], a2, macro_jump(H, a2))
                + LinearConstraints.fix(n, m.dofs([pin]), m.points[pin] @ H.T))
        sol = solve_constrained(P, np.zeros(n), cons, redundant="drop")
        assert len(sol.reduction.dropped) >= 1                              # the corner rows are dependent
        ue = m.points @ H.T
        eu = np.abs(sol.u[:, 0].reshape(-1, 2) - ue).max() / np.abs(ue).max()
        r = sol.reactions[:, 0].reshape(-1, 2)
        V = L[0] * L[1] * TH2D
        sbar = (r.T @ m.points) / V
        es = np.abs(sbar - sig).max() / np.abs(sig).max()
        # the lattice builder gives the same solution
        cons2 = periodic_lattice(m, [a1, a2], H) + LinearConstraints.fix(n, m.dofs([pin]), m.points[pin] @ H.T)
        sol2 = solve_constrained(P, np.zeros(n), cons2, redundant="drop")
        eu2 = np.abs(sol2.u[:, 0].reshape(-1, 2) - ue).max() / np.abs(ue).max()
        print(f"\n  CT3a {name}: u {eu:.1e}, average stress {es:.1e}, lattice builder {eu2:.1e}")
        assert eu <= 1e-10 and es <= 1e-10 and eu2 <= 1e-10


@pytest.mark.parametrize("name", ["H8", "T4"])
def test_ct3b_periodic_skew_cell_3d(name):
    m0 = mesh_of(name, n=(2, 2, 2), amp=0.15)
    Aff = np.array([[1.0, 0.2, 0.1], [0.05, 1.0, 0.15], [0.1, 0.05, 1.0]])
    m = replace(m0, points=m0.points @ Aff.T)
    L = np.array(L3D)
    lattice = [Aff @ (np.eye(3)[i] * L[i]) for i in range(3)]
    H = np.array([[0.003, -0.001, 0.0015], [0.002, -0.002, 0.0005], [-0.001, 0.0025, 0.001]])
    D = lame_D(3)
    sig = sigma_of(D, H)
    P = ElasticProblem(m, D)
    pin = int(np.argmin(np.linalg.norm(m.points, axis=1)))
    cons = periodic_lattice(m, lattice, H) + LinearConstraints.fix(P.ndof, m.dofs([pin]), m.points[pin] @ H.T)
    sol = solve_constrained(P, np.zeros(P.ndof), cons, redundant="drop")
    ue = m.points @ H.T
    eu = np.abs(sol.u[:, 0].reshape(-1, 3) - ue).max() / np.abs(ue).max()
    V = abs(np.linalg.det(Aff)) * np.prod(L)
    sbar = (sol.reactions[:, 0].reshape(-1, 3).T @ m.points) / V
    es = np.abs(sbar - sig).max() / np.abs(sig).max()
    print(f"\n  CT3b {name}: u {eu:.1e}, average stress {es:.1e}")
    assert eu <= 1e-10 and es <= 1e-10


# ---------------------------------------------------------------------------------------------- CT4, CT5
def _n_small():
    m = rectangle(2, 2, element="T3")
    return m, ElasticProblem(m, lame_D(2))


def test_ct4_contradictory_and_duplicate_constraints():
    m, P = _n_small()
    n = P.ndof
    with pytest.raises(ContradictoryConstraints, match="row 1"):
        (LinearConstraints.fix(n, [4], 1.0) + LinearConstraints.fix(n, [4], 2.0)).eliminate(P.active)
    with pytest.raises(ContradictoryConstraints):
        (LinearConstraints.tie(n, [2], [6], 1.0) + LinearConstraints.tie(n, [6], [2], 1.0)).eliminate(P.active)
    dup = LinearConstraints.tie(n, [2], [6], 1.0) + LinearConstraints.tie(n, [6], [2], -1.0)
    with pytest.raises(RedundantConstraints, match="row 1"):
        dup.eliminate(P.active)
    red = dup.eliminate(P.active, redundant="drop")
    assert red.dropped == [1] and red.rows == [0]
    # same solution as the single row
    base = (LinearConstraints.fix(n, m.dofs(m.node_sets["left"]), 0.0)
            + LinearConstraints.tie(n, [2 * m.node_sets["right"][0]], [2 * m.node_sets["right"][1]], 0.01))
    f = P.traction(m.facet_sets["right"], [1.0, 0.0])
    s1 = solve_constrained(P, f, base)
    twice = base + LinearConstraints.tie(n, [2 * m.node_sets["right"][1]], [2 * m.node_sets["right"][0]], -0.01)
    with pytest.raises(RedundantConstraints):
        solve_constrained(P, f, twice)
    s2 = solve_constrained(P, f, twice, redundant="drop")
    assert np.array_equal(s1.u, s2.u)


def test_ct5_cyclic_chain_refused_and_acyclic_chain_accepted():
    m, P = _n_small()
    n = P.ndof
    a, b, c, d = 2, 6, 8, 10
    cyc = LinearConstraints.tie(n, [a, b, c], [b, c, a], 0.0)
    with pytest.raises(RedundantConstraints, match="rank deficient"):
        cyc.eliminate(P.active)
    bad = LinearConstraints.tie(n, [a, b, c], [b, c, a], 1.0)
    with pytest.raises(ContradictoryConstraints):
        bad.eliminate(P.active)
    chain = LinearConstraints.tie(n, [a, b, c], [b, c, d], [0.1, 0.2, 0.3])
    red = chain.eliminate(P.active)
    assert len(red.slaves) == 3 and len(red.masters) == len(P.active) - 3
    base = LinearConstraints.fix(n, m.dofs(m.node_sets["left"]), 0.0)
    sol = solve_constrained(P, P.traction(m.facet_sets["right"], [1.0, 0.5]), base + chain)
    u = sol.u[:, 0]
    assert abs(u[a] - u[b] - 0.1) + abs(u[b] - u[c] - 0.2) + abs(u[c] - u[d] - 0.3) <= 1e-12 * np.abs(u).max()


# ---------------------------------------------------------------------------------------------- CT6, CT7
def test_ct6_multiplier_forces_balance_the_applied_loads():
    m, P, cons, f = _kkt_problem()
    # add a periodic pair of the top/bottom sides is not needed: Dirichlet + MPC + tie is the mixed set
    sol = solve_constrained(P, f, cons)
    Fc = sol.constraint_forces[:, 0].reshape(-1, 2)                     # C^T lambda, rebuilt from the multipliers
    Fa = f.reshape(-1, 2)
    scale = np.abs(Fa).sum()
    p = np.array([0.4, 0.3])
    bal = np.abs(Fc.sum(axis=0) + Fa.sum(axis=0)).max() / scale
    mom = abs((nodal_moment(m.points, Fc, p) + nodal_moment(m.points, Fa, p))[0]) / (scale * np.abs(m.points).max())
    res = np.abs(sol.constraint_forces - sol.reactions).max() / np.abs(sol.reactions).max()
    print(f"\n  CT6: force balance {bal:.1e}, moment balance {mom:.1e}, C^T lambda = K u - f {res:.1e}")
    assert bal <= 1e-10 and mom <= 1e-10 and res <= 1e-9


def test_ct6_periodic_cell_with_loads_balances():
    m0 = mesh_of("T6", n=(3, 3))
    Aff = np.array([[1.0, 0.3], [0.0, 1.0]])
    m = replace(m0, points=m0.points @ Aff.T)
    P = ElasticProblem(m, lame_D(2), thickness=TH2D)
    L = L2D
    a1, a2 = np.array([L[0], 0.0]), Aff @ np.array([0.0, L[1]])
    f = P.body_force([0.2, -0.1])
    n = P.ndof
    cons = (periodic(m, m.node_sets["left"], m.node_sets["right"], a1) + periodic(m, m.node_sets["bottom"],
            m.node_sets["top"], a2) + LinearConstraints.fix(n, m.dofs([0]), 0.0))
    sol = solve_constrained(P, f, cons, redundant="drop")
    Fc = sol.constraint_forces[:, 0].reshape(-1, 2)
    Fa = f.reshape(-1, 2)
    bal = np.abs(Fc.sum(axis=0) + Fa.sum(axis=0)).max() / np.abs(Fa).sum()
    mom = abs((nodal_moment(m.points, Fc, np.zeros(2)) + nodal_moment(m.points, Fa, np.zeros(2)))[0]) / \
        (np.abs(Fa).sum() * np.abs(m.points).max())
    print(f"\n  CT6 periodic: force {bal:.1e}, moment {mom:.1e}")
    assert bal <= 1e-10 and mom <= 1e-10


def test_ct7_lagrange_and_penalty_against_elimination_with_conditioning():
    m, P, cons, f = _kkt_problem()
    el = solve_constrained(P, f, cons)
    lag = solve_lagrange(P, f, cons)
    scale = np.abs(el.u).max()
    eu = np.abs(lag.u - el.u).max() / scale
    em = np.abs(lag.multipliers + el.multipliers).max() / np.abs(el.multipliers).max()
    Kr = el.reduction.reduce_matrix(P.K)
    cond_el = condition_1norm(Kr, Factorization(Kr, "spd"))
    errs, conds = {}, {}
    for alpha in (1e3, 1e5):
        pen = solve_penalty(P, f, cons, alpha=alpha)
        errs[alpha] = np.abs(pen.u - el.u).max() / scale
        conds[alpha] = pen.cond1
    ratio_e = errs[1e3] / errs[1e5]
    ratio_c = conds[1e5] / conds[1e3]
    print(f"\n  CT7: lagrange u {eu:.1e}, mu+lambda {em:.1e}; cond1: elimination {cond_el:.2e}, lagrange "
          f"{lag.cond1:.2e}, penalty(1e3) {conds[1e3]:.2e}, penalty(1e5) {conds[1e5]:.2e}; penalty error "
          f"{errs[1e3]:.2e} -> {errs[1e5]:.2e} (ratio {ratio_e:.1f}), cond ratio {ratio_c:.1f}")
    assert eu <= 1e-9 and em <= 1e-8
    assert 30 <= ratio_e <= 300
    assert ratio_c >= 10


def test_ct8_heat_problem_with_a_tie():
    m = rectangle(4, 3, lx=1.2, ly=0.9, element="T3")
    H = HeatProblem(m, kappa=1.7, thickness=0.7)
    n = H.ndof
    right = m.node_sets["right"]
    cons = LinearConstraints.fix(n, m.node_sets["left"], 1.0) + LinearConstraints.tie(n, [right[0]], [right[-1]], 0.5)
    f = H.flux(m.facet_sets["right"], 2.0)
    sol = solve_constrained(H, f, cons)
    T = sol.u[:, 0]
    assert abs(T[right[0]] - T[right[-1]] - 0.5) <= 1e-12 * np.abs(T).max()
    bal = abs(sol.constraint_forces.sum() + f.sum()) / np.abs(f).sum()
    assert bal <= 1e-10


# ---------------------------------------------------------------------------------------------- negative controls
def _periodic_setup(H, pin=True, jump_sign=1.0, zero_jump=False):
    m0 = mesh_of("Q4", n=(4, 3))
    Aff = np.array([[1.0, 0.35], [0.0, 1.0]])
    m = replace(m0, points=m0.points @ Aff.T)
    a1, a2 = np.array([L2D[0], 0.0]), Aff @ np.array([0.0, L2D[1]])
    P = ElasticProblem(m, lame_D(2), thickness=TH2D)
    j1 = 0.0 if zero_jump else jump_sign * macro_jump(H, a1)
    j2 = 0.0 if zero_jump else jump_sign * macro_jump(H, a2)
    cons = (periodic(m, m.node_sets["left"], m.node_sets["right"], a1, j1)
            + periodic(m, m.node_sets["bottom"], m.node_sets["top"], a2, j2))
    if pin:
        cons = cons + LinearConstraints.fix(P.ndof, m.dofs([0]), 0.0)
    return m, P, cons


H_NEG = np.array([[0.004, -0.0025], [0.0035, -0.002]])


@pytest.mark.parametrize("variant", ["wrong_sign", "no_jump"])
def test_cn1_wrong_periodic_jump_is_detected(variant):
    m, P, cons = _periodic_setup(H_NEG, jump_sign=-1.0 if variant == "wrong_sign" else 1.0,
                                 zero_jump=variant == "no_jump")
    sol = solve_constrained(P, np.zeros(P.ndof), cons, redundant="drop")
    ue = m.points @ H_NEG.T
    err = np.abs(sol.u[:, 0].reshape(-1, 2) - ue).max() / np.abs(ue).max()
    print(f"\n  CN1 {variant}: {err:.2f}")
    assert err > 0.1


def test_cn2_periodic_cell_without_a_pin_is_a_mechanism():
    m, P, cons = _periodic_setup(H_NEG, pin=False)
    with pytest.raises(MechanismError, match=r"2 free rigid motion"):
        solve_constrained(P, np.zeros(P.ndof), cons, redundant="drop")


def test_cn3_missing_k_u0_term_changes_the_nonzero_dirichlet_solution():
    m, P = _t6_problem()
    nodes = m.node_sets["left"]
    n = P.ndof
    cons = LinearConstraints.fix(n, m.dofs(nodes), np.tile([0.02, -0.01], len(nodes)))
    f = P.traction(m.facet_sets["right"], [1.0, 0.0])
    good = solve_constrained(P, f, cons)
    red = good.reduction
    K = P.K
    z_bad = Factorization(red.reduce_matrix(K), "spd").solve(f[red.masters])        # forgot - K u0
    u_bad = red.expand(z_bad.ravel())
    err = np.abs(u_bad - good.u[:, 0]).max() / np.abs(good.u).max()
    print(f"\n  CN3: {err:.2f}")
    assert err > 1e-3


def test_cn4_invalid_inputs_raise():
    m = rectangle(1, 1, element="T3")
    pts = np.vstack([m.points, [[5.0, 5.0]]])
    mu = Mesh(pts, m.cells, "T3")
    P = ElasticProblem(mu, lame_D(2))
    with pytest.raises(ConstraintError, match="belong to no cell"):
        solve_constrained(P, np.zeros(P.ndof), LinearConstraints.fix(P.ndof, [2 * 4], 0.0))
    m2 = mesh_of("Q4", n=(3, 3))
    with pytest.raises(ConstraintError, match="no partner"):
        periodic(m2, m2.node_sets["left"], m2.node_sets["right"], [0.7, 0.0])
    with pytest.raises(ConstraintError, match="ambiguous"):
        periodic(m2, np.array([m2.node_sets["left"][0]] * 2), np.array(m2.node_sets["right"][:2]), [L2D[0], 0.0])
    with pytest.raises(ConstraintError, match="different size"):
        periodic(m2, m2.node_sets["left"], m2.node_sets["right"][:2], [L2D[0], 0.0])
    with pytest.raises(ValueError, match="right-hand-side"):
        LinearConstraints.fix(8, [1, 2], [1.0, 2.0, 3.0])
    P2 = ElasticProblem(m2, lame_D(2))
    two = LinearConstraints.fix(P2.ndof, m2.dofs(m2.node_sets["left"]), np.zeros((len(m2.dofs(m2.node_sets["left"])), 2)))
    with pytest.raises(ValueError, match="right-hand-side column"):
        solve_constrained(P2, np.zeros((P2.ndof, 3)), two)
    with pytest.raises(ValueError, match="out of range"):
        LinearConstraints.fix(8, [9], 0.0)
    with pytest.raises(ConstraintError, match="lattice vector"):
        periodic_lattice(m2, [[5.0, 5.0]])
    assert sp.issparse(LinearConstraints.empty(4).C)

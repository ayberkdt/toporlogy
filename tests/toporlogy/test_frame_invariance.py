"""Frame invariance and objectivity: the checks I1 to I5 pre-registered in docs/olcumler/dogrulama_turu.md (section I,
before the first run). The thresholds are those of the table there and are not tuned.

I1  2-D, rotate the problem by +90 deg about z (grid (nx, ny) -> (ny, nx), loads, supports, element moduli, fibre
    angle theta -> theta + 90): compliance and Tsai-Wu field (<= 1e-10), first buckling mu_1 (<= 1e-8); Q4 and Q4IM.
I2  2-D mirror x -> L - x with theta -> -theta: compliance (<= 1e-10); Q4 and Q4IM.
I3  3-D, rotate by 90 deg about z with per-element material axes R_e -> Rz R_e: compliance (<= 1e-10); H8 and H8IM.
I4  Objectivity of the geometrically non-linear model: a large rigid rotation superimposed on a deformed state leaves
    the strain energy (<= 1e-12 relative) and rotates the nodal internal force (<= 1e-11 relative to max|f|).
I5  Pure rigid motion from u = 0 (1 rad rotation + translation): internal force and energy vanish
    (|f| <= 1e-12 |K0|_1 |u|_2, W <= 1e-12 * 1/2 u^T K0 u).

The point, vector and element maps of a transformed problem are built by ``_Map`` from the coordinates and checked
against them (and against hand-computed indices on a tiny grid) before any physics is compared. I4 and I5 use no
energy interpolation: its linear part (1 - gamma^2) K0 u is not objective by construction (void elements are linear).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from toporlogy.core.domain import Domain
from toporlogy.core.material import (Orthotropic, OrthotropicStrength, rotations_about_z, rotations_from_axes,
                                     unit_elasticity_field)
from toporlogy.fem.buckling import LinearBuckling
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.nonlinear import NonlinearElasticity
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.strength import make_criterion
from toporlogy.optimization.stress import FailureIndex

MAT = Orthotropic(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)
STRENGTH = OrthotropicStrength(Xt=15.0, Xc=12.0, Yt=0.5, Yc=2.0, S12=0.7)
H = 0.5
THICK = 0.8

ROT90_2D = np.array([[0.0, -1.0], [1.0, 0.0]])
ROT90_3D = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MIRROR_2D = np.diag([-1.0, 1.0])


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


# ======================================================================================================== the maps
class _Map:
    """The rigid map y = A x + t of a domain onto ``dst`` (x on a grid of ``src``): node, element and DOF maps built
    from the coordinates, and asserted against them. ``A`` is a signed permutation (90 deg rotation, mirror)."""

    def __init__(self, src: Domain, dst: Domain, A: np.ndarray, t: np.ndarray) -> None:
        self.src, self.dst, self.A, self.t = src, dst, np.asarray(A, float), np.asarray(t, float)
        dim = src.dim
        X = src.node_coords()
        Y = X @ self.A.T + self.t
        ijk = np.rint((Y - dst.origin) / dst.h).astype(np.int64)
        assert np.allclose(dst.origin + dst.h * ijk, Y), "a mapped node is off the destination grid"
        assert (ijk >= 0).all() and (ijk <= np.array(dst.node_shape) - 1).all()
        self.nmap = dst.node_number(ijk)
        assert len(np.unique(self.nmap)) == src.nn
        assert np.allclose(dst.node_coords()[self.nmap], Y), "node map: coordinates do not match"
        C = src.element_centers()
        Yc = C @ self.A.T + self.t
        eijk = np.floor((Yc - dst.origin) / dst.h).astype(np.int64)
        self.emap = np.ravel_multi_index(tuple(eijk.T), dst.shape)
        assert len(np.unique(self.emap)) == src.ne
        assert np.allclose(dst.element_centers()[self.emap], Yc), "element map: centres do not match"
        self.comp = np.argmax(np.abs(self.A), axis=0)                  # component c of the source -> comp[c]
        assert np.allclose(np.abs(self.A[self.comp, np.arange(dim)]), 1.0)
        # a nodal vector field transforms as v' = A v: the position field maps to Y - t
        assert np.allclose(self.vec(X.ravel()).reshape(dst.nn, dim)[self.nmap], Y - self.t), \
            "vector map: positions do not match"

    def vec(self, v: np.ndarray) -> np.ndarray:
        """(ndof,) nodal vector (load, displacement) of the source in the destination numbering, rotated by A."""
        out = np.zeros((self.dst.nn, self.src.dim))
        out[self.nmap] = np.asarray(v, float).reshape(self.src.nn, self.src.dim) @ self.A.T
        return out.ravel()

    def dofs(self, fixed: np.ndarray) -> np.ndarray:
        dim = self.src.dim
        fixed = np.asarray(fixed, np.int64)
        return np.sort(dim * self.nmap[fixed // dim] + self.comp[fixed % dim])

    def per_element(self, a: np.ndarray) -> np.ndarray:
        """Element array of the source (axis 0 = element) placed at the mapped element numbers."""
        a = np.asarray(a)
        out = np.empty_like(a)
        out[self.emap] = a
        return out


def _rotated(d: Domain, A: np.ndarray) -> _Map:
    nx, ny = d.shape[0], d.shape[1]
    dst = Domain((ny, nx) + d.shape[2:], h=d.h, thickness=d.thickness)
    t = np.zeros(d.dim)
    t[0] = ny * d.h                                                   # (x, y) -> (H - y, x), H = ny h
    return _Map(d, dst, A, t)


def _mirrored(d: Domain) -> _Map:
    dst = Domain(d.shape, h=d.h, thickness=d.thickness)
    t = np.zeros(d.dim)
    t[0] = d.shape[0] * d.h                                           # x -> L - x
    return _Map(d, dst, MIRROR_2D, t)


def test_maps_on_a_tiny_grid():
    """Hand-computed indices of a (3, 2) grid (h = 1), rotation by +90 deg and mirror; then a 3-D check."""
    d = Domain((3, 2), h=1.0)
    m = _rotated(d, ROT90_2D)                       # destination (2, 3): node n = i' * 4 + j', element e = i' * 3 + j'
    assert m.dst.shape == (2, 3)
    assert m.nmap[0] == 2 * 4 + 0                   # node (0, 0) -> (H - 0, 0) = (2, 0)
    assert m.nmap[d.node_number(np.array([3, 0]))] == 2 * 4 + 3        # (3, 0) -> (2, 3)
    assert m.nmap[d.node_number(np.array([0, 2]))] == 0                # (0, 2) -> (0, 0)
    assert m.emap[2 * 2 + 0] == 1 * 3 + 2           # element (2, 0) -> (ny - 1 - 0, 2) = (1, 2)
    assert m.emap[0 * 2 + 1] == 0 * 3 + 0           # element (0, 1) -> (ny - 1 - 1, 0) = (0, 0)
    f = np.zeros(d.ndof)
    f[2 * 5 + 0] = 1.0                              # node 5 = (1, 2), force +x -> +y on node (H - 2, 1) = (0, 1)
    g = m.vec(f)
    assert g[2 * m.dst.node_number(np.array([0, 1])) + 1] == 1.0 and np.count_nonzero(g) == 1
    assert list(m.dofs(np.array([2 * 5 + 0]))) == [2 * int(m.dst.node_number(np.array([0, 1]))) + 1]
    mm = _mirrored(d)
    assert mm.nmap[0] == 3 * 3 + 0                  # mirror, node (0, 0) -> (3, 0): n = 3 * (ny + 1) + 0
    assert mm.emap[0] == 2 * 2 + 0                  # element (0, 0) -> (2, 0)
    d3 = Domain((2, 3, 1), h=1.0)
    m3 = _rotated(d3, ROT90_3D)                     # destination (3, 2, 1); node (i, j, k) -> (ny - j, i, k)
    n = int(d3.node_number(np.array([1, 2, 1])))
    assert m3.nmap[n] == m3.dst.node_number(np.array([1, 1, 1]))
    assert m3.emap[int(np.ravel_multi_index((1, 2, 0), d3.shape))] == int(np.ravel_multi_index((0, 1, 0), m3.dst.shape))


# ======================================================================================================== 2-D problem
def _clamped_2d(d: Domain) -> np.ndarray:
    X = d.node_coords()
    left = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    roller = d.dofs(np.flatnonzero((np.abs(X[:, 0] - X[:, 0].max()) < 1e-9) & (X[:, 1] < 1e-9)), 1)
    return np.union1d(left, roller)


def _problem_2d(nx: int, ny: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    d = Domain((nx, ny), h=H, thickness=THICK)
    fixed = _clamped_2d(d)
    X = d.node_coords()
    F = np.zeros(d.ndof)
    tip = np.flatnonzero(np.abs(X[:, 0] - X[:, 0].max()) < 1e-9)
    F[2 * tip + 1] = -1.0 / len(tip)                                  # bending: compression along the lower fibre
    for n in rng.choice(d.nn, 6, replace=False):
        F[2 * n:2 * n + 2] += 0.3 * rng.standard_normal(2)
    theta = rng.uniform(-90.0, 90.0, d.ne)
    E = 40.0 * (0.3 + 0.7 * rng.random(d.ne))
    return d, fixed, F, theta, E


def _solve_2d(d, fixed, F, theta, E, element):
    R = rotations_about_z(theta)
    D = unit_elasticity_field(MAT, 2, "stress", R)
    m = Elasticity(d, fixed, element=element, elasticity=D)
    U = m.solve_numpy(E, F)[:, 0]
    return m, U, R


def _tsai_wu(m: Elasticity, U: np.ndarray, R: np.ndarray) -> np.ndarray:
    d = m.domain
    ctx = EvalContext(domain=d, xphys=torch.ones(d.shape, dtype=torch.float64),
                      E=torch.full(d.shape, 40.0, dtype=torch.float64), U=torch.as_tensor(U[:, None]))
    fi = FailureIndex(d, make_criterion("tsai_wu", STRENGTH, R), D=m.D, E0=40.0)
    return fi.solid_ratio(ctx)[:, 0].numpy()


def _mu1(m: Elasticity, E: np.ndarray, U: np.ndarray) -> float:
    mu, _ = LinearBuckling(m).modes(E, E, U, k=3)
    return float(mu[0])


@pytest.mark.parametrize("element", ["Q4", "Q4IM"])
def test_i1_rotation_by_90_degrees_2d(element):
    nx, ny = 7, 4
    d, fixed, F, theta, E = _problem_2d(nx, ny)
    m1, U1, R1 = _solve_2d(d, fixed, F, theta, E, element)
    mp = _rotated(d, ROT90_2D)
    d2 = mp.dst
    assert d2.shape == (ny, nx) and d2.thickness == d.thickness and d2.h == d.h
    theta2 = mp.per_element(theta + 90.0)
    m2, U2, R2 = _solve_2d(d2, mp.dofs(fixed), mp.vec(F), theta2, mp.per_element(E), element)
    c1, c2 = float(F @ U1), float(mp.vec(F) @ U2)
    tw1, tw2 = _tsai_wu(m1, U1, R1), _tsai_wu(m2, U2, R2)
    mu1, mu2 = _mu1(m1, E, U1), _mu1(m2, mp.per_element(E), U2)
    e_c, e_tw, e_mu = abs(c2 - c1) / abs(c1), float(np.max(np.abs(tw2[mp.emap] - tw1)) / np.max(np.abs(tw1))), \
        abs(mu2 - mu1) / abs(mu1)
    print(f"\n  I1 {element}: compliance {e_c:.2e} (c = {c1:.6g}), Tsai-Wu field {e_tw:.2e} (max {tw1.max():.4g}), "
          f"mu_1 {e_mu:.2e} (mu_1 = {mu1:.6g})")
    assert mu1 > 0.0
    assert e_c <= 1e-10
    assert e_tw <= 1e-10
    assert e_mu <= 1e-8


@pytest.mark.parametrize("element", ["Q4", "Q4IM"])
def test_i2_mirror_2d(element):
    d, fixed, F, theta, E = _problem_2d(7, 4, seed=11)
    m1, U1, _ = _solve_2d(d, fixed, F, theta, E, element)
    mp = _mirrored(d)
    m2, U2, _ = _solve_2d(mp.dst, mp.dofs(fixed), mp.vec(F), mp.per_element(-theta), mp.per_element(E), element)
    c1, c2 = float(F @ U1), float(mp.vec(F) @ U2)
    err = abs(c2 - c1) / abs(c1)
    print(f"\n  I2 {element}: compliance {err:.2e} (c = {c1:.6g})")
    assert err <= 1e-10


# ======================================================================================================== 3-D problem
@pytest.mark.parametrize("element", ["H8", "H8IM"])
def test_i3_rotation_by_90_degrees_3d(element):
    nx, ny, nz = 4, 3, 2
    rng = np.random.default_rng(5)
    d = Domain((nx, ny, nz), h=H)
    X = d.node_coords()
    fixed = np.union1d(d.dofs(np.flatnonzero(X[:, 0] < 1e-9)),
                       d.dofs(np.flatnonzero((np.abs(X[:, 0] - X[:, 0].max()) < 1e-9) & (X[:, 1] < 1e-9)
                                             & (X[:, 2] < 1e-9)), 2))
    F = np.zeros(d.ndof)
    tip = np.flatnonzero(np.abs(X[:, 0] - X[:, 0].max()) < 1e-9)
    F[3 * tip + 1] = -1.0 / len(tip)
    for n in rng.choice(d.nn, 6, replace=False):
        F[3 * n:3 * n + 3] += 0.3 * rng.standard_normal(3)
    R = rotations_from_axes(rng.standard_normal((d.ne, 3)), rng.standard_normal((d.ne, 3)))
    E = 40.0 * (0.3 + 0.7 * rng.random(d.ne))

    def solve(dom, fx, f, Rm, Em):
        m = Elasticity(dom, fx, element=element, elasticity=unit_elasticity_field(MAT, 3, "stress", Rm))
        return float(f @ m.solve_numpy(Em, f)[:, 0])

    mp = _rotated(d, ROT90_3D)
    assert mp.dst.shape == (ny, nx, nz)
    c1 = solve(d, fixed, F, R, E)
    c2 = solve(mp.dst, mp.dofs(fixed), mp.vec(F), mp.per_element(ROT90_3D @ R), mp.per_element(E))
    err = abs(c2 - c1) / abs(c1)
    print(f"\n  I3 {element}: compliance {err:.2e} (c = {c1:.6g})")
    assert err <= 1e-10


# ======================================================================================================== non-linear
def _unit_axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation(dim: int, angle: float, axis: np.ndarray | None = None) -> np.ndarray:
    if dim == 2:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])
    return _unit_axis_rotation(np.array([0.0, 0.0, 1.0]) if axis is None else axis, angle)


# (id, dim, material, plane, orthotropic field)
NL_CASES = [
    ("svk_2d_stress", 2, "svk", "stress", False),
    ("svk_2d_strain", 2, "svk", "strain", False),
    ("svk_field_2d_stress", 2, "svk", "stress", True),
    ("svk_field_2d_strain", 2, "svk", "strain", True),
    ("neo_2d_stress", 2, "neo_hooke", "stress", False),
    ("neo_2d_strain", 2, "neo_hooke", "strain", False),
    ("svk_3d", 3, "svk", "stress", False),
    ("svk_field_3d", 3, "svk", "stress", True),
    ("neo_3d", 3, "neo_hooke", "stress", False),
]
NL_PARAMS = [pytest.param(c[1:], id=c[0]) for c in NL_CASES]


def _nl_model(case) -> tuple[NonlinearElasticity, Domain]:
    dim, material, plane, field = case
    rng = np.random.default_rng(21)
    d = Domain((6, 3), h=H, thickness=THICK) if dim == 2 else Domain((3, 2, 2), h=H)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    D = None
    if field:
        if dim == 2:
            R = rotations_about_z(rng.uniform(-90.0, 90.0, d.n_active))
        else:
            R = rotations_from_axes(rng.standard_normal((d.n_active, 3)), rng.standard_normal((d.n_active, 3)))
        D = unit_elasticity_field(MAT, dim, plane, R)
    return NonlinearElasticity(d, fixed, material=material, plane=plane, elasticity=D), d


def _deformed_state(nl: NonlinearElasticity, d: Domain, seed: int = 3):
    """A random finite deformation of the reference grid: stretched and rotated by 0.5 rad, plus node noise; random E."""
    rng = np.random.default_rng(seed)
    X = d.node_coords()
    G = _rotation(d.dim, 0.5) @ (np.eye(d.dim) + 0.2 * np.diag(rng.random(d.dim))) - np.eye(d.dim)
    u = (X @ G.T).ravel() + 0.08 * d.h * rng.standard_normal(d.ndof)
    E = np.exp(rng.uniform(np.log(2e-4), 0.0, nl.n_elements))
    return u, E


@pytest.mark.parametrize("case", NL_PARAMS)
def test_i4_objectivity_under_a_large_rigid_rotation(case):
    nl, d = _nl_model(case)
    u, E = _deformed_state(nl, d)
    X = d.node_coords()
    axis = np.random.default_rng(8).standard_normal(3)
    Q = _rotation(d.dim, 1.0, axis)
    uQ = ((X + u.reshape(-1, d.dim)) @ Q.T - X).ravel()                # u_Q = Q (X + u) - X
    W, WQ = nl.strain_energy(u, E), nl.strain_energy(uQ, E)
    f, fQ = nl.internal_force(u, E), nl.internal_force(uQ, E)
    want = (f.reshape(-1, d.dim) @ Q.T).ravel()                        # Q f_int(u) node-wise
    e_w = abs(WQ - W) / abs(W)
    e_f = float(np.max(np.abs(fQ - want)) / np.max(np.abs(f)))
    print(f"\n  I4 {case}: energy {e_w:.2e} (W = {W:.6g}), internal force {e_f:.2e} (max|f| = {np.max(np.abs(f)):.4g})")
    assert W > 0.0
    assert e_w <= 1e-12
    assert e_f <= 1e-11


def _full_k0(nl: NonlinearElasticity, E: np.ndarray) -> sp.csr_matrix:
    """Small-displacement stiffness (ndof x ndof, all DOFs) from the element tangents at u = 0."""
    Ke = nl.element_state(np.zeros(nl.domain.ndof), E)["K"]
    nloc = Ke.shape[1]
    rows = np.repeat(nl.edof, nloc, axis=1).ravel()
    cols = np.tile(nl.edof, (1, nloc)).ravel()
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(nl.domain.ndof,) * 2).tocsr()


@pytest.mark.parametrize("case", NL_PARAMS)
def test_i5_pure_rigid_motion_has_no_force_and_no_energy(case):
    nl, d = _nl_model(case)
    _, E = _deformed_state(nl, d)
    X = d.node_coords()
    rng = np.random.default_rng(9)
    Q = _rotation(d.dim, 1.0, rng.standard_normal(3))
    t = rng.standard_normal(d.dim)
    u = (X @ Q.T + t - X).ravel()                                      # rigid motion from u = 0
    K0 = _full_k0(nl, E)
    k1 = float(spla.norm(K0, 1))                                       # induced 1-norm (>= the 2-norm, K0 symmetric)
    uK0u = float(u @ (K0 @ u))
    f = nl.internal_force(u, E)
    W = nl.strain_energy(u, E)
    f_bound = 1e-12 * k1 * np.linalg.norm(u)
    w_bound = 1e-12 * 0.5 * uK0u
    print(f"\n  I5 {case}: |f|_2 {np.linalg.norm(f):.2e} <= {f_bound:.2e} (|K0|_1 |u|_2 = {k1 * np.linalg.norm(u):.3g}); "
          f"W {W:.2e} <= {w_bound:.2e} (1/2 u'K0u = {0.5 * uK0u:.3g})")
    assert uK0u > 0.0
    assert np.linalg.norm(f) <= f_bound
    assert abs(W) <= w_bound

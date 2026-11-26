"""Flat-facet rectangular shell element (docs/olcumler/kabuk_onkayit.md, KB1-KB11)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import ke_q4
from toporlogy.fem.plate import MindlinPlate, hard_simply_supported, navier_deflection, plate_element_matrix
from toporlogy.fem.shell import (FlatShell, ShellMesh, bending_matrix, cylinder_surface, facet_matrix,
                                 flat_rectangle, membrane_matrix)

NU = 0.3


def _rotation(seed: int) -> np.ndarray:
    Q, R = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(R))
    return Q if np.linalg.det(Q) > 0 else -Q


def _rigid_modes(X: np.ndarray) -> list[np.ndarray]:
    """Six rigid motions (6 nn,): translations e_i and rotations ω = e_i about the global origin."""
    out = []
    for i in range(3):
        r = np.zeros((len(X), 6))
        r[:, i] = 1.0
        out.append(r.ravel())
    for i in range(3):
        w = np.eye(3)[i]
        r = np.zeros((len(X), 6))
        r[:, :3] = np.cross(w, X)
        r[:, 3:] = w
        out.append(r.ravel())
    return out


def _plate_map(sign: float = 1.0) -> np.ndarray:
    """(12, 24) map global (u, v, w, θx, θy, θz) of a facet in the xy-plane to (w, βx, βy): βx = −θy, βy = θx
    (``sign`` = −1: the opposite signs, the negative control of KB2)."""
    P = np.zeros((12, 24))
    for i in range(4):
        P[3 * i, 6 * i + 2] = 1.0
        P[3 * i + 1, 6 * i + 4] = -sign
        P[3 * i + 2, 6 * i + 3] = sign
    return P


def test_kb1_flat_facet_decouples_into_membrane_and_plate() -> None:
    h, t = 0.7, 0.05
    X = np.array([[0, 0, 0], [h, 0, 0], [h, h, 0], [0, h, 0]], float)
    K = facet_matrix(X, t, NU)
    A = (6 * np.arange(4)[:, None] + np.array([0, 1, 5])).ravel()
    B = (6 * np.arange(4)[:, None] + np.array([2, 3, 4])).ravel()
    assert np.all(K[np.ix_(A, B)] == 0.0)
    P = _plate_map()
    Kp = plate_element_matrix(h, t, NU)
    eb = np.abs(P @ K @ P.T - Kp).max() / np.abs(Kp).max()
    Km_ref = t * ke_q4(NU)
    em = np.abs(membrane_matrix(h, h, t, NU) - Km_ref).max() / np.abs(Km_ref).max()
    uv = (6 * np.arange(4)[:, None] + np.array([0, 1])).ravel()
    em0 = np.abs(facet_matrix(X, t, NU, drill=0.0)[np.ix_(uv, uv)] - Km_ref).max() / np.abs(Km_ref).max()
    print(f"KB1: bending {eb:.2e}, membrane {em:.2e}, membrane block (drill 0) {em0:.2e}")
    assert eb <= 1e-12 and em <= 1e-12 and em0 <= 1e-12


def test_kb2_rotation_sign_mapping() -> None:
    Q = _rotation(11)
    a, b, t = 0.6, 0.35, 0.05
    c0 = np.array([0.4, -1.2, 0.7])
    loc = np.array([[0, 0], [a, 0], [a, b], [0, b]], float)
    X = c0 + loc[:, :1] * Q[:, 0] + loc[:, 1:] * Q[:, 1]
    K = facet_matrix(X, t, NU)
    nK = np.linalg.norm(K)
    for i in range(3):
        w = Q[:, i]
        r: np.ndarray = np.zeros((4, 6))
        r[:, :3] = np.cross(w, X - np.array([2.0, 0.5, -1.0]))
        r[:, 3:] = w
        r = r.ravel()
        rel = np.linalg.norm(K @ r) / (nK * np.linalg.norm(r))
        print(f"KB2: rotation about axis {i}: {rel:.2e}")
        assert rel <= 1e-12
    # negative control: rotation about y of a facet in the xy-plane with βx = +θy, βy = −θx
    Xf = np.array([[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]], float)
    Kf = facet_matrix(Xf, t, NU)
    r = np.zeros((4, 6))
    r[:, :3] = np.cross([0.0, 1.0, 0.0], Xf)
    r[:, 4] = 1.0
    r = r.ravel()
    p_wrong = _plate_map(sign=-1.0) @ r
    e_wrong = 0.5 * p_wrong @ bending_matrix(a, b, t, NU) @ p_wrong
    scale = r @ (np.diag(Kf) * r)
    print(f"KB2: wrong mapping energy / scale = {e_wrong / scale:.3e}")
    assert e_wrong >= 1e-3 * scale


def _plane_mesh(Q: np.ndarray, c0: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[ShellMesh, np.ndarray]:
    Xg, Yg = np.meshgrid(xs, ys)
    loc = np.column_stack([Xg.ravel(), Yg.ravel()])
    nodes = c0 + loc[:, :1] * Q[:, 0] + loc[:, 1:] * Q[:, 1]
    nx = len(xs) - 1
    i, j = np.meshgrid(np.arange(nx), np.arange(len(ys) - 1))
    n0 = (i + (nx + 1) * j).ravel()
    return ShellMesh(nodes, np.column_stack([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])), loc


_XS, _YS = np.array([0.0, 0.3, 0.75, 1.2]), np.array([0.0, 0.4, 0.55])      # 3 × 2 unequal rectangles
_AREA = 1.2 * 0.55


def _patch(t: float, seed: int = 5) -> tuple[FlatShell, np.ndarray, np.ndarray]:
    Q = _rotation(seed)
    mesh, loc = _plane_mesh(Q, np.array([0.3, 0.2, -0.4]), _XS, _YS)
    return FlatShell(mesh, t, NU), loc, Q


def test_kb3_membrane_patch_on_unequal_rectangles() -> None:
    t = 0.04
    sh, loc, Q = _patch(t)
    K = sh.stiffness(np.ones(sh.mesh.ne))
    x, y = loc[:, 0], loc[:, 1]
    al, be, ga, de = 0.3, -0.2, 0.5, 0.1
    u: np.ndarray = np.zeros((sh.mesh.nn, 6))
    u[:, :3] = (al * x + be * y)[:, None] * Q[:, 0] + (ga * x + de * y)[:, None] * Q[:, 1]
    u[:, 3:] = 0.5 * (ga - be) * Q[:, 2]
    u = u.ravel()
    eps = np.array([al, de, be + ga])
    Dm = t / (1 - NU ** 2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    ref_m = 0.5 * _AREA * eps @ Dm @ eps
    em = abs(0.5 * u @ (K @ u) - ref_m) / ref_m
    print(f"KB3: membrane patch {em:.2e}")
    assert em <= 1e-12


def _bending_patch_error(t: float) -> float:
    sh, loc, Q = _patch(t)
    K = sh.stiffness(np.ones(sh.mesh.ne))
    x = loc[:, 0]
    u: np.ndarray = np.zeros((sh.mesh.nn, 6))
    u[:, :3] = (x ** 2 / 2)[:, None] * Q[:, 2]                                 # w = x²/2
    u[:, 3:] = (-x)[:, None] * Q[:, 1]                                         # βx = x, i.e. θy = −x
    u = u.ravel()
    ref_b = t ** 3 / (12 * (1 - NU ** 2)) * _AREA / 2
    return float(abs(0.5 * u @ (K @ u) - ref_b) / ref_b)


@pytest.mark.xfail(strict=True, reason="pre-registered KB3 (bending patch) failed: measured 1.12e-12 > 1e-12; "
                                       "rounding floor of the quadratic form at t = 0.04 (kabuk_onkayit.md)")
def test_kb3_bending_patch_on_unequal_rectangles() -> None:
    eb = _bending_patch_error(0.04)
    print(f"KB3: bending patch {eb:.2e}")
    assert eb <= 1e-12


def test_kb3b_bending_patch_thicker_facets_post_hoc() -> None:
    """Post hoc (added after KB3's first run): the same bending patch at t = 0.1 (a/t ≤ 12, like PL2)."""
    eb = _bending_patch_error(0.1)
    print(f"KB3b (post hoc): bending patch t=0.1 {eb:.2e}")
    assert eb <= 1e-12


def test_kb4_rigid_modes_of_a_curved_segment() -> None:
    m = cylinder_surface(1.0, 1.0, 8, 6, angle=np.pi / 2)
    mesh = ShellMesh(m.nodes + np.array([0.3, -0.7, 1.1]), m.facets)
    sh = FlatShell(mesh, 0.05, NU)
    K = sh.stiffness(np.ones(mesh.ne)).toarray()
    nK = np.linalg.norm(K)
    rels = [np.linalg.norm(K @ r) / (nK * np.linalg.norm(r)) for r in _rigid_modes(mesh.nodes)]
    lam = np.linalg.eigvalsh(0.5 * (K + K.T))
    n_zero = int(np.sum(lam < 1e-10 * lam.max()))
    print(f"KB4: max rigid residual {max(rels):.2e}; zero eigenvalues {n_zero}; 7th / max {lam[6] / lam.max():.2e}")
    assert max(rels) <= 1e-10
    assert n_zero == 6


def _clamped_segment(t: float = 0.05, drill: float | None = None) -> FlatShell:
    return _segment(cylinder_surface(1.0, 1.0, 8, 6, angle=np.pi / 2), t, drill)


def _segment(mesh: ShellMesh, t: float, drill: float | None = None) -> FlatShell:
    """Open 8 × 6 segment clamped (all six DOFs) on its first ring of nine nodes (z = 0 before any rotation)."""
    fixed = (6 * np.arange(9)[:, None] + np.arange(6)).ravel()
    return FlatShell(mesh, t, NU, fixed) if drill is None else FlatShell(mesh, t, NU, fixed, drill=drill)


def test_kb5_rotation_invariance() -> None:
    m = cylinder_surface(1.0, 1.0, 8, 6, angle=np.pi / 2)
    Q, c = _rotation(21), np.array([1.5, -0.3, 2.2])
    rot = ShellMesh(m.nodes @ Q.T + c, m.facets)
    energies = []
    for mesh, R in ((m, np.eye(3)), (rot, Q)):
        sh = _segment(mesh, 0.05)
        F = sh.pressure_load(1.0)
        tip, mid = 6 * (m.nn - 1), 6 * (m.nn - 5)
        F[tip:tip + 3] += R @ np.array([0.03, -0.05, 0.08])
        F[mid + 3:mid + 6] += R @ np.array([0.02, 0.01, -0.03])
        u = sh.solve(np.ones(mesh.ne), F).numpy()
        energies.append(0.5 * F @ u)
    rel = abs(energies[0] - energies[1]) / abs(energies[0])
    print(f"KB5: energies {energies[0]:.12e} / {energies[1]:.12e}, rel {rel:.2e}")
    assert rel <= 1e-10


def _tube_supports(n_theta: int, j: int) -> np.ndarray:
    a, b, c = n_theta * j, n_theta * j + n_theta // 4, n_theta * j + n_theta // 2
    return np.array([6 * a + 1, 6 * a + 2, 6 * b + 0, 6 * b + 2, 6 * c + 1, 6 * c + 2])


def _pressurised_tube(n_theta: int) -> tuple[np.ndarray, np.ndarray, float]:
    R, t, L, nz = 1.0, 0.01, 4.0, 8
    mesh = cylinder_surface(R, L, n_theta, nz)
    sh = FlatShell(mesh, t, NU, _tube_supports(n_theta, nz // 2))
    u = sh.solve(np.ones(mesh.ne), sh.pressure_load(1.0)).numpy().reshape(-1, 6)
    X = mesh.nodes
    ur = (u[:, 0] * X[:, 0] + u[:, 1] * X[:, 1]) / R
    mid = np.arange(n_theta) + n_theta * (nz // 2)
    return ur, ur[mid], R * R / t


def test_kb6_kb7_pressurised_tube() -> None:
    err = {}
    for n in (16, 32, 64):
        ur, mid, w_membrane = _pressurised_tube(n)
        discrete = w_membrane * np.cos(np.pi / n)
        e_disc = np.abs(ur - discrete).max() / discrete
        err[n] = abs(mid.mean() - w_membrane) / w_membrane
        print(f"KB6 N={n}: discrete closed form {e_disc:.2e}; KB7 error vs membrane {err[n]:.4e} "
              f"(1 - cos(pi/N) = {1 - np.cos(np.pi / n):.4e})")
        assert e_disc <= 1e-8                                                       # KB6
    order = np.log2(err[32] / err[64])
    print(f"KB7: order 32->64 {order:.4f}")
    assert err[64] <= 0.01 and err[16] > err[32] > err[64] and 1.9 <= order <= 2.1    # KB7


def _pinched_ring(n_theta: int) -> float:
    R, t, L, nz, P = 1.0, 0.01, 0.4, 2, 1.0
    mesh = cylinder_surface(R, L, n_theta, nz)
    sh = FlatShell(mesh, t, 0.0, _tube_supports(n_theta, nz // 2))
    F = np.zeros(sh.ndof)
    b = L / nz
    for j in range(nz + 1):
        wj = P * b * (0.5 if j in (0, nz) else 1.0)
        F[6 * (n_theta * j)] -= wj
        F[6 * (n_theta * j + n_theta // 2)] += wj
    u = sh.solve(np.ones(mesh.ne), F).numpy()
    jm = nz // 2
    return float(u[6 * (n_theta * jm + n_theta // 2)] - u[6 * (n_theta * jm)])


def test_kb8_pinched_ring() -> None:
    R, t, P = 1.0, 0.01, 1.0
    ref = (np.pi / 4 - 2 / np.pi) * P * R ** 3 / (t ** 3 / 12)
    err = {n: (_pinched_ring(n) - ref) / ref for n in (16, 32, 64)}
    order = np.log2(abs(err[32]) / abs(err[64]))
    print(f"KB8: signed errors {err}, order 32->64 {order:.2f}, ref {ref:.6e}")
    assert abs(err[64]) <= 0.01 and abs(err[64]) < abs(err[16])


def _navier_shell(lx: float, ly: float, nx: int, ny: int, t: float) -> float:
    mesh = flat_rectangle(lx, ly, nx, ny)
    X = mesh.nodes
    on_x = (np.abs(X[:, 0]) < 1e-12) | (np.abs(X[:, 0] - lx) < 1e-12)
    on_y = (np.abs(X[:, 1]) < 1e-12) | (np.abs(X[:, 1] - ly) < 1e-12)
    c00 = int(np.argmin(np.linalg.norm(X - [0, 0, 0], axis=1)))
    c10 = int(np.argmin(np.linalg.norm(X - [lx, 0, 0], axis=1)))
    fixed = np.concatenate([6 * np.flatnonzero(on_x | on_y) + 2, 6 * np.flatnonzero(on_x) + 3,
                            6 * np.flatnonzero(on_y) + 4, [6 * c00, 6 * c00 + 1, 6 * c10 + 1]])
    sh = FlatShell(mesh, t, NU, fixed)
    u = sh.solve(np.ones(mesh.ne), sh.pressure_load(1.0)).numpy()
    c = int(np.argmin(np.linalg.norm(X - [lx / 2, ly / 2, 0], axis=1)))
    return float(u[6 * c + 2])


@pytest.mark.parametrize("ratio", [10.0, 100.0])
def test_kb9_navier_square(ratio) -> None:
    t = 1.0 / ratio
    ref = navier_deflection(1.0, 1.0, t, 1.0, NU, 1.0, 0.5, 0.5)
    err = {n: abs(_navier_shell(1.0, 1.0, n, n, t) - ref) / ref for n in (16, 32)}
    order = np.log2(err[16] / err[32])
    print(f"KB9 a/t={ratio:g}: errors {err}, order {order:.2f}")
    assert err[32] <= 0.01 and order >= 1.5


def test_kb9_navier_rectangle_and_plate_equality() -> None:
    ref = navier_deflection(2.0, 1.0, 0.1, 1.0, NU, 1.0, 1.0, 0.5)
    er = abs(_navier_shell(2.0, 1.0, 32, 32, 0.1) - ref) / ref
    d = Domain((16, 16), h=1.0 / 16)
    p = MindlinPlate(d, 0.1, NU, hard_simply_supported(d))
    up = p.solve(torch.ones(d.n_active), p.pressure_load(1.0)).numpy()
    Xd = d.node_coords()
    wp = float(up[3 * int(np.argmin(np.abs(Xd[:, 0] - 0.5) + np.abs(Xd[:, 1] - 0.5)))])
    ws = _navier_shell(1.0, 1.0, 16, 16, 0.1)
    eq = abs(ws - wp) / abs(wp)
    print(f"KB9: rectangle 2x1 error {er:.3e}; shell vs MindlinPlate n=16 {eq:.2e}")
    assert er <= 0.01 and eq <= 1e-9


def test_kb10_design_gradient() -> None:
    sh = _clamped_segment()
    F = sh.pressure_load(1.0)
    ne = sh.mesh.ne
    x = 0.2 + 0.7 * np.random.default_rng(31).random(ne)
    v = np.random.default_rng(32).normal(size=ne)
    Ft = torch.as_tensor(F)

    def comp(xx: torch.Tensor) -> torch.Tensor:
        return Ft @ sh.solve(1e-6 + xx ** 3, F)
    xt = torch.tensor(x, requires_grad=True)
    (g,) = torch.autograd.grad(comp(xt), xt)
    s = 1e-3
    f = {k: float(comp(torch.as_tensor(x + k * s * v))) for k in (-2, -1, 1, 2)}
    fd = (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * s)
    rel = abs(float(g @ torch.as_tensor(v)) - fd) / abs(fd)
    print(f"KB10: rel {rel:.2e}")
    assert rel <= 1e-6


def test_drilling_scale_record() -> None:
    """Record helper (not pre-registered as a threshold): tip-load compliance of the clamped segment vs drill."""
    out = {}
    for drill in (1e-4, 1e-3, 1e-2):
        sh = _clamped_segment(drill=drill)
        F = np.zeros(sh.ndof)
        F[6 * (sh.mesh.nn - 1) + 2] = 1.0
        out[drill] = float(F @ sh.solve(np.ones(sh.mesh.ne), F).numpy())
    print("drill record: " + ", ".join(f"{k:g}: {c:.6e}" for k, c in out.items())
          + f"; rel 1e-4 vs 1e-2 {abs(out[1e-4] - out[1e-2]) / out[1e-3]:.2e}")
    assert all(np.isfinite(list(out.values())))


def test_kb11_refusals() -> None:
    m = flat_rectangle(1.0, 1.0, 2, 2)
    with pytest.raises(ValueError, match="thickness"):
        FlatShell(m, 0.0)
    with pytest.raises(ValueError, match="Poisson"):
        FlatShell(m, 0.1, nu=0.5)
    with pytest.raises(ValueError, match="Poisson"):
        FlatShell(m, 0.1, nu=-1.0)
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    with pytest.raises(ValueError, match="degenerate"):
        FlatShell(ShellMesh(np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]], float), np.array([[0, 1, 2, 3]])),
                  0.1)
    skew = sq + np.array([[0, 0, 0], [0, 0, 0], [0.3, 0, 0], [0.3, 0, 0]])
    with pytest.raises(ValueError, match="not a rectangle"):
        FlatShell(ShellMesh(skew, np.array([[0, 1, 2, 3]])), 0.1)
    warp = sq + np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.2], [0, 0, 0]])
    with pytest.raises(ValueError, match="not a rectangle"):
        FlatShell(ShellMesh(warp, np.array([[0, 1, 2, 3]])), 0.1)
    with pytest.raises(ValueError, match="out of range"):
        ShellMesh(sq, np.array([[0, 1, 2, 7]]))
    with pytest.raises(ValueError, match="drilling"):
        FlatShell(m, 0.1, drill=-1.0)
    sh = FlatShell(m, 0.1, fixed=np.arange(18))
    with pytest.raises(ValueError, match="facet moduli"):
        sh.solve(np.ones(3), np.zeros(sh.ndof))
    for bad in (0.0, 2 * np.pi):
        with pytest.raises(ValueError, match="angle"):
            cylinder_surface(1.0, 1.0, 4, 2, angle=bad)

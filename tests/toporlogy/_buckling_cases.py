"""Problems of docs/olcumler/burkulma_geometri_onkayit.md (§A–§C, §E), shared by the tests and the measurement script
(one implementation per number).

Every function returns plain numbers or arrays; nothing here asserts.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

import toporlogy as T
from toporlogy.fem.buckling import BucklingClusterWarning, LinearBuckling

# =========================================================================================== A1, A2: Euler column
B_COL, L_COL, E_COL = 4.0, 160.0, 1000.0


def euler_timoshenko(b: float = B_COL, L: float = L_COL, E: float = E_COL, nu: float = 0.0) -> float:
    """Clamped-free column, P_e = π² E I / (4 L²), with the shear correction P_T = P_e / (1 + P_e / (κ G A)),
    κ = 5/6, plane stress unit thickness."""
    I, A, G = b ** 3 / 12.0, b, E / (2.0 * (1.0 + nu))
    pe = np.pi ** 2 * E * I / (4.0 * L ** 2)
    return float(pe / (1.0 + pe / (5.0 / 6.0 * G * A)))


def _compressed(d: T.Domain, fixed: np.ndarray, top: np.ndarray, nu: float, element: str | None = None):
    X = d.node_coords()
    F = d.load_vector()
    w = np.ones(len(top))
    xs = X[top, 0]
    w[np.isclose(xs, xs.min()) | np.isclose(xs, xs.max())] = 0.5
    F[2 * top + 1, 0] = -w / w.sum()                                  # unit compressive resultant
    return T.Elasticity(d, fixed, nu, element), F


def euler_column(nb: int, element: str | None = None) -> float:
    """λ1 of the b × L column with ``nb`` elements across (A1, A2)."""
    h = B_COL / nb
    d = T.Domain((nb, int(round(L_COL / h))), h)
    X = d.node_coords()
    top = np.flatnonzero(np.isclose(X[:, 1], L_COL))
    m, F = _compressed(d, d.dofs(np.flatnonzero(X[:, 1] == 0)), top, 0.0, element)
    E = np.full(d.n_active, E_COL)
    mu, _ = LinearBuckling(m).modes(E, E, m.solve_numpy(E, F)[:, 0], 2)
    return float(1.0 / mu[0])


# =========================================================================================== A3, SP1: embedded member
B_MEM, L_MEM, WIDE = 4.0, 80.0, 5           # column b × L in the middle of a WIDE·b wide design domain
EG_METHODS = ("stiffness", "pg=1", "pg=2", "pg=3", "pg=4")


def stress_modulus(x: np.ndarray, emin: float, method: str, E0: float = 1.0, p: float = 3.0) -> np.ndarray:
    """E_G of the methods of §B: ``stiffness`` = the SIMP stiffness E0 (emin + (1 − emin) x^p); ``pg=q`` =
    E0 x^q without a lower bound."""
    if method == "stiffness":
        return E0 * (emin + (1.0 - emin) * x ** p)
    if method.startswith("pg="):
        return E0 * x ** float(method[3:])
    raise ValueError(method)


def embedded_member(nb: int, emin: float = 1e-6, method: str = "pg=3", alone: bool = False) -> dict[str, float]:
    """λ1 and the diagnostics of mode 1 of the 0/1 column in a void design domain (bottom edge clamped over the full
    width, unit compression on the column top); ``alone``: the column by itself on the same mesh."""
    h = B_MEM / nb
    ny = int(round(L_MEM / h))
    nx = nb if alone else WIDE * nb
    d = T.Domain((nx, ny), h)
    X = d.node_coords()
    x0 = 0.0 if alone else (WIDE - 1) / 2 * B_MEM
    xc = d.element_centers()[:, 0]
    x = ((xc > x0) & (xc < x0 + B_MEM)).astype(float)
    top = np.flatnonzero(np.isclose(X[:, 1], L_MEM) & (X[:, 0] > x0 - 1e-9) & (X[:, 0] < x0 + B_MEM + 1e-9))
    m, F = _compressed(d, d.dofs(np.flatnonzero(X[:, 1] == 0)), top, 0.3)
    xa = x[m.elements]
    E = emin + (1.0 - emin) * xa ** 3
    EG = stress_modulus(xa, emin, method)
    U = m.solve_numpy(E, F)[:, 0]
    lb = LinearBuckling(m)
    mu, modes = lb.modes(E, EG, U, 2)
    return {"lambda1": float(1.0 / mu[0]), "low_density_share": float(lb.low_density_share(E, modes, xa)[0]),
            "stress_energy_share": float(lb.stress_energy_share(EG, U, modes, xa)[0]),
            "localization": float(lb.localization(E, modes)[0]), "dofs": int(len(m.free))}


# =========================================================================================== A4–A6: singular wall
def wall(n: int, exclude_r: float | None = None) -> tuple[float, float, float]:
    """The solid wall of Ferrari, Sigmund & Guest (2021) §5.2 on an n × n mesh (n = 40 k; supported node columns of
    the 320 mesh in units of n/40), opening void, x-load 1e-2 on the left edge. Returns λ1, its localisation and
    the energy share of mode 1 within 0.02 of a clamped-patch end. ``exclude_r``: no stress stiffness within that
    distance of a clamped-patch end."""
    h = 1.0 / n
    d = T.Domain((n, n), h)
    t = n // 40
    ix, iy = np.indices((n, n))
    row, col = n - 1 - iy, ix
    void = (row >= 2 * n // 5 - 1) & (col >= 2 * n // 5 - 1) & (col <= n - n // 5 - 1)
    x = np.where(void, 0.0, 1.0).ravel()
    X = d.node_coords()
    xi = np.rint(X[:, 0] / h).astype(int)
    cols = np.r_[0:t + 1, 2 * n // 5 - t - 1:2 * n // 5, 4 * n // 5:4 * n // 5 + t + 1, n - t:n + 1]
    fixed = d.dofs(np.flatnonzero(np.isclose(X[:, 1], 0) & np.isin(xi, cols)))
    left = np.flatnonzero(np.isclose(X[:, 0], 0))
    F = d.load_vector()
    w = np.ones(len(left))
    w[[0, -1]] = 0.5
    F[2 * left, 0] = 1e-2 * w / w.sum()
    m = T.Elasticity(d, fixed, 0.3)
    E = 1e-6 + x ** 3 * (1 - 1e-6)
    U = m.solve_numpy(E, F)[:, 0]
    c = d.element_centers()
    ends = np.array([t, 2 * n // 5 - t - 1, 2 * n // 5 - 1, 4 * n // 5, 4 * n // 5 + t, n - t]) * h
    dist = np.min(np.hypot(c[:, 0][:, None] - ends[None, :], c[:, 1][:, None]), axis=1)
    EG = x ** 3 if exclude_r is None else x ** 3 * (dist >= exclude_r)
    lb = LinearBuckling(m)
    mu, modes = lb.modes(E, EG, U, 2)
    _, kE = lb.mode_forms(modes)
    en = E * kE[:, 0]
    _WALL_ENERGY[(n, exclude_r)] = en
    return 1 / mu[0], lb.localization(E, modes)[0], float(en[dist < 0.02].sum() / en.sum())


_WALL_ENERGY: dict[tuple[int, float | None], np.ndarray] = {}


def wall_mode_energy(n: int, exclude_r: float | None = None) -> np.ndarray:
    """Mode-1 strain energy per element (element order of the n × n domain) of :func:`wall`."""
    if (n, exclude_r) not in _WALL_ENERGY:
        wall(n, exclude_r)
    return _WALL_ENERGY[(n, exclude_r)]


# =========================================================================================== D2: a grey design
def d2_study(buckling: Any = None, target: float | None = None) -> tuple[T.Study, T.Domain]:
    """The column problem of test_buckling.test_constraint_raises_the_load_factor (16 × 48, volume 0.35,
    E_min 1e-6, filter 2.0, MMA move 0.1); with ``buckling`` (a BucklingKS) the constraint λ1 ≥ ``target``."""
    d = T.Domain((16, 48), 1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 1] == 0))
    top = np.flatnonzero((X[:, 1] == 48) & (np.abs(X[:, 0] - 8) <= 2))
    F = d.load_vector()
    F[2 * top + 1, 0] = -1.0 / len(top)
    cons = [T.Constraint(T.Volume(), 0.35)]
    if buckling is not None:
        assert target is not None
        cons.append(T.Constraint(buckling, 1.0 / target, scale=target))
    st = T.Study(d, T.Elasticity(d, fixed, 0.3), F, T.standard_chain(d, 2.0, projection=False),
                 T.SIMP(d, E0=1.0, emin_ratio=1e-6), T.Compliance(), cons, T.MMA(move=0.1))
    return st, d


def d2_design() -> tuple[np.ndarray, np.ndarray, T.Study, T.Domain]:
    """(physical density (domain shape), design variables, study, domain) of the compliance + volume optimum of
    :func:`d2_study` (60 iterations)."""
    st, d = d2_study()
    res = st.run(max_iter=60, tol=1e-4)
    return np.asarray(res.xphys, float).reshape(d.shape), np.asarray(res.x, float), st, d


def evaluate_density(st: T.Study, d: T.Domain, x: np.ndarray, emin: float = 1e-6, method: str = "pg=3",
                     n_modes: int = 6) -> dict[str, Any]:
    """Compliance, λ1 and mode-1 diagnostics of the density ``x`` on the study's mesh and load (E0 = 1, p = 3),
    stress modulus ``method``; the max centroid von Mises of the solid-material stress in elements x ≥ 0.5;
    the number of bodies of x ≥ 0.5 (8-connected)."""
    from scipy import ndimage

    from toporlogy.core.tensor import von_mises_squared
    m = st.physics
    xa = np.asarray(x, float).ravel()[m.elements]
    E = emin + (1.0 - emin) * xa ** 3
    EG = stress_modulus(xa, emin, method)
    F = np.asarray(st.F, float)
    U = m.solve_numpy(E, F)[:, 0]
    lb = LinearBuckling(m)
    mu, modes = lb.modes(E, EG, U, n_modes)
    s = lb.element_stress(np.ones_like(xa), U).mean(axis=1)                    # solid-material centroid stress
    vm = np.sqrt(von_mises_squared(np.stack([s[:, 0], s[:, 1], np.zeros(len(s)), np.zeros(len(s)),
                                             np.zeros(len(s)), s[:, 2]], axis=1)))
    solid = xa >= 0.5
    lab = ndimage.label(np.asarray(x).reshape(d.shape) >= 0.5, structure=np.ones((3, 3)))[1]
    return {"compliance": float(F[:, 0] @ U), "lambda1": float(1.0 / mu[0]) if mu[0] > 0 else float("inf"),
            "low_density_share": float(lb.low_density_share(E, modes, xa)[0]),
            "stress_energy_share": float(lb.stress_energy_share(EG, U, modes, xa)[0]),
            "localization": float(lb.localization(E, modes)[0]),
            "vm_max_solid": float(vm[solid].max()) if solid.any() else float("nan"),
            "volume": float(xa.sum()), "bodies": int(lab)}


def volume_variants(x: np.ndarray) -> dict[str, np.ndarray]:
    """§E: V0 grey; V1 the volume-preserving 0/1 threshold; V2, V3 Gaussian blur σ = 1, 2 elements, then the
    volume-preserving 0/1 threshold. Volume-preserving = the round(Σ x) largest values are solid."""
    from scipy import ndimage

    def top(f: np.ndarray) -> np.ndarray:
        n = int(round(float(x.sum())))
        out = np.zeros(f.size)
        out[np.argsort(-f.ravel(), kind="stable")[:n]] = 1.0
        return out.reshape(f.shape)
    return {"V0_grey": x.copy(), "V1_threshold": top(x),
            "V2_blur1": top(ndimage.gaussian_filter(x, 1.0, mode="nearest")),
            "V3_blur2": top(ndimage.gaussian_filter(x, 2.0, mode="nearest"))}


# =========================================================================================== C: square column
def square_column_study(n_modes: int) -> tuple[T.Study, Any, T.Domain]:
    """3 × 3 × 24 column (h 0.5, x = 1, standard_chain 1.2, SIMP E0 50): double λ1. Returns (study, response,
    domain); the buckling response (ρ 20, scale 1) is constraint 1."""
    from toporlogy.optimization.buckling import BucklingKS
    d = T.Domain((3, 3, 24), 0.5)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 2] == 0))
    top = np.flatnonzero(np.isclose(X[:, 2], X[:, 2].max()))
    F = d.load_vector()
    F[3 * top + 2, 0] = -1.0 / len(top)
    resp = BucklingKS(n_modes=n_modes, rho=20.0, scale=1.0)
    st = T.Study(d, T.Elasticity(d, fixed, 0.3), F, T.standard_chain(d, 1.2), T.SIMP(d, E0=50.0),
                 T.Compliance(), [T.Constraint(T.Volume(), 0.6), T.Constraint(resp, 1.0, scale=1.0)], T.MMA())
    return st, resp, d


def response_value(st: T.Study, resp: Any, x: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BucklingClusterWarning)
        ctx = st.context(torch.as_tensor(x), True)
        return float(resp(ctx))


def one_sided_derivatives(st: T.Study, resp: Any, x0: np.ndarray, d: np.ndarray, h: float = 1e-4
                          ) -> tuple[float, float]:
    """(D+, D−): one-sided differences along ``d`` with Richardson's step doubling, 2 D(h) − D(2h), second order
    (RL2)."""
    j0 = response_value(st, resp, x0)

    def side(s: float) -> float:
        a = (response_value(st, resp, x0 + s * h * d) - j0) / (s * h)
        b = (response_value(st, resp, x0 + 2 * s * h * d) - j0) / (2 * s * h)
        return 2 * a - b
    return side(1.0), side(-1.0)


# =========================================================================================== D: stepped bar
def stepped_bar_sdf(X: np.ndarray) -> np.ndarray:
    """GC: 8 × 8 bar along x ∈ [0, 32] (y, z ∈ [2, 10]) with a 4 × 8 neck (y ∈ [4, 8]) for x ∈ [12, 20]."""
    from toporlogy.post.primitives import BoxPrimitive
    a = BoxPrimitive((0, 2, 2), (12, 10, 10)).sdf(X)
    b = BoxPrimitive((20, 2, 2), (32, 10, 10)).sdf(X)
    c = BoxPrimitive((12, 4, 2), (20, 8, 10)).sdf(X)
    return np.minimum(np.minimum(a, b), c)

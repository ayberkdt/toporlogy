"""Geometrically non-linear elasticity (``toporlogy.fem.nonlinear``) and end compliance (``optimization.nonlinear``).

Tolerances are the pre-registered ones of ``docs/olcumler/geometrik_dogrusal_olmama.md`` §0.2 (written and committed
before the first run, 25 Nov 2026 17:58 +03:00); the V-numbers below refer to that table. Checks added after the
first run are marked "post hoc". The elastica reference is ``tests/toporlogy/_elastica.py`` (two independent
solutions of the same equation).
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from toporlogy.configuration.build import study_from_dict
from toporlogy.configuration.registry import ConfigError
from toporlogy.core.domain import Domain
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import D_iso, D_plane_strain, D_plane_stress
from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearConvergenceError, NonlinearElasticity
from toporlogy.optimization.compose import Displacement
from toporlogy.optimization.nonlinear import EndCompliance
from toporlogy.optimization.study import Study
from toporlogy.optimizers.mma import MMA

sys.path.insert(0, str(Path(__file__).parent))
from _elastica import elastica_quadrature  # noqa: E402

CONFIGS = [("svk", 2, "stress"), ("svk", 2, "strain"), ("svk", 3, "stress"),
           ("neo_hooke", 2, "stress"), ("neo_hooke", 2, "strain"), ("neo_hooke", 3, "stress")]
IDS = [f"{m}-{d}D-{p}" for m, d, p in CONFIGS]


def _D(dim: int, plane: str, nu: float) -> np.ndarray:
    return D_iso(nu) if dim == 3 else (D_plane_strain(nu) if plane == "strain" else D_plane_stress(nu))


def _cantilever_domain(dim: int) -> tuple[Domain, np.ndarray, np.ndarray]:
    """Small cantilever: clamped at x = 0, unit vertical load spread over the x = max face."""
    d = Domain((6, 3) if dim == 2 else (4, 2, 2), h=0.5, thickness=0.8)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - X[:, 0].max()) < 1e-9)
    F = np.zeros(d.ndof)
    F[dim * tip + 1] = -1.0 / len(tip)
    return d, fixed, F


def _grey(n: int, seed: int, lo: float = 0.05) -> np.ndarray:
    return lo + (1 - lo) * np.random.default_rng(seed).random(n)


# ============================================================================================== V1, V1b: small load
@pytest.mark.parametrize("material,dim,plane", CONFIGS, ids=IDS)
def test_v1_small_load_equals_linear(material, dim, plane):
    """V1: ‖u_NL − u_L‖/‖u_L‖ ≤ 1e-6 at max|∇u| ≤ 1e-8; V1b: compliance gradient relative ≤ 1e-6 (pre-registered).
    Energy interpolation on, grey design."""
    d, fixed, F = _cantilever_domain(dim)
    nu = 0.3
    lin = Elasticity(d, fixed, elasticity=_D(dim, plane, nu))
    E = _grey(lin.n_elements, 1)
    UL = lin.solve_numpy(E, F)[:, 0]
    grad_u = np.abs(UL).max() / d.h
    scale = 1e-8 / grad_u
    nl = NonlinearElasticity(d, fixed, nu=nu, material=material, plane=plane, rtol=1e-12,
                             energy_interpolation=EnergyInterpolation(E0=1.0))
    UN = nl.solve_numpy(E, scale * F)[:, 0] / scale
    rel = np.linalg.norm(UN - UL) / np.linalg.norm(UL)
    print(f"V1 {material} {dim}D {plane}: rel = {rel:.2e}")
    assert rel <= 1e-6

    Et = torch.tensor(E, requires_grad=True)
    cL = (torch.as_tensor(F[:, None]) * lin.solve(Et, F)).sum()
    (gL,) = torch.autograd.grad(cL, Et)
    Et2 = torch.tensor(E, requires_grad=True)
    cN = (torch.as_tensor(scale * F[:, None]) * nl.solve(Et2, scale * F)).sum() / scale ** 2
    (gN,) = torch.autograd.grad(cN, Et2)
    relg = float((gN - gL).abs().max() / gL.abs().max())
    print(f"V1b {material} {dim}D {plane}: gradient rel = {relg:.2e}")
    assert relg <= 1e-6


# ============================================================================================== V2: uniaxial tension
def _uniaxial_exact(material: str, dim: int, plane: str, nu: float, P: float) -> tuple[float, float]:
    """(λ1, λ2): axial and lateral stretch under the dead nominal stress P (E = 1), lateral faces free."""
    if material == "svk":
        Eeff, nueff = (1.0 / (1 - nu ** 2), nu / (1 - nu)) if (dim == 2 and plane == "strain") else (1.0, nu)
        lam1 = brentq(lambda l: Eeff * l * 0.5 * (l * l - 1) - P, 1.0, 3.0, xtol=1e-15, rtol=1e-15)
        E1 = 0.5 * (lam1 ** 2 - 1)
        return lam1, math.sqrt(1 + 2 * (-nueff * E1))
    kappa, mu = 1 / (3 * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    lateral3 = not (dim == 2 and plane == "strain")     # plane stress and 3-D: λ3 = λ2

    def S(l1: float, l2: float) -> tuple[float, float]:
        l3 = l2 if lateral3 else 1.0
        J = l1 * l2 * l3
        I1 = l1 ** 2 + l2 ** 2 + l3 ** 2
        Jm = J ** (-2 / 3)
        Si = [kappa * J * (J - 1) / li ** 2 + mu * Jm * (1 - I1 / (3 * li ** 2)) for li in (l1, l2)]
        return Si[0], Si[1]

    def lam2(l1: float) -> float:
        return brentq(lambda l2: S(l1, l2)[1], 0.3, 1.5, xtol=1e-15, rtol=1e-15)

    lam1 = brentq(lambda l1: l1 * S(l1, lam2(l1))[0] - P, 1.0, 3.0, xtol=1e-15, rtol=1e-15)
    return lam1, lam2(lam1)


@pytest.mark.parametrize("material,dim,plane", CONFIGS, ids=IDS)
def test_v2_uniaxial_tension_closed_form(material, dim, plane):
    """V2: homogeneous uniaxial tension under a dead load P = 0.3 E; stretches within 1e-9 (pre-registered)."""
    nu, P = 0.3, 0.3
    shape = (3, 2) if dim == 2 else (2, 2, 2)
    d = Domain(shape, h=1.0, thickness=0.5)
    X = d.node_coords()
    L = X[:, 0].max()
    fixed = [d.dofs(np.flatnonzero(X[:, 0] < 1e-9), 0)]
    origin = np.flatnonzero(np.all(np.abs(X) < 1e-9, axis=1))
    fixed.append(d.dofs(origin, list(range(1, dim))))
    if dim == 3:
        y_edge = np.flatnonzero((X[:, 0] < 1e-9) & (np.abs(X[:, 1] - X[:, 1].max()) < 1e-9) & (X[:, 2] < 1e-9))
        fixed.append(d.dofs(y_edge, 2))
    F = np.zeros(d.ndof)
    face = np.abs(X[:, 0] - L) < 1e-9
    # consistent nodal forces of a uniform traction on the bilinear face elements
    if dim == 2:
        ys = np.sort(np.unique(X[face, 1]))
        for a, b in zip(ys[:-1], ys[1:], strict=True):
            for y in (a, b):
                n = np.flatnonzero(face & (np.abs(X[:, 1] - y) < 1e-9))
                F[2 * n] += P * d.thickness * (b - a) / 2
    else:
        ys, zs = np.sort(np.unique(X[face, 1])), np.sort(np.unique(X[face, 2]))
        for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
            for z0, z1 in zip(zs[:-1], zs[1:], strict=True):
                for y in (y0, y1):
                    for z in (z0, z1):
                        n = np.flatnonzero(face & (np.abs(X[:, 1] - y) < 1e-9) & (np.abs(X[:, 2] - z) < 1e-9))
                        F[3 * n] += P * (y1 - y0) * (z1 - z0) / 4
    nl = NonlinearElasticity(d, np.concatenate(fixed), nu=nu, material=material, plane=plane, rtol=1e-12)
    U = nl.solve_numpy(np.ones(nl.n_elements), F)[:, 0].reshape(-1, dim)
    lam1_fe = 1 + U[face, 0].mean() / L
    top = np.abs(X[:, 1] - X[:, 1].max()) < 1e-9
    lam2_fe = 1 + U[top, 1].mean() / X[:, 1].max()
    lam1, lam2 = _uniaxial_exact(material, dim, plane, nu, P)
    e1, e2 = abs(lam1_fe - lam1), abs(lam2_fe - lam2)
    spread = np.ptp(U[face, 0])                         # homogeneous: every face node moves alike
    print(f"V2 {material} {dim}D {plane}: lambda1 = {lam1:.12f} (err {e1:.1e}), lambda2 = {lam2:.12f} "
          f"(err {e2:.1e}), face spread {spread:.1e}")
    assert e1 <= 1e-9 and e2 <= 1e-9


# ============================================================================================== V3, V3b: tangent
def _deformed_state(material, dim, plane, seed=3):
    d = Domain((4, 3) if dim == 2 else (2, 2, 2), h=1.0, thickness=0.6)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    nl = NonlinearElasticity(d, fixed, nu=0.3, material=material, plane=plane,
                             energy_interpolation=EnergyInterpolation(E0=1.0))
    rng = np.random.default_rng(seed)
    # large rigid rotation + stretch + random part: |∇u| of order 0.5
    th = 0.6
    R = np.eye(dim)
    R[:2, :2] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    G = R @ (np.eye(dim) + 0.2 * np.diag(rng.random(dim))) - np.eye(dim)
    # the base state need not satisfy the supports (δu does): zeroing the clamped DOFs after a 0.6 rad rotation
    # inverted the elements next to the clamp (J down to -0.81, first run), outside neo-Hooke's domain
    u = (X @ G.T).ravel() + 0.08 * d.h * rng.standard_normal(d.ndof)
    # moduli across the interpolation's transition (ρ̃ = E^(1/3) ≈ 0.1 at E = 1e-3)
    E = np.exp(rng.uniform(np.log(2e-4), 0.0, nl.n_elements))
    return nl, u, E


@pytest.mark.parametrize("material,dim,plane", CONFIGS, ids=IDS)
def test_v3_consistent_tangent_finite_difference(material, dim, plane):
    """V3: ‖K δu − central FD of R‖ / ‖K δu‖ ≤ 1e-6; V3b: ∂f_e/∂E_e by central FD, relative ≤ 1e-6 (pre-registered).
    Large rotation and stretch, energy interpolation across its transition."""
    nl, u, E = _deformed_state(material, dim, plane)
    rng = np.random.default_rng(11)
    du = np.zeros_like(u)
    du[nl.free] = rng.standard_normal(len(nl.free))
    du /= np.abs(du).max()
    eps = 1e-6
    K = nl.tangent(u, E)
    Kd = K @ du[nl.free]
    fd = (nl.internal_force(u + eps * du, E) - nl.internal_force(u - eps * du, E))[nl.free] / (2 * eps)
    rel = np.linalg.norm(Kd - fd) / np.linalg.norm(Kd)
    st = nl.element_state(u, E, need_K=False, need_dE=True)
    dE = 1e-6 * E
    fp = nl.element_state(u, E + dE, need_K=False)["f"]
    fm = nl.element_state(u, E - dE, need_K=False)["f"]
    fdE = (fp - fm) / (2 * dE[:, None])
    relE = np.abs(st["dfdE"] - fdE).max() / np.abs(fdE).max()
    print(f"V3 {material} {dim}D {plane}: tangent rel = {rel:.2e}; V3b dfdE rel = {relE:.2e}")
    assert rel <= 1e-6
    assert relE <= 1e-6


@pytest.mark.parametrize("material,dim,plane", CONFIGS, ids=IDS)
def test_strain_energy_gradient_is_internal_force(material, dim, plane):
    """Post hoc: f_int = ∂Π/∂u (central FD of the stored energy along a random direction), relative ≤ 1e-6."""
    nl, u, E = _deformed_state(material, dim, plane, seed=5)
    du = np.random.default_rng(2).standard_normal(len(u))
    eps = 1e-6
    fd = (nl.strain_energy(u + eps * du, E) - nl.strain_energy(u - eps * du, E)) / (2 * eps)
    an = float(nl.internal_force(u, E) @ du)
    assert abs(fd - an) <= 1e-6 * abs(an)


# ============================================================================================== V4, V5: cantilever
def _beam(n: int, material: str, alpha: float, slender: int = 50, **kw):
    """Cantilever L/H = slender, n elements through the thickness, ν = 0, plane stress, dead tip load α EI/L²."""
    H = 1.0
    h = H / n
    d = Domain((slender * n, n), h=h)
    X = d.node_coords()
    L = slender * H
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - L) < 1e-9)
    w = np.ones(len(tip))
    w[[0, -1]] = 0.5
    w /= w.sum()
    P = alpha * 1.0 * (H ** 3 / 12) / L ** 2
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -P * w
    nl = NonlinearElasticity(d, fixed, nu=0.0, material=material, rtol=1e-10, **kw)
    U = nl.solve_numpy(np.ones(nl.n_elements), F)[:, 0]
    u = -np.sum(w * U[2 * tip]) / L                 # shortening
    v = -np.sum(w * U[2 * tip + 1]) / L             # deflection
    return u, v, nl.last_reports[0]


def _orders(residuals: list[float], floor: float = 1e-11) -> list[float]:
    r = [x for x in residuals if x > floor]
    return [math.log(r[k + 1] / r[k]) / math.log(r[k] / r[k - 1]) for k in range(1, len(r) - 1)
            if r[k] < r[k - 1]]


V4_NH_REASON = (
    "pre-registered V4 not met for neo-Hooke (25 Nov 2026): at load factors 0.1 and 0.6-1.0 the largest three-point "
    "order of the relative residual per step is 1.79 and 0.77-0.94 (SVK: 2.3-18 at every step). The measure is unsound for this "
    "problem: in the slender beam the residual first jumps to 1e2-5e3 x |lambda F| and the three residuals of each "
    "step straddle the pre-asymptotic phase. The tangent is consistent (V3: <= 1.9e-10) and the error contraction from a "
    "nearby converged state is quadratic (post hoc test_newton_error_contraction_is_quadratic).")


@pytest.mark.parametrize("material", ["svk", pytest.param("neo_hooke", marks=pytest.mark.xfail(strict=True,
                                                                                             reason=V4_NH_REASON))])
def test_v4_newton_converges_quadratically(material):
    """V4: per load step with a usable triple (residuals above 1e-11), the largest observed order ≥ 1.8
    (pre-registered). Coarse cantilever, α = 2. Neo-Hooke does not meet it: strict xfail, see ``V4_NH_REASON``."""
    u, v, rep = _beam(4, material, 2.0, slender=50, load_steps=10)
    best = []
    for s in rep.steps:
        assert s.converged
        p = _orders(s.residuals)
        print(f"V4 {material} step {s.load_factor:.2f}: residuals " + " ".join(f"{r:.1e}" for r in s.residuals)
              + (f"  max p = {max(p):.2f}" if p else ""))
        if p:
            best.append(max(p))
    assert best, "no step had three residuals above the floor"
    assert min(best) >= 1.8


@pytest.mark.parametrize("material", ["svk", "neo_hooke"])
def test_newton_error_contraction_is_quadratic(material):
    """Post hoc (replaces V4 as the convergence-order evidence): Newton at the full load started from the converged
    state at 0.9 F, stocky cantilever (16 x 6, max|u| = 0.1 L, cond(K_T) ≈ 1.2e4). Errors e_k = |u_k − u*|∞/|u*|∞
    against the solution at rtol 1e-14; while e_{k+1} > 1e-11 (reference floor ~1e-12), C_k = e_{k+1}/e_k² ≤ 10 over
    at least two iterations and three decades. Measured 25 Nov 2026: e = 1.0e-1, 6.3e-4 (SVK) / 5.5e-4 (NH), 1.2e-6,
    ≤ 1e-12; C_k ≤ 3.7. A linearly converging iteration with rate r has C_k = r/e_k, growing without bound."""
    import scipy.sparse.linalg as spla
    d = Domain((16, 6), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - 16.0) < 1e-9)
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -0.02 / len(tip)
    nl = NonlinearElasticity(d, fixed, nu=0.3, material=material, rtol=1e-14)
    E = np.ones(nl.n_elements)
    us = nl.solve_numpy(E, F)[:, 0]
    u = nl.solve_numpy(E, 0.9 * F)[:, 0]
    errs = []
    for _ in range(5):
        errs.append(np.abs(u - us).max() / np.abs(us).max())
        R = (nl.internal_force(u, E) - F)[nl.free]
        u[nl.free] -= spla.spsolve(nl.tangent(u, E).tocsc(), R)
    C = [errs[k + 1] / errs[k] ** 2 for k in range(len(errs) - 1) if errs[k + 1] > 1e-11]
    print(f"contraction {material}: e = " + " ".join(f"{e:.2e}" for e in errs) + "; C = "
          + " ".join(f"{c:.2f}" for c in C))
    assert len(C) >= 2 and max(C) <= 10.0
    assert errs[0] / errs[len(C)] >= 1e3


ALPHAS = (1.0, 2.0)


@pytest.mark.slow
@pytest.mark.parametrize("material", ["svk", "neo_hooke"])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_v5_large_deflection_cantilever_elastica(material, alpha):
    """V5a: n = 16 errors ≤ 0.5 % (v) / 1.0 % (u); V5b: Richardson n = 8, 16 ≤ 0.1 % / 0.2 %; V5c: errors decrease
    n = 4 → 8 → 16 (pre-registered)."""
    u_ref, v_ref, _ = elastica_quadrature(alpha)
    res = {n: _beam(n, material, alpha)[:2] for n in (4, 8, 16)}
    eu = {n: (res[n][0] - u_ref) / u_ref for n in res}
    ev = {n: (res[n][1] - v_ref) / v_ref for n in res}
    uR = res[16][0] + (res[16][0] - res[8][0]) / 3
    vR = res[16][1] + (res[16][1] - res[8][1]) / 3
    print(f"V5 {material} alpha={alpha}: ref u/L={u_ref:.6f} v/L={v_ref:.6f}; "
          + "; ".join(f"n={n}: u {eu[n]:+.3%} v {ev[n]:+.3%}" for n in res)
          + f"; Richardson u {(uR - u_ref) / u_ref:+.3%} v {(vR - v_ref) / v_ref:+.3%}")
    assert abs(ev[16]) <= 5e-3 and abs(eu[16]) <= 1e-2
    assert abs(vR - v_ref) / v_ref <= 1e-3 and abs(uR - u_ref) / u_ref <= 2e-3
    assert abs(ev[4]) > abs(ev[8]) > abs(ev[16]) and abs(eu[4]) > abs(eu[8]) > abs(eu[16])


# ============================================================================================== V6: design gradient
def _study(domain: Domain, physics, F, objective, rmin=1.5) -> Study:
    chain = Chain([Embed(domain), DensityFilter(domain, rmin), Passive(domain)])
    return Study(domain, physics, F, chain, SIMP(domain, E0=1.0, penal=3.0, emin_ratio=1e-6), objective, [],
                 MMA())


def _fd_gradient(st: Study, x: np.ndarray, idx: np.ndarray, h: float = 1e-5) -> np.ndarray:
    out = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        out.append((st.evaluate(xp, gradients=False).f0 - st.evaluate(xm, gradients=False).f0) / (2 * h))
    return np.asarray(out)


@pytest.mark.parametrize("material,dim", [("svk", 2), ("neo_hooke", 2), ("svk", 3)])
def test_v6_end_compliance_gradient_finite_difference(material, dim):
    """V6: end-compliance (and a displacement response) gradient through filter + SIMP + Newton, central FD with
    step 1e-5: max|g_AD − g_FD| / max|g_FD| ≤ 1e-5, with max displacement ≥ 5 % of the span (pre-registered)."""
    if dim == 2:
        d = Domain((16, 6), h=1.0)
    else:
        d = Domain((8, 3, 3), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero((np.abs(X[:, 0] - X[:, 0].max()) < 1e-9) & (X[:, 1] < 1e-9))
    span = X[:, 0].max()
    F = np.zeros(d.ndof)
    load = 0.02 if dim == 2 else 0.05
    F[dim * tip + 1] = -load / len(tip)
    nl = NonlinearElasticity(d, fixed, nu=0.3, material=material, rtol=1e-12,
                             energy_interpolation=EnergyInterpolation(E0=1.0))
    x = 0.3 + 0.6 * np.random.default_rng(4).random(d.n_design)
    idx = np.random.default_rng(5).choice(d.n_design, 6, replace=False)
    for resp in (EndCompliance(), Displacement(np.array([dim * tip[0] + 1]), np.array([1.0]), 0, "tip")):
        st = _study(d, nl, F, resp)
        ev = st.evaluate(x)
        umax = np.abs(ev.U).max()
        fd = _fd_gradient(st, x, idx)
        rel = np.abs(ev.df0[idx] - fd).max() / np.abs(fd).max()
        print(f"V6 {material} {dim}D {resp.name}: max|u|/span = {umax / span:.3f}, rel = {rel:.2e}, "
              f"cutoffs {nl.cutoffs()}")
        assert umax >= 0.05 * span
        assert rel <= 1e-5


# ============================================================================================== failure is explicit
def test_newton_failure_raises_with_history():
    """Post hoc: an unreachable equilibrium (one iteration allowed, no halving room) raises, naming the load factor
    reached and the residuals; nothing is returned."""
    d, fixed, F = _cantilever_domain(2)
    nl = NonlinearElasticity(d, fixed, material="svk", load_steps=1, max_iter=1, min_step=0.6, rtol=1e-14)
    with pytest.raises(NonlinearConvergenceError, match="load factor 0 of 1"):
        nl.solve_numpy(np.ones(nl.n_elements), 1e-2 * F)


def test_zero_load_case_and_report():
    """Post hoc: a zero load column returns zero without iterations; reports carry the cutoff and residuals."""
    d, fixed, F = _cantilever_domain(2)
    nl = NonlinearElasticity(d, fixed, energy_interpolation=EnergyInterpolation())
    U = nl.solve_numpy(np.ones(nl.n_elements), np.column_stack([np.zeros(d.ndof), 1e-3 * F]))
    assert not U[:, 0].any() and U[:, 1].any()
    assert nl.last_reports[0].iterations == 0
    assert nl.last_reports[1].cutoff == 0.1 and nl.last_reports[1].relative_residual <= 1e-9


def test_rejects_unsupported_options():
    d, fixed, _ = _cantilever_domain(2)
    with pytest.raises(ValueError, match="Q4"):
        NonlinearElasticity(d, fixed, element="Q4IM")
    with pytest.raises(ValueError, match="svk only"):
        NonlinearElasticity(d, fixed, material="neo_hooke", elasticity=D_plane_stress(0.3))
    with pytest.raises(ValueError, match="material"):
        NonlinearElasticity(d, fixed, material="mooney")


# ============================================================================================== YAML
def _yaml(**physics) -> dict:
    return {
        "domain": {"type": "box", "shape": [16, 6], "h": 1.0},
        "material": {"E0": 1.0, "nu": 0.3},
        "physics": {"type": "nonlinear_elasticity", **physics},
        "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
        "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                                  "force": [0, -0.01]}]}],
        "objective": "end_compliance",
        "constraints": [{"response": "volume", "upper": 0.5}],
        "optimizer": {"type": "mma", "move": 0.1},
        "run": {"max_iter": 2},
    }


def test_yaml_builds_and_runs():
    st = study_from_dict(_yaml(material="neo_hooke"))
    assert isinstance(st.physics, NonlinearElasticity) and st.physics.material == "neo_hooke"
    assert st.physics.interpolation is not None and isinstance(st.objective, EndCompliance)
    res = st.run(max_iter=2)
    assert np.isfinite(res.objective) and res.U is not None


def test_yaml_end_compliance_needs_nonlinear_physics():
    cfg = _yaml()
    cfg["physics"] = "elasticity"
    with pytest.raises(ConfigError, match="nonlinear_elasticity"):
        study_from_dict(cfg)
    cfg2 = copy.deepcopy(_yaml(energy_interpolation=False))
    assert study_from_dict(cfg2).physics.interpolation is None


# ============================================================================================== V7: benchmark
@pytest.mark.slow
def test_v7_nonlinear_design_beats_linear_design_under_large_load():
    """V7a: C_NL(x_NL) ≤ 0.99 C_NL(x_L); V7b: C_L(x_L) ≤ 1.005 C_L(x_NL) (pre-registered). Clamped-clamped beam,
    mid-span load scaled so that the linear design's linear mid deflection is 10 % of the span; the problem and its
    parameters are ``BENCH`` of ``docs/olcumler/geometrik_dogrusal_olmama.py`` (the record is written by that script).
    Comparison with the numbers of Buhl, Pedersen & Sigmund (2000): source not read, NOT VERIFIED."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geometrik_dogrusal_olmama", Path(__file__).parents[2] / "docs" / "olcumler" / "geometrik_dogrusal_olmama.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.benchmark(save=False)
    assert out["V7a_ratio_C_NL"] <= 0.99
    assert out["V7b_ratio_C_L"] <= 1.005

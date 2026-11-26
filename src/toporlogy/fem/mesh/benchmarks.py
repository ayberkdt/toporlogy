"""Benchmark problems of the Poisson-ratio / locking report (docs/olcumler/fem_genel_onkayit.md §4).

* :func:`mms_case`: the manufactured solution of :mod:`toporlogy.fem.mesh.manufactured` on the unit square (plane
  strain) / cube with u = 0 on the whole boundary, for a general-mesh element, the mixed u-p formulation, or a
  regular-grid element (``"Q4IM"``, ``"H8IM"``, ``"grid:Q4"``, ``"grid:H8"`` through :class:`toporlogy.fem.elastic.
  Elasticity`, loads from the same consistent body force; grid L2 errors use the nodal (compatible) part).
* :func:`cantilever_case`: the cantilever with a parabolic end shear, closed-form plane-stress solution of Timoshenko
  & Goodier (transcribed and checked in ``tests/toporlogy/test_timoshenko_cantilever.py``, H1); both ends carry the
  exact tractions, three (3-D: six) DOFs hold the rigid motion, and the measure is R = U_h / U_exact with
  ``U_h = ½ fᵀu`` (≤ 1 for conforming elements). In 3-D the beam has width = depth and ν = 0, for which the plane
  solution is the exact 3-D solution.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from toporlogy.core.domain import Domain
from toporlogy.core.linear_elastic import (isotropic_moduli, isotropic_plane_strain, isotropic_plane_stress,
                                           isotropic_stiffness)
from toporlogy.fem.mesh.manufactured import Manufactured
from toporlogy.fem.mesh.mesh import Mesh, box, perturb, rectangle
from toporlogy.fem.mesh.mixed import MixedElasticity
from toporlogy.fem.mesh.problem import Dirichlet, ElasticProblem
from toporlogy.fem.mesh.reference import get

__all__ = ["moduli", "mms_case", "cantilever_case", "cantilever_energy", "MIXED"]

MIXED = {"TH:T6-T3": ("T6", "vertex"), "TH:Q9-Q4": ("Q9", "vertex"), "TH:T10-T4": ("T10", "vertex"),
         "Q4-P0": ("Q4", "P0")}
GRID = {"Q4IM": ("Q4IM", 2), "H8IM": ("H8IM", 3), "grid:Q4": ("Q4", 2), "grid:H8": ("H8", 3)}


def moduli(nu: float, fixed: str = "E") -> tuple[float, float, float]:
    """(E, μ, λ) with E = 1 (``fixed="E"``) or μ = G = 1 (``fixed="G"``); ν = 0.5 gives λ = ∞."""
    if fixed not in ("E", "G"):
        raise ValueError("fixed: E | G")
    if nu == 0.5:
        mu = 1.0 / 3.0 if fixed == "E" else 1.0
        return 3.0 * mu, mu, math.inf
    E = 1.0 if fixed == "E" else 2.0 * (1.0 + nu)
    m = isotropic_moduli(E=E, nu=nu)
    return E, m.G, m.lam


def _unit_mesh(element: str, n: int, distort: float, seed: int) -> Mesh:
    dim = get(element).dim
    m = rectangle(n, n, element=element) if dim == 2 else box(n, n, n, element=element)
    return perturb(m, distort / n, seed=seed) if distort else m


@dataclass
class _Timer:
    t0: float = 0.0

    def __enter__(self) -> _Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a: Any) -> None:
        self.dt = time.perf_counter() - self.t0


def mms_case(method: str, n: int, nu: float, fixed: str = "E", variant: str = "A", distort: float = 0.0,
             seed: int = 0, dim: int | None = None, plane: str = "strain") -> dict[str, Any]:
    """Errors of one method on the manufactured problem.

    ``method``: a general-mesh element (``Q4 Q9 T3 T6 H8 T4 T10``), a mixed pair (:data:`MIXED`), or a grid element
    (``Q4IM H8IM grid:Q4 grid:H8``). ``variant``: "A" (c = 0, divergence free, p = 0) or "B" (c = μ/λ, finite
    pressure). Returns L2 / energy errors (energy: displacement-only methods with finite λ), the pressure error of
    mixed methods (zero-mean comparison), sizes and times. ``plane="stress"`` (2-D displacement elements only):
    plane-stress D and the plane-stress λ* = 2λμ / (λ + 2μ) in the body force.
    """
    E, mu, lam = moduli(nu, fixed)
    if plane == "stress":
        if method in MIXED or method in GRID:
            raise ValueError("plane stress: general-mesh displacement elements only")
        lam = 2 * lam * mu / (lam + 2 * mu)
    elif plane != "strain":
        raise ValueError("plane: strain | stress")
    if method in MIXED:
        element, pspace = MIXED[method]
    elif method in GRID:
        element = GRID[method][0]
        pspace = ""
    else:
        element, pspace = method, ""
    d = get(element.replace("IM", "")).dim if dim is None else dim
    if variant == "A":
        ms = Manufactured(d, 0.0, 0.0)
    elif variant == "B":
        ms = Manufactured.incompressible_limit(d, mu, lam)
    else:
        raise ValueError("variant: A | B")
    out: dict[str, Any] = {"method": method, "n": n, "nu": nu, "fixed": fixed, "variant": variant, "E": E, "mu": mu,
                           "lam": lam, "distort": distort, "plane": plane if d == 2 else "3d"}

    def body(X: np.ndarray) -> np.ndarray:
        return ms.body_force(mu, X)

    if method in GRID:
        return _grid_mms(method, n, nu, E, mu, lam, ms, body, out)
    m = _unit_mesh(element, n, distort, seed)
    k = get(element).degree
    with _Timer() as ta:
        if method in MIXED:
            prob: Any = MixedElasticity(m, mu, lam, pressure=pspace)
            disp = prob.disp
        else:
            D = isotropic_stiffness(E, nu) if d == 3 else \
                (isotropic_plane_strain(E, nu) if plane == "strain" else isotropic_plane_stress(E, nu))
            prob = disp = ElasticProblem(m, D)
            _ = disp.K
    f = disp.body_force(body, degree=2 * k + 4)
    bc = Dirichlet.fixed(m.dofs(m.boundary_nodes()))
    with _Timer() as ts:
        s = prob.solve(f, bc)
    u = s.u[:, 0]
    if method in MIXED:
        err = ElasticProblem.errors(disp, u, ms.u, ms.grad)
        out["l2"] = err["l2"]
        out["energy"] = float("nan")
        out["p_error"] = prob.pressure_error(s.p[:, 0], ms.pressure, zero_mean=True) if variant == "B" else \
            float(np.sqrt(np.mean(s.p[:, 0] ** 2)))
        out["gauge"] = s.gauge
        out["ndof"] = int(disp.ndof + prob.n_p)
    else:
        err = disp.errors(u, ms.u, ms.grad)
        out.update(err)
        out["ndof"] = int(disp.ndof)
        fac = disp.factorization()
        out["fill_nnz"] = None if fac is None else fac.fill_nnz
    out["nnz"] = int(disp.K.nnz)
    out["t_assembly"] = ta.dt
    out["t_solve"] = ts.dt
    return out


def _grid_mms(method: str, n: int, nu: float, E: float, mu: float, lam: float, ms: Manufactured, body: Any,
              out: dict[str, Any]) -> dict[str, Any]:
    from toporlogy.fem.elastic import Elasticity
    from toporlogy.fem.elements import D_plane_strain, D_iso
    element, dim = GRID[method]
    dom = Domain((n,) * dim, h=1.0 / n)
    m = Mesh.from_domain(dom)
    Dunit = D_plane_strain(nu) if dim == 2 else D_iso(nu)
    lin = ElasticProblem(m, E * Dunit)
    f = lin.body_force(body, degree=6)
    bnd = m.dofs(m.boundary_nodes())
    with _Timer() as ts:
        el = Elasticity(dom, bnd, element=element, elasticity=Dunit)
        U = el.solve_numpy(np.full(dom.n_active, E * (dom.h if dim == 3 else 1.0)) / (dom.h if dim == 3 else 1.0),
                           f)
    err = lin.errors(U[:, 0], ms.u, ms.grad)
    out["l2"] = err["l2"]
    out["energy"] = err["energy"] if element in ("Q4", "H8") else float("nan")
    out["ndof"] = int(dom.ndof)
    out["t_solve"] = ts.dt
    return out


# ---------------------------------------------------------------------------------------------- cantilever
def _beam(P: float, D: float, L: float, E: float, nu: float) -> tuple[Any, Any, float]:
    I_ = D ** 3 / 12

    def u(X: np.ndarray) -> np.ndarray:
        x, y = X[:, 0], X[:, 1]
        ux = -P * y / (6 * E * I_) * ((6 * L - 3 * x) * x + (2 + nu) * (y ** 2 - D ** 2 / 4))
        uy = P / (6 * E * I_) * (3 * nu * y ** 2 * (L - x) + (4 + 5 * nu) * D ** 2 * x / 4 + (3 * L - x) * x ** 2)
        return np.column_stack([ux, uy] + ([np.zeros(len(x))] if X.shape[1] == 3 else []))

    def sigma(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = X[:, 0], X[:, 1]
        return -P * (L - x) * y / I_, P * (D ** 2 / 4 - y ** 2) / (2 * I_)
    return u, sigma, I_


def cantilever_energy(P: float, D: float, L: float, E: float, nu: float, width: float = 1.0) -> tuple[float, float]:
    """(total strain energy, shear part) of the exact plane-stress cantilever, per ``width``:
    bending P² L³ / (6 E I) and shear 0.6 P² L / (G A) with I = D³/12, A = D (per unit width)."""
    G = E / (2 * (1 + nu))
    Ib = D ** 3 / 12
    ub = P ** 2 * L ** 3 / (6 * E * Ib)
    us = 0.6 * P ** 2 * L / (G * D)
    return (ub + us) * width, us * width


def cantilever_case(method: str, slenderness: float, ny: int, nx: int | None = None, nu: float = 0.3,
                    dim: int = 2) -> dict[str, Any]:
    """R = U_h / U_exact of the end-loaded cantilever (depth D = 1, length L = slenderness), ny elements through
    the depth, nx along (default square elements, nx = slenderness · ny). 3-D: width = depth, nz = ny, ν must be 0.
    ``method``: general-mesh element, or ``Q4IM``/``grid:Q4``/``H8IM``/``grid:H8`` on the regular grid (square
    elements only)."""
    D, P, E = 1.0, 1.0, 1.0
    L = float(slenderness)
    nx = int(round(L * ny)) if nx is None else int(nx)
    if dim == 3 and nu != 0.0:
        raise ValueError("the 3-D cantilever reference is exact for ν = 0 only")
    u_ex, sig, I_ = _beam(P, D, L, E, nu)
    width = 1.0 if dim == 2 else D
    U_ex, U_sh = cantilever_energy(P, D, L, E, nu, width)
    grid = method in GRID
    element = GRID[method][0] if grid else method
    if grid:
        if abs(L * ny - nx) > 1e-9:
            raise ValueError("grid elements are squares: nx = slenderness · ny")
        h = D / ny
        shape = (nx, ny) if dim == 2 else (nx, ny, ny)
        dom = Domain(shape, h=h, origin=(0.0, -D / 2) if dim == 2 else (0.0, -D / 2, 0.0))
        m = Mesh.from_domain(dom)
    else:
        if dim == 2:
            m = rectangle(nx, ny, lx=L, ly=D, element=element, origin=(0.0, -D / 2))
        else:
            m = box(nx, ny, ny, lx=L, ly=D, lz=D, element=element, origin=(0.0, -D / 2, 0.0))
    Dm = isotropic_plane_stress(E, nu) if dim == 2 else isotropic_stiffness(E, nu)
    lin = ElasticProblem(m, Dm)
    zero = np.zeros

    def t_left(X: np.ndarray) -> np.ndarray:
        sx, txy = sig(X)
        return np.column_stack([-sx, -txy] + ([zero(len(X))] if dim == 3 else []))

    def t_right(X: np.ndarray) -> np.ndarray:
        sx, txy = sig(X)
        return np.column_stack([sx, txy] + ([zero(len(X))] if dim == 3 else []))
    f = lin.traction(m.facets_where(lambda X: np.abs(X[:, 0]) < 1e-9), t_left) + \
        lin.traction(m.facets_where(lambda X: np.abs(X[:, 0] - L) < 1e-9), t_right)
    X = m.points

    def node_at(p: tuple[float, ...]) -> int:
        used = m.used_nodes()
        return int(used[np.argmin(np.linalg.norm(X[used] - np.array(p), axis=1))])
    a = node_at((0.0, 0.0) + ((0.0,) if dim == 3 else ()))
    if dim == 2:
        b = node_at((0.0, D / 2))
        fixed = np.array([2 * a, 2 * a + 1, 2 * b])
    else:
        b = node_at((0.0, 0.0, D))
        c = node_at((0.0, D / 2, 0.0))
        fixed = np.array([3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * c])
    t0 = time.perf_counter()
    if grid:
        from toporlogy.fem.elastic import Elasticity
        from toporlogy.fem.elements import D_iso, D_plane_stress
        el = Elasticity(dom, fixed, element=element, elasticity=D_plane_stress(nu) if dim == 2 else D_iso(nu))
        u = el.solve_numpy(np.full(dom.n_active, E), f)[:, 0]
    else:
        u = lin.solve(f, Dirichlet.fixed(fixed)).u[:, 0]
    dt = time.perf_counter() - t0
    Uh = 0.5 * float(f @ u)
    return {"method": method, "dim": dim, "slenderness": L, "ny": ny, "nx": nx, "aspect": (L / nx) / (D / ny),
            "nu": nu, "R": Uh / U_ex, "shear_fraction": U_sh / U_ex, "ndof": int(len(u)), "t_solve": dt}

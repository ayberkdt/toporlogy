"""PR-09: independent verification and validation (docs/yol_haritasi_2026-11-18.md §3, row PR-09).

Unlike tests/toporlogy/test_regression.py, whose oracles are frozen copies of this repository's own earlier drivers,
the references here do not come from this repository:

1. Top88 and Top3D: NumPy transcriptions of the published MATLAB codes (tests/toporlogy/published_codes.py), run on
   the same problem as toporlogy; first objective and sensitivity of fixed designs, then the optimisation history.
2. An end-loaded plane-stress cantilever with an exact polynomial elasticity solution, under mesh refinement.
3. Free vibration of a square and a cube with slip walls (normal displacement fixed on every face): exact
   eigenvalues, several of them repeated.

Every tolerance below was written down before the comparison was first run (pre-registration file of this PR,
2026-11-24 19:24 +03:00, in the session notes); none was changed after seeing a result. Each docstring also says why
the two sides can differ at all. The measured numbers are printed (run with ``-s``).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from published_codes import Top3D, Top88, lk_H8
from toporlogy import OC, SIMP, Compliance, Constraint, Domain, Elasticity, Study, Volume, standard_chain
from toporlogy.configuration.build import load_config
from toporlogy.fem.elements import element_matrix
from toporlogy.fem.modal import Modal, find_clusters

REPO = Path(__file__).resolve().parents[2]
SEED = 20261124


# ================================================================================ helpers: layouts and constraints
def _g_scale(c: Constraint) -> float:
    """The factor ``s`` of ``g = (response − upper)·s`` (Constraint.g)."""
    return c.scale if c.scale is not None else (1.0 / abs(c.upper) if c.upper != 0 else 1.0)


def _from_matlab_2d(a: np.ndarray) -> np.ndarray:
    """top88 field (nely, nelx), row 0 at the top → toporlogy grid (nx, ny), j = 0 at the bottom."""
    return np.ascontiguousarray(np.asarray(a)[::-1, :].T)


def _from_matlab_3d(a: np.ndarray) -> np.ndarray:
    """top3d field (nely, nelx, nelz), row 0 at the top → toporlogy grid (nx, ny, nz)."""
    return np.ascontiguousarray(np.asarray(a)[::-1].transpose(1, 0, 2))


def _dof_set(xyz: np.ndarray, dofs: np.ndarray, dim: int) -> set[tuple[int, ...]]:
    """{(coordinates..., direction)} of DOF numbers whose node coordinates are ``xyz[node]``."""
    return {(*(int(round(v)) for v in xyz[d // dim]), int(d % dim)) for d in np.asarray(dofs, np.int64)}


def _load_map(xyz: np.ndarray, F: np.ndarray, dim: int) -> dict[tuple[int, ...], float]:
    nz = np.flatnonzero(F)
    return {(*(int(round(v)) for v in xyz[d // dim]), int(d % dim)): float(F[d]) for d in nz}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# ================================================================================ 1. Top88
def _mbb_study() -> Study:
    return load_config(REPO / "examples" / "mbb2d.yaml")


def _top88_for(study: Study) -> Top88:
    """The reference with the parameters read from the YAML example (not retyped)."""
    cfg = study.config
    assert cfg is not None
    nelx, nely = cfg["domain"]["shape"]
    rmin = [c for c in cfg["chain"] if c["type"] == "density_filter"][0]["rmin"]
    return Top88(nelx, nely, cfg["constraints"][0]["upper"], cfg["interpolation"]["penal"], rmin,
                 nu=cfg["material"]["nu"])


def test_top88_same_problem_as_mbb2d_yaml():
    """examples/mbb2d.yaml is top88's problem: same loads, supports, element matrix, filter radius and OC settings.

    Loads and supports are compared as sets of (node coordinates, direction) because the two codes number nodes
    differently (top88: column-major from the top-left node; toporlogy: x-major from the bottom-left node).
    Pre-registered: exact equality (integers, the unit load, the 8×8 matrix to 1e-15 in absolute value).

    Note (not a difference in behaviour): toporlogy's Q4 matrix ``ke_q4`` is itself typed in from top88's A11/A12/
    B11/B12, so the element-matrix check in 2-D is not independent; the 3-D test below compares two independently
    written hexahedra.
    """
    study = _mbb_study()
    ref = _top88_for(study)
    d = study.domain
    xy_t = d.node_coords()
    xy_r = ref.node_xy(np.arange((ref.nelx + 1) * (ref.nely + 1)))
    assert _dof_set(xy_t, study.physics.fixed, 2) == _dof_set(xy_r, ref.fixeddofs, 2)
    assert _load_map(xy_t, study.F[:, 0], 2) == _load_map(xy_r, ref.F, 2)
    assert np.max(np.abs(element_matrix(2, "Q4", 0.3) - ref.KE)) <= 1e-15
    assert study.interp.emin_ratio == ref.Emin and study.interp.penal == ref.penal
    assert isinstance(study.optimizer, OC) and study.optimizer.move == 0.2 and study.optimizer.bisection_tol == 1e-3
    assert study.optimizer.l1 == 0.0 and study.optimizer.l2 == 1e9 and study.optimizer.dv_scale == d.n_active


@pytest.mark.parametrize("design", ["uniform", "random"])
def test_top88_fixed_design_objective_and_sensitivity(design):
    """Objective, sensitivity and volume gradient of one fixed design: toporlogy (autograd through the filter and the
    adjoint solve) against top88's hand-coded ft = 2 chain rule ``dc(:) = H*(dc(:)./Hs)``.

    Designs: x = 0.5 everywhere, and a seeded random x in [0.2, 1]. Pre-registered tolerances:
    |c_T − c_R|/c_R ≤ 1e-9; max|dc_T − dc_R| / max|dc_R| ≤ 1e-8; max|n·dg_T/s − dv_R| ≤ 1e-12 (s: constraint scale);
    max|x̄_T − x̄_R| ≤ 1e-13.

    Why they can differ at all: rounding only. Different sparse solvers (SuperLU / PARDISO against SciPy's spsolve on
    top88's symmetrised (K + Kᵀ)/2), different summation order in assembly and filtering.
    """
    study = _mbb_study()
    ref = _top88_for(study)
    if design == "uniform":
        x = np.full(ref.shape_f, ref.volfrac)
    else:
        x = np.random.default_rng(SEED).uniform(0.2, 1.0, ref.shape_f)
    r = ref.evaluate(x)
    ev = study.evaluate(_from_matlab_2d(x).ravel())
    n = study.domain.n_active
    c_rel = _rel(ev.f0, float(r["c"]))
    dc_R = _from_matlab_2d(r["dc"]).ravel()
    dc_rel = float(np.max(np.abs(ev.df0 - dc_R)) / np.max(np.abs(dc_R)))
    dv_abs = float(np.max(np.abs(n * ev.dg[0] / _g_scale(study.constraints[0]) - _from_matlab_2d(r["dv"]).ravel())))
    x_abs = float(np.max(np.abs(ev.xphys - _from_matlab_2d(r["xPhys"]))))
    print(f"\n  top88 fixed {design}: c={ev.f0:.10g} c_rel={c_rel:.2e} dc_rel={dc_rel:.2e} dv_abs={dv_abs:.2e} "
          f"x_abs={x_abs:.2e}")
    assert c_rel <= 1e-9
    assert dc_rel <= 1e-8
    assert dv_abs <= 1e-12
    assert x_abs <= 1e-13


def test_top88_history_first_50_iterations():
    """Optimisation history over the first 50 OC iterations from top88's uniform start x = V*.

    Pre-registered: max_k |c_T,k − c_R,k| / c_R,k ≤ 1e-6 and max|x̄_T − x̄_R| ≤ 1e-5 after iteration 50 (equal
    numbers of iterations).

    Algorithmic differences found when reading both codes, and why none changes the iterates beyond rounding:

    * top88 starts with ``xPhys = x`` unfiltered; toporlogy filters x = 0.5. The filter of a constant field is that
      constant up to rounding (normalised weights).
    * top88 tests ``while (l2−l1)/(l1+l2) > 1e-3`` before each bisection step, toporlogy after it (and stops on
      ``< tol``); both compute at least one step and end on the same bracket except when the gap equals 1e-3 exactly.
    * The YAML constraint is mean(x̄) − 0.5 ≤ 0 scaled by s = 1/0.5 = 2 and OC multiplies ∂g/∂x by n: the bisection
      therefore works on λ_T = λ_R / 2 inside [0, 1e9], i.e. λ_R ∈ [0, 2e9]. That bracket is top88's [0, 1e9] plus
      one extra halving whenever λ_R < 1e9, so the dyadic midpoints and the stopping bracket are the same.
    * toporlogy clamps ``max(−df0, 0)`` and ``max(dv, 1e-30)`` and caps the bisection at 80 halvings; compliance
      sensitivities are ≤ 0 and dv > 0 here, and about 40 halvings are needed, so these are inactive.
    """
    study = _mbb_study()
    ref = _top88_for(study)
    iters = 50
    old = ref.optimise(max_loop=iters)
    new = study.run(max_iter=iters, tol=0.01)
    c_R, c_T = np.asarray(old["compliance"]), np.asarray(new.history["f0"])
    assert len(c_R) == len(c_T) == iters
    c_rel = float(np.max(np.abs(c_T - c_R) / c_R))
    x_abs = float(np.max(np.abs(new.xphys - _from_matlab_2d(old["xPhys"]))))
    print(f"\n  top88 50 it: c_rel={c_rel:.2e} x_abs={x_abs:.2e} c_50={c_T[-1]:.8f}")
    assert c_rel <= 1e-6
    assert x_abs <= 1e-5


@pytest.mark.slow
def test_top88_history_to_convergence():
    """Both codes run to their stopping rule ``change ≤ 0.01`` (top88's ``while change > 0.01``; the YAML's run.tol
    0.01, max_iter 300; the reference is capped at 300 too).

    Pre-registered: the same number of iterations; final compliance |c_T − c_R|/c_R ≤ 1e-5; max|x̄_T − x̄_R| ≤ 1e-3.
    Differences: as in the 50-iteration test (rounding only); the looser bound allows for rounding growth over
    many more iterations.
    """
    study = _mbb_study()
    ref = _top88_for(study)
    run = study.config["run"]
    old = ref.optimise(max_loop=int(run["max_iter"]))
    new = study.run(max_iter=int(run["max_iter"]), tol=float(run["tol"]))
    c_R, c_T = np.asarray(old["compliance"]), np.asarray(new.history["f0"])
    c_rel = _rel(float(c_T[-1]), float(c_R[-1]))
    x_abs = float(np.max(np.abs(new.xphys - _from_matlab_2d(old["xPhys"]))))
    print(f"\n  top88 converged: iterations R={len(c_R)} T={len(c_T)} c_R={c_R[-1]:.8f} c_T={c_T[-1]:.8f} "
          f"c_rel={c_rel:.2e} x_abs={x_abs:.2e}")
    assert len(c_R) == len(c_T)
    assert c_rel <= 1e-5
    assert x_abs <= 1e-3


# ================================================================================ 2. Top3D
TOP3D = dict(nelx=24, nely=8, nelz=4, volfrac=0.3, penal=3.0, rmin=1.5)


def _top3d_pair(scale: float | None = 1.0) -> tuple[Top3D, Study]:
    """top3d(24, 8, 4, 0.3, 3, 1.5) and the same problem built with toporlogy's Python API.

    top3d: face x = 0 clamped; −1 in y on each of the nelz + 1 nodes of the edge x = nelx, y = 0 (not normalised).
    ``scale`` is the volume constraint's scale s; 1 with ``dv_scale = n`` is top3d's ``Σx̄ ≤ V*·n`` (see
    :func:`test_oc_bisection_depends_on_the_constraint_scale` for why this matters).
    """
    p = TOP3D
    ref = Top3D(p["nelx"], p["nely"], p["nelz"], p["volfrac"], p["penal"], p["rmin"])
    d = Domain((p["nelx"], p["nely"], p["nelz"]))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    tip = d.nodes_where(lambda X: (X[:, 0] == p["nelx"]) & (X[:, 1] == 0))
    F = d.load_vector()
    F[d.dofs(tip, 1), 0] = -1.0
    study = Study(d, Elasticity(d, fixed, nu=0.3, element="H8"), F, standard_chain(d, p["rmin"], projection=False),
                  SIMP(d, penal=p["penal"], emin_ratio=1e-9), Compliance(),
                  [Constraint(Volume(), p["volfrac"], scale=scale)],
                  OC(move=0.2, bisection_tol=1e-3, dv_scale=d.n_active))
    return ref, study


def test_top3d_same_problem_and_hexahedron():
    """Same loads and supports (as coordinate sets), and the element matrix: top3d's closed-form ``lk_H8`` (1/144
    erratum included) against toporlogy's 2×2×2 Gauss-integrated H8, both on the unit cube with ν = 0.3.

    Pre-registered: loads and supports equal as sets. The hexahedron comparison was added after the pre-registration
    file but before any run: max|KE_T − KE_R| ≤ 1e-12 (rounding of the exact rational entries; both claim the exact
    trilinear stiffness, in the same corner order (0,0,0), (1,0,0), (1,1,0), (0,1,0), then z = 1).
    """
    ref, study = _top3d_pair()
    d = study.domain
    xyz_r = ref.node_xyz(np.arange((ref.nelx + 1) * (ref.nely + 1) * (ref.nelz + 1)))
    assert _dof_set(d.node_coords(), study.physics.fixed, 3) == _dof_set(xyz_r, ref.fixeddofs, 3)
    assert _load_map(d.node_coords(), study.F[:, 0], 3) == _load_map(xyz_r, ref.F, 3)
    ke_abs = float(np.max(np.abs(element_matrix(3, "H8", 0.3) - lk_H8(0.3))))
    print(f"\n  top3d lk_H8 vs H8: max abs {ke_abs:.2e}")
    assert ke_abs <= 1e-12


@pytest.mark.parametrize("design", ["uniform", "random"])
def test_top3d_fixed_design_objective_and_sensitivity(design):
    """As the top88 fixed-design test, on top3d's cantilever. Pre-registered tolerances (the same four):
    c ≤ 1e-9 relative; dc ≤ 1e-8 relative to max|dc|; n·dg/s against dv ≤ 1e-12; x̄ ≤ 1e-13.
    Differences: rounding only (solver, summation order, the two element matrices' rounding).
    """
    ref, study = _top3d_pair()
    if design == "uniform":
        x = np.full(ref.shape_f, ref.volfrac)
    else:
        x = np.random.default_rng(SEED).uniform(0.2, 1.0, ref.shape_f)
    r = ref.evaluate(x)
    ev = study.evaluate(_from_matlab_3d(x).ravel())
    n = study.domain.n_active
    c_rel = _rel(ev.f0, float(r["c"]))
    dc_R = _from_matlab_3d(r["dc"]).ravel()
    dc_rel = float(np.max(np.abs(ev.df0 - dc_R)) / np.max(np.abs(dc_R)))
    dv_abs = float(np.max(np.abs(n * ev.dg[0] / _g_scale(study.constraints[0]) - _from_matlab_3d(r["dv"]).ravel())))
    x_abs = float(np.max(np.abs(ev.xphys - _from_matlab_3d(r["xPhys"]))))
    print(f"\n  top3d fixed {design}: c={ev.f0:.10g} c_rel={c_rel:.2e} dc_rel={dc_rel:.2e} dv_abs={dv_abs:.2e} "
          f"x_abs={x_abs:.2e}")
    assert c_rel <= 1e-9
    assert dc_rel <= 1e-8
    assert dv_abs <= 1e-12
    assert x_abs <= 1e-13


def test_top3d_history_first_30_iterations():
    """First 30 OC iterations of top3d against toporlogy. Pre-registered: c_k ≤ 1e-6 relative for every k,
    max|x̄_T − x̄_R| ≤ 1e-5 after iteration 30. Differences: as for top88 (top3d's OC loop is top88's verbatim).

    Record of the first run (2026-11-24): with the constraint's default scale s = 1/V* = 1/0.3 this test FAILED,
    c_rel = 1.90e-3 (first above 1e-9 at iteration 2, the first OC update), max|Δx̄| = 1.24e-3. Cause established by
    experiment, see :func:`test_oc_bisection_depends_on_the_constraint_scale`. The comparison now uses s = 1, which
    is top3d's own constraint Σx̄ ≤ V*·n; the tolerances are the pre-registered ones, unchanged.
    """
    ref, study = _top3d_pair()
    iters = 30
    old = ref.optimise(max_loop=iters)
    new = study.run(max_iter=iters, tol=0.01)
    c_R, c_T = np.asarray(old["compliance"]), np.asarray(new.history["f0"])
    assert len(c_R) == len(c_T) == iters
    c_rel = float(np.max(np.abs(c_T - c_R) / c_R))
    x_abs = float(np.max(np.abs(new.xphys - _from_matlab_3d(old["xPhys"]))))
    print(f"\n  top3d 30 it: c_rel={c_rel:.2e} x_abs={x_abs:.2e} c_30={c_T[-1]:.8f}")
    assert c_rel <= 1e-6
    assert x_abs <= 1e-5


def test_oc_bisection_depends_on_the_constraint_scale():
    """Finding of this PR (post hoc: found by the failed first run of the 30-iteration test, not pre-registered).

    toporlogy's OC bisects the multiplier of the *scaled* constraint, λ_T = λ_R / s, in the fixed bracket [0, 1e9]
    and stops at a relative bracket width of 1e-3; the returned λ is the last midpoint, so it is determined only to
    that width, and *which* point of the bracket comes out depends on s. When s is a power of two, the bracket
    [0, 1e9·s] in top3d's units is top3d's bracket plus or minus whole halvings, the midpoints fall on the same
    dyadic grid and the iterates agree to rounding; otherwise they differ by up to the bisection tolerance from the
    first update on. Measured on 30 iterations: s ∈ {0.5, 1, 2, 4} → c_rel 9.9e-12; s = 3 → 2.39e-3; s = 1/0.3 (the
    default 1/V*, also what a YAML ``{response: volume, upper: 0.3}`` gets, e.g. examples/cantilever3d.yaml) →
    1.90e-3. Both are valid OC iterations within the bisection tolerance; "the same as top3d / top88" holds to
    rounding only for power-of-two scales (top88's 1/0.5 = 2 is one).

    This test pins the behaviour on 5 iterations: s = 2 agrees ≤ 1e-6, s = 1/0.3 differs > 1e-6. A change that makes
    the bisection independent of s should turn the second assertion and must update this docstring.
    """
    ref = _top3d_pair()[0]
    c_R = np.asarray(ref.optimise(max_loop=5)["compliance"])
    rel = {}
    for s in (2.0, None):
        c_T = np.asarray(_top3d_pair(scale=s)[1].run(max_iter=5, tol=0.01).history["f0"])
        rel[s] = float(np.max(np.abs(c_T - c_R) / c_R))
    print(f"\n  OC bisection vs constraint scale, 5 it: s=2 c_rel={rel[2.0]:.2e}; s=1/0.3 c_rel={rel[None]:.2e}")
    assert rel[2.0] <= 1e-6
    assert rel[None] > 1e-6


@pytest.mark.slow
def test_top3d_history_to_convergence():
    """top3d's own stopping rule (change ≤ tolx = 0.01, maxloop = 200) on both sides. Pre-registered: the same
    number of iterations; final c ≤ 1e-5 relative; max|x̄_T − x̄_R| ≤ 1e-3."""
    ref, study = _top3d_pair()
    old = ref.optimise(max_loop=200, tolx=0.01)
    new = study.run(max_iter=200, tol=0.01)
    c_R, c_T = np.asarray(old["compliance"]), np.asarray(new.history["f0"])
    c_rel = _rel(float(c_T[-1]), float(c_R[-1]))
    x_abs = float(np.max(np.abs(new.xphys - _from_matlab_3d(old["xPhys"]))))
    print(f"\n  top3d converged: iterations R={len(c_R)} T={len(c_T)} c_R={c_R[-1]:.8f} c_T={c_T[-1]:.8f} "
          f"c_rel={c_rel:.2e} x_abs={x_abs:.2e}")
    assert len(c_R) == len(c_T)
    assert c_rel <= 1e-5
    assert x_abs <= 1e-3


# ================================================================================ 3. Elasticity: exact cantilever
# Plane stress, 0 ≤ x ≤ L, −c ≤ y ≤ c, unit thickness; shear load P on the end x = 0 (parabolic), the end x = L
# carries the reaction tractions; y = ±c traction free. Stresses σx = −P x y / I, σy = 0, τxy = −P (c² − y²) / (2I),
# I = 2c³/3. Displacements integrated from Hooke's law by hand (this file; checked below, not quoted from a book):
#   u0 = −P x² y/(2EI) − ν P y³/(6EI) + P y³/(6IG) − P c² y/(2IG),   v0 = ν P x y²/(2EI) + P x³/(6EI)
# plus a rigid motion fixing u = v = 0 at (L, 0) and u = 0 at (L, c), the three DOFs the FE model fixes.
CANT = dict(L=4.0, c=0.5, E=1.0, nu=0.3, P=1.0)


class _Cantilever:
    def __init__(self, L: float, c: float, E: float, nu: float, P: float) -> None:
        self.L, self.c, self.E, self.nu, self.P = L, c, E, nu, P
        self.G = E / (2 * (1 + nu))
        self.I = 2 * c ** 3 / 3
        self.a = -self.u0(L, 0.0)
        self.th = (self.u0(L, c) + self.a) / c
        self.b = -self.v0(L, 0.0) - self.th * L

    def u0(self, x, y):
        P, E, I, G, nu, c = self.P, self.E, self.I, self.G, self.nu, self.c
        return -P * x ** 2 * y / (2 * E * I) - nu * P * y ** 3 / (6 * E * I) + P * y ** 3 / (6 * I * G) \
            - P * c ** 2 * y / (2 * I * G)

    def v0(self, x, y):
        P, E, I, nu = self.P, self.E, self.I, self.nu
        return nu * P * x * y ** 2 / (2 * E * I) + P * x ** 3 / (6 * E * I)

    def u(self, x, y):
        return self.u0(x, y) + self.a - self.th * y

    def v(self, x, y):
        return self.v0(x, y) + self.b + self.th * x

    def stress(self, x, y):
        sx = -self.P * x * y / self.I
        return sx, 0.0 * x * y, -self.P * (self.c ** 2 - y ** 2) / (2 * self.I)

    def traction(self, x_face: float, y):
        """Traction on the end face x = x_face (outward normal −x at x = 0, +x at x = L)."""
        sx, _, txy = self.stress(x_face, y)
        sgn = -1.0 if x_face == 0.0 else 1.0
        return sgn * sx, sgn * txy


def _d4(f: Callable[[float], float], t: float, h: float = 1e-2) -> float:
    """Fourth-order central difference; exact up to rounding for polynomials of degree ≤ 4."""
    return (-f(t + 2 * h) + 8 * f(t + h) - 8 * f(t - h) + f(t - 2 * h)) / (12 * h)


_GX, _GW = np.polynomial.legendre.leggauss(3)          # exact for degree ≤ 5


def test_cantilever_exact_solution_satisfies_the_field_equations():
    """The reference is established, not quoted: at 49 points the fields satisfy Hooke's law (strains from the
    displacements = compliance × stresses), equilibrium, traction-free long edges and the load resultants; the
    boundary work equals twice the strain energy (Clapeyron). Pre-registered: every residual ≤ 1e-10 relative to the
    field's scale; boundary work against 2 × strain energy ≤ 1e-12 relative.
    """
    s = _Cantilever(**CANT)
    L, c, E, nu, G = s.L, s.c, s.E, s.nu, s.G
    eps_scale = s.P * L * c / (E * s.I)              # largest bending strain
    div_scale = s.P * L / s.I                        # largest stress / depth scale c
    worst = 0.0
    for x in np.linspace(0.3, L - 0.3, 7):
        for y in np.linspace(-c + 0.05, c - 0.05, 7):
            sx, sy, txy = s.stress(x, y)
            ex = _d4(lambda t: s.u(t, y), x)
            ey = _d4(lambda t: s.v(x, t), y)
            gxy = _d4(lambda t: s.u(x, t), y) + _d4(lambda t: s.v(t, y), x)
            hooke = max(abs(ex - (sx - nu * sy) / E), abs(ey - (sy - nu * sx) / E), abs(gxy - txy / G))
            div_x = _d4(lambda t: s.stress(t, y)[0], x) + _d4(lambda t: s.stress(x, t)[2], y)
            div_y = _d4(lambda t: s.stress(t, y)[2], x) + _d4(lambda t: s.stress(x, t)[1], y)
            worst = max(worst, hooke / eps_scale, abs(div_x) / div_scale, abs(div_y) / div_scale)
    free = max(abs(v) for x in np.linspace(0, L, 9) for y in (-c, c) for v in s.stress(x, y)[1:])
    y_g, w_g = c * _GX, c * _GW
    res0 = np.sum(w_g * s.traction(0.0, y_g)[1]) - s.P
    resL_y = np.sum(w_g * s.traction(L, y_g)[1]) + s.P
    resL_x = np.sum(w_g * s.traction(L, y_g)[0])
    momL = np.sum(w_g * s.traction(L, y_g)[0] * y_g) + s.P * L    # moment of the end-x=L tractions about (L, 0)
    work = sum(np.sum(w_g * (tx * s.u(xf, y_g) + ty * s.v(xf, y_g)))
               for xf in (0.0, L) for tx, ty in [s.traction(xf, y_g)])
    xg, wx = L / 2 * (_GX + 1), L / 2 * _GW
    X, Y = np.meshgrid(xg, y_g, indexing="ij")
    W = np.outer(wx, w_g)
    sxx, _, txy = s.stress(X, Y)
    energy = float(np.sum(W * (sxx ** 2 / E + txy ** 2 / G)) / 2)
    print(f"\n  cantilever exact field: Hooke/equilibrium worst rel {worst:.2e}, free-edge {free:.1e}, "
          f"resultants {abs(res0):.1e} {abs(resL_y):.1e} {abs(resL_x):.1e} {abs(momL):.1e}, "
          f"work/(2U) - 1 = {work / (2 * energy) - 1:.1e}")
    assert worst <= 1e-10
    assert free <= 1e-10 and max(abs(res0), abs(resL_y), abs(resL_x), abs(momL) / L) <= 1e-10
    assert abs(work / (2 * energy) - 1) <= 1e-12


def _cantilever_fe(m: int) -> dict[str, float]:
    """Q4 solution on (4m) × m square elements of size 1/m; consistent nodal loads of the exact end tractions."""
    s = _Cantilever(**CANT)
    L, c = s.L, s.c
    d = Domain((int(round(4 * m)), m), h=1.0 / m, origin=(0.0, -c))
    X = d.node_coords()
    tol = 1e-9 / m
    F = d.load_vector()
    for xf in (0.0, L):
        nodes = d.nodes_where(lambda Z, xf=xf: np.abs(Z[:, 0] - xf) < tol)
        nodes = nodes[np.argsort(X[nodes, 1])]
        for n0, n1 in zip(nodes[:-1], nodes[1:], strict=True):
            y0, y1 = X[n0, 1], X[n1, 1]
            yg = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * _GX
            wg = 0.5 * (y1 - y0) * _GW
            tx, ty = s.traction(xf, yg)
            for node, N in ((n0, (y1 - yg) / (y1 - y0)), (n1, (yg - y0) / (y1 - y0))):
                F[2 * node, 0] += np.sum(wg * tx * N)
                F[2 * node + 1, 0] += np.sum(wg * ty * N)
    A = d.nodes_where(lambda Z: (np.abs(Z[:, 0] - L) < tol) & (np.abs(Z[:, 1]) < tol))
    B = d.nodes_where(lambda Z: (np.abs(Z[:, 0] - L) < tol) & (np.abs(Z[:, 1] - c) < tol))
    tip = d.nodes_where(lambda Z: (np.abs(Z[:, 0]) < tol) & (np.abs(Z[:, 1]) < tol))
    assert len(A) == len(B) == len(tip) == 1
    fixed = np.concatenate([d.dofs(A), d.dofs(B, 0)])
    phys = Elasticity(d, fixed, nu=s.nu)
    U = phys.solve_numpy(np.full(d.n_active, s.E), F)[:, 0]
    ue, ve = s.u(X[:, 0], X[:, 1]), s.v(X[:, 0], X[:, 1])
    err = np.hypot(U[0::2] - ue, U[1::2] - ve)
    # exact compliance: work of the exact tractions on the exact displacements (degree 5 → 3-point Gauss exact)
    yg, wg = c * _GX, c * _GW
    c_exact = sum(float(np.sum(wg * (tx * s.u(xf, yg) + ty * s.v(xf, yg))))
                  for xf in (0.0, L) for tx, ty in [s.traction(xf, yg)])
    t = int(tip[0])
    return {"c_h": float(F[:, 0] @ U), "c": c_exact, "tip_h": float(U[2 * t + 1]), "tip": float(ve[t]),
            "max_err": float(err.max() / np.hypot(ue, ve).max())}


def test_cantilever_q4_converges_to_the_exact_solution():
    """Mesh refinement m = 2, 4, 8, 16, 32 elements per unit length (L = 4, depth 1: up to 128 × 32 Q4).

    Pre-registered: (ii) c_h ≤ c_exact on every mesh (relative slack 1e-12; displacement FE with work-equivalent
    loads is too stiff); (iii) observed orders log2(e_m / e_2m) on the two finest pairs ≥ 1.8 for the energy error
    (c − c_h)/c, the tip deflection error at (0, 0) and the max nodal displacement error (expected 2 for bilinear
    elements and a smooth solution); (iv) tip deflection relative error on the finest mesh ≤ 2e-2.
    Sources of difference: discretisation only (the loads are exact for the FE space, the supports match the exact
    solution's gauge).
    """
    ms = [2, 4, 8, 16, 32]
    rows = [_cantilever_fe(m) for m in ms]
    e_c = [(r["c"] - r["c_h"]) / r["c"] for r in rows]
    e_tip = [abs(r["tip_h"] - r["tip"]) / abs(r["tip"]) for r in rows]
    e_max = [r["max_err"] for r in rows]
    for m, r, a, b, e in zip(ms, rows, e_c, e_tip, e_max, strict=True):
        print(f"\n  cantilever m={m:2d}: c_h={r['c_h']:.8f} c={r['c']:.8f} e_c={a:.3e} tip_h={r['tip_h']:.6f} "
              f"tip={r['tip']:.6f} e_tip={b:.3e} e_max={e:.3e}", end="")
    order = {name: [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
             for name, e in (("energy", e_c), ("tip", e_tip), ("max", e_max))}
    print("\n  orders: " + "; ".join(f"{k} " + ", ".join(f"{p:.3f}" for p in v) for k, v in order.items()))
    assert all(r["c_h"] <= r["c"] * (1 + 1e-12) for r in rows)
    for name, p in order.items():
        assert min(p[-2:]) >= 1.8, f"{name}: observed orders {p}"
    assert e_tip[-1] <= 2e-2


# ================================================================================ 4. Modal: slip-wall square / cube
# Slip walls (normal displacement and tangential traction zero on every face) separate the Navier equations:
# u = ∇ψ (P waves, λ = (λ' + 2μ)|k|²/ρ) and divergence-free fields (S waves, λ = μ|k|²/ρ), k = π(l, m[, q]) / a, with
# sin in the normal component and cos in the tangential ones. λ' + 2μ = E/(1 − ν²) in plane stress,
# E(1 − ν)/((1 + ν)(1 − 2ν)) in 3-D. Derived in this file; the 2-D fields are checked against the Navier equations
# below. Number of S polarisations: (number of non-zero k components) − 1; one P mode for every k ≠ 0.
def _analytic_spectrum(dim: int, E: float, nu: float, a: float = 1.0, kmax: int = 4) -> list[tuple[float, str, tuple]]:
    mu = E / (2 * (1 + nu))
    cp = E / (1 - nu ** 2) if dim == 2 else E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    out = []
    for k in np.ndindex(*([kmax + 1] * dim)):
        k2 = sum(v * v for v in k)
        if k2 == 0:
            continue
        out.append((cp * math.pi ** 2 * k2 / a ** 2, "P", k))
        for p in range(sum(v > 0 for v in k) - 1):
            out.append((mu * math.pi ** 2 * k2 / a ** 2, f"S{p}", k))
    return sorted(out, key=lambda t: t[0])


def _analytic_clusters(lam: np.ndarray) -> list[list[int]]:
    groups, cur = [], [0]
    for i in range(1, len(lam)):
        if abs(lam[i] - lam[i - 1]) <= 1e-12 * lam[i]:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [g for g in groups if len(g) > 1]


def _mode_2d(kind: str, l: int, m: int, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    al, be = l * math.pi, m * math.pi
    if kind == "P":                      # ∇(cos αx cos βy)
        return -al * np.sin(al * X) * np.cos(be * Y), -be * np.cos(al * X) * np.sin(be * Y)
    return be * np.sin(al * X) * np.cos(be * Y), -al * np.cos(al * X) * np.sin(be * Y)   # curl(sin αx sin βy)


def _slip_walls(d: Domain) -> np.ndarray:
    top = d.h * np.asarray(d.shape)
    tol = 1e-9 * d.h
    out = [d.dofs(d.nodes_where(lambda Z, i=i: (np.abs(Z[:, i]) < tol) | (np.abs(Z[:, i] - top[i]) < tol)), i)
           for i in range(d.dim)]
    return np.unique(np.concatenate(out))


def _slip_modal(dim: int, n: int, k: int) -> tuple[Modal, np.ndarray, np.ndarray]:
    d = Domain((n,) * dim, h=1.0 / n)
    m = Modal(d, _slip_walls(d), nu=0.3, n_modes=k, element="Q4" if dim == 2 else "H8")
    lam, Phi = m.solve_numpy(np.ones(d.n_active), np.ones(d.n_active), k)
    return m, lam, Phi


def test_slip_square_analytic_modes_satisfy_navier():
    """The 9 lowest analytic 2-D modes satisfy μ∇²u + (λ'+μ)∇(∇·u) + λρu = 0 and the slip conditions at sampled
    points (fourth-order central differences, step 1e-3). Pre-registered as "verified in the test"; the tolerance
    1e-6 relative to λ|u| was fixed with this check before it ran (truncation (kh)⁴ ≈ 1e-8, rounding ≈ 1e-10)."""
    E, nu = 1.0, 0.3
    mu, lp = E / (2 * (1 + nu)), E * nu / (1 - nu ** 2)
    worst = 0.0
    for lam, kind, (l, m) in _analytic_spectrum(2, E, nu)[:9]:
        def U(x, y, kind=kind, l=l, m=m):
            ux, uy = _mode_2d(kind[0], l, m, np.asarray(x, float), np.asarray(y, float))
            return np.array([ux, uy])
        h = 1e-3
        for x in (0.13, 0.41, 0.77):
            for y in (0.21, 0.58, 0.9):
                def dx(f, x=x, y=y):
                    return _d4(lambda t: f(t, y), x, h)

                def dy(f, x=x, y=y):
                    return _d4(lambda t: f(x, t), y, h)
                lap = np.array([_d4(lambda t: _d4(lambda s: U(s, y)[i], t, h), x, h)
                                + _d4(lambda t: _d4(lambda s: U(x, s)[i], t, h), y, h) for i in range(2)])

                def div(xx, yy):
                    return (_d4(lambda t: U(t, yy)[0], xx, h) + _d4(lambda t: U(xx, t)[1], yy, h))
                gdiv = np.array([dx(div), dy(div)])
                r = mu * lap + (lp + mu) * gdiv + lam * U(x, y)
                worst = max(worst, float(np.max(np.abs(r))) / (lam * max(float(np.max(np.abs(U(x, y)))), 1e-3)))
        # slip conditions: normal displacement and shear stress zero on x = 0, 1 and y = 0, 1
        for t in (0.17, 0.5, 0.83):
            for wall in (0.0, 1.0):
                assert abs(U(wall, t)[0]) <= 1e-12 and abs(U(t, wall)[1]) <= 1e-12
                txy_x = _d4(lambda s: U(wall, s)[0], t, h) + _d4(lambda s: U(s, t)[1], wall, h)
                txy_y = _d4(lambda s: U(s, wall)[1], t, h) + _d4(lambda s: U(t, s)[0], wall, h)
                worst = max(worst, abs(txy_x) / lam, abs(txy_y) / lam)
    print(f"\n  slip square Navier/slip residual worst {worst:.2e}")
    assert worst <= 1e-6


def test_slip_square_q4_eigenvalues_converge_with_repeated_pairs():
    """Lowest 9 eigenvalues of the slip-wall unit square, plane stress, E = 1, ν = 0.3, ρ = 1, Q4 with consistent
    mass, meshes n = 8, 16, 32. Exact: S(1,1); P(1,0), P(0,1); S(1,2), S(2,1); P(1,1); S(2,2); S(1,3), S(3,1) —
    multiplicities 1, 2, 2, 1, 1, 2.

    Pre-registered: (i) λ_h ≥ λ (relative slack 1e-10; Rayleigh–Ritz with the slip conditions imposed exactly);
    (ii) observed order between n = 16 and 32 ≥ 1.8 for each of the 9; (iii) finest relative error ≤ 3e-2;
    (iv) repeated pairs: relative gap ≤ 1e-8, and ``find_clusters(λ_h, 1e-3)`` returns exactly the analytic clusters;
    (v) each analytic mode (nodal interpolant) projected M-orthogonally onto the FE modes of its cluster keeps ≥ 0.99
    of its M-norm². Differences: discretisation only; the mesh has the square's symmetry, so a pair can split only by
    rounding.
    """
    spec = _analytic_spectrum(2, 1.0, 0.3)[:9]
    lam_ex = np.array([s[0] for s in spec])
    clusters = _analytic_clusters(lam_ex)
    assert [len(g) for g in clusters] == [2, 2, 2]
    errs = {}
    for n in (8, 16, 32):
        m, lam, Phi = _slip_modal(2, n, 10)
        lam = lam[:9]
        errs[n] = (lam - lam_ex) / lam_ex
        gaps = [abs(lam[g[1]] - lam[g[0]]) / lam[g[0]] for g in clusters]
        found = [list(g) for g in find_clusters(lam, 1e-3)]
        X = m.domain.node_coords()
        M = m.mass_matrix(np.ones(m.domain.n_active))
        keep = []
        for i, (_, kind, (l, q)) in enumerate(spec):
            g = next((c for c in clusters if i in c), [i])
            ux, uy = _mode_2d(kind[0], l, q, X[:, 0], X[:, 1])
            phi = np.empty(m.domain.ndof)
            phi[0::2], phi[1::2] = ux, uy
            pf = phi[m.free]
            a = Phi[:, g].T @ (M @ pf)
            keep.append(float(a @ a / (pf @ (M @ pf))))
        print(f"\n  slip square n={n}: rel err " + " ".join(f"{e:.3e}" for e in errs[n])
              + f" | pair gaps {' '.join(f'{v:.1e}' for v in gaps)} | clusters {found} | min kept {min(keep):.5f}",
              end="")
        assert np.all(errs[n] >= -1e-10)
        assert max(gaps) <= 1e-8
        assert found == clusters
        assert min(keep) >= 0.99
    order = np.log2(errs[16] / errs[32])
    print("\n  orders 16->32: " + " ".join(f"{p:.3f}" for p in order))
    assert np.all(order >= 1.8)
    assert np.max(errs[32]) <= 3e-2


def test_slip_cube_h8_eigenvalues_converge_with_repeated_triplets():
    """Lowest 8 eigenvalues of the slip-wall unit cube, E = 1, ν = 0.3, ρ = 1, H8 with consistent mass, n = 4, 8, 12.
    Exact: S(1,1,0) ×3 (2μπ²), S(1,1,1) ×2 (3μπ²), P(1,0,0) ×3 ((λ+2μ)π²).

    Pre-registered: λ_h ≥ λ (slack 1e-10); finest (n = 12) relative error ≤ 3e-2; order between n = 8 and 12 ≥ 1.8;
    cluster gaps ≤ 1e-8; ``find_clusters(λ_h, 1e-3)`` returns exactly the three clusters (sizes 3, 2, 3).
    """
    spec = _analytic_spectrum(3, 1.0, 0.3)[:8]
    lam_ex = np.array([s[0] for s in spec])
    clusters = _analytic_clusters(lam_ex)
    assert [len(g) for g in clusters] == [3, 2, 3]
    errs = {}
    for n in (4, 8, 12):
        _, lam, _ = _slip_modal(3, n, 9)
        lam = lam[:8]
        errs[n] = (lam - lam_ex) / lam_ex
        gaps = [(lam[g].max() - lam[g].min()) / lam[g].min() for g in clusters]
        found = [list(g) for g in find_clusters(lam, 1e-3)]
        print(f"\n  slip cube n={n}: rel err " + " ".join(f"{e:.3e}" for e in errs[n])
              + f" | gaps {' '.join(f'{v:.1e}' for v in gaps)} | clusters {found}", end="")
        assert np.all(errs[n] >= -1e-10)
        assert max(gaps) <= 1e-8
        assert found == clusters
    order = np.log(errs[8] / errs[12]) / np.log(12 / 8)
    print("\n  orders 8->12: " + " ".join(f"{p:.3f}" for p in order))
    assert np.all(order >= 1.8)
    assert np.max(errs[12]) <= 3e-2

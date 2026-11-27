"""Nonlinear transient / steady heat conduction k(T), rho c_p(T) (``toporlogy.fem.mesh.nonlinear_heat``). Checks NH1-NH9 of
docs/foundations/prereg/NH_nonlinear_heat.md (thresholds pre-registered and committed in a0d687e before this file was
run). All numbers are synthetic, not material data.

References are independent of the production code: the manufactured field and its source from sympy, the enthalpy
polynomial and Gauss quadrature of the 1-D bar written here, the Kirchhoff closed forms and ``scipy.optimize.brentq``.
Comparisons labelled "regression" call shared kernels (``HeatProblem``, ``kernels.mass``)."""
from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import brentq

from toporlogy.core import properties as P
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.history import StepRejected
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.fem.mesh.nonlinear_heat import NonlinearHeat
from toporlogy.fem.mesh.problem import Dirichlet, HeatProblem
from toporlogy.fem.mesh.quadrature import rule

K0, BETA, C0, GAM, TR = 1.0, 0.01, 2.0, 0.005, 300.0
DOM = (250.0, 450.0)
KU, CU = "W/(m K)", "J/(m^3 K)"
T_END = 0.1


def kpoly(cls: type = P.Polynomial) -> P.PropertyFunction:
    return cls([K0, K0 * BETA], TR, DOM, KU, source="synthetic", positive=True)


def cpoly() -> P.PropertyFunction:
    return P.Polynomial([C0, C0 * GAM], TR, DOM, CU, source="synthetic", positive=True)


def H_exact(T: np.ndarray) -> np.ndarray:
    """Enthalpy of the linear capacity law, written here (not the property integral)."""
    th = np.asarray(T) - TR
    return C0 * (th + 0.5 * GAM * th ** 2)


# ------------------------------------------------------------------------------------------------ manufactured field
@cache
def manufactured():
    sy = pytest.importorskip("sympy")
    x, y, t = sy.symbols("x y t")
    g = 1 + sy.Rational(1, 2) * sy.sin(30 * t)
    T = 300 + 20 * x + 10 * y + 15 * g * sy.sin(sy.pi * x) * sy.sin(2 * sy.pi * y)
    k = K0 * (1 + BETA * (T - 300))
    C = C0 * (1 + GAM * (T - 300))
    divk = sy.diff(k * sy.diff(T, x), x) + sy.diff(k * sy.diff(T, y), y)
    Q = C * sy.diff(T, t) - divk
    lam = lambda e: sy.lambdify((x, y, t), e, "numpy")  # noqa: E731
    return lam(T), lam(Q), lam(divk)


def exact_T(X: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(manufactured()[0](X[:, 0], X[:, 1], t), dtype=float) * np.ones(len(X))


def l2_error(mesh, T: np.ndarray, t: float) -> float:
    g = kn.geometry(mesh.ref, mesh.cell_points(), rule(mesh.ref.cell, 8))
    Th = np.einsum("qa,ea->eq", g.N, T[mesh.cells])
    ex = exact_T(g.x.reshape(-1, 2), t).reshape(Th.shape)
    return float(np.sqrt(np.sum((Th - ex) ** 2 * g.dV) / np.sum(ex ** 2 * g.dV)))


def run_mms(element: str, n: int, dt: float, discrete: bool) -> tuple[float, list[int]]:
    """Relative L2 error at T_END of the manufactured problem; ``discrete``: the time-consistent source."""
    _, Qc, divk = manufactured()
    Tf = manufactured()[0]
    mesh = rectangle(n, n, 1.0, 1.0, element)
    pts = mesh.points
    model = NonlinearHeat(mesh, kpoly(), cpoly(), exact_T(pts, 0.0))
    nodes = np.unique(mesh.boundary_nodes())
    model.add_dirichlet(nodes, lambda t: exact_T(pts[nodes], t))
    if discrete:
        def src(X, t):
            Th1 = Tf(X[:, 0], X[:, 1], t)
            Th0 = Tf(X[:, 0], X[:, 1], t - dt)
            return (H_exact(Th1) - H_exact(Th0)) / dt - divk(X[:, 0], X[:, 1], t)
    else:
        def src(X, t):
            return Qc(X[:, 0], X[:, 1], t)
    model.set_source(src)
    res = model.advance_to(T_END, dt)
    return l2_error(mesh, np.array(model.T), model.t), [r.iterations for r in res]


def slope(h: list[float], e: list[float]) -> float:
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


# ------------------------------------------------------------------------------------------------ NH1 tangent vs FD
def _tangent_case(element: str, form: str, k, C) -> tuple[float, float]:
    mesh = rectangle(3, 3, 1.0, 1.0, element)
    rng = np.random.default_rng(7)
    T0 = 300.0 + 40.0 * rng.random(mesh.n_nodes)
    model = NonlinearHeat(mesh, k, C, T0, form=form)
    model.add_robin(mesh.facet_sets["right"], 12.0, 330.0)
    dt = 0.01
    Tt = T0 + 10.0 * rng.random(mesh.n_nodes) - 3.0
    R0, J = model.residual_jacobian(Tt, dt, dt)
    Jd = J.toarray()
    Jfd = np.zeros_like(Jd)
    h = 1e-3
    for j in range(mesh.n_nodes):
        e = np.zeros(mesh.n_nodes)
        e[j] = h
        Rp, _ = model.residual_jacobian(Tt + e, dt, dt)
        Rm, _ = model.residual_jacobian(Tt - e, dt, dt)
        Jfd[:, j] = (Rp - Rm) / (2 * h)
    return float(np.abs(Jd - Jfd).max() / np.abs(Jd).max()), float(np.abs(Jd).max())


def _tables():
    Tk = [250.0, 300.0, 350.0, 400.0, 450.0]
    k = P.Pchip(Tk, [0.8, 1.0, 1.5, 1.9, 2.6], KU, source="synthetic", positive=True)
    C = P.PiecewiseLinear(Tk, [1.5, 2.0, 2.6, 3.0, 3.1], CU, source="synthetic", positive=True)
    return k, C


@pytest.mark.parametrize("element", ["Q4", "T6"])
def test_nh1_tangent_enthalpy_form_against_central_fd(element: str) -> None:
    k, C = _tables()
    err, _ = _tangent_case(element, "enthalpy", k, C)
    assert err <= 1e-6, f"max|J - J_fd| / max|J| = {err:.3e}"


@pytest.mark.parametrize("element", ["Q4", "T6"])
def test_nh1_tangent_capacity_form_against_central_fd(element: str) -> None:
    err, _ = _tangent_case(element, "capacity", kpoly(), cpoly())
    assert err <= 1e-6, f"max|J - J_fd| / max|J| = {err:.3e}"


class _NoDk(P.Polynomial):
    """Negative control: the derivative of the conductivity is (wrongly) zero."""

    def _derivative(self, T: np.ndarray) -> np.ndarray:
        return np.zeros(T.shape)


def test_nh1_negative_control_missing_conductivity_derivative_is_detected() -> None:
    err, _ = _tangent_case("Q4", "enthalpy", kpoly(_NoDk), cpoly())
    assert err >= 1e-3, f"the FD check did not see a tangent without k' (mismatch {err:.3e})"


# ------------------------------------------------------------------------------------------------ NH2 / NH9 spatial
_XF_T3 = ("prereg check failed: least-squares slope over n = 2..16 is 1.740 (errors 2.548e-2, 9.162e-3, 2.662e-3, "
          "6.909e-4): the coarsest level is pre-asymptotic; the order of the finest pair is 1.946 "
          "(see the post hoc test)")
_XF_Q9 = ("prereg check failed: least-squares slope over n = 2..8 is 1.995 (errors 1.125e-3, 5.526e-4, 7.082e-5): "
          "n = 2 to 4 is pre-asymptotic (order 1.03); n = 4 to 8 gives 2.964 (see the post hoc test)")


@pytest.mark.parametrize("element,levels,lo,hi", [
    pytest.param("T3", (2, 4, 8, 16), 1.85, 2.2, marks=pytest.mark.xfail(strict=True, reason=_XF_T3)),
    ("Q4", (2, 4, 8, 16), 1.85, 2.2), ("T6", (2, 4, 8), 2.8, 3.3),
    pytest.param("Q9", (2, 4, 8), 2.8, 3.3, marks=pytest.mark.xfail(strict=True, reason=_XF_Q9))])
def test_nh2_spatial_order_time_consistent_source(element: str, levels: tuple[int, ...], lo: float, hi: float) -> None:
    errs = [run_mms(element, n, 0.0125, True)[0] for n in levels]
    s = slope([1.0 / n for n in levels], errs)
    assert all(a > b for a, b in zip(errs, errs[1:])), f"errors not decreasing: {errs}"
    assert lo <= s <= hi, f"{element}: slope {s:.3f}, errors {errs}"


@pytest.mark.parametrize("element,levels,lo,hi", [("T3", (2, 4, 8, 16), 1.85, 2.2), ("Q4", (2, 4, 8, 16), 1.85, 2.2),
                                                  ("T6", (2, 4, 8, 16), 2.8, 3.3), ("Q9", (2, 4, 8, 16), 2.8, 3.3)])
def test_nh2_post_hoc_order_of_the_two_finest_levels(element: str, levels: tuple[int, ...], lo: float,
                                                     hi: float) -> None:
    """POST HOC (written after the NH2 windows were seen to fail on the coarsest level): the same windows applied to
    the order log2(e_{n}/e_{2n}) of the two finest levels n = 8, 16."""
    e = [run_mms(element, n, 0.0125, True)[0] for n in levels[-2:]]
    p = math.log2(e[0] / e[1])
    assert lo <= p <= hi, f"{element}: order {p:.3f}, errors {e}"


def test_nh9_newton_iterations_at_most_six() -> None:
    _, its = run_mms("Q4", 8, 0.0125, True)
    assert max(its) <= 6, its


# ------------------------------------------------------------------------------------------------ NH3 temporal order
def test_nh3_temporal_first_order_and_spatial_floor() -> None:
    dts = [0.0125, 0.00625, 0.003125, 0.0015625]
    errs = [run_mms("Q9", 16, dt, False)[0] for dt in dts]        # n = 16: n = 8 failed the floor condition (record)
    s = slope(dts, errs)
    floor = run_mms("Q9", 16, 0.0125, True)[0]
    assert 0.9 <= s <= 1.15, f"temporal slope {s:.3f}, errors {errs}"
    assert floor <= 0.1 * errs[-1], f"spatial floor {floor:.3e} vs smallest time error {errs[-1]:.3e}"


# ------------------------------------------------------------------------------------------------ NH4 / NH5 bar balance
LY, DT_BAR, NSTEP = 0.1, 0.05, 6
_GX, _GW = np.polynomial.legendre.leggauss(3)


def _bar_integral(xs: np.ndarray, f_of_xi) -> float:
    """sum over elements of int f dx, f a function of the element index and xi in [0, 1] (3-point Gauss, exact for
    the quadratic integrands here); times the width LY."""
    tot = 0.0
    for i in range(len(xs) - 1):
        hx = xs[i + 1] - xs[i]
        xi = 0.5 * (_GX + 1.0)
        tot += float(np.sum(0.5 * _GW * hx * f_of_xi(i, xi)))
    return tot * LY


@cache
def bar_run(form: str) -> list[dict]:
    mesh = rectangle(8, 1, 1.0, LY)
    model = NonlinearHeat(mesh, kpoly(), cpoly(), 300.0, form=form)
    model.add_flux(mesh.facet_sets["left"], lambda t: 50.0 * (1.0 + t))
    model.add_robin(mesh.facet_sets["right"], 20.0, lambda t: 320.0 + 10.0 * t)
    model.set_source(lambda X, t: 100.0 * (1.0 + X[:, 0] ** 2) * (1.0 + t))
    bottom = mesh.node_sets["bottom"]
    xs = mesh.points[bottom, 0]
    order = np.argsort(xs)
    xs, bottom = xs[order], bottom[order]
    top = mesh.node_sets["top"][np.argsort(mesh.points[mesh.node_sets["top"], 0])]
    out = []
    prev = np.array(model.T)
    for _ in range(NSTEP):
        r = model.step(DT_BAR)
        new = np.array(model.T)
        assert np.abs(new[bottom] - new[top]).max() < 1e-8          # 1-D state
        Tp, Tn = prev[bottom], new[bottom]

        def dH(i, xi):
            a = Tp[i] + (Tp[i + 1] - Tp[i]) * xi
            b = Tn[i] + (Tn[i + 1] - Tn[i]) * xi
            return H_exact(b) - H_exact(a)

        def dT2(i, xi):
            return ((Tn[i] - Tp[i]) * (1 - xi) + (Tn[i + 1] - Tp[i + 1]) * xi) ** 2

        t1 = r.t
        stored_ind = _bar_integral(xs, dH)
        flux_ind = DT_BAR * LY * 50.0 * (1.0 + t1)
        robin_ind = DT_BAR * LY * 20.0 * (320.0 + 10.0 * t1 - Tn[-1])
        source_ind = DT_BAR * LY * 100.0 * (1.0 + t1) * (1.0 + 1.0 / 3.0)
        scale = max(abs(stored_ind), abs(flux_ind), abs(robin_ind), abs(source_ind))
        out.append({"r": r, "stored_ind": stored_ind, "imbalance": stored_ind - flux_ind - robin_ind - source_ind,
                    "scale": scale, "pred_capacity": -0.5 * C0 * GAM * _bar_integral(xs, dT2),
                    "H0": prev[bottom].copy(), "H1": Tn.copy(), "xs": xs})
        prev = new
    return out


def test_nh4a_bar_energy_balance_enthalpy_form_to_rounding() -> None:
    for s in bar_run("enthalpy"):
        assert abs(s["imbalance"]) <= 1e-9 * s["scale"], (s["imbalance"], s["scale"])
        assert abs(s["r"].stored - s["stored_ind"]) <= 1e-11 * abs(s["stored_ind"])
        assert abs(s["r"].energy_residual) <= 1e-9 * s["r"].energy_scale
        assert s["r"].dirichlet_heat == 0.0


def test_nh4c_enthalpy_increments_telescope() -> None:
    run = bar_run("enthalpy")
    xs = run[0]["xs"]
    T_first = np.full(len(xs), 300.0)
    T_last = run[-1]["H1"]

    def dH(i, xi):
        a = T_first[i] + (T_first[i + 1] - T_first[i]) * xi
        b = T_last[i] + (T_last[i + 1] - T_last[i]) * xi
        return H_exact(b) - H_exact(a)

    total = _bar_integral(xs, dH)
    assert abs(sum(s["r"].stored for s in run) - total) <= 1e-11 * abs(total)


def test_nh4b_general_mesh_all_boundary_kinds_energy_residual() -> None:
    k, C = _tables()
    mesh = rectangle(4, 3, 1.0, 0.75, "T6")
    model = NonlinearHeat(mesh, k, C, 300.0)
    left = np.unique(mesh.node_sets["left"])
    model.add_dirichlet(left, lambda t: 300.0 + 400.0 * t)
    model.add_robin(mesh.facet_sets["right"], 15.0, lambda t: 310.0 + 100.0 * t)
    model.add_flux(mesh.facet_sets["bottom"], lambda t: 30.0 * (1.0 + 2.0 * t))
    model.set_source(lambda X, t: 40.0 * (1.0 + X[:, 0] * X[:, 1]))
    res = model.advance_to(0.1, 0.02)
    assert len(res) == 5
    for r in res:
        assert abs(r.energy_residual) <= 1e-9 * r.energy_scale, (r.energy_residual, r.energy_scale)
    assert abs(res[-1].dirichlet_heat) >= 1e-3 * res[-1].energy_scale   # the reservoir term is not vacuous


def test_nh5_capacity_form_negative_control_imbalance_matches_closed_form() -> None:
    for s in bar_run("capacity"):
        pred = s["pred_capacity"]
        assert pred < 0.0
        assert abs(s["imbalance"] - pred) <= 1e-6 * abs(pred), (s["imbalance"], pred)
        assert abs(s["imbalance"]) >= 1e-3 * s["scale"], (s["imbalance"], s["scale"])
        assert abs(s["r"].stored - s["r"].stored_scheme) > 0.0           # the schemes differ


# ------------------------------------------------------------------------------------------------ NH6 linear limit
@pytest.mark.parametrize("element", ["Q4", "T3"])
def test_nh6_linear_limit_equals_heat_problem(element: str) -> None:
    kc, cc, h, Tinf, Q0, TL = 1.7, 2.3, 7.0, 290.0, 40.0, 310.0
    mesh = rectangle(6, 4, 1.5, 1.0, element)
    left = np.unique(mesh.node_sets["left"])
    k = P.Constant(kc, KU, source="synthetic", positive=True)
    C = P.Constant(cc, CU, source="synthetic", positive=True)
    # existing solver
    hp = HeatProblem(mesh, kc)
    f = hp.robin(mesh.facet_sets["right"], h, Tinf) + hp.body_force(Q0)
    Tref = hp.solve(f, Dirichlet([(left, TL)])).u[:, 0]
    model = NonlinearHeat(mesh, k, C, 300.0)
    model.add_dirichlet(left, TL)
    model.add_robin(mesh.facet_sets["right"], h, Tinf)
    model.set_source(Q0)
    steady = model.steady()
    assert np.abs(steady.T - Tref).max() <= 1e-9 * (1.0 + np.abs(Tref).max())
    # transient: dense backward Euler from HeatProblem.K and kernels.mass (regression)
    g = kn.geometry(mesh.ref, mesh.cell_points(), rule(mesh.ref.cell, 2 * mesh.ref.degree))
    M = hp.pattern.assemble(kn.mass(g, cc, ncomp=1))
    A = (M / 0.1 + hp.K).toarray()
    free = np.setdiff1d(np.unique(mesh.cells), left)
    T = np.full(mesh.n_nodes, 300.0)
    tr = NonlinearHeat(mesh, k, C, 300.0)
    tr.add_dirichlet(left, TL)
    tr.add_robin(mesh.facet_sets["right"], h, Tinf)
    tr.set_source(Q0)
    tr.advance_to(0.5, 0.1)
    for _ in range(5):
        rhs = f + (M / 0.1) @ T                                    # the OLD temperatures, also at the Dirichlet nodes
        T[left] = TL
        T[free] = np.linalg.solve(A[np.ix_(free, free)], rhs[free] - A[np.ix_(free, left)] @ T[left])
    assert np.abs(np.array(tr.T) - T).max() <= 1e-9 * (1.0 + np.abs(T).max())
    assert sp.issparse(M)


# ------------------------------------------------------------------------------------------------ NH7 Kirchhoff
def _phi(theta: np.ndarray) -> np.ndarray:
    return K0 * (theta + 0.5 * BETA * theta ** 2)


def _theta_of_phi(phi: np.ndarray) -> np.ndarray:
    return (-1.0 + np.sqrt(1.0 + 2.0 * BETA * phi / K0)) / BETA


def kirchhoff_case(nx: int, variant: str) -> tuple[float, int]:
    mesh = rectangle(nx, 1, 1.0, 0.1)
    model = NonlinearHeat(mesh, kpoly(), cpoly(), 300.0)
    model.add_dirichlet(np.unique(mesh.node_sets["left"]), 300.0)
    x = mesh.points[:, 0]
    if variant == "dirichlet":
        model.add_dirichlet(np.unique(mesh.node_sets["right"]), 400.0)
        phi = _phi(np.array(100.0)) * x
    elif variant == "flux":
        model.add_flux(mesh.facet_sets["right"], 100.0)
        phi = 100.0 * x
    else:
        h, Tinf = 1.0, 420.0
        thR = brentq(lambda th: _phi(th) - h * (Tinf - 300.0 - th) * 1.0, 0.0, 120.0, xtol=1e-14)
        model.add_robin(mesh.facet_sets["right"], h, Tinf)
        phi = _phi(np.array(thR)) * x
    res = model.steady()
    exact = 300.0 + _theta_of_phi(phi)
    return float(np.abs(np.array(model.T) - exact).max()), res.iterations


_XF_NH7 = ("prereg check failed: the nodal error is at rounding level (<= 6.3e-13 K at every level and variant; "
           "the Dirichlet case is 0.0 at nx = 4), not O(h^2): with exact quadrature the element flux is "
           "Phi(T_{i+1}) - Phi(T_i), so the nodal values are exact "
           "(see the post hoc test); the slope of rounding noise is meaningless")


@pytest.mark.xfail(strict=True, reason=_XF_NH7)
@pytest.mark.parametrize("variant", ["dirichlet", "flux", "robin"])
def test_nh7_steady_one_dimensional_kirchhoff(variant: str) -> None:
    levels = (4, 8, 16, 32)
    runs = [kirchhoff_case(n, variant) for n in levels]
    errs = [e for e, _ in runs]
    s = slope([1.0 / n for n in levels], errs)
    assert s >= 1.7, f"{variant}: slope {s:.3f}, errors {errs}"
    assert errs[-1] <= 0.1, f"{variant}: error {errs[-1]:.3e} K at nx = 32"
    assert max(i for _, i in runs) <= 10, runs


@pytest.mark.parametrize("variant", ["dirichlet", "flux", "robin"])
def test_nh7_post_hoc_nodal_values_are_exact(variant: str) -> None:
    """POST HOC. Derivation: for a linear element in 1-D, k(T_h) T_h' is the derivative of Phi(T_h(x)), so with exact
    quadrature the element flux is Phi(T_{i+1}) - Phi(T_i); the equations make Phi(T_i) linear in i, i.e. the nodal
    values are those of the Kirchhoff closed form for any mesh. Bound: 1e-10 K at every level, Newton <= 10."""
    runs = [kirchhoff_case(n, variant) for n in (4, 8, 16, 32)]
    assert max(e for e, _ in runs) <= 1e-10, runs
    assert max(i for _, i in runs) <= 10, runs


# ------------------------------------------------------------------------------------------------ NH8 refusals
def _bar(**kw) -> tuple[object, NonlinearHeat]:
    mesh = rectangle(4, 1, 1.0, 0.1)
    return mesh, NonlinearHeat(mesh, kpoly(), cpoly(), 300.0, **kw)


def test_nh8a_property_validation() -> None:
    mesh = rectangle(2, 1, 1.0, 0.1)
    free = P.Polynomial([1.0, 0.0], TR, DOM, KU, source="synthetic")             # positive=False
    with pytest.raises(ValueError, match="positive"):
        NonlinearHeat(mesh, free, cpoly(), 300.0)
    wrong = P.Polynomial([1.0, 0.0], TR, DOM, "W/m", source="synthetic", positive=True)
    with pytest.raises(ValueError, match="units"):
        NonlinearHeat(mesh, wrong, cpoly(), 300.0)
    with pytest.raises(ValueError, match="PropertyFunction"):
        NonlinearHeat(mesh, 1.0, cpoly(), 300.0)
    with pytest.raises(ValueError, match="form"):
        NonlinearHeat(mesh, kpoly(), cpoly(), 300.0, form="lumped")


def test_nh8b_initial_temperature_outside_validity() -> None:
    mesh = rectangle(2, 1, 1.0, 0.1)
    with pytest.raises(ValueError, match="validity"):
        NonlinearHeat(mesh, kpoly(), cpoly(), 500.0)
    with pytest.raises(ValueError, match="> 0"):
        NonlinearHeat(mesh, kpoly(), cpoly(), -3.0)


def test_nh8c_rejection_changes_nothing_and_later_step_is_bitwise_clean() -> None:
    def make() -> NonlinearHeat:
        mesh = rectangle(4, 1, 1.0, 0.1)
        m = NonlinearHeat(mesh, kpoly(), cpoly(), 300.0)
        m.add_dirichlet(np.unique(mesh.node_sets["left"]), lambda t: 460.0 if t < 0.015 else 330.0)
        m.add_robin(mesh.facet_sets["right"], 10.0, 300.0)
        return m

    a, b = make(), make()
    T_before, t_before = np.array(a.T), a.t
    with pytest.raises(StepRejected, match="validity"):
        a.step(0.01)
    assert a.t == t_before and np.array_equal(a.T, T_before)
    ra = a.step(0.02)
    rb = b.step(0.02)
    assert np.array_equal(ra.T, rb.T) and ra.iterations == rb.iterations


def test_nh8d_non_convergence_is_rejected_without_state_change() -> None:
    mesh = rectangle(4, 1, 1.0, 0.1)
    m = NonlinearHeat(mesh, kpoly(), cpoly(), 300.0)
    m.add_dirichlet(np.unique(mesh.node_sets["left"]), 400.0)
    T_before = np.array(m.T)
    with pytest.raises(StepRejected, match="no convergence"):
        m.step(0.01, max_iter=1)
    assert m.t == 0.0 and np.array_equal(m.T, T_before)
    assert m.step(0.01).iterations > 1                                  # the same step converges with the default


class _SignFlip(P.Constant):
    """Declared positive, but k < 0 above 320 K: the run-time guard must catch it."""

    def _value(self, T: np.ndarray) -> np.ndarray:
        return np.where(T > 320.0, -1.0, 1.0)


def test_nh8e_non_positive_property_value_at_run_time_is_rejected() -> None:
    mesh = rectangle(4, 1, 1.0, 0.1)
    k = _SignFlip(1.0, KU, source="synthetic", domain=DOM, positive=True)
    m = NonlinearHeat(mesh, k, cpoly(), 300.0)
    m.add_dirichlet(np.unique(mesh.node_sets["left"]), 330.0)
    with pytest.raises(StepRejected, match="conductivity"):
        m.step(0.01)
    assert m.t == 0.0


def test_nh8f_invalid_arguments() -> None:
    mesh, m = _bar()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.step(bad)
    with pytest.raises(ValueError, match="max_iter"):
        m.step(0.01, max_iter=0)
    with pytest.raises(ValueError, match="steady problem"):
        m.steady()
    m.add_dirichlet(np.unique(mesh.node_sets["left"]), -5.0)
    with pytest.raises(ValueError, match="> 0 K"):
        m.step(0.01)
    with pytest.raises(ValueError, match="already"):
        m.add_dirichlet(np.unique(mesh.node_sets["left"]), 310.0)
    with pytest.raises(ValueError):
        m.add_robin(mesh.facet_sets["right"], -1.0, 300.0)
    assert m.t == 0.0
    assert math.isfinite(float(np.array(m.T).sum()))

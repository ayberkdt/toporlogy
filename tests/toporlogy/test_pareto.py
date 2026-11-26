"""Multi-objective scalarisations: records P1-P5 of ``docs/olcumler/laminat_uq_onkayit.md``."""
from __future__ import annotations

import numpy as np
import pytest

import toporlogy as T
from toporlogy.optimization import pareto as P

WEIGHTS = (0.1, 0.3, 0.5, 0.7, 0.9)


# --------------------------------------------------------------------------------------------------- P1
def test_p1_dominance_filter() -> None:
    F = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 3.0], [4.0, 1.0], [5.0, 2.0], [2.0, 3.0]])
    assert P.nondominated(F).tolist() == [True, True, False, True, False, True]
    assert P.dominates([1, 1], [1, 2]) and not P.dominates([1, 1], [1, 1]) and not P.dominates([0, 3], [1, 2])
    with pytest.raises(ValueError):
        P.nondominated(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        P.dominates([1.0], [1.0, 2.0])


# --------------------------------------------------------------------------------------------------- P2-P3
def f1(x: np.ndarray) -> float:
    return float(x[0])


def f2(x: np.ndarray) -> float:
    g = 1.0 + 9.0 * x[1]
    return float(g * (1.0 - (x[0] / g) ** 2))


BOX = [(0.0, 1.0), (0.0, 1.0)]


def nonconvex_front(n_weights: int = 21, n_eps: int = 11) -> tuple[list[P.ParetoPoint], list[P.ParetoPoint]]:
    ws = [P.weighted_sum([f1, f2], [w, 1 - w], BOX) for w in np.linspace(0, 1, n_weights)]
    eps = [P.epsilon_constraint([f1, f2], 1, {0: float(e)}, BOX) for e in np.linspace(0, 1, n_eps)]
    return ws, eps


def test_p2a_weighted_sum_misses_the_interior_of_a_nonconvex_front() -> None:
    ws, _ = nonconvex_front()
    for p in ws:
        assert min(abs(p.objectives[0]), abs(p.objectives[0] - 1)) <= 1e-3, (p.parameter, p.objectives)


def test_p2b_epsilon_constraint_covers_the_nonconvex_front() -> None:
    _, eps = nonconvex_front()
    assert all(p.feasible for p in eps)
    v = np.array([p.objectives for p in eps])
    assert np.abs(v[:, 1] - (1 - v[:, 0] ** 2)).max() <= 1e-5
    assert np.diff(np.sort(v[:, 0])).max() <= 0.15
    assert P.nondominated(v, tol=1e-9).all()


def test_p3_weighted_sum_covers_a_convex_front() -> None:
    g1 = lambda x: float(x[0] ** 2)                                          # noqa: E731
    g2 = lambda x: float((x[0] - 1) ** 2)                                    # noqa: E731
    for w in np.linspace(0.05, 0.95, 10):
        p = P.weighted_sum([g1, g2], [w, 1 - w], [(-1.0, 2.0)])
        assert abs(p.x[0] - (1 - w)) <= 1e-6


# --------------------------------------------------------------------------------------------------- P4
def structural_problem():
    """Cantilever 40 × 20, two separate load cases: downward at the lower right corner, axial pull at mid height."""
    d = T.Domain((40, 20))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    F = d.load_vector(2)
    F[d.dofs(d.nodes_where(lambda X: (X[:, 0] > 40 - 1e-9) & (X[:, 1] < 1e-9)), 1), 0] = -1.0
    F[d.dofs(d.nodes_where(lambda X: (X[:, 0] > 40 - 1e-9) & (np.abs(X[:, 1] - 10) < 1e-9)), 0), 1] = 1.0
    return d, fixed, F


def structural_run(objective: T.Response, extra: list, iterations: int = 100, x0: np.ndarray | None = None) -> dict:
    d, fixed, F = structural_problem()
    chain = T.standard_chain(d, 1.5)
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), objective,
                 [T.Constraint(T.Volume(), 0.4), *extra], T.MMA(move=0.2))
    res = st.run(x0=x0, max_iter=iterations, tol=1e-3)
    c = np.sum(F * res.U, axis=0)
    return {"c": c, "volume": res.constraint_values["volume"], "feasible": res.feasible,
            "iterations": res.iterations, "xphys": res.xphys, "x": res.x}


def structural_sweep(iterations: int = 100) -> dict:
    ws = {w: structural_run(T.Compliance(weights=[w, 1 - w]), [], iterations) for w in WEIGHTS}
    eps = float(ws[0.5]["c"][1])
    e = structural_run(P.CaseCompliance(0), [T.Constraint(P.CaseCompliance(1), eps)], iterations)
    return {"weighted": ws, "epsilon": eps, "constrained": e}


@pytest.fixture(scope="module")
def sweep() -> dict:
    return structural_sweep()


def test_p4a_weighted_points_are_feasible_and_mutually_nondominated(sweep: dict) -> None:
    v = np.array([sweep["weighted"][w]["c"] for w in WEIGHTS])
    assert all(sweep["weighted"][w]["feasible"] for w in WEIGHTS)
    assert P.nondominated(v).all(), v


def test_p4b_front_is_monotone_in_the_weight(sweep: dict) -> None:
    v = np.array([sweep["weighted"][w]["c"] for w in WEIGHTS])          # increasing weight on case 0
    assert np.all(v[1:, 0] <= v[:-1, 0] * 1.01), v
    assert np.all(v[1:, 1] >= v[:-1, 1] * 0.99), v


@pytest.mark.xfail(strict=True, reason="pre-registered P4c failed: from the default start the epsilon-constraint run "
                                       "ends in another local minimum (see docstring)")
def test_p4c_epsilon_constraint_reaches_the_weighted_point(sweep: dict) -> None:
    """Pre-registered P4c, first run (26 Nov 2026), not met: c2 = 17.3896 holds the bound, but c1 = 123.77 against
    111.50 of the weighted point (+11 %, bound +5 %). Post hoc: 300 iterations (c1 = 123.70, design change below tol)
    and move 0.05 (123.79) give the same minimum; started from the weighted design it stays there (next test)."""
    e = sweep["constrained"]
    assert e["c"][1] <= sweep["epsilon"] * (1 + 1e-3), (e["c"], sweep["epsilon"])
    assert e["c"][0] <= 1.05 * sweep["weighted"][0.5]["c"][0], (e["c"], sweep["weighted"][0.5]["c"])


def test_post_hoc_p4c_from_the_weighted_design(sweep: dict) -> None:
    """POST HOC: warm-started from the weighted design the ε-constraint problem keeps that point (c1 ≤ c1_w, c2 ≤ ε),
    so the P4c miss is the local minimum reached from the default start, not the ε-constraint formulation."""
    w = sweep["weighted"][0.5]
    e = structural_run(P.CaseCompliance(0), [T.Constraint(P.CaseCompliance(1), sweep["epsilon"])], 50, x0=w["x"])
    assert e["c"][1] <= sweep["epsilon"] * (1 + 1e-3)
    assert e["c"][0] <= w["c"][0] * (1 + 1e-3)


# --------------------------------------------------------------------------------------------------- P5
def test_p5_bad_input_and_unreachable_bounds() -> None:
    with pytest.raises(ValueError):
        P.weighted_sum([f1, f2], [-1.0, 2.0], BOX)
    with pytest.raises(ValueError):
        P.weighted_sum([f1, f2], [0.0, 0.0], BOX)
    with pytest.raises(ValueError):
        P.weighted_sum([f1, f2], [1.0], BOX)
    with pytest.raises(ValueError):
        P.epsilon_constraint([f1, f2], 1, {1: 0.0}, BOX)
    with pytest.raises(ValueError):
        P.epsilon_constraint([f1, f2], 0, {1: 0.5}, [(0.0, np.inf), (0.0, 1.0)])
    p = P.epsilon_constraint([f1, f2], 0, {1: -1.0}, BOX)                   # f2 >= 0 on the box
    assert not p.feasible
    with pytest.raises(ValueError):
        P.CaseCompliance(-1)

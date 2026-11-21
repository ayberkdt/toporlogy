"""Response algebra and the displacement response (optimization/compose.py): values against direct computation,
gradients at the pre-registered 1e-6 (Richardson for the ones through the solve)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.optimization.compose import Displacement, Functional, Ratio, Shifted, WeightedSum
from toporlogy.optimization.diagnostics import study_gradcheck

TOL = 1e-6


def _study(objective, n_cases: int = 2):
    d = T.Domain((9, 5), 1.0, thickness=0.6)
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    tip = d.nodes_where(lambda X: X[:, 0] == 9)
    F = d.load_vector(n_cases)
    F[2 * tip + 1, 0] = -1.0 / len(tip)
    F[2 * tip, 1] = 0.5 / len(tip)
    return T.Study(d, T.Elasticity(d, fixed), F, T.standard_chain(d, 1.5, beta=2.0), T.SIMP(d), objective(d),
                   [T.Constraint(T.Volume(), 0.5)], T.MMA()), d


def _fd4(check, h: float = 1e-3) -> float:
    r1, r2 = check(h), check(2 * h)
    return max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
               for a, b in zip(r1["directions"], r2["directions"], strict=True))


def test_displacement_value_and_gradient() -> None:
    def obj(d):
        mid = d.nodes_where(lambda X: (X[:, 0] == 5) & (X[:, 1] >= 2))
        return Displacement(2 * mid + 1, case=0)             # not proportional to the load: a real adjoint solve
    st, d = _study(obj)
    x0 = 0.3 + 0.5 * np.random.default_rng(0).random(d.n_design)
    ev = st.evaluate(x0)
    mid = d.nodes_where(lambda X: (X[:, 0] == 5) & (X[:, 1] >= 2))
    assert ev.f0 == pytest.approx(ev.U[2 * mid + 1, 0].mean(), rel=1e-13)
    n0 = st.physics.n_adjoint_solves
    assert _fd4(lambda h: study_gradcheck(st, x0, "objective", n_dirs=4, h=h, seed=1)) <= TOL
    assert st.physics.n_adjoint_solves > n0


def test_displacement_proportional_to_the_load_is_compliance() -> None:
    """L = F of case 0: the response is the compliance of that case, and no adjoint solve is needed."""
    st, d = _study(lambda d: T.Compliance(weights=[1.0, 0.0]))
    dofs = np.flatnonzero(st.F[:, 0])
    disp = Displacement(dofs, st.F[dofs, 0], case=0)
    x0 = np.full(d.n_design, 0.5)
    ctx = st.context(torch.as_tensor(x0).requires_grad_(True), True)
    n0 = st.physics.n_adjoint_solves
    v = disp(ctx)
    (g,) = torch.autograd.grad(v, ctx.x)
    assert float(v) == pytest.approx(st.evaluate(x0).f0, rel=1e-12) and st.physics.n_adjoint_solves == n0
    assert np.allclose(g.numpy(), st.evaluate(x0).df0, rtol=1e-10)


def test_algebra_values_and_gradients() -> None:
    def obj(d):
        c, v = T.Compliance(), T.Volume()
        f = Functional(lambda ctx: (ctx.xphys ** 2).mean(), "mean_square")
        return WeightedSum([Ratio(c, v), Shifted(f, 3.0, -0.2), c], [0.5, 2.0, 0.1], references="first")
    st, d = _study(obj)
    x0 = 0.3 + 0.5 * np.random.default_rng(2).random(d.n_design)
    ws = st.objective
    ev = st.evaluate(x0)
    c = float((st.F * ev.U).sum())
    v = float(ev.xphys.mean())
    f = 3.0 * float((ev.xphys ** 2).mean()) - 0.2
    assert ws.references == pytest.approx([c / v, abs(f), c])
    assert ev.f0 == pytest.approx(0.5 + 2.0 * np.sign(f) + 0.1)        # every term is 1 (times its weight) at first
    assert ws.needs == frozenset({"u"}) and "compliance_per_volume" in ws.name
    assert _fd4(lambda h: study_gradcheck(st, x0, "objective", n_dirs=4, h=h, seed=3)) <= TOL
    ws.reset_scale()
    assert ws.references is None


def test_errors() -> None:
    c = T.Compliance()
    for bad in (lambda: WeightedSum([]), lambda: WeightedSum([c], [1.0, 2.0]), lambda: WeightedSum([c], references=[1, 2]),
                lambda: Displacement([]), lambda: Displacement([1, 2], [1.0])):
        with pytest.raises(ValueError):
            bad()
    st, d = _study(lambda d: Functional(lambda ctx: ctx.xphys, "vector"))
    with pytest.raises(ValueError, match="scalar"):
        st.evaluate(np.full(d.n_design, 0.5))
    st, d = _study(lambda d: Displacement([0], case=7))
    with pytest.raises(ValueError, match="load case"):
        st.evaluate(np.full(d.n_design, 0.5))

"""The worked examples of docs/deriving_methods.md, executed: a new local criterion (strain energy density), a new
global response (worst load case), a new chain link (volume-preserving projection) and a new optimizer (projected
gradient). Each is derived from the building blocks only and must pass the same gates as a built-in component."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.optimization.aggregates import ks
from toporlogy.optimization.benchmark import compare_runs, format_table
from toporlogy.optimization.compose import Functional
from toporlogy.optimization.diagnostics import study_gradcheck
from toporlogy.optimization.local import Aggregated, AugmentedLagrangian, LocalField

TOL = 1e-6


# --------------------------------------------------------------------------- 1. a local criterion in ten lines
class EnergyDensity(LocalField):
    """Strain energy density of every active element and load case over a limit: w_e / w_lim − 1 ≤ 0, with
    w_e = E_e u_eᵀ KE u_e · geom / (2 V_e)."""

    name = "energy_density"
    needs = frozenset({"u"})

    def __init__(self, w_lim: float) -> None:
        self.w_lim = float(w_lim)

    def __call__(self, ctx: T.EvalContext) -> torch.Tensor:
        m = ctx.physics
        d = ctx.domain
        ue = ctx.U[torch.as_tensor(m.edof, dtype=torch.long)]                       # (na, nloc, nc)
        e = torch.einsum("eac,ab,ebc->ec", ue, torch.as_tensor(m.KE), ue)
        E = ctx.E.reshape(-1)[torch.as_tensor(m.elements, dtype=torch.long)]
        vol = d.h ** d.dim if d.dim == 3 else d.h ** 2 * d.thickness
        return 0.5 * m.geom * E[:, None] * e / vol / self.w_lim - 1.0


def _study(objective, constraints, optimizer=None, n_cases: int = 2, chain=None, shape=(14, 7)):
    d = T.Domain(shape, 1.0)
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    tip = d.nodes_where(lambda X: X[:, 0] == shape[0])
    F = d.load_vector(n_cases)
    F[2 * tip + 1, 0] = -1.0 / len(tip)
    if n_cases > 1:
        F[2 * tip, 1] = 2.0 / len(tip)
    ch = chain(d) if chain else T.standard_chain(d, 1.5, beta=2.0)
    return T.Study(d, T.Elasticity(d, fixed), F, ch, T.SIMP(d), objective, constraints, optimizer or T.MMA()), d


def _fd4(check, h: float = 1e-3) -> float:
    r1, r2 = check(h), check(2 * h)
    return max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
               for a, b in zip(r1["directions"], r2["directions"], strict=True))


def test_new_local_criterion_gets_both_strategies_and_their_gradients() -> None:
    field = EnergyDensity(w_lim=5.0)
    x0 = None
    for objective, constraints, which in (
            (T.Compliance(), [T.Constraint(Aggregated(field, "ks", 10.0), 0.0, scale=1.0)], 0),
            (AugmentedLagrangian(T.Volume(), field, reference=1.0), [T.Constraint(T.Volume(), 1.0)], "objective")):
        st, d = _study(objective, constraints)
        x0 = 0.3 + 0.5 * np.random.default_rng(0).random(d.n_design)
        if which == "objective":
            st.evaluate(x0)
            objective.update()                               # non-zero multipliers
        assert _fd4(lambda h, st=st, which=which: study_gradcheck(st, x0, which, n_dirs=4, h=h, seed=1)) <= TOL
    # the total strain energy of a load case is half its compliance: the field is what it claims to be
    st, d = _study(T.Compliance(weights=[1.0, 0.0]), [T.Constraint(T.Volume(), 0.5)])
    ctx = st.context(torch.as_tensor(x0), True)
    w = (field(ctx) + 1.0) * field.w_lim
    assert float(w[:, 0].sum()) == pytest.approx(0.5 * float(st.objective(ctx)), rel=1e-12)


# --------------------------------------------------------------------------- 2. a global response in one line
def test_worst_load_case_compliance_from_a_function() -> None:
    worst = Functional(lambda ctx: ks((torch.as_tensor(ctx.F) * ctx.U).sum(dim=0), 0.5), "worst_case", {"u"})
    st, d = _study(worst, [T.Constraint(T.Volume(), 0.5)])
    x0 = 0.3 + 0.5 * np.random.default_rng(2).random(d.n_design)
    ev = st.evaluate(x0)
    per_case = (st.F * ev.U).sum(axis=0)
    assert per_case.max() <= ev.f0 <= per_case.max() + math.log(2) / 0.5
    assert _fd4(lambda h: study_gradcheck(st, x0, "objective", n_dirs=4, h=h, seed=3)) <= TOL
    res = st.run(max_iter=15)
    sum_st, _ = _study(T.Compliance(), [T.Constraint(T.Volume(), 0.5)])
    ref = sum_st.run(max_iter=15)
    worst_of = lambda r: float((st.F * r.U).sum(axis=0).max())                 # noqa: E731
    print(f"\n  worst load-case compliance after 15 iterations: min-max design {worst_of(res):.2f}, "
          f"sum design {worst_of(ref):.2f}")
    assert worst_of(res) <= worst_of(ref) * 1.001                              # it optimises what it says


# --------------------------------------------------------------------------- 3. a chain link
class VolumePreservingProjection(torch.nn.Module):
    """tanh projection whose threshold η is chosen at every call so that the mean over the part is unchanged
    (bisection on the detached field). The threshold depends on the field: dη/dx follows from the implicit function
    theorem, written here in closed form (the link is NOT just a projection with a constant η)."""

    def __init__(self, domain: T.Domain, beta: float = 4.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.register_buffer("mask", torch.as_tensor(domain.active))

    def _proj(self, x: torch.Tensor, eta: torch.Tensor | float) -> torch.Tensor:
        b = self.beta
        e = torch.as_tensor(eta, dtype=x.dtype)
        return (torch.tanh(b * e) + torch.tanh(b * (x - e))) / (torch.tanh(b * e) + torch.tanh(b * (1 - e)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xa = x[self.mask]
        target = float(xa.detach().mean())
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if float(self._proj(xa.detach(), mid).mean()) > target else (lo, mid)
        eta0 = torch.tensor(0.5 * (lo + hi), dtype=x.dtype, requires_grad=True)
        with torch.enable_grad():
            r = self._proj(xa.detach(), eta0).mean()
            (dr_deta,) = torch.autograd.grad(r, eta0)
        # residual R(x, η) = mean(proj(x, η)) − mean(x) = 0  →  η(x) = η0 − R(x, η0) / (∂R/∂η), exact to first order
        eta = eta0.detach() - (self._proj(xa, eta0.detach()).mean() - xa.mean()) / dr_deta
        return torch.where(self.mask, self._proj(x, eta), torch.zeros_like(x))


def test_new_link_preserves_volume_and_has_the_right_gradient() -> None:
    d = T.Domain((10, 6))
    link = VolumePreservingProjection(d, beta=6.0)
    x0 = 0.15 + 0.7 * np.random.default_rng(4).random(d.shape)
    y = link(torch.as_tensor(x0))
    assert float(y.mean()) == pytest.approx(x0.mean(), abs=1e-12)
    w = torch.as_tensor(np.random.default_rng(5).random(d.shape))
    r = T.gradcheck(lambda x: (w * link(x) ** 2).sum(), x0, n_dirs=5)
    assert r["max_rel_error"] <= TOL, r

    class Naive(VolumePreservingProjection):                                   # the tempting shortcut: constant η
        def forward(self, x):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if float(self._proj(x.detach(), mid).mean()) > float(x.detach().mean()) \
                    else (lo, mid)
            return self._proj(x, 0.5 * (lo + hi))

    bad = T.gradcheck(lambda x: (w * Naive(d, 6.0)(x) ** 2).sum(), x0, n_dirs=5)
    assert bad["max_rel_error"] > 1e-3                                         # the gate catches it


# --------------------------------------------------------------------------- 4. an optimizer
class ProjectedGradient(T.Optimizer):
    """Steepest descent with a fixed step, projected on the box and shifted uniformly to meet one volume-like
    constraint (bisection through the chain, as OC). Thirty lines; a baseline, not a recommendation."""

    name = "PG"
    max_constraints = 1
    needs_constraint_fn = True

    def __init__(self, step: float = 0.1) -> None:
        self.step_size = float(step)

    def step(self, x, f0, df0, g, dg, constraint_fn=None):
        y = x - self.step_size * df0 / max(float(np.abs(df0).max()), 1e-300)
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if float(constraint_fn(np.clip(y - mid, 0, 1))[0]) > 0 else (lo, mid)
        return np.clip(y - hi, 0.0, 1.0)


def test_new_optimizer_runs_in_the_study_and_is_compared_on_equal_terms() -> None:
    runs = {}
    for name, opt in (("PG", ProjectedGradient()), ("OC", T.OC()), ("MMA", T.MMA())):
        st, _ = _study(T.Compliance(), [T.Constraint(T.Volume(), 0.5)], opt, n_cases=1,
                       chain=lambda d: T.standard_chain(d, 1.5, projection=False), shape=(36, 12))
        runs[name] = (st, st.run(max_iter=80, tol=0.0))
    rows = compare_runs(runs)
    print("\n" + format_table(rows))
    by = {r["name"]: r for r in rows}
    assert all(r["constraints"]["volume"]["verdict"] == "PASS" for r in rows)
    assert by["PG"]["evaluations"] == by["OC"]["evaluations"] == by["MMA"]["evaluations"] == 81   # equal budget
    # measured 21 Nov 2026: see docs/deriving_methods.md §4; the gate only says the baseline works at all
    assert by["PG"]["objective_binary"] <= 1.5 * by["MMA"]["objective_binary"]

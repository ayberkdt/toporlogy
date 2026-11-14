"""Finite-difference gradient checks of every chain link, interpolation and response (docs/mimari_plan.md §3.1).

Pre-registered threshold: relative error ≤ 1e-6 (float64, CPU) of the directional derivative along random unit
directions, :func:`toporlogy.checks.gradcheck`. Designs are random interior points (x ∈ [0.2, 0.8]) so no
gradient is degenerate. The domains are NOT boxes (holes, L shape, partial frac, keep and void regions) so masking
and passive handling are covered.

Step size. Chain links and interpolations are checked with gradcheck's own central difference (h = 1e-6).
Responses through the linear solve are not: the direct solve returns f with a relative error of ~1e-13
(inferred from an h sweep, 13 Nov 2026: f ≈ 9.3e3 carries ~7e-10 of noise), so the central difference has a noise floor
``1e-13·|f| / (h·|∇f·d|)`` that reaches 1e-5…1e-4 at h = 1e-6 along directions where |∇f·d| ≪ |f| (measured
2.9e-6 … 1.9e-4 for the cases below; at h = 1e-4 truncation and noise still meet at ~1e-6). For those responses the
same gradcheck is run at h and 2h (same seed → same directions) and Richardson-extrapolated,
``fd = (4 fd(h) − fd(2h)) / 3`` (the fourth-order five-point stencil), at h = 1e-3: truncation O(h⁴), noise
~1e-13·|f|/(1e-3·|∇f·d|). The threshold is unchanged; the h = 1e-6 value is printed alongside.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy import (RAMP, SIMP, Compliance, Constraint, DensityFilter, Domain, Elasticity, Embed,
                          EvalContext, HeavisideProjection, Passive, Response, Study, Volume, gradcheck,
                          standard_chain, study_gradcheck)
from toporlogy.optim import Optimizer

TOL = 1e-6           # pre-registered
NDIR = 5


def _interior(rng: np.random.Generator, shape) -> np.ndarray:
    return 0.2 + 0.6 * rng.random(shape)


def _domain_3d() -> Domain:
    shape = (8, 5, 4)
    mask = np.ones(shape, bool)
    mask[5:, 3:, :] = False                       # L shape
    mask[2:4, 1:3, 1:3] = False                   # cavity
    rng = np.random.default_rng(7)
    frac = np.where(mask, 0.6 + 0.4 * rng.random(shape), 0.0)
    keep = np.zeros(shape, bool)
    keep[-1, :2, :] = True
    void = np.zeros(shape, bool)
    void[4, 0, :2] = True
    return Domain(shape, 0.5, frac=frac, keep=keep, void=void)


def _domain_2d() -> Domain:
    shape = (10, 6)
    mask = np.ones(shape, bool)
    mask[6:, 4:] = False
    mask[3:5, 2:4] = False
    keep = np.zeros(shape, bool)
    keep[-1, :2] = True
    void = np.zeros(shape, bool)
    void[1, 5] = True
    return Domain(shape, 1.0, frac=mask.astype(float), keep=keep, void=void, thickness=0.7)


def _report(name: str, r: dict) -> float:
    print(f"\n  gradcheck {name}: max_rel_error = {r['max_rel_error']:.2e}")
    return r["max_rel_error"]


def _fd4(name: str, check, h: float = 1e-3) -> float:
    """Richardson-extrapolated directional check: ``check(h)`` must call gradcheck / study_gradcheck with a fixed
    seed. Returns the max relative error of (4 fd(h) − fd(2h))/3 against autograd."""
    r1, r2, plain = check(h), check(2 * h), check(1e-6)
    err = 0.0
    for a, b in zip(r1["directions"], r2["directions"], strict=True):
        assert a["ad"] == b["ad"]
        fd = (4 * a["fd"] - b["fd"]) / 3
        err = max(err, abs(fd - a["ad"]) / max(abs(fd), abs(a["ad"]), 1e-300))
    print(f"\n  gradcheck {name}: max_rel_error = {err:.2e} (5-point, h={h:g}); "
          f"central h=1e-6: {plain['max_rel_error']:.2e}")
    return err


def _weights(rng, shape) -> torch.Tensor:
    return torch.as_tensor(rng.standard_normal(shape), dtype=torch.float64)


# ---------------------------------------------------------------------------------------------- chain links alone
@pytest.mark.parametrize("dim", [2, 3])
def test_embed(dim):
    d = _domain_3d() if dim == 3 else _domain_2d()
    rng = np.random.default_rng(1)
    W, link = _weights(rng, d.shape), Embed(d)
    r = gradcheck(lambda x: (W * link(x) ** 2).sum(), _interior(rng, d.n_design), n_dirs=NDIR)
    assert _report(f"Embed {dim}-D", r) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_density_filter_masked(dim):
    d = _domain_3d() if dim == 3 else _domain_2d()
    rng = np.random.default_rng(2)
    W = _weights(rng, d.shape)
    for rmin in (1.5, 2.6):
        link = DensityFilter(d, rmin)
        r = gradcheck(lambda g, link=link: (W * link(g) ** 2).sum(), _interior(rng, d.shape), n_dirs=NDIR)
        assert _report(f"DensityFilter {dim}-D rmin={rmin}", r) <= TOL


@pytest.mark.parametrize("beta", [1.0, 8.0])
def test_heaviside_projection(beta):
    rng = np.random.default_rng(3)
    W = _weights(rng, (6, 5, 3))
    link = HeavisideProjection(beta, 0.45)
    r = gradcheck(lambda g: (W * link(g)).sum(), _interior(rng, (6, 5, 3)), n_dirs=NDIR)
    assert _report(f"HeavisideProjection beta={beta}", r) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_passive_keep_void(dim):
    d = _domain_3d() if dim == 3 else _domain_2d()
    rng = np.random.default_rng(4)
    W, link = _weights(rng, d.shape), Passive(d)
    g0 = _interior(rng, d.shape)
    r = gradcheck(lambda g: (W * link(g) ** 2).sum(), g0, n_dirs=NDIR)
    assert _report(f"Passive {dim}-D", r) <= TOL
    gt = torch.as_tensor(g0).requires_grad_(True)
    (W * link(gt)).sum().backward()
    frozen = d.keep | d.void | ~d.active
    assert torch.all(gt.grad[torch.as_tensor(frozen)] == 0)
    assert torch.allclose(gt.grad[torch.as_tensor(d.design)], W[torch.as_tensor(d.design)])


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("beta", [1.0, 8.0])
def test_standard_chain_composed(dim, beta):
    d = _domain_3d() if dim == 3 else _domain_2d()
    rng = np.random.default_rng(5)
    W = _weights(rng, d.shape)
    chain = standard_chain(d, 1.8, projection=True, beta=beta, eta=0.5)
    r = gradcheck(lambda x: (W * chain(x) ** 2).sum(), _interior(rng, d.n_design), n_dirs=NDIR)
    assert _report(f"standard_chain {dim}-D beta={beta}", r) <= TOL


@pytest.mark.parametrize("kind", ["SIMP", "RAMP"])
def test_interpolation(kind):
    d = _domain_3d()
    rng = np.random.default_rng(6)
    W = _weights(rng, d.shape)
    interp = SIMP(d, E0=2.0, penal=3.0, emin_ratio=1e-6) if kind == "SIMP" else RAMP(d, E0=2.0, q=8.0)
    r = gradcheck(lambda g: (W * interp(g)).sum(), _interior(rng, d.shape), n_dirs=NDIR)
    assert _report(kind, r) <= TOL


# ---------------------------------------------------------------------------------------------- responses
class TipDisplacement(Response):
    """u at one DOF of one load case — NOT self-adjoint when the DOF is unloaded (adjoint solve)."""

    name = "tip_u"
    needs = frozenset({"u"})

    def __init__(self, dof: int, case: int = 0, sign: float = 1.0) -> None:
        self.dof, self.case, self.sign = dof, case, sign

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return self.sign * ctx.U[self.dof, self.case]


class DiagonalEnergy(Response):
    """Σ_c Uᵀ M U with a fixed positive diagonal M (∂J/∂U = 2 M U, not a multiple of F)."""

    name = "umu"
    needs = frozenset({"u"})

    def __init__(self, m: np.ndarray) -> None:
        self.m = torch.as_tensor(m, dtype=torch.float64)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return (self.m[:, None] * ctx.U ** 2).sum()


def _study(d: Domain, objective: Response, constraints=(), n_cases: int = 1, beta: float = 4.0,
           interp=None) -> Study:
    """Cantilever on the masked domain: clamped at x = 0, loads at the free end (case 2: transverse y)."""
    eps = 1e-6 * d.h
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: abs(X[:, 0] - d.shape[0] * d.h) < eps)
    F = d.load_vector(n_cases)
    F[d.dofs(tip, d.dim - 1), 0] = -1.0 / len(tip)
    if n_cases > 1:
        F[d.dofs(tip, 1 if d.dim == 3 else 0), 1] = 0.7 / len(tip)
    chain = standard_chain(d, 1.6, projection=True, beta=beta)
    return Study(d, Elasticity(d, fixed), F, chain, interp or SIMP(d, emin_ratio=1e-6), objective,
                 list(constraints), Optimizer())


@pytest.mark.parametrize("dim", [2, 3])
def test_compliance_single_case(dim):
    d = _domain_3d() if dim == 3 else _domain_2d()
    st = _study(d, Compliance())
    n_adj = st.physics.n_adjoint_solves
    x = _interior(np.random.default_rng(8), d.n_design)
    assert _fd4(f"Compliance {dim}-D", lambda h: study_gradcheck(st, x, n_dirs=NDIR, h=h)) <= TOL
    assert st.physics.n_adjoint_solves == n_adj, "compliance must use the self-adjoint shortcut (no extra solve)"


def test_compliance_multi_case_weighted_ramp():
    d = _domain_3d()
    st = _study(d, Compliance(weights=[0.3, 2.5]), n_cases=2, interp=RAMP(d, q=5.0))
    n_adj = st.physics.n_adjoint_solves
    x = _interior(np.random.default_rng(9), d.n_design)
    assert _fd4("Compliance 2 cases weighted (RAMP)", lambda h: study_gradcheck(st, x, n_dirs=NDIR, h=h)) <= TOL
    assert st.physics.n_adjoint_solves == n_adj


def test_volume_weighted_and_unweighted():
    d = _domain_3d()
    st = _study(d, Compliance(), [Constraint(Volume(), 0.4), Constraint(Volume(weighted=True), 0.35, scale=3.0)])
    x = _interior(np.random.default_rng(10), d.n_design)
    for i, name in enumerate(["Volume", "Volume(weighted)"]):
        r = study_gradcheck(st, x, which=i, n_dirs=NDIR)
        assert _report(name, r) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_non_self_adjoint_displacement(dim):
    d = _domain_3d() if dim == 3 else _domain_2d()
    st0 = _study(d, Compliance())
    # an unloaded free DOF in the middle of the part, transverse direction
    tipish = d.nodes_where(lambda X: abs(X[:, 0] - (d.shape[0] // 2) * d.h) < 1e-9)
    dof = int(np.intersect1d(d.dofs(tipish, d.dim - 1), st0.physics.free)[0])
    assert not np.any(st0.F[dof])
    st = _study(d, TipDisplacement(dof, sign=-1.0))
    n_adj = st.physics.n_adjoint_solves
    x = _interior(np.random.default_rng(11), d.n_design)
    assert _fd4(f"displacement (adjoint) {dim}-D", lambda h: study_gradcheck(st, x, n_dirs=NDIR, h=h)) <= TOL
    assert st.physics.n_adjoint_solves > n_adj, "the adjoint branch was not exercised"


def test_non_self_adjoint_umu_two_cases():
    d = _domain_3d()
    m = 0.5 + np.random.default_rng(12).random(d.ndof)
    st = _study(d, DiagonalEnergy(m), n_cases=2)
    n_adj = st.physics.n_adjoint_solves
    x = _interior(np.random.default_rng(13), d.n_design)
    assert _fd4("U^T M U, 2 cases (adjoint)", lambda h: study_gradcheck(st, x, n_dirs=NDIR, h=h)) <= TOL
    assert st.physics.n_adjoint_solves > n_adj


def test_mixed_self_adjoint_and_adjoint_cases():
    """Case 0 enters as compliance (shortcut), case 1 through a displacement (adjoint): one backward, both paths."""

    class Mixed(Response):
        name = "mixed"
        needs = frozenset({"u"})

        def __init__(self, dof):
            self.dof = dof

        def __call__(self, ctx):
            F0 = torch.as_tensor(ctx.F[:, 0])
            return (F0 * ctx.U[:, 0]).sum() + 50.0 * ctx.U[self.dof, 1]

    d = _domain_3d()
    st0 = _study(d, Compliance(), n_cases=2)
    dof = int(np.flatnonzero(st0.F[:, 1])[0])     # loaded in case 1 → ∂J/∂U_1 is NOT ∝ F_1 (single entry)
    st = _study(d, Mixed(dof), n_cases=2)
    x = _interior(np.random.default_rng(14), d.n_design)
    assert _fd4("compliance case 0 + displacement case 1",
                lambda h: study_gradcheck(st, x, n_dirs=NDIR, h=h)) <= TOL


def test_evaluate_objective_and_u_constraint():
    """Study.evaluate differentiates the objective and a U-dependent constraint from ONE forward solve.

    Both gradients are compared with five-point differences (h = 1e-3, see module docstring) of
    :meth:`Study.evaluate` values along random directions.
    """
    d = _domain_3d()
    st0 = _study(d, Compliance())
    dof = int(np.intersect1d(d.dofs(d.nodes_where(lambda X: abs(X[:, 0] - 2 * d.h) < 1e-9), 2),
                             st0.physics.free)[0])
    st = _study(d, Compliance(), [Constraint(TipDisplacement(dof, sign=-1.0), 1e-3)])
    x = _interior(np.random.default_rng(15), d.n_design)
    n_fac = st.physics.solver.n_factor
    ev = st.evaluate(x)
    assert st.physics.solver.n_factor == n_fac + 1, "one factorisation for objective and constraint"
    rng = np.random.default_rng(16)
    h, worst = 1e-3, 0.0

    def vals(xx):
        e = st.evaluate(xx, gradients=False)
        return np.array([e.f0, e.g[0]])

    for _ in range(NDIR):
        v = rng.standard_normal(x.size)
        v /= np.linalg.norm(v)
        fd = (8 * (vals(x + h * v) - vals(x - h * v)) - (vals(x + 2 * h * v) - vals(x - 2 * h * v))) / (12 * h)
        for f, ad in zip(fd, (ev.df0 @ v, ev.dg[0] @ v), strict=True):
            worst = max(worst, abs(f - ad) / max(abs(f), abs(ad)))
    print(f"\n  gradcheck evaluate(objective + U-constraint): max_rel_error = {worst:.2e}")
    assert worst <= TOL


def test_factorisation_token_restored_in_backward():
    """Forward at x1, forward at x2 (refactorises), then backward of the x1 graph: the solve must refactor K(x1).

    The gradient must equal the one from a fresh forward/backward at x1; afterwards the x2 graph's backward must
    also still be right (it refactors K(x2) again).
    """
    d = _domain_3d()
    st0 = _study(d, Compliance())
    dof = int(np.intersect1d(d.dofs(d.nodes_where(lambda X: abs(X[:, 0] - 3 * d.h) < 1e-9), 2),
                             st0.physics.free)[0])
    st = _study(d, TipDisplacement(dof))
    rng = np.random.default_rng(17)
    x1, x2 = _interior(rng, d.n_design), _interior(rng, d.n_design)

    def graph(x):
        xt = torch.as_tensor(x).clone().requires_grad_(True)
        return xt, st.objective(st.context(xt, True))

    def fresh_grad(x):
        xt, J = graph(x)
        J.backward()
        return xt.grad.numpy().copy()

    g1_ref, g2_ref = fresh_grad(x1), fresh_grad(x2)
    a, Ja = graph(x1)
    tok_a = st.physics._token
    b, Jb = graph(x2)
    assert st.physics._token == tok_a + 1
    n_fac = st.physics.solver.n_factor
    Ja.backward()
    assert st.physics.solver.n_factor == n_fac + 1, "backward of the x1 graph must refactorise K(x1)"
    Jb.backward()
    assert st.physics.solver.n_factor == n_fac + 2, "backward of the x2 graph must refactorise K(x2)"
    e1 = np.max(np.abs(a.grad.numpy() - g1_ref)) / np.max(np.abs(g1_ref))
    e2 = np.max(np.abs(b.grad.numpy() - g2_ref)) / np.max(np.abs(g2_ref))
    print(f"\n  token restore: rel diff x1 {e1:.2e}, x2 {e2:.2e}")
    assert e1 <= 1e-10 and e2 <= 1e-10

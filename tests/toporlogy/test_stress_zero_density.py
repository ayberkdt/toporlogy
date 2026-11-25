"""The derivative of the stress relaxation at zero density (docs/olcumler/gerilme_onkayit.md §6, Z1-Z4 (Z5 is the example run in gerilme.md §3.5),
pre-registered 2026-11-26 01:32 +0300 before ``optimization/stress.py`` changed).

For 0 < a < 1 the one-sided derivative of x̄^a at x̄ = 0 is +∞; torch turned the zero incoming gradient of a p-norm
aggregate times it into NaN and the L-bracket examples stopped with non-finite moduli. The library now sets the
derivative there to 0 and leaves every value unchanged. The old formulas are written out here as the reference.
"""
from __future__ import annotations

import pytest
import torch

from toporlogy.core.domain import Domain
from toporlogy.core.material import Strength
from toporlogy.fem.elastic import Elasticity
from toporlogy.optimization.local import Aggregated
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.strength import make_criterion
from toporlogy.optimization.stress import FailureIndex, VonMises


def _problem(seed: int = 0):
    d = Domain((6, 4))
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    F = d.load_vector()
    F[2 * d.nodes_where(lambda X: (X[:, 0] == 6) & (X[:, 1] == 0))[0] + 1, 0] = -1.0
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(d.shape, dtype=torch.float64, generator=g)
    x[2, 3] = x[0, 0] = x[5, 1] = 0.0                                  # exact zeros
    return d, fixed, F, x


def _fields(d, form, a):
    q, penal = (a, 3.0) if form != "polynomial" else (0.5, a)
    return [VonMises(d, 1.0, 0.3, E0=1.0, form=form, q=q, penal=penal),
            FailureIndex(d, make_criterion("von_mises", Strength(1.0)), E0=1.0, form=form, q=q, penal=penal)]


def _old(field, ctx, x):
    """The formula of ``VonMises.__call__`` before the change (gerilme_onkayit.md §6, Z2)."""
    r = field.solid_ratio(ctx)
    xe = x.reshape(-1)[torch.as_tensor(field.elements)][:, None]
    if field.form == "polynomial":
        lam = r - 1.0
        return xe ** field.penal * lam * (lam * lam + 1.0)
    v = r if field.q == 0 else xe.clamp_min(0.0) ** field.q * r
    return v if field.form == "ratio" else v - 1.0


def test_z1_aggregated_gradient_is_finite_at_zero_density():
    d, fixed, F, _ = _problem()
    x = torch.full(d.shape, 0.6, dtype=torch.float64)
    x[2, 3] = 0.0
    x.requires_grad_(True)
    E = 1e-6 + x ** 3
    U = Elasticity(d, fixed, 0.3).solve(E.reshape(-1), F)
    value = Aggregated(VonMises(d, 1.0, 0.3, E0=1.0, q=0.5), "pnorm", 8.0)(EvalContext(d, x, E, U, F))
    (g,) = torch.autograd.grad(value, x)
    assert bool(torch.isfinite(g).all())


CASES = [(form, a) for form in ("ratio", "margin") for a in (0.5, 0.25)] + [("polynomial", 0.5), ("polynomial", 0.25)]


@pytest.mark.parametrize(("form", "a"), CASES)
def test_z2_z3_values_unchanged_derivative_zero_only_at_zero_density(form, a):
    d, fixed, F, x0 = _problem()
    E = 1e-6 + x0 ** 3
    U = Elasticity(d, fixed, 0.3).solve(E.reshape(-1), F).detach()    # the state does not depend on x here
    w = torch.rand((d.n_active, 1), dtype=torch.float64, generator=torch.Generator().manual_seed(1)) + 0.5
    for field in _fields(d, form, a):
        x_new = x0.clone().requires_grad_(True)
        x_old = x0.clone().requires_grad_(True)
        new = field(EvalContext(d, x_new, E, U, F))
        old = _old(field, EvalContext(d, x_old, E, U, F), x_old)
        assert torch.equal(new, old), f"Z2 {type(field).__name__} {form} a={a}"
        (g_new,) = torch.autograd.grad((w[field.elements] * new).sum(), x_new)
        (g_old,) = torch.autograd.grad((w[field.elements] * old).sum(), x_old)
        zero = x0 == 0
        assert torch.equal(g_new[~zero], g_old[~zero]), f"Z3 {type(field).__name__} {form} a={a}"
        assert bool((g_new[zero] == 0).all()), f"Z3 zero {type(field).__name__} {form} a={a}"


@pytest.mark.parametrize(("form", "a"), [("ratio", 1.0), ("polynomial", 3.0)])
def test_z4_exponents_with_a_finite_derivative_are_unchanged(form, a):
    d, fixed, F, x0 = _problem()
    E = 1e-6 + x0 ** 3
    U = Elasticity(d, fixed, 0.3).solve(E.reshape(-1), F).detach()
    for field in _fields(d, form, a):
        x_new = x0.clone().requires_grad_(True)
        x_old = x0.clone().requires_grad_(True)
        (g_new,) = torch.autograd.grad(field(EvalContext(d, x_new, E, U, F)).sum(), x_new)
        (g_old,) = torch.autograd.grad(_old(field, EvalContext(d, x_old, E, U, F), x_old).sum(), x_old)
        assert torch.equal(g_new, g_old)
        if form == "ratio":                                            # derivative of x · r at 0 is r, not 0
            assert bool((g_new[x0 == 0] != 0).all())

"""Stress field, aggregated constraint and augmented Lagrangian (optimization/stress.py, local.py).

Closed forms: uniform tension (2-D and 3-D, exact for the element), pure bending of a 2-D strip (within 3 %: the
bilinear element is not exact in bending), von Mises identities.
Gradients: pre-registered 1e-6 with the Richardson check of test_gradients (responses through the linear solve).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy import (MMA, SIMP, Compliance, Constraint, Domain, Elasticity, EvalContext, Study, Volume,
                       standard_chain, study_gradcheck)
from toporlogy.optimization.local import Aggregated, AugmentedLagrangian, FunctionField, MultiplierUpdate
from toporlogy.optimization.stress import VonMises, centroid_stress, von_mises_matrix

TOL = 1e-6


def _fd4(check, h: float = 1e-3) -> float:
    r1, r2 = check(h), check(2 * h)
    err = 0.0
    for a, b in zip(r1["directions"], r2["directions"], strict=True):
        fd = (4 * a["fd"] - b["fd"]) / 3
        err = max(err, abs(fd - a["ad"]) / max(abs(fd), abs(a["ad"]), 1e-300))
    return err


def test_von_mises_matrix_identities() -> None:
    rng = np.random.default_rng(0)
    V3, V2 = von_mises_matrix(3), von_mises_matrix(2)
    for _ in range(20):
        s = rng.standard_normal(6)
        sx, sy, sz, tyz, txz, txy = s
        ref = 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (tyz ** 2 + txz ** 2 + txy ** 2)
        assert abs(s @ V3 @ s - ref) < 1e-12
        p = rng.standard_normal()
        hyd = s + np.array([p, p, p, 0, 0, 0])
        assert abs(hyd @ V3 @ hyd - ref) < 1e-10          # hydrostatic stress does not change it
        a = np.array([sx, sy, txy])
        assert abs(a @ V2 @ a - np.array([sx, sy, 0, 0, 0, txy]) @ V3 @ np.array([sx, sy, 0, 0, 0, txy])) < 1e-12


def _tension_2d(nx: int, ny: int, h: float, t: float, traction: float):
    d = Domain((nx, ny), h, thickness=t)
    X = d.node_coords()
    left = np.flatnonzero(X[:, 0] == 0)
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(left[:1], 1)])
    right = np.flatnonzero(np.isclose(X[:, 0], nx * h))
    F = d.load_vector()
    w = np.full(len(right), h)
    w[[0, -1]] = h / 2                                    # consistent nodal loads of a uniform traction
    F[2 * right, 0] = traction * t * w
    return d, fixed, F


def test_uniform_tension_2d_exact() -> None:
    E0, nu, s0 = 210.0, 0.3, 4.0
    d, fixed, F = _tension_2d(6, 4, 0.5, 0.7, s0)
    U = torch.as_tensor(Elasticity(d, fixed, nu).solve_numpy(np.full(d.n_active, E0), F))
    s = centroid_stress(d, U, nu, E0)
    assert torch.allclose(s[:, 0, 0], torch.full((d.n_active,), s0, dtype=torch.float64), rtol=1e-10)
    assert float(s[:, 1:, 0].abs().max()) < 1e-9
    ctx = EvalContext(d, torch.ones(d.shape, dtype=torch.float64), torch.ones(d.shape), U, F)
    vm = VonMises(d, sigma_lim=2.0, nu=nu, E0=E0, q=0.0)
    assert torch.allclose(vm(ctx), torch.full((d.n_active, 1), 2.0, dtype=torch.float64), rtol=1e-9)
    assert abs(vm.last_solid_ratio - 2.0) < 1e-8


def test_uniform_tension_3d_exact() -> None:
    E0, nu, s0, h = 70.0, 0.25, 3.0, 0.5
    d = Domain((4, 3, 2), h)
    X = d.node_coords()
    left = np.flatnonzero(X[:, 0] == 0)
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(left[:1], [1, 2]), d.dofs(left[1:2], 1)])
    right = np.flatnonzero(np.isclose(X[:, 0], 4 * h))
    F = d.load_vector()
    yz = X[right][:, 1:]
    w = np.ones(len(right))
    for k, n in ((0, 3), (1, 2)):
        w *= np.where(np.isclose(yz[:, k], 0) | np.isclose(yz[:, k], n * h), 0.5, 1.0)
    F[3 * right, 0] = s0 * h * h * w
    for element in ("H8", "H8IM"):
        U = torch.as_tensor(Elasticity(d, fixed, nu, element).solve_numpy(np.full(d.n_active, E0), F))
        s = centroid_stress(d, U, nu, E0)
        assert torch.allclose(s[:, 0, 0], torch.full((d.n_active,), s0, dtype=torch.float64), rtol=1e-9), element
        assert float(s[:, 1:, 0].abs().max()) < 1e-8, element


def test_pure_bending_centroid_stress() -> None:
    """Strip under linearly varying end traction σ_x = k (y − H/2): the centroid stress of row j is k (y_c − H/2).
    Not exact for the bilinear element (parasitic shear makes it too stiff in bending): measured 21 Nov 2026 on this
    12 x 6 strip, the worst centroid stress is 1.8 % off; the pre-set tolerance is 3 % of the outer-fibre value."""
    nx, ny, h, k, E0, nu = 12, 6, 1.0, 2.0, 100.0, 0.0
    d = Domain((nx, ny), h)
    X = d.node_coords()
    left = np.flatnonzero(X[:, 0] == 0)
    mid = left[np.argmin(np.abs(X[left, 1] - ny * h / 2))]
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(np.array([mid]), 1)])
    right = np.flatnonzero(np.isclose(X[:, 0], nx * h))
    F = d.load_vector()
    y = X[right, 1]
    for a, b in zip(right[:-1], right[1:], strict=True):   # consistent loads of a linear traction on each edge
        ya, yb = X[a, 1], X[b, 1]
        sa, sb = k * (ya - ny * h / 2), k * (yb - ny * h / 2)
        F[2 * a, 0] += h * (2 * sa + sb) / 6
        F[2 * b, 0] += h * (sa + 2 * sb) / 6
    assert abs(F[:, 0].sum()) < 1e-12 and y.size == ny + 1
    U = torch.as_tensor(Elasticity(d, fixed, nu).solve_numpy(np.full(d.n_active, E0), F))
    s = centroid_stress(d, U, nu, E0).numpy()[:, :, 0]
    yc = d.element_centers()[:, 1]
    ref = k * (yc - ny * h / 2)
    err = np.abs(s[:, 0] - ref).max() / np.abs(ref).max()
    print(f"\n  bending: worst centroid stress error {err:.2%} of the outer-fibre value")
    assert err < 0.03
    assert np.abs(s[:, 1:]).max() < 0.03 * np.abs(ref).max()


def _study(dim: int, response_of, objective=None):
    if dim == 2:
        shape = (10, 6)
        mask = np.ones(shape, bool)
        mask[6:, 4:] = False
        d = Domain(shape, 1.0, frac=mask.astype(float), thickness=0.7)
    else:
        shape = (6, 4, 3)
        mask = np.ones(shape, bool)
        mask[4:, 2:, :] = False
        d = Domain(shape, 0.5, frac=mask.astype(float))
    X = d.node_coords()
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] == 0))
    F = d.load_vector(2)
    tip = d.nodes_where(lambda X: np.isclose(X[:, 0], X[:, 0].max()))
    F[d.dim * tip + 1, 0] = -1.0 / len(tip)
    F[d.dim * tip, 1] = 0.7 / len(tip)
    chain = standard_chain(d, 1.5, beta=2.0)
    interp = SIMP(d, E0=10.0)
    r = response_of(d)
    obj = objective(d, r) if objective is not None else Compliance()
    cons = [Constraint(Volume(), 0.5)] if objective is not None else [Constraint(r, 1.0)]
    assert X.shape[0] == d.nn
    return Study(d, Elasticity(d, fixed, 0.3), F, chain, interp, obj, cons, MMA()), d


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["pnorm", "pmean", "ks", "ks_lower", "induced_exponential", "induced_power"])
def test_gradient_aggregated_stress(dim: int, kind: str) -> None:
    study, d = _study(dim, lambda d: Aggregated(VonMises(d, 0.8, 0.3, form="ratio", q=0.5), kind, 8.0))
    x0 = 0.2 + 0.6 * np.random.default_rng(3).random(d.n_design)
    err = _fd4(lambda h: study_gradcheck(study, x0, 0, n_dirs=4, h=h, seed=5))
    assert err <= TOL, err


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("form", ["margin", "polynomial"])
def test_gradient_augmented_lagrangian(dim: int, form: str) -> None:
    def obj(d, r):
        al = AugmentedLagrangian(Volume(), VonMises(d, 0.8, 0.3, form=form), mu0=5.0, reference=1.0)
        return al
    study, d = _study(dim, lambda d: None, obj)
    al = study.objective
    x0 = 0.2 + 0.6 * np.random.default_rng(4).random(d.n_design)
    study.evaluate(x0)                                    # fills last_g
    al.update()                                           # non-zero multipliers: both branches of h are exercised
    assert float(al.lam.max()) > 0 and float(al.lam.min()) >= 0 and al.mu == pytest.approx(5.5)
    err = _fd4(lambda h: study_gradcheck(study, x0, "objective", n_dirs=4, h=h, seed=6))
    assert err <= TOL, err


def test_adaptive_scale_reaches_true_maximum() -> None:
    study, d = _study(2, lambda d: Aggregated(VonMises(d, 0.8, 0.3), "pmean", 8.0, adaptive=1.0))
    r = study.constraints[0].response
    x0 = np.full(d.n_design, 0.6)
    ev = study.evaluate(x0)
    assert ev.values[r.name] == pytest.approx(r.last_max, rel=1e-12)      # a = 1: scaled value = true maximum
    assert r.last_aggregate < r.last_max
    c = r.scale.value
    study.evaluate(x0, gradients=False)                   # no update without gradients (line searches, final check)
    assert r.scale.value == c
    r.reset_scale()
    assert r.scale.value is None


def test_augmented_lagrangian_solves_a_small_problem() -> None:
    """min Σx  s.t. x_j ≥ a_j (g_j = a_j − x_j ≤ 0): the solution is x = a, multipliers 1/N · N = f-scale."""
    a = torch.as_tensor([0.2, 0.5, 0.8, 0.35], dtype=torch.float64)
    field = FunctionField(lambda ctx: a - ctx.x, "lower")

    class Sum(Volume):
        name = "sum"

        def __call__(self, ctx):
            return ctx.x.sum()

    al = AugmentedLagrangian(Sum(), field, mu0=4.0, alpha=1.5, mu_max=1e3, reference=1.0)
    sched = MultiplierUpdate(al, every=1, tol=1e-6)
    x = torch.full((4,), 0.5, dtype=torch.float64)
    for _ in range(40):
        for _ in range(200):                              # inner: plain gradient descent, step 1/(mu/N)
            xt = x.clone().requires_grad_(True)
            ctx = EvalContext(None, xt, xt, x=xt)          # type: ignore[arg-type]
            (g,) = torch.autograd.grad(al(ctx), xt)
            x = (x - g * (4.0 / al.mu) * 0.9).clamp(0.0, 1.0)
        assert sched.update(0, 0.0) is False              # default: the optimizer is not reset at an update
    assert torch.allclose(x, a, atol=1e-6), x
    assert torch.allclose(al.lam, torch.full((4,), 4.0, dtype=torch.float64), atol=1e-4)   # dJ/dx = 1 = lam/N
    assert sched.finished and set(sched.state()) == {"al_mu", "al_updates"}
    sched.reset()
    assert al.lam is None and al.mu == 4.0 and not sched.finished
    assert MultiplierUpdate(al, every=1, reset_optimizer=True).update(0, 0.0) is True


def test_field_options_and_errors() -> None:
    d = Domain((6, 4), 1.0)
    ex = np.zeros(d.shape, bool)
    ex[0, :] = True
    vm = VonMises(d, 1.0, exclude=ex, cases=[1])
    assert len(vm.elements) == 20
    U = torch.zeros((d.ndof, 3), dtype=torch.float64)
    ctx = EvalContext(d, torch.ones(d.shape, dtype=torch.float64), torch.ones(d.shape), U, None, interp=SIMP(d))
    assert vm(ctx).shape == (20, 1) and vm.weights(ctx).shape == (20, 1)
    for bad in (lambda: VonMises(d, 0.0), lambda: VonMises(d, 1.0, form="x"), lambda: VonMises(d, 1.0, q=-1),
                lambda: VonMises(d, 1.0, exclude=np.ones(d.shape, bool)),
                lambda: VonMises(d, 1.0, exclude=np.ones((2, 2), bool)),
                lambda: AugmentedLagrangian(Volume(), vm, mu0=0.0), lambda: MultiplierUpdate(None, every=0)):
        with pytest.raises(ValueError):
            bad()

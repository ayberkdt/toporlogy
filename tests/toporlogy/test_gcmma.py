"""GCMMA (optimizers/gcmma.py): conservativeness of accepted steps, the 2-cycle case of plain MMA, monotone
objective on feasible iterates, agreement with MMA on the MBB beam, cost accounting."""
from __future__ import annotations

import numpy as np
import pytest

import toporlogy as T
from toporlogy.optimizers.gcmma import GCMMA, conservative_subproblem
from toporlogy.optimizers.mma import MMA


def _drive(opt, x, fun, n_iter):
    opt.value_fn = lambda xc: (fun(xc)[0], fun(xc)[2])
    hist = []
    for _ in range(n_iter):
        f0, df0, g, dg = fun(x)
        hist.append((f0, float(np.max(g))))
        xn = opt.step(x, f0, df0, g, dg)
        if np.max(np.abs(xn - x)) < 1e-11:
            x = xn
            break
        x = xn
    return x, hist


def _quadratic():
    t = np.array([1.0, 1.0, 1.0, 1.0, 1.4, -0.3])
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], float)
    b = np.array([0.6, 1.0, 1.5])
    return lambda x: (float(np.sum((x - t) ** 2)), 2 * (x - t), A @ x / b - 1.0, A / b[:, None])


def test_subproblem_reproduces_values_and_gradients_at_x() -> None:
    rng = np.random.default_rng(0)
    n, m = 12, 2
    x = rng.uniform(0.2, 0.8, n)
    df, f = rng.normal(size=(m + 1, n)), rng.normal(size=m + 1)
    low, upp = x - 0.5, x + 0.5
    rho = np.array([0.3, 0.2, 0.1])
    _, lam, approx, info = conservative_subproblem(x, low, upp, df, f, rho)
    assert np.allclose(approx(x), f, atol=1e-12) and lam.shape == (m,) and info["residual"] < 1e-5
    h = 1e-6
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        assert np.allclose((approx(x + e) - approx(x - e)) / (2 * h), df[:, j], atol=1e-6)   # first-order exact
    d2 = (approx(x + e) - 2 * approx(x) + approx(x - e)) / h ** 2
    assert np.all(d2 > 0)                                                                    # strictly convex


def test_no_two_cycle_at_an_interior_optimum() -> None:
    """min Σ(x − t)², optimum of the free variables interior: plain MMA ends in a 2-cycle of amplitude 4e-3
    (mma.py notes); GCMMA converges."""
    def fun(x):
        return float(np.sum((x - 0.2) ** 2)), 2 * (x - 0.2), np.array([x.mean() - 0.9]), np.full((1, x.size), 1 / x.size)

    x_g, _ = _drive(GCMMA(), np.full(5, 0.7), fun, 300)
    np.testing.assert_allclose(x_g, 0.2, atol=1e-5)          # measured 3.4e-7 after 300 steps
    x_m = np.full(5, 0.7)
    opt, tail = MMA(), []
    for _ in range(300):
        f0, df0, g, dg = fun(x_m)
        x_m = opt.step(x_m, f0, df0, g, dg)
        tail.append(x_m[0])
    assert np.ptp(tail[-10:]) > 1e-3                      # the documented limitation is still there in plain MMA


def test_quadratic_two_active_constraints_and_monotone_descent() -> None:
    fun = _quadratic()
    opt = GCMMA()
    x, hist = _drive(opt, np.full(6, 0.5), fun, 400)
    np.testing.assert_allclose(x, [0.3, 0.3, 0.5, 0.5, 1.0, 0.0], atol=1e-6)
    lam = opt.kkt_multipliers()
    assert lam[0] > 0 and lam[1] > 0 and abs(lam[2]) < 1e-7
    f = np.array([h[0] for h in hist])
    feas = np.array([h[1] <= 1e-9 for h in hist])
    k0 = int(np.argmax(feas))
    assert feas[k0:].all(), "a feasible iterate must stay feasible"
    assert np.all(np.diff(f[k0:]) <= 1e-9 * (1 + np.abs(f[k0:-1]))), "objective must not increase once feasible"
    assert opt.n_not_conservative == 0 and opt.n_inner_total >= len(hist) - 1


def test_strong_curvature_needs_inner_iterations() -> None:
    """f = 200 Σ (x − 0.45)²: the curvature (400) exceeds that of the first MMA approximation (about 4 |∂f| + ρ), so
    the first candidate overshoots and is not conservative; the inner loop must raise ρ. (A concave objective would
    not do: it lies below its tangent, so every convex approximation is conservative.)"""
    def fun(x):
        return (float(200 * np.sum((x - 0.45) ** 2)), 400 * (x - 0.45), np.array([x.mean() - 0.9]),
                np.full((1, x.size), 1 / x.size))

    opt = GCMMA()
    x, hist = _drive(opt, np.linspace(0.48, 0.6, 8), fun, 200)
    f = np.array([h[0] for h in hist])
    # monotone up to the acceptance tolerance: conservative_tol (1e-9) in units of the scaled objective f / f(x0),
    # i.e. 1.5e-8 here; measured: the iterates end in a cycle of amplitude 1.3e-9 in f at f = 5e-9
    assert np.all(np.diff(f) <= 1e-9 * (1 + f[:-1] / f[0]) * f[0])
    assert opt.n_inner_total > len(hist) and opt.n_not_conservative == 0   # some step tested several candidates
    np.testing.assert_allclose(x, 0.45, atol=1e-5)


def _mbb(optimizer):
    d = T.Domain((40, 14))
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    roller = d.nodes_where(lambda X: (X[:, 0] == 40) & (X[:, 1] == 0))
    top_left = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == 14))
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(roller, 1)])
    F = d.load_vector()[:, 0]
    F[d.dofs(top_left, 1)] = -1.0
    return T.Study(d, T.Elasticity(d, fixed), F, T.standard_chain(d, 1.8, projection=False),
                   T.SIMP(d, emin_ratio=1e-9), T.Compliance(), [T.Constraint(T.Volume(), 0.5)], optimizer)


def test_mbb_gcmma_vs_mma() -> None:
    rg = _mbb(GCMMA()).run(max_iter=60, tol=1e-3)
    rm = _mbb(MMA()).run(max_iter=60, tol=1e-3)
    print(f"\n  MBB 40x14, 60 iterations: GCMMA c = {rg.objective:.4f} ({rg.settings['inner_evaluations']} inner "
          f"evaluations), MMA c = {rm.objective:.4f}")
    assert rg.feasible and rg.max_violation <= 1e-3
    f = np.array(rg.history["f0"])
    v = np.array(rg.history["volume"])
    k0 = int(np.argmax(v <= 0.5 * (1 + 1e-9)))
    assert np.all(np.diff(f[k0:]) <= 1e-7 * f[k0:-1]), "compliance must not increase once the volume bound holds"
    assert rg.objective <= 1.05 * rm.objective            # not worse than plain MMA by more than 5 %
    assert rg.settings["inner_evaluations"] >= rg.iterations
    assert _mbb(GCMMA()).optimizer.value_fn is None       # Study clears what it set


def test_validation() -> None:
    for bad in (lambda: GCMMA(move=0.0), lambda: GCMMA(normalise="current"), lambda: GCMMA(rho_init=0.0),
                lambda: GCMMA(asydecr=1.5)):
        with pytest.raises(ValueError):
            bad()
    opt = GCMMA()
    with pytest.raises(ValueError, match="value_fn"):
        opt.step(np.full(3, 0.5), 1.0, np.ones(3), np.array([0.0]), np.ones((1, 3)))
    with pytest.raises(ValueError, match="at least one constraint"):
        opt.step(np.full(3, 0.5), 1.0, np.ones(3), np.zeros(0), np.zeros((0, 3)))

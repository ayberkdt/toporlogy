"""SiMPL (optimizers/simpl.py): helper identities, problems with known solutions, the MBB beam against OC.

The paper could not be read (see the module notes); these tests check what the implementation claims: bounds by
construction, the constraint met after every step, descent with a line search, and convergence to known optima."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize as so

import toporlogy as T
from toporlogy.optimizers.simpl import SiMPL, bregman_divergence, logit, sigmoid


def test_helpers() -> None:
    psi = np.array([-800.0, -30.0, -1.0, 0.0, 2.0, 40.0, 900.0])
    s = sigmoid(psi)
    assert np.all(np.isfinite(s)) and s[0] == 0.0 and s[3] == 0.5 and s[-1] == 1.0
    assert np.allclose(logit(s[2:5]), psi[2:5]) and abs(logit(np.array([0.0]))[0]) < 28   # clipped at 1e-12
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0.01, 0.99, 50), rng.uniform(0.01, 0.99, 50)
    assert bregman_divergence(a, a) == 0.0 and bregman_divergence(a, b) > 0
    assert bregman_divergence(np.array([0.0, 1.0]), np.array([0.3, 0.6])) == pytest.approx(-np.log(0.7) - np.log(0.6))
    # D is the Bregman divergence of the entropy: D(a, b) = phi(a) - phi(b) - phi'(b) (a - b), phi' = logit
    phi = lambda v: np.sum(v * np.log(v) + (1 - v) * np.log(1 - v))
    assert bregman_divergence(a, b) == pytest.approx(phi(a) - phi(b) - logit(b) @ (a - b), rel=1e-10)


def _drive(opt, x, fun, vmax, n_iter=400, tol=1e-9):
    cfn = lambda xc: np.array([xc.mean() / vmax - 1.0])
    opt.value_fn = lambda xc: (fun(xc)[0], cfn(xc))
    fs = []
    for _ in range(n_iter):
        f, df = fun(x)
        fs.append(f)
        xn = opt.step(x, f, df, cfn(x), np.full((1, x.size), 1 / (x.size * vmax)), cfn)
        assert np.all((xn >= 0) & (xn <= 1)) and cfn(xn)[0] <= 1e-8
        done = np.max(np.abs(xn - x)) < tol
        x = xn
        if done:
            break
    return x, np.array(fs)


@pytest.mark.parametrize("line_search", ["armijo", "bregman"])
def test_convex_quadratic_with_volume_bound(line_search: str) -> None:
    rng = np.random.default_rng(1)
    n, vmax = 30, 0.4
    t = rng.uniform(-0.2, 1.2, n)
    w = rng.uniform(0.5, 3.0, n)
    fun = lambda x: (float(np.sum(w * (x - t) ** 2)), 2 * w * (x - t))
    ref = so.minimize(lambda x: fun(x)[0], np.full(n, 0.3), jac=lambda x: fun(x)[1], bounds=[(0, 1)] * n,
                      constraints=[{"type": "ineq", "fun": lambda x: vmax - x.mean()}], method="SLSQP",
                      options={"ftol": 1e-14, "maxiter": 500})
    opt = SiMPL(line_search=line_search)
    x, fs = _drive(opt, np.full(n, vmax), fun, vmax)
    err_f, err_x = fun(x)[0] / ref.fun - 1, float(np.max(np.abs(x - ref.x)))
    print(f"\n  quadratic, {line_search}: {len(fs)} steps, f/f* - 1 = {err_f:.2e}, max|x - x*| = {err_x:.2e}")
    # variables at a bound are approached only asymptotically (x = sigmoid(psi) never reaches 0 or 1)
    assert err_f + 1e-12 >= 0 and err_f < 1e-4 and err_x < 5e-3
    # descent up to the accuracy of the shift (shift_tol = 1e-10): measured increases up to 4.4e-11 at convergence
    assert np.all(np.diff(fs) <= 1e-9 * (1 + np.abs(fs[:-1]))), "a line search must give descent"
    assert opt.last_shift > 0 and opt.n_inner_total >= len(fs) - 1


def test_inactive_constraint_gives_zero_shift_and_interior_optimum() -> None:
    fun = lambda x: (float(np.sum((x - 0.3) ** 2)), 2 * (x - 0.3))
    opt = SiMPL()
    x, _ = _drive(opt, np.full(6, 0.6), fun, 0.9)
    np.testing.assert_allclose(x, 0.3, atol=1e-6)
    assert opt.last_shift == 0.0


def test_linear_objective_goes_to_the_vertices() -> None:
    """min cᵀx, mean(x) ≤ v: no constraint is active for c > 0 (x → 0); for c < 0 the cheapest entries fill up."""
    c = -np.arange(1.0, 11.0)
    x, _ = _drive(SiMPL(line_search="none"), np.full(10, 0.3), lambda x: (float(c @ x), c), 0.3, n_iter=300)
    assert x.mean() <= 0.3 + 1e-8 and np.all(x[-3:] > 0.99) and np.all(x[:6] < 0.01)


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


def test_mbb_against_oc() -> None:
    rs = _mbb(SiMPL()).run(max_iter=80, tol=1e-3)
    ro = _mbb(T.OC()).run(max_iter=80, tol=1e-3)
    grey = lambda r: T.grey_fraction(r.xphys)
    print(f"\n  MBB 40x14, <= 80 iterations: SiMPL c = {rs.objective:.3f} ({rs.iterations} it, "
          f"{rs.settings['inner_evaluations']} line-search evaluations, grey {grey(rs):.3f}); "
          f"OC c = {ro.objective:.3f} ({ro.iterations} it, grey {grey(ro):.3f})")
    assert rs.feasible
    f = np.array(rs.history["f0"])
    assert np.all(np.diff(f) <= 1e-9 * f[:-1]), "Armijo: compliance must decrease"
    assert rs.objective <= 1.05 * ro.objective


def test_validation() -> None:
    for bad in (lambda: SiMPL(line_search="wolfe"), lambda: SiMPL(c1=2.0), lambda: SiMPL(alpha_min=1.0, alpha_max=0.5)):
        with pytest.raises(ValueError):
            bad()
    opt = SiMPL()
    x, one = np.full(4, 0.5), np.full((1, 4), 0.25)
    cfn = lambda xc: np.array([xc.mean() - 0.5])
    with pytest.raises(ValueError, match="constraint_fn"):
        opt.step(x, 1.0, np.ones(4), np.zeros(1), one)
    with pytest.raises(ValueError, match="value_fn"):
        opt.step(x, 1.0, np.ones(4), np.zeros(1), one, cfn)
    with pytest.raises(ValueError, match="volume-like"):
        opt.step(x, 1.0, np.ones(4), np.zeros(1), -one, cfn)
    with pytest.raises(ValueError, match="exactly one"):
        opt.step(x, 1.0, np.ones(4), np.zeros(2), np.ones((2, 4)), cfn)
    opt.value_fn = lambda xc: (1.0, cfn(xc))
    with pytest.raises(ValueError, match="cannot be met"):
        opt.step(x, 1.0, np.ones(4), np.ones(1), one, lambda xc: np.array([1.0]))

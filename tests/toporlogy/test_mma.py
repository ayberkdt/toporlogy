"""MMA optimizer of the new core (toporlogy.optim.mma).

1. Analytic problems with known optima (``step`` alone, m = 1 and m = 2).
2. Oracle: 2-D half MBB beam against ``topo_oracles.topology.topopt(..., optimizer="MMA")``. Same filter,
   SIMP, start, objective scaling (c/c0) and volume constraint (Σx̄/(V n) − 1); the only difference is the
   subproblem solver (old: bisection on the 1-D dual, tol 1e-14; new: L-BFGS-B on the m-D dual), so the histories
   agree to the dual tolerance (measured 13 Nov 2026: ≤ 3.5e-9 relative per iteration, same iteration count).
3. Two constraints (left / right half volume bounds) on the MBB beam.
4. 3-D cantilever with β continuation (1 → 2 → 4): MMA vs OC from the same start; ``reset()`` at every β step.

conftest.py imports toporlogy first (PARDISO before torch).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.optim.mma import MMA


# ----------------------------------------------------------------------------- helpers
def _run_steps(opt: MMA, x: np.ndarray, fun, n_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    for _ in range(n_iter):
        f0, df0, g, dg = fun(x)
        xn = opt.step(x, f0, df0, g, dg)
        if np.max(np.abs(xn - x)) < tol:
            return xn
        x = xn
    return x


def _mbb(nelx: int, nely: int, rmin: float, constraints, optimizer) -> tuple[T.Domain, T.Study]:
    """Half MBB beam: load −1 (y) at the top-left node, x fixed on the left edge, y fixed at the bottom-right node."""
    d = T.Domain((nelx, nely))
    left = d.nodes_where(lambda X: X[:, 0] == 0)
    roller = d.nodes_where(lambda X: (X[:, 0] == nelx) & (X[:, 1] == 0))
    top_left = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == nely))
    assert len(roller) == 1 and len(top_left) == 1
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(roller, 1)])
    F = d.load_vector()[:, 0]
    F[d.dofs(top_left, 1)] = -1.0
    st = T.Study(d, T.Elasticity(d, fixed, element="Q4"), F, T.standard_chain(d, rmin, projection=False),
                 T.SIMP(d, penal=3.0, emin_ratio=1e-9), T.Compliance(), constraints(d), optimizer)
    return d, st


class MaskVolume(T.Response):
    """Mean physical density over a boolean element mask."""

    def __init__(self, mask: np.ndarray, name: str) -> None:
        self.mask = torch.as_tensor(np.asarray(mask, bool))
        self.name = name

    def __call__(self, ctx: T.EvalContext) -> torch.Tensor:
        return ctx.xphys[self.mask].mean()


# ----------------------------------------------------------------------------- 1. analytic
def test_reciprocal_one_constraint_analytic_optimum():
    """min Σ w_j / x_j s.t. Σ x_j ≤ V  →  x_j = V √w_j / Σ √w (KKT), here (0.2, 0.4, 0.6, 0.8)."""
    w = np.array([1.0, 4.0, 9.0, 16.0])
    V = 2.0

    def fun(x):
        return float(np.sum(w / x)), -w / x ** 2, np.array([x.sum() / V - 1.0]), np.full((1, 4), 1.0 / V)

    opt = MMA()
    x = _run_steps(opt, np.full(4, 0.5), fun)
    np.testing.assert_allclose(x, V * np.sqrt(w) / np.sqrt(w).sum(), atol=1e-6)
    assert opt.last_lambda.shape == (1,) and opt.last_lambda[0] > 0


def test_quadratic_two_constraints_and_bounds():
    """min Σ (x_j − t_j)² s.t. x1 + x2 ≤ 0.6, x3 + x4 ≤ 1.0, x5 + x6 ≤ 1.5 (inactive), 0 ≤ x ≤ 1.

    Optimum: (0.3, 0.3, 0.5, 0.5, 1, 0): first two constraints active, the third inactive (λ = 0), x5 at its upper
    bound (t5 = 1.4), x6 at its lower bound (t6 = −0.3).

    Not tested on purpose: a variable with an *interior* optimum that no active constraint touches (e.g. t6 = 0.2).
    Plain MMA (not GCMMA) then ends in a 2-cycle around the optimum, amplitude 0.0043 measured, because the
    asymptotes are clamped at asymin = 0.01 from x (see the module notes of toporlogy.optim.mma)."""
    t = np.array([1.0, 1.0, 1.0, 1.0, 1.4, -0.3])
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], float)
    b = np.array([0.6, 1.0, 1.5])

    def fun(x):
        return float(np.sum((x - t) ** 2)), 2 * (x - t), A @ x / b - 1.0, A / b[:, None]

    opt = MMA()
    x = _run_steps(opt, np.full(6, 0.5), fun, n_iter=500)
    np.testing.assert_allclose(x, [0.3, 0.3, 0.5, 0.5, 1.0, 0.0], atol=1e-6)
    lam = opt.last_lambda
    assert lam[0] > 0 and lam[1] > 0 and abs(lam[2]) < 1e-8


def test_move_limit_reset_and_validation():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(0.2, 0.8, n)
    df0 = rng.normal(size=n)
    g, dg = np.array([0.1]), np.full((1, n), 1.0 / n)
    opt = MMA(move=0.05)
    x1 = opt.step(x, 3.0, df0, g, dg)
    assert np.all(np.abs(x1 - x) <= 0.05 + 1e-12) and np.all((x1 >= 0) & (x1 <= 1))
    opt.step(x1, 2.5, df0, g, dg)
    assert opt.k == 2 and opt.f0_scale == pytest.approx(1 / 3.0)
    opt.reset()
    assert opt.k == 0 and opt.low is None and opt.xold1 is None and opt.f0_scale is None and opt.n_resets == 1
    # after reset the optimizer behaves exactly like a fresh one
    np.testing.assert_array_equal(opt.step(x, 3.0, df0, g, dg), MMA(move=0.05).step(x, 3.0, df0, g, dg))
    # the objective scale makes the step invariant to a positive factor on f0 (normalise="first")
    np.testing.assert_allclose(MMA().step(x, 30.0, 10 * df0, g, dg), MMA().step(x, 3.0, df0, g, dg), atol=1e-10)
    with pytest.raises(ValueError):
        MMA(move=0.0)
    with pytest.raises(ValueError):
        MMA(normalise="max")
    with pytest.raises(ValueError):
        MMA().step(x, 1.0, df0, np.array([0.0, 0.0]), dg)


# ----------------------------------------------------------------------------- 2. oracle (2-D MBB)
def test_mbb_matches_old_topopt_mma():
    from topo_oracles.topology import mbb_half, topopt

    nelx, nely, vol, rmin = 30, 10, 0.5, 1.5
    old = topopt(mbb_half(nelx, nely), vol, rmin, ft=2, optimizer="MMA", move=0.2, tol=0.01, max_iter=200)
    _, st = _mbb(nelx, nely, rmin, lambda d: [T.Constraint(T.Volume(), vol)], MMA(move=0.2))
    new = st.run(max_iter=200, tol=0.01)
    c_old, c_new = np.asarray(old["compliance"]), np.asarray(new.history["f0"])
    print(f"\nMBB {nelx}x{nely}: iterations old {old['iteration']} new {new.iterations}; "
          f"c old {c_old[-1]:.10g} new {c_new[-1]:.10g}")
    assert new.iterations == old["iteration"]
    rel = np.abs(c_new - c_old) / c_old
    xo, xn = old["xfiz"], new.xphys.T[::-1, :]          # old Grid2D rows count from the top
    print(f"  max rel. compliance difference per iteration {rel.max():.2e}; mean|xphys_old - xphys_new| "
          f"{np.abs(xo - xn).mean():.2e}; volume old {xo.mean():.8f} new {new.xphys.mean():.8f}")
    assert rel.max() < 1e-6                              # dual-solver tolerance (measured 2.4e-9)
    assert np.abs(xo - xn).max() < 1e-5
    assert new.xphys.mean() == pytest.approx(xo.mean(), abs=1e-8)


# ----------------------------------------------------------------------------- 3. two constraints
def test_mbb_two_half_volume_constraints():
    nelx, nely, rmin = 48, 16, 2.0
    left = ((np.arange(nelx) + 0.5) < nelx / 2)[:, None] & np.ones((1, nely), bool)

    def split(bl, br):
        return lambda d: [T.Constraint(MaskVolume(left, "vol_left"), bl),
                          T.Constraint(MaskVolume(~left, "vol_right"), br)]

    _, st = _mbb(nelx, nely, rmin, lambda d: [T.Constraint(T.Volume(), 0.4)], MMA())
    rg = st.run(max_iter=300, tol=0.01)
    c_glob = st.evaluate(rg.x, gradients=False).f0
    vl, vr = rg.xphys[left].mean(), rg.xphys[~left].mean()
    out = {}
    for key, (bl, br) in {"own": (vl, vr), "tight_left": (0.3, 0.5)}.items():
        opt = MMA()
        _, s2 = _mbb(nelx, nely, rmin, split(bl, br), opt)
        r = s2.run(x0=np.full(s2.domain.n_design, 0.4), max_iter=300, tol=0.01)
        ev = s2.evaluate(r.x, gradients=False)
        out[key] = (ev.f0, ev.g, opt.last_lambda, r.iterations)
        print(f"\n{key}: bounds {bl:.4f}/{br:.4f}  c {ev.f0:.6g} (global {c_glob:.6g}, ratio {ev.f0 / c_glob:.4f})"
              f"  g {ev.g}  lambda {opt.last_lambda}  it {r.iterations}")
        assert r.converged
        assert np.all(ev.g <= 1e-3)                    # feasible (relative violation)
        assert np.all(ev.g >= -1e-3)                   # both active: more material always lowers compliance
        assert np.all(opt.last_lambda > 0)
    # bounds equal to the global optimum's own split: the global design is feasible, so the split problem can only
    # match it; MMA lands in a nearby local optimum (measured +1.1 % on 60x20)
    assert out["own"][0] == pytest.approx(c_glob, rel=0.03)
    # a restrictive split (same total 0.4) cannot beat the global bound
    assert out["tight_left"][0] > c_glob


# ----------------------------------------------------------------------------- 4. 3-D with β continuation
def test_cantilever_3d_mma_vs_oc_with_beta_continuation():
    nx, ny, nz, vol = 24, 8, 4, 0.3

    def run(opt):
        d = T.Domain((nx, ny, nz))
        clamp = d.nodes_where(lambda X: X[:, 0] == 0)
        tip = d.nodes_where(lambda X: (X[:, 0] == nx) & (X[:, 1] == 0))
        F = d.load_vector()[:, 0]
        F[d.dofs(tip, 1)] = -1.0 / len(tip)
        ch = T.standard_chain(d, 1.5, projection=True, beta=1.0)
        sch = T.BetaContinuation(ch, beta0=1.0, beta_max=4.0, every=20, tol=0.01)
        st = T.Study(d, T.Elasticity(d, d.dofs(clamp), element="H8"), F, ch, T.SIMP(d, penal=3.0, emin_ratio=1e-9),
                     T.Compliance(), [T.Constraint(T.Volume(), vol)], opt, sch)
        r = st.run(max_iter=80, tol=0.01)
        return r, st.evaluate(r.x, gradients=False), sch

    ro, eo, so = run(T.OC())
    mma = MMA()
    rm, em, sm = run(mma)
    print(f"\n3-D {nx}x{ny}x{nz}: OC c {eo.f0:.6g} (it {ro.iterations}, conv {ro.converged}, g {eo.g[0]:.1e}); "
          f"MMA c {em.f0:.6g} (it {rm.iterations}, conv {rm.converged}, g {em.g[0]:.1e}); "
          f"MMA/OC - 1 = {em.f0 / eo.f0 - 1:+.4f}; resets {mma.n_resets}; "
          f"mean|d xphys| {np.abs(ro.xphys - rm.xphys).mean():.3f}; grey OC {T.grey_fraction(ro.xphys):.3f} "
          f"MMA {T.grey_fraction(rm.xphys):.3f}")
    assert so.beta == sm.beta == 4.0
    assert mma.n_resets == 2                           # β 1 → 2 → 4
    assert rm.converged
    assert em.g[0] <= 1e-3 and eo.g[0] <= 1e-3
    assert em.f0 == pytest.approx(eo.f0, rel=0.03)   # measured +0.02 % (β_max 8: −2.2 %, OC oscillating)


def test_subproblem_failure_is_reported(monkeypatch):
    """B-04: the dual of the MMA subproblem is solved with L-BFGS-B, whose result was taken without a check. Now
    every step keeps the outcome and the dual's projected-gradient residual in last_subproblem; a solve that stopped
    early AND is far from the dual optimum warns (MMASubproblemWarning) and is counted."""
    import scipy.optimize as so

    from toporlogy.optim import _mma_core
    from toporlogy.optim.mma import MMASubproblemWarning

    x = np.full(4, 0.5)
    df0, g, dg = -np.ones(4), np.array([-0.1]), np.ones((1, 4)) / 4
    opt = MMA(move=0.2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", MMASubproblemWarning)            # a converged dual must not warn
        opt.step(x, 1.0, df0, g, dg)
    assert opt.last_subproblem["success"] is True and opt.subproblem_failures == 0

    real = so.minimize

    def stopped(*a, early: bool = False, **kw):
        res = real(*a, **kw)
        res.success, res.message = False, "ABNORMAL (injected)"
        if early:                                                         # stop at the start point: not converged
            kw["options"] = {**kw.get("options", {}), "maxiter": 0}
            res2 = real(*a, **kw)
            res.x, res.jac = res2.x, res2.jac
        return res

    # flag "not success" but the dual at its optimum (line search stalled at machine precision): no warning
    monkeypatch.setattr(_mma_core.so, "minimize", stopped)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MMASubproblemWarning)
        opt.step(x, 1.0, df0, g, dg)
    assert opt.subproblem_failures == 0 and opt.last_subproblem["residual"] < opt.dual_tol
    # flag "not success" and multipliers far from the optimum: warned and counted
    monkeypatch.setattr(_mma_core.so, "minimize", lambda *a, **kw: stopped(*a, early=True, **kw))
    with pytest.warns(MMASubproblemWarning, match="did not converge"):
        opt.step(x, 1.0, df0, g, dg)
    assert opt.subproblem_failures == 1 and opt.last_subproblem["message"] == "ABNORMAL (injected)"
    opt.restart()
    assert opt.subproblem_failures == 0                                  # counted per run

"""Optimizer-independent first-order optimality measure (docs/olcumler/dogrulama_turu.md §4, P1-P4,
pre-registered 2026-11-25 22:22 +0300 before the code was written).

P1-P3 on min Σ (x_i − a_i)² s.t. Σ x_i ≤ V, 0 ≤ x ≤ 1, whose solution is x = clip(a − λ/2, 0, 1) with λ from
Σ x = V (found here by bisection to machine precision). P4 on the 30 x 10 MBB of test_result_contract.py.
"""
from __future__ import annotations

import numpy as np
import pytest

import toporlogy as T
from test_result_contract import _mbb
from toporlogy.optimization.diagnostics import kkt_estimate
from toporlogy.optimizers import GCMMA, SiMPL

A = np.array([1.4, 0.9, 0.7, 0.55, 0.3, -0.2, 1.1, 0.05])
V = 3.0


def _solution():
    lo, hi = 0.0, 10.0
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        if np.clip(A - lam / 2, 0, 1).sum() > V:
            lo = lam
        else:
            hi = lam
    return np.clip(A - lam / 2, 0, 1), lam


def _problem(x, V=V):
    return 2 * (x - A), np.array([x.sum() / V - 1.0]), np.full((1, x.size), 1.0 / V)


def test_p1_exact_at_the_closed_form_solution():
    x, lam = _solution()
    assert (x == 0).any() and (x == 1).any() and ((x > 0) & (x < 1)).any()       # every bound case present
    df0, g, dg = _problem(x)
    k = kkt_estimate(x, df0, g, dg)
    lam_scaled = lam * V                                                          # multiplier of g = Σx/V − 1
    print(f"\n  P1: residual {k['residual']:.2e}, lambda {k['multipliers'][0]:.15g} vs {lam_scaled:.15g}")
    assert k["residual"] <= 1e-12
    assert abs(k["multipliers"][0] - lam_scaled) <= 1e-10 * lam_scaled


def test_p2_away_from_the_solution():
    x, _ = _solution()
    rng = np.random.default_rng(0)
    inner = (x > 0) & (x < 1)
    y = x.copy()
    y[inner] += 1e-3 * rng.choice([-1.0, 1.0], inner.sum())
    k = kkt_estimate(y, *_problem(y))
    print(f"\n  P2: residual {k['residual']:.2e}")
    assert k["residual"] >= 1e-4


def test_p3_inactive_constraint_gets_no_multiplier():
    x = np.clip(A, 0, 1) * 0.5 + 0.1                                              # Σx well below 10
    df0, g, dg = _problem(x, V=10.0)
    k = kkt_estimate(x, df0, g, dg)
    assert not k["active"][0] and k["multipliers"][0] == 0.0
    want = np.max(np.abs(df0)) / np.max(np.abs(df0))                              # interior: projected = full
    assert abs(k["residual"] - want) <= 1e-14


@pytest.mark.parametrize("make", [lambda: T.MMA(move=0.2), lambda: T.OC(move=0.2, dv_scale=300.0),
                                  lambda: GCMMA(), lambda: SiMPL()], ids=["MMA", "OC", "GCMMA", "SiMPL"])
def test_p4_falls_at_convergence_for_every_optimizer(make):
    first = _mbb(make()).run(max_iter=1, tol=1e-3)
    res = _mbb(make()).run(max_iter=400, tol=1e-3)
    print(f"\n  P4 {type(make()).__name__}: 1 iteration {first.stationarity:.3e}, {res.iterations} iterations "
          f"({res.termination}) {res.stationarity:.3e}, ratio {res.stationarity / first.stationarity:.2e}; "
          f"optimizer multipliers: {res.kkt_residual}")
    assert res.stationarity < 0.1 * first.stationarity

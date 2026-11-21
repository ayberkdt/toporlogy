"""Aggregates (optimization/aggregates.py): the stated bounds for random inputs, limits, weights, gradients."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from toporlogy import gradcheck
from toporlogy.optimization import aggregates as A

TOL = 1e-12


def _vals(seed: int, n: int, positive: bool) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    v = rng.random(n) * 3.0 if positive else rng.standard_normal(n) * 2.0
    return torch.as_tensor(v, dtype=torch.float64)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 200), st.floats(1.0, 200.0))
def test_bounds_of_every_aggregate(seed: int, n: int, par: float) -> None:
    g, gp = _vals(seed, n, False), _vals(seed, n, True)
    m, mp = float(g.max()), float(gp.max())
    slack = 1e-9 * (1 + abs(m))
    assert m - slack <= float(A.ks(g, par)) <= m + math.log(n) / par + slack
    assert m - math.log(n) / par - slack <= float(A.ks_lower(g, par)) <= m + slack
    assert float(g.mean()) - slack <= float(A.induced_exponential(g, par)) <= m + slack
    assert mp * (1 - 1e-9) <= float(A.pnorm(gp, par)) <= n ** (1 / par) * mp * (1 + 1e-9)
    assert n ** (-1 / par) * mp * (1 - 1e-9) <= float(A.pmean(gp, par)) <= mp * (1 + 1e-9)
    assert float(A.pmean(gp, par)) * (1 - 1e-9) <= float(A.induced_power(gp, par)) <= mp * (1 + 1e-9)


def test_exact_for_constant_field_and_limits() -> None:
    c = torch.full((37,), 1.7, dtype=torch.float64)
    for kind in ("ks_lower", "pmean", "induced_exponential", "induced_power"):
        assert abs(float(A.aggregate(c, kind)) - 1.7) < TOL, kind
    g = _vals(3, 50, True)
    m = float(g.max())
    for kind, par in (("ks", 1e4), ("ks_lower", 1e4), ("pnorm", 2e3), ("pmean", 2e3), ("induced_exponential", 1e4),
                      ("induced_power", 2e3)):
        assert abs(float(A.aggregate(g, kind, par)) - m) < 2e-3 * m, kind
    assert float(A.aggregate(g, "max")) == m
    assert A.is_upper_bound("ks") and A.is_upper_bound("pnorm") and not A.is_upper_bound("pmean")


def test_no_overflow_at_large_values_and_parameters() -> None:
    g = torch.as_tensor([1e6, 2e6, 3e6], dtype=torch.float64)
    for kind in A.AGGREGATES:
        v = float(A.aggregate(g, kind, 500.0))
        assert math.isfinite(v) and 2.9e6 <= v <= 3.1e6, (kind, v)


def test_weights_reproduce_repetition() -> None:
    """Integer weights equal repeating the entries: the aggregate approximates an integral, not a count."""
    g = _vals(5, 6, True)
    w = torch.as_tensor([1, 2, 3, 1, 4, 2], dtype=torch.float64)
    rep = torch.repeat_interleave(g, w.long())
    for kind in A.AGGREGATES:
        a, b = float(A.aggregate(g, kind, 7.0, weights=w)), float(A.aggregate(rep, kind, 7.0))
        assert abs(a - b) <= 1e-12 * abs(b), kind


@pytest.mark.parametrize("kind", sorted(A.AGGREGATES))
def test_gradients(kind: str) -> None:
    x0 = 0.2 + np.random.default_rng(1).random(40)
    r = gradcheck(lambda x: A.aggregate(x * x + 0.1, kind, 9.0), x0, n_dirs=5)
    assert r["max_rel_error"] <= 1e-6, r


def test_errors() -> None:
    g = torch.as_tensor([-1.0, 2.0], dtype=torch.float64)
    for bad in (lambda: A.pnorm(g), lambda: A.induced_power(g), lambda: A.aggregate(g, "nope"),
                lambda: A.ks(g[:0]), lambda: A.ks(g, weights=torch.ones(3)), lambda: A.ks(g, -1.0)):
        with pytest.raises(ValueError):
            bad()


def test_adaptive_scale() -> None:
    s = A.AdaptiveScale(0.5)
    assert s.update(2.0, 4.0) == 0.5                        # first update: the plain ratio
    assert s.update(3.0, 3.0) == 0.75                       # 0.5 * 1 + 0.5 * 0.5
    s.frozen = True
    assert s.update(9.0, 1.0) == 0.75
    s.frozen = False
    s.reset()
    assert s.value is None and s.update(0.0, 0.0) == 1.0    # nothing to scale: neutral

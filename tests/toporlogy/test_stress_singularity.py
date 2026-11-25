"""Peak-stress diagnosis (docs/olcumler/gerilme_onkayit.md §4, pre-registered 2026-11-26 00:24 +0300 before the first
run): point load, re-entrant corner and clamped edge are reported as singular (S1-S5), a load spread over a finite
footprint and a smooth bending field are not (S6, S7). The rule itself is checked on closed-form sequences."""
from __future__ import annotations

from functools import cache

import numpy as np
import pytest

import _stress_cases as S
from toporlogy.optimization.stress_singularity import (CONVERGED, NOT_CONVERGED, NOT_VERIFIED, SINGULAR,
                                                       assess_peak, region_stress)

H = [1.0 / n for n in S.LEVELS]


@cache
def _regions(name: str):
    build, lo, hi, _ = S.PEAK_PROBLEMS[name]
    return [S.centroid_region(build(n), lo, hi) for n in S.LEVELS]


def _report(name: str):
    rows = _regions(name)
    a = assess_peak(H, [r.peak for r in rows])
    print(f"\n  {name}: peaks " + " ".join(f"{r.peak:.4g}" for r in rows) + " | slopes "
          + " ".join(f"{s:.3f}" for s in a.slopes) + f" | ratio {a.increment_ratio} | {a.status}"
          + " | means " + " ".join(f"{r.mean:.5g}" for r in rows))
    return rows, a


SINGULAR_CASES = ["point_load", "l_corner", "clamped_edge"]


@pytest.mark.parametrize("name", SINGULAR_CASES)
def test_s1_singular_status(name):
    _, a = _report(name)
    assert a.status == SINGULAR == "singular — max stress not converged"


@pytest.mark.parametrize("name", SINGULAR_CASES)
def test_s2_peak_location_at_the_singular_point(name):
    finest = _regions(name)[-1]
    dist = float(np.linalg.norm(np.subtract(finest.location, S.PEAK_PROBLEMS[name][3])))
    print(f"\n  S2 {name}: peak at {finest.location}, {dist / H[-1]:.2f} h from the singular point")
    assert dist <= 2 * H[-1]


@pytest.mark.parametrize("name,band", [("point_load", (0.8, 1.2)), ("l_corner", (0.30, 0.55))],
                         ids=["S3", "S4"])
def test_s3_s4_growth_exponent(name, band):
    _, a = _report(name)
    assert band[0] <= a.slopes[-1] <= band[1]


def test_s4_reference_exponent_of_the_270_degree_corner():
    """Mode I eigenvalue of the traction-free wedge of opening 2α = 270°: λ sin 2α + sin 2λα = 0 (Williams 1952,
    written from the textbook form; the paper was not read). The pre-registered band is around 1 − λ."""
    alpha = 3 * np.pi / 4
    lo, hi = 0.5, 0.6
    f = lambda lam: lam * np.sin(2 * alpha) + np.sin(2 * lam * alpha)      # noqa: E731
    assert f(lo) * f(hi) < 0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    assert abs(lo - 0.544484) < 1e-6 and abs((1 - lo) - 0.4555) < 1e-4


@pytest.mark.parametrize("name", SINGULAR_CASES)
def test_s5_regional_mean_settles(name):
    means = [r.mean for r in _regions(name)]
    change = abs(means[-1] - means[-2]) / means[-1]
    print(f"\n  S5 {name}: regional means " + " ".join(f"{m:.5g}" for m in means) + f", last change {change:.2%}")
    assert change <= 0.03


def test_s6_finite_footprint_is_not_singular():
    _, a = _report("footprint_load")
    assert a.status == CONVERGED


def test_s7_growing_but_settling_peak_is_not_called_singular():
    n = [8, 16, 32, 64]
    peaks = [S.bending_peak(k).peak for k in n]
    h = [S.H / k for k in n]
    fine, coarse = assess_peak(h, peaks), assess_peak(h[:3], peaks[:3])
    print("\n  S7: peaks " + " ".join(f"{p:.5f}" for p in peaks) + f" | {coarse.status} -> {fine.status}")
    assert all(np.diff(peaks) > 0)
    assert fine.status == CONVERGED
    assert coarse.status == NOT_CONVERGED


# ----------------------------------------------------------------------------------------------- the rule
def test_rule_on_closed_form_sequences():
    h = np.array([0.1, 0.05, 0.025, 0.0125])
    power = assess_peak(h, 2.0 + h ** -0.45)
    assert power.status == SINGULAR
    assert abs(assess_peak(h, h ** -0.45).slopes[-1] - 0.45) < 1e-12
    assert assess_peak(h, 3.0 - np.log(h)).status == SINGULAR                       # logarithmic divergence
    assert assess_peak(h, 5.0 - h ** 2).status == CONVERGED
    assert assess_peak(h, 5.0 - 20 * h).status == NOT_CONVERGED                     # first order, still far
    assert assess_peak(h, np.array([1.0, 1.5, 1.2, 1.6])).status == NOT_CONVERGED   # oscillating
    assert assess_peak(h, np.full(4, 7.0)).status == CONVERGED
    assert assess_peak(h[:2], np.array([1.0, 2.0])).status == NOT_VERIFIED
    assert assess_peak(h, np.array([1.0, 2.0, np.nan, 3.0])).status == NOT_VERIFIED
    with pytest.raises(ValueError, match="decreasing"):
        assess_peak(h[::-1], h)


def test_region_stress():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    r = region_stress(np.array([1.0, 5.0, 9.0]), X, lo=[0.0, -1.0], hi=[1.0, 1.0], weights=np.array([3.0, 1.0, 1.0]))
    assert (r.peak, r.location, r.n_elements) == (5.0, (1.0, 0.0), 2)
    assert r.mean == pytest.approx(2.0)
    with pytest.raises(ValueError, match="region"):
        region_stress(np.ones(3), X, lo=[5.0, 5.0])

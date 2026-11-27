"""R07_bordered: `arc_length` at an exactly singular tangent (bordered / previous-increment predictor).

Preregistered in ``docs/foundations/prereg/R07_bordered.md`` (checks R1-R6; R7 is the promotion of the strict xfails in
``test_bifurcation.py``). References are closed forms derived there: the primary path of model A (v = -lam / k_a, u = 0)
and the two-bar truss P(w) of ``_truss.py``. R5 compares the new code with itself switched off (a regression of the
neutrality, labelled so); the independent evidence of neutrality is the sha256 fingerprints of the unchanged code in the
prereg. "Fix disabled" = ``continuation._singular_predictor`` replaced by a function that finds no predictor, which is the
code path of the old behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from toporlogy.solvers import continuation
from toporlogy.solvers.continuation import ArcLengthSettings, ContinuationError, TangentSpectrum, arc_length
from toporlogy.solvers.newton import LinearSolveFailure, Linearization

sys.path.insert(0, str(Path(__file__).parent))
from _truss import TwoBarTruss  # noqa: E402

DENSE = TangentSpectrum()


class ModelA:
    """Model A of ``catallanma_onkayit.md`` (re-derived): U = ka e^2 / 2 + kr u^2 / 2 + beta u^3 / 6, e = v + u^2 / 2L,
    R = grad U - lam q, q = (0, -1); primary path u = 0, v = -lam / ka."""

    conservative = True
    load = np.array([0.0, -1.0])

    def __init__(self, L: float = 1.0, ka: float = 100.0, kr: float = 1.0, beta: float = 3.0) -> None:
        self.L, self.ka, self.kr, self.beta = L, ka, kr, beta

    def linearize(self, x: np.ndarray, lam: float) -> Linearization:
        u, v = x
        L, ka, kr, b = self.L, self.ka, self.kr, self.beta
        e = v + u * u / (2 * L)
        g = np.array([ka * e * u / L + kr * u + 0.5 * b * u * u, ka * e])
        K = np.array([[ka * (u / L) ** 2 + ka * e / L + kr + b * u, ka * u / L], [ka * u / L, ka]])
        return Linearization(g - lam * self.load, K, float(np.abs(K).sum(axis=0).max()))

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray:
        try:
            out = np.linalg.solve(lin.matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise LinearSolveFailure(f"LinAlgError: {e}") from e
        if not np.all(np.isfinite(out)):
            raise LinearSolveFailure("non-finite solution")
        return out

    def det_sign(self, lin: Linearization) -> int:
        return int(np.sign(np.linalg.det(lin.matrix)))


class ExactZeroTruss(TwoBarTruss):
    """The truss with the rounding of exact zeros removed: tangent entries of size <= 1e-12 ||K||_1 are set to 0.0
    (K_vv at the limit point w_1 is exactly zero in exact arithmetic and ~1e-18 in floating point). No other entry
    changes (K_uu ~ 2, K_vv elsewhere >> 1e-12 ||K||_1 except within ~1e-10 h of the two limit points)."""

    def linearize(self, x: np.ndarray, lam: float) -> Linearization:
        lin = super().linearize(x, lam)
        K = lin.matrix.copy()
        K[np.abs(K) <= 1e-12 * lin.norm1] = 0.0
        return Linearization(lin.residual, K, lin.norm1, lin.energy)


def _off(monkeypatch) -> None:
    """Disable the fix: no predictor at a singular tangent (the old behaviour)."""
    monkeypatch.setattr(continuation, "_singular_predictor", lambda *a, **k: None)


def _a_settings(ds: float) -> ArcLengthSettings:
    return ArcLengthSettings(ds=ds, rtol=1e-12, eta_tol=0.0, max_steps=500)


def _signature(path) -> tuple:
    """Everything an accepted point and a record carries, as bytes / reprs (for the bitwise comparison)."""
    pts = [(np.float64(p.lam).tobytes(), np.asarray(p.x, float).tobytes(), np.float64(p.ds).tobytes(), p.iterations,
            np.float64(p.residual).tobytes(), p.det_sign, p.index,
            None if p.eigenvalues is None else np.asarray(p.eigenvalues).tobytes()) for p in path.points]
    recs = [repr((r.load_factor, r.converged, r.code, r.rolled_back, r.notes, r.residuals, r.backward_errors,
                  r.updates, r.reason)) for r in path.records]
    return pts, recs, path.rejected


# ====================================================================== R1: model A, exact singular accepted point
def test_r1_model_a_passes_the_exactly_singular_accepted_point():
    """R1: ds = 0.01 accepts (0, -0.01) at lam = 1 with K11 = 0.0 exactly; the path continues on v = -lam / k_a."""
    m = ModelA()
    path = arc_length(m, np.zeros(2), 0.0, _a_settings(0.01), lambda x, lam: lam >= 1.5)
    K1 = m.linearize(path.points[1].x, path.points[1].lam).matrix
    lam, u, v = path.lam, path.component(0), path.component(1)
    print(f"R1: {len(path.points)} points, K11 at point 1 = {K1[0, 0]!r}, lam {lam}, v {v}, notes {path.records[1].notes}")
    assert K1[0, 0] == 0.0                                           # the precondition of the check
    assert len(path.points) == 3
    assert np.abs(lam - np.arange(3.0)).max() <= 1e-12
    assert np.abs(v + lam / m.ka).max() <= 1e-12 and np.abs(u).max() <= 1e-14
    assert np.all(np.diff(lam) > 0) and np.all(np.diff(v) < 0)       # orientation: the previous increment
    assert any("predictor: previous increment" in n for n in path.records[1].notes)
    assert path.records[1].code == "converged" and path.records[1].reason == "" and path.rejected == 0


# ====================================================================== R2 / R3: truss from the exact limit point
def _truss_path(tr: TwoBarTruss, start_tangent, direction: float = 1.0, ds: float = 0.01, w_stop: float = 2.5):
    x0 = np.array([0.0, -tr.w1])
    lam0 = float(tr.P(tr.w1) / tr.P_ref)
    st = ArcLengthSettings(ds=ds, rtol=1e-12, eta_tol=0.0, max_steps=4000, direction=direction)
    if start_tangent is not None and start_tangent[0][1] > 0:           # w decreasing: stop before the origin side ends
        stop = lambda x, lam: -x[1] <= -0.5 * tr.h  # noqa: E731
    else:
        stop = lambda x, lam: -x[1] >= w_stop * tr.h  # noqa: E731
    if start_tangent is None and direction >= 0:
        stop = lambda x, lam: -x[1] <= -0.5 * tr.h  # noqa: E731
    return arc_length(tr, x0, lam0, st, stop, spectrum=DENSE, start_tangent=start_tangent)


def _curve_error(tr: TwoBarTruss, path) -> tuple[float, float]:
    w = -path.component(1)
    return float(np.abs(path.lam - tr.P(w) / tr.P_ref).max()), float(np.abs(path.component(0)).max())


def test_r2_truss_exactly_singular_start_downward_turns_at_both_limit_points():
    """R2 (w increasing): the exact limit point w1 is the start; the load factor leaves it downward, turns at the
    second limit point (lam_min ~ -1) and rises again; every point on the closed form."""
    tr = ExactZeroTruss()
    lin = tr.linearize(np.array([0.0, -tr.w1]), 1.0)
    assert lin.matrix[1, 1] == 0.0 and lin.matrix[0, 0] > 1.0           # exactly singular start, K_uu = O(2)
    path = _truss_path(tr, ((0.0, -1.0), 0.0))
    e_curve, e_u = _curve_error(tr, path)
    lam = path.lam
    imin = int(np.argmin(lam))
    dv1 = path.points[1].x[1] - path.points[0].x[1]
    print(f"R2: {len(lam)} points, curve {e_curve:.1e}, |u| {e_u:.1e}, lam[1] {lam[1]:.9f}, lam_min {lam[imin]:.6f} at "
          f"point {imin}, dv1 {dv1:.12f}, notes {path.records[0].notes}")
    assert e_curve <= 1e-9 and e_u <= 1e-12
    assert dv1 < 0 and abs(abs(dv1) - 0.01) <= 1e-12                   # along the start tangent, |dx| = ds
    assert abs(lam[0] - 1.0) <= 1e-12 and lam[1] < 1.0 and np.all(np.diff(lam[: imin + 1]) < 0)
    assert 0 < imin < len(lam) - 1 and np.all(np.diff(lam[imin:]) > 0)
    assert -1.0 - 1e-9 <= lam[imin] <= -0.97
    assert any("bordered null vector" in n for n in path.records[0].notes)


def test_r2_truss_exactly_singular_start_upward_orientation_follows_the_start_tangent():
    """R2 (opposite orientation): start tangent (0, +1; 0) sends the path toward the origin (w decreasing)."""
    tr = ExactZeroTruss()
    path = _truss_path(tr, ((0.0, 1.0), 0.0))
    e_curve, e_u = _curve_error(tr, path)
    dv1 = path.points[1].x[1] - path.points[0].x[1]
    print(f"R2 up: {len(path.lam)} points, curve {e_curve:.1e}, |u| {e_u:.1e}, lam[1] {path.lam[1]:.9f}, dv1 {dv1:.12f}")
    assert e_curve <= 1e-9 and e_u <= 1e-12
    assert dv1 > 0 and abs(dv1 - 0.01) <= 1e-12 and path.lam[1] < 1.0 and np.all(np.diff(path.lam) < 0)


@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["direction+", "direction-"])
def test_r3_truss_exactly_singular_start_without_start_tangent(direction):
    """R3: no start tangent: the null vector (0, 1, 0) of [K, -q] with t_lam = 0 is oriented by the largest-component
    rule times `direction`: + -> v increases (w decreases), - -> v decreases."""
    tr = ExactZeroTruss()
    path = _truss_path(tr, None, direction=direction)
    e_curve, e_u = _curve_error(tr, path)
    dv1 = path.points[1].x[1] - path.points[0].x[1]
    print(f"R3 direction {direction:+.0f}: {len(path.lam)} points, curve {e_curve:.1e}, |u| {e_u:.1e}, lam[1] "
          f"{path.lam[1]:.9f}, dv1 {dv1:+.12f}, notes {path.records[0].notes}")
    assert e_curve <= 1e-9 and e_u <= 1e-12
    assert np.sign(dv1) == direction and abs(abs(dv1) - 0.01) <= 1e-12 and path.lam[1] < 1.0
    assert any("null vector of [K, -q]" in n for n in path.records[0].notes)


# ====================================================================== R4: singular bifurcation start
def test_r4_bifurcation_start_has_no_unique_direction_without_a_start_tangent():
    """R4a: model A with k_r = 0 at the origin (q in range K, null [K, -q] two-dimensional): nothing is invented."""
    with pytest.raises(ContinuationError) as ei:
        arc_length(ModelA(kr=0.0), np.zeros(2), 0.0, _a_settings(0.01), lambda x, lam: lam >= 1.5)
    assert ei.value.code == "step_limit" and ei.value.cause == "singular_tangent"
    assert len(ei.value.path.points) == 1


def test_r4_bifurcation_start_with_a_start_tangent_follows_the_primary_path():
    """R4b: with start_tangent = ((0, -0.01), 1) the path runs through the singular start on v = -lam / k_a."""
    m = ModelA(kr=0.0)
    path = arc_length(m, np.zeros(2), 0.0, _a_settings(0.01), lambda x, lam: lam >= 1.5,
                      start_tangent=(np.array([0.0, -0.01]), 1.0))
    lam, u, v = path.lam, path.component(0), path.component(1)
    print(f"R4b: {len(lam)} points, lam {lam}, notes {path.records[0].notes}")
    assert lam[-1] >= 1.5 and np.abs(lam - np.arange(len(lam))).max() <= 1e-12
    assert np.abs(v + lam / m.ka).max() <= 1e-12 and np.abs(u).max() <= 1e-14


# ====================================================================== R5: neutrality where K is regular
def _runs():
    tr = TwoBarTruss()
    yield "truss ds=0.01", lambda: arc_length(tr, np.zeros(2), 0.0,
                                              ArcLengthSettings(ds=0.01, rtol=1e-12, eta_tol=0.0, max_steps=4000),
                                              lambda x, lam: -x[1] >= 2.5 * tr.h, spectrum=DENSE)
    yield "model A ds=0.0137", lambda: arc_length(ModelA(), np.zeros(2), 0.0, _a_settings(0.0137),
                                                  lambda x, lam: lam >= 1.5, spectrum=DENSE)


@pytest.mark.parametrize("name,run", list(_runs()), ids=[n for n, _ in _runs()])
def test_r5_regular_tangent_paths_are_bitwise_unchanged(name, run, monkeypatch):
    """R5 (regression of the new code against itself switched off): no accepted point has a singular tangent, so the
    switched-off run is the old code and the paths are equal bitwise. The independent evidence is the sha256
    fingerprint of the unchanged code in the prereg."""
    new = run()
    _off(monkeypatch)
    old = run()
    assert len(new.points) > 2
    assert _signature(new) == _signature(old)
    assert not any("predictor:" in n for r in new.records for n in r.notes)


# ====================================================================== R6: negative control
def test_r6_fix_disabled_reproduces_the_n4b_failure(monkeypatch):
    """R6a: model A ds = 0.01 without the fix: ContinuationError step_limit / singular_tangent, 2 accepted points, last
    lam = 1 (the N4b record)."""
    _off(monkeypatch)
    with pytest.raises(ContinuationError) as ei:
        arc_length(ModelA(), np.zeros(2), 0.0, _a_settings(0.01), lambda x, lam: lam >= 1.5)
    pts = ei.value.path.points
    assert ei.value.code == "step_limit" and ei.value.cause == "singular_tangent"
    assert len(pts) == 2 and abs(pts[-1].lam - 1.0) <= 1e-12


@pytest.mark.parametrize("start_tangent", [((0.0, -1.0), 0.0), None], ids=["with-start-tangent", "without"])
def test_r6_fix_disabled_truss_exact_limit_start_fails(start_tangent, monkeypatch):
    """R6b: the exactly singular truss start without the fix: the first predictor solve fails, step_limit."""
    _off(monkeypatch)
    with pytest.raises(ContinuationError) as ei:
        _truss_path(ExactZeroTruss(), start_tangent)
    assert ei.value.code == "step_limit" and ei.value.cause == "singular_tangent"
    assert len(ei.value.path.points) == 1

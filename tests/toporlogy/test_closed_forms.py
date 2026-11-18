"""Forward values against their closed forms (adversarial audit, 18 Nov 2026: these were gradient-checked only).

A gradient check compares autograd with finite differences of the SAME forward function, so a wrong forward formula
passes it. Each expected value here is written out from the mathematical definition in the docstring of the class
under test and evaluated independently with numpy, at points on both sides of every branch.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.design.interp import RAMP, SIMP
from toporlogy.fem.heat import ConductivityInterp
from toporlogy.optimization import dynamics as dyn
from toporlogy.optimization.checks import gradcheck, grey_fraction
from toporlogy.optimization.robust import Robust

X = np.array([0.0, 0.02, 0.05, 0.0999, 0.1, 0.1001, 0.3, 0.5, 0.77, 1.0])


def _domain():
    frac = np.ones((5, 2))
    frac[1, 0], frac[3, 1] = 0.4, 0.75                          # partial voxels: the frac factor must be applied
    return T.Domain((5, 2), 1.0, frac=frac)


def _grid(d, values):
    return torch.as_tensor(np.asarray(values, float).reshape(d.shape))


def test_simp_and_ramp_forward():
    d = _domain()
    x = X.reshape(d.shape)
    E0, emin = 2.5, 1e-3
    for p in (1.0, 3.0, 4.5):
        got = SIMP(d, E0=E0, penal=p, emin_ratio=emin)(_grid(d, x)).numpy()
        want = (emin * E0 + x ** p * (E0 - emin * E0)) * d.frac
        assert np.allclose(got, want, rtol=1e-14, atol=0)
    for q in (0.0, 8.0):
        got = RAMP(d, E0=E0, q=q, emin_ratio=emin)(_grid(d, x)).numpy()
        want = (emin * E0 + x / (1 + q * (1 - x)) * (E0 - emin * E0)) * d.frac
        assert np.allclose(got, want, rtol=1e-14, atol=0)
    # RAMP is not SIMP: at x = 0.5, q = 8 it is 0.1 of the stiffness, SIMP p=3 gives 0.125
    r = RAMP(d, q=8.0, emin_ratio=0.0)(_grid(d, np.full(d.shape, 0.5))).numpy() / d.frac
    assert np.allclose(r, 0.5 / 5.0, rtol=1e-14)


def test_mass_and_two_range_interpolations_on_both_branches():
    d = _domain()
    x = X.reshape(d.shape)
    t = 0.1
    got = dyn.LinearMass(d, rho0=7.8, rhomin_ratio=0.01)(_grid(d, x)).numpy()
    assert np.allclose(got, 7.8 * (0.01 + 0.99 * x) * d.frac, rtol=1e-14, atol=0)
    q = 6.0
    got = dyn.TwoRangeMass(d, rho0=2.0, threshold=t, q=q)(_grid(d, x)).numpy()
    want = 2.0 * np.where(x >= t, x, x ** q / t ** (q - 1)) * d.frac
    assert np.allclose(got, want, rtol=1e-13, atol=1e-300)
    for p, qq in ((3.0, 1.0), (3.0, 2.0)):
        got = dyn.TwoRangeSIMP(d, E0=3.0, penal=p, threshold=t, q=qq, emin_ratio=1e-9)(_grid(d, x)).numpy()
        s = np.where(x >= t, x ** p, t ** (p - qq) * x ** qq)
        assert np.allclose(got, (3e-9 + s * (3.0 - 3e-9)) * d.frac, rtol=1e-13, atol=0)
    # continuity at the threshold from both sides (both branches are C0 there by construction)
    m = dyn.TwoRangeSIMP(d, penal=3.0, threshold=t, q=1.0, emin_ratio=0.0)
    lo, hi = m(_grid(d, np.full(d.shape, t - 1e-12))).numpy() / d.frac, m(_grid(d, np.full(d.shape, t))).numpy() / d.frac
    assert np.allclose(lo, hi, rtol=1e-9)


def test_conductivity_interpolation():
    d = _domain()
    x = X.reshape(d.shape)
    got = ConductivityInterp(d, k0=4.0, penal=3.0, kmin_ratio=1e-3)(_grid(d, x)).numpy()
    assert np.allclose(got, (1e-3 + x ** 3 * (1 - 1e-3)) * 4.0 * d.frac, rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match="kmin_ratio"):
        ConductivityInterp(d, kmin_ratio=1.0)


def test_ks_and_pnorm_lower_bounds_of_the_smallest_eigenvalue():
    lam = torch.tensor([2.0, 2.0, 3.0, 7.5], dtype=torch.float64)
    for rho in (10.0, 40.0):
        ks = dyn.MinEigenvalueKS(4, rho=rho, scale=1.5, kind="ks").aggregate(lam)
        want = -(1.5 / rho) * math.log(sum(math.exp(-rho * v / 1.5) for v in lam.tolist()))
        assert float(ks) == pytest.approx(want, rel=1e-14)
        assert float(ks) <= 2.0                                   # a lower bound of min(lam)
        pn = dyn.MinEigenvalueKS(4, rho=rho, scale=1.5, kind="pnorm").aggregate(lam)
        want = 1.5 * sum((v / 1.5) ** (-rho) for v in lam.tolist()) ** (-1.0 / rho)
        assert float(pn) == pytest.approx(want, rel=1e-13)
    auto = dyn.MinEigenvalueKS(4, rho=40.0, kind="ks")             # scale taken from the first value seen
    auto.aggregate(lam)
    assert auto.scale == 2.0


def test_frequency_margin_value(monkeypatch):
    """1 - lambda_low / lambda_min with lambda_min = (2 pi f_min)^2 / c, c = E / (rho L^2): the (2 pi)^2 and the
    unit factor must be there."""
    lam = torch.tensor([39.0, 80.0, 81.0], dtype=torch.float64)
    monkeypatch.setattr(dyn, "eigenvalues", lambda ctx, n: lam[:n])
    c = dyn.omega_squared_factor(E_unit=70e3, rho_unit=2.7e-9, length_unit=1.0)
    assert c == pytest.approx(70e3 / 2.7e-9, rel=1e-15)
    f_min = 0.9 * math.sqrt(39.0 * c) / (2 * math.pi)                # lambda_1 sits 1/0.81 above lambda_min
    m = dyn.FrequencyMargin(f_min, 1, 40.0, E_unit=70e3, rho_unit=2.7e-9)
    lam_min = (2 * math.pi * f_min) ** 2 / c
    assert m.lam_min == pytest.approx(lam_min, rel=1e-13)
    assert float(m(None)) == pytest.approx(1 - 39.0 / lam_min, rel=1e-12)
    assert float(m(None)) == pytest.approx(1 - 1 / 0.81, rel=1e-12)      # negative: the constraint holds
    m3 = dyn.FrequencyMargin(f_min, 3, 40.0, E_unit=70e3, rho_unit=2.7e-9)
    low = -(lam_min / 40.0) * math.log(sum(math.exp(-40.0 * v / lam_min) for v in lam.tolist()))
    assert float(m3(None)) == pytest.approx(1 - low / lam_min, rel=1e-12)


@pytest.mark.parametrize("kind", ["ks", "pnorm", "max"])
def test_robust_aggregate_of_the_realisations(kind, monkeypatch):
    """Robust = worst case of the realisations: KS (scale = max |v| at the first evaluation), p-norm, or max."""
    v = torch.tensor([3.0, 5.0, 4.0], dtype=torch.float64)
    r = Robust(T.Compliance(), robust=None, aggregate=kind, rho=50.0, p=16.0)
    monkeypatch.setattr(r, "values", lambda ctx: v)
    got = float(r(None))
    if kind == "max":
        want = 5.0
    elif kind == "pnorm":
        want = sum(x ** 16.0 for x in v.tolist()) ** (1 / 16.0)
    else:
        want = 5.0 / 50.0 * math.log(sum(math.exp(50.0 * x / 5.0) for x in v.tolist()))
    assert got == pytest.approx(want, rel=1e-14)
    assert got >= 5.0 - 1e-12                                            # an upper bound of the worst case
    assert r.last_values == {"eroded": 3.0, "intermediate": 5.0, "dilated": 4.0}


def test_grey_fraction_counts_the_open_interval():
    x = np.array([0.0, 0.1, 0.1000001, 0.5, 0.8999999, 0.9, 1.0])
    assert grey_fraction(x) == pytest.approx(3 / 7, rel=1e-15)          # 0.1 and 0.9 themselves are not grey
    mask = np.array([1, 1, 1, 1, 0, 0, 0], bool)
    assert grey_fraction(x, mask) == pytest.approx(2 / 4, rel=1e-15)
    assert grey_fraction(x, mask, lo=0.0, hi=1.0) == pytest.approx(3 / 4, rel=1e-15)
    assert grey_fraction(np.zeros(0)) == 0.0


def test_gradcheck_reports_a_wrong_gradient():
    """About 60 gradient tests rely on gradcheck; it must flag a backward that is off, and pass a correct one."""
    class Wrong(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x ** 3).sum()

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * 3.0 * x ** 2 * 1.01                              # 1 % too large

    x = np.random.default_rng(4).uniform(0.2, 1.0, 12)
    bad = gradcheck(Wrong.apply, x)
    good = gradcheck(lambda t: (t ** 3).sum(), x)
    # measured: wrong 9.9e-3, correct 2.4e-8 (round-off of the h = 1e-6 central difference); the suite's gradient
    # tolerance is 1e-6, so a 1 % error is flagged with a margin of 10^4
    assert bad["max_rel_error"] > 5e-3 and good["max_rel_error"] < 1e-6
    sign = gradcheck(lambda t: -(t ** 3).sum() + 0 * t.sum(), x)
    assert sign["max_rel_error"] < 1e-6

"""Real density in the energy interpolation of the non-linear physics (docs/olcumler/gercek_yogunluk_onkayit.md,
E4a-E4e; E4f is the baseline run)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.domain import Domain
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.interp import RAMP, SIMP, inverse_modulus
from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearElasticity
from toporlogy.optimization.nonlinear import EndCompliance
from toporlogy.optimization.study import Study
from toporlogy.optimizers.mma import MMA


@pytest.mark.parametrize("kind,par", [("simp", 1.0), ("simp", 2.0), ("simp", 3.0), ("ramp", 8.0)])
@pytest.mark.parametrize("frac", [1.0, 0.5])
def test_e4a_inverse_modulus(kind, par, frac) -> None:
    x = np.concatenate([np.geomspace(1e-3, 1.0, 200), [1.0]])
    d = Domain((len(x), 1), frac=np.full((len(x), 1), frac))
    interp = SIMP(d, penal=par, emin_ratio=1e-9) if kind == "simp" else RAMP(d, q=par, emin_ratio=1e-9)
    E = interp(torch.as_tensor(x[:, None])).numpy().ravel()
    xr, dx = inverse_modulus(interp, E, np.full_like(x, frac))
    assert np.abs(xr - x).max() <= 1e-12
    h = 1e-7
    xc = x[:, None]
    fd = (interp(torch.as_tensor(xc + h)) - interp(torch.as_tensor(xc - h))).numpy().ravel() / (2 * h)
    assert np.allclose(dx[5:-5] * fd[5:-5], 1.0, rtol=1e-5)


def _beam(penal: float):
    d = Domain((16, 6))
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero((np.abs(X[:, 0] - X[:, 0].max()) < 1e-9) & (X[:, 1] < 1e-9))
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -0.02 / len(tip)
    nl = NonlinearElasticity(d, fixed, nu=0.3, rtol=1e-12, energy_interpolation=EnergyInterpolation(E0=1.0))
    simp = SIMP(d, E0=1.0, penal=penal, emin_ratio=1e-6)
    st = Study(d, nl, F, Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)]), simp, EndCompliance(), [], MMA())
    return d, nl, st


@pytest.mark.parametrize("penal", [1.0, 2.0, 3.0])
def test_e4b_gamma_at_the_real_density(penal) -> None:
    d, nl, st = _beam(penal)
    x = np.random.default_rng(1).random(d.n_design)
    ev = st.evaluate(x, False)
    xbar = np.asarray(ev.xphys).ravel()[d.active_elements]
    E = st.interp(torch.as_tensor(ev.xphys)).numpy().ravel()[d.active_elements]
    it = nl.interpolation
    g = nl._gamma(E, it.cutoff)[0]
    g_true = it.gamma(E, it.cutoff, (xbar, np.ones_like(xbar)))[0]
    assert np.abs(g - g_true).max() <= 1e-12


def test_e4c_gradient_with_penal_2() -> None:
    d, nl, st = _beam(2.0)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(d.n_design)
    idx = np.random.default_rng(5).choice(d.n_design, 6, replace=False)
    ev = st.evaluate(x)
    fd = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-5
        xm[i] -= 1e-5
        fd.append((st.evaluate(xp, False).f0 - st.evaluate(xm, False).f0) / 2e-5)
    fd = np.asarray(fd)
    assert np.abs(ev.df0[idx] - fd).max() / np.abs(fd).max() <= 1e-5


def test_e4d_passive_void_gradient_is_finite() -> None:
    d0 = Domain((16, 6))
    void = np.zeros(d0.shape, bool)
    void[6:9, 2:4] = True
    d = Domain((16, 6), void=void)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero((np.abs(X[:, 0] - X[:, 0].max()) < 1e-9) & (X[:, 1] < 1e-9))
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -0.02 / len(tip)
    nl = NonlinearElasticity(d, fixed, nu=0.3, rtol=1e-12, energy_interpolation=EnergyInterpolation(E0=1.0))
    st = Study(d, nl, F, Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)]), SIMP(d, penal=2.0), EndCompliance(),
               [], MMA())
    ev = st.evaluate(np.full(d.n_design, 0.5))
    assert np.all(np.isfinite(ev.df0)) and np.isfinite(ev.f0)


def test_e4e_physics_alone_keeps_its_rule() -> None:
    d, nl, _ = _beam(3.0)
    alone = NonlinearElasticity(d, nl.fixed if hasattr(nl, "fixed") else d.dofs(np.flatnonzero(
        d.node_coords()[:, 0] < 1e-9)), nu=0.3, rtol=1e-12, energy_interpolation=EnergyInterpolation(E0=1.0))
    assert alone.density_map is None
    E = np.linspace(1e-6, 1.0, d.n_active)
    g, dg = alone._gamma(E, 0.1)
    rho = E ** (1.0 / 3.0)
    g_old = 0.5 * (1.0 + np.tanh(0.5 * 120.0 * (rho - 0.1)))
    assert np.array_equal(g, g_old)

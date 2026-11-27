"""Viscoelastic topology optimisation under a temperature history (docs/foundations/prereg/VT_viscoelastic_temperature.md,
VT1-VT5)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.constitutive import TimeStep
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries, WLFShift
from toporlogy.fem.mesh.history import HistoryProblem
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure

from test_viscoelastic_topology import TIMES, _cantilever, _general, _schedule, _weights

MPA = 1e6
SHIFT = WLFShift(T_ref=300.0, C1=17.4, C2=51.6)


def _shifted(c: float = 1.0) -> GeneralizedMaxwell:
    g = _general()
    return GeneralizedMaxwell(PronySeries(c * g.shear.inf, c * g.shear.amplitudes, g.shear.taus),
                              PronySeries(c * g.bulk.inf, c * g.bulk.amplitudes, g.bulk.taus), shift=SHIFT)


TEMPS = np.minimum(300.0 + 30.0 * TIMES / 3.5, 330.0)            # ramp to 330 K by 3.5 s, then hold


def _setup(mat, temps=TEMPS, T0=300.0, times=TIMES):
    m, fixed, tip = _cantilever()
    loads = _schedule(m.n_nodes * 2, tip, times, 2.0)
    return m, fixed, loads, ViscoelasticStructure(m, mat, times, loads, fixed, temperatures=temps, temperature0=T0)


@pytest.mark.parametrize("c", [1.0, 0.37])
def test_vt1_forward_equals_history_fem(c) -> None:
    m, fixed, loads, st = _setup(_shifted())
    u = st.response(torch.full((m.n_cells,), c, dtype=torch.float64)).u.detach().numpy()
    hp = HistoryProblem(m, _shifted(c))
    tp, Tp, err = 0.0, 300.0, 0.0
    for k, (t, T) in enumerate(zip(TIMES, TEMPS, strict=True)):
        r = hp.step(TimeStep(tp, t - tp, float(T), Tp), f_ext=loads[k], prescribed=(fixed, np.zeros(len(fixed))),
                    tol=1e-13)
        tp, Tp = t, float(T)
        err = max(err, float(np.abs(r.u - u[k]).max() / np.abs(r.u).max()))
    print(f"VT1 c={c}: {err:.2e}")
    assert err <= 1e-10


def test_vt2_gradient_fd() -> None:
    m, fixed, loads, st = _setup(_shifted())
    rng = np.random.default_rng(4)
    x = 0.3 + 0.6 * rng.random(m.n_cells)
    d = rng.normal(size=m.n_cells)
    xt = torch.as_tensor(x).requires_grad_(True)
    st.response(_weights(xt)).compliance[-1].backward()
    g = float(xt.grad.numpy() @ d)
    errs = []
    with torch.no_grad():
        for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            jp = float(st.response(_weights(torch.as_tensor(x + h * d))).compliance[-1])
            jm = float(st.response(_weights(torch.as_tensor(x - h * d))).compliance[-1])
            errs.append(abs((jp - jm) / (2 * h) - g) / abs(g))
    print(f"VT2 FD {[f'{e:.1e}' for e in errs]}")
    assert min(errs) <= 1e-6


def _J_and_grad(st, x):
    xt = torch.as_tensor(x).requires_grad_(True)
    J = st.response(_weights(xt)).compliance[-1]
    J.backward()
    return float(J), xt.grad.numpy().copy()


def test_vt3_reference_temperature_equals_unshifted() -> None:
    m, fixed, loads, st = _setup(_shifted(), temps=np.full(len(TIMES), 300.0))
    plain = ViscoelasticStructure(m, _general(), TIMES, loads, fixed)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    J1, g1 = _J_and_grad(st, x)
    J2, g2 = _J_and_grad(plain, x)
    assert J1 == J2 and np.array_equal(g1, g2)


def test_vt4_time_temperature_superposition() -> None:
    T = 320.0
    aT = float(SHIFT.aT(T))
    m, fixed, loads, st = _setup(_shifted(), temps=np.full(len(TIMES), T), T0=T)
    plain = ViscoelasticStructure(m, _general(), TIMES / aT, loads, fixed)
    w = torch.as_tensor(_weights(0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)))
    with torch.no_grad():
        u1, u2 = st.response(w).u.numpy(), plain.response(w).u.numpy()
    rel = np.abs(u1 - u2).max() / np.abs(u2).max()
    print(f"VT4 a_T(320 K) = {aT:.4e}, rel {rel:.2e}")
    assert rel <= 1e-12


def test_vt5_refusals() -> None:
    m, fixed, tip = _cantilever()
    loads = _schedule(m.n_nodes * 2, tip, TIMES, 2.0)
    bounded = GeneralizedMaxwell(_general().shear, _general().bulk,
                                 shift=WLFShift(T_ref=300.0, C1=17.4, C2=51.6, T_min=290.0, T_max=320.0))
    for kw in [dict(), dict(temperatures=TEMPS), dict(temperature0=300.0), dict(temperatures=TEMPS[:3],
                                                                              temperature0=300.0)]:
        with pytest.raises(ValueError):
            ViscoelasticStructure(m, _shifted(), TIMES, loads, fixed, **kw)
    with pytest.raises(ValueError):
        ViscoelasticStructure(m, bounded, TIMES, loads, fixed, temperatures=TEMPS, temperature0=300.0)
    with pytest.raises(ValueError):
        ViscoelasticStructure(m, _general(), TIMES, loads, fixed, temperatures=TEMPS, temperature0=300.0)

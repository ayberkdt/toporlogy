"""Creep topology optimisation under a temperature history (docs/foundations/prereg/CT_creep_temperature.md, CT1-CT4)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.constitutive import TimeStep
from toporlogy.core.creep import ArrheniusFactor, NortonCreep
from toporlogy.fem.mesh.history import HistoryProblem
from toporlogy.optimization.creep_topology import CreepStructure

from test_creep_topology import A, G0, K0, SREF, TIMES, _cantilever, _w

ARR = ArrheniusFactor(80e3, 300.0)
TEMPS = np.minimum(300.0 + 30.0 * TIMES / 10.0, 330.0)


def _st(temps=TEMPS, T0=300.0):
    m, fixed, loads = _cantilever()
    return m, fixed, loads, CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12, arrhenius=ARR,
                                           temperatures=temps, temperature0=T0)


@pytest.mark.parametrize("c", [1.0, 0.4])
def test_ct1_forward_equals_history_fem(c) -> None:
    m, fixed, loads, st = _st()
    u = st.response(torch.full((m.n_cells,), c, dtype=torch.float64)).u.detach().numpy()
    hp = HistoryProblem(m, NortonCreep(c * K0, c * G0, A, SREF, 3.0, arrhenius=ARR))
    tp, Tp, err = 0.0, 300.0, 0.0
    for k, (t, T) in enumerate(zip(TIMES, TEMPS, strict=True)):
        r = hp.step(TimeStep(tp, t - tp, float(T), Tp), f_ext=loads[k], prescribed=(fixed, np.zeros(len(fixed))),
                    tol=1e-12)
        tp, Tp = t, float(T)
        err = max(err, float(np.abs(r.u - u[k]).max() / np.abs(r.u).max()))
    print(f"CT1 c={c}: {err:.2e}")
    assert err <= 1e-9


def test_ct2_gradient_fd() -> None:
    m, fixed, loads, st = _st()
    rng = np.random.default_rng(4)
    x = 0.4 + 0.5 * rng.random(m.n_cells)
    d = rng.normal(size=m.n_cells)
    xt = torch.as_tensor(x).requires_grad_(True)
    st.response(_w(xt)).compliance[-1].backward()
    g = float(xt.grad.numpy() @ d)
    errs = []
    with torch.no_grad():
        for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            jp = float(st.response(_w(torch.as_tensor(x + h * d))).compliance[-1])
            jm = float(st.response(_w(torch.as_tensor(x - h * d))).compliance[-1])
            errs.append(abs((jp - jm) / (2 * h) - g) / abs(g))
    print(f"CT2 FD {[f'{e:.1e}' for e in errs]}")
    assert min(errs) <= 1e-6


def test_ct3_reference_temperature_equals_isothermal() -> None:
    m, fixed, loads, st = _st(temps=np.full(len(TIMES), 300.0))
    iso = CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12)
    x = 0.4 + 0.5 * np.random.default_rng(4).random(m.n_cells)
    out = []
    for s in (st, iso):
        xt = torch.as_tensor(x).requires_grad_(True)
        J = s.response(_w(xt)).compliance[-1]
        J.backward()
        out.append((float(J), xt.grad.numpy().copy()))
    assert out[0][0] == out[1][0] and np.array_equal(out[0][1], out[1][1])


def test_ct4_refusals() -> None:
    m, fixed, loads = _cantilever()
    base = (m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed)
    for kw in [dict(arrhenius=ARR), dict(arrhenius=ARR, temperatures=TEMPS), dict(temperatures=TEMPS),
               dict(arrhenius=ARR, temperatures=TEMPS[:3], temperature0=300.0)]:
        with pytest.raises(ValueError):
            CreepStructure(*base, **kw)

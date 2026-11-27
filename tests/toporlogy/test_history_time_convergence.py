"""Time-step convergence of the FE history solver (docs/foundations/prereg/H7_time_convergence.md, H7a-H7c)."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core.constitutive import TimeStep
from toporlogy.core.creep import NortonCreep
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries
from toporlogy.fem.mesh.history import HistoryProblem
from toporlogy.fem.mesh.mesh import rectangle

MPA = 1e6
T_END = 100.0
LEVELS = (10, 20, 40, 80, 160)
N_REF = 1280


def _tip_u(model, F, n):
    m = rectangle(8, 4, 2.0, 1.0)
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    fixed = np.concatenate([2 * left, 2 * left + 1])
    tip = int(np.flatnonzero(np.isclose(P[:, 0], 2.0) & np.isclose(P[:, 1], 0.0))[0])
    hp = HistoryProblem(m, model)
    f = np.zeros(hp.ndof)
    f[2 * tip + 1] = -F
    pres = (fixed, np.zeros(len(fixed)))
    hp.step(TimeStep(0.0, 0.0), f_ext=f, prescribed=pres, tol=1e-12)
    dt = T_END / n
    for k in range(n):
        r = hp.step(TimeStep(k * dt, dt), f_ext=f, prescribed=pres, tol=1e-12)
    return float(r.u[2 * tip + 1])


def _orders(model, F):
    ref = _tip_u(model, F, N_REF)
    e = np.array([abs(_tip_u(model, F, n) - ref) for n in LEVELS])
    p = np.log2(e[:-1] / e[1:])
    return e, p


@pytest.mark.slow
def test_h7a_norton_first_order() -> None:
    e, p = _orders(NortonCreep(3.33e9, 0.77e9, 1e-4, 10 * MPA, 3.0), 2.0e5)
    print(f"H7a errors {e}, orders {p}")
    assert np.all(np.diff(e) < 0)                                            # H7c
    assert np.all((p[-2:] >= 0.9) & (p[-2:] <= 1.1))


@pytest.mark.slow
def test_h7b_generalized_maxwell_second_order() -> None:
    gm = GeneralizedMaxwell(PronySeries(1 * MPA, [2 * MPA, 3 * MPA], [1.0, 100.0]), PronySeries(5 * MPA, [4 * MPA], [10.0]))
    e, p = _orders(gm, 1.0e2)
    print(f"H7b errors {e}, orders {p}")
    assert np.all(np.diff(e) < 0)                                            # H7c
    assert np.all((p[-2:] >= 1.8) & (p[-2:] <= 2.2))

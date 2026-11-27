"""The physical history models inside the FE history solver (docs/foundations/prereg/H6_physical_models_fe.md,
H6a-H6e)."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core.constitutive import TimeStep, drive_material_point
from toporlogy.core.creep import NortonCreep
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries
from toporlogy.core.viscoplastic import Perzyna, PhysicalJ2
from toporlogy.fem.mesh.history import HistoryProblem, StepRejected
from toporlogy.fem.mesh.mesh import box

MPA = 1e6
K0, G0 = 3.33e9, 0.77e9
TIMES = np.concatenate([np.linspace(0.25, 1.0, 4), np.logspace(-1, 3, 13)[1:] + 1.0])


def _models():
    return {
        "H6a": (NortonCreep(K0, G0, 1e-6, 10 * MPA, 3.0), lambda t: 15 * MPA * min(t, 1.0)),
        "H6b": (Perzyna(K0, G0, 20 * MPA, 100 * MPA, 10.0, 10 * MPA, 1.0), lambda t: 25 * MPA * min(t, 1.0)),
        "H6c": (GeneralizedMaxwell(PronySeries(1 * MPA, [2 * MPA, 3 * MPA], [1.0, 100.0]),
                                   PronySeries(5 * MPA, [4 * MPA], [10.0])), lambda t: 1e3 * min(t, 1.0)),
        "H6d": (PhysicalJ2(K0, G0, 20 * MPA, 100 * MPA), lambda t: 25 * MPA * min(t, 1.0)),
    }


def _bar():
    m = box(2, 1, 1, 2.0, 1.0, 1.0)
    P = m.points
    sup = sorted([3 * n for n in np.flatnonzero(np.isclose(P[:, 0], 0.0))]
                 + [3 * n + 1 for n in np.flatnonzero(np.isclose(P[:, 1], 0.0))]
                 + [3 * n + 2 for n in np.flatnonzero(np.isclose(P[:, 2], 0.0))])
    right = np.flatnonzero(np.isclose(P[:, 0], 2.0))
    return m, np.array(sup), right


def _fe_run(model, load, max_iter_at=None):
    m, sup, right = _bar()
    hp = HistoryProblem(m, model)
    out = []
    tp = 0.0
    for k, t in enumerate(TIMES):
        f = np.zeros(hp.ndof)
        f[3 * right] = load(t) / len(right)
        kw = {} if max_iter_at != k else {"max_iter": 0}
        r = hp.step(TimeStep(tp, t - tp), f_ext=f, prescribed=(sup, np.zeros(len(sup))), tol=1e-12, **kw)
        out.append(r)
        tp = t
    return hp, out, right


@pytest.mark.parametrize("case", ["H6a", "H6b", "H6c", "H6d"])
def test_h6_bar_equals_material_point(case) -> None:
    model, load = _models()[case]
    hp, out, right = _fe_run(model, load)
    stress = np.zeros((len(TIMES), 6))
    stress[:, 0] = [load(t) for t in TIMES]
    mp = drive_material_point(model, np.zeros((len(TIMES), 6)), stress, np.zeros(6, bool), times=TIMES,
                              tol=1e-10 * float(np.abs(stress).max()))
    for k, r in enumerate(out):
        ux = r.u[3 * right].mean()
        ref = 2.0 * mp.strain[k, 0]
        assert abs(ux - ref) <= 1e-8 * abs(ref), (case, k, ux, ref)


def test_h6e_creep_rollback() -> None:
    model, load = _models()["H6a"]
    m, sup, right = _bar()
    ref_hp, ref_out, _ = _fe_run(model, load)
    hp = HistoryProblem(m, model)
    tp = 0.0
    for k, t in enumerate(TIMES[:8]):
        f = np.zeros(hp.ndof)
        f[3 * right] = load(t) / len(right)
        if k == 7:
            before = {key: v.copy() for key, v in hp.store.committed.items()}
            with pytest.raises(StepRejected):
                hp.step(TimeStep(tp, t - tp), f_ext=f, prescribed=(sup, np.zeros(len(sup))), tol=1e-12, max_iter=0)
            assert all(np.array_equal(hp.store.committed[key], before[key]) for key in before)
        r = hp.step(TimeStep(tp, t - tp), f_ext=f, prescribed=(sup, np.zeros(len(sup))), tol=1e-12)
        tp = t
    assert np.array_equal(r.u, ref_out[7].u)

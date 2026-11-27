"""The later material models inside the FE history solver (docs/foundations/prereg/H8_more_models_fe.md, H8a-H8e)."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core.constitutive import TimeStep, drive_material_point
from toporlogy.core.creep_plasticity import CreepPlasticJ2
from toporlogy.core.thermomechanics import AdiabaticPerzyna
from toporlogy.core.viscoplastic import DruckerPrager
from toporlogy.fem.mesh.history import HistoryProblem
from toporlogy.fem.mesh.mesh import box

import test_adiabatic as TA
import test_cure as TC
import test_thermoplastic as TT

MPA = 1e6
K0, G0 = 3.33e9, 0.77e9


def _bar(both_ends: bool = False):
    m = box(2, 1, 1, 2.0, 1.0, 1.0)
    P = m.points
    xs = np.isclose(P[:, 0], 0.0) | (np.isclose(P[:, 0], 2.0) if both_ends else False)
    sup = sorted([3 * n for n in np.flatnonzero(xs)] + [3 * n + 1 for n in np.flatnonzero(np.isclose(P[:, 1], 0.0))]
                 + [3 * n + 2 for n in np.flatnonzero(np.isclose(P[:, 2], 0.0))])
    return m, np.array(sup), np.flatnonzero(np.isclose(P[:, 0], 2.0))


def _force_case(model, times, load):
    m, sup, right = _bar()
    hp = HistoryProblem(m, model)
    tp, ux = 0.0, []
    for t in times:
        f = np.zeros(hp.ndof)
        f[3 * right] = load(t) / len(right)
        r = hp.step(TimeStep(tp, t - tp), f_ext=f, prescribed=(sup, np.zeros(len(sup))), tol=1e-12)
        ux.append(r.u[3 * right].mean())
        tp = t
    s = np.zeros((len(times), 6))
    s[:, 0] = [load(t) for t in times]
    mp = drive_material_point(model, np.zeros((len(times), 6)), s, np.zeros(6, bool), times=times,
                              tol=1e-10 * float(np.abs(s).max()))
    ref = 2.0 * mp.strain[:, 0]
    return float(np.max(np.abs(np.array(ux) - ref)) / np.abs(ref).max())


def test_h8a_drucker_prager_nonassociated() -> None:
    times = np.linspace(0.1, 1.0, 10)
    err = _force_case(DruckerPrager(K0, G0, 0.3, 20 * MPA, 100 * MPA, a_flow=0.1), times, lambda t: -30 * MPA * t)
    print(f"H8a {err:.2e}")
    assert err <= 1e-8


def test_h8b_creep_plasticity() -> None:
    times = np.concatenate([np.linspace(0.25, 1.0, 4), 1.0 + np.logspace(-1, 3, 13)[1:]])
    err = _force_case(CreepPlasticJ2(K0, G0, 1e-6, 10 * MPA, 3.0, 20 * MPA, 100 * MPA), times,
                      lambda t: 25 * MPA * min(t, 1.0))
    print(f"H8b {err:.2e}")
    assert err <= 1e-8


def test_h8c_adiabatic_perzyna_displacement_control() -> None:
    model = AdiabaticPerzyna(K=TA.K0, G=TA.G0, yield_stress=TA.sy_fn(), hardening=TA.HH, tau0=TA.TAU0,
                             sigma_ref=TA.SREF, m=TA.MM, density=TA.RHO, specific_heat=TA.cp_fn(),
                             taylor_quinney=0.5, temperature0=TA.TREF)
    times = np.linspace(0.5, 10.0, 20)
    m, sup, right = _bar()
    hp = HistoryProblem(m, model)
    tp, sxx, temp = 0.0, [], []
    for t in times:
        e = 0.03 * t / 10.0
        dofs = np.concatenate([sup, 3 * right])
        vals = np.concatenate([np.zeros(len(sup)), np.full(len(right), 2.0 * e)])
        o = np.argsort(dofs)
        r = hp.step(TimeStep(tp, t - tp), prescribed=(dofs[o], vals[o]), tol=1e-12)
        sxx.append(r.stress.reshape(-1, 6)[:, 0])
        temp.append(np.asarray(hp.store.committed["temperature"]).reshape(-1))
        tp = t
    strain = np.zeros((len(times), 6))
    strain[:, 0] = 0.03 * times / 10.0
    mp = drive_material_point(model, strain, np.zeros((len(times), 6)), np.array([True] + [False] * 5), times=times,
                              tol=1e-6)
    s_ref = mp.stress[:, 0]
    T_ref = np.array([float(np.asarray(st["temperature"]).reshape(-1)[0]) for st in mp.states])
    es = max(float(np.abs(s - s_ref[k]).max()) for k, s in enumerate(sxx)) / np.abs(s_ref).max()
    eT = max(float(np.abs(T - T_ref[k]).max()) for k, T in enumerate(temp)) / np.abs(T_ref - TA.TREF).max()
    print(f"H8c stress {es:.2e}, temperature rise {eT:.2e}")
    assert es <= 1e-8 and eT <= 1e-8


def _thermal_constrained(model, times, temps, T0, state_key):
    m, sup, _right = _bar(both_ends=True)
    hp = HistoryProblem(m, model)
    tp, Tp, sxx, st = 0.0, T0, [], []
    for t, T in zip(times, temps, strict=True):
        r = hp.step(TimeStep(tp, t - tp, float(T), Tp), prescribed=(sup, np.zeros(len(sup))), tol=1e-12)
        sxx.append(r.stress.reshape(-1, 6)[:, 0])
        st.append(np.asarray(hp.store.committed[state_key]).reshape(-1))
        tp, Tp = t, float(T)
    mp = drive_material_point(model, np.zeros((len(times), 6)), np.zeros((len(times), 6)),
                              np.array([True] + [False] * 5), times=times, temperatures=temps, temperature0=T0,
                              tol=1e-6)
    s_ref = mp.stress[:, 0]
    a_ref = np.array([float(np.asarray(s[state_key]).reshape(-1)[0]) for s in mp.states])
    es = max(float(np.abs(s - s_ref[k]).max()) for k, s in enumerate(sxx)) / max(np.abs(s_ref).max(), 1e-300)
    ea = max(float(np.abs(a - a_ref[k]).max()) for k, a in enumerate(st)) / max(np.abs(a_ref).max(), 1e-300)
    return es, ea


def test_h8d_curing_point_constrained_bar() -> None:
    times = np.linspace(20.0, 600.0, 30)
    temps = 300.0 + 120.0 * times / 600.0
    es, ea = _thermal_constrained(TC.make_point(), times, temps, 300.0, "alpha")
    print(f"H8d stress {es:.2e}, cure {ea:.2e}")
    assert es <= 1e-8 and ea <= 1e-8


def test_h8e_crystallising_point_constrained_bar() -> None:
    times = np.linspace(2.0, 60.0, 30)
    temps = 400.0 - 70.0 * times / 60.0
    es, ea = _thermal_constrained(TT.make_point(), times, temps, 400.0, "X")
    print(f"H8e stress {es:.2e}, crystallinity {ea:.2e}")
    assert es <= 1e-8 and ea <= 1e-8

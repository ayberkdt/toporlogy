"""Physical time in the material-point contract (docs/foundations/prereg/P01_contract.md, CT1-CT8)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (HistoryUpdate, J2Plasticity, Kinematics, MaterialRequest, StateStore,
                                         TimeStep, drive_material_point, require_step,
                                         small_strain_history_evaluate)


# ----------------------------------------------------------------------------------------------- CT1, CT2
def test_ct1_time_step_validation() -> None:
    for bad in [dict(t=0.0, dt=-1e-9), dict(t=float("nan"), dt=1.0), dict(t=0.0, dt=float("inf")),
                dict(t=0.0, dt=1.0, temperature=0.0), dict(t=0.0, dt=1.0, temperature=-20.0),
                dict(t=0.0, dt=1.0, temperature=float("nan")), dict(t=0.0, dt=1.0, temperature_prev=300.0),
                dict(t=0.0, dt=1.0, temperature=300.0, temperature_prev=-1.0)]:
        with pytest.raises(ValueError):
            TimeStep(**bad)
    st = TimeStep(1.0, 0.5, 310.0)
    assert st.temperature_prev == 310.0 and st.t_next == 1.5
    st = TimeStep(0.0, 0.0, np.array([300.0, 301.0]), np.array([299.0, 300.0]))
    assert isinstance(st.temperature, np.ndarray) and not st.temperature.flags.writeable


def test_ct2_require_step() -> None:
    m = J2Plasticity(0.3, 1e-3)
    with pytest.raises(ValueError, match="load factor is not time"):
        require_step(m, MaterialRequest())
    with pytest.raises(ValueError, match="temperature"):
        require_step(m, MaterialRequest(step=TimeStep(0.0, 1.0)), temperature=True)
    st = TimeStep(0.0, 1.0, 300.0)
    assert require_step(m, MaterialRequest(step=st), temperature=True) is st


# ----------------------------------------------------------------------------------------------- CT3, CT6
def _j2_path(k: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctrl = np.array([True, False, False, False, False, True])
    ex = np.concatenate([np.linspace(0, 4e-3, k // 2 + 1)[1:], np.linspace(4e-3, -1e-3, k - k // 2 + 1)[1:]])
    strain = np.zeros((k, 6))
    strain[:, 0] = ex
    strain[:, 5] = 0.3 * ex
    return strain, np.zeros((k, 6)), ctrl


def _old_loop(model: Any, strain: np.ndarray, stress: np.ndarray, control: np.ndarray,
              tol: float = 1e-14, max_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """The loop of drive_material_point before P01 (regression reference, written out here)."""
    free = ~control
    store = StateStore(model, 1)
    e = np.zeros(6)
    out_e, out_s = [], []
    for k in range(len(strain)):
        e = e.copy()
        e[control] = strain[k, control]
        for it in range(max_iter + 1):
            r = model.evaluate(Kinematics(T.voigt_to_strain(e)[None], "small"), None, store.committed,
                               MaterialRequest(energy=True))
            res = r.stress[0, free] - stress[k, free]
            if not free.any() or np.abs(res).max() <= tol:
                break
            Ce = getattr(model, "Ce6", None)
            Cff = (Ce if (it == 0 and Ce is not None) else r.tangent[0])[np.ix_(free, free)]
            e[free] -= np.linalg.solve(Cff, res)
        store.set_trial(r.trial_state)
        store.commit()
        out_e.append(e.copy())
        out_s.append(r.stress[0].copy())
    return np.array(out_e), np.array(out_s)


def test_ct3_untimed_driver_unchanged() -> None:
    m = J2Plasticity(0.3, 2e-3, hardening=0.05)
    strain, stress, ctrl = _j2_path()
    h = drive_material_point(m, strain, stress, ctrl)
    e_ref, s_ref = _old_loop(m, strain, stress, ctrl)
    assert np.array_equal(h.strain, e_ref) and np.array_equal(h.stress, s_ref)
    assert h.times is None and h.temperatures is None


def test_ct6_restart_equals_uninterrupted() -> None:
    m = J2Plasticity(0.3, 2e-3, hardening=0.05)
    strain, stress, ctrl = _j2_path()
    full = drive_material_point(m, strain, stress, ctrl)
    first = drive_material_point(m, strain[:5], stress[:5], ctrl)
    store = StateStore(m, 1)
    store.restore(first.states[-1])
    second = drive_material_point(m, strain[5:], stress[5:], ctrl, store=store, strain0=first.strain[-1])
    assert np.array_equal(np.vstack([first.stress, second.stress]), full.stress)
    assert np.array_equal(np.vstack([first.strain, second.strain]), full.strain)
    for a, b in zip(first.states + second.states, full.states):
        assert all(np.array_equal(a[key], b[key]) for key in a)


# ----------------------------------------------------------------------------------------------- probes
class _Probe:
    """Linear isotropic history probe (σ = C ε, state = a counter of committed steps); records every request."""

    name = "probe"
    kinematics = "small"
    conservative = False
    physical = True
    state_fields = ("n_steps",)

    def __init__(self) -> None:
        K, G = 2.0, 1.0
        one = np.array([1.0, 1, 1, 0, 0, 0])
        Idev = np.diag([1.0, 1, 1, 0.5, 0.5, 0.5]) - np.outer(one, one) / 3.0
        self.C = K * np.outer(one, one) + 2 * G * Idev
        self.steps: list[TimeStep | None] = []
        self.seen: list[tuple[np.ndarray, bool]] = []

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"n_steps": np.zeros(n)}

    def update(self, eps: np.ndarray, committed: Any, request: MaterialRequest) -> HistoryUpdate:
        self.steps.append(request.step)
        c = committed["n_steps"]
        self.seen.append((np.array(c, copy=True), bool(c.flags.writeable)))
        ev = T.strain_to_voigt(eps, rtol=None)
        sv = ev @ self.C.T
        return HistoryUpdate(T.voigt_to_stress(sv), np.broadcast_to(self.C, (len(eps), 6, 6)).copy(),
                             {"n_steps": np.asarray(c) + 1.0})

    def evaluate(self, kin: Kinematics, params: Any, committed: Any, request: MaterialRequest) -> Any:
        return small_strain_history_evaluate(self, kin, committed, request)


def test_ct4_time_and_temperature_schedule() -> None:
    p = _Probe()
    times = [0.5, 1.5, 1.5, 4.0]
    temps = [300.0, 310.0, 320.0, 305.0]
    strain = np.zeros((4, 6))
    strain[:, 0] = [1e-3, 2e-3, 3e-3, 4e-3]
    ctrl = np.array([True] * 6)
    drive_material_point(p, strain, np.zeros((4, 6)), ctrl, times=times, temperatures=temps, t0=0.1,
                         temperature0=295.0)
    tp = [0.1] + times[:-1]
    Tp = [295.0] + temps[:-1]
    assert len(p.steps) == 4
    for k, st in enumerate(p.steps):
        assert st is not None
        assert (st.t, st.dt, st.temperature, st.temperature_prev) == (tp[k], times[k] - tp[k], temps[k], Tp[k])
    for bad in [dict(times=[1.0, 0.5, 2.0, 3.0]), dict(times=[1.0, 2.0]), dict(times=[0.05, 1, 2, 3], t0=0.1),
                dict(temperatures=temps, temperature0=295.0), dict(times=times, temperatures=temps),
                dict(times=times, temperatures=temps[:2], temperature0=295.0)]:
        with pytest.raises(ValueError):
            drive_material_point(_Probe(), strain, np.zeros((4, 6)), ctrl, **bad)


def test_ct5_no_commit_inside_newton() -> None:
    p = _Probe()
    k = 3
    strain = np.zeros((k, 6))
    strain[:, 0] = [1e-3, 2e-3, 3e-3]
    stress = np.zeros((k, 6))
    ctrl = np.array([True, False, False, False, False, False])               # lateral and shear stresses free
    h = drive_material_point(p, strain, stress, ctrl, times=[1.0, 2.0, 3.0])
    assert all(not w for _, w in p.seen)                                      # read-only committed state
    per_step: dict[float, set[float]] = {}
    for (c, _), st in zip(p.seen, p.steps):
        assert st is not None
        per_step.setdefault(st.t, set()).add(float(c[0]))
    assert [sorted(per_step[t]) for t in sorted(per_step)] == [[0.0], [1.0], [2.0]]
    assert len(p.seen) > k                                                    # stress control iterated
    assert float(h.states[-1]["n_steps"][0]) == k


def test_ct7_nonlinear_elasticity_refuses_physical_model() -> None:
    from toporlogy.core.domain import Domain
    from toporlogy.fem.nonlinear import NonlinearElasticity
    d = Domain((4, 2))
    with pytest.raises(ValueError, match="element modulus"):
        NonlinearElasticity(d, [0, 1], material=_Probe())


def test_ct8_plane_strain_embedding_and_plane_stress_refusal() -> None:
    p = _Probe()
    rng = np.random.default_rng(3)
    H2 = 1e-3 * rng.normal(size=(5, 2, 2))
    H3 = np.zeros((5, 3, 3))
    H3[:, :2, :2] = H2
    req = MaterialRequest(step=TimeStep(0.0, 1.0))
    st = p.initial_state(5)
    r2 = small_strain_history_evaluate(p, Kinematics(H2, "small", "strain"), st, req)
    r3 = small_strain_history_evaluate(p, Kinematics(H3, "small"), st, req)
    ip = [0, 1, 5]
    assert np.array_equal(r2.stress, r3.stress[:, ip])
    assert np.array_equal(r2.tangent, r3.tangent[:, ip][:, :, ip])
    assert np.array_equal(r2.diagnostics["sigma_zz"], r3.stress[:, 2])
    with pytest.raises(ValueError, match="plane strain only"):
        small_strain_history_evaluate(p, Kinematics(H2, "small", "stress"), st, req)

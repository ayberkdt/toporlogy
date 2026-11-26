"""Transient heat conduction and Newmark dynamics as Study physics (docs/olcumler/gecici_study_onkayit.md, HS1-HS7 and
DS1-DS7; G1 is the baseline run and the resolved-config snapshot)."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.fem.heat import element_nodes
from toporlogy.optimization.transient import time_weights

ERRORS = (ValueError, ConfigError)


def _base(dim: int) -> dict:
    shape = [12, 6] if dim == 2 else [6, 3, 3]
    return {"domain": {"type": "box", "shape": shape, "h": 1.0}, "material": {"E0": 1.0, "nu": 0.3, "rho": 1.0},
            "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
            "chain": [{"type": "density_filter", "rmin": 1.5}],
            "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma"}


def heat_cfg(dim: int = 2, **physics: object) -> dict:
    cfg = _base(dim)
    cfg["physics"] = {"type": "transient_heat", "dt": 0.5, "n_steps": 12, "theta": 0.5, "source_density": 1.0,
                      **physics}
    cfg["objective"] = "transient_thermal_compliance"
    return cfg


def dyn_cfg(dim: int = 2, **physics: object) -> dict:
    cfg = _base(dim)
    at = ["max", "min"] if dim == 2 else ["max", "min", "min"]
    f = [0, -1] if dim == 2 else [0, -1, 0]
    cfg["load_cases"] = [{"loads": [{"type": "force", "where": {"type": "point", "at": at}, "force": f}]}]
    cfg["physics"] = {"type": "dynamics_newmark", "dt": 2.0, "n_steps": 30, **physics}
    cfg["objective"] = "dynamic_compliance"
    return cfg


def _x(st, seed: int = 2) -> np.ndarray:
    return 0.2 + 0.7 * np.random.default_rng(seed).random(st.n_var)


def _active(st, xphys: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """(E_active, x̄) of the study at a physical density."""
    xt = torch.as_tensor(xphys)
    E = st.interp(xt).reshape(-1)[torch.as_tensor(st.domain.active_elements)]
    return E, xt


def _fd(st, x: np.ndarray, seed: int = 7) -> tuple[float, float]:
    v = np.random.default_rng(seed).normal(size=st.n_var)
    ev = st.evaluate(x)
    t = 1e-6
    fd = (st.evaluate(x + t * v, False).f0 - st.evaluate(x - t * v, False).f0) / (2 * t)
    return float(ev.df0 @ v), fd


# ============================================================================================ heat
def test_hs1_value_equals_direct_run() -> None:
    st = build_setup(heat_cfg(initial_temperature=0.3)).study
    ev = st.evaluate(_x(st), False)
    ph = st.physics
    E, xt = _active(st, ev.xphys)
    T = ph.heat.run(E * ph.conductivity / ph.E0, ph.capacities(xt), torch.as_tensor(ph.T0), ph.dt, ph.n_steps,
                    torch.as_tensor(ph.q))
    w = time_weights(ph.n_steps, ph.dt)
    J = float(sum(w[n] * float(T[n] @ torch.as_tensor(ph.q)) for n in range(ph.n_steps + 1)))
    print(f"HS1: J = {ev.f0:.12g}, direct {J:.12g}, rel {abs(ev.f0 - J) / abs(J):.2e}")
    assert abs(ev.f0 - J) <= 1e-12 * abs(J)


def _q4_matrices(h: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Conduction ∫ t ∇N·∇N dA and capacity ∫ t N N dA of a square Q4 of side h, own 2 x 2 Gauss loop (corner order
    (-1,-1), (1,-1), (1,1), (-1,1))."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    g = 1.0 / math.sqrt(3.0)
    Kc, Cc = np.zeros((4, 4)), np.zeros((4, 4))
    for xi in (-g, g):
        for eta in (-g, g):
            N = (1 + corners[:, 0] * xi) * (1 + corners[:, 1] * eta) / 4
            dxi = corners[:, 0] * (1 + corners[:, 1] * eta) / 4
            deta = corners[:, 1] * (1 + corners[:, 0] * xi) / 4
            B = np.stack([dxi, deta]) * (2.0 / h)            # dN/dx, dN/dy
            jac = (h / 2) ** 2
            Kc += t * B.T @ B * jac
            Cc += t * np.outer(N, N) * jac
    return Kc, Cc


def test_hs2_value_equals_dense_theta_method() -> None:
    st = build_setup(heat_cfg(initial_temperature=0.3, capacity_min_ratio=0.1)).study
    ev = st.evaluate(_x(st), False)
    ph, d = st.physics, st.domain
    E, xt = _active(st, ev.xphys)
    k = (E * ph.conductivity / ph.E0).numpy()
    c = ph.capacities(xt).numpy()
    Kc, Cc = _q4_matrices(d.h, d.thickness)
    nodes = element_nodes(d, d.active_elements)
    K, C = np.zeros((d.nn, d.nn)), np.zeros((d.nn, d.nn))
    for e, nd in enumerate(nodes):
        K[np.ix_(nd, nd)] += k[e] * Kc
        C[np.ix_(nd, nd)] += c[e] * Cc
    fixed = np.asarray(ph.heat.fixed)
    free = np.setdiff1d(np.arange(d.nn), fixed)
    dt, th, q = ph.dt, ph.heat.theta, ph.q
    A = C / dt + th * K
    Bm = C / dt - (1 - th) * K
    T = np.full(d.nn, 0.3)
    T[fixed] = 0.0
    w = time_weights(ph.n_steps, dt)
    J = w[0] * q @ T
    for n in range(1, ph.n_steps + 1):
        rhs = Bm @ T + q
        Tn = np.zeros(d.nn)
        Tn[free] = np.linalg.solve(A[np.ix_(free, free)], rhs[free] - A[np.ix_(free, fixed)] @ Tn[fixed])
        T = Tn
        J += w[n] * q @ T
    print(f"HS2: J = {ev.f0:.12g}, dense {J:.12g}, rel {abs(ev.f0 - J) / abs(J):.2e}")
    assert abs(ev.f0 - J) <= 1e-10 * abs(J)


@pytest.mark.parametrize(("dim", "theta"), [(2, 0.5), (3, 1.0)])
def test_hs3_gradient_finite_difference(dim: int, theta: float) -> None:
    st = build_setup(heat_cfg(dim, theta=theta, capacity_min_ratio=0.1)).study
    an, fd = _fd(st, _x(st))
    print(f"HS3 {dim}-D theta {theta}: rel {abs(an - fd) / abs(fd):.2e}")
    assert abs(an - fd) <= 1e-6 * abs(fd)


def _dense(st, ev, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stiffness-like, mass-like) dense matrices on the free unknowns and the free load (heat: K, C, q; dynamics: K,
    M, F) from the library element matrices at the evaluation's density."""
    ph = st.physics
    E, xt = _active(st, ev.xphys)
    if kind == "heat":
        Kp = ph.heat.system
        K = Kp.matrix(torch.stack([torch.zeros_like(E), E * ph.conductivity / ph.E0]).numpy()).toarray()
        M = Kp.matrix(torch.stack([ph.capacities(xt), torch.zeros_like(E)]).numpy()).toarray()
        free, load = Kp.free, ph.q
    else:
        nm = ph.nm
        K = nm.effective.matrix(torch.stack([E, torch.zeros_like(E)]).numpy()).toarray()
        M = nm.mass.matrix(ph.densities(xt).numpy()).toarray()
        free, load = nm.effective.free, st.F[:, 0]
    return K[np.ix_(free, free)], M[np.ix_(free, free)], load[free]


def test_hs4_long_time_limit_is_the_static_thermal_compliance() -> None:
    cfg = heat_cfg(theta=1.0)
    st = build_setup(cfg).study
    x = _x(st)
    K, C, q = _dense(st, st.evaluate(x, False), "heat")
    lam1 = float(sla.eigh(K, C, eigvals_only=True, subset_by_index=[0, 0])[0])
    c_static = float(q @ np.linalg.solve(K, q))
    cfg["physics"].update(dt=0.1 / lam1, n_steps=2000)
    J = build_setup(cfg).study.evaluate(x, False).f0
    dev = J / c_static - 1.0
    print(f"HS4: lambda1 {lam1:.4g}, J {J:.6g}, static {c_static:.6g}, J/static - 1 = {dev:.3e} "
          f"(bound 1/(lambda1 t_N) = {1 / 200:.1e})")
    assert abs(dev) <= 1e-2


def test_hs5_yaml_runs() -> None:
    res = build_setup(heat_cfg()).study.run(max_iter=5)
    f0 = res.history["f0"]
    print(f"HS5: f0 {f0[0]:.6g} -> {f0[-1]:.6g}")
    assert np.isfinite(res.objective) and np.isfinite(res.constraint_values["volume"])
    assert f0[-1] < f0[0]


HEAT_REFUSALS = {
    "load_cases": "takes no load_cases",
    "roller": "every component",
    "oblique": "oblique rollers",
    "q4im": r"Q4 \(2-D\) and H8 \(3-D\) only",
    "theta": r"theta must lie in \[0, 1\]",
    "dt": "dt > 0",
    "n_steps": "n_steps must be an integer >= 1",
    "no_input": "needs a heat input",
    "no_dt": "needs dt and n_steps",
    "compliance": r"unsupported combination \(catalog\): response compliance with physics transient_heat",
    "elasticity": "needs physics transient_heat",
}


HEAT_REFUSALS["roller_posthoc"] = "every component"      # POST HOC (added after the first run, see "roller")
_ROLLER_XFAIL = pytest.mark.xfail(strict=True, reason=(
    "pre-registered HS6 'roller' FAILED on the first run: the setup is refused, but by the build's support check "
    "before the physics ('supports: rigid-body: 1 unrestrained rigid-body mode(s) of a component with 91 nodes: "
    "translation along (0, 1)'), not with the expected 'every component'; the probe (the only support fixes x) is "
    "mechanically under-constrained. Post hoc probe roller_posthoc reaches the physics check"))


@pytest.mark.parametrize("change", [pytest.param(c, marks=_ROLLER_XFAIL) if c == "roller" else c
                                    for c in HEAT_REFUSALS])
def test_hs6_rejects(change: str) -> None:
    cfg = heat_cfg()
    ph = cfg["physics"]
    if change == "load_cases":
        cfg["load_cases"] = dyn_cfg()["load_cases"]
    elif change == "roller":
        cfg["supports"][0]["dofs"] = "x"
    elif change == "roller_posthoc":           # a clamped edge restrains the body; a y-roller adds a partial node
        cfg["supports"].append({"type": "fixed", "where": {"type": "point", "at": ["max", "max"]}, "dofs": "y"})
    elif change == "oblique":
        cfg["supports"].append({"type": "roller", "where": {"type": "point", "at": ["max", "max"]}, "normal": [1, 1]})
    elif change == "q4im":
        ph["element"] = "Q4IM"
    elif change == "theta":
        ph["theta"] = 1.5
    elif change == "dt":
        ph["dt"] = 0.0
    elif change == "n_steps":
        ph["n_steps"] = 0
    elif change == "no_input":
        del ph["source_density"]
    elif change == "no_dt":
        del ph["dt"]
    elif change == "compliance":
        cfg["objective"] = "compliance"
    else:
        cfg = dyn_cfg()
        del cfg["physics"]
        cfg["objective"] = "transient_thermal_compliance"
    with pytest.raises(ERRORS, match=HEAT_REFUSALS[change]):
        build_setup(cfg)


def test_hs7_one_factorisation_per_evaluation() -> None:
    st = build_setup(heat_cfg()).study
    sys_ = st.physics.heat.system
    n0 = sys_.n_factorizations
    st.evaluate(_x(st))
    print(f"HS7: factorisations {sys_.n_factorizations - n0}")
    assert sys_.n_factorizations - n0 == 1


# ============================================================================================ dynamics
def test_ds1_value_equals_direct_run() -> None:
    st = build_setup(dyn_cfg()).study
    ev = st.evaluate(_x(st), False)
    ph = st.physics
    E, xt = _active(st, ev.xphys)
    F = torch.as_tensor(st.F[:, 0])
    zero = torch.zeros(st.domain.ndof, dtype=torch.float64)
    res = ph.nm.run(E, ph.densities(xt), zero, zero, ph.dt, ph.n_steps, lambda t: F)
    w = time_weights(ph.n_steps, ph.dt)
    J = float(sum(w[n] * float(F @ res.u[n]) for n in range(ph.n_steps + 1)))
    print(f"DS1: J = {ev.f0:.12g}, direct {J:.12g}, rel {abs(ev.f0 - J) / abs(J):.2e}")
    assert abs(ev.f0 - J) <= 1e-12 * abs(J)


def test_ds2_value_equals_dense_newmark() -> None:
    a_m, b_k = 0.01, 0.02
    st = build_setup(dyn_cfg(profile="half_sine", pulse_duration=15.0, rayleigh_alpha=a_m,
                             rayleigh_beta=b_k)).study
    ev = st.evaluate(_x(st), False)
    ph = st.physics
    K, M, F = _dense(st, ev, "dynamics")
    C = a_m * M + b_k * K
    beta, gamma, dt = 0.25, 0.5, ph.dt
    s = [ph.load_factor(t) for t in ph.times()]
    u = np.zeros(len(F))
    v = np.zeros(len(F))
    a = np.linalg.solve(M, s[0] * F)
    w = time_weights(ph.n_steps, dt)
    J = w[0] * s[0] * F @ u
    A = M + gamma * dt * C + beta * dt * dt * K
    for n in range(1, ph.n_steps + 1):
        up = u + dt * v + (0.5 - beta) * dt * dt * a
        vp = v + (1 - gamma) * dt * a
        a = np.linalg.solve(A, s[n] * F - C @ vp - K @ up)
        u = up + beta * dt * dt * a
        v = vp + gamma * dt * a
        J += w[n] * s[n] * F @ u
    print(f"DS2: J = {ev.f0:.12g}, dense {J:.12g}, rel {abs(ev.f0 - J) / abs(J):.2e}")
    assert abs(ev.f0 - J) <= 1e-9 * abs(J)


@pytest.mark.parametrize("dim", [2, 3])
def test_ds3_gradient_finite_difference(dim: int) -> None:
    extra = (dict(profile="half_sine", pulse_duration=15.0, rayleigh_alpha=0.01, rayleigh_beta=0.02) if dim == 2
             else dict(n_steps=20))
    st = build_setup(dyn_cfg(dim, **extra)).study
    an, fd = _fd(st, _x(st))
    print(f"DS3 {dim}-D: rel {abs(an - fd) / abs(fd):.2e}")
    assert abs(an - fd) <= 1e-6 * abs(fd)


def test_ds4_quasi_static_limit_is_the_static_compliance() -> None:
    cfg = dyn_cfg()
    st = build_setup(cfg).study
    x = _x(st)
    K, M, F = _dense(st, st.evaluate(x, False), "dynamics")
    w1 = math.sqrt(float(sla.eigh(K, M, eigvals_only=True, subset_by_index=[0, 0])[0]))
    T1 = 2 * math.pi / w1
    c_static = float(F @ np.linalg.solve(K, F))
    cfg["physics"].update(dt=T1 / 40, n_steps=2000)
    J = build_setup(cfg).study.evaluate(x, False).f0
    dev = J / c_static - 1.0
    print(f"DS4: T1 {T1:.4g}, J {J:.6g}, static {c_static:.6g}, J/static - 1 = {dev:.3e} "
          f"(bound 1/(omega1 t_N) = {1 / (2 * math.pi * 50):.1e})")
    assert abs(dev) <= 1e-2


def test_ds5_yaml_runs() -> None:
    res = build_setup(dyn_cfg()).study.run(max_iter=5)
    f0 = res.history["f0"]
    print(f"DS5: f0 {f0[0]:.6g} -> {f0[-1]:.6g}")
    assert np.isfinite(res.objective) and np.isfinite(res.constraint_values["volume"])
    assert f0[-1] < f0[0]


DYN_REFUSALS = {
    "two_cases": "exactly one load case",
    "per_case_supports": "exactly one load case",
    "oblique": "oblique rollers",
    "q4im": r"Q4 \(2-D\) and H8 \(3-D\) only",
    "field": "one material D",
    "dt": "dt > 0",
    "n_steps": "n_steps must be an integer >= 1",
    "profile": "profile: step",
    "half_sine": "half_sine needs pulse_duration",
    "compliance": r"unsupported combination \(catalog\): response compliance with physics dynamics_newmark",
    "elasticity": "needs physics dynamics_newmark",
}


@pytest.mark.parametrize("change", list(DYN_REFUSALS))
def test_ds6_rejects(change: str, tmp_path) -> None:
    cfg = dyn_cfg()
    ph = cfg["physics"]
    if change == "two_cases":
        cfg["load_cases"].append(copy.deepcopy(cfg["load_cases"][0]))
    elif change == "per_case_supports":
        cfg["load_cases"].append(copy.deepcopy(cfg["load_cases"][0]))
        cfg["load_cases"][1]["supports"] = [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "max"}}]
    elif change == "oblique":
        cfg["supports"].append({"type": "roller", "where": {"type": "point", "at": ["max", "max"]}, "normal": [1, 1]})
    elif change == "q4im":
        ph["element"] = "Q4IM"
    elif change == "field":
        from test_element_orientation import ORTHO
        f = tmp_path / "angle.npy"
        np.save(f, np.full((12, 6), 25.0))
        cfg["material"] = {"E0": float(ORTHO["E1"]), "nu": 0.3, "orthotropic": {**ORTHO, "angle_field": str(f)}}
    elif change == "dt":
        ph["dt"] = -1.0
    elif change == "n_steps":
        ph["n_steps"] = 0
    elif change == "profile":
        ph["profile"] = "ramp"
    elif change == "half_sine":
        ph["profile"] = "half_sine"
    elif change == "compliance":
        cfg["objective"] = "compliance"
    else:
        del cfg["physics"]
    with pytest.raises(ERRORS, match=DYN_REFUSALS[change]):
        build_setup(cfg)


def test_ds7_one_factorisation_per_evaluation() -> None:
    st = build_setup(dyn_cfg()).study
    nm = st.physics.nm
    e0, m0 = nm.effective.n_factorizations, nm.mass.n_factorizations
    st.evaluate(_x(st))
    print(f"DS7: K_eff {nm.effective.n_factorizations - e0}, mass {nm.mass.n_factorizations - m0}")
    assert nm.effective.n_factorizations - e0 == 1 and nm.mass.n_factorizations - m0 == 1

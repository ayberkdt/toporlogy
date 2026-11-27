"""Topology optimisation under Norton creep with the exact history adjoint (docs/foundations/prereg/CO_creep_topology.md,
CO1-CO5)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.constitutive import MaterialRequest, TimeStep
from toporlogy.core.creep import NortonCreep
from toporlogy.fem.mesh.history import HistoryProblem
from toporlogy.fem.mesh.mesh import perturb, rectangle
from toporlogy.optimization.creep_topology import CreepStructure, norton_update_torch

MPA = 1e6
K0, G0, A, SREF = 3.33e9, 0.77e9, 1e-6, 10 * MPA
TIMES = np.array([0.25, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])


@pytest.mark.parametrize("n", [1.0, 3.0, 5.0])
def test_co1_torch_update_equals_numpy(n) -> None:
    rng = np.random.default_rng(int(n))
    P = 40
    eps = 5e-3 * rng.normal(size=(P, 3, 3))
    eps = 0.5 * (eps + np.swapaxes(eps, 1, 2))
    e0 = 1e-3 * rng.normal(size=(P, 3, 3))
    e0 = 0.5 * (e0 + np.swapaxes(e0, 1, 2))
    e0 -= (np.trace(e0, axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)
    model = NortonCreep(K0, G0, A, SREF, n)
    ref = model.update(eps, {"creep_strain": e0, "p_cr": np.zeros(P)}, MaterialRequest(step=TimeStep(0.0, 50.0)))
    sig, ecr, dp = norton_update_torch(torch.as_tensor(eps), torch.as_tensor(e0), torch.full((P,), K0, dtype=torch.float64),
                                       torch.full((P,), G0, dtype=torch.float64), 50.0, A, SREF, n)
    sc = np.abs(ref.stress).max()
    assert np.abs(sig.numpy() - ref.stress).max() <= 1e-12 * sc
    assert np.abs(ecr.numpy() - ref.trial_state["creep_strain"]).max() <= 1e-12 * np.abs(
        ref.trial_state["creep_strain"]).max()
    assert np.abs(dp.numpy() - ref.diagnostics["delta_p"]).max() <= 1e-12 * np.abs(ref.diagnostics["delta_p"]).max()


def _cantilever(nx=8, ny=4, distort=True, load=2.0e5):
    m = rectangle(nx, ny, 2.0, 1.0)
    if distort:
        m = perturb(m, 0.06, seed=3)
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    fixed = np.concatenate([2 * left, 2 * left + 1])
    tip = int(np.flatnonzero(np.isclose(P[:, 0], 2.0) & np.isclose(P[:, 1], 0.0))[0])
    loads = np.zeros((len(TIMES), m.n_nodes * 2))
    loads[:, 2 * tip + 1] = -load * np.minimum(TIMES, 1.0)
    return m, fixed, loads


@pytest.mark.parametrize("c", [1.0, 0.4])
def test_co2_forward_equals_history_fem(c) -> None:
    m, fixed, loads = _cantilever()
    st = CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12)
    u = st.response(torch.full((m.n_cells,), c, dtype=torch.float64)).u.detach().numpy()
    hp = HistoryProblem(m, NortonCreep(c * K0, c * G0, A, SREF, 3.0))
    tp, err = 0.0, 0.0
    for k, t in enumerate(TIMES):
        r = hp.step(TimeStep(tp, t - tp), f_ext=loads[k], prescribed=(fixed, np.zeros(len(fixed))), tol=1e-12)
        tp = t
        err = max(err, float(np.abs(r.u - u[k]).max() / np.abs(r.u).max()))
    print(f"CO2 c={c}: {err:.2e}")
    assert err <= 1e-9


def _w(x, p=3.0, emin=1e-9):
    return emin + x ** p * (1.0 - emin)


def _grad_and_fd(st, x, d, detach=False):
    xt = torch.as_tensor(x).requires_grad_(True)
    st.response(_w(xt), detach_history=detach).compliance[-1].backward()
    g = float(xt.grad.numpy() @ d)
    return g


def test_co3_gradient_matches_fd() -> None:
    m, fixed, loads = _cantilever()
    st = CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12)
    rng = np.random.default_rng(4)
    x = 0.4 + 0.5 * rng.random(m.n_cells)
    d = rng.normal(size=m.n_cells)
    g = _grad_and_fd(st, x, d)
    errs = []
    with torch.no_grad():
        for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            jp = float(st.response(_w(torch.as_tensor(x + h * d))).compliance[-1])
            jm = float(st.response(_w(torch.as_tensor(x - h * d))).compliance[-1])
            errs.append(abs((jp - jm) / (2 * h) - g) / abs(g))
    print(f"CO3 FD errors {[f'{e:.2e}' for e in errs]}")
    assert min(errs) <= 1e-6, errs


@pytest.mark.xfail(strict=True, reason="pre-registered CO4 FAILED on the first run: detaching the creep history changes "
                   "the gradient by only 6.29e-4 (< 1e-3) at this load (2e5 N tip load, weak creep): the negative "
                   "control is not sensitive in this set-up; post hoc CO4b at 5x the load")
def test_co4_detached_history_changes_gradient() -> None:
    m, fixed, loads = _cantilever()
    st = CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12)
    rng = np.random.default_rng(4)
    x = 0.4 + 0.5 * rng.random(m.n_cells)
    gs = []
    for det in (False, True):
        xt = torch.as_tensor(x).requires_grad_(True)
        st.response(_w(xt), detach_history=det).compliance[-1].backward()
        gs.append(xt.grad.numpy().copy())
    rel = np.linalg.norm(gs[1] - gs[0]) / np.linalg.norm(gs[0])
    print(f"CO4 detached relative difference {rel:.3e}")
    assert rel >= 1e-3


def test_co4b_detached_history_strong_creep_posthoc() -> None:
    """POST HOC (after CO4 failed): the same control at 5x the load (strong creep); the full gradient is also checked
    against FD there, since the history term is then large."""
    m, fixed, loads = _cantilever(load=1.0e6)
    st = CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12)
    rng = np.random.default_rng(4)
    x = 0.4 + 0.5 * rng.random(m.n_cells)
    d = rng.normal(size=m.n_cells)
    gs = []
    for det in (False, True):
        xt = torch.as_tensor(x).requires_grad_(True)
        st.response(_w(xt), detach_history=det).compliance[-1].backward()
        gs.append(xt.grad.numpy().copy())
    rel = np.linalg.norm(gs[1] - gs[0]) / np.linalg.norm(gs[0])
    g = float(gs[0] @ d)
    with torch.no_grad():
        h = 1e-5
        fd = (float(st.response(_w(torch.as_tensor(x + h * d))).compliance[-1])
              - float(st.response(_w(torch.as_tensor(x - h * d))).compliance[-1])) / (2 * h)
    print(f"CO4b detached relative difference {rel:.3e}; FD error {abs(fd - g) / abs(g):.2e}")
    assert rel >= 1e-3 and abs(fd - g) <= 1e-6 * abs(g)


@pytest.mark.slow
def test_co6_creep_design_beats_elastic_design_at_service_time() -> None:
    """CO6 (docs/foundations/prereg/CO6_creep_topology.md)."""
    from toporlogy.optimization.viscoelastic_topology import optimize_viscoelastic
    nx, ny = 30, 10
    m = rectangle(nx, ny, float(nx), float(ny))
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    corner = int(np.flatnonzero(np.isclose(P[:, 0], nx) & np.isclose(P[:, 1], 0.0))[0])
    fixed = np.concatenate([2 * left, [2 * corner + 1]])
    tl = int(np.flatnonzero(np.isclose(P[:, 0], 0.0) & np.isclose(P[:, 1], ny))[0])
    times = np.array([0.0, 1, 3, 10, 30, 100, 300, 1000.0])
    loads = np.zeros((len(times), m.n_nodes * 2))
    loads[:, 2 * tl + 1] = -5e6
    st = CreepStructure(m, K0, G0, 1e-4, SREF, 3.0, times, loads, fixed)
    kw = dict(volume=0.5, rmin=1.5, max_iter=1500, kkt_tol=1e-4)
    el = optimize_viscoelastic(st, step_index=0, **kw)                    # type: ignore[arg-type]
    cr = optimize_viscoelastic(st, step_index=-1, x0=el.x, **kw)          # type: ignore[arg-type]
    with torch.no_grad():
        def J(xp):
            c = st.response(_w(torch.as_tensor(xp))).compliance
            return float(c[0]), float(c[-1])
        je, jc = J(el.xphys), J(cr.xphys)
    print(f"CO6 elastic design J(0), J(1000 s) = {je}; creep design = {jc}; gain at 1000 s "
          f"{(je[1] - jc[1]) / je[1]:.4%}; iterations {el.iterations} ({el.termination}), {cr.iterations} "
          f"({cr.termination}); kkt {el.kkt_residual}, {cr.kkt_residual}")
    for r in (el, cr):                                                                     # CO6b
        assert (r.converged and r.kkt_residual <= 1e-4) or (not r.converged and r.iterations == 1500)
    assert jc[1] <= je[1]                                                                  # CO6a
    assert jc[0] >= je[0]                                                                  # CO6c

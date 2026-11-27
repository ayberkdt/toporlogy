"""Accumulated creep-strain constraint (docs/foundations/prereg/CC_creep_constraint.md, CC1-CC4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from toporlogy.core.constitutive import TimeStep
from toporlogy.core.creep import NortonCreep
from toporlogy.fem.mesh.history import HistoryProblem
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.creep_topology import CreepStructure, pnorm_creep_strain
from toporlogy.optimization.viscoelastic_topology import optimize_viscoelastic

from test_creep_topology import A, G0, K0, SREF, TIMES, _cantilever, _w
from test_viscoelastic_robust import _mbb


def test_cc1_accumulated_creep_strain_equals_history_fem() -> None:
    m, fixed, loads = _cantilever()
    st = CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12)
    with torch.no_grad():
        p = st.response(torch.ones(m.n_cells, dtype=torch.float64)).p_cr.numpy()
    hp = HistoryProblem(m, NortonCreep(K0, G0, A, SREF, 3.0))
    tp = 0.0
    for k, t in enumerate(TIMES):
        hp.step(TimeStep(tp, t - tp), f_ext=loads[k], prescribed=(fixed, np.zeros(len(fixed))), tol=1e-12)
        tp = t
    ref = np.asarray(hp.store.committed["p_cr"]).reshape(-1)
    err = float(np.abs(p - ref).max() / np.abs(ref).max())
    print(f"CC1 {err:.2e}")
    assert err <= 1e-9


def test_cc2_pnorm_gradient_fd() -> None:
    m, fixed, loads = _cantilever(load=1.0e6)
    st = CreepStructure(m, K0, G0, A, SREF, 3.0, TIMES, loads, fixed, tol=1e-12)
    rng = np.random.default_rng(4)
    x = 0.4 + 0.5 * rng.random(m.n_cells)
    d = rng.normal(size=m.n_cells)
    xt = torch.as_tensor(x).requires_grad_(True)
    pnorm_creep_strain(st.response(_w(xt)), 8.0).backward()
    g = float(xt.grad.numpy() @ d)
    errs = []
    with torch.no_grad():
        for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            jp = float(pnorm_creep_strain(st.response(_w(torch.as_tensor(x + h * d))), 8.0))
            jm = float(pnorm_creep_strain(st.response(_w(torch.as_tensor(x - h * d))), 8.0))
            errs.append(abs((jp - jm) / (2 * h) - g) / abs(g))
    print(f"CC2 FD {[f'{e:.1e}' for e in errs]}")
    assert min(errs) <= 1e-6


def test_cc4_no_extra_constraints_unchanged() -> None:
    st, _ = _mbb()
    r = optimize_viscoelastic(st, 0.5, 1.5, max_iter=5, extra_constraints=[])
    old = np.array(json.loads((Path(__file__).resolve().parent / "baseline" / "ks1_fingerprint_78bedbb.json")
                              .read_text(encoding="utf-8"))["x"])
    assert np.array_equal(r.x, old)
    with pytest.raises(ValueError, match="single objective"):
        optimize_viscoelastic(st, 0.5, 1.5, steps=[0, -1], max_iter=1, extra_constraints=[(lambda res: res.compliance[0], 1.0)])


@pytest.mark.slow
def test_cc3_creep_strain_constraint_respected() -> None:
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
    el = optimize_viscoelastic(st, step_index=0, **kw)
    cr = optimize_viscoelastic(st, step_index=-1, x0=el.x, **kw)

    def stats(xp):
        with torch.no_grad():
            res = st.response(_w(torch.as_tensor(xp)))
            return float(pnorm_creep_strain(res, 8.0)), float(res.p_cr.max()), float(res.compliance[-1])
    pn0, pmax0, J0 = stats(cr.xphys)
    lim = 0.5 * pn0
    cc = optimize_viscoelastic(st, step_index=-1, x0=cr.x, extra_constraints=[(lambda r: pnorm_creep_strain(r, 8.0),
                                                                                  lim)], **kw)
    pn1, pmax1, J1 = stats(cc.xphys)
    print(f"CC3 unconstrained creep design: pnorm {pn0:.4e} max {pmax0:.4e} J {J0:.4e} ({cr.iterations} it, "
          f"{cr.termination}); constrained: pnorm {pn1:.4e} (limit {lim:.4e}) max {pmax1:.4e} J {J1:.4e} "
          f"({cc.iterations} it, {cc.termination}, kkt {cc.kkt_residual}); J price {(J1 - J0) / J0:.3%}")
    assert pn1 <= lim * (1 + 1e-3)
    assert pmax1 < pmax0

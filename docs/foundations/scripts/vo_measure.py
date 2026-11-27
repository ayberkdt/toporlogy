"""Measured values of VO1-VO6 (docs/foundations/prereg/VO_viscoelastic_topology.md) -> docs/foundations/VO_measurements.json.

    python docs/foundations/scripts/vo_measure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "toporlogy")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import test_viscoelastic_topology as T  # noqa: E402
from toporlogy.core.constitutive import TimeStep  # noqa: E402
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries  # noqa: E402
from toporlogy.fem.mesh.history import HistoryProblem  # noqa: E402
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure, optimize_viscoelastic  # noqa: E402

out: dict = {}
mat = T._general()
m, fixed, tip = T._cantilever()
loads = T._schedule(m.n_nodes * 2, tip, T.TIMES, 2.0)
st = ViscoelasticStructure(m, mat, T.TIMES, loads, fixed)
for c in (1.0, 0.37):
    u = st.response(torch.full((m.n_cells,), c, dtype=torch.float64)).u.detach().numpy()
    sc = GeneralizedMaxwell(PronySeries(c * mat.shear.inf, c * mat.shear.amplitudes, mat.shear.taus),
                            PronySeries(c * mat.bulk.inf, c * mat.bulk.amplitudes, mat.bulk.taus))
    hp, tp, err = HistoryProblem(m, sc), 0.0, 0.0
    for k, t in enumerate(T.TIMES):
        r = hp.step(TimeStep(tp, t - tp), f_ext=loads[k], prescribed=(fixed, np.zeros(len(fixed))), tol=1e-13)
        tp, err = t, max(err, float(np.abs(r.u - u[k]).max() / np.abs(r.u).max()))
    out[f"VO1_c{c}"] = err
rng = np.random.default_rng(4)
x = 0.3 + 0.6 * rng.random(m.n_cells)
d = rng.normal(size=m.n_cells)
xt = torch.as_tensor(x).requires_grad_(True)
J = st.response(T._weights(xt)).compliance[-1]
J.backward()
g = float(xt.grad.numpy() @ d)
errs = {}
with torch.no_grad():
    for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        jp = float(st.response(T._weights(torch.as_tensor(x + h * d))).compliance[-1])
        jm = float(st.response(T._weights(torch.as_tensor(x - h * d))).compliance[-1])
        errs[str(h)] = abs((jp - jm) / (2 * h) - g) / abs(g)
out["VO2_fd_errors"] = errs
out["VO3_ratio_proportional"] = T._ratio_test(T._proportional())
out["VO4_ratio_nonproportional"] = T._ratio_test(T._nonprop())
gs = []
for det in (False, True):
    xt = torch.as_tensor(x).requires_grad_(True)
    st.response(T._weights(xt), detach_history=det).compliance[-1].backward()
    gs.append(xt.grad.numpy().copy())
out["VO5_detached_rel_diff"] = float(np.linalg.norm(gs[1] - gs[0]) / np.linalg.norm(gs[0]))
nx, ny = 30, 10
from toporlogy.fem.mesh.mesh import rectangle  # noqa: E402
mb = rectangle(nx, ny, float(nx), float(ny))
P = mb.points
left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
corner = int(np.flatnonzero(np.isclose(P[:, 0], nx) & np.isclose(P[:, 1], 0.0))[0])
fx = np.concatenate([2 * left, [2 * corner + 1]])
tl = int(np.flatnonzero(np.isclose(P[:, 0], 0.0) & np.isclose(P[:, 1], ny))[0])
times = np.concatenate([[0.0], np.logspace(-1, 2, 12)])
ld = np.zeros((len(times), mb.n_nodes * 2))
ld[:, 2 * tl + 1] = -1.0
s2 = ViscoelasticStructure(mb, T._nonprop(), times, ld, fx)
rs = optimize_viscoelastic(s2, 0.5, 1.5, step_index=-1, max_iter=40)
r0 = optimize_viscoelastic(s2, 0.5, 1.5, step_index=0, max_iter=40)
with torch.no_grad():
    def cross(xp):
        return [float(v) for v in s2.response(torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))).compliance[[0, -1]]]
    out["VO6"] = {"mean_abs_dxphys": float(np.abs(rs.xphys - r0.xphys).mean()),
                  "service_design_J0_Jts": cross(rs.xphys), "instant_design_J0_Jts": cross(r0.xphys),
                  "volumes": [rs.volume, r0.volume], "iterations": 40}
(ROOT / "docs" / "foundations" / "VO_measurements.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))

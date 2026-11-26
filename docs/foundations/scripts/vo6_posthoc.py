"""POST HOC study of VO6 (docs/foundations/prereg/VO_viscoelastic_topology.md): 40 vs 120 MMA iterations and a warm
start from the instantaneous design. Writes docs/foundations/VO6_posthoc.json. About 6 min."""
import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "toporlogy")]
import numpy as np
import torch
import test_viscoelastic_topology as T
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure, optimize_viscoelastic
nx, ny = 30, 10
mb = rectangle(nx, ny, float(nx), float(ny)); P = mb.points
left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
corner = int(np.flatnonzero(np.isclose(P[:, 0], nx) & np.isclose(P[:, 1], 0.0))[0])
fx = np.concatenate([2 * left, [2 * corner + 1]])
tl = int(np.flatnonzero(np.isclose(P[:, 0], 0.0) & np.isclose(P[:, 1], ny))[0])
times = np.concatenate([[0.0], np.logspace(-1, 2, 12)])
ld = np.zeros((len(times), mb.n_nodes * 2)); ld[:, 2 * tl + 1] = -1.0
s2 = ViscoelasticStructure(mb, T._nonprop(), times, ld, fx)
def J(xp, i):
    with torch.no_grad():
        return float(s2.response(torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))).compliance[i])
res = {}
for iters in (40, 120):
    rs = optimize_viscoelastic(s2, 0.5, 1.5, step_index=-1, max_iter=iters)
    r0 = optimize_viscoelastic(s2, 0.5, 1.5, step_index=0, max_iter=iters)
    res[iters] = {"service_design_Jts": J(rs.xphys, -1), "instant_design_Jts": J(r0.xphys, -1),
                  "service_design_J0": J(rs.xphys, 0), "instant_design_J0": J(r0.xphys, 0),
                  "dx": float(np.abs(rs.xphys - r0.xphys).mean()), "last_change_s": rs.history["change"][-1],
                  "last_change_0": r0.history["change"][-1]}
    if iters == 40:
        warm = optimize_viscoelastic(s2, 0.5, 1.5, step_index=-1, max_iter=80, x0=r0.x)
        res["warm_from_instant_80"] = {"Jts": J(warm.xphys, -1), "J0": J(warm.xphys, 0), "vol": warm.volume}
print(json.dumps(res, indent=1))
(ROOT / "docs" / "foundations" / "VO6_posthoc.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")

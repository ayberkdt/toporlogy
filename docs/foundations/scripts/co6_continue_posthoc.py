"""POST HOC (after CO6): continue the non-converged creep-design run for up to 1500 more MMA iterations from where it
stopped. Writes docs/foundations/CO6_continue_posthoc.json. About 65 min."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "toporlogy")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from toporlogy.fem.mesh.mesh import rectangle  # noqa: E402
from toporlogy.optimization.creep_topology import CreepStructure  # noqa: E402
from toporlogy.optimization.viscoelastic_topology import optimize_viscoelastic  # noqa: E402

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
st = CreepStructure(m, 3.33e9, 0.77e9, 1e-4, 10e6, 3.0, times, loads, fixed)
kw = dict(volume=0.5, rmin=1.5, max_iter=1500, kkt_tol=1e-4)
el = optimize_viscoelastic(st, step_index=0, **kw)
cr = optimize_viscoelastic(st, step_index=-1, x0=el.x, **kw)
cr2 = optimize_viscoelastic(st, step_index=-1, x0=cr.x, **kw)


def J(xp):
    with torch.no_grad():
        c = st.response(torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))).compliance
    return [float(c[0]), float(c[-1])]


out = {"elastic": {"J": J(el.xphys), "it": el.iterations, "stop": el.termination, "kkt": el.kkt_residual},
       "creep_first_1500": {"J": J(cr.xphys), "it": cr.iterations, "stop": cr.termination, "kkt": cr.kkt_residual},
       "creep_continued": {"J": J(cr2.xphys), "it": cr2.iterations, "stop": cr2.termination, "kkt": cr2.kkt_residual},
       "note": "MMA restarted from the first run's x (asymptotes reset)"}
(ROOT / "docs" / "foundations" / "CO6_continue_posthoc.json").write_text(json.dumps(out, indent=1) + "\n",
                                                                       encoding="utf-8")
print(json.dumps(out, indent=1))

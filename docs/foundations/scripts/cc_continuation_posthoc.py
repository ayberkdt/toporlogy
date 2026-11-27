"""POST HOC (after CC3 failed): continuation in the creep-strain limit (0.9, 0.8, 0.7 x the unconstrained p-norm),
each run warm-started from the previous design, KKT stop 1e-4, at most 600 iterations. Writes
docs/foundations/CC_continuation_posthoc.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "toporlogy")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from toporlogy.fem.mesh.mesh import rectangle  # noqa: E402
from toporlogy.optimization.creep_topology import CreepStructure, pnorm_creep_strain  # noqa: E402
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


def stats(xp):
    with torch.no_grad():
        r = st.response(torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9)))
        return {"pnorm": float(pnorm_creep_strain(r, 8.0)), "max": float(r.p_cr.max()), "J": float(r.compliance[-1])}


kw = dict(volume=0.5, rmin=1.5, kkt_tol=1e-4)
el = optimize_viscoelastic(st, step_index=0, max_iter=1500, **kw)
cr = optimize_viscoelastic(st, step_index=-1, x0=el.x, max_iter=1500, **kw)
base = stats(cr.xphys)
out = {"unconstrained": {**base, "it": cr.iterations, "stop": cr.termination}, "continuation": []}
x = cr.x
for f in (0.9, 0.8, 0.7):
    lim = f * base["pnorm"]
    r = optimize_viscoelastic(st, step_index=-1, x0=x, max_iter=600,
                              extra_constraints=[(lambda res: pnorm_creep_strain(res, 8.0), lim)], **kw)
    s = stats(r.xphys)
    out["continuation"].append({"factor": f, "limit": lim, **s, "feasible": s["pnorm"] <= lim * (1 + 1e-3),
                                "it": r.iterations, "stop": r.termination, "kkt": r.kkt_residual})
    x = r.x
    print(out["continuation"][-1], flush=True)
(ROOT / "docs" / "foundations" / "CC_continuation_posthoc.json").write_text(json.dumps(out, indent=1) + "\n",
                                                                          encoding="utf-8")

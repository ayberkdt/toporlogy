"""Re-run the VM and CO6 optimisations and save their designs for the figure (docs/foundations/designs.npz).

    python docs/foundations/scripts/save_designs.py      # about 50 min (the creep run dominates)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "toporlogy")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from test_viscoelastic_minmax_switch import _problem  # noqa: E402
from toporlogy.fem.mesh.mesh import rectangle  # noqa: E402
from toporlogy.optimization.creep_topology import CreepStructure  # noqa: E402
from toporlogy.optimization.viscoelastic_topology import optimize_viscoelastic  # noqa: E402

out: dict[str, np.ndarray] = {}
st, kA, kB = _problem()
kw = dict(volume=0.4, rmin=0.075, max_iter=1500, kkt_tol=1e-4)
for name, r in {"vm_minmax": optimize_viscoelastic(st, steps=[kA, kB], **kw),
                "vm_A": optimize_viscoelastic(st, step_index=kA, **kw),
                "vm_B": optimize_viscoelastic(st, step_index=kB, **kw)}.items():
    with torch.no_grad():
        c = st.response(torch.as_tensor(1e-9 + r.xphys ** 3 * (1 - 1e-9))).compliance
    out[name] = r.xphys
    out[name + "_J"] = np.array([float(c[kA]), float(c[kB])])
out["vm_centroids"] = st.mesh.cell_centroids()
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
cs = CreepStructure(m, 3.33e9, 0.77e9, 1e-4, 10e6, 3.0, times, loads, fixed)
kw2 = dict(volume=0.5, rmin=1.5, max_iter=1500, kkt_tol=1e-4)
el = optimize_viscoelastic(cs, step_index=0, **kw2)
cr = optimize_viscoelastic(cs, step_index=-1, x0=el.x, **kw2)
cr = optimize_viscoelastic(cs, step_index=-1, x0=cr.x, **kw2) if not cr.converged else cr
for name, r in {"co_elastic": el, "co_creep": cr}.items():
    with torch.no_grad():
        c = cs.response(torch.as_tensor(1e-9 + r.xphys ** 3 * (1 - 1e-9))).compliance
    out[name] = r.xphys
    out[name + "_J"] = np.array([float(c[0]), float(c[-1])])
out["co_centroids"] = m.cell_centroids()
np.savez(ROOT / "docs" / "foundations" / "designs.npz", **out)
print({k: v for k, v in out.items() if k.endswith("_J")})

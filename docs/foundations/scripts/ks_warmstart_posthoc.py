"""POST HOC (after KS2/KS3 failed): warm-start the robust and the min-max objective from the KKT-stopped single-time
design. Writes docs/foundations/KS_warmstart_posthoc.json. About 4 min."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "toporlogy")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from test_viscoelastic_robust import _mbb, _step_of  # noqa: E402
from test_viscoelastic_uncertain import SAMPLES, _set  # noqa: E402
from toporlogy.optimization.viscoelastic_topology import RobustViscoelastic, optimize_viscoelastic  # noqa: E402

st, times = _mbb()
fixed = np.setdiff1d(np.arange(st.ndof), st.free)
sts = _set(st.mesh, times, st.loads, fixed, SAMPLES)


def w(xp):
    return torch.as_tensor(1e-9 + xp ** 3 * (1 - 1e-9))


def robust_value(xp):
    with torch.no_grad():
        J = np.array([float(s.response(w(xp)).compliance[-1]) for s in sts])
    return float(J.mean() + 2 * J.std())


def worst(xp, ks):
    with torch.no_grad():
        c = st.response(w(xp)).compliance
    return max(float(c[k]) for k in ks)


out = {}
nom = optimize_viscoelastic(sts[4], 0.5, 1.5, max_iter=1500, kkt_tol=1e-4)
rob_w = optimize_viscoelastic(RobustViscoelastic(sts, kappa=2.0), 0.5, 1.5, max_iter=1500, kkt_tol=1e-4, x0=nom.x)
out["robust"] = {"nominal_design": robust_value(nom.xphys), "robust_warm_from_nominal": robust_value(rob_w.xphys),
                 "iterations": rob_w.iterations, "termination": rob_w.termination, "kkt": rob_w.kkt_residual}
ks = [0, _step_of(times, 10.0), len(times) - 1]
s100 = optimize_viscoelastic(st, 0.5, 1.5, step_index=ks[-1], max_iter=1500, kkt_tol=1e-4)
mm_w = optimize_viscoelastic(st, 0.5, 1.5, steps=ks, max_iter=1500, kkt_tol=1e-4, x0=s100.x)
out["minmax"] = {"t100_design_worst": worst(s100.xphys, ks), "minmax_warm_from_t100": worst(mm_w.xphys, ks),
                 "iterations": mm_w.iterations, "termination": mm_w.termination, "kkt": mm_w.kkt_residual}
(ROOT / "docs" / "foundations" / "KS_warmstart_posthoc.json").write_text(json.dumps(out, indent=1) + "\n",
                                                                       encoding="utf-8")
print(json.dumps(out, indent=1))

"""Full chain on a small real optimisation result: 3-D cantilever (30×10×10) → finalize → checks → export → T10.

Slow (optimisation + gmsh in a subprocess + T10 solve). Acceptance (assignment): one body, watertight, quality
report written, T10 re-check runs; volume of the finished part within 3 % of the optimised volume (closing adds
fillets) and the T10 compliance within a factor 2 of the voxel model (a sanity bound — the measured ratio is in the
report, not a pass criterion of the method).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from toporlogy import (OC, SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, Study, Volume,
                          standard_chain)
from toporlogy.post import BoxPrimitive, Keep, finalize, manufacturing_checks, reverify


@pytest.mark.slow
@pytest.mark.gmsh
def test_cantilever_full_chain(tmp_path) -> None:
    pytest.importorskip("gmsh")                      # T10 re-analysis below; conftest: skip without gmsh
    nx, ny, nz = 30, 10, 10
    keep = np.zeros((nx, ny, nz), bool)
    keep[0] = True
    keep[-1, :, 0] = True
    d = Domain((nx, ny, nz), h=1.0, keep=keep)
    L, eps = nx * d.h, 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: (np.abs(X[:, 0] - L) < eps) & (X[:, 2] < eps))
    F = d.load_vector(1)
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    chain = standard_chain(d, 1.5, projection=True)
    st = Study(d, Elasticity(d, fixed, nu=0.3), F, chain, SIMP(d), Compliance(),
               [Constraint(Volume(), 0.3, scale=1.0)], OC(dv_scale=d.n_active),
               BetaContinuation(chain, 1.0, 8.0, 15, 0.01))
    opt = st.run(max_iter=40, tol=0.01)
    x = opt.xphys
    pads = [Keep(BoxPrimitive((0, 0, 0), (1, ny, nz)), name="clamp plate"), Keep(BoxPrimitive((L - 1, 0, 0), (L, ny, 1)), name="load pad")]
    res = finalize(d, x, preserve=pads)
    r = res.report
    assert r["topology_mesh"]["bodies"] == 1 and r["topology_grid"]["bodies"] == 1
    assert r["edges"]["open"] == 0 and r["edges"]["non_manifold"] == 0 and r["edges"]["misoriented"] == 0
    assert abs(r["volume_vs_target"]) < 0.03
    ck = manufacturing_checks(res, min_wall=1.0)
    assert ck.items[0]["passed"] and ck.measured["topology"]["bodies"] == 1
    sup = [lambda X: X[:, 0] < 1e-6]
    loads = {"tip": [(lambda X: (X[:, 0] > L - 0.5) & (X[:, 2] < 0.5), (0.0, 0.0, -1.0))]}
    rv = reverify(res, sup, loads, E=1.0, nu=0.3, size=1.0, domain=d, xphys=x, timeout=900,
                  exclusions=[lambda X: X[:, 0] < 1.0, lambda X: X[:, 0] > L - 1.0])
    c = rv["t10"]["cases"][0]
    assert np.isfinite(c["compliance"]) and c["compliance"] > 0
    assert abs(c["applied_force"][2] + 1.0) < 1e-9
    assert 0.5 < rv["change"][0]["compliance_ratio"] < 2.0
    files = res.export(tmp_path, "cantilever", formats=("stl", "ply"), checks=ck, extra={"reverify": rv})
    rep = json.loads(Path(files["report"]).read_text(encoding="utf-8"))
    assert rep["checks"]["items"][0]["passed"] and "pipeline" in rep and "reverify" in rep

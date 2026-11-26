"""Bitwise fingerprint of the path followers without a spectrum (catallanma_onkayit.md, B1).

Run on the base commit before the solver changes (``--write catallanma_parmakizi_oncesi.json``) and on the final code
(``--check catallanma_parmakizi_oncesi.json``): every SHA-256 of the accepted points (λ, x), the per-iteration
residual histories and the rejection count must be equal.

    PYTHONPATH=src python docs/olcumler/catallanma_parmakizi.py --write docs/olcumler/catallanma_parmakizi_oncesi.json
    PYTHONPATH=src python docs/olcumler/catallanma_parmakizi.py --check docs/olcumler/catallanma_parmakizi_oncesi.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests" / "toporlogy"))

from _truss import TwoBarTruss  # noqa: E402

from toporlogy.core.domain import Domain  # noqa: E402
from toporlogy.fem.nonlinear import NonlinearElasticity  # noqa: E402
from toporlogy.solvers.continuation import ArcLengthSettings, arc_length, displacement_control  # noqa: E402


def _digest(path) -> dict[str, str | int]:
    h_pts, h_rec = hashlib.sha256(), hashlib.sha256()
    for p in path.points:
        h_pts.update(np.float64(p.lam).tobytes())
        h_pts.update(np.ascontiguousarray(p.x, np.float64).tobytes())
        h_pts.update(str(p.det_sign).encode())
    for r in path.records:
        h_rec.update(np.asarray(r.residuals, np.float64).tobytes())
        h_rec.update(np.asarray(r.updates, np.float64).tobytes())
        h_rec.update(f"{r.code}|{r.converged}|{r.rolled_back}|{r.notes}".encode())
    return {"points": h_pts.hexdigest(), "records": h_rec.hexdigest(), "n_points": len(path.points),
            "rejected": path.rejected}


def configurations() -> dict[str, dict[str, str | int]]:
    out = {}
    tr = TwoBarTruss()
    st = ArcLengthSettings(ds=0.005, rtol=1e-12, eta_tol=0.0, max_steps=2000)
    out["truss_arc_length"] = _digest(arc_length(tr, np.zeros(2), 0.0, st, lambda x, lam: -x[1] >= 2.5 * tr.h))
    targets = list(-np.linspace(0.0, 2.5 * tr.h, 51)[1:])
    out["truss_displacement_control"] = _digest(displacement_control(tr, np.zeros(2), 0.0, 1, targets, rtol=1e-12,
                                                                     eta_tol=0.0))
    d = Domain((16, 6), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - 16) < 1e-9)
    F = np.zeros(d.ndof)
    F[2 * tip + 1] = -0.06 / len(tip)
    nl = NonlinearElasticity(d, fixed, nu=0.3, material="svk", rtol=1e-12)
    E = np.ones(nl.n_elements)
    prob = nl.continuation_problem(E, F)
    u_end = nl.solve_numpy(E, F)[:, 0]
    ds = 0.08 * float(np.linalg.norm(u_end[nl.free]))
    out["fe_cantilever_arc_length"] = _digest(arc_length(prob, np.zeros(len(nl.free)), 0.0,
                                                         ArcLengthSettings(ds=ds, rtol=1e-12),
                                                         lambda x, lam: lam >= 1.0))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("--write", "--check"):
        print(__doc__)
        return 2
    got = configurations()
    target = Path(argv[2])
    if argv[1] == "--write":
        target.write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(got, indent=2))
        return 0
    ref = json.loads(target.read_text(encoding="utf-8"))
    same = {k: got[k] == ref[k] for k in ref}
    for k, v in same.items():
        print(f"{k}: {'bitwise equal' if v else 'DIFFERENT'} ({got[k]['n_points']} points, {got[k]['rejected']} "
              f"rejected)")
    print(f"{sum(same.values())}/{len(same)} equal")
    return 0 if all(same.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

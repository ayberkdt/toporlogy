"""Measurements of docs/olcumler/gerilme.md (pre-registration: gerilme_onkayit.md).

    python docs/olcumler/gerilme.py            # from the repository root: gerilme.json and the figures
    python docs/olcumler/gerilme.py --figures  # figures only, from gerilme.json (needs datawiz, not in the repo)

The problems are the ones of the tests (``tests/toporlogy/_stress_cases.py``, ``_lbracket_stress.py``), so a number
in the record and the number a test checks come from one implementation. About two minutes.

The L-bracket part of ``gerilme.json`` was written at ``8d1f83e``, before the examples got the load pad and the
one-body protection (gerilme_onkayit.md §7); re-running this script now measures the current examples under the
same keys. Their load-path record is ``gerilme_yukyolu.py`` / ``gerilme_yukyolu.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests" / "toporlogy")]
OUT = HERE / "gerilme.json"
FIELDS = HERE / "gerilme_alanlar.json"                       # the two 0/1 designs and their stress ratios (figure 4)


def measure() -> None:
    import toporlogy  # noqa: F401  (before torch: PARDISO, see the package)
    import numpy as np
    import torch

    import _lbracket_stress as B
    import _stress_cases as S
    import _verification as V
    from toporlogy.optimization.stress_singularity import assess_peak

    t0 = time.perf_counter()
    out: dict[str, Any] = {"environment": {**V.environment(), "torch_threads": torch.get_num_threads()}}

    # ---------------------------------------------------------------- closed forms
    out["tension_max_error"] = {e: max(S.errors(S.tension(e), S.tension_exact, S.SIGMA0).values())
                                for e in ("Q4", "Q4IM", "H8", "H8IM")}
    out["bending_incompatible_max_error"] = {
        e: max(S.errors(S.bending(e, n), S.bending_exact, S.SMAX).values()) for e, n in (("Q4IM", 8), ("H8IM", 4))}
    bending: dict[str, Any] = {}
    for dim, element in ((2, "Q4"), (3, "H8")):
        regions = {"R": S.REGION_R, "R_strict": S.REGION_R_STRICT} if dim == 2 else {"R": S.REGION_R}
        for name, region in {**regions, "all": None}.items():
            rows = [S.errors(S.bending(element, n), S.bending_exact, S.SMAX, region) for n in S.MESHES[dim]]
            errors = {k: [r[k] for r in rows] for k in S.TYPES}
            bending[f"{element}_{name}"] = {"n": list(S.MESHES[dim]), "error": errors,
                                            "order": {k: S.orders(v) for k, v in errors.items()}}
    out["pure_bending"] = bending

    # ---------------------------------------------------------------- peaks under refinement
    h = [1.0 / n for n in S.LEVELS]
    peaks: dict[str, Any] = {}
    for name, (build, lo, hi, at) in S.PEAK_PROBLEMS.items():
        rows = [S.centroid_region(build(n), lo, hi) for n in S.LEVELS]
        a = assess_peak(h, [r.peak for r in rows])
        peaks[name] = {"n": list(S.LEVELS), "peak": [r.peak for r in rows], "mean": [r.mean for r in rows],
                       "location": [list(r.location) for r in rows],
                       "distance_over_h": [float(np.linalg.norm(np.subtract(r.location, at)) / k)
                                           for r, k in zip(rows, h, strict=True)],
                       "slopes": list(a.slopes), "increment_ratio": a.increment_ratio, "status": a.status}
    rows = [S.centroid_region(S.l_corner(n, fillet=0.125), (0.25, 0.25), (0.75, 0.75)) for n in S.LEVELS]
    a = assess_peak(h, [r.peak for r in rows])
    peaks["staircase_fillet_r0125"] = {"n": list(S.LEVELS), "peak": [r.peak for r in rows],
                                       "mean": [r.mean for r in rows], "location": [list(r.location) for r in rows],
                                       "slopes": list(a.slopes), "increment_ratio": a.increment_ratio,
                                       "status": a.status}
    n = [8, 16, 32, 64]
    values = [S.bending_peak(k).peak for k in n]
    peaks["bending_control"] = {"n": n, "peak": values, "status_8_to_32": assess_peak([1 / k for k in n[:3]],
                                                                                      values[:3]).status,
                                "status": assess_peak([1 / k for k in n], values).status}
    out["peaks"] = peaks

    # ---------------------------------------------------------------- L-bracket
    lb: dict[str, Any] = {"sigma_lim": B.SIGMA_LIM, "historical": B.HISTORICAL, "unchanged": {}, "posthoc": {},
                          "refinement": {}}
    fields: dict[str, Any] = {}
    for method in B.METHODS:
        lb["unchanged"][method] = B.attempt_unchanged(method)
        r = B.run(method)
        lb["posthoc"][method] = {k: v for k, v in r.items() if k != "arrays"}
        fields[method] = {"x_binary": r["arrays"]["x_binary"].tolist(),
                          "ratio_binary": np.round(r["arrays"]["ratio_binary"], 5).tolist(),
                          "x_grey": np.round(r["arrays"]["x_grey"], 5).tolist()}
    for which in ("solid", *B.METHODS):
        lb["refinement"][which] = B.refinement(which)
    out["lbracket"] = lb
    out["seconds"] = time.perf_counter() - t0
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    FIELDS.write_text(json.dumps(fields, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name}, {FIELDS.name} in {out['seconds']:.0f} s")


if __name__ == "__main__":
    if "--figures" not in sys.argv:
        measure()
    import gerilme_figur                                      # noqa: E402  (next to this file)

    gerilme_figur.main()

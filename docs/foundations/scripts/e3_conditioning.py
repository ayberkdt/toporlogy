"""E3_conditioning raw data: the accuracy regime table and the R02 family (docs/foundations/prereg/E3_conditioning.md).

    python docs/foundations/scripts/e3_conditioning.py      -> docs/foundations/E3_conditioning.json

Uses the measurement functions of tests/toporlogy/test_elastic_conditioning.py (mpmath reference on the float64 inputs).
All tensors are synthetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tests.toporlogy import test_elastic_conditioning as E  # noqa: E402

levels = {f"{k:.0e}": v for k, v in E.measure_levels().items()}
fam = E.r02_family()
summary = {}
for lo, hi in ((0, 1e15), (1e15, 1e20), (1e20, 1e30), (1e30, 1e99)):
    rows = [r for r in fam if lo <= r["kappa_exact"] < hi]
    summary[f"{lo:.0e}..{hi:.0e}"] = {
        "n": len(rows), "accepted": sum(r["accepted"] for r in rows),
        "float_cholesky_failed": sum(not r["float_cholesky_ok"] for r in rows),
        "stored_float_matrix_not_spd_in_mp": sum(not r["float_matrix_spd"] for r in rows),
        "max_map_error_over_u": max((r["map_error"] for r in rows), default=0.0) / E.U,
        "min_lambda_min_rel_error": min((r["lambda_min_rel_error"] for r in rows), default=None),
        "max_lambda_min_rel_error": max((r["lambda_min_rel_error"] for r in rows), default=None)}
out = {"u": E.U, "c": E.C_FACTOR, "levels": levels, "ec7": E.ec7_counts(), "r02_summary": summary, "r02_rows": fam}
(REPO / "docs" / "foundations" / "E3_conditioning.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps({"levels": levels, "ec7": out["ec7"], "r02_summary": summary}, indent=1))

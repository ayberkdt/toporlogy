"""Load path of the two L-bracket stress examples after the load pad and the one-body protection
(gerilme_onkayit.md §7, Y1-Y5). Writes gerilme_yukyolu.json; run from the repository root (about one minute):

    python docs/olcumler/gerilme_yukyolu.py

The numbers come from the helpers the slow tests use (``tests/toporlogy/_lbracket_stress.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests" / "toporlogy")]

import _lbracket_stress as B  # noqa: E402

out = {}
for m in B.METHODS:
    r = B.run(m)
    c0 = B.solid_compliance(m)
    old = B.gate(m, B.old_design(m))
    new = B.gate(m, r["arrays"]["x_grey"])
    row = {"iterations": r["iterations"], "state_solves": r["state_solves"], "termination": r["termination"],
           "feasible_flag": r["feasible_flag"], "objective": r["objective"],
           "optimizer_constraint_values": r["optimizer_constraint_values"], "compliance_solid": c0,
           "Y2_gate": new["verdict"], "Y2_reasons": new.get("reasons"),
           "Y5_old_design_gate": old["verdict"], "Y5_reasons": old.get("reasons")}
    for v in ("grey", "binary"):
        g = r[v]
        row[v] = {"volume": g["volume"], "raw_solid_max": g["raw_solid_max"]["ratio"],
                  "relaxed_qp_max": g["relaxed_qp_max"]["ratio"], "load_path": g["load_path"],
                  "compliance_over_solid": g["load_path"]["compliance"] / c0}
    out[m] = row
with open(REPO / "docs" / "olcumler" / "gerilme_yukyolu.json", "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

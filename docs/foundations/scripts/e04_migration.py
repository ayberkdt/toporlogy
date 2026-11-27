"""E04 migration: the historical auxetic cell (docs/olcumler/auxetic_hucre.json) under both Poisson conventions.

    python docs/foundations/scripts/e04_migration.py      -> docs/foundations/E04_poisson_migration.json

Reads only the stored homogenised C of the historical run (no re-optimisation); the historical json is not changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
src = json.loads((REPO / "docs" / "olcumler" / "auxetic_hucre.json").read_text(encoding="utf-8"))
out = {"source": "docs/olcumler/auxetic_hucre.json", "convention_note": "nu_ij = -eps_j/eps_i under uniaxial "
       "stress along i = -S_ji/S_ii; legacy = C12/C11", "designs": {}}
for key in ("grey", "binary"):
    C = np.array(src[key]["C"], float)
    S = np.linalg.solve(C, np.eye(3))
    out["designs"][key] = {
        "legacy_stored": src[key]["nu"], "legacy_recomputed_C12_over_C11": C[0, 1] / C[0, 0],
        "nu12_uniaxial": -S[1, 0] / S[0, 0], "nu21_uniaxial": -S[0, 1] / S[1, 1],
        "C11": C[0, 0], "C22": C[1, 1], "coupling_max_abs_C16_C26_over_C11": max(abs(C[0, 2]), abs(C[1, 2])) / C[0, 0],
        "min_eig_C": float(np.linalg.eigvalsh(C).min())}
(REPO / "docs" / "foundations" / "E04_poisson_migration.json").write_text(json.dumps(out, indent=1) + "\n",
                                                                           encoding="utf-8")
print(json.dumps(out["designs"], indent=1))

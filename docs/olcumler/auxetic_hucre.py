"""The negative Poisson ratio cell of auxetic_hucre_onkayit.md (A1-A4): writes auxetic_hucre.json. About 20 s.

    python docs/olcumler/auxetic_hucre.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests" / "toporlogy")]

import numpy as np  # noqa: E402

import _auxetic as A  # noqa: E402

r = A.run()
b = r["binary"]
out = {"problem": {"n": A.N, "nu0": A.NU, "penal": A.P, "emin": A.EMIN, "rmin": A.RMIN, "volume": A.VOL,
                   "C11_min": A.C11_MIN, "max_iter": A.MAX_ITER},
       "iterations": r["iterations"], "grey": {"nu": r["nu"], "volume": r["vol"], "C": r["C"].tolist(),
                                               "m_nd": float(np.mean(4 * r["xphys"] * (1 - r["xphys"])))},
       "binary": {"nu": r["nu_binary"], "volume": r["vol_binary"], "C": r["C_binary"].tolist(),
                  "nu21": float(r["C_binary"][0, 1] / r["C_binary"][1, 1]),
                  "min_eig": float(np.linalg.eigvalsh(r["C_binary"]).min()),
                  "periodic_one_body": A.periodic_one_body(b),
                  "rows": ["".join("#" if v else "." for v in row) for row in b.T[::-1]]},
       "history_every_10": r["history"][::10]}
(REPO / "docs" / "olcumler" / "auxetic_hucre.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

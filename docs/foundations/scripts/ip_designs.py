"""Re-run the IP3/IP4 settings (docs/foundations/prereg/IP_projection.md) and store the projected designs for the
figure: docs/foundations/data/ip_cells_x_phys.json and docs/foundations/IP_designs.json (values, no fields).

    python docs/foundations/scripts/ip_designs.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.cell_topology import optimize_cell

HERE = Path(__file__).resolve().parents[1]
NU = 0.3
K, MU = 1 / (2 * (1 - NU)), 1 / (2 * (1 + NU))
HS = 0.5 * K * MU / (0.5 * K + MU)


def main() -> None:
    m0 = rectangle(40, 40, 1.0, 1.0, "T3")
    A = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2]])
    cases = {"square_void_hole": (m0, np.eye(2), np.array([0.5, 0.5]), 1.0),
             "rhombus_void_hole": (replace(m0, points=m0.points @ A.T), A.T.copy(),
                                   np.array([0.75, np.sqrt(3.0) / 4]), np.sqrt(3.0) / 2)}
    xs, vals = {}, {}
    for key, (m, L, centre, area) in cases.items():
        x0 = np.where(np.linalg.norm(m.cell_centroids() - centre, axis=1) <= np.sqrt(0.5 * area / np.pi), 0.0, 1.0)
        r = optimize_cell(m, L, D_plane_stress(NU), 0.5, 0.06, penal=3.5, emin=1e-9, move=0.2, x0=x0,
                          kkt_tol=1e-4, projection=(0.5, (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)), stage_iter=100)
        xs[key] = r.x_phys.tolist()
        vals[key] = {"kappa": r.objective, "ratio_hs": r.objective / HS, "volume": r.volume,
                     "grey": float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9))), "iterations": r.iterations,
                     "kkt": r.kkt, "C": r.C.tolist()}
        print(key, vals[key]["ratio_hs"], vals[key]["grey"])
    (HERE / "data" / "ip_cells_x_phys.json").write_text(json.dumps(xs), encoding="utf-8")
    (HERE / "IP_designs.json").write_text(json.dumps(vals, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

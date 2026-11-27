"""Re-run the IA2 setting (docs/foundations/prereg/IA_auxetic_cell.md) and store the design and its full C and S:
docs/foundations/data/ia_cell_x_phys.json, docs/foundations/IA_design.json.

    python docs/foundations/scripts/ia_design.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.homogenization import homogenize_mesh_elasticity
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.cell_topology import optimize_cell

HERE = Path(__file__).resolve().parents[1]


def physical(C: np.ndarray) -> dict[str, float]:
    """Lateral strain ratios from the full compliance (shear coupling included)."""
    S = np.linalg.inv(C)
    return {"nu12_stress_x": float(-S[1, 0] / S[0, 0]), "nu21_stress_y": float(-S[0, 1] / S[1, 1]),
            "C12_over_C11": float(C[0, 1] / C[0, 0]), "C12_over_C22": float(C[0, 1] / C[1, 1])}


def main() -> None:
    m = rectangle(40, 40, 1.0, 1.0, "T3")
    c = m.cell_centroids()
    x0 = 0.4 + 0.2 * np.cos(2 * np.pi * c[:, 0]) * np.cos(2 * np.pi * c[:, 1])
    D = D_plane_stress(0.3)
    r = optimize_cell(m, np.eye(2), D, 0.4, 0.06, objective="poisson", penal=3.5, emin=1e-9, move=0.2, x0=x0,
                      kkt_tol=1e-4, projection=(0.5, (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)), stage_iter=100,
                      min_C11=0.02)
    solid = r.x_phys >= 0.5
    Cb = homogenize_mesh_elasticity(m, D, np.eye(2), np.where(solid, 1.0, 1e-9)).C
    out = {"C": r.C.tolist(), "C_thresholded": Cb.tolist(), "design": physical(r.C), "thresholded": physical(Cb),
           "volume": r.volume, "grey": float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9))), "iterations": r.iterations}
    (HERE / "data" / "ia_cell_x_phys.json").write_text(json.dumps({"square_auxetic": r.x_phys.tolist()}),
                                                       encoding="utf-8")
    (HERE / "IA_design.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("C", "C_thresholded")}, indent=1))
    print("C", np.array2string(r.C, precision=6))


if __name__ == "__main__":
    main()

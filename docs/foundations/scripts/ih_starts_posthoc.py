"""Post hoc (not preregistered; written after IH3(c)/IH4(c) failed): does the start decide the IH bulk-modulus result?

    python docs/foundations/scripts/ih_starts_posthoc.py <variant> <out.json>

Variants: ``square_void_hole`` / ``rhombus_void_hole`` (x0 = 1 outside and 0 inside a central circle of area 0.5,
the solid fraction exactly at the limit up to the mesh), ``square_random`` (x0 uniform in [0, 1], seed 0, scaled to
mean 0.5). Same settings as IH3/IH4 otherwise (40 x 40 T3, rmin 0.06, penal 3.5, emin 1e-9, move 0.2, 300 iterations,
kkt_tol 1e-4). Writes kappa, kappa/HS, volume, iterations, stop, KKT, grey fraction, the start's own kappa and the
designs (x_phys) as lists.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace

import numpy as np

from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.mesh.mesh import rectangle
from toporlogy.optimization.cell_topology import PeriodicDensityFilter, cell_objective, optimize_cell

NU, PENAL, EMIN, RMIN = 0.3, 3.5, 1e-9, 0.06
K, MU = 1 / (2 * (1 - NU)), 1 / (2 * (1 + NU))
HS = 0.5 * K * MU / (0.5 * K + MU)


def main(variant: str, out: str) -> None:
    m0 = rectangle(40, 40, 1.0, 1.0, "T3")
    if variant.startswith("rhombus"):
        A = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2]])
        m, L, centre = replace(m0, points=m0.points @ A.T), A.T.copy(), np.array([0.75, np.sqrt(3.0) / 4])
        area = np.sqrt(3.0) / 2
    else:
        m, L, centre, area = m0, np.eye(2), np.array([0.5, 0.5]), 1.0
    c = m.cell_centroids()
    if variant.endswith("void_hole"):
        r = np.sqrt(0.5 * area / np.pi)
        x0 = np.where(np.linalg.norm(c - centre, axis=1) <= r, 0.0, 1.0)
    elif variant == "square_random":
        x0 = np.random.default_rng(0).uniform(size=m.n_cells)
        x0 = np.clip(x0 * 0.5 / x0.mean(), 0.0, 1.0)
    else:
        raise SystemExit(f"unknown variant {variant}")
    D = D_plane_stress(NU)
    filt = PeriodicDensityFilter(m, L, RMIN)
    k_start = cell_objective(m, L, D, filt(x0), PENAL, EMIN)[0]
    t0 = time.perf_counter()
    r = optimize_cell(m, L, D, 0.5, RMIN, penal=PENAL, emin=EMIN, max_iter=300, move=0.2, x0=x0, kkt_tol=1e-4)
    grey = float(np.mean((r.x_phys > 0.1) & (r.x_phys < 0.9)))
    rec = {"variant": variant, "kappa": r.objective, "ratio_hs": r.objective / HS, "kappa_start": k_start,
           "ratio_start": k_start / HS, "volume": r.volume, "iterations": r.iterations, "stop": r.stop, "kkt": r.kkt,
           "grey": grey, "C": r.C.tolist(), "seconds": time.perf_counter() - t0,
           "points": m.points.tolist(), "cells": m.cells.tolist(), "x_phys": r.x_phys.tolist()}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    print({k: v for k, v in rec.items() if k not in ("points", "cells", "x_phys")})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Measurements for docs/yol_haritasi_2026-11-18.md §2.3 / PR-10: symmetry of the Ideathon bracket run.

    PYTHONPATH=src python docs/olcumler/simetri.py [ROOT]

ROOT holds ``studies/final_2026/ansys/`` (git-ignored, default: the current directory). The study is rebuilt from
its YAML (domain, keep with the connectivity ties); x̄ is read from the stored run (``run/result.npz``), the final
surface from ``run/part.stl``. The field numbers go through the ``symmetry_check`` post step, i.e. the same code
that writes ``summary.json``. Writes ``simetri.json`` next to this file.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.resolve import read_config
from toporlogy.geometry import read_stl
from toporlogy.post.mesh import mirror_distance

OUT = Path(__file__).resolve().with_name("simetri.json")
os.chdir(sys.argv[1] if len(sys.argv) > 1 else ".")
STUDY = Path("studies/final_2026/ansys/toporlogy_2026-11-18")
CAD = Path("studies/final_2026/ansys/official_2026-11-05/visuals/initial_exact.stl")

Vc, Fc = read_stl(CAD)
y_mid = float(0.5 * (Vc[:, 1].min() + Vc[:, 1].max()))           # "parça ortası" of §2.3
planes = [["x", 500.0], ["y", round(y_mid, 2)]]
raw = read_config(STUDY / "official_bracket.yaml")
raw["post"] = [{"type": "symmetry_check", "planes": planes}]
t = time.perf_counter()
setup = build_setup(raw)
t_build = time.perf_counter() - t
xphys = np.load(STUDY / "run" / "result.npz")["xphys"]
step = dict(setup.post)["symmetry_check"]          # connectivity.gate adds its own step
field = step(SimpleNamespace(xphys=xphys), setup.study, None)
Vp, Fp = read_stl(STUDY / "run" / "part.stl")
t = time.perf_counter()
surface = {"part.stl": [mirror_distance(Vp, Fp, a, c) for a, c in planes],
           "initial_exact.stl (CAD)": [mirror_distance(Vc, Fc, a, c) for a, c in planes]}
t_mesh = time.perf_counter() - t

ref = {"x": {"frac": 0.012, "xphys": 0.478, "keep": 0.224, "keep_mismatch_cells": 228, "grid_shift_max": 0.0},
       "y": {"frac": 0.117, "xphys": 0.474, "grid_shift_max": 3.62}}
print(f"{setup.study.domain}  (build {t_build:.1f} s, surface distances {t_mesh:.1f} s)")
print(f"y mid of the CAD bounding box: {y_mid:.4f}")
for row in field["planes"]:
    f = row["fields"]
    r = ref[row["axis"]]
    print(f"{row['axis']} = {row['coordinate']:g}: shift {row['grid_shift_max']:.3f} (§2.3 {r['grid_shift_max']}), "
          f"outside box {row['cells_outside_box']}")
    for k in ("frac", "keep", "xphys"):
        print(f"   e_sym({k:5s}) = {f[k]['e_sym']:.6f}  (§2.3: {r.get(k, '-')}), mismatch cells {f[k]['mismatch_cells']}")
for what, rows in surface.items():
    for (a, c), m in zip(planes, rows):
        print(f"{what:24s} {a} = {c:g}: d_H {m['hausdorff']:.3f}, mean {m['mean']:.3f}, p95 {m['p95']:.3f} "
              f"(bbox diagonal {m['bbox_diagonal']:.1f}, {m['samples']} samples)")
OUT.write_text(json.dumps({
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "system": platform.system(), "machine": platform.machine()},
    "domain": str(setup.study.domain), "y_mid_cad": y_mid, "reference_2_3": ref, "symmetry_check": field,
    "surface": surface, "seconds": {"build": t_build, "surface": t_mesh}}, indent=1), encoding="utf-8")
print(f"-> {OUT}")

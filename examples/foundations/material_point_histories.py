"""Physical material models at a material point (Foundations II): one stress-controlled creep test per model.

    python examples/foundations/material_point_histories.py

Synthetic parameters only (no real material). Each model gets the same uniaxial history: the axial stress ramps to
its target in 1 s and is held to 1000 s, the lateral stresses stay zero; physical time is passed explicitly.
"""
from __future__ import annotations

import numpy as np

from toporlogy.core.constitutive import drive_material_point
from toporlogy.core.creep import NortonCreep
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries
from toporlogy.core.viscoplastic import Perzyna, PhysicalJ2

MPA = 1e6
K, G = 3.33e9, 0.77e9                                   # Pa (synthetic, about E = 2.15 GPa, nu = 0.395)
times = np.concatenate([np.linspace(0.25, 1.0, 4), 1.0 + np.logspace(-1, 3, 13)[1:]])
models = {
    "generalized Maxwell (1 kPa)": (GeneralizedMaxwell(PronySeries(1 * MPA, [2 * MPA, 3 * MPA], [1.0, 100.0]),
                                                       PronySeries(5 * MPA, [4 * MPA], [10.0])), 1e3),
    "Norton creep (15 MPa)": (NortonCreep(K, G, A=1e-6, sigma_ref=10 * MPA, n=3.0), 15 * MPA),
    "Perzyna (25 MPa)": (Perzyna(K, G, 20 * MPA, 100 * MPA, tau0=10.0, sigma_ref=10 * MPA, m=1.0), 25 * MPA),
    "physical J2 (25 MPa)": (PhysicalJ2(K, G, 20 * MPA, 100 * MPA), 25 * MPA),
}
print(f"{'model':30s} {'eps_xx(1 s)':>12s} {'eps_xx(1001 s)':>15s} {'eps_yy/eps_xx at end':>22s}")
for name, (model, s) in models.items():
    stress = np.zeros((len(times), 6))
    stress[:, 0] = s * np.minimum(times, 1.0)
    h = drive_material_point(model, np.zeros((len(times), 6)), stress, np.zeros(6, bool), times=times,
                             tol=1e-10 * s)
    e = h.strain
    print(f"{name:30s} {e[3, 0]:12.4e} {e[-1, 0]:15.4e} {e[-1, 1] / e[-1, 0]:22.4f}")

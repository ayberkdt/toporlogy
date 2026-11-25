"""Combinations of the parallel physics branches, checked at their merge (2026-11-25): several phases (design.phases)
on an orthotropic material, uniform and per element (core.material angle_field).

Pre-registered before the first run (2026-11-25 19:14 +03:00, coordinator, written into this docstring first;
first run: M1 0 and 0, M2 2.5e-7, M3 0, all PASS):
M1  phases + orthotropic: a per-element ``angle_field`` of 25 degrees everywhere gives the same compliance as
    ``angle: 25`` (relative ≤ 1e-14) and the same gradient (≤ 1e-13; the batched per-element product rounds
    differently from the single matrix, as A's P6 measured 1.3e-14);
M2  phases + a varying angle field: the compliance gradient against central finite differences, relative ≤ 1e-6
    on 6 random components (the tolerance of B's V4);
M3  two phases with the same E on an orthotropic material = one orthotropic material (SIMP) when the second
    phase's field is 0: relative ≤ 1e-14 (B's V2a, here through the YAML and with a rotated D).
"""
from __future__ import annotations

import numpy as np

from toporlogy.configuration.build import study_from_dict

ORTHO = {"E1": 40.0, "E2": 9.0, "E3": 9.0, "G12": 4.0, "G13": 4.0, "G23": 3.2, "nu12": 0.28, "nu13": 0.28,
         "nu23": 0.4}
SHAPE = (12, 4)


def _cfg(orient: dict, phases: list | None) -> dict:
    mat: dict = {"E0": 40.0, "nu": 0.3, "orthotropic": {**ORTHO, **orient}}
    cfg: dict = {"domain": {"type": "box", "shape": list(SHAPE), "h": 1.0}, "material": mat,
                 "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
                 "load_cases": [{"name": "tip", "loads": [{"type": "force",
                                                           "where": {"type": "point", "at": ["max", "min"]},
                                                           "force": [0, -1]}]}],
                 "objective": "compliance", "optimizer": {"type": "mma", "move": 0.1}, "run": {"max_iter": 2}}
    if phases is None:
        cfg["constraints"] = [{"response": "volume", "upper": 0.5}]
    else:
        mat["phases"] = phases
        cfg["interpolation"] = {"type": "multiphase_simp"}
        cfg["constraints"] = [{"response": {"type": "phase_volume"}, "upper": 0.5}]
    return cfg


PHASES = [{"name": "soft", "E": 10.0}, {"name": "stiff", "E": 40.0}]


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def test_m1_phases_uniform_angle_field_equals_angle(tmp_path):
    f = tmp_path / "angle.npy"
    np.save(f, np.full(SHAPE, 25.0))
    a = study_from_dict(_cfg({"angle": 25.0}, PHASES))
    b = study_from_dict(_cfg({"angle_field": str(f)}, PHASES))
    assert b.physics.per_element
    x = np.random.default_rng(1).uniform(0.2, 0.9, a.default_start().shape)
    ea, eb = a.evaluate(x), b.evaluate(x)
    print(f"\n  M1: f0 {_rel(eb.f0, ea.f0):.1e}, df0 {_rel(eb.df0, ea.df0):.1e}")
    assert _rel(eb.f0, ea.f0) <= 1e-14
    assert _rel(eb.df0, ea.df0) <= 1e-13


def test_m2_phases_varying_angle_gradient(tmp_path):
    f = tmp_path / "angle.npy"
    ix, iy = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    np.save(f, 10.0 * ix - 15.0 * iy)
    st = study_from_dict(_cfg({"angle_field": str(f)}, PHASES))
    rng = np.random.default_rng(2)
    x = rng.uniform(0.2, 0.9, st.default_start().shape)
    ev = st.evaluate(x)
    errs = []
    for i in rng.choice(x.size, 6, replace=False):
        h = 1e-5
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (st.evaluate(xp).f0 - st.evaluate(xm).f0) / (2 * h)
        errs.append(abs(fd - ev.df0[i]) / max(abs(fd), 1e-12))
    print(f"\n  M2: largest relative FD error {max(errs):.1e}")
    assert max(errs) <= 1e-6


def test_m3_two_equal_phases_equal_one_orthotropic_material():
    one = study_from_dict(_cfg({"angle": 30.0}, None))
    two = study_from_dict(_cfg({"angle": 30.0}, [{"name": "a", "E": 40.0}, {"name": "b", "E": 40.0}]))
    n = one.default_start().size
    x1 = np.random.default_rng(3).uniform(0.2, 0.9, n)
    x2 = two.default_start().copy()
    assert x2.size == 2 * n                        # design vector field-major: [t_1, t_2]
    x2[:n], x2[n:] = x1, 0.0
    e1, e2 = one.evaluate(x1), two.evaluate(x2)
    print(f"\n  M3: compliance {e1.f0:.12g} / {e2.f0:.12g}, rel {_rel(e2.f0, e1.f0):.1e}")
    assert _rel(e2.f0, e1.f0) <= 1e-14

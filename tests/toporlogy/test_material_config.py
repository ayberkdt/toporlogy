"""The material section of the YAML schema (plane, orthotropic, strength) and the components that read it: physics
elasticity, stress (criterion, default limit), buckling (preload_case), interpolation hashin_shtrikman.

Every case builds a study from a mapping and checks what was built against the library objects constructed
directly; the error cases check that bad input is refused with a message naming the key."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from toporlogy.configuration.build import study_from_dict
from toporlogy.configuration.registry import ConfigError
from toporlogy.core.material import Orthotropic, rotation_about_z, unit_elasticity
from toporlogy.design.interp import HashinShtrikman
from toporlogy.fem.elements import D_plane_strain, D_plane_stress
from toporlogy.optimization.stress import FailureIndex, VonMises

ORTHO = {"E1": 40.0, "E2": 9.0, "E3": 9.0, "G12": 4.0, "G13": 4.0, "G23": 3.2, "nu12": 0.28, "nu13": 0.28,
         "nu23": 0.4, "angle": 30.0}


def _base(**material) -> dict:
    return {
        "domain": {"type": "box", "shape": [12, 4], "h": 1.0},
        "material": {"E0": 1.0, "nu": 0.3, **material},
        "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
        "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                                  "force": [0, -1]}]}],
        "objective": "compliance",
        "constraints": [{"response": "volume", "upper": 0.5}],
        "optimizer": {"type": "mma", "move": 0.1},
        "run": {"max_iter": 2},
    }


def _stress(cfg: dict, **opts) -> dict:
    c = copy.deepcopy(cfg)
    c["constraints"].append({"response": {"type": "stress", **opts}})
    return c


def test_defaults_build_the_original_objects():
    st = study_from_dict(_stress(_base(), sigma_lim=2.0))
    assert np.array_equal(st.physics.D, D_plane_stress(0.3)) and st.physics.nu == 0.3
    field = st.constraints[1].response.field
    assert type(field) is VonMises and field.sigma_lim == 2.0
    assert st.config["material"]["plane"] == "stress" and st.config["material"]["orthotropic"] is None


def test_strength_gives_the_default_limit():
    st = study_from_dict(_stress(_base(strength={"tension": 300.0, "safety_factor": 1.5})))
    assert type(st.constraints[1].response.field) is VonMises
    assert st.constraints[1].response.field.sigma_lim == 200.0
    st = study_from_dict(_stress(_base(strength={"tension": 300.0, "compression": 500.0}), criterion="raghava"))
    f = st.constraints[1].response.field
    assert type(f) is FailureIndex and f.criterion.kind == "raghava" and f.criterion.sc == 500.0


def test_plane_strain():
    st = study_from_dict(_stress(_base(plane="strain"), sigma_lim=2.0))
    assert np.allclose(st.physics.D, D_plane_strain(0.3), rtol=1e-14)
    f = st.constraints[1].response.field
    assert type(f) is FailureIndex and f.szz_row is not None and f.criterion.kind == "von_mises"
    st.evaluate(st.default_start())


def test_orthotropic_with_tsai_wu():
    cfg = _base(orthotropic=ORTHO, strength={"Xt": 1500.0, "Xc": 1200.0, "Yt": 50.0, "Yc": 200.0, "S12": 70.0})
    cfg["material"]["E0"] = 40.0
    st = study_from_dict(_stress(cfg))
    mat = Orthotropic(*(ORTHO[k] for k in ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23")))
    assert np.allclose(st.physics.D, unit_elasticity(mat, 2, "stress", rotation_about_z(30.0)), rtol=1e-14)
    f = st.constraints[1].response.field
    assert f.criterion.kind == "tsai_wu" and np.allclose(f.criterion.R, rotation_about_z(30.0))
    ev = st.evaluate(st.default_start())
    assert np.all(np.isfinite(ev.dg))


def test_buckling_preload_and_hashin_shtrikman():
    cfg = _base()
    cfg["material"]["E0"] = 100.0                      # E0 = 1: the preload alone would buckle (checked: refused)
    cfg["domain"]["shape"] = [6, 18]
    cfg["supports"] = [{"type": "fixed", "where": {"type": "plane", "axis": "y", "at": "min"}}]
    top = {"type": "plane", "axis": "y", "at": "max"}
    cfg["load_cases"] = [{"name": "live", "loads": [{"type": "force", "where": top, "force": [0.02, -1.0]}]},
                         {"name": "dead", "loads": [{"type": "force", "where": top, "force": [0.0, -0.3]}]}]
    cfg["constraints"].append({"response": {"type": "buckling", "load_factor": 0.01, "n_modes": 4,
                                            "preload_case": 1}})
    st = study_from_dict(cfg)
    assert st.constraints[1].response.preload_case == 1
    st.evaluate(st.default_start())
    cfg2 = _base()
    cfg2["interpolation"] = {"type": "hashin_shtrikman"}
    assert isinstance(study_from_dict(cfg2).interp, HashinShtrikman)


@pytest.mark.parametrize("material,stress,match", [
    ({"orthotropic": ORTHO}, None, "E0 must equal"),
    ({"plane": "membrane"}, None, "plane"),
    ({"strength": {"Xt": 1, "Xc": 1, "Yt": 1, "Yc": 1, "S12": 1}}, None, "orthotropic materials take"),
    ({"strength": {"tension": 1.0, "bogus": 2.0}}, None, "strength"),
    ({"strength": {"tension": 3.0}}, {"sigma_lim": 2.0}, "not both"),
    ({}, {"criterion": "drucker_prager", "sigma_lim": 2.0}, "needs material.strength"),
    ({"strength": {"tension": 3.0}}, {"criterion": "tsai_wu"}, "orthotropic"),
])
def test_bad_material_input_is_refused(material, stress, match):
    cfg = _base(**material)
    if stress is not None:
        cfg = _stress(cfg, **stress)
    with pytest.raises((ConfigError, ValueError), match=match):
        study_from_dict(cfg)


def test_plane_strain_is_refused_in_3d():
    cfg = _base(plane="strain")
    cfg["domain"]["shape"] = [6, 3, 2]
    cfg["load_cases"][0]["loads"][0]["force"] = [0, -1, 0]
    cfg["load_cases"][0]["loads"][0]["where"]["at"] = ["max", "min", "min"]
    with pytest.raises((ConfigError, ValueError), match="2-D domains only"):
        study_from_dict(cfg)

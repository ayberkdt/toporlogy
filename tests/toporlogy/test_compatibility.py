"""Unsupported combinations of the compatibility matrix fail closed (08 §3, ``toporlogy.configuration.catalog``).

Every ``UNSUPPORTED`` cell of ``RESPONSE_MATRIX`` and every ``UNSUPPORTED`` rule of ``RULES`` that the build is said to
enforce (``enforced`` = component or catalog) has a probe here: the build must raise with a message matching the
probe's pattern. A meta-test checks that no such cell or rule is left without a probe.

POST HOC (label of docs/vv_katalog.md §5): an exploratory probe run on 2026-11-26 (00:2x +0300, scratch script, not
pre-registered) built 44 combinations before these tests were written; it found two combinations that failed late
with unrelated errors (modal + a state response: ``AttributeError: 'Modal' object has no attribute 'solve'``;
buckling with oblique supports: ``ValueError: wrong M dimensions``) and the build now refuses them
(:func:`toporlogy.configuration.catalog.check_build`). The patterns below were written after that run.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.catalog import RESPONSE_MATRIX, RULES, UNSUPPORTED, check_build
from toporlogy.configuration.registry import ConfigError

B2: dict[str, Any] = {
    "domain": {"type": "box", "shape": [12, 6], "h": 1.0},
    "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
    "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]}, "force": [0, -1]}]}],
    "chain": [{"type": "density_filter", "rmin": 1.5}],
    "constraints": [{"response": "volume", "upper": 0.5}],
    "optimizer": "mma",
}
B3 = copy.deepcopy(B2)
B3["domain"] = {"type": "box", "shape": [8, 4, 4], "h": 1.0}
B3["load_cases"][0]["loads"][0] = {"type": "force", "where": {"type": "point", "at": ["max", "min", "min"]},
                                   "force": [0, -1, 0]}
ORTHO = {"E1": 1.0, "E2": 0.5, "E3": 0.5, "G12": 0.3, "G13": 0.3, "G23": 0.2, "nu12": 0.3, "nu13": 0.3, "nu23": 0.3,
         "angle": 30.0}
PHASES = [{"name": "a", "E": 0.5, "density": 0.5, "cost": 1.0}, {"name": "b", "E": 1.0, "density": 1.0, "cost": 2.0}]
VOL = {"response": "volume", "upper": 0.5}
BUCK = {"response": {"type": "buckling", "load_factor": 0.02}}


def mod(base: dict[str, Any], **kw: Any) -> dict[str, Any]:
    c = copy.deepcopy(base)
    c.update(copy.deepcopy(kw))
    return c


def two_case_supports(base: dict[str, Any]) -> list[dict[str, Any]]:
    """Two load cases, the second with its own supports (per-case supports)."""
    case = base["load_cases"][0]
    own = [{"type": "fixed", "where": {"type": "plane", "axis": "y", "at": "min"}}]
    return [copy.deepcopy(case), dict(copy.deepcopy(case), supports=own)]


def physics_for(p: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(base config, physics spec) able to build physics ``p`` with a volume constraint."""
    return (B3 if p == "elasticity_gmg" else B2), {"type": p}


STATE = {  # response spec (objective) for a state response
    "compliance": "compliance",
    "robust_compliance": "robust_compliance",
    "displacement": {"type": "displacement", "where": {"type": "point", "at": ["max", "min"]}, "direction": "y"},
    "stress": {"type": "stress", "sigma_lim": 1.0},
    "stress_augmented_lagrangian": {"type": "stress_augmented_lagrangian", "objective": "volume",
                                    "sigma_lim": 1.0},
    "end_compliance": "end_compliance",
}
EIGEN = {"frequency": {"type": "maximize", "response": {"type": "frequency"}},
         "eigenvalue": {"type": "maximize", "response": {"type": "eigenvalue"}},
         "min_eigenvalue_ks": {"type": "maximize", "response": {"type": "min_eigenvalue_ks"}},
         "min_frequency": None}           # constraint only (see below)


def cell_probe(response: str, physics: str) -> dict[str, Any]:
    base, ph = physics_for(physics)
    if response == "buckling":
        obj = "end_compliance" if physics == "nonlinear_elasticity" else "volume" if physics == "modal" else \
            "compliance"
        cons = [BUCK] if physics == "modal" else [VOL, BUCK]
        return mod(base, physics=ph, objective=obj, constraints=cons)
    if response == "min_frequency":
        return mod(base, physics=ph, constraints=[VOL, {"response": {"type": "min_frequency", "f_min": 1.0}}])
    if response in EIGEN:
        return mod(base, physics=ph, objective=EIGEN[response])
    spec = STATE[response]
    c = mod(base, physics=ph, objective=spec)
    if response == "robust_compliance":
        c["chain"] = [{"type": "density_filter", "rmin": 1.5}, {"type": "heaviside_projection", "beta": 1.0}]
    if response == "displacement" and base is B3:
        c["objective"]["where"]["at"] = ["max", "min", "min"]
    return c


def cell_pattern(response: str, physics: str) -> str:
    if response in EIGEN:
        return r"needs physics modal"
    if response == "end_compliance":
        return r"needs physics nonlinear_elasticity"
    if physics == "modal":
        return rf"unsupported combination \(catalog\): response {response} with physics modal: needs the static state u"
    return rf"unsupported combination \(catalog\): response buckling with physics {physics}"


UNSUPPORTED_CELLS = [(r, p) for r, row in RESPONSE_MATRIX.items() for p, c in row.items()
                     if c.status == UNSUPPORTED and c.enforced in ("component", "catalog")]


@pytest.mark.parametrize(("response", "physics"), UNSUPPORTED_CELLS, ids=[f"{r}-x-{p}" for r, p in UNSUPPORTED_CELLS])
def test_unsupported_cell_fails_closed_at_build(response: str, physics: str) -> None:
    with pytest.raises(ConfigError, match=cell_pattern(response, physics)):
        build_setup(cell_probe(response, physics))


def _gmg_field() -> Any:
    from toporlogy.core.domain import Domain
    from toporlogy.fem.elastic_gmg import ElasticityGMG
    from toporlogy.fem.elements import D_iso
    d = Domain((4, 4, 4))
    return ElasticityGMG(d, np.arange(12), elasticity=np.repeat(D_iso(0.3)[None], d.n_active, axis=0))


def _cfg(c: dict[str, Any]) -> Callable[[], Any]:
    return lambda: build_setup(c)


OBLIQUE = {"type": "roller", "where": {"type": "plane", "axis": "x", "at": "max"}, "normal": [1, 1]}
OBLIQUE3 = {"type": "roller", "where": {"type": "plane", "axis": "x", "at": "max"}, "normal": [1, 1, 0]}
FIBRE = {"name": "f", "orthotropic": {"E1": 1.0, "E2": 0.1, "E3": 0.1, "G12": 0.05, "G13": 0.05, "G23": 0.04,
                                      "nu12": 0.3, "nu13": 0.3, "nu23": 0.3}}
TENSOR = mod(B2, interpolation={"type": "tensor_phases", "phases": [FIBRE]}, physics="tensor_elasticity")
ANGLE_CHAIN = [{"type": "embed", "angle": {"rmin": 1.5}}, {"type": "density_filter", "rmin": 1.5}]
RULE_PROBES: dict[str, tuple[Callable[[], Any], type[BaseException], str]] = {
    "gmg_2d": (_cfg(mod(B2, physics="elasticity_gmg")), ConfigError, r"3-D only"),
    "gmg_material_field": (_gmg_field, ValueError, r"per-element elasticity .* is not supported"),
    "gmg_per_case_supports": (_cfg(mod(B3, physics="elasticity_gmg", load_cases=two_case_supports(B3))), ConfigError,
                              r"elasticity_gmg needs one support set"),
    "gmg_oblique_supports": (_cfg(mod(B3, physics="elasticity_gmg", supports=B3["supports"] + [OBLIQUE3])),
                             ConfigError, r"elasticity_gmg supports zero displacements along x / y / z only"),
    "modal_per_case_supports": (_cfg(mod(B2, physics="modal", objective=EIGEN["frequency"],
                                         load_cases=two_case_supports(B2))), ConfigError,
                                r"modal physics needs one support set"),
    "nonlinear_per_case_supports": (_cfg(mod(B2, physics="nonlinear_elasticity", objective="end_compliance",
                                             load_cases=two_case_supports(B2))), ConfigError,
                                    r"nonlinear_elasticity needs one support set"),
    "buckling_oblique_supports": (_cfg(mod(B2, supports=B2["supports"] + [OBLIQUE], constraints=[VOL, BUCK])),
                                  ConfigError, r"buckling with oblique supports or springs"),
    "buckling_per_case_supports": (_cfg(mod(B2, load_cases=two_case_supports(B2), constraints=[VOL, BUCK])),
                                   ConfigError, r"buckling with per-case supports"),
    "plane_strain_3d": (_cfg(mod(B3, material={"plane": "strain"})), ConfigError, r"applies to 2-D domains only"),
    "hashin_shtrikman_orthotropic": (_cfg(mod(B2, interpolation="hashin_shtrikman", material={"orthotropic": ORTHO})),
                                     ConfigError, r"isotropic material only"),
    "phases_responses": (_cfg(mod(B2, material={"phases": PHASES}, interpolation="multiphase_simp")), ConfigError,
                         r"only Compliance, PhaseVolume, Mass, Cost"),
    "phases_oc": (_cfg(mod(B2, material={"phases": PHASES}, interpolation="multiphase_simp", optimizer="oc",
                           constraints=[{"response": "phase_volume", "upper": 0.5}])), ConfigError,
                  r"OC does not handle several phases"),
    "phases_links": (_cfg(mod(B2, material={"phases": PHASES}, interpolation="multiphase_simp",
                              chain=[{"type": "helmholtz_filter", "rmin": 1.5}],
                              constraints=[{"response": "phase_volume", "upper": 0.5}])), ConfigError,
                     r"chain may hold only"),
    "nonlinear_incompatible_modes": (_cfg(mod(B2, physics={"type": "nonlinear_elasticity", "element": "Q4IM"},
                                              objective="end_compliance")), ConfigError, r"Q4IM"),
    "finalize_2d": (_cfg(mod(B2, post=["finalize"])), ConfigError, r"finalize needs a 3-D domain"),
    # material design (docs/olcumler/malzeme_tasarimi.md)
    "tensor_responses": (_cfg(mod(TENSOR, objective=STATE["displacement"])), ConfigError,
                         r"with physics tensor_elasticity: only Compliance"),
    "tensor_interpolation": (_cfg(mod(TENSOR, interpolation="simp")), ConfigError,
                             r"physics tensor_elasticity with interpolation simp"),
    "tensor_phases_physics": (_cfg(mod(TENSOR, physics="elasticity")), ConfigError,
                              r"tensor_phases needs physics tensor_elasticity"),
    "tensor_incompatible_modes": (_cfg(mod(TENSOR, physics={"type": "tensor_elasticity", "element": "Q4IM"})),
                                  ConfigError, r"linear in D"),
    "tensor_per_case_supports": (_cfg(mod(TENSOR, load_cases=two_case_supports(B2))), ConfigError,
                                 r"tensor_elasticity needs one support set"),
    "tensor_oblique_supports": (_cfg(mod(TENSOR, supports=B2["supports"] + [OBLIQUE])), ConfigError,
                                r"tensor_elasticity supports zero displacements along x / y / z only"),
    "tensor_post": (_cfg(mod(TENSOR, post=["discrete_check"])), ConfigError, r"post step\(s\) \['discrete_check'\]"),
    "angle_3d": (_cfg(mod(TENSOR, domain=B3["domain"], load_cases=B3["load_cases"], chain=ANGLE_CHAIN)), ConfigError,
                 r"angle block is 2-D only"),
    "angle_phases": (_cfg(mod(TENSOR, chain=ANGLE_CHAIN, interpolation=dict(TENSOR["interpolation"],
                                                                          phases=[FIBRE, dict(FIBRE, name="g")]))),
                     ConfigError, r"angle block with 2 phases is not supported"),
    "angle_interpolation": (_cfg(mod(B2, chain=ANGLE_CHAIN)), ConfigError, r"angle block needs interpolation "
                                                                          r"tensor_phases"),
    "angle_isotropic": (_cfg(mod(TENSOR, chain=ANGLE_CHAIN, interpolation=dict(
        TENSOR["interpolation"], phases=[{"name": "iso", "isotropic": {"E": 1.0, "nu": 0.3}}]))), ConfigError,
        r"does not depend on the in-plane angle"),
    "angle_oc": (_cfg(mod(TENSOR, chain=ANGLE_CHAIN, optimizer="oc")), ConfigError, r"angle block with optimizer oc"),
}
ENFORCED_RULES = [r.id for r in RULES if r.status == UNSUPPORTED and r.enforced in ("component", "catalog")]


@pytest.mark.parametrize("rule", ENFORCED_RULES)
def test_unsupported_rule_fails_closed(rule: str) -> None:
    action, exc, pattern = RULE_PROBES[rule]
    with pytest.raises(exc, match=pattern):
        action()


def test_every_enforced_rule_has_a_probe() -> None:
    assert sorted(RULE_PROBES) == sorted(ENFORCED_RULES)


@pytest.mark.parametrize("cfg", [
    pytest.param(mod(B2, constraints=[VOL, BUCK]), id="elasticity-buckling"),
    pytest.param(mod(B2, physics="modal", objective=EIGEN["frequency"]), id="modal-frequency"),
    pytest.param(mod(B2, physics="nonlinear_elasticity", objective="end_compliance"), id="nonlinear-end_compliance"),
    pytest.param(mod(B3, physics="elasticity_gmg"), id="gmg-compliance"),
])
def test_supported_combinations_still_build(cfg: dict[str, Any]) -> None:
    build_setup(cfg)


def test_check_build_reads_nested_responses() -> None:
    """A buckling response nested in a composite is found as well (maximize / weighted_sum specs)."""
    resolved = {"physics": {"type": "elasticity_gmg"},
                "objective": {"type": "weighted_sum", "responses": [{"type": "compliance"}, {"type": "buckling"}]},
                "constraints": []}
    with pytest.raises(ConfigError, match="buckling with physics elasticity_gmg"):
        check_build(resolved, object(), [])

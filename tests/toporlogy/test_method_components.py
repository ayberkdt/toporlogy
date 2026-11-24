"""YAML components of the methods added on 21 Nov 2026: built objects, default bounds, error messages."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.design.pde_filter import HelmholtzFilter
from toporlogy.optimization.buckling import BucklingKS
from toporlogy.optimization.compose import Displacement, WeightedSum
from toporlogy.optimization.local import Aggregated, AugmentedLagrangian, MultiplierUpdate
from toporlogy.optimizers import GCMMA, MMA, SiMPL


def _raw(**over):
    raw = {"name": "t", "domain": {"type": "box", "shape": [12, 6]}, "material": {"E0": 2.0, "nu": 0.25},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"name": "a", "loads": [{"type": "force", "where": {"type": "plane", "axis": "x",
                                                                              "at": "max"}, "force": [0, -1]}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}], "objective": "compliance",
           "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma",
           "run": {"max_iter": 3, "tol": 0.01}}
    raw.update(over)
    return raw


def test_stress_constraint_defaults_and_exclusion() -> None:
    st = build_setup(_raw(constraints=[
        {"response": "volume", "upper": 0.5},
        {"response": {"type": "stress", "sigma_lim": 3.0, "aggregate": "ks", "parameter": 20.0,
                      "exclude": [{"type": "box", "lo": [10, 0], "hi": [12, 6]}]}}])).study
    c = st.constraints[1]
    r = c.response
    assert isinstance(r, Aggregated) and c.upper == 1.0 and c.scale == 1.0 and c.key == "stress"
    assert r.kind == "ks" and r.parameter == 20.0 and r.scale is not None and r.scale.damping == 0.5
    assert r.field.sigma_lim == 3.0 and r.field.nu == 0.25 and r.field.E0 == 2.0
    assert len(r.field.elements) == 60 and "u" in r.needs          # 12 of the 72 elements excluded
    res = st.run(max_iter=3)
    assert np.isfinite(res.constraint_values["stress"])


def test_buckling_constraint_bound_from_load_factor() -> None:
    st = build_setup(_raw(constraints=[{"response": "volume", "upper": 0.5},
                                       {"response": {"type": "buckling", "load_factor": 4.0, "n_modes": 3}}])).study
    c = st.constraints[1]
    assert isinstance(c.response, BucklingKS) and c.upper == 0.25 and c.scale == 4.0 and c.response.E0 == 2.0
    with pytest.raises(ConfigError, match="load_factor > 0"):
        build_setup(_raw(constraints=[{"response": {"type": "buckling", "load_factor": 0.0}}]))


def test_displacement_and_weighted_sum() -> None:
    st = build_setup(_raw(objective={"type": "weighted_sum", "references": "first", "weights": [1.0, 0.5],
                                     "responses": ["compliance",
                                                   {"type": "displacement", "direction": [0, -2],
                                                    "where": {"type": "plane", "axis": "x", "at": "max"}}]})).study
    ws = st.objective
    assert isinstance(ws, WeightedSum) and isinstance(ws.responses[1], Displacement)
    disp = ws.responses[1]
    assert disp.dofs.size == 7 and np.allclose(disp.weights, -1.0 / 7)      # unit vector −y, mean over 7 nodes
    ev = st.evaluate(st.default_start())
    assert ev.f0 == pytest.approx(1.5)                                      # both terms 1 at the first evaluation


def test_augmented_lagrangian_objective_and_schedule() -> None:
    raw = _raw(objective={"type": "stress_augmented_lagrangian", "objective": "volume", "sigma_lim": 3.0, "mu0": 2.0},
               schedule={"type": "multiplier_update", "every": 2, "beta0": 1.0, "beta_max": 4.0},
               chain=[{"type": "density_filter", "rmin": 1.5}, {"type": "heaviside_projection"}])
    st = build_setup(raw).study
    al, sch = st.objective, st.schedule
    assert isinstance(al, AugmentedLagrangian) and isinstance(sch, MultiplierUpdate) and sch.al is al
    assert al.mu0 == 2.0 and al.field.form == "polynomial" and sch.inner is not None and not sch.reset_optimizer
    res = st.run(max_iter=5)
    assert al.n_updates == 2 and set(res.history) >= {"al_mu", "al_updates", "beta"}
    assert res.history["al_mu"][0] == 2.0 and res.history["al_mu"][-1] > 2.0
    with pytest.raises(ConfigError, match="augmented-Lagrangian objective"):
        build_setup(_raw(schedule={"type": "multiplier_update"}))


def test_optimizers_filter_and_post_step(tmp_path) -> None:
    from toporlogy.execution.runner import run_setup
    for name, cls in (("gcmma", GCMMA), ("simpl", SiMPL), ("mma", MMA)):
        setup = build_setup(_raw(optimizer=name, chain=[{"type": "helmholtz_filter", "rmin": 2.0}],
                                 post=[{"type": "discrete_check", "max_ratio": 100.0}]))
        st = setup.study
        assert isinstance(st.optimizer, cls) and isinstance(st.chain.find(HelmholtzFilter), HelmholtzFilter)
        run_setup(setup, tmp_path / name)
        assert st.optimizer.value_fn is None
    with pytest.raises((ConfigError, ValueError), match="state solve|needs U|cannot use"):
        build_setup(_raw(optimizer="simpl", constraints=[{"response": {"type": "stress", "sigma_lim": 1.0}}]))


def test_two_runs_of_one_study_are_identical_with_scaled_responses() -> None:
    """Responses that freeze a scale at their first evaluation (buckling KS, adaptive stress scale, weighted sum with
    references "first") must take it again in every run: README, "running a study twice gives the same result".
    Red before Study.run reset the scales (21 Nov 2026): the second run of the first study below differed from the
    first. The eigenvalue-KS study is a guard only: it gave identical runs without the reset as well (measured)."""
    raw = _raw(objective={"type": "weighted_sum", "references": "first",
                          "responses": ["compliance", {"type": "buckling", "n_modes": 3}]},
               constraints=[{"response": "volume", "upper": 0.5},
                            {"response": {"type": "stress", "sigma_lim": 2.0, "adaptive": 0.5}, "upper": 5.0}])
    st = build_setup(raw).study
    a = st.run(max_iter=6)
    scales = (list(st.objective.references), st.objective.responses[1].scale, st.constraints[1].response.scale.value)
    b = st.run(max_iter=6)
    assert np.array_equal(a.x, b.x) and a.history["f0"] == b.history["f0"]
    assert scales == (list(st.objective.references), st.objective.responses[1].scale,
                      st.constraints[1].response.scale.value)

    modal = {"name": "m", "domain": {"type": "box", "shape": [16, 4]}, "material": {"E0": 1.0, "nu": 0.3, "rho": 1.0},
             "physics": {"type": "modal", "n_modes": 3},
             "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
             "chain": [{"type": "density_filter", "rmin": 1.5}], "interpolation": {"type": "two_range_simp"},
             "objective": {"type": "maximize", "response": {"type": "min_eigenvalue_ks", "n_modes": 2}},
             "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": {"type": "mma", "move": 0.1},
             "run": {"max_iter": 5, "tol": 0.001}}
    sm = build_setup(modal).study
    c, d = sm.run(max_iter=5), sm.run(max_iter=5)
    assert np.array_equal(c.x, d.x)

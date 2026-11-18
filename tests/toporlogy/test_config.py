"""Declarative studies (toporlogy.config + registry): validation messages, equivalence with a Python-built Study,
resolved-config round trip, overrides, sweeps, registry extension. Problems are tiny (CPU shared with other jobs)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from toporlogy import OC, SIMP, Compliance, Constraint, Domain, Elasticity, Study, Volume, standard_chain
from toporlogy import config as C
from toporlogy.registry import REGISTRY, ConfigError, register


def mbb(nx: int = 16, ny: int = 6, rmin: float = 1.5, vf: float = 0.5, max_iter: int = 8) -> dict:
    """Half MBB beam config (2-D)."""
    return {
        "name": "mbb",
        "domain": {"type": "box", "shape": [nx, ny], "h": 1.0},
        "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}, "dofs": "x"},
                     {"type": "fixed", "where": {"type": "point", "at": ["max", "min"]}, "dofs": "y"}],
        "load_cases": [{"name": "centre", "loads": [{"type": "force", "where": {"type": "point", "at": ["min", "max"]},
                                                     "force": [0, -1]}]}],
        "chain": [{"type": "density_filter", "rmin": rmin}],
        "interpolation": {"type": "simp", "penal": 3.0, "emin_ratio": 1e-9},
        "constraints": [{"response": "volume", "upper": vf}],
        "optimizer": {"type": "oc", "bisection_tol": 1e-3},
        "run": {"max_iter": max_iter, "tol": 0.01},
    }


def errors_of(raw: dict) -> dict[str, str]:
    with pytest.raises(ConfigError) as ei:
        C.build_setup(raw)
    return dict(ei.value.errors)


# ---------------------------------------------------------------------------------------------- validation
def test_missing_required_key():
    raw = mbb()
    del raw["domain"]
    errs = errors_of(raw)
    assert "required" in errs["domain"]


def test_wrong_type_names_key_path():
    raw = mbb()
    raw["material"] = {"E0": "abc"}
    raw["run"]["max_iter"] = 2.5
    errs = errors_of(raw)
    assert errs["material.E0"] == 'expected number, got "abc"'
    assert errs["run.max_iter"].startswith("expected integer")


def test_unknown_component_did_you_mean():
    raw = mbb()
    raw["chain"][0]["type"] = "density_filtr"
    raw["optimizer"] = "mmaa"
    errs = errors_of(raw)
    assert errs["chain.0.type"] == 'unknown link "density_filtr"; did you mean "density_filter"?'
    assert 'did you mean "mma"' in errs["optimizer.type"]


def test_unknown_parameter_and_top_level_key():
    raw = mbb()
    raw["chain"][0] = {"type": "density_filter", "rmn": 1.5}
    raw["optimiser"] = "oc"
    errs = errors_of(raw)
    assert 'did you mean "rmin"' in errs["chain.0.rmn"]
    assert "required" in errs["chain.0.rmin"]
    assert 'did you mean "optimizer"' in errs["optimiser"]


def test_lenient_input():
    """Numbers as text (PyYAML reads 1e-9 as text), names in any case / with dashes, constraint shorthand."""
    raw = mbb()
    raw["interpolation"] = {"type": "SIMP", "emin_ratio": "1e-9"}
    raw["chain"] = [{"type": "Density-Filter", "rmin": 2}]
    raw["constraints"] = [{"type": "volume", "upper": "0.4"}]
    r = C.resolve(raw)
    assert r["interpolation"] == {"type": "simp", "penal": 3.0, "emin_ratio": 1e-9}
    assert [lk["type"] for lk in r["chain"]] == ["embed", "density_filter", "passive"]
    assert r["chain"][1]["rmin"] == 2.0
    assert r["constraints"][0]["response"] == {"type": "volume", "weighted": False}
    assert r["constraints"][0]["upper"] == 0.4


def test_empty_selection_reports_component_path():
    raw = mbb()
    raw["supports"][0]["where"]["at"] = 3.3          # between node lines
    errs = errors_of(raw)
    assert "selects no node" in errs["supports.0.where"]


def test_yaml_reader(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(mbb()), encoding="utf-8")
    st = C.load_config(p)
    assert st.domain.shape == (16, 6) and st.config["name"] == "mbb"
    (tmp_path / "bad.yaml").write_text("domain: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot parse"):
        C.read_config(tmp_path / "bad.yaml")


# ---------------------------------------------------------------------------------------------- equivalence
def test_config_equals_python_study():
    """The same half MBB beam built from a config and in Python: identical objective history and design."""
    nx, ny, max_iter = 16, 6, 8
    st_cfg = C.study_from_dict(mbb(nx, ny, max_iter=max_iter))
    h_cfg = st_cfg.run(max_iter=max_iter, tol=0.01)

    d = Domain((nx, ny))
    load = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == ny))
    F = d.load_vector()
    F[d.dofs(load, 1), 0] = -1.0
    fixed = np.concatenate([d.dofs(d.nodes_where(lambda X: X[:, 0] == 0), 0),
                            d.dofs(d.nodes_where(lambda X: (X[:, 0] == nx) & (X[:, 1] == 0)), 1)])
    py = Study(d, Elasticity(d, fixed, nu=0.3, element="Q4"), F, standard_chain(d, 1.5, projection=False),
               SIMP(d, penal=3.0, emin_ratio=1e-9), Compliance(), [Constraint(Volume(), 0.5)],
               OC(move=0.2, bisection_tol=1e-3, dv_scale=d.n_active))
    h_py = py.run(max_iter=max_iter, tol=0.01)

    assert np.array_equal(st_cfg.F, py.F)
    assert np.array_equal(st_cfg.physics.fixed, py.physics.fixed)
    assert h_cfg.history["f0"] == h_py.history["f0"]                 # bit-identical
    assert np.array_equal(h_cfg.xphys, h_py.xphys)


def test_load_case_weights_reach_compliance():
    raw = mbb()
    raw["load_cases"] = [dict(raw["load_cases"][0], weight=1.0),
                         {"name": "side", "weight": 0.25,
                          "loads": [{"type": "force", "where": {"type": "point", "at": ["min", "max"]},
                                     "force": [0.5, 0]}]}]
    s = C.build_setup(raw)
    assert s.study.F.shape[1] == 2
    assert s.study.objective.weights.tolist() == [1.0, 0.25]
    assert s.cases[1]["resultant"] == [0.5, 0.0]


# ---------------------------------------------------------------------------------------------- run folder
def test_resolved_config_round_trip_and_run_folder(tmp_path):
    s = C.build_setup(mbb(max_iter=3))
    sm = C.run_setup(s, tmp_path / "run", source="test")
    files = {p.name for p in (tmp_path / "run").iterdir()}
    assert {"config.resolved.yaml", "setup.json", "history.jsonl", "result.json", "result.npz",
            "summary.json"} <= files
    saved = yaml.safe_load((tmp_path / "run" / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert set(saved["provenance"]) == {"git", "created", "source", "overrides"}
    again = C.resolve(saved)
    strip = lambda r: {k: v for k, v in r.items() if k != "provenance"}
    assert strip(again) == strip(s.config)
    assert sm["iterations"] == 3 and sm["status"] == "not converged"
    assert sm["constraints"][0]["name"] == "volume" and sm["constraints"][0]["satisfied"]
    back = C.read_run(tmp_path / "run")
    assert back["objective"] == sm["objective"] and len(back["history"]) == 3


def test_summary_constraints_belong_to_the_returned_design(tmp_path):
    """summary.json reports objective AND constraints of the design in result.npz, not of the last analysed
    iterate (history[-1] is x_{k}; the returned design is x_{k+1})."""
    s = C.build_setup(mbb(max_iter=3))
    sm = C.run_setup(s, tmp_path / "run", source="test")
    x = np.load(tmp_path / "run" / "result.npz")["x"]
    ev = s.study.evaluate(x, gradients=False)
    hist = [json.loads(ln) for ln in (tmp_path / "run" / "history.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    assert sm["objective"] == pytest.approx(ev.f0, rel=1e-12)
    for c in sm["constraints"]:
        assert c["value"] == pytest.approx(ev.values[c["name"]], rel=1e-12), c["name"]
    # the test is sensitive: the last analysed iterate has a different volume than the returned design
    last = hist[-1]["volume"]
    assert abs(last - ev.values["volume"]) > 1e-9


# ---------------------------------------------------------------------------------------------- overrides / sweeps
def test_overrides():
    raw = mbb()
    out = C.apply_overrides(raw, ["chain.0.rmin=2.5", "optimizer.move=0.1", "objective.weights=[1]",
                                  "run.max_iter=4"])
    assert out["chain"][0]["rmin"] == 2.5 and out["optimizer"]["move"] == 0.1 and out["run"]["max_iter"] == 4
    assert out["objective"] == {"weights": [1]}                       # created, type missing → validation error
    assert raw["chain"][0]["rmin"] == 1.5                             # input untouched
    out2 = C.apply_overrides({"optimizer": "oc"}, ["optimizer.move=0.3"])
    assert out2["optimizer"] == {"type": "oc", "move": 0.3}           # component name expanded to a mapping
    with pytest.raises(ConfigError, match="index out of range"):
        C.apply_overrides(raw, ["chain.5.rmin=1"])
    with pytest.raises(ConfigError, match="key.path=value"):
        C.apply_overrides(raw, ["chain.0.rmin"])


def test_sweep_parsing():
    sw = C.parse_sweep(["chain.0.rmin=1.5,2", "domain.shape=[8,4],[10,4]"])
    assert sw == [("chain.0.rmin", [1.5, 2]), ("domain.shape", [[8, 4], [10, 4]])]
    combos = C.sweep_combinations(sw)
    assert len(combos) == 4 and combos[1] == [("chain.0.rmin", 1.5), ("domain.shape", [10, 4])]
    assert C.combination_label(combos[0]) == "chain.0.rmin=1.5__domain.shape=-8-4-"


# ---------------------------------------------------------------------------------------------- registry
def test_register_new_components_one_line(tmp_path):
    """A new link and a post step register with one line each and are usable from a config."""
    from toporlogy.chain import HeavisideProjection

    class Clamp(HeavisideProjection):
        pass

    def stats(ctx, prefix: str = "x"):
        return lambda res, study, folder: {f"{prefix}_max": float(res.xphys.max())}

    register("link", "test_clamp", Clamp)
    register("post", "test_stats", stats)
    try:
        assert REGISTRY.get("link", "test_clamp").params["beta"].default == 1.0
        raw = mbb(max_iter=2)
        raw["chain"].append({"type": "test_clamp", "beta": 2})
        raw["post"] = [{"type": "test_stats", "prefix": "rho"}]
        s = C.build_setup(raw)
        assert isinstance(s.study.chain[2], Clamp) and s.study.chain[2].beta == 2.0
        sm = C.run_setup(s, tmp_path / "r")
        assert 0 < sm["post"]["test_stats"]["rho_max"] <= 1
        with pytest.raises(ValueError, match="already registered"):
            register("link", "test_clamp", Clamp)
    finally:
        REGISTRY.unregister("link", "test_clamp")
        REGISTRY.unregister("post", "test_stats")


def test_every_core_component_lists_parameters():
    for kind in REGISTRY.kinds():
        for name in REGISTRY.names(kind):
            params = REGISTRY.get(kind, name).params
            json.dumps({k: (None if p.required else p.default) for k, p in params.items()})


def test_duplicate_response_names_rejected():
    raw = mbb()
    raw["constraints"] = [{"response": "volume", "upper": 0.5}, {"response": "volume", "upper": 0.6}]
    raw["optimizer"] = "mma"
    errs = errors_of(copy.deepcopy(raw))
    assert "constraints.1.name" in errs["constraints.1"]
    raw["constraints"][1]["name"] = "volume_b"
    s = C.build_setup(raw)
    assert s.constraint_names == ["volume", "volume_b"]


def test_oc_with_a_state_dependent_constraint_is_rejected_at_build():
    """B-07: OC evaluates its constraint inside the bisection without a solve, so a constraint that needs U failed
    at the first step ("OC needs constraint_fn"). It is now rejected when the study is built, from YAML and Python."""
    raw = mbb()
    raw["objective"] = "volume"
    raw["constraints"] = [{"response": "compliance", "name": "stiffness", "upper": 500.0}]
    errs = errors_of(raw)
    assert "state solve" in errs["constraints"] and "stiffness" in errs["constraints"] and "MMA" in errs["constraints"]
    raw["optimizer"] = "mma"
    assert C.build_setup(raw).study.optimizer.name == "MMA"              # MMA takes it
    d = Domain((6, 3))
    with pytest.raises(ValueError, match="state solve"):
        Study(d, Elasticity(d, d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))), d.load_vector(), standard_chain(d, 1.5),
              SIMP(d), Volume(), [Constraint(Compliance(), 1.0)], OC())

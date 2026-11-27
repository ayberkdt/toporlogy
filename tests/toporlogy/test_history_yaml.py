"""History-dependent topology optimisation from YAML / the CLI (docs/foundations/prereg/HY_history_yaml.md, HY1-HY9).

HY1 / HY2 / HY8 / HY6 (objective part) compare two construction paths of the same kernel: REGRESSION tests, not
independent physics checks. HY3 / HY5 are independent (closed-form literals). All materials and loads are SYNTHETIC."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from toporlogy.configuration.build import Setup, build_setup
from toporlogy.configuration.history_build import HistorySetup, is_history_config, resolve_history
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config, resolve
from toporlogy.core.linear_elastic import isotropic_moduli
from toporlogy.core.vtu import read_vtu
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries
from toporlogy.execution.cli import main
from toporlogy.fem.mesh.mesh import perturb, rectangle
from toporlogy.optimization.creep_topology import CreepStructure
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure, optimize_viscoelastic

REPO = Path(__file__).resolve().parents[2]
EX = REPO / "examples" / "foundations"
VE_EXAMPLE, CR_EXAMPLE = EX / "viscoelastic_service.yaml", EX / "creep_service.yaml"
MPA = 1e6
TIMES = [0.5, 1.0, 1.7, 2.0, 3.5, 6.0, 9.0, 15.0, 30.0, 60.0, 120.0, 300.0]
_DEL = object()


# ------------------------------------------------------------------------------------------------ builders
def _ve_card() -> dict[str, Any]:
    return {"name": "hy_gm", "model": "generalized_maxwell", "units": {"stress": "MPa", "time": "s"},
            "parameters": {"shear": {"inf": 1.0, "amplitudes": [2.0, 3.0], "taus": [1.0, 100.0]},
                           "bulk": {"inf": 5.0, "amplitudes": [4.0], "taus": [10.0]}},
            "provenance": {"source": "synthetic test card", "synthetic": True, "date": "2026-11-27"},
            "validity": {"time": [1.0e-3, 1.0e+5]}}


def _cr_card() -> dict[str, Any]:
    return {"name": "hy_norton", "model": "norton", "units": {"stress": "Pa", "rate": "1/s"},
            "parameters": {"elasticity": {"E": 7.0e10, "nu": 0.33}, "A": 1.0e-6, "sigma_ref": 1.0e8, "n": 3.0},
            "provenance": {"source": "synthetic test card", "synthetic": True, "date": "2026-11-27"},
            "validity": {"stress": [0.0, 2.0e8]}}


def _cfg(card: dict[str, Any], times: list[float], force: float, n: tuple[int, int] = (8, 4),
         rmin: float = 0.2, ramp: float = 2.0, at: float | None = None) -> dict[str, Any]:
    return {"name": "hy", "analysis": {"type": "history"},
            "domain": {"generator": "rectangle", "n": list(n), "size": [2.0, 1.0], "element": "Q4", "perturb": 0.06,
                       "seed": 3},
            "material": card, "supports": [{"set": "left", "fix": "all"}],
            "loads": [{"point": [2.0, 0.0], "force": [0.0, force], "time": {"ramp": ramp}}],
            "times": list(times), "objective": {"compliance_at": times[-1] if at is None else at},
            "constraints": [{"response": "volume", "upper": 0.4}],
            "chain": [{"type": "density_filter", "rmin": rmin}],
            "optimizer": {"type": "mma", "max_iter": 3}}


def _ve_cfg() -> dict[str, Any]:
    return _cfg(_ve_card(), TIMES, -1.0e3)


def _mesh_and_fixed(n: tuple[int, int]) -> tuple[Any, np.ndarray, int]:
    """The API-side mesh, fixed DOFs and tip node, found from coordinates (independent of the config reader)."""
    m = perturb(rectangle(n[0], n[1], 2.0, 1.0), 0.06, seed=3)
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    tip = int(np.flatnonzero(np.isclose(P[:, 0], 2.0) & np.isclose(P[:, 1], 0.0))[0])
    return m, np.concatenate([2 * left, 2 * left + 1]), tip


def _tip_loads(m: Any, tip: int, times: list[float], force: float, ramp: float) -> np.ndarray:
    loads = np.zeros((len(times), 2 * m.n_nodes))
    for k, t in enumerate(times):
        loads[k, 2 * tip + 1] = force * min(t / ramp, 1.0)
    return loads


def _ve_api() -> ViscoelasticStructure:
    mat = GeneralizedMaxwell(PronySeries(1.0 * MPA, [2.0 * MPA, 3.0 * MPA], [1.0, 100.0]),
                             PronySeries(5.0 * MPA, [4.0 * MPA], [10.0]))
    m, fixed, tip = _mesh_and_fixed((8, 4))
    return ViscoelasticStructure(m, mat, TIMES, _tip_loads(m, tip, TIMES, -1.0e3, 2.0), fixed)


CR_TIMES = [0.5, 1.0, 5.0, 20.0, 100.0]


def _cr_cfg() -> dict[str, Any]:
    c = _cfg(_cr_card(), CR_TIMES, -5.0e6, n=(6, 3), rmin=0.3, ramp=1.0)
    c["optimizer"]["max_iter"] = 2
    return c


def _cr_api() -> CreepStructure:
    mo = isotropic_moduli(E=7.0e10, nu=0.33)
    m, fixed, tip = _mesh_and_fixed((6, 3))
    return CreepStructure(m, mo.K, mo.G, 1.0e-6, 1.0e8, 3.0, CR_TIMES, _tip_loads(m, tip, CR_TIMES, -5.0e6, 1.0), fixed)


def _same_run(a: Any, b: Any) -> bool:
    return (np.array_equal(a.x, b.x) and np.array_equal(a.xphys, b.xphys) and a.objective == b.objective
            and a.history["f0"] == b.history["f0"] and a.history["volume"] == b.history["volume"]
            and a.iterations == b.iterations and a.termination == b.termination
            and (a.objectives is None) == (b.objectives is None)
            and (a.objectives is None or np.array_equal(a.objectives, b.objectives)))


# ------------------------------------------------------------------------------------------------ HY1
@pytest.mark.parametrize("objective", ["single", "minmax"])
def test_hy1_viscoelastic_yaml_equals_api(objective: str) -> None:
    """REGRESSION (two construction paths): loads, J, dJ/dw and a 3-iteration run are bitwise equal."""
    cfg = _ve_cfg()
    if objective == "minmax":
        cfg["objective"] = {"max_compliance_at": [6.0, 300.0]}
    s = build_setup(cfg)
    assert isinstance(s, HistorySetup) and s.kind == "viscoelastic"
    ref = _ve_api()
    assert np.array_equal(s.structure.loads, ref.loads) and np.array_equal(s.structure.times, ref.times)
    assert np.array_equal(s.structure.free, ref.free) and np.array_equal(s.structure.dV, ref.dV)
    w = torch.as_tensor(0.2 + 0.7 * np.random.default_rng(5).random(ref.ne)).requires_grad_(True)
    k = s.steps[-1]
    assert k == 11
    out = []
    for st in (s.structure, ref):
        c = st.response(w).compliance
        g, = torch.autograd.grad(c[k], w)
        out.append((c.detach(), g))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])
    steps = None if objective == "single" else [5, 11]
    r1 = s.optimize()
    r2 = optimize_viscoelastic(ref, volume=0.4, rmin=0.2, step_index=11, penal=3.0, emin=1e-9, max_iter=3, move=0.2,
                               steps=steps, tol=0.0, kkt_tol=None)
    print(f"HY1 {objective}: J {float(out[0][0][k]):.6e}, run objective {r1.objective:.6e} vs {r2.objective:.6e}")
    assert s.steps == ([11] if objective == "single" else [5, 11])
    assert _same_run(r1, r2)


# ------------------------------------------------------------------------------------------------ HY2
def test_hy2_creep_yaml_equals_api() -> None:
    """REGRESSION (two construction paths) for the Norton creep structure."""
    s = build_setup(_cr_cfg())
    assert isinstance(s, HistorySetup) and s.kind == "creep"
    ref = _cr_api()
    assert np.array_equal(s.structure.loads, ref.loads)
    assert (s.structure.K0, s.structure.G0, s.structure.A, s.structure.sref, s.structure.n) == \
        (ref.K0, ref.G0, ref.A, ref.sref, ref.n)
    w = torch.as_tensor(0.3 + 0.6 * np.random.default_rng(6).random(ref.ne)).requires_grad_(True)
    out = []
    for st in (s.structure, ref):
        c = st.response(w).compliance
        g, = torch.autograd.grad(c[-1], w)
        out.append((c.detach(), g))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])
    r1 = s.optimize()
    r2 = optimize_viscoelastic(ref, volume=0.4, rmin=0.3, step_index=len(CR_TIMES) - 1, penal=3.0, emin=1e-9,
                               max_iter=2, move=0.2, steps=None, tol=0.0, kkt_tol=None)
    print(f"HY2: J {float(out[0][0][-1]):.6e}, run objective {r1.objective:.6e} vs {r2.objective:.6e}")
    assert _same_run(r1, r2)


# ------------------------------------------------------------------------------------------------ HY3
GRID3 = [5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0]
EXPECT3 = {"ramp": ({"ramp": 10.0}, [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
           "hold": ({"hold": True}, [1.0] * 8),
           "table": ({"table": [[0, 0], [10, 1], [100, 1], [200, 0.5]]}, [0.5, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.5])}


@pytest.mark.parametrize("form", list(EXPECT3))
def test_hy3_set_load_and_time_functions_match_closed_forms(form: str) -> None:
    """Independent: -900 N on the right edge of a 4 x 2 mesh = -300 N on each of its 3 nodes times s(t_k) (literals)."""
    spec, s_exact = EXPECT3[form]
    cfg = _cfg(_ve_card(), GRID3, -900.0, n=(4, 2))
    cfg["domain"].update(size=[2.0, 1.0], perturb=0.0)
    cfg["loads"] = [{"set": "right", "force": [0.0, -900.0], "time": spec}]
    cfg["objective"] = {"compliance_at": 300.0}
    L = build_setup(cfg).structure.loads
    m = rectangle(4, 2, 2.0, 1.0)
    right = np.flatnonzero(np.isclose(m.points[:, 0], 2.0))
    assert len(right) == 3
    expected = np.zeros_like(L)
    for node in right:
        expected[:, 2 * node + 1] = -300.0 * np.array(s_exact)
    assert np.allclose(L, expected, rtol=1e-14, atol=0.0)
    assert np.count_nonzero(L[:, 0::2]) == 0               # no x-force anywhere
    assert np.count_nonzero(L[0]) == 3


# ------------------------------------------------------------------------------------------------ HY4
def test_hy4_units_with_factors_equal_si() -> None:
    si = _cfg(_ve_card(), TIMES, -1.0e3, n=(6, 3))
    si["domain"].update(thickness=1.0, perturb=0.0)
    mm = copy.deepcopy(si)
    mm["units"] = {"length": {"name": "mm", "factor": 1.0e-3}, "force": {"name": "kN", "factor": 1.0e3},
                   "time": {"name": "h", "factor": 3600.0}}
    mm["domain"].update(size=[2000.0, 1000.0], thickness=1000.0)
    mm["loads"] = [{"point": [2000.0, 0.0], "force": [0.0, -1.0], "time": {"ramp": 2.0 / 3600.0}}]
    mm["times"] = [t / 3600.0 for t in TIMES]
    mm["objective"] = {"compliance_at": 300.0 / 3600.0}
    mm["chain"] = [{"type": "density_filter", "rmin": 200.0}]
    a, b = build_setup(si), build_setup(mm)
    assert np.allclose(b.structure.loads, a.structure.loads, rtol=1e-13, atol=0.0)
    assert np.allclose(b.times, a.times, rtol=1e-13, atol=0.0)
    assert b.steps == a.steps and abs(b.options["rmin"] - 0.2) <= 1e-13
    w = torch.full((a.structure.ne,), 0.4, dtype=torch.float64)
    ca, cb = a.structure.response(w).compliance.numpy(), b.structure.response(w).compliance.numpy()
    print(f"HY4: max relative difference {np.abs(cb / ca - 1).max():.2e}")
    assert np.allclose(cb, ca, rtol=1e-9, atol=0.0)


# ------------------------------------------------------------------------------------------------ HY5
def test_hy5_logspace_and_events_grid() -> None:
    cfg = _cfg(_ve_card(), TIMES, -1.0e3, n=(4, 2), ramp=10.0)
    cfg["times"] = {"logspace": [1.0, 1.0e3, 7], "events": [250.0, 10.0]}
    cfg["objective"] = {"compliance_at": 1000.0}
    t = resolve_history(cfg)["times"]
    expect = [1.0 * (1.0e3 / 1.0) ** (k / 6) for k in (0, 1)] + [10.0] + [1.0e3 ** (3 / 6), 1.0e3 ** (4 / 6), 250.0,
                                                                           1.0e3 ** (5 / 6), 1.0e3]
    assert len(t) == 8 and 10.0 in t and 250.0 in t
    assert np.allclose(t, sorted(expect), rtol=1e-12, atol=0.0)


# ------------------------------------------------------------------------------------------------ HY6
def _reduced(example: Path, tmp: Path) -> Path:
    cfg = read_config(example)
    cfg["domain"]["n"] = [8, 4]
    cfg["chain"] = [{"type": "density_filter", "rmin": 0.4}]
    cfg["optimizer"]["max_iter"] = 2
    p = tmp / f"{example.stem}_small.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


@pytest.mark.parametrize("example", [VE_EXAMPLE, CR_EXAMPLE], ids=["viscoelastic", "creep"])
def test_hy6_cli_validate_and_run(example: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(example)]) == 0
    assert "history" in capsys.readouterr().out
    cfgp = _reduced(example, tmp_path)
    folder = tmp_path / "run"
    code = main(["run", str(cfgp), "--out", str(folder), "-q"])
    printed = capsys.readouterr().out
    assert code == 1 and "RESULT" in printed and "not converged" in printed      # 2 iterations, tol 0
    for f in ("design.vtu", "history_summary.json", "summary.json", "history.jsonl", "result.npz",
              "config.resolved.yaml"):
        assert (folder / f).is_file(), f
    sm = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    hist = [json.loads(ln) for ln in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    extra = json.loads((folder / "history_summary.json").read_text(encoding="utf-8"))
    vtu = read_vtu(folder / "design.vtu")
    with np.load(folder / "result.npz") as z:
        xphys, comp = z["xphys"], z["compliance"]
    k = sm["objective_steps"][0]
    print(f"HY6 {example.stem}: exit {code}, objective {sm['objective']:.6e}, volume "
          f"{sm['constraints'][0]['value']:.4f}, step {k}")
    assert len(hist) == 2 and sm["iterations"] == 2 and sm["analysis"] == "history"
    assert np.isfinite(sm["objective"]) and sm["objective_name"] == "compliance"
    assert len(vtu.connectivity) == 32 and set(vtu.cell_data) == {"xphys", "x", "weight"}
    assert np.array_equal(vtu.cell_data["xphys"], xphys)
    assert f"displacement_step{k}" in vtu.point_data and vtu.point_data[f"displacement_step{k}"].shape == (45, 3)
    assert sm["objective"] == float(comp[k]) == extra["compliance_returned_design"][k]
    assert len(extra["times_s"]) == len(comp) and extra["objective_steps"] == [k]
    setup = build_setup(read_config(cfgp))
    wref = setup.weights(xphys)
    with torch.no_grad():
        again = setup.structure.response(wref).compliance[k]
    assert float(again) == sm["objective"]                  # REGRESSION: the same kernel at the returned design
    assert main(["show", str(folder)]) == 1
    assert "compliance" in capsys.readouterr().out


def test_hy6_cli_refuses_a_bad_config_with_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _ve_cfg()
    cfg["analysis"]["plane"] = "stress"
    cfg["objective"] = {"compliance_at": 7.0}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["validate", str(p)]) == 2
    err = capsys.readouterr().err
    assert "analysis.plane" in err and "objective.compliance_at" in err and "2 problems" in err
    assert main(["run", str(p), "--out", str(tmp_path / "r")]) == 2
    assert not (tmp_path / "r").exists()
    capsys.readouterr()
    assert main(["run", str(VE_EXAMPLE), "--sweep", "optimizer.max_iter=1,2", "--out", str(tmp_path / "s")]) == 2
    assert "sweep" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ HY7
def _set(cfg: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    node: Any = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    key: Any = int(parts[-1]) if isinstance(node, list) else parts[-1]
    if value is _DEL:
        del node[key]
    else:
        node[key] = value
    return cfg


def _card_with(path: str, value: Any) -> dict[str, Any]:
    return _set(_ve_card(), path, value)


M = EX / "materials"
REFUSALS: dict[str, tuple[str, Any, str, str]] = {      # id -> (key to set, value, path of the error, message part)
    "unknown_top_key": ("symmetry", {"x": 1.0}, "symmetry", "unknown"),
    "grid_only_key": ("load_cases", [{"loads": []}], "load_cases", "history path"),
    "run_key": ("run", {"max_iter": 5}, "run", "optimizer"),
    "schedule_key": ("schedule", {"type": "beta_continuation"}, "schedule", "not supported"),
    "unknown_optimizer_key": ("optimizer.foo", 1, "optimizer.foo", "unknown"),
    "unknown_domain_key": ("domain.shape", "disk", "domain.shape", "unknown"),
    "analysis_type": ("analysis.type", "static", "analysis.type", "history"),
    "plane_stress": ("analysis.plane", "stress", "analysis.plane", "plane stress"),
    "box_3d": ("domain.generator", "box", "domain.generator", "3-D"),
    "element_h8": ("domain.element", "H8", "domain.element", "Q4"),
    "domain_n_length": ("domain.n", [8], "domain.n", "expected a list of 2"),
    "material_unsupported_model": ("material", str(M / "physical_j2.yaml"), "material.model", "not supported"),
    "material_shift": ("material", str(M / "generalized_maxwell.yaml"), "material.parameters.shift", "isothermal"),
    "material_arrhenius": ("material", str(M / "norton.yaml"), "material.parameters.arrhenius", "isothermal"),
    "material_missing_card_key": ("material", _card_with("units", _DEL), "material.units", "missing"),
    "material_bad_parameter": ("material", _card_with("parameters.shear.inf", -1.0), "material.parameters.shear.inf",
                               ">= 0"),
    "material_missing_file": ("material", str(M / "no_such_card.yaml"), "material", "not found"),
    "material_validity_time": ("material", _card_with("validity.time", [1.0e-3, 100.0]), "times", "validity"),
    "objective_off_grid": ("objective.compliance_at", 7.0, "objective.compliance_at", "step-grid"),
    "objective_both": ("objective", {"compliance_at": 300.0, "max_compliance_at": [6.0, 300.0]}, "objective",
                       "exactly one"),
    "objective_one_time_list": ("objective", {"max_compliance_at": [300.0]}, "objective.max_compliance_at",
                                "at least 2"),
    "objective_duplicate": ("objective", {"max_compliance_at": [6.0, 6.0]}, "objective.max_compliance_at.1", "twice"),
    "ramp_off_grid": ("loads.0.time.ramp", 2.5, "loads.0.time.ramp", "step-grid"),
    "table_off_grid": ("loads.0.time", {"table": [[0, 0], [2.5, 1]]}, "loads.0.time.table.1", "step-grid"),
    "table_first_knot": ("loads.0.time", {"table": [[1.0, 0], [2.0, 1]]}, "loads.0.time.table.0", "t = 0"),
    "time_two_forms": ("loads.0.time", {"ramp": 2.0, "hold": True}, "loads.0.time", "exactly one"),
    "time_missing": ("loads.0.time", _DEL, "loads.0.time", "required"),
    "hold_false": ("loads.0.time", {"hold": False}, "loads.0.time.hold", "true"),
    "unit_without_factor": ("units", {"length": "mm"}, "units.length", "factor"),
    "unit_dict_without_factor": ("units", {"length": {"name": "mm"}}, "units.length.factor", "factor"),
    "unit_bad_factor": ("units", {"force": {"name": "kN", "factor": -1000.0}}, "units.force.factor", "> 0"),
    "unit_unknown_kind": ("units", {"temperature": "K"}, "units.temperature", "unknown"),
    "traction": ("loads.0", {"set": "right", "force": [0.0, -1.0], "distribute": "traction", "time": {"hold": True}},
                 "loads.0.distribute", "traction"),
    "unknown_support_set": ("supports", [{"set": "middle", "fix": ["x"]}], "supports.0.set", "node set"),
    "unknown_load_set": ("loads.0", {"set": "nowhere", "force": [0.0, -1.0], "time": {"hold": True}},
                         "loads.0.set", "node set"),
    "rigid_motion": ("supports", [{"set": "left", "fix": ["x"]}], "supports", "rigid"),
    "zero_force": ("loads.0.force", [0.0, 0.0], "loads.0.force", "zero"),
    "times_not_increasing": ("times", [1.0, 1.0, 300.0], "times", "strictly increasing"),
    "times_negative": ("times", [-1.0, 300.0], "times.0", ">= 0"),
    "logspace_zero_start": ("times", {"logspace": [0.0, 10.0, 3]}, "times.logspace.0", "> 0"),
    "logspace_one_point": ("times", {"logspace": [1.0, 10.0, 1]}, "times.logspace.2", ">= 2"),
    "logspace_reversed": ("times", {"logspace": [10.0, 1.0, 3]}, "times.logspace", "t0 < t1"),
    "times_unknown_key": ("times", {"logspace": [1.0, 10.0, 3], "extra": 1}, "times.extra", "unknown"),
    "volume_range": ("constraints.0.upper", 1.5, "constraints.0.upper", "(0, 1]"),
    "constraints_two": ("constraints", [{"response": "volume", "upper": 0.4}, {"response": "volume", "upper": 0.3}],
                        "constraints", "exactly one"),
    "constraint_compliance": ("constraints", [{"response": "compliance", "upper": 1.0}], "constraints.0.response",
                              "volume"),
    "chain_missing": ("chain", _DEL, "chain", "required"),
    "chain_two_links": ("chain", [{"type": "density_filter", "rmin": 0.2}, {"type": "passive"}], "chain", "exactly one"),
    "chain_heaviside": ("chain", [{"type": "heaviside_projection"}], "chain.0.type", "not supported"),
    "chain_rmin_missing": ("chain", [{"type": "density_filter"}], "chain.0.rmin", "required"),
    "optimizer_oc": ("optimizer.type", "oc", "optimizer.type", "mma"),
    "optimizer_max_iter": ("optimizer.max_iter", 0, "optimizer.max_iter", ">= 1"),
    "interpolation_ramp": ("interpolation", {"type": "ramp"}, "interpolation.type", "simp"),
    "post_stl": ("post", [{"type": "stl"}], "post.0.type", "not supported"),
    "post_name_clash": ("post", [{"type": "summary", "name": "summary"}], "post.0.name", "summary"),
    "missing_loads": ("loads", _DEL, "loads", "missing"),
    "missing_times": ("times", _DEL, "times", "missing"),
    "missing_objective": ("objective", _DEL, "objective", "required"),
    "missing_supports": ("supports", _DEL, "supports", "missing"),
    "missing_material": ("material", _DEL, "material", "required"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_hy7_every_refusal_names_the_key_path(case: str) -> None:
    key, value, path, part = REFUSALS[case]
    cfg = _set(_ve_cfg(), key, copy.deepcopy(value) if value is not _DEL else _DEL)
    with pytest.raises(ConfigError) as err:
        build_setup(cfg)
    hits = [m for p, m in err.value.errors if p == path]
    print(f"HY7 {case}: {err.value.errors}")
    assert hits, err.value.errors
    assert any(part.lower() in m.lower() for m in hits), (part, hits)
    assert all("history" in m for _, m in err.value.errors)


def test_hy7_all_problems_are_listed_at_once() -> None:
    cfg = _ve_cfg()
    cfg["analysis"]["plane"] = "stress"
    cfg["domain"]["generator"] = "box"
    cfg["objective"] = {"compliance_at": 7.0}
    cfg["bogus"] = 1
    with pytest.raises(ConfigError) as err:
        build_setup(cfg)
    paths = {p for p, _ in err.value.errors}
    assert {"analysis.plane", "domain.generator", "objective.compliance_at", "bogus"} <= paths


def test_hy7_unknown_key_gets_a_suggestion() -> None:
    cfg = _set(_ve_cfg(), "optimzer", {"type": "mma"})
    with pytest.raises(ConfigError) as err:
        build_setup(cfg)
    assert any(p == "optimzer" and 'did you mean "optimizer"' in m for p, m in err.value.errors)


# ------------------------------------------------------------------------------------------------ HY8
@pytest.mark.parametrize("example", [VE_EXAMPLE, CR_EXAMPLE], ids=["viscoelastic", "creep"])
def test_hy8_resolve_is_idempotent_and_examples_validate(example: Path) -> None:
    r1 = resolve(read_config(example))
    r2 = resolve(copy.deepcopy(r1))
    assert r1 == r2
    assert isinstance(r1["material"], dict) and r1["units"] == {"length": "m", "force": "N", "time": "s"}
    s = build_setup(read_config(example))
    assert isinstance(s, HistorySetup) and s.card.synthetic
    s2 = build_setup(yaml.safe_load(yaml.safe_dump(r1)))
    w = torch.full((s.structure.ne,), 0.4, dtype=torch.float64)
    assert torch.equal(s.structure.response(w).compliance[:3], s2.structure.response(w).compliance[:3])


def test_hy8_resolved_run_config_rebuilds_the_same_problem(tmp_path: Path) -> None:
    p = _reduced(VE_EXAMPLE, tmp_path)
    folder = tmp_path / "run"
    main(["run", str(p), "--out", str(folder), "-q"])
    resolved = yaml.safe_load((folder / "config.resolved.yaml").read_text(encoding="utf-8"))
    a, b = build_setup(read_config(p)), build_setup(resolved)
    w = torch.full((a.structure.ne,), 0.4, dtype=torch.float64)
    assert torch.equal(a.structure.response(w).compliance, b.structure.response(w).compliance)
    assert "provenance" in resolved and resolved["analysis"] == {"type": "history", "plane": "strain"}


# ------------------------------------------------------------------------------------------------ HY9
def test_hy9_other_paths_are_untouched() -> None:
    grid = {"name": "g", "domain": {"type": "box", "shape": [12, 4], "h": 0.25}, "material": {"E0": 1.0, "nu": 0.3},
            "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
            "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point",
                                                                                 "at": ["max", "min"]},
                                                      "force": [0, -1]}]}],
            "chain": [{"type": "density_filter", "rmin": 2.0}], "objective": "compliance",
            "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma", "run": {"max_iter": 3}}
    mesh = read_config(REPO / "examples" / "research" / "mesh_cantilever2d.yaml")
    assert not is_history_config(grid) and not is_history_config(mesh) and is_history_config(read_config(VE_EXAMPLE))
    assert resolve(grid)["domain"]["type"] == "box" and resolve(mesh)["domain"]["type"] == "mesh"
    assert type(build_setup(mesh)) is Setup and type(build_setup(grid)) is Setup
    # a typo in the grid path is still reported by the grid path, not swallowed by the history path
    with pytest.raises(ConfigError) as err:
        build_setup({**grid, "symmetry": {"x": 1.5}})
    assert any(p == "symmetry" and "history" not in m for p, m in err.value.errors)

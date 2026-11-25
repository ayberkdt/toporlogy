"""Fault injection (08 §2): bad input must fail closed, with an error that names what is wrong.

Pre-registered 2026-11-26 00:27 +0300 (``docs/vv_fault_injection.md`` §0), before the first run of any case here.
Each case gives an action, the exception types accepted and a regular expression the message must match
(case-insensitive). The expectation is "fails closed with a clear message": a silent success, an exception of
another type (``AttributeError``, ``IndexError``, a bare ``LinAlgError``) or a message that does not name the
culprit counts as a FAIL. A case that fails its pre-registered expectation stays as a strict xfail with the
observed behaviour in its reason (``docs/vv_fault_injection.md`` §1); the expectation is not changed afterwards.
"""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config
from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity

BASE: dict[str, Any] = {
    "domain": {"type": "box", "shape": [12, 6], "h": 1.0},
    "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
    "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]}, "force": [0, -1]}]}],
    "chain": [{"type": "density_filter", "rmin": 1.5}],
    "constraints": [{"response": "volume", "upper": 0.5}],
    "optimizer": "mma",
}
ORTHO = {"E1": 1.0, "E2": 0.5, "E3": 0.5, "G12": 0.3, "G13": 0.3, "G23": 0.2, "nu12": 0.3, "nu13": 0.3, "nu23": 0.3,
         "angle": 0.0}


def cfg(**changes: Any) -> dict[str, Any]:
    c = copy.deepcopy(BASE)
    for k, v in changes.items():
        c[k] = v
    return c


def with_force(force: list[float]) -> dict[str, Any]:
    c = cfg()
    c["load_cases"][0]["loads"][0]["force"] = force
    return c


def cantilever_model() -> tuple[Domain, Elasticity, np.ndarray]:
    d = Domain((8, 4), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    m = Elasticity(d, fixed, nu=0.3)
    F = np.zeros(d.ndof)
    F[d.dim * int(np.argmax(X[:, 0] + 1e-3 * X[:, 1])) + 1] = -1.0
    return d, m, F


def solve_with(E_mod: Callable[[np.ndarray], np.ndarray] | None = None,
               F_mod: Callable[[np.ndarray], np.ndarray] | None = None) -> Any:
    d, m, F = cantilever_model()
    E = np.ones(d.n_active)
    E = E if E_mod is None else E_mod(E)
    F = F if F_mod is None else F_mod(F)
    return m.solve_numpy(E, F)


def write_and_read(tmp: Path, text: str, name: str = "bad.yaml") -> Any:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return build_setup(read_config(p))


def _nan(E: np.ndarray) -> np.ndarray:
    E = E.copy()
    E[3] = math.nan
    return E


def _inf_load(F: np.ndarray) -> np.ndarray:
    F = F.copy()
    F[-1] = math.inf
    return F


def _neg_E(E: np.ndarray) -> np.ndarray:
    E = E.copy()
    E[0] = -1.0
    return E


def _wrong_len(E: np.ndarray) -> np.ndarray:
    return E[:-1]


def _wrong_rows(F: np.ndarray) -> np.ndarray:
    return F[:-2]


def _evaluate_nan_design(_: Path) -> Any:
    s = build_setup(cfg())
    x = s.study.default_start()
    x[0] = math.nan
    return s.study.evaluate(x)


def _zero_load(_: Path) -> Any:
    return build_setup(with_force([0, 0]))


def _no_support(_: Path) -> Any:
    c = cfg()
    c["supports"] = []
    return build_setup(c)


def _one_node_support(_: Path) -> Any:
    c = cfg()
    c["supports"] = [{"type": "fixed", "where": {"type": "point", "at": ["min", "min"]}}]
    return build_setup(c)


def _finished_folder(tmp: Path) -> Any:
    from toporlogy.execution.cli import main
    p = tmp / "mbb.yaml"
    import yaml
    c = cfg()
    c["run"] = {"max_iter": 2}
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    out = tmp / "run"
    first = main(["run", str(p), "--out", str(out), "-q"])
    second = main(["run", str(p), "--out", str(out), "-q"])
    raise RuntimeError(f"cli exit codes first={first} second={second}")


# (id, action(tmp_path), accepted exception types, message pattern)
CASES: list[tuple[str, Callable[[Path], Any], tuple[type[BaseException], ...], str]] = [
    # --- NaN / Inf -------------------------------------------------------------------------------------------
    ("nan_modulus_solve", lambda t: solve_with(E_mod=_nan), (ValueError,), r"finite|nan"),
    ("inf_load_solve", lambda t: solve_with(F_mod=_inf_load), (ValueError,), r"finite|inf"),
    ("nan_force_config", lambda t: build_setup(with_force([math.nan, -1.0])), (ConfigError,), r"finite|nan"),
    ("inf_E0_config", lambda t: build_setup(cfg(material={"E0": math.inf})), (ConfigError,), r"E0"),
    ("nan_design_evaluate", _evaluate_nan_design, (ValueError,), r"finite|nan"),
    # --- invalid (non-SPD) material ---------------------------------------------------------------------------
    ("negative_E0_config", lambda t: build_setup(cfg(material={"E0": -1.0})), (ConfigError,), r"E0"),
    ("zero_E0_config", lambda t: build_setup(cfg(material={"E0": 0.0})), (ConfigError,), r"E0"),
    ("nu_half_3d", lambda t: Elasticity(Domain((2, 2, 2)), [0, 1, 2], nu=0.5), (ValueError,),
     r"nu|poisson|positive definite"),
    ("nu_above_half_config", lambda t: build_setup(cfg(material={"nu": 0.7})), (ConfigError,),
     r"nu|poisson|positive definite"),
    ("nu_below_minus_one_config", lambda t: build_setup(cfg(material={"nu": -1.2})), (ConfigError,),
     r"nu|poisson|positive definite"),
    ("orthotropic_not_spd_config", lambda t: build_setup(cfg(material={"orthotropic": dict(ORTHO, nu12=2.5)})),
     (ConfigError,), r"positive definite|nu12|admissib"),
    ("negative_modulus_solve", lambda t: solve_with(E_mod=_neg_E), (ValueError,), r"modul|positive|E"),
    # --- inverted / degenerate element -------------------------------------------------------------------------
    ("negative_h_domain", lambda t: Domain((4, 4), h=-1.0), (ValueError,), r"h > 0"),
    ("negative_h_config", lambda t: build_setup(cfg(domain={"type": "box", "shape": [12, 6], "h": -1.0})),
     (ConfigError,), r"h"),
    ("negative_thickness_domain", lambda t: Elasticity(Domain((4, 4), thickness=-1.0), [0, 1, 2, 3]),
     (ValueError,), r"thickness"),
    # --- singular boundary conditions --------------------------------------------------------------------------
    ("no_supports_config", _no_support, (ConfigError,), r"support"),
    ("one_node_support_config", _one_node_support, (ConfigError,), r"rigid|mechanism|singular|rotation"),
    ("zero_load_config", _zero_load, (ConfigError,), r"load|zero"),
    # --- wrong shapes ------------------------------------------------------------------------------------------
    ("modulus_wrong_length_solve", lambda t: solve_with(E_mod=_wrong_len), (ValueError,), r"shape|length|active"),
    ("load_wrong_rows_solve", lambda t: solve_with(F_mod=_wrong_rows), (ValueError,), r"shape|rows|dof"),
    ("domain_shape_1d_config", lambda t: build_setup(cfg(domain={"type": "box", "shape": [12]})), (ConfigError,),
     r"shape"),
    ("force_3_components_2d", lambda t: build_setup(with_force([0, -1, 0])), (ConfigError,), r"component"),
    # --- wrong units / values ----------------------------------------------------------------------------------
    ("h_with_unit_text", lambda t: build_setup(cfg(domain={"type": "box", "shape": [12, 6], "h": "1 mm"})),
     (ConfigError,), r"number"),
    ("negative_rmin", lambda t: build_setup(cfg(chain=[{"type": "density_filter", "rmin": -1.5}])), (ConfigError,),
     r"rmin"),
    ("volume_upper_above_one", lambda t: build_setup(cfg(constraints=[{"response": "volume", "upper": 50.0}])),
     (ConfigError,), r"upper|fraction|volume"),
    ("negative_max_iter", lambda t: build_setup(cfg(run={"max_iter": -5})), (ConfigError,), r"max_iter"),
    ("negative_tol", lambda t: build_setup(cfg(run={"tol": -0.01})), (ConfigError,), r"tol"),
    # --- malformed input ---------------------------------------------------------------------------------------
    ("yaml_syntax_error", lambda t: write_and_read(t, "domain: {type: box, shape: [12, 6]\nsupports: ["),
     (ConfigError,), r"cannot parse"),
    ("yaml_top_level_list", lambda t: write_and_read(t, "- domain\n- supports\n"), (ConfigError,), r"mapping"),
    ("yaml_unknown_top_key", lambda t: build_setup(dict(cfg(), domian={"type": "box"})), (ConfigError,),
     r"domian"),
    ("yaml_duplicate_key", lambda t: write_and_read(
        t, "domain: {type: box, shape: [12, 6]}\ndomain: {type: box, shape: [4, 4]}\n"
           "supports: [{type: fixed, where: {type: plane, axis: x, at: min}}]\n"
           "load_cases: [{loads: [{type: force, where: {type: point, at: [max, min]}, force: [0, -1]}]}]\n"),
     (ConfigError,), r"duplicate"),
    ("missing_file", lambda t: read_config(t / "nope.yaml"), (ConfigError,), r"not found"),
    # --- stale run folder --------------------------------------------------------------------------------------
    ("rerun_into_finished_folder", _finished_folder, (RuntimeError,), r"first=\d+ second=2\b"),
]


@pytest.mark.parametrize(("action", "types", "pattern"), [pytest.param(a, t, p, id=i) for i, a, t, p in CASES])
def test_fails_closed_with_a_clear_message(action: Callable[[Path], Any], types: tuple[type[BaseException], ...],
                                           pattern: str, tmp_path: Path) -> None:
    try:
        out = action(tmp_path)
    except types as e:
        assert re.search(pattern, str(e), re.IGNORECASE), f"{type(e).__name__} without a clear message: {e}"
        return
    except Exception as e:  # noqa: BLE001 - the wrong exception type is the finding
        pytest.fail(f"wrong exception type {type(e).__name__}: {str(e)[:300]}")
    pytest.fail(f"accepted silently, returned {type(out).__name__}")

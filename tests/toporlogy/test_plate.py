"""Mindlin-Reissner plate element (docs/olcumler/plaka_onkayit.md, PL1-PL7)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.plate import MindlinPlate, hard_simply_supported, navier_deflection, plate_element_matrix

NU = 0.3


def test_pl1_rigid_modes() -> None:
    h, t = 0.7, 0.05
    K = plate_element_matrix(h, t, NU)
    c = np.array([[0, 0], [h, 0], [h, h], [0, h]], float)
    modes = []
    for w, bx, by in [(np.ones(4), 0 * c[:, 0], 0 * c[:, 0]), (c[:, 0], np.ones(4), 0 * c[:, 0]),
                      (c[:, 1], 0 * c[:, 0], np.ones(4))]:
        r = np.zeros(12)
        r[0::3], r[1::3], r[2::3] = w, bx, by
        modes.append(r)
    for r in modes:
        assert np.abs(K @ r).max() <= 1e-12 * np.abs(K).max()


def test_pl2_constant_curvature_patch() -> None:
    h, t = 0.5, 0.1
    K = plate_element_matrix(h, t, NU)
    c = np.array([[0, 0], [h, 0], [h, h], [0, h]], float) + np.array([1.3, -0.4])
    u = np.zeros(12)
    u[0::3], u[1::3] = c[:, 0] ** 2 / 2, c[:, 0]
    D11 = t ** 3 / (12 * (1 - NU ** 2))
    energy = 0.5 * u @ K @ u
    assert abs(energy - D11 * h * h / 2) <= 1e-12 * D11 * h * h / 2


def _centre(n: int, ratio: float, shear: str = "reduced") -> tuple[float, float]:
    a = 1.0
    t = a / ratio
    d = Domain((n, n), h=a / n)
    p = MindlinPlate(d, t, NU, hard_simply_supported(d), shear=shear)
    u = p.solve(torch.ones(d.n_active), p.pressure_load(1.0)).numpy()
    X = d.node_coords()
    c = int(np.argmin(np.abs(X[:, 0] - a / 2) + np.abs(X[:, 1] - a / 2)))
    return float(u[3 * c]), t


@pytest.mark.parametrize("ratio", [10.0, 100.0])
def test_pl3_navier_convergence(ratio) -> None:
    ref = navier_deflection(1.0, 1.0, 1.0 / ratio, 1.0, NU, 1.0, 0.5, 0.5)
    err = {n: abs(_centre(n, ratio)[0] - ref) / ref for n in (8, 16, 32)}
    order = np.log2(err[16] / err[32])
    print(f"PL3 a/t={ratio:g}: errors {err}, order 16->32 {order:.2f}, ref {ref:.6e}")
    assert err[32] <= 0.01
    assert order >= 1.5


def test_pl4_full_shear_integration_locks() -> None:
    ref = navier_deflection(1.0, 1.0, 1e-3, 1.0, NU, 1.0, 0.5, 0.5)
    w, _ = _centre(16, 1000.0, shear="full")
    print(f"PL4: full / Navier = {w / ref:.4f}")
    assert w <= 0.5 * ref


def test_pl5_thin_limit_kirchhoff() -> None:
    ref = navier_deflection(1.0, 1.0, 1e-3, 1.0, NU, 1.0, 0.5, 0.5, kirchhoff=True)
    w, _ = _centre(32, 1000.0)
    print(f"PL5: error {abs(w - ref) / ref:.3e}")
    assert abs(w - ref) <= 0.02 * ref


def test_pl6_design_gradient() -> None:
    d = Domain((10, 10), h=0.1)
    p = MindlinPlate(d, 0.02, NU, hard_simply_supported(d))
    F = p.pressure_load(1.0)
    x = 0.2 + 0.7 * np.random.default_rng(3).random(d.n_active)
    v = np.random.default_rng(4).normal(size=d.n_active)

    def comp(xx: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(F) @ p.solve(1e-6 + xx ** 3, F)
    xt = torch.tensor(x, requires_grad=True)
    (g,) = torch.autograd.grad(comp(xt), xt)
    s = 1e-6
    fd = (float(comp(torch.as_tensor(x + s * v))) - float(comp(torch.as_tensor(x - s * v)))) / (2 * s)
    print(f"PL6: rel {abs(float(g @ torch.as_tensor(v)) - fd) / abs(fd):.2e}")
    assert abs(float(g @ torch.as_tensor(v)) - fd) <= 1e-6 * abs(fd)


def test_pl7_refusals() -> None:
    with pytest.raises(ValueError, match="2-D"):
        MindlinPlate(Domain((3, 3, 3)), 0.1)
    with pytest.raises(ValueError, match="thickness"):
        MindlinPlate(Domain((3, 3)), 0.0)
    with pytest.raises(ValueError, match="Poisson"):
        MindlinPlate(Domain((3, 3)), 0.1, nu=0.5)


def test_navier_series_truncation_is_small() -> None:
    """Record helper (not pre-registered as a threshold): change of the series from 399 to 799 terms."""
    for ratio in (10.0, 100.0, 1000.0):
        a = navier_deflection(1.0, 1.0, 1 / ratio, 1.0, NU, 1.0, 0.5, 0.5, terms=399)
        b = navier_deflection(1.0, 1.0, 1 / ratio, 1.0, NU, 1.0, 0.5, 0.5, terms=799)
        print(f"truncation a/t={ratio:g}: {abs(a - b) / abs(b):.2e}")
        assert abs(a - b) <= 1e-3 * abs(b)


def _plate_study(n: int = 12, volume: float = 0.4):
    from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
    from toporlogy.design.interp import SIMP
    from toporlogy.fem.plate import PlatePhysics
    from toporlogy.optimization.responses import Compliance, Volume
    from toporlogy.optimization.study import Constraint, Study
    from toporlogy.optimizers.mma import MMA
    d = Domain((n, n), h=1.0 / n)
    plate = MindlinPlate(d, 0.02, NU, hard_simply_supported(d))
    F = plate.pressure_load(1.0)
    st = Study(d, PlatePhysics(plate), F, Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)]),
               SIMP(d, E0=1.0, penal=3.0, emin_ratio=1e-6), Compliance(), [Constraint(Volume(), volume)], MMA())
    return d, plate, F, st


def test_pq1_pq2_study_value_and_gradient() -> None:
    d, plate, F, st = _plate_study()
    x = 0.2 + 0.7 * np.random.default_rng(6).random(st.n_var)
    v = np.random.default_rng(7).normal(size=st.n_var)
    ev = st.evaluate(x)
    E = st.interp(torch.as_tensor(ev.xphys)).reshape(-1)[torch.as_tensor(d.active_elements)]
    c = float(torch.as_tensor(F) @ plate.solve(E, F))
    assert abs(ev.f0 - c) <= 1e-12 * abs(c)                                              # PQ1
    s = 1e-6
    fd = (st.evaluate(x + s * v, False).f0 - st.evaluate(x - s * v, False).f0) / (2 * s)
    print(f"PQ2: rel {abs(ev.df0 @ v - fd) / abs(fd):.2e}")
    assert abs(ev.df0 @ v - fd) <= 1e-6 * abs(fd)                                         # PQ2


def test_pq3_plate_optimisation_runs() -> None:
    d, plate, F, st = _plate_study(n=24)
    res = st.run(max_iter=30)
    c0 = res.history["f0"][0]
    print(f"PQ3: compliance {c0:.4e} -> {res.objective:.4e}, volume {res.constraint_values['volume']:.4f}")
    assert res.objective < c0
    assert res.constraint_values["volume"] <= 0.4 * (1 + 1e-3)


def test_pq4_wrong_load_rows_refused() -> None:
    from toporlogy.design.chain import Chain, Embed
    from toporlogy.design.interp import SIMP
    from toporlogy.fem.plate import PlatePhysics
    from toporlogy.optimization.responses import Compliance
    from toporlogy.optimization.study import Study
    from toporlogy.optimizers.mma import MMA
    d = Domain((4, 4))
    plate = MindlinPlate(d, 0.1, NU, hard_simply_supported(d))
    with pytest.raises(ValueError, match=f"{3 * d.nn} state DOFs"):
        Study(d, PlatePhysics(plate), np.zeros(d.ndof), Chain([Embed(d)]), SIMP(d), Compliance(), [], MMA())


def _yaml(n: int = 12, edge: str = "simply_supported", **phys) -> dict:
    p = {"type": "plate", "thickness": 0.02, "edge": edge, **phys}
    if "pressure" not in phys and "force" not in phys:
        p["pressure"] = 1.0
    return {"domain": {"type": "box", "shape": [n, n], "h": 1.0 / n}, "material": {"E0": 1.0, "nu": NU},
            "physics": p, "load_cases": [],
            "supports": [{"type": "fixed", "where": {"type": "plane", "axis": a, "at": e}}
                         for a in ("x", "y") for e in ("min", "max")],
            "chain": [{"type": "density_filter", "rmin": 1.5}], "objective": "compliance",
            "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma"}


def test_py1_yaml_equals_python_plate_study() -> None:
    from toporlogy.configuration.build import build_setup
    from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
    from toporlogy.design.interp import SIMP
    from toporlogy.fem.plate import PlatePhysics
    from toporlogy.optimization.responses import Compliance, Volume
    from toporlogy.optimization.study import Constraint, Study
    from toporlogy.optimizers.mma import MMA
    a = build_setup(_yaml()).study
    d = a.domain
    X = d.node_coords()
    edge = np.flatnonzero((X[:, 0] < 1e-9) | (X[:, 0] > 1 - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > 1 - 1e-9))
    plate = MindlinPlate(d, 0.02, NU, 3 * edge)
    F = plate.pressure_load(1.0)
    b = Study(d, PlatePhysics(plate), F, Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)]),
              SIMP(d, E0=1.0, penal=3.0, emin_ratio=a.interp.emin_ratio), Compliance(), [Constraint(Volume(), 0.4)],
              MMA())
    x = 0.2 + 0.7 * np.random.default_rng(8).random(a.n_var)
    ea, eb = a.evaluate(x), b.evaluate(x)
    print(f"PY1: value {abs(ea.f0 - eb.f0) / abs(eb.f0):.2e}, grad {np.abs(ea.df0 - eb.df0).max() / np.abs(eb.df0).max():.2e}"
          f", emin {a.interp.emin_ratio}")
    assert abs(ea.f0 - eb.f0) <= 1e-12 * abs(eb.f0)
    assert np.abs(ea.df0 - eb.df0).max() <= 1e-12 * np.abs(eb.df0).max()


def test_py2_yaml_clamped_point_load_runs() -> None:
    from toporlogy.configuration.build import build_setup
    st = build_setup(_yaml(16, "clamped", force=-1.0, at={"type": "point", "at": [0.5, 0.5]})).study
    res = st.run(max_iter=15)
    c0 = res.history["f0"][0]
    print(f"PY2: compliance {c0:.4e} -> {res.objective:.4e}, volume {res.constraint_values['volume']:.4f}")
    assert res.objective < c0 and res.constraint_values["volume"] <= 0.4 * (1 + 1e-3)


@pytest.mark.parametrize("change,match", [
    ("load_cases", "no load_cases"), ("partial", "fixed on every"), ("no_load", "needs a load"),
    ("thickness", "thickness"), ("three_d", "2-D"), ("force_no_at", "needs at")])
def test_py3_yaml_refusals(change, match) -> None:
    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.registry import ConfigError
    cfg = _yaml()
    if change == "load_cases":
        cfg["load_cases"] = [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "max"]},
                                         "force": [0, -1]}]}]
    elif change == "partial":
        cfg["supports"].append({"type": "roller", "where": {"type": "point", "at": [0.5, 0.5]}, "normal": [1, 0]})
    elif change == "no_load":
        cfg["physics"]["pressure"] = 0.0
    elif change == "thickness":
        cfg["physics"]["thickness"] = 0.0
    elif change == "three_d":
        cfg["domain"] = {"type": "box", "shape": [4, 4, 2], "h": 0.25}
        cfg["supports"] = [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}]
    else:
        cfg["physics"].pop("pressure")
        cfg["physics"]["force"] = -1.0
    with pytest.raises((ValueError, ConfigError), match=match):
        build_setup(cfg)

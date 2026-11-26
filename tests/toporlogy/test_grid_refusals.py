"""Grid-only components refuse a non-grid domain with a clear error (docs/olcumler/ag_topoloji_onkayit.md, RF1-RF3)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from toporlogy.core.domain import Domain, is_grid
from toporlogy.core.material import Strength
from toporlogy.design import manufacturing as mf
from toporlogy.design.chain import DensityFilter, Embed, HeavisideProjection, Passive, standard_chain
from toporlogy.design.interp import RAMP, SIMP
from toporlogy.design.orientation import PeriodicAngleFilter
from toporlogy.design.pde_filter import HelmholtzFilter
from toporlogy.design.phases import MultiPhaseSIMP, TensorPhases, phase_report
from toporlogy.fem.elements import D_iso, D_plane_stress
from toporlogy.fem.mesh import Mesh, rectangle
from toporlogy.fem.shell import FlatShell, cylinder_surface
from toporlogy.optimization import connectivity as cn
from toporlogy.optimization.dynamics import LinearMass, TwoRangeMass, TwoRangeSIMP
from toporlogy.optimization.lengthscale import GreyLevel, LocalVolume
from toporlogy.optimization.mass_responses import CentreOfMass, InertiaComponent
from toporlogy.optimization.mesh_study import MeshDomain, MeshElasticity, MeshFilterLink, MeshVolume
from toporlogy.optimization.responses import Compliance, Cost, EvalContext, Mass, Volume
from toporlogy.optimization.robust import RobustSet
from toporlogy.optimization.shell_study import ShellDomain, ShellPhysics
from toporlogy.optimization.strength import make_criterion
from toporlogy.optimization.stress import FailureIndex, VonMises
from toporlogy.optimization.stress_recovery import StressRecovery

PHASES = [{"name": "a", "E": 0.5}, {"name": "b", "E": 1.0}]


def _domains() -> dict[str, Any]:
    return {"mesh2d": MeshDomain(rectangle(8, 4, 2.0, 1.0, "Q4")),
            "mesh3d": MeshDomain(Mesh.from_domain(Domain((4, 3, 2)))),
            "shell": ShellDomain(cylinder_surface(1.0, 1.0, 8, 6, angle=np.pi / 2))}


DOMAINS = _domains()


def _x(d: Any) -> torch.Tensor:
    return torch.full((d.n_design,), 0.5, dtype=torch.float64)


def _ctx(d: Any, chain: Any = None, fields: Any = None) -> EvalContext:
    x = _x(d)
    return EvalContext(d, x, x.clone(), torch.zeros((d.ndof, 1), dtype=torch.float64), np.zeros((d.ndof, 1)),
                       interp=SIMP(d), chain=chain, fields=fields or [])


def _ifc(d: Any) -> dict[str, np.ndarray]:
    a, b = np.zeros(d.shape, bool), np.zeros(d.shape, bool)
    a[0], b[-1] = True, True
    return {"a": a, "b": b}


def _recovery(d: Any) -> Any:
    if isinstance(d, ShellDomain):                       # the shell physics has no grid domain at all
        return StressRecovery(ShellPhysics(FlatShell(d.mesh, 0.05, 0.3, np.arange(6))), np.zeros(d.ndof))
    D = D_plane_stress(0.3) if d.dim == 2 else D_iso(0.3)
    return StressRecovery(MeshElasticity(d, D, np.arange(d.dim)), np.zeros(d.ndof))


def _robust(d: Any) -> Any:
    from toporlogy.design.chain import Chain
    ch = Chain([Embed(d), MeshFilterLink(d, 0.3), HeavisideProjection(2.0), Passive(d)])
    fields, v = [], _x(d)
    for link in ch.links:
        v = link(v)
        fields.append(v)
    return RobustSet().realisation(_ctx(d, ch, fields), "eroded")


TYING = "a Tying with explicit element maps"
CHAIN = "a Chain of Embed, MeshFilterLink, Passive"
# component name in the message -> (call on a domain, alternative named in the message or None)
REFUSED: dict[str, tuple[Callable[[Any], Any], str | None]] = {
    "DensityFilter": (lambda d: DensityFilter(d, 1.5), "MeshFilterLink"),
    "standard_chain": (lambda d: standard_chain(d, 1.5), CHAIN),
    "manufacturing_chain": (lambda d: mf.manufacturing_chain(d, 1.5), CHAIN),
    "HelmholtzFilter": (lambda d: HelmholtzFilter(d, 1.5), "MeshFilterLink"),
    "PeriodicAngleFilter": (lambda d: PeriodicAngleFilter(d, 1.5), None),
    "AMFilter": (lambda d: mf.AMFilter(d), None),
    "Casting": (lambda d: mf.Casting(d, "+y"), None),
    "SymmetryTying": (lambda d: mf.SymmetryTying(d, {"x": 1.0}), TYING),
    "PatternTying.translational": (lambda d: mf.PatternTying.translational(d, [0.5] + [0.0] * (d.dim - 1)), TYING),
    "PatternTying.cyclic": (lambda d: mf.PatternTying.cyclic(d, 4, [0.0] * d.dim), TYING),
    "mirror_closure": (lambda d: mf.mirror_closure(d, np.zeros(d.shape, bool), [("x", 1.0)]), None),
    "ExtrusionTying": (lambda d: mf.ExtrusionTying(d, "y"), TYING),
    "overhang_map": (lambda d: mf.overhang_map(_x(d).numpy(), domain=d), None),
    "overhang_violations": (lambda d: mf.overhang_violations(_x(d).numpy(), domain=d), None),
    "undercut_map": (lambda d: mf.undercut_map(_x(d).numpy(), "+y", domain=d), None),
    "undercuts": (lambda d: mf.undercuts(_x(d).numpy(), "+y", domain=d), None),
    "MultiPhaseSIMP": (lambda d: MultiPhaseSIMP(d, PHASES), None),
    "TensorPhases": (lambda d: TensorPhases(d, PHASES), None),
    "phase_report": (lambda d: phase_report(MultiPhaseSIMP(Domain((2, 2)), PHASES), np.zeros((2, 4)), d), None),
    "Volume": (lambda d: Volume()(_ctx(d)), "MeshVolume"),
    "Mass": (lambda d: Mass()(_ctx(d)), None),
    "Cost": (lambda d: Cost()(_ctx(d)), None),
    "LocalVolume": (lambda d: LocalVolume(2.0)(_ctx(d)), None),
    "VonMises": (lambda d: VonMises(d, 1.0), None),
    "FailureIndex": (lambda d: FailureIndex(d, make_criterion("von_mises", Strength(1.0))), None),
    "StressRecovery": (_recovery, None),
    "ConductionConnectivity": (lambda d: cn.ConductionConnectivity(d, _ifc(d)), None),
    "CouplingCompliance": (lambda d: cn.CouplingCompliance(d, _ifc(d)), None),
    "analyse_connectivity": (lambda d: cn.analyse_connectivity(d, _x(d).numpy(), _ifc(d)), None),
    "tie_regions": (lambda d: cn.tie_regions(d, _ifc(d)), None),
    "bridge": (lambda d: cn.bridge(d, _x(d).numpy(), _ifc(d)), None),
    "one_body": (lambda d: cn.one_body(d, _ifc(d)), None),
    "CentreOfMass": (lambda d: CentreOfMass()(_ctx(d)), None),
    "InertiaComponent": (lambda d: InertiaComponent()(_ctx(d)), None),
    "RobustSet": (_robust, None),
}


@pytest.mark.parametrize("where", list(DOMAINS))
@pytest.mark.parametrize("component", list(REFUSED))
def test_rf1_grid_only_component_refuses_non_grid_domain(component: str, where: str) -> None:
    call, alternative = REFUSED[component]
    d = DOMAINS[where]
    with pytest.raises(ValueError) as err:
        call(d)
    msg = str(err.value)
    who = "ShellPhysics" if (component, where) == ("StressRecovery", "shell") else type(d).__name__
    assert component in msg and "grid" in msg and who in msg, msg
    assert (alternative in msg) if alternative else ("no mesh or shell counterpart" in msg), msg


KEEP: dict[str, Callable[[Any], Any]] = {
    "Embed": lambda d: Embed(d)(_x(d)),
    "Passive": lambda d: Passive(d)(_x(d)),
    "HeavisideProjection": lambda d: HeavisideProjection(4.0)(_x(d)),
    "SIMP": lambda d: SIMP(d)(_x(d)),
    "RAMP": lambda d: RAMP(d)(_x(d)),
    "LinearMass": lambda d: LinearMass(d)(_x(d)),
    "TwoRangeMass": lambda d: TwoRangeMass(d)(_x(d)),
    "TwoRangeSIMP": lambda d: TwoRangeSIMP(d)(_x(d)),
    "MeshFilterLink": lambda d: MeshFilterLink(d, 0.3)(_x(d)),
    "MeshVolume": lambda d: MeshVolume(d)(_ctx(d)),
    "Compliance": lambda d: Compliance()(_ctx(d)),
    "GreyLevel": lambda d: GreyLevel()(_ctx(d)),
    "Tying(explicit maps)": lambda d: mf.Tying(d, [np.arange(d.ne)])(torch.full((d.n_design,), 0.5,
                                                                               dtype=torch.float64)),
}


@pytest.mark.parametrize("where", list(DOMAINS))
@pytest.mark.parametrize("component", list(KEEP))
def test_rf2_components_that_work_on_non_grid_domains_keep_working(component: str, where: str) -> None:
    d = DOMAINS[where]
    out = KEEP[component](d)
    assert torch.isfinite(torch.as_tensor(out)).all()


def test_rf3_is_grid_truth_table() -> None:
    frac = np.ones((5, 4))
    frac[0, 0] = 0.0
    frac[1, 1] = 0.5
    keep, void = np.zeros((5, 4), bool), np.zeros((5, 4), bool)
    keep[2, 2], void[3, 3] = True, True
    grids = [Domain((5, 4)), Domain((3, 2, 2)), Domain((5, 4), frac=frac, keep=keep, void=void)]
    table = {repr(g): is_grid(g) for g in grids} | {k: is_grid(d) for k, d in DOMAINS.items()}
    print("RF3:", table)
    assert all(is_grid(g) for g in grids)
    assert not any(is_grid(d) for d in DOMAINS.values())

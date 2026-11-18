"""Compatibility module: ``toporlogy.bc`` moved to the :mod:`toporlogy.boundary` package (docs/gelistirme_plani.md
G4). Every name that was importable from it, private ones included, still is and refers to the same object; new
code imports ``toporlogy.boundary`` (same public names) or its modules: ``part``, ``selectors``, ``supports``,
``constrained``, ``loads``, ``checks``, ``cases``.
"""
from __future__ import annotations

from toporlogy.boundary.part import (  # noqa: F401
    EmptySelectionError, PartInfo, _INFO, _cross, _mom, _vec, part_info,
)
from toporlogy.boundary.selectors import (  # noqa: F401
    AllNodes, AtPoints, BoreFace, HalfSpaceNodes, InBox, InShape, InSphere, Nearest, NodeIds, OnCylinder, OnPlane,
    Selector, SurfaceNodes, Where, _SetOp, _cyl_coords, _facing_ok,
)
from toporlogy.boundary.supports import (  # noqa: F401
    ElasticSupport, Fixed, PinnedBore, Roller, Support, Symmetry,
)
from toporlogy.boundary.constrained import (  # noqa: F401
    ConstrainedElasticity, ConstraintSet, PerCaseSupports, _constraint_key, _resolve, reaction_forces,
)
from toporlogy.boundary.loads import (  # noqa: F401
    Applied, BearingLoad, BodyForce, BodyForceCompliance, Load, NodalForces, PointLoad, Pressure, RemoteForce,
    RemoteMoment, SurfaceForce, Torque, Traction, _LoadSolve, _bearing_vectors, _cylinder_samples, _scatter,
    _weights, body_force_matrix, design_load_solve, rigid_distribution,
)
from toporlogy.boundary.checks import (  # noqa: F401
    CheckReport, Issue, _decimals, _describe_mode, _fmt_vec, _format_loads, _rb_basis, _table,
)
from toporlogy.boundary.cases import (  # noqa: F401
    BoundaryConditions, Combination, LoadCase, clearance_mask,
)

__all__ = [
    # selectors
    "Selector", "AllNodes", "SurfaceNodes", "InBox", "OnPlane", "HalfSpaceNodes", "OnCylinder", "BoreFace",
    "InSphere", "InShape", "AtPoints", "Nearest", "NodeIds", "Where", "EmptySelectionError",
    # supports
    "Support", "Fixed", "Roller", "Symmetry", "PinnedBore", "ElasticSupport",
    # loads
    "Load", "Applied", "NodalForces", "PointLoad", "SurfaceForce", "Traction", "Pressure", "BearingLoad",
    "RemoteForce", "RemoteMoment", "Torque", "BodyForce",
    # cases, physics, checks
    "LoadCase", "Combination", "BoundaryConditions", "ConstraintSet", "ConstrainedElasticity", "PerCaseSupports",
    "Issue", "CheckReport", "part_info", "PartInfo", "rigid_distribution", "reaction_forces",
    "body_force_matrix", "design_load_solve", "BodyForceCompliance", "clearance_mask",
]

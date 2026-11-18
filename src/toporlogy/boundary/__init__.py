"""Boundary conditions by geometry: node selectors, supports, loads, load cases, non-design pads and checks.

Users never handle DOF numbers. A *selector* picks nodes of the part by geometry; a *support* constrains selected
nodes; a *load* turns an intended resultant into consistent nodal forces; :class:`BoundaryConditions` collects
named load cases (each may have its own supports), builds ``F`` (ndof, n_cases) and the physics, and checks
everything (:meth:`BoundaryConditions.check`, :meth:`BoundaryConditions.summary`).

Part surface and normals
------------------------
A node is on the surface when it touches at least one but fewer than 2^dim active elements (boundary of the active
elements; the box boundary counts). Exposed element faces (active element, inactive neighbour) carry area
``h²`` (3-D) or ``h·thickness`` (2-D) and an axis normal — the *staircase* surface. Each node also gets a smooth
outward normal ``n̂ = −m/|m|`` from the solid-fraction moment ``m = Σ_e frac_e (c_e − x) w(|c_e − x|)`` over the
4^dim surrounding elements (cone weight ``w = max(0, 2.5h − d)``); with partial volumes it follows curved surfaces
closely. The *geometric* area of a staircase face is ``A_f |n_f · n̂|`` (Σ_f n_f,k² over the staircase faces under
a smooth surface patch equals its area), used for area-weighted distributions and springs.

Selectors exclude what they should: :class:`OnCylinder` (bore face) keeps only surface nodes whose normal points
toward the axis within ``max_angle``, so the end faces at the bore mouths are not selected; :class:`OnPlane` keeps
only surface nodes whose normal is (anti)parallel to the plane normal.

Supports and how they enter the solver
--------------------------------------
Every support yields, per node, the *directions* in which the displacement is zero. Per node the directions of all
supports are combined (SVD, rank k): k = dim → all DOFs fixed; k axis-aligned directions → those DOFs fixed (plain
:class:`~toporlogy.fem.Elasticity`); otherwise the node gets a local orthonormal frame T (constrained directions
first) and the constrained local DOFs are removed after the transformation ``K' = Tᵀ K T`` (exact multi-point
constraint by transformation). Elastic supports add nodal spring blocks ``k A_i (n nᵀ | I | d dᵀ)`` to K. Both
need :class:`ConstrainedElasticity`, a subclass of ``Elasticity`` that overrides ``matrix`` and ``_solve_free``;
the autograd adjoint of the core works unchanged (springs do not depend on the design; ``U = T S⁻¹ Tᵀ F`` is
self-adjoint for compliance).

Loads (never a single node by default)
--------------------------------------
* :class:`PointLoad` — the force spread over the surface nodes within ``radius`` (default 1.5 h) with the
  rigid-body distribution below, so it is statically equivalent to the force at the point (zero moment about it);
* :class:`RemoteForce`, :class:`RemoteMoment`, :class:`Torque` — weighted least-squares distribution
  ``f_i = w_i (λ + μ × r_i)`` (minimises Σ|f_i|²/w_i under the resultant force and moment, a rigid-body field;
  the "distributed / deformable" remote load of commercial codes, source not read in this session);
* :class:`SurfaceForce` (total force ∝ geometric area), :class:`Traction` (force per area),
  :class:`Pressure` (−p n on staircase faces: a closed pressurised cavity gives exactly zero net force) — consistent
  nodal loads of a uniform traction on bilinear faces (A/4 per corner in 3-D, A/2 per end in 2-D);
* :class:`BearingLoad` — the radial part of the force over the loaded half of a bore with weight
  ``A_i max(0, cos θ_i)`` (projected-area cosine distribution, ported and generalised from
  the repository's earlier voxel tools);
  ``mode="parallel"`` (nodal forces parallel to the load) or ``"radial"`` (normal contact pressure ∝ cos θ). The
  documentation of the commercial "Bearing Load" was not read in this session;
* body forces: :func:`body_force_matrix` (design-independent or as a linear map of x̄) and the design-dependent
  response :class:`BodyForceCompliance` with :func:`design_load_solve` (adjoint through the load).

Every load records its intended force (and moment about a reference point when it prescribes one);
:meth:`BoundaryConditions.check` compares them with the assembled nodal forces to 1e-9 relative.

Per-load-case supports
----------------------
The core ``Elasticity`` has one set of fixed DOFs and ``Study`` takes one physics. :meth:`BoundaryConditions.physics`
returns a single physics when all cases share their supports, else a :class:`PerCaseSupports` wrapper that solves
each group of columns with its own physics and concatenates (autograd flows through each solve). Responses that run
extra solves through ``ctx.physics`` (adjoint-based stress etc.) must use the group physics.
"""
from __future__ import annotations

from toporlogy.boundary.part import (
    EmptySelectionError, PartInfo, part_info,
)
from toporlogy.boundary.selectors import (
    AllNodes, AtPoints, BoreFace, HalfSpaceNodes, InBox, InShape, InSphere, Nearest, NodeIds, OnCylinder, OnPlane,
    Selector, SurfaceNodes, Where,
)
from toporlogy.boundary.supports import (
    ElasticSupport, Fixed, PinnedBore, Roller, Support, Symmetry,
)
from toporlogy.boundary.constrained import (
    ConstrainedElasticity, ConstraintSet, PerCaseSupports, reaction_forces,
)
from toporlogy.boundary.loads import (
    Applied, BearingLoad, BodyForce, BodyForceCompliance, Load, NodalForces, PointLoad, Pressure, RemoteForce,
    RemoteMoment, SurfaceForce, Torque, Traction, body_force_matrix, design_load_solve, rigid_distribution,
)
from toporlogy.boundary.checks import (
    CheckReport, Issue,
)
from toporlogy.boundary.cases import (
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

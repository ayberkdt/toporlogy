"""General-mesh finite elements (03_FEM_ELEMENTS, research round 2): reference elements, quadrature, isoparametric
mapping with a signed detJ check, unstructured meshes with physical tags and boundary facets, sparse assembly with a
reusable pattern, essential/natural boundary conditions with reactions and a support-rank check, and a mixed u-p
(Taylor–Hood) formulation for nearly incompressible material.

Separate from the regular-grid fast path (:mod:`toporlogy.fem.elastic`), which is unchanged; :meth:`Mesh.from_domain`
turns a :class:`~toporlogy.core.domain.Domain` into a Q4 / H8 mesh with the same DOF numbers. Records:
``docs/olcumler/fem_genel_onkayit.md`` (pre-registration) and ``docs/olcumler/fem_genel.md`` (results).
"""
from __future__ import annotations

from toporlogy.fem.mesh.kernels import DegenerateElementError, InvertedElementError
from toporlogy.fem.mesh.mesh import Mesh, box, perturb, rectangle, straighten
from toporlogy.fem.mesh.mixed import MixedElasticity
from toporlogy.fem.mesh.problem import Dirichlet, ElasticProblem, HeatProblem, UnsupportedBoundaryCondition
from toporlogy.fem.mesh.reference import ELEMENTS, get

__all__ = ["Mesh", "rectangle", "box", "perturb", "straighten", "ElasticProblem", "HeatProblem", "Dirichlet",
           "MixedElasticity", "InvertedElementError", "DegenerateElementError", "UnsupportedBoundaryCondition",
           "ELEMENTS", "get"]

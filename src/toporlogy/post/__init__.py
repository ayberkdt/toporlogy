"""toporlogy.post — from an optimised density to a manufacturing-ready, verified surface.

Pipeline (:func:`finalize`, defaults in :class:`PostParams`): fine grid → normalised upsampling and blur →
volume-preserving threshold → sub-voxel signed distance → island removal → topology-safe adaptive opening and
closing → exact re-imposition of functional geometry (:class:`Keep`, :class:`Cut`, :class:`Bound` with
:class:`Cylinder`, :class:`Plane`, :class:`Box`, :class:`Sphere`, …) → flying-edges surface → Taubin λ|μ smoothing
with functional vertices projected back, tangential relaxation onto the implicit shape → report.

Then :func:`manufacturing_checks` (gate), :func:`reverify` (gmsh T10 + direct solve, change vs the voxel model) and
:func:`export_result` (STL/PLY/OBJ, print variant with machining stock, JSON report).

Import :mod:`toporlogy` first in a process that also uses torch (PARDISO before torch; see that package).
"""
from toporlogy.post.checks import CheckResult, CheckThresholds, sdf_local_thickness, manufacturing_checks, overhang
from toporlogy.post.export import export_result, read_stl, write_json, write_obj, write_ply, write_stl
from toporlogy.post.pipeline import (Implicit, PostParams, PostResult, finalize, measure_interfaces,
                                        optimised_volume, print_variant)
from toporlogy.post.primitives import (Bound, Box, Cut, Cylinder, GridPrimitive, ImplicitPrimitive, Keep,
                                          Plane, Primitive, Role, Sphere, assemble, element_fraction)
from toporlogy.post.verify import TetMesh, reverify, solve_t10, tet_mesh, voxel_reference

__all__ = ["finalize", "PostParams", "PostResult", "Implicit", "measure_interfaces", "optimised_volume",
           "print_variant", "Primitive", "Sphere", "Cylinder", "Plane", "Box", "GridPrimitive", "ImplicitPrimitive",
           "Role", "Keep", "Cut", "Bound", "assemble", "element_fraction", "manufacturing_checks", "CheckThresholds",
           "CheckResult", "sdf_local_thickness", "overhang", "export_result", "write_stl", "read_stl", "write_ply",
           "write_obj", "write_json", "tet_mesh", "solve_t10", "voxel_reference", "reverify", "TetMesh"]

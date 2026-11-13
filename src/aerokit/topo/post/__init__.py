"""aerokit.topo.post — from an optimised density to a manufacturing-ready, verified surface.

Pipeline (:func:`finalize`, defaults in :class:`PostParams`): fine grid → normalised upsampling and blur →
volume-preserving threshold → sub-voxel signed distance → island removal → topology-safe adaptive opening and
closing → exact re-imposition of functional geometry (:class:`Keep`, :class:`Cut`, :class:`Bound` with
:class:`Cylinder`, :class:`Plane`, :class:`Box`, :class:`Sphere`, …) → flying-edges surface → Taubin λ|μ smoothing
with functional vertices projected back, tangential relaxation onto the implicit shape → report.

Then :func:`manufacturing_checks` (gate), :func:`reverify` (gmsh T10 + direct solve, change vs the voxel model) and
:func:`export_result` (STL/PLY/OBJ, print variant with machining stock, JSON report).

Import :mod:`aerokit.topo` first in a process that also uses torch (PARDISO before torch; see that package).
"""
from aerokit.topo.post.checks import CheckResult, CheckThresholds, local_thickness, manufacturing_checks, overhang
from aerokit.topo.post.export import export_result, read_stl, write_json, write_obj, write_ply, write_stl
from aerokit.topo.post.pipeline import (Implicit, PostParams, PostResult, finalize, measure_interfaces,
                                        optimised_volume, print_variant)
from aerokit.topo.post.primitives import (Bound, Box, Cut, Cylinder, GridShape, Implicit as ImplicitShape, Keep,
                                          Plane, Role, Shape, Sphere, assemble, element_fraction)
from aerokit.topo.post.verify import TetMesh, reverify, solve_t10, tet_mesh, voxel_reference

__all__ = ["finalize", "PostParams", "PostResult", "Implicit", "measure_interfaces", "optimised_volume",
           "print_variant", "Shape", "Sphere", "Cylinder", "Plane", "Box", "GridShape", "ImplicitShape", "Role",
           "Keep", "Cut", "Bound", "assemble", "element_fraction", "manufacturing_checks", "CheckThresholds",
           "CheckResult", "local_thickness", "overhang", "export_result", "write_stl", "read_stl", "write_ply",
           "write_obj", "write_json", "tet_mesh", "solve_t10", "voxel_reference", "reverify", "TetMesh"]

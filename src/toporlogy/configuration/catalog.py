"""Scientific catalogs on top of the component registry: elements, physics/solvers, benchmarks, materials and the
compatibility matrix (08_VV_CATALOG_INFRA §3, first slice).

One source. Component names are the registry's (:mod:`toporlogy.configuration.registry`); literature keys are
:data:`toporlogy.references.REFERENCES`. This module adds only what the registry does not know: element and physics
facts, verification evidence (test node ids) and the support status of combinations. Tests check that every
registered physics, solver and response has an entry here and that every reference key and evidence test exists
(``tests/toporlogy/test_catalog.py``). The human-readable catalog and its JSON are generated from this module::

    python -m toporlogy.configuration.catalog --write docs      # docs/vv_katalog.md and docs/vv_katalog.json

Status of a combination (``STATUSES``):

* ``SUPPORTED``: a fast-suite test checks the combination against a reference (closed form, manufactured solution,
  independent implementation, external solver) or its gradient by finite differences / Taylor remainder, and passes;
* ``EXPERIMENTAL``: it runs and some test exercises it, but the check is partial (value or gradient only, slow suite
  only, or the same number by construction);
* ``UNSUPPORTED``: refused; the build (or the first evaluation, as recorded in ``enforced``) raises with a message
  that names the combination (``tests/toporlogy/test_compatibility.py``);
* ``NOT VERIFIED``: accepted by the build, no test known. Being registered does not make a combination supported.

The build calls :func:`check_build` before it constructs the :class:`~toporlogy.optimization.study.Study`; it
refuses the ``UNSUPPORTED`` rules whose ``enforced`` is ``"catalog"`` (those that would otherwise fail late or with an
unrelated error).
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from toporlogy.configuration.registry import ConfigError

__all__ = ["SCHEMA_VERSION", "STATUSES", "SUPPORTED", "EXPERIMENTAL", "UNSUPPORTED", "NOT_VERIFIED", "Element",
           "Physics", "Solver", "Benchmark", "MaterialRecord", "MATERIAL_KINDS", "validate_material",
           "material_section", "Cell", "Rule", "ELEMENTS", "PHYSICS", "SOLVERS",
           "BENCHMARKS", "MATERIALS", "RESPONSE_MATRIX", "RULES", "PHYSICS_INDEPENDENT", "COMPOSITE_RESPONSES",
           "status", "check_build", "as_dict", "render_markdown", "write"]

SCHEMA_VERSION = "1.1.0"         # semantic version of the catalog layout (fields), not of its content

SUPPORTED, EXPERIMENTAL, UNSUPPORTED, NOT_VERIFIED = "SUPPORTED", "EXPERIMENTAL", "UNSUPPORTED", "NOT VERIFIED"
STATUSES = (SUPPORTED, EXPERIMENTAL, UNSUPPORTED, NOT_VERIFIED)

T = "tests/toporlogy/"


# ---------------------------------------------------------------------------------------------- elements
@dataclass(frozen=True)
class Element:
    """A finite element of the library (``physics.element`` in a config, or a verification-only element)."""

    id: str
    dim: int
    geometry: str
    nodes: int
    dofs_per_node: int
    interpolation: str
    quadrature: str
    physics: tuple[str, ...]              # registry physics names that accept it ("post:reverify" = post step only)
    locking: str
    distortion: str
    references: tuple[str, ...]
    evidence: tuple[str, ...]             # test node ids (file::function)
    source: str                           # where it is implemented


_GRID = "regular grid, undistorted unit square/cube elements only (no Jacobian varies)"
_MESH = ("general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or "
         "degenerate cells refused; distorted-mesh patch test and MMS rates measured")
ELEMENTS: tuple[Element, ...] = (
    Element("Q4", 2, "square", 4, 2, "bilinear", "2 x 2 Gauss", ("elasticity", "modal", "nonlinear_elasticity",
                                                                  "tensor_elasticity", "thermoelastic"),
            "shear-locks in bending (parasitic shear)", _GRID, ("andreassen_2011", "cook_2002"),
            (T + "test_fem_mms.py::test_manufactured_solution_converges_at_second_order",
             T + "test_skfem_crosscheck.py::test_x1_stiffness",
             T + "test_independent_vv.py::test_top88_fixed_design_objective_and_sensitivity"),
            "fem/elements.py: ke_q4, ke_q4_from_D"),
    Element("Q4IM", 2, "square", 4, 2, "bilinear + 2 incompatible modes per component (Q6), statically condensed",
            "2 x 2 Gauss", ("elasticity", "modal"), "removes bending shear locking on undistorted squares",
            _GRID, ("wilson_1973", "taylor_beresford_wilson_1976"),
            (T + "test_fem_mms_extended.py::test_m1_q4im_converges_at_second_order",
             T + "test_buckling_physics.py::test_gradient_q4im_with_modes_in_kg"),
            "fem/elements.py: ke_q4im_from_D"),
    Element("H8", 3, "cube", 8, 3, "trilinear", "2 x 2 x 2 Gauss",
            ("elasticity", "elasticity_gmg", "modal", "nonlinear_elasticity", "tensor_elasticity", "thermoelastic"),
            "shear-locks in bending", _GRID,
            ("liu_tovar_2014", "cook_2002"),
            (T + "test_fem_mms.py::test_manufactured_solution_converges_at_second_order",
             T + "test_regression.py::test_topo3d_box_no_projection",
             T + "test_gmg.py::test_gmg_matches_direct_masked_domain"),
            "fem/elements.py: ke_h8, ke_h8_from_D"),
    Element("H8IM", 3, "cube", 8, 3, "trilinear + 9 Wilson incompatible modes, statically condensed",
            "2 x 2 x 2 Gauss", ("elasticity", "elasticity_gmg", "modal"),
            "removes bending shear locking on undistorted cubes", _GRID,
            ("taylor_beresford_wilson_1976", "cook_2002"),
            (T + "test_regression.py::test_topo3d_h8im", T + "test_buckling.py::test_euler_column_3d_h8im",
             T + "test_gmg.py::test_gmg_matches_direct_masked_domain"),
            "fem/elements.py: ke_h8(element='H8IM')"),
    Element("T10", 3, "tetrahedron", 10, 3, "quadratic", "4-point rule", ("post:reverify",),
            "none expected for quadratic tetrahedra (not measured here)",
            "general gmsh meshes; inverted or degenerate Jacobians are refused", ("zienkiewicz_taylor_2013",
                                                                                 "keast_1986", "cook_2002"),
            (T + "test_post_verify.py::test_t10_patch_uniaxial_tension",
             T + "test_t10_convergence.py::test_mesh_convergence_needs_three_sizes"),
            "post/verify.py (verification of the finished surface only; needs gmsh)"),
    # general-mesh elements (03_FEM_ELEMENTS, docs/olcumler/fem_genel.md): Python API only, no YAML physics yet
    *(Element(f"mesh:{name}", dim, geom, nodes, dim, interp, quad, (), lock, _MESH, refs,
              (T + "test_mesh_elements.py::test_e4_completeness",
               T + "test_mesh_assembly.py::test_g2_affine_patch_test",
               T + "test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution",
               T + "test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh"),
              "fem/mesh (ElasticProblem)")
      for name, dim, geom, nodes, interp, quad, lock, refs in (
          ("Q4", 2, "quadrilateral", 4, "bilinear (isoparametric)", "2 x 2 Gauss",
           "shear-locks on elongated elements (R = 0.29 at aspect 5); volumetric locking in plane strain (L2 error "
           "0.94 at nu = 0.4999, n = 16)", ("cook_2002", "zienkiewicz_taylor_2013")),
          ("Q9", 2, "quadrilateral", 9, "biquadratic Lagrange (isoparametric)", "3 x 3 Gauss",
           "no shear locking measured (R >= 0.9997, aspect up to 5); partial volumetric locking (L2 error 0.013 at "
           "nu = 0.4999 vs 3.5e-4 at 0.3)", ("zienkiewicz_taylor_2013",)),
          ("T3", 2, "triangle", 3, "linear", "centroid (exact for B^T D B)",
           "locks in bending and volumetrically (L2 error 0.97 at nu = 0.4999)", ("cook_2002",)),
          ("T6", 2, "triangle", 6, "quadratic", "3-point (degree 2)",
           "partial volumetric locking (L2 error 0.018 at nu = 0.4999)", ("zienkiewicz_taylor_2013",
                                                                          "dunavant_1985")),
          ("H8", 3, "hexahedron", 8, "trilinear (isoparametric)", "2 x 2 x 2 Gauss",
           "shear-locks on elongated elements (R = 0.24 at aspect 5); volumetric locking", ("cook_2002",)),
          ("T4", 3, "tetrahedron", 4, "linear", "centroid (exact for B^T D B)",
           "locks volumetrically (L2 error 0.996 at nu = 0.4999, n = 6)", ("cook_2002",)),
          ("T10", 3, "tetrahedron", 10, "quadratic (gmsh node order)", "4-point Keast (degree 2)",
           "curved T10 fails the patch test with the 4-point rule (1.7e-4); a degree-3 rule passes",
           ("keast_1986", "zienkiewicz_taylor_2013")))),
    *(Element(f"mesh:{name}", dim, geom, nodes, dim, interp + " displacement + continuous linear pressure "
              "(Taylor-Hood u-p, sigma = 2 mu eps - p I, p = -lambda div u)", quad, (),
              "locking-free near and at nu = 0.5 (L1c, L1e); inf-sup constant bounded under refinement (L2)",
              _MESH + "; plane strain and 3-D only (plane stress refused), lambda = 0 refused",
              ("zienkiewicz_taylor_2013",),
              (T + "test_mesh_locking.py::test_l1c_taylor_hood_does_not_lock",
               T + "test_mesh_locking.py::test_l1e_taylor_hood_converges_near_and_at_incompressibility",
               T + "test_mesh_locking.py::test_l2_taylor_hood_inf_sup_is_bounded"),
              "fem/mesh/mixed.py (MixedElasticity)")
      for name, dim, geom, nodes, interp, quad in (
          ("TH-T6T3", 2, "triangle", 6, "quadratic", "3-point (degree 2)"),
          ("TH-Q9Q4", 2, "quadrilateral", 9, "biquadratic", "3 x 3 Gauss"),
          ("TH-T10T4", 3, "tetrahedron", 10, "quadratic", "4-point Keast (degree 2)"))),
)


# ---------------------------------------------------------------------------------------------- physics, solvers
@dataclass(frozen=True)
class Physics:
    """A registered physics (registry kind ``physics``)."""

    id: str
    residual: str
    tangent: str
    matrix: str                           # symmetry / definiteness of the system matrix
    solve: str
    devices: tuple[str, ...]
    dims: tuple[int, ...]
    elements: tuple[str, ...]
    supports: str                         # support kinds accepted
    convergence: str
    memory: str
    evidence: tuple[str, ...]
    scalar_modulus: str                   # how the design enters (the scalar-E contract)


PHYSICS: tuple[Physics, ...] = (
    Physics("elasticity", "K(E) u = f (linear, small strain)", "K(E), linear in the element moduli",
            "symmetric positive definite on the free DOFs (refuses floating parts)", "direct (solver: direct)",
            ("cpu",), (2, 3), ("Q4", "Q4IM", "H8", "H8IM"),
            "fixed DOFs; oblique rollers and springs (ConstrainedElasticity); per-case supports (PerCaseSupports)",
            "direct solve; relative residual <= 1e-6 or backward error <= 1e-14 accepted, else SolveResidualError",
            "assembled sparse K and its factor", (T + "test_gradients.py::test_compliance_single_case",
                                                    T + "test_skfem_crosscheck.py::test_x4_cantilever_solution",
                                                    T + "test_fem_mms.py::test_manufactured_solution_converges_at_"
                                                        "second_order"),
            "E_active (n_active,) scales one unit-modulus D per element (D may differ per element)"),
    Physics("elasticity_gmg", "K(E) u = f (linear, small strain)", "matrix-free K(E)",
            "symmetric positive definite", "geometric multigrid preconditioned CG; direct below direct_below DOFs",
            ("cpu", "cuda"), (3,), ("H8", "H8IM"), "axis-aligned fixed DOFs, one support set",
            "PCG to tol (default 1e-8); fail-closed without fallback, logged fallback to the direct solver",
            "matrix-free; multigrid hierarchy", (T + "test_gmg.py::test_gmg_matches_direct_masked_domain",
                                                 T + "test_gmg.py::test_gradients",
                                                 T + "test_gmg.py::test_fail_closed_without_fallback"),
            "E_active scales ONE unit D (per-element D refused)"),
    Physics("modal", "K(E) phi = lambda M(x) phi (generalised eigenproblem)", "K(E), M(x)",
            "K SPD on the free DOFs, M SPD; repeated eigenvalues detected (KS aggregation)",
            "shift-invert eigensolver (sigma) with the direct solver", ("cpu",), (2, 3), ("Q4", "Q4IM", "H8", "H8IM"),
            "axis-aligned fixed DOFs, one support set; no static state u",
            "eigensolver tolerance (not recorded per run)", "assembled K and M",
            (T + "test_modal.py::test_eigenpairs_match_dense", T + "test_independent_matrices.py::test_k4_modal_"
                                                                   "eigenvalues",
             T + "test_modal.py::test_response_gradients"),
            "E_active scales K; mass from the density (linear or two-range)"),
    Physics("nonlinear_elasticity", "r(u; E) = f_int(u) - f = 0 (total Lagrangian, SVK or neo-Hooke; API: "
                                    "small-strain models of core.constitutive incl. J2 plasticity, no sensitivity)",
            "consistent tangent K_T(u)", "symmetric; positive definite only on the stable branch",
            "general Newton (solvers.newton) with load steps; arc length / displacement control through "
            "continuation_problem (API); one adjoint solve with K_T per case", ("cpu",), (2, 3), ("Q4", "H8"),
            "axis-aligned fixed DOFs, one support set", "Newton residual and error contraction; "
                                                        "NonlinearConvergenceError with the history",
            "assembled K_T per Newton step", (T + "test_nonlinear.py::test_v3_consistent_tangent_finite_difference",
                                               T + "test_nonlinear.py::test_v5_large_deflection_cantilever_elastica",
                                               T + "test_nonlinear.py::test_v6_end_compliance_gradient_finite_"
                                                   "difference"),
            "E_active through an energy interpolation; general D with material svk only"),
    Physics("tensor_elasticity", "K(C) u = f (linear, small strain)",
            "K(C), linear in the element tensors C_e (Q4, H8: KE(C) = sum_a c_a KE(S_a))",
            "symmetric positive definite on the free DOFs (every C_e a convex combination of SPD tensors)",
            "direct (solver: direct)", ("cpu",), (2, 3), ("Q4", "H8"), "axis-aligned fixed DOFs, one support set",
            "direct solve; relative residual <= 1e-6 or backward error <= 1e-14 accepted, else SolveResidualError",
            "assembled sparse K, element matrices (n_active, nloc, nloc) refilled per evaluation",
            (T + "test_material_design.py::test_k8b_scalar_limit_matches_multiphase_simp",
             T + "test_material_design.py::test_k7_independent_assembly",
             T + "test_material_design.py::test_g1_compliance_gradient_finite_difference"),
            "per-element elasticity tensor C_e (MaterialField of interpolation tensor_phases); the scalar grid the "
            "study passes is only checked against the field"),
    Physics("thermoelastic", "K(E) u = f + f_th(E), f_th = sum_e E_e G dT_e (one-way, small strain, prescribed "
                             "uniform temperature rise)", "K(E) and f_th(E), both linear in the element moduli",
            "symmetric positive definite on the free DOFs", "direct (GridSPD supernodal / SuperLU factor)",
            ("cpu",), (2, 3), ("Q4", "H8"), "axis-aligned fixed DOFs, one support set",
            "direct solve (GridSPD)", "assembled sparse K and its factor",
            (T + "test_thermoelastic_study.py::test_ts2_value_equals_direct_thermoelastic",
             T + "test_thermoelastic_study.py::test_ts3_gradient_finite_difference",
             T + "test_thermoelastic.py::test_te1_free_expansion"),
            "E_active scales ONE unit D (per-element D refused) and the thermal load; compliance is F_totᵀ U"),
)


@dataclass(frozen=True)
class Solver:
    """A registered linear solver (registry kind ``solver``) or a solver backend behind one."""

    id: str
    registry: bool                        # registered under kind "solver"
    method: str
    backends: tuple[str, ...]
    status: str
    note: str


SOLVERS: tuple[Solver, ...] = (
    Solver("direct", True, "sparse Cholesky / LU of the upper triangle", ("SuperLU (scipy)", "PARDISO (pypardiso)"),
           SUPPORTED, "SuperLU always available; PARDISO optional (fast extra), VERIFIED only where the pardiso tests "
                      "ran (backends.yml)"),
    Solver("gmg", False, "geometric multigrid PCG (inside physics elasticity_gmg)", ("torch CPU", "torch CUDA"),
           EXPERIMENTAL, "CPU path tested in the fast suite; CUDA path NOT VERIFIED without a GPU"),
)


# ---------------------------------------------------------------------------------------------- benchmarks
BENCHMARK_KINDS = ("analytic", "mms", "published", "internal_oracle", "independent_same_author", "external_solver",
                   "gradient", "experimental")


@dataclass(frozen=True)
class Benchmark:
    """A benchmark: a test that compares the library against a reference."""

    id: str
    kind: str                             # one of BENCHMARK_KINDS
    test: str                             # test node id (file::function)
    quantity: str
    reference: str                        # REFERENCES key, oracle module or "closed form"
    tolerance: str
    runtime: str                          # fast | slow
    validation: str = "code verification" # code verification | solution verification | validation
    note: str = ""


_BENCHMARK_ROWS: tuple[tuple[str, ...], ...] = (
    (
        'mms_q4_h8',
        'mms',
        'test_fem_mms.py::test_manufactured_solution_converges_at_second_order',
        'nodal error, observed order (Q4, H8, H8IM)',
        'manufactured solution (docs/gelistirme_plani.md V1)',
        'rates in [1.8, 2.25]; finest e < 1e-3 / 1e-2 / 1.2e-2',
        'fast',
        'solution verification',
    ),
    (
        'mms_wrong_constant',
        'mms',
        'test_fem_mms.py::test_manufactured_solution_detects_a_wrong_material_constant',
        'order with a wrong material constant (negative control)',
        'manufactured solution',
        'last rate < 1.0, error > 5x',
        'fast',
        'solution verification',
    ),
    (
        'mms_q4im',
        'mms',
        'test_fem_mms_extended.py::test_m1_q4im_converges_at_second_order',
        'Q4IM nodal error order',
        'manufactured solution (dogrulama_turu M1)',
        'rates in (1.8, 2.25); finest <= 1e-3',
        'fast',
        'solution verification',
    ),
    (
        'mms_orientation_field',
        'mms',
        'test_fem_mms_extended.py::test_m2_orientation_field_converges_at_second_order',
        'orthotropic orientation field order',
        'manufactured solution (M2)',
        'rates in (1.8, 2.25); finest <= 5e-3 / 5e-2',
        'fast',
        'solution verification',
    ),
    (
        'mms_heat',
        'mms',
        'test_fem_mms_extended.py::test_m3_heat_conduction_converges_at_second_order',
        'conduction order',
        'manufactured solution (M3)',
        'rates in (1.8, 2.25)',
        'fast',
        'solution verification',
    ),
    (
        'mms_nonlinear',
        'mms',
        'test_fem_mms_extended.py::test_m4_nonlinear_manufactured_solution_converges_at_second_order',
        'SVK / neo-Hooke order',
        'manufactured solution (M4)',
        'rates in (1.8, 2.25)',
        'fast (3-D cases slow)',
        'solution verification',
    ),
    (
        'skfem_stiffness',
        'external_solver',
        'test_skfem_crosscheck.py::test_x1_stiffness',
        'stiffness matrix',
        'scikit-fem',
        'relative <= 1e-12',
        'fast',
        'code verification',
    ),
    (
        'skfem_mass',
        'external_solver',
        'test_skfem_crosscheck.py::test_x2_mass',
        'consistent mass',
        'scikit-fem',
        'relative <= 1e-12',
        'fast',
        'code verification',
    ),
    (
        'skfem_conduction',
        'external_solver',
        'test_skfem_crosscheck.py::test_x3_conduction',
        'conduction matrix',
        'scikit-fem',
        'relative <= 1e-12',
        'fast',
        'code verification',
    ),
    (
        'skfem_cantilever',
        'external_solver',
        'test_skfem_crosscheck.py::test_x4_cantilever_solution',
        'cantilever displacement',
        'scikit-fem',
        'relative <= 1e-10',
        'fast',
        'code verification',
    ),
    (
        'top88_fixed_design',
        'internal_oracle',
        'test_independent_vv.py::test_top88_fixed_design_objective_and_sensitivity',
        'objective, sensitivity',
        'published_codes.py (Top88 port)',
        'c 1e-9, dc 1e-8',
        'fast',
        'code verification',
    ),
    (
        'top88_history',
        'internal_oracle',
        'test_independent_vv.py::test_top88_history_first_50_iterations',
        'compliance history, 50 iterations',
        'published_codes.py (Top88 port)',
        'c 1e-6, x 1e-5',
        'fast',
        'code verification',
    ),
    (
        'top3d_fixed_design',
        'internal_oracle',
        'test_independent_vv.py::test_top3d_fixed_design_objective_and_sensitivity',
        'objective, sensitivity',
        'published_codes.py (Top3D port)',
        'c 1e-9, dc 1e-8',
        'fast',
        'code verification',
    ),
    (
        'topo3d_regression',
        'internal_oracle',
        'test_regression.py::test_topo3d_box_no_projection',
        'compliance history, density',
        'topo_oracles.topo3d (frozen driver)',
        'c 1e-6, x 1e-6',
        'fast',
        'code verification',
    ),
    (
        'cantilever_exact',
        'analytic',
        'test_independent_vv.py::test_cantilever_q4_converges_to_the_exact_solution',
        'compliance bound and tip error order',
        'exact polynomial cantilever solution',
        'order >= 1.8; tip <= 2e-2',
        'fast',
        'solution verification',
    ),
    (
        'slip_square_modes',
        'analytic',
        'test_independent_vv.py::test_slip_square_q4_eigenvalues_converge_with_repeated_pairs',
        'eigenvalues with repeated pairs',
        'analytic slip-wall modes',
        'order >= 1.8; gaps <= 1e-8',
        'fast',
        'solution verification',
    ),
    (
        'independent_K',
        'independent_same_author',
        'test_independent_matrices.py::test_k1_k2_reduced_stiffness',
        'reduced stiffness (8 element/material variants)',
        '_independent_fe.py',
        'relative <= 1e-12',
        'fast',
        'code verification',
    ),
    (
        'independent_buckling',
        'independent_same_author',
        'test_independent_buckling.py::test_load_factors_equal_the_dense_independent_solution',
        'first six load factors',
        '_independent_fe.py (dense eigh)',
        '|dmu| <= 1e-9 |mu1|',
        'fast',
        'code verification',
    ),
    (
        'independent_stress',
        'independent_same_author',
        'test_independent_stress.py::test_s2_von_mises_index',
        'von Mises index',
        '_independent_fe.py',
        'relative <= 1e-12',
        'fast',
        'code verification',
    ),
    (
        'timoshenko_tip',
        'analytic',
        'test_timoshenko_cantilever.py::test_h4_tip_deflection',
        'tip deflection',
        'timoshenko_goodier_1970',
        'Q4 <= 2e-2, Q4IM <= 2e-3',
        'fast',
        'solution verification',
    ),
    (
        'timoshenko_order_q4',
        'analytic',
        'test_timoshenko_cantilever.py::test_h2_q4_converges_at_second_order',
        'nodal error order (strict xfail)',
        'timoshenko_goodier_1970',
        'rates in [1.8, 2.25]',
        'fast',
        'solution verification',
        'FAILED pre-registered (ledger)',
    ),
    (
        'fs2019_q4_column',
        'published',
        'test_fs2019_column.py::test_published_q4_error_20x2',
        'buckling error Q4 20x2',
        'ferrari_sigmund_2019 (Fig. 1 table)',
        '|e - 0.119| <= 5e-4',
        'fast',
        'code verification',
    ),
    (
        'fs2019_q6_column',
        'published',
        'test_fs2019_column.py::test_published_q6_error_20x2',
        'buckling error Q4IM vs published Q6 (strict xfail)',
        'ferrari_sigmund_2019',
        '|e + 6.6e-3| <= 5e-5',
        'fast',
        'code verification',
        'FAILED pre-registered (ledger)',
    ),
    (
        'euler_column_orthotropic',
        'analytic',
        'test_buckling_physics.py::test_orthotropic_column_buckles_with_its_axial_modulus',
        'column buckling load, refinement ratios',
        'Euler column',
        'err <= 5e-3; ratios >= 2.5',
        'fast',
        'solution verification',
    ),
    (
        'euler_bernoulli_modal_2d',
        'analytic',
        'test_modal.py::test_cantilever_vs_euler_bernoulli_2d',
        'first frequency ratio',
        'Euler-Bernoulli beam',
        '0.98 < r < 1.0, monotone',
        'fast',
        'solution verification',
    ),
    (
        'heat_series_bar',
        'analytic',
        'test_heat.py::test_bar_series_resistance_exact',
        'end temperature',
        'series resistance',
        'relative <= 1e-10',
        'fast',
        'code verification',
    ),
    (
        'nl_uniaxial',
        'analytic',
        'test_nonlinear.py::test_v2_uniaxial_tension_closed_form',
        'stretches under dead load',
        'closed form',
        '<= 1e-9',
        'fast',
        'code verification',
    ),
    (
        'nl_elastica',
        'independent_same_author',
        'test_nonlinear.py::test_v5_large_deflection_cantilever_elastica',
        'tip displacement',
        '_elastica.py',
        'v <= 5e-3, u <= 1e-2 at n = 16',
        'slow',
        'solution verification',
    ),
    (
        'deng_pnorm_after_mma',
        'published',
        'test_published_values.py::test_deng_cantilever_published_value_is_after_the_first_mma_step',
        'p-norm after one MMA step (post hoc)',
        'Deng, Vulimiri & To 2021 §6.2',
        'relative <= 5e-3',
        'fast',
        'code verification',
        'post-hoc reading of the paper',
    ),
    (
        'topbuck250_wall',
        'published',
        'test_published_values.py::test_topbuck250_wall_solid_blf_against_published_value',
        'solid wall load factor (strict xfail)',
        'ferrari_sigmund_guest_2021 §5.2',
        '|lambda - 3.15| <= 0.0051',
        'slow',
        'code verification',
        'FAILED (ledger)',
    ),
    (
        't10_mesh_convergence',
        'mms',
        'test_t10_convergence.py::test_the_same_cantilever_converges_on_fine_meshes_and_not_on_coarse_ones',
        'three-grid GCI verdict',
        'celik_2008',
        'CONVERGED on fine, NOT CONVERGED on coarse',
        'slow',
        'solution verification',
        'needs gmsh',
    ),
    (
        'taylor_t1',
        'gradient',
        'test_taylor_remainder.py::test_t1_taylor_orders',
        'Taylor remainder order of 11 responses',
        'docs/olcumler/dogrulama_turu.md T',
        'two consecutive p in (1.8, 2.2)',
        'fast',
        'code verification',
    ),
    (
        'gradients_compliance',
        'gradient',
        'test_gradients.py::test_compliance_single_case',
        'compliance gradient',
        'five-point finite difference',
        'relative <= 1e-6',
        'fast',
        'code verification',
    ),
    (
        'constitutive_ad_oracle',
        'internal_oracle',
        'test_constitutive.py::test_p1_energy_stress_tangent_match_the_ad_oracle',
        'SVK / neo-Hooke energy, PK2 stress, tangent (3-D, plane strain, plane stress)',
        'torch autograd of the energy (core.constitutive.energy_oracle)',
        'psi, S <= 1e-11; C <= 1e-10 relative',
        'fast',
        'code verification',
    ),
    (
        'j2_uniaxial_cyclic',
        'analytic',
        'test_constitutive.py::test_j2_cyclic_uniaxial_closed_form_and_elastic_unloading',
        'J2 uniaxial stress, cyclic, isotropic hardening',
        'closed form (1-D isotropic hardening, leg by leg)',
        'stress <= 1e-10 sigma_y; alpha <= 1e-10 eps_y',
        'fast',
        'code verification',
    ),
    (
        'j2_consistent_tangent',
        'gradient',
        'test_constitutive.py::test_j4_consistent_tangent_matches_finite_difference',
        'J2 algorithmic tangent (linear and Voce hardening)',
        'central finite difference of the return map',
        'relative <= 1e-6',
        'fast',
        'code verification',
    ),
    (
        'j2_fe_homogeneous_bar',
        'internal_oracle',
        'test_nonlinear_solvers.py::test_j7_homogeneous_bar_matches_the_material_point',
        'FE J2 bar (Q4 plane strain, H8): loading and unloading strain',
        'mixed-control material point driver (core.constitutive.drive_material_point)',
        'relative <= 1e-9',
        'fast',
        'code verification',
    ),
    (
        'truss_snap_through',
        'analytic',
        'test_nonlinear_solvers.py::test_a1_a3_arc_length_traces_the_truss_snap_through',
        'arc-length path of the shallow two-bar truss through both limit points',
        'closed form P(w) (tests/toporlogy/_truss.py)',
        '|lambda - P(w)/P_ref| <= 1e-9; det K sign 100 %',
        'fast',
        'code verification',
    ),
    (
        'fe_arc_length_vs_load_control',
        'internal_oracle',
        'test_nonlinear_solvers.py::test_a5_fe_arc_length_matches_load_control',
        'arc-length path of a stocky cantilever (SVK)',
        'load control at the same load factors',
        'relative <= 1e-7',
        'fast',
        'code verification',
    ),
    (
        'end_strain_energy_taylor',
        'gradient',
        'test_nonlinear_solvers.py::test_t1_t2_end_strain_energy_taylor_remainder',
        'end strain energy gradient (fixed cutoff)',
        'Taylor remainder order',
        'two consecutive orders in [1.8, 2.2]; x1.01 control <= 1.3',
        'fast',
        'code verification',
    ),
)
BENCHMARKS: tuple[Benchmark, ...] = tuple(Benchmark(r[0], r[1], T + r[2], *r[3:])
                                          for r in _BENCHMARK_ROWS)


# ---------------------------------------------------------------------------------------------- materials
@dataclass(frozen=True)
class MaterialRecord:
    """A material data record (08 §3). ``kind``: synthetic | nominal | typical | test_measured | design_allowable;
    ``family``: isotropic (values E0, nu, rho) | orthotropic (E1 … nu23, rho); ``units``: "dimensionless" (consistent
    user units, copied as they are) or "SI" (Pa, kg/m^3; converted into the config's ``units`` section).

    A record that is not synthetic needs a source, a page or table and a licence (:func:`validate_material`). No
    real material value is entered without having read its source and licence; the records below are synthetic.
    """

    id: str
    kind: str
    family: str
    values: Mapping[str, float]
    units: str
    source: str
    used_by: tuple[str, ...] = ()
    note: str = ""
    grade: str = ""
    condition: str = ""
    temperature: str = ""
    uncertainty: Mapping[str, str] = field(default_factory=dict)
    page_or_table: str = ""
    licence: str = ""
    applicability: str = ""


MATERIAL_KINDS = ("synthetic", "nominal", "typical", "test_measured", "design_allowable")
MATERIAL_UNITS = ("dimensionless", "SI")
_ISO_KEYS, _ORTHO_KEYS = ("E0", "nu"), ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23")
_MODULI = {"E0", "E1", "E2", "E3", "G12", "G13", "G23"}
_PRESSURE_DIM, _DENSITY_DIM = (-1, 1, -2, 0), (-3, 1, 0, 0)

MATERIALS: tuple[MaterialRecord, ...] = (
    MaterialRecord("synthetic_unit_isotropic", "synthetic", "isotropic", {"E0": 1.0, "nu": 0.3, "rho": 1.0},
                   "dimensionless", "schema default (configuration/schema.py _MATERIAL)",
                   ("examples/*.yaml without a material block",),
                   "not a real material; the default of every config"),
    MaterialRecord("synthetic_auxetic_isotropic", "synthetic", "isotropic", {"E0": 1.0, "nu": -0.5, "rho": 1.0},
                   "dimensionless", "chosen for tests (docs/olcumler/malzeme_katalog_onkayit.md C4)",
                   (T + "test_material_catalog.py",), "not a real material: a negative Poisson ratio inside (-1, 1/2)"),
    MaterialRecord("synthetic_orthotropic_ply", "synthetic", "orthotropic",
                   {"E1": 40.0, "E2": 9.0, "E3": 9.0, "G12": 4.0, "G13": 4.0, "G23": 3.2, "nu12": 0.28, "nu13": 0.28,
                    "nu23": 0.4, "rho": 1.0}, "dimensionless", "the README physics example (round numbers)",
                   ("README.md",), "not a real material: ratios of the order of a fibre ply"),
    MaterialRecord("synthetic_orthotropic_high_nu23", "synthetic", "orthotropic",
                   {"E1": 10.0, "E2": 1.0, "E3": 1.0, "G12": 0.5, "G13": 0.5, "G23": 0.3, "nu12": 0.05, "nu13": 0.05,
                    "nu23": 0.65, "rho": 1.0}, "dimensionless",
                   "chosen for tests (docs/olcumler/malzeme_katalog_onkayit.md C4)", (T + "test_material_catalog.py",),
                   "not a real material: positive definite with nu23 > 1/2 (anisotropic bounds differ from isotropic)"),
    MaterialRecord("synthetic_si_isotropic", "synthetic", "isotropic", {"E0": 2.0e11, "nu": 0.3, "rho": 8000.0},
                   "SI", "chosen for tests (docs/olcumler/malzeme_katalog_onkayit.md C6)",
                   (T + "test_material_catalog.py",),
                   "not a real material: round numbers of the order of a steel, in Pa and kg/m^3"),
)


def validate_material(m: MaterialRecord) -> None:
    """Refuse a record that is incomplete or physically inadmissible; the message names the field."""
    from toporlogy.core.linear_elastic import isotropic_moduli
    from toporlogy.core.material import Orthotropic
    if m.kind not in MATERIAL_KINDS:
        raise ValueError(f"{m.id}: kind {m.kind!r} (one of {', '.join(MATERIAL_KINDS)})")
    if m.units not in MATERIAL_UNITS:
        raise ValueError(f"{m.id}: units {m.units!r} (one of {', '.join(MATERIAL_UNITS)})")
    if not m.source:
        raise ValueError(f"{m.id}: source is required")
    if m.kind != "synthetic":
        for f in ("page_or_table", "licence"):
            if not getattr(m, f):
                raise ValueError(f"{m.id}: a {m.kind} record needs {f} (08 §3: traceable source, page and licence)")
    v = dict(m.values)
    if "rho" in v and not v["rho"] > 0:
        raise ValueError(f"{m.id}: rho > 0 is required")
    if m.family == "isotropic":
        if [k for k in _ISO_KEYS if k not in v] or set(v) - {*_ISO_KEYS, "rho"}:
            raise ValueError(f"{m.id}: isotropic values are E0, nu (and rho), got {sorted(v)}")
        try:
            isotropic_moduli(E=float(v["E0"]), nu=float(v["nu"]))
        except ValueError as e:
            raise ValueError(f"{m.id}: E0 / nu not admissible: {e}") from None
    elif m.family == "orthotropic":
        if [k for k in _ORTHO_KEYS if k not in v] or set(v) - {*_ORTHO_KEYS, "rho"}:
            raise ValueError(f"{m.id}: orthotropic values are {', '.join(_ORTHO_KEYS)} (and rho), got {sorted(v)}")
        try:
            Orthotropic(*(float(v[k]) for k in _ORTHO_KEYS))
        except ValueError as e:
            raise ValueError(f"{m.id}: orthotropic constants not admissible: {e}") from None
    else:
        raise ValueError(f"{m.id}: family {m.family!r} (isotropic or orthotropic)")


def material_section(raw: Mapping[str, Any], system: Any) -> dict[str, Any]:
    """The ``material`` section with ``catalog: <id>`` replaced by the record's values: E0 / nu / rho (isotropic) or
    E0 = E1 and an ``orthotropic`` block. An SI record needs the config's ``units`` section (``system``, a
    :class:`~toporlogy.configuration.units.UnitSystem`) and is converted into it. A key given beside ``catalog`` that
    the record also sets is refused (one source per value)."""
    ids = {m.id: m for m in MATERIALS}
    rid = str(raw["catalog"])
    if rid not in ids:
        from toporlogy.configuration.registry import suggest
        raise ValueError(f'no material "{rid}" in the catalog{suggest(rid, list(ids)) or ""} '
                         f"(ids: {', '.join(ids)})")
    m = ids[rid]
    v = dict(m.values)
    if m.units == "SI":
        if system is None:
            raise ValueError(f"{rid} is in SI units: add a top-level units section, e.g. units: {{length: mm, "
                             f"force: N}}")
        v = {k: (system.from_si(x, _PRESSURE_DIM) if k in _MODULI else
                 system.from_si(x, _DENSITY_DIM) if k == "rho" else x) for k, x in v.items()}
    out: dict[str, Any] = {k: x for k, x in raw.items() if k != "catalog"}
    if m.family == "isotropic":
        sets: dict[str, Any] = {k: v[k] for k in ("E0", "nu", "rho") if k in v}
    else:
        sets = {"E0": v["E1"], "orthotropic": {k: v[k] for k in _ORTHO_KEYS}}
        if "rho" in v:
            sets["rho"] = v["rho"]
    clash = sorted(set(sets) & set(out))
    if clash:
        raise ValueError(f"{', '.join(clash)} given both by catalog {rid} and in the config")
    return {**out, **sets}


# ---------------------------------------------------------------------------------------------- compatibility
@dataclass(frozen=True)
class Cell:
    status: str
    evidence: tuple[str, ...] = ()
    reason: str = ""
    enforced: str = ""                    # UNSUPPORTED: component | catalog | first_evaluation


PHYS = ("elasticity", "elasticity_gmg", "modal", "nonlinear_elasticity", "tensor_elasticity", "thermoelastic")

# responses that do not read the study's physics (no state solve, or an own auxiliary solve); evidence with elasticity
PHYSICS_INDEPENDENT: dict[str, tuple[str, ...]] = {
    "volume": (T + "test_gradients.py::test_volume_weighted_and_unweighted",),
    "centre_of_mass": (T + "test_mass_properties.py::test_mp7_mp8_study_responses_value_and_gradient",),
    "inertia": (T + "test_mass_properties.py::test_mp7_mp8_study_responses_value_and_gradient",),
    "effective_stiffness": (T + "test_homogenization.py::test_h15_h16_study_responses_value_and_gradient",),
    "effective_poisson": (T + "test_homogenization.py::test_h15_h16_study_responses_value_and_gradient",),
    "dilated_volume": (T + "test_robust.py::test_gradcheck_robust_compliance_and_dilated_volume",),
    "local_volume": (T + "test_lengthscale.py::test_gradcheck_local_volume_and_grey",),
    "grey_level": (T + "test_lengthscale.py::test_gradcheck_local_volume_and_grey",),
    "conduction_connectivity": (T + "test_connectivity.py::test_gradcheck_conduction_connectivity",),
    "coupling_compliance": (T + "test_connectivity.py::test_gradcheck_coupling_compliance",),
    "phase_volume": (T + "test_multi_material.py::test_v4_gradients_finite_difference",),
    "mass": (T + "test_multi_material.py::test_v4_gradients_finite_difference",),
    "cost": (T + "test_multi_material.py::test_v4_gradients_finite_difference",),
}
# composites: their status is the weakest status of their parts
COMPOSITE_RESPONSES = ("maximize", "weighted_sum")

_MODAL_ONLY = Cell(UNSUPPORTED, reason="needs physics modal", enforced="component")
_NO_STATE = Cell(UNSUPPORTED, reason="needs the static state u; physics modal solves an eigenproblem only",
                 enforced="catalog")
_NV = Cell(NOT_VERIFIED)


def _row(**cells: Cell) -> dict[str, Cell]:
    return {p: cells.get(p, _NV) for p in PHYS}


RESPONSE_MATRIX: dict[str, dict[str, Cell]] = {
    "compliance": _row(
        elasticity=Cell(SUPPORTED, (T + "test_gradients.py::test_compliance_single_case",
                                    T + "test_independent_vv.py::test_top88_fixed_design_objective_and_sensitivity")),
        elasticity_gmg=Cell(SUPPORTED, (T + "test_gmg.py::test_gradients",
                                        T + "test_gmg.py::test_study_regression_against_direct")),
        modal=_NO_STATE,
        nonlinear_elasticity=Cell(EXPERIMENTAL, (T + "test_nonlinear.py::test_v6_end_compliance_gradient_finite_"
                                                     "difference",),
                                  "same number as end_compliance by construction (EndCompliance subclasses "
                                  "Compliance); only end_compliance is tested"),
        tensor_elasticity=Cell(SUPPORTED, (
            T + "test_material_design.py::test_k8b_scalar_limit_matches_multiphase_simp",
            T + "test_material_design.py::test_g1_compliance_gradient_finite_difference")),
        thermoelastic=Cell(SUPPORTED, (T + "test_thermoelastic_study.py::test_ts2_value_equals_direct_thermoelastic",
                                       T + "test_thermoelastic_study.py::test_ts3_gradient_finite_difference"),
                           "c = F_totᵀ U with the design-dependent thermal load")),
    "robust_compliance": _row(
        elasticity=Cell(SUPPORTED, (T + "test_robust.py::test_gradcheck_robust_compliance_and_dilated_volume",)),
        modal=_NO_STATE),
    "scenario_compliance": _row(
        elasticity=Cell(SUPPORTED, (T + "test_yaml_research_responses.py::test_value_and_gradient",
                                    T + "test_uq.py::test_u10b_robust_gradients_match_finite_differences")),
        modal=_NO_STATE),
    "compliance_case": _row(
        elasticity=Cell(SUPPORTED, (T + "test_yaml_research_responses.py::test_value_and_gradient",)),
        modal=_NO_STATE),
    "displacement": _row(
        elasticity=Cell(SUPPORTED, (T + "test_gradients.py::test_non_self_adjoint_displacement",
                                    T + "test_compose.py::test_displacement_value_and_gradient")),
        modal=_NO_STATE,
        nonlinear_elasticity=Cell(SUPPORTED, (T + "test_nonlinear.py::test_v5_large_deflection_cantilever_elastica",
                                              T + "test_nonlinear.py::test_v6_end_compliance_gradient_finite_"
                                                  "difference"))),
    "stress": _row(
        elasticity=Cell(SUPPORTED, (T + "test_stress.py::test_gradient_aggregated_stress",
                                    T + "test_independent_stress.py::test_s2_von_mises_index")),
        modal=_NO_STATE,
        nonlinear_elasticity=Cell(NOT_VERIFIED, reason="builds; the small-strain stress of the stress response on a "
                                                       "large-displacement state is not a verified measure (risk)")),
    "stress_augmented_lagrangian": _row(
        elasticity=Cell(SUPPORTED, (T + "test_stress.py::test_gradient_augmented_lagrangian",)),
        modal=_NO_STATE),
    "buckling": _row(
        elasticity=Cell(SUPPORTED, (T + "test_buckling.py::test_gradient_buckling_ks",
                                    T + "test_independent_buckling.py::test_load_factors_equal_the_dense_"
                                        "independent_solution")),
        elasticity_gmg=Cell(UNSUPPORTED, reason="buckling needs plain Elasticity (direct solver)", enforced="catalog"),
        modal=_NO_STATE,
        nonlinear_elasticity=Cell(UNSUPPORTED, reason="linear buckling about a linear prestress; not defined on the "
                                                      "non-linear state", enforced="catalog")),
    "end_strain_energy": _row(
        elasticity=Cell(UNSUPPORTED, reason="needs physics nonlinear_elasticity", enforced="component"),
        elasticity_gmg=Cell(UNSUPPORTED, reason="needs physics nonlinear_elasticity", enforced="component"),
        modal=Cell(UNSUPPORTED, reason="needs physics nonlinear_elasticity", enforced="component"),
        nonlinear_elasticity=Cell(SUPPORTED, (T + "test_nonlinear_solvers.py::"
                                                  "test_t1_t2_end_strain_energy_taylor_remainder",),
                                  "hyperelastic materials only (svk, neo_hooke); a stateful law is refused"),
        tensor_elasticity=Cell(UNSUPPORTED, reason="needs physics nonlinear_elasticity", enforced="component")),
    "end_compliance": _row(
        elasticity=Cell(UNSUPPORTED, reason="needs physics nonlinear_elasticity", enforced="component"),
        elasticity_gmg=Cell(UNSUPPORTED, reason="needs physics nonlinear_elasticity", enforced="component"),
        modal=Cell(UNSUPPORTED, reason="needs physics nonlinear_elasticity", enforced="component"),
        nonlinear_elasticity=Cell(SUPPORTED, (T + "test_nonlinear.py::test_v6_end_compliance_gradient_finite_"
                                                  "difference",
                                              T + "test_nonlinear_orientation.py::test_n4_end_compliance_gradient_"
                                                  "with_a_field"))),
    **{r: _row(elasticity=_MODAL_ONLY, elasticity_gmg=_MODAL_ONLY, nonlinear_elasticity=_MODAL_ONLY,
               modal=Cell(SUPPORTED, (T + "test_modal.py::test_response_gradients",
                                      T + "test_modal.py::test_eigenpairs_match_dense")))
       for r in ("frequency", "eigenvalue", "min_eigenvalue_ks", "min_frequency")},
}


@dataclass(frozen=True)
class Rule:
    """A compatibility rule outside the physics x response matrix (material, element, supports, phases ...)."""

    id: str
    when: Mapping[str, str]               # axis -> value, all must hold
    status: str
    reason: str = ""
    evidence: tuple[str, ...] = ()
    enforced: str = ""


RULES: tuple[Rule, ...] = (
    Rule("gmg_2d", {"physics": "elasticity_gmg", "dim": "2"}, UNSUPPORTED, "elasticity_gmg is 3-D only",
         enforced="component"),
    Rule("gmg_material_field", {"physics": "elasticity_gmg", "material": "orthotropic_field"}, UNSUPPORTED,
         "matrix-free kernels assume one element matrix", enforced="component"),
    Rule("gmg_per_case_supports", {"physics": "elasticity_gmg", "supports": "per_case"}, UNSUPPORTED,
         "one support set only", enforced="component"),
    Rule("gmg_oblique_supports", {"physics": "elasticity_gmg", "supports": "oblique_or_spring"}, UNSUPPORTED,
         "axis-aligned fixed DOFs only", enforced="component"),
    Rule("modal_per_case_supports", {"physics": "modal", "supports": "per_case"}, UNSUPPORTED,
         "one support set only", enforced="component"),
    Rule("nonlinear_per_case_supports", {"physics": "nonlinear_elasticity", "supports": "per_case"}, UNSUPPORTED,
         "one support set only", enforced="component"),
    Rule("buckling_oblique_supports", {"response": "buckling", "supports": "oblique_or_spring"}, UNSUPPORTED,
         "LinearBuckling assumes axis-aligned fixed DOFs (ConstrainedElasticity gave 'wrong M dimensions' at the "
         "first evaluation before this rule)", enforced="catalog"),
    Rule("buckling_per_case_supports", {"response": "buckling", "supports": "per_case"}, UNSUPPORTED,
         "LinearBuckling needs one support set", enforced="catalog"),
    Rule("plane_strain_3d", {"material": "plane_strain", "dim": "3"}, UNSUPPORTED, "plane applies to 2-D only",
         enforced="component"),
    Rule("hashin_shtrikman_orthotropic", {"interpolation": "hashin_shtrikman", "material": "orthotropic"},
         UNSUPPORTED, "isotropic bound", enforced="component"),
    Rule("phases_responses", {"material": "phases", "response": "other_than_compliance_phase_volume_mass_cost"},
         UNSUPPORTED, "only Compliance, PhaseVolume, Mass, Cost verified with several phases", enforced="component"),
    Rule("phases_oc", {"material": "phases", "optimizer": "oc"}, UNSUPPORTED,
         "OC assumes one positive-gradient constraint", enforced="component"),
    Rule("phases_links", {"material": "phases", "link": "other_than_embed_density_filter_projection_passive"},
         UNSUPPORTED, "only Embed, DensityFilter, HeavisideProjection, Passive verified", enforced="component"),
    Rule("phases_elasticity", {"material": "phases", "physics": "elasticity"}, SUPPORTED,
         evidence=(T + "test_multi_material.py::test_v4_gradients_finite_difference",
                   T + "test_multi_material.py::test_v3_two_material_bar_closed_form")),
    Rule("phases_gmg", {"material": "phases", "physics": "elasticity_gmg"}, NOT_VERIFIED),
    Rule("phases_modal", {"material": "phases", "physics": "modal"}, NOT_VERIFIED,
         "builds; the mass interpolation of several phases is not tested"),
    Rule("phases_nonlinear", {"material": "phases", "physics": "nonlinear_elasticity"}, NOT_VERIFIED,
         "builds with compliance; end_compliance is refused with phases"),
    Rule("orthotropic_elasticity", {"material": "orthotropic_field", "physics": "elasticity"}, SUPPORTED,
         evidence=(T + "test_fem_mms_extended.py::test_m2_orientation_field_converges_at_second_order",
                   T + "test_independent_matrices.py::test_k1_k2_reduced_stiffness")),
    Rule("orthotropic_buckling", {"material": "orthotropic_field", "response": "buckling"}, SUPPORTED,
         evidence=(T + "test_element_orientation.py::test_p3c_buckling_gradient_q4im",
                   T + "test_buckling_physics.py::test_orthotropic_column_buckles_with_its_axial_modulus")),
    Rule("orthotropic_modal", {"material": "orthotropic", "physics": "modal"}, NOT_VERIFIED,
         "builds; modal tests use isotropic material"),
    Rule("orthotropic_gmg_uniform", {"material": "orthotropic", "physics": "elasticity_gmg"}, NOT_VERIFIED,
         "one rotated D is accepted; not tested"),
    Rule("orthotropic_nonlinear", {"material": "orthotropic_field", "physics": "nonlinear_elasticity"}, SUPPORTED,
         "material svk only (neo-Hooke with a general D is refused)",
         (T + "test_nonlinear_orientation.py::test_n3_consistent_tangent_with_a_field",
          T + "test_nonlinear_orientation.py::test_n4_end_compliance_gradient_with_a_field")),
    Rule("nonlinear_incompatible_modes", {"physics": "nonlinear_elasticity", "element": "Q4IM|H8IM"}, UNSUPPORTED,
         "incompatible-mode elements not implemented for the non-linear physics", enforced="component"),
    Rule("plane_strain_elasticity", {"material": "plane_strain", "physics": "elasticity"}, EXPERIMENTAL,
         "D checked against D_plane_strain; no solution benchmark in plane strain",
         (T + "test_material_config.py::test_plane_strain",)),
    Rule("plane_strain_modal", {"material": "plane_strain", "physics": "modal"}, NOT_VERIFIED),
    Rule("finalize_2d", {"post": "finalize", "dim": "2"}, UNSUPPORTED, "finalize needs a 3-D domain",
         enforced="component"),
    Rule("cuda_gmg", {"physics": "elasticity_gmg", "device": "cuda"}, NOT_VERIFIED,
         "no GPU in CI or in this record"),
    Rule("mesh_grid_equivalence", {"element": "mesh:Q4|mesh:H8", "mesh": "from_domain"}, SUPPORTED,
         "the general mesh of a Domain reproduces the grid path (K <= 1e-14, u <= 1e-12)",
         (T + "test_mesh_assembly.py::test_g1_from_domain_q4_equals_grid_path",
          T + "test_mesh_assembly.py::test_g1_from_domain_h8_equals_grid_path")),
    Rule("mesh_material_field", {"element": "mesh:*", "material": "per_cell_D"}, SUPPORTED,
         evidence=(T + "test_mesh_assembly.py::test_g9_two_material_bar",
                   T + "test_mesh_assembly.py::test_g8_pattern_reuse")),
    Rule("mesh_nearly_incompressible_displacement_only", {"element": "mesh:Q4|mesh:T3|mesh:H8|mesh:T4",
                                                           "material": "nu_near_0.5"}, EXPERIMENTAL,
         "runs, but locks (L2 error 0.94-0.996 at nu = 0.4999): use the Taylor-Hood elements",
         (T + "test_mesh_locking.py::test_l1b_displacement_elements_lock",)),
    Rule("mesh_incompressible_mixed", {"element": "mesh:TH-*", "material": "nu_0.5"}, SUPPORTED,
         "lambda = inf exact; the zero-mean pressure gauge is added when the constant pressure is in the null space",
         (T + "test_mesh_locking.py::test_l1e_taylor_hood_converges_near_and_at_incompressibility",)),
    # material design (docs/olcumler/malzeme_tasarimi.md)
    Rule("tensor_phases_elasticity", {"interpolation": "tensor_phases", "physics": "tensor_elasticity"}, SUPPORTED,
         "phases with their own tensors, rules simplex and recursive",
         (T + "test_material_design.py::test_k7_independent_assembly",
          T + "test_material_design.py::test_k6_series_parallel_bar")),
    Rule("angle_elasticity", {"link": "embed.angle", "physics": "tensor_elasticity"}, SUPPORTED,
         "density + fibre angle, 2-D, one or several phases", (
             T + "test_material_design.py::test_a5_off_axis_tension_and_gradient",
             T + "test_material_design.py::test_a8_yaml_example_runs")),
    Rule("tensor_responses", {"physics": "tensor_elasticity",
                              "response": "other_than_compliance_volume_phase_volume_mass_cost"}, UNSUPPORTED,
         "only Compliance, Volume, PhaseVolume, Mass, Cost verified with tensor_elasticity", enforced="catalog"),
    Rule("tensor_interpolation", {"physics": "tensor_elasticity", "interpolation": "other_than_tensor_phases"},
         UNSUPPORTED, "tensor_elasticity reads the material field of interpolation tensor_phases",
         enforced="catalog"),
    Rule("tensor_phases_physics", {"interpolation": "tensor_phases", "physics": "other_than_tensor_elasticity"},
         UNSUPPORTED, "the scalar-modulus physics cannot take the element tensors", enforced="component"),
    Rule("tensor_incompatible_modes", {"physics": "tensor_elasticity", "element": "Q4IM|H8IM"}, UNSUPPORTED,
         "condensed incompatible modes are not linear in D", enforced="component"),
    Rule("tensor_per_case_supports", {"physics": "tensor_elasticity", "supports": "per_case"}, UNSUPPORTED,
         "one support set only", enforced="component"),
    Rule("tensor_oblique_supports", {"physics": "tensor_elasticity", "supports": "oblique_or_spring"}, UNSUPPORTED,
         "axis-aligned fixed DOFs only", enforced="component"),
    Rule("tensor_post", {"physics": "tensor_elasticity", "post": "other_than_phase_report"}, UNSUPPORTED,
         "the other post steps assume a scalar modulus or an isotropic material", enforced="catalog"),
    Rule("angle_3d", {"link": "embed.angle", "dim": "3"}, UNSUPPORTED, "3-D orientation is not implemented",
         enforced="component"),
    Rule("angle_phases", {"link": "embed.angle", "interpolation": "tensor_phases with several phases"}, SUPPORTED,
         "one angle field turns every phase with rotate true (docs/olcumler/faz_aci_onkayit.md)",
         (T + "test_phase_angle.py::test_ma1_two_phase_angle_study_reduces_to_one_phase",
          T + "test_phase_angle.py::test_ma2_gradients_finite_difference",
          T + "test_phase_angle.py::test_ma4_yaml_example_runs")),
    Rule("angle_phases_no_density", {"link": "embed.angle density false", "interpolation":
                                     "tensor_phases with several phases"}, UNSUPPORTED,
         "which phase fills the part is undefined", enforced="component"),
    Rule("angle_interpolation", {"link": "embed.angle", "interpolation": "other_than_tensor_phases"}, UNSUPPORTED,
         "the angle turns the phase tensors of tensor_phases", enforced="component"),
    Rule("angle_isotropic", {"link": "embed.angle", "material": "isotropic phase"}, UNSUPPORTED,
         "the material does not depend on the angle (zero gradient)", enforced="component"),
    Rule("angle_oc", {"link": "embed.angle", "optimizer": "oc"}, UNSUPPORTED,
         "OC assumes a constraint gradient positive in every variable; the angle has none", enforced="catalog"),
)


def status(physics: str, response: str) -> Cell:
    """Status of ``response`` with ``physics`` (registry names). Composite responses: not decidable alone."""
    if response in PHYSICS_INDEPENDENT:          # tested with elasticity only; elsewhere read from the code, not tested
        if physics == "elasticity":
            return Cell(SUPPORTED, PHYSICS_INDEPENDENT[response], "tested with physics elasticity")
        return Cell(NOT_VERIFIED, reason="does not read the study's physics (read from the code); combination not "
                                         "tested")
    if response in COMPOSITE_RESPONSES:
        return Cell(NOT_VERIFIED, reason="composite: the weakest status of its parts")
    row = RESPONSE_MATRIX.get(response)
    if row is None:
        return Cell(NOT_VERIFIED, reason="not in the catalog")
    return row.get(physics, _NV)


# ---------------------------------------------------------------------------------------------- build-time check
def _response_types(spec: Any) -> list[str]:
    """Every component type in a resolved response spec (nested responses of maximize / weighted_sum included)."""
    out: list[str] = []
    if isinstance(spec, dict):
        if "type" in spec:
            out.append(str(spec["type"]))
        for v in spec.values():
            if isinstance(v, dict | list):
                out += _response_types(v)
    elif isinstance(spec, list):
        for v in spec:
            out += _response_types(v)
    return out


def check_build(resolved: Mapping[str, Any], physics: Any, responses: Sequence[Any]) -> None:
    """Refuse the ``UNSUPPORTED`` combinations this catalog enforces (``enforced == "catalog"``).

    ``resolved``: the resolved config; ``physics``: the built physics object; ``responses``: the built objective and
    constraint responses. Raises :class:`ConfigError` naming the rule.
    """
    from toporlogy.boundary.constrained import ConstrainedElasticity, PerCaseSupports

    ptype = resolved["physics"]["type"]
    kinds = _response_types(resolved.get("objective")) + _response_types(
        [c.get("response") for c in resolved.get("constraints") or []])
    # the matrix cells this catalog enforces; nested specs (maximize, weighted_sum, robust wrappers) included
    for kind in dict.fromkeys(kinds):
        cell = RESPONSE_MATRIX.get(kind, {}).get(ptype)
        if cell is not None and cell.status == UNSUPPORTED and cell.enforced == "catalog":
            raise ConfigError(f"unsupported combination (catalog): response {kind} with physics {ptype}: "
                              f"{cell.reason}", "physics")
    if ptype == "modal":          # responses outside the matrix (user components) that need the static state
        state = [getattr(r, "name", type(r).__name__) for r in responses if "u" in getattr(r, "needs", ())]
        if state:
            raise ConfigError(f"unsupported combination (catalog): response(s) {state} with physics modal: needs the "
                              "static state u; physics modal solves an eigenproblem only", "physics")
    _check_material_design(resolved, physics, responses)
    if "buckling" in kinds and isinstance(physics, ConstrainedElasticity | PerCaseSupports):
        what = "oblique supports or springs" if isinstance(physics, ConstrainedElasticity) else "per-case supports"
        raise ConfigError(f"unsupported combination (catalog): response buckling with {what}: linear buckling "
                          "needs axis-aligned fixed DOFs and one support set", "supports")


_TENSOR_RESPONSES = ("Compliance", "Volume", "PhaseVolume", "Mass", "Cost")


def _check_material_design(resolved: Mapping[str, Any], physics: Any, responses: Sequence[Any]) -> None:
    """Rules tensor_interpolation, tensor_responses, tensor_post and angle_oc."""
    from toporlogy.fem.tensor_elastic import TensorElasticity
    angle = any(isinstance(lk, Mapping) and lk.get("type") == "embed" and lk.get("angle") is not None
                for lk in resolved.get("chain") or [])
    if angle and (resolved.get("optimizer") or {}).get("type") == "oc":
        raise ConfigError("unsupported combination (catalog): an angle block with optimizer oc: OC assumes a "
                          "constraint gradient positive in every variable; use mma or gcmma", "optimizer")
    if not isinstance(physics, TensorElasticity):
        return
    itype = resolved["interpolation"]["type"]
    if itype != "tensor_phases":
        raise ConfigError(f"unsupported combination (catalog): physics tensor_elasticity with interpolation {itype}: "
                          "it reads the material field of interpolation tensor_phases", "interpolation")
    bad = [getattr(r, "name", type(r).__name__) for r in responses if type(r).__name__ not in _TENSOR_RESPONSES]
    if bad:
        raise ConfigError(f"unsupported combination (catalog): response(s) {bad} with physics tensor_elasticity: "
                          f"only {', '.join(_TENSOR_RESPONSES)} are verified", "physics")
    posts = [p["type"] for p in resolved.get("post") or [] if p["type"] != "phase_report"]
    if posts:
        raise ConfigError(f"unsupported combination (catalog): post step(s) {posts} with physics tensor_elasticity: "
                          "only phase_report is available", "post")


# ---------------------------------------------------------------------------------------------- output
def as_dict() -> dict[str, Any]:
    """The whole catalog as JSON-able data (the content of ``docs/vv_katalog.json``)."""
    from toporlogy.configuration.registry import REGISTRY
    import toporlogy.configuration.build  # noqa: F401  (registers the built-in components)

    def plain(x: Any) -> Any:
        return json.loads(json.dumps(asdict(x), default=list))

    responses = REGISTRY.names("response")
    return {
        "schema_version": SCHEMA_VERSION, "statuses": list(STATUSES), "benchmark_kinds": list(BENCHMARK_KINDS),
        "elements": [plain(e) for e in ELEMENTS], "physics": [plain(p) for p in PHYSICS],
        "solvers": [plain(s) for s in SOLVERS], "benchmarks": [plain(b) for b in BENCHMARKS],
        "materials": [plain(m) for m in MATERIALS],
        "compatibility": {"physics": list(PHYS),
                          "physics_x_response": {r: {p: plain(status(p, r)) for p in PHYS} for r in responses},
                          "rules": [plain(r) for r in RULES]},
    }


def _short(test_id: str) -> str:
    return test_id.removeprefix(T)


def render_markdown(data: Mapping[str, Any]) -> str:
    """Human-readable catalog generated from :func:`as_dict` (``docs/vv_katalog.md``)."""
    L: list[str] = [
        "# Bilimsel kataloglar ve uyumluluk matrisi (üretilmiş dosya, elle düzenleme)", "",
        f"Şema sürümü {data['schema_version']}. Kaynak: `src/toporlogy/configuration/catalog.py` (ve "
        "`_catalog_benchmarks.py`); yeniden üretim: `python -m toporlogy.configuration.catalog --write docs`. "
        "Bileşen adları registry'den, kaynak anahtarları `toporlogy.references`'tan gelir; "
        "`tests/toporlogy/test_catalog.py` ikisini ve bu dosyanın güncelliğini denetler.", "",
        "Durumlar: **SUPPORTED** hızlı pakette bir referansa ya da sonlu farka karşı geçen test var; "
        "**EXPERIMENTAL** koşuyor, denetim kısmi; **UNSUPPORTED** reddediliyor (`enforced`: bileşen, katalog ya da "
        "ilk değerlendirme); **NOT VERIFIED** kurulum kabul ediyor, bilinen test yok. Registry'de olmak "
        "desteklenmek değildir.", "", "## 1. Elemanlar", "",
        "| Eleman | Boyut | Düğüm × SD | İnterpolasyon | Quadrature | Fizik | Kilitlenme | Bozulma | Kanıt |",
        "|:---|---:|:---|:---|:---|:---|:---|:---|:---|"]
    for e in data["elements"]:
        L.append(f"| {e['id']} | {e['dim']} | {e['nodes']} × {e['dofs_per_node']} | {e['interpolation']} | "
                 f"{e['quadrature']} | {', '.join(e['physics'])} | {e['locking']} | {e['distortion']} | "
                 f"{'<br>'.join(_short(t) for t in e['evidence'])} |")
    L += ["", "## 2. Fizik ve çözücüler", "",
          "| Fizik | Rezidü | Matris | Çözüm | Cihaz | Boyut | Eleman | Mesnet | Yakınsama | Tasarım girişi |",
          "|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|"]
    for p in data["physics"]:
        L.append(f"| {p['id']} | {p['residual']} | {p['matrix']} | {p['solve']} | {', '.join(p['devices'])} | "
                 f"{', '.join(map(str, p['dims']))} | {', '.join(p['elements'])} | {p['supports']} | "
                 f"{p['convergence']} | {p['scalar_modulus']} |")
    L += ["", "| Çözücü | Registry'de | Yöntem | Arka uçlar | Durum | Not |", "|:---|:---|:---|:---|:---|:---|"]
    for s in data["solvers"]:
        L.append(f"| {s['id']} | {'evet' if s['registry'] else 'hayır'} | {s['method']} | "
                 f"{', '.join(s['backends'])} | {s['status']} | {s['note']} |")
    L += ["", "## 3. Benchmark kataloğu", "",
          "| Kimlik | Tür | Test | Nicelik | Referans | Tolerans | Süre | Düzey |",
          "|:---|:---|:---|:---|:---|:---|:---|:---|"]
    for b in data["benchmarks"]:
        L.append(f"| {b['id']} | {b['kind']} | {_short(b['test'])} | {b['quantity']} | {b['reference']} | "
                 f"{b['tolerance']} | {b['runtime']} | {b['validation']} |")
    L += ["", "Deney verisiyle doğrulama (validation) yok: her satır code/solution verification. Aynı yazarın "
              "ikinci uygulaması (`independent_same_author`) dış ekip doğrulaması sayılmaz.",
          "", "## 4. Malzeme veri kataloğu", "",
          "Yalnız sentetik kayıt var; kaynağı, sayfası ve lisansı bu çalışmada okunmamış gerçek malzeme değeri "
          "girilmez. YAML: `material: {catalog: <kimlik>}` (SI kayıt `units` bölümü ister ve ona çevrilir; "
          "`docs/olcumler/malzeme_katalog_onkayit.md`).", "",
          "| Kimlik | Tür | Aile | Değerler | Birim | Kaynak | Lisans | Not |",
          "|:---|:---|:---|:---|:---|:---|:---|:---|"]
    for m in data["materials"]:
        vals = ", ".join(f"{k} = {v:g}" for k, v in m["values"].items())
        L.append(f"| {m['id']} | {m['kind']} | {m['family']} | {vals} | {m['units']} | {m['source']}"
                 f"{' (' + m['page_or_table'] + ')' if m['page_or_table'] else ''} | {m['licence'] or '—'} | "
                 f"{m['note']} |")
    comp = data["compatibility"]
    phys = comp["physics"]
    L += ["", "## 5. Uyumluluk: fizik × yanıt", "", "| Yanıt | " + " | ".join(phys) + " |",
          "|:---|" + ":---|" * len(phys)]
    for r, row in comp["physics_x_response"].items():
        cells = []
        for p in phys:
            c = row[p]
            tag = c["status"] + (f" ({c['enforced']})" if c["enforced"] else "")
            cells.append(tag)
        L.append(f"| {r} | " + " | ".join(cells) + " |")
    L += ["", "Gerekçe ve kanıt (yalnız boş olmayanlar):", ""]
    for r, row in comp["physics_x_response"].items():
        for p in phys:
            c = row[p]
            if c["reason"] or c["evidence"]:
                ev = "; ".join(_short(t) for t in c["evidence"])
                L.append(f"- `{r}` × `{p}`: {c['status']}" + (f" — {c['reason']}" if c["reason"] else "") +
                         (f" [{ev}]" if ev else ""))
    L += ["", "## 6. Diğer kurallar (malzeme, eleman, mesnet, faz, cihaz)", "",
          "| Kural | Koşul | Durum | Uygulama | Gerekçe | Kanıt |", "|:---|:---|:---|:---|:---|:---|"]
    for ru in comp["rules"]:
        cond = ", ".join(f"{k} = {v}" for k, v in ru["when"].items())
        L.append(f"| {ru['id']} | {cond} | {ru['status']} | {ru['enforced'] or '—'} | {ru['reason'] or '—'} | "
                 f"{'<br>'.join(_short(t) for t in ru['evidence']) or '—'} |")
    return "\n".join(L) + "\n"


def write(folder: str | Path) -> list[Path]:
    """Write ``vv_katalog.json`` and ``vv_katalog.md`` into ``folder``."""
    folder = Path(folder)
    data = as_dict()
    j, m = folder / "vv_katalog.json", folder / "vv_katalog.md"
    j.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    m.write_text(render_markdown(data), encoding="utf-8", newline="\n")
    return [j, m]


def _iter_evidence(data: Mapping[str, Any]) -> Iterable[str]:
    for e in data["elements"]:
        yield from e["evidence"]
    for p in data["physics"]:
        yield from p["evidence"]
    for b in data["benchmarks"]:
        yield b["test"]
    for row in data["compatibility"]["physics_x_response"].values():
        for c in row.values():
            yield from c["evidence"]
    for r in data["compatibility"]["rules"]:
        yield from r["evidence"]


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the generated catalog (vv_katalog.json, vv_katalog.md).")
    ap.add_argument("--write", metavar="FOLDER", required=True)
    a = ap.parse_args(argv)
    for p in write(a.write):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


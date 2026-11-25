"""Declarative studies: a YAML/JSON config → :class:`~toporlogy.optimization.study.Study`, run folders, overrides,
sweeps.

Schema (every key optional unless marked *required*; defaults in brackets). Components are a registered name or a
mapping ``{type: <name>, <param>: <value>}`` (:mod:`toporlogy.configuration.registry`; ``python -m toporlogy list``
shows every component with its parameters and defaults).

Units: any consistent set. Lengths (``h``, ``origin``, selector coordinates, ``t_max``, ``min_solid``) share one
unit, e.g. m; ``E0`` is force/length² (Pa with N and m); forces are force units (N); ``rho`` is mass/length³;
``rmin`` and the ``radius`` of ``local_volume`` are in elements.

.. code-block:: yaml

    name: mbb                      # [config file stem] run-folder suffix
    domain: {type: box, shape: [60, 20], h: 1.0}                    # *required*; also csg, stl (keep / void shapes)
    material: {E0: 1.0, nu: 0.3, rho: 1.0}                          # [1.0, 0.3, 1.0]; also plane, orthotropic,
                                                                    # strength (see below)
    physics: {type: elasticity, element: null, solver: direct}      # element null = H8 (3-D) / Q4 (2-D); or modal
                                                                    # or nonlinear_elasticity (large displacements,
                                                                    # objective end_compliance)
    supports:                                                       # *required*, >= 1 (bc supports)
      - {type: fixed, where: {type: plane, axis: x, at: min}, dofs: x}
    load_cases:                                                     # *required* (optional for physics modal)
      - name: tip                  # [case<i>]
        weight: 1.0                # [1.0] used by compliance
        supports: null             # [null] own supports of this case (null = the supports above)
        loads: [{type: force, where: {type: point, at: [min, max]}, force: [0, -1]}]
    combinations: []               # [[]] {name, factors: {case: factor}, weight}: extra columns (same supports)
    chain: [{type: density_filter, rmin: 1.5}]                      # embed / passive added at the ends if absent
    interpolation: simp            # [simp]
    objective: compliance          # [compliance]
    constraints:                   # [[]]  response <= upper, g = (value - upper) * scale (scale null = 1/|upper|)
      - {response: volume, upper: 0.5, scale: null, name: null}    # upper null: the response's own bound
    optimizer: oc                  # [oc]
    schedule: null                 # [null] e.g. {type: beta_continuation, every: 25}
    robust: null                   # [null] {min_solid, min_void, eta_e, eta_d, eta_i, beta, kernel, eta_max}
    connectivity: null             # [null] {interfaces, ties, constraint, eta, bound, sink, gate, repair, ...}
    run: {max_iter: 100, tol: 0.01, seed: 0, feasibility_tol: 0.001, move_beta_ref: null}
    output: {folder: runs, snapshot_every: 0}
    post: []                       # [[]] post-processing components

``tol`` is the max design change for convergence; ``feasibility_tol`` the relative constraint violation still
judged PASS (constraint verdict); ``seed`` seeds NumPy and torch; ``move_beta_ref`` (null = off) shrinks the
optimizer's move limit to move0 · min(1, move_beta_ref / β) after every β step of the schedule. The resolved
config (all defaults filled, auto links explicit) is written as ``config.resolved.yaml`` with a ``provenance``
block (git revision, time, source, overrides); loading it again builds the identical study.

Material. ``plane`` (2-D only): ``stress`` [default] or ``strain``. ``orthotropic`` [null = isotropic E0, nu]:
{E1, E2, E3, G12, G13, G23, nu12, nu13, nu23} in the material axes, oriented by ``angle`` (degrees about z) or
``axes: [axis1, axis2]`` (3-D), or per element by ``angle_field`` (2-D: a ``.npy`` file of the element grid shape,
degrees about z) or ``axis_field`` (3-D: ``.npy`` of shape (*grid, 3) for axis 1 with ``axis2: [x, y, z]``, or
(*grid, 2, 3) with both axes); E0 must then equal E1 (the modulus the interpolation scales). The fibre angle is
data, not a design variable. ``strength`` [null]:
{tension, compression, shear, safety_factor} (isotropic) or {Xt, Xc, Yt, Yc, S12, Zt, Zc, S13, S23, f12,
safety_factor} (orthotropic); used by the ``stress`` responses (``criterion``) and their default limit.
(:mod:`toporlogy.core.material`, :mod:`toporlogy.optimization.strength`.)

Several phases. ``material.phases`` [null = one material]: ``[{name, E, density, cost}, ...]`` in increasing E
(density [1.0] mass/length³, cost [1.0] per unit volume of dense phase); the phases share ν (and the orthotropic
axes), so E scales one unit elasticity matrix. Needs ``interpolation: multiphase_simp``; the embed link then makes
one design field per phase (design vector field-major). Objective / constraints: ``compliance``, ``phase_volume``
(``phases``: null = solid, a name or a list), ``mass``, ``cost``; optimizer ``mma`` or ``gcmma``; chain links
``density_filter``, ``heaviside_projection``; post step ``phase_report`` (label field, per-phase volumes / mass /
cost, voxel surfaces per phase in 3-D). (:mod:`toporlogy.design.phases`.)

Boundary conditions. Supports and loads build a :class:`toporlogy.boundary.BoundaryConditions` (selectors, supports,
loads, per-case supports, combinations); its ``check()`` runs in :func:`build` — an error (e.g. an unrestrained
rigid-body mode) is a :class:`ConfigError`; warnings are kept in :attr:`Setup.bc_report`. The physics comes from it
(``ConstrainedElasticity`` for oblique supports and springs, ``PerCaseSupports`` for per-case supports).

``robust``: ``min_solid`` / ``min_void`` (length) set the density filter radius and the thresholds
(:func:`toporlogy.optimization.robust.robust_chain_settings`; the ``rmin`` written into the resolved chain), or give
``eta_e`` / ``eta_d`` directly. A ``heaviside_projection`` link is added when the chain has none. The thresholds are
shared by ``robust_compliance``, ``dilated_volume``, ``local_volume(source: dilated)`` and ``robust_continuation``.

``connectivity`` (one body): ``interfaces`` (name → selector; null = one per support and per load),
``ties`` (section in elements of frozen minimal ties, null / 0 = none; the interface cells are frozen as well,
otherwise a void interface cell cuts the tie), ``constraint`` (conduction | coupling | none),
``eta``, ``bound`` (null = 20 / 2.5), ``sink``, ``options`` (extra response parameters), ``gate`` (adds the
``connectivity_gate`` post step: verdict in ``summary.json``, exit 1 on FAIL), ``repair`` (bridge | none),
``repair_section``, ``threshold``, ``symmetry`` (mirror planes as in the ``symmetry`` link, {x: 500} or
[[x, 500], [y, mid]]: the frozen ties are placed together with their mirror images; null = the planes of a
``symmetry`` link in the chain, else none).

Public functions: :func:`read_config`, :func:`resolve`, :func:`build`, :func:`study_from_dict`, :func:`load_config`,
:func:`apply_overrides`, :func:`parse_sweep`, :func:`sweep_combinations`, :func:`run_setup`, :func:`read_run`.
"""
from __future__ import annotations

from typing import Any

from toporlogy.configuration.registry import Param


SCHEMA_KEYS = ("name", "domain", "material", "physics", "supports", "load_cases", "combinations", "chain",
               "interpolation", "objective", "constraints", "optimizer", "schedule", "robust", "connectivity", "run",
               "output", "post", "provenance")


_REQUIRED = ("domain", "supports")                     # load_cases too, unless physics is modal


_MATERIAL = {"E0": Param("E0", 1.0, float, unit="force/length^2"), "nu": Param("nu", 0.3, float),
             "rho": Param("rho", 1.0, float, unit="mass/length^3"),
             "plane": Param("plane", "stress", str),
             "orthotropic": Param("orthotropic", None, None, nullable=True),
             "strength": Param("strength", None, None, nullable=True, unit="force/length^2"),
             "phases": Param("phases", None, None, nullable=True)}


_RUN = {"max_iter": Param("max_iter", 100, int), "tol": Param("tol", 0.01, float),
        "seed": Param("seed", 0, int), "feasibility_tol": Param("feasibility_tol", 1e-3, float),
        "move_beta_ref": Param("move_beta_ref", None, float, nullable=True)}


_OUTPUT = {"folder": Param("folder", "runs", str), "snapshot_every": Param("snapshot_every", 0, int)}


_CASE = {"name": Param("name", None, str, nullable=True), "weight": Param("weight", 1.0, float)}


_COMBINATION = {"name": Param("name"), "factors": Param("factors"), "weight": Param("weight", 1.0, float)}


_CONSTRAINT = {"upper": Param("upper", None, float, nullable=True),
               "scale": Param("scale", None, float, nullable=True), "name": Param("name", None, str, nullable=True)}


_ROBUST = {"min_solid": Param("min_solid", None, float, nullable=True, unit="length"),
           "min_void": Param("min_void", None, float, nullable=True, unit="length"),
           "eta_e": Param("eta_e", None, float, nullable=True), "eta_d": Param("eta_d", None, float, nullable=True),
           "eta_i": Param("eta_i", None, float, nullable=True), "beta": Param("beta", None, float, nullable=True),
           "kernel": Param("kernel", "discrete", str), "eta_max": Param("eta_max", 0.75, float)}


_CONNECTIVITY = {"interfaces": Param("interfaces", None, None, nullable=True), "ties": Param("ties", 1, None),
                 "constraint": Param("constraint", "conduction", str, nullable=True),
                 "eta": Param("eta", 0.7, float, nullable=True), "bound": Param("bound", None, float, nullable=True),
                 "sink": Param("sink", None, str, nullable=True), "options": Param("options", None, None,
                                                                                   nullable=True),
                 "gate": Param("gate", True, bool), "repair": Param("repair", "none", str),
                 "repair_section": Param("repair_section", 2, int), "threshold": Param("threshold", 0.5, float),
                 "symmetry": Param("symmetry", None, None, nullable=True)}


_DEFAULTS: dict[str, Any] = {"physics": "elasticity", "chain": [{"type": "density_filter", "rmin": 1.5}],
                             "interpolation": "simp", "objective": "compliance", "constraints": [],
                             "optimizer": "oc", "schedule": None, "post": []}


_TYING = ("symmetry", "extrusion", "pattern")         # links mapping a reduced vector → design vector (before embed)


_NEEDS_FINALIZE = ("manufacturing_checks", "export", "reverify")


STATUS_OK = "ok"

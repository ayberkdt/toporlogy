"""Component constructors: physics.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np

from toporlogy.configuration.helpers import checked_selector, material_model
from toporlogy.configuration.registry import BuildContext, register


def register_physics() -> None:
    """Register the physics components."""

    # ---------------------------------------------------------------- interpolation, physics (dynamics)
    def two_range_simp(ctx: BuildContext, penal: float = 3.0, threshold: float = 0.1, q: float = 1.0,
                       emin_ratio: float = 1e-9) -> Any:
        """SIMP x^p above threshold, threshold^(p-q) x^q below (no spurious low-density modes); E0 from material."""
        from toporlogy.optimization.dynamics import TwoRangeSIMP
        return TwoRangeSIMP(ctx.domain, E0=ctx.material["E0"], penal=penal, threshold=threshold, q=q,
                            emin_ratio=emin_ratio)

    register("interpolation", "two_range_simp", two_range_simp)

    def modal(ctx: BuildContext, element: str | None = None, n_modes: int = 6, sigma: float = 0.0, tol: float = 0.0,
              mass: str = "linear", rhomin_ratio: float = 0.0, mass_threshold: float = 0.1, mass_q: float = 6.0,
              added_mass: float = 0.0, added_mass_at: Any = None, cluster_tol: float = 1e-3,
              repeated: str = "warn") -> Any:
        """Free vibration K phi = lambda M phi (no static solve; load_cases optional); rho0 = material.rho; mass
        "linear" or "two_range"; added_mass (total) spread over the nodes of added_mass_at. Supports: fixed DOFs
        along axes only, one support set."""
        from toporlogy.optimization.dynamics import LinearMass, TwoRangeMass
        from toporlogy.fem.modal import Modal
        bcs, d = ctx.bc, ctx.domain
        sets = {id(bcs.constraints(i)) for i in range(len(bcs.cases))}
        if len(sets) > 1:
            raise ValueError("modal physics needs one support set (no per-case supports)")
        cs = bcs.constraints(0)
        if cs.needs_extension:
            raise ValueError("modal physics supports zero displacements along x / y / z only (no oblique rollers, "
                             "pinned bores or springs)")
        rho0 = float(ctx.material.get("rho", 1.0))
        m: LinearMass | TwoRangeMass
        if mass == "linear":
            m = LinearMass(d, rho0=rho0, rhomin_ratio=rhomin_ratio)
        elif mass == "two_range":
            m = TwoRangeMass(d, rho0=rho0, threshold=mass_threshold, q=mass_q)
        else:
            raise ValueError(f'mass: linear | two_range, got "{mass}"')
        phys = Modal(d, cs.fixed, nu=ctx.material["nu"], element=element, mass=m, n_modes=n_modes, sigma=sigma,
                     tol=tol, cluster_tol=cluster_tol, repeated=repeated, elasticity=material_model(ctx).elasticity)
        if added_mass:
            if added_mass_at is None:
                raise ValueError("added_mass needs added_mass_at (a selector)")
            nodes = checked_selector(ctx, added_mass_at, "added_mass_at").select(d)
            phys.add_point_mass(nodes, float(added_mass) / len(nodes))
        return phys

    def elasticity_gmg(ctx: BuildContext, element: str | None = None, tol: float = 1e-8, maxit: int = 300,
                       direct_below: int = 4000, fallback_max_elements: int = 120_000, warm_start: bool = True,
                       allow_inexact: bool = False) -> Any:
        """Static elasticity by geometric-multigrid PCG (matrix-free; 3-D only; tol = relative residual). Supports:
        zero displacements along x / y / z, one support set (no oblique rollers, pinned bores, springs or per-case
        supports)."""
        bcs, d = ctx.bc, ctx.domain
        if d.dim != 3:
            raise ValueError("elasticity_gmg is 3-D only; use elasticity in 2-D")
        if len({id(bcs.constraints(i)) for i in range(len(bcs.cases))}) > 1:
            raise ValueError("elasticity_gmg needs one support set (no per-case supports); use elasticity")
        if bcs.constraints(0).needs_extension:
            raise ValueError("elasticity_gmg supports zero displacements along x / y / z only (no oblique rollers, "
                             "pinned bores or springs); use elasticity")
        from toporlogy.fem.elastic_gmg import ElasticityGMG
        return ElasticityGMG(d, bcs.constraints(0).fixed, nu=ctx.material["nu"], element=element, tol=tol,
                             maxit=maxit, direct_below=direct_below, fallback_max_elements=fallback_max_elements,
                             warm_start=warm_start, allow_inexact=allow_inexact,
                             elasticity=material_model(ctx).elasticity)

    register("physics", "elasticity_gmg", elasticity_gmg, units={"direct_below": "active elements",
                                                                 "fallback_max_elements": "active elements"})
    register("physics", "modal", modal, nested={"added_mass_at": "selector"},
             units={"added_mass": "mass (total)", "sigma": "eigenvalue"})

    def nonlinear_elasticity(ctx: BuildContext, material: str = "svk", element: str | None = None,
                             energy_interpolation: bool = True, beta: float = 120.0, cutoff: float = 0.1,
                             cutoff_step: float = 0.02, cutoff_max: float = 1.0, density_exponent: float = 3.0,
                             load_steps: int = 10, max_iter: int = 30, rtol: float = 1e-9, min_step: float = 1e-4,
                             warm_start: bool = False, solver: Any = "direct", cutoff_policy: str = "adaptive",
                             line_search: str = "none", merit: str = "residual") -> Any:
        """Static geometrically non-linear elasticity (large displacements, Total Lagrangian, Newton-Raphson, dead
        loads; Q4 / H8). material svk (any material D) | neo_hooke (isotropic); nu, plane, E0 from material.
        energy_interpolation: void elements linear (Wang et al. 2014 scheme; beta, cutoff, cutoff_step, cutoff_max,
        density_exponent; cutoff_policy adaptive | monotone). line_search none | backtracking, merit residual |
        energy (fem.nonlinear, docs/olcumler/nonlineer_cozucu.md). Raises when Newton does not reach the full load.
        Supports: zero displacements along x / y / z only, one support set."""
        from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearElasticity
        bcs, d = ctx.bc, ctx.domain
        if len({id(bcs.constraints(i)) for i in range(len(bcs.cases))}) > 1:
            raise ValueError("nonlinear_elasticity needs one support set (no per-case supports)")
        cs = bcs.constraints(0)
        if cs.needs_extension:
            raise ValueError("nonlinear_elasticity supports zero displacements along x / y / z only (no oblique "
                             "rollers, pinned bores or springs)")
        mm = material_model(ctx)
        if material == "neo_hooke" and mm.orthotropic:
            raise ValueError("neo_hooke is isotropic; use material svk with an orthotropic material")
        ei = None
        if energy_interpolation:
            ei = EnergyInterpolation(E0=float(ctx.material["E0"]), beta=beta, cutoff=cutoff, cutoff_step=cutoff_step,
                                     cutoff_max=cutoff_max, density_exponent=density_exponent, policy=cutoff_policy)
        return NonlinearElasticity(d, cs.fixed, nu=ctx.material["nu"], material=material,
                                   plane=str(ctx.material.get("plane", "stress")) if d.dim == 2 else "stress",
                                   element=element, elasticity=mm.elasticity if material == "svk" else None,
                                   energy_interpolation=ei, load_steps=load_steps, max_iter=max_iter, rtol=rtol,
                                   min_step=min_step, warm_start=warm_start, solver=solver, line_search=line_search,
                                   merit=merit)

    def thermoelastic(ctx: BuildContext, alpha: Any = None, delta_T: float = 0.0, element: str | None = None) -> Any:
        """Linear elasticity with the thermal load of a uniform temperature rise delta_T (one-way, small strain;
        fem.thermoelastic, docs/olcumler/termoelastik_study_onkayit.md). alpha: thermal expansion (number or
        symmetric tensor); nu, plane and D from material. compliance is then F_totᵀ U with the design-dependent
        thermal load. Q4 / H8; supports: zero displacements along x / y / z only, one support set."""
        from toporlogy.fem.thermoelastic import ThermoElastic, ThermoElasticPhysics
        bcs, d = ctx.bc, ctx.domain
        if alpha is None:
            raise ValueError("thermoelastic needs alpha (thermal expansion coefficient)")
        if len({id(bcs.constraints(i)) for i in range(len(bcs.cases))}) > 1:
            raise ValueError("thermoelastic needs one support set (no per-case supports)")
        cs = bcs.constraints(0)
        if cs.needs_extension:
            raise ValueError("thermoelastic supports zero displacements along x / y / z only (no oblique rollers, "
                             "pinned bores or springs)")
        mm = material_model(ctx)
        D = None if mm.elasticity is None else np.asarray(mm.elasticity, float)
        if D is not None and D.ndim != 2:
            raise ValueError("thermoelastic needs one material D (no per-element material field)")
        plane = str(ctx.material.get("plane", "stress")) if d.dim == 2 else "stress"
        if d.dim == 2 and plane == "strain" and mm.orthotropic:
            raise ValueError("thermoelastic in plane strain is isotropic only (the thermal stress uses the 3-D "
                             "isotropic C)")
        te = ThermoElastic(d, cs.fixed, alpha, nu=float(ctx.material["nu"]), element=element, plane=plane, D=D)
        return ThermoElasticPhysics(te, np.full(d.nn, te.T_ref + float(delta_T)))

    register("physics", "thermoelastic", thermoelastic, types={"alpha": None})

    def _grid_element(physics: str, dim: int, element: str | None) -> str:
        want = "Q4" if dim == 2 else "H8"
        if element is not None and element != want:
            raise ValueError(f"{physics} supports Q4 (2-D) and H8 (3-D) only, got {element!r} in {dim}-D")
        return want

    def _time_grid(physics: str, dt: float | None, n_steps: int | None) -> None:
        if dt is None or n_steps is None:
            raise ValueError(f"{physics} needs dt and n_steps (the time step and the number of steps)")

    def transient_heat(ctx: BuildContext, dt: float | None = None, n_steps: int | None = None, theta: float = 1.0,
                       conductivity: float = 1.0, capacity: float = 1.0, capacity_min_ratio: float = 1e-3,
                       initial_temperature: float = 0.0, sink_temperature: float = 0.0, source_density: float = 0.0,
                       power: float = 0.0, source: Any = None, element: str | None = None) -> Any:
        """Transient heat conduction C(x) dT/dt + K(E) T = q by the theta-method (theta 1 backward Euler, 0.5
        Crank-Nicolson; fem.transient_heat, docs/olcumler/gecici_study_onkayit.md): conductivity k = conductivity x
        E / E0, capacity c = capacity x (capacity_min_ratio + (1 - capacity_min_ratio) x). The supports are the heat
        sinks (T = sink_temperature on their nodes; support type fixed); no load_cases: the heat input is
        source_density (per volume, every active element) and / or power (total) spread over the nodes of source.
        Response transient_thermal_compliance. Q4 / H8."""
        from toporlogy.fem.transient_heat import TransientHeat, TransientHeatPhysics, nodal_source
        d = ctx.domain
        if ctx.config.get("load_cases"):
            raise ValueError("transient_heat takes no load_cases: the heat input is physics.source_density and / or "
                             "physics.power at physics.source")
        _time_grid("transient_heat", dt, n_steps)
        _grid_element("transient_heat", d.dim, element)
        cs = ctx.bc.constraints(0)
        if cs.needs_extension:
            raise ValueError("transient_heat: the supports are heat sinks; zero displacements along x / y / z only (no "
                             "oblique rollers, pinned bores or springs)")
        nodes, count = np.unique(np.asarray(cs.fixed, np.int64) // d.dim, return_counts=True)
        if np.any(count != d.dim):
            raise ValueError("transient_heat: the supports are heat sinks (temperature fixed on their nodes); use "
                             "support type fixed on every component (a node with only some components fixed is no "
                             "sink)")
        q = nodal_source(d, source_density) if source_density else np.zeros(d.nn)
        if power:
            if source is None:
                raise ValueError("transient_heat: power needs source (a selector)")
            at = np.asarray(checked_selector(ctx, source, "source").select(d), np.int64)
            np.add.at(q, at, float(power) / at.size)
        elif source is not None:
            raise ValueError("transient_heat: source needs power (the total heat input)")
        heat = TransientHeat(d, nodes, float(sink_temperature), theta=theta)
        return TransientHeatPhysics(heat, q, cast(float, dt), cast(int, n_steps), T0=float(initial_temperature),
                                    conductivity=conductivity, E0=float(ctx.material["E0"]), capacity=capacity,
                                    capacity_min_ratio=capacity_min_ratio)

    def plate(ctx: BuildContext, thickness: float | None = None, edge: str = "simply_supported",
              pressure: float = 0.0, force: float = 0.0, at: Any = None, shear: str = "reduced") -> Any:
        """Mindlin-Reissner plate bending of the 2-D domain (fem.plate, docs/olcumler/plaka_onkayit.md): DOFs
        (w, beta_x, beta_y) per node, thickness t, nu from material. No load_cases: the load is pressure (uniform q)
        and / or force (total transverse force spread over the nodes of at). The supports select the supported
        nodes (support type fixed); edge simply_supported fixes w there, clamped also both rotations. compliance
        is then F^T w. Q4 with reduced shear (shear: reduced | full)."""
        from toporlogy.fem.plate import MindlinPlate, PlatePhysics
        d = ctx.domain
        if d.dim != 2:
            raise ValueError("plate needs a 2-D domain (the plate mid-surface)")
        if ctx.config.get("load_cases"):
            raise ValueError("plate takes no load_cases: the load is physics.pressure and / or physics.force at "
                             "physics.at")
        if thickness is None:
            raise ValueError("plate needs thickness")
        if edge not in ("simply_supported", "clamped"):
            raise ValueError(f"plate edge: simply_supported | clamped, got {edge!r}")
        cs = ctx.bc.constraints(0)
        if cs.needs_extension:
            raise ValueError("plate: the supports select the supported nodes (support type fixed only)")
        nodes, count = np.unique(np.asarray(cs.fixed, np.int64) // d.dim, return_counts=True)
        if np.any(count != d.dim):
            raise ValueError("plate: the supports select the supported nodes; use support type fixed on every "
                             "component (a node with only some components fixed is ambiguous)")
        if nodes.size == 0:
            raise ValueError("plate needs supports (the supported nodes)")
        fixed = 3 * nodes if edge == "simply_supported" else (3 * nodes[:, None] + np.arange(3)).ravel()
        pl = MindlinPlate(d, float(thickness), float(ctx.material["nu"]), fixed, shear=shear)
        F = pl.pressure_load(float(pressure)) if pressure else np.zeros(pl.ndof)
        if force:
            if at is None:
                raise ValueError("plate: force needs at (a selector)")
            where = np.asarray(checked_selector(ctx, at, "at").select(d), np.int64)
            np.add.at(F, 3 * where, float(force) / where.size)
        elif at is not None:
            raise ValueError("plate: at needs force (the total transverse force)")
        if not np.any(F):
            raise ValueError("plate needs a load: pressure and / or force")
        return PlatePhysics(pl, loads=F)

    register("physics", "plate", plate, nested={"at": "selector"})

    def dynamics_newmark(ctx: BuildContext, dt: float | None = None, n_steps: int | None = None,
                         profile: str = "step", pulse_duration: float | None = None, period: float | None = None,
                         rayleigh_alpha: float = 0.0, rayleigh_beta: float = 0.0, rhomin_ratio: float = 1e-6,
                         element: str | None = None) -> Any:
        """Linear structural dynamics M(x) u'' + C u' + K(E) u = s(t) F from rest by Newmark average acceleration
        (fem.dynamics_newmark, docs/olcumler/gecici_study_onkayit.md): consistent mass rho = material.rho x
        (rhomin_ratio + (1 - rhomin_ratio) x), Rayleigh C = rayleigh_alpha M + rayleigh_beta K, F the single load
        case, profile step | half_sine (pulse_duration) | sine (period). Response dynamic_compliance. Q4 / H8, one
        material D; supports: zero displacements along x / y / z only, one load case."""
        from toporlogy.fem.dynamics_newmark import Newmark, NewmarkPhysics
        bcs, d = ctx.bc, ctx.domain
        _time_grid("dynamics_newmark", dt, n_steps)
        el = _grid_element("dynamics_newmark", d.dim, element)
        if int(ctx.n_cases) != 1:
            raise ValueError(f"dynamics_newmark needs exactly one load case (no combinations), got {ctx.n_cases}")
        cs = bcs.constraints(0)
        if cs.needs_extension:
            raise ValueError("dynamics_newmark supports zero displacements along x / y / z only (no oblique rollers, "
                             "pinned bores or springs)")
        if rayleigh_alpha < 0 or rayleigh_beta < 0:
            raise ValueError(f"Rayleigh coefficients must be >= 0, got {rayleigh_alpha}, {rayleigh_beta}")
        mm = material_model(ctx)
        D = None if mm.elasticity is None else np.asarray(mm.elasticity, float)
        if D is not None and D.ndim != 2:
            raise ValueError("dynamics_newmark needs one material D (no per-element material field)")
        nm = Newmark(d, cs.fixed, nu=float(ctx.material["nu"]), D=D, element=el,
                     rayleigh=(float(rayleigh_alpha), float(rayleigh_beta)))
        return NewmarkPhysics(nm, cast(float, dt), cast(int, n_steps), profile=profile, pulse_duration=pulse_duration,
                              period=period, rho0=float(ctx.material.get("rho", 1.0)), rhomin_ratio=rhomin_ratio)

    register("physics", "transient_heat", transient_heat, nested={"source": "selector"},
             units={"dt": "time"})
    register("physics", "dynamics_newmark", dynamics_newmark,
             units={"dt": "time", "pulse_duration": "time", "period": "time"})
    register("physics", "nonlinear_elasticity", nonlinear_elasticity, nested={"solver": "solver"},
             units={"min_step": "load factor"})

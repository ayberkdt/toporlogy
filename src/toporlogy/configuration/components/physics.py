"""Component constructors: physics (moved from toporlogy.registry).
"""
from __future__ import annotations

from typing import Any

from toporlogy.configuration.helpers import checked_selector
from toporlogy.configuration.registry import BuildContext, register


def register_physics() -> None:
    """Register the physics components (moved from registry._register_features)."""

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
        if mass == "linear":
            m = LinearMass(d, rho0=rho0, rhomin_ratio=rhomin_ratio)
        elif mass == "two_range":
            m = TwoRangeMass(d, rho0=rho0, threshold=mass_threshold, q=mass_q)
        else:
            raise ValueError(f'mass: linear | two_range, got "{mass}"')
        phys = Modal(d, cs.fixed, nu=ctx.material["nu"], element=element, mass=m, n_modes=n_modes, sigma=sigma,
                     tol=tol, cluster_tol=cluster_tol, repeated=repeated)
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
                             warm_start=warm_start, allow_inexact=allow_inexact)

    register("physics", "elasticity_gmg", elasticity_gmg, units={"direct_below": "active elements",
                                                                 "fallback_max_elements": "active elements"})
    register("physics", "modal", modal, nested={"added_mass_at": "selector"},
             units={"added_mass": "mass (total)", "sigma": "eigenvalue"})

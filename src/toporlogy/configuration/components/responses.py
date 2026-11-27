"""Component constructors: responses.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from toporlogy.configuration.helpers import (
    _case_weights, _field_source, _interfaces, _region_mask, _robust_set, _vector, checked_selector, material_model,
)
from toporlogy.configuration.registry import BuildContext, register


def register_responses() -> None:
    """Register the responses components."""
    length = "length"

    # ---------------------------------------------------------------- responses
    def _need_modal(ctx: BuildContext, what: str) -> None:
        from toporlogy.fem.modal import Modal
        if not isinstance(ctx.physics, Modal):
            raise ValueError(f"{what} needs physics modal")

    def robust_compliance(ctx: BuildContext, realisations: Sequence[str] = ("eroded", "intermediate", "dilated"),
                          aggregate: str = "ks", rho: float = 50.0, p: float = 16.0, scale: float | None = None,
                          weights: list | None = None) -> Any:
        """Worst-case compliance over the eroded / intermediate / dilated realisations (one solve each); thresholds
        from the robust block; weights null = load-case weights."""
        from toporlogy.optimization.robust import RobustCompliance
        return RobustCompliance(_robust_set(ctx), weights=_case_weights(ctx, weights), realisations=realisations,
                                aggregate=aggregate, rho=rho, p=p, scale=scale)

    def dilated_volume(ctx: BuildContext, weighted: bool = False) -> Any:
        """Volume of the dilated realisation (robust formulation); robust_continuation keeps the blueprint at the
        constraint's initial bound."""
        from toporlogy.optimization.robust import DilatedVolume
        return DilatedVolume(_robust_set(ctx), weighted)

    def local_volume(ctx: BuildContext, t_max: float | None = None, radius: float | None = None,
                     member: str | None = None, p: float = 16.0, aggregate: str = "pmean", adaptive: float = 0.0,
                     outside: Any = "void", source: str = "xphys") -> Any:
        """Maximum member thickness: local volume fraction in a ball <= alpha. t_max (length) sets radius and the
        constraint bound alpha (max_thickness_settings); or radius (elements) with an explicit upper. source:
        xphys | dilated | intermediate | eroded."""
        from toporlogy.optimization.lengthscale import LocalVolume, max_thickness_settings
        d = ctx.domain
        s = None
        if t_max is not None:
            s = max_thickness_settings(t_max, d.h, d.dim, radius, member)
            radius = s["radius"]
        elif radius is None:
            raise ValueError("give t_max (length) or radius (elements)")
        r = LocalVolume(radius, p=p, aggregate=aggregate, adaptive=adaptive, outside=outside,
                        source=_field_source(ctx, source))
        if s is not None:
            r.config_upper = float(s["alpha"])
            r.settings = s
        return r

    def grey_level(ctx: BuildContext, source: str = "xphys") -> Any:
        """M_nd = mean 4 x (1 - x) over the design elements (monitor or constraint, e.g. <= 0.05)."""
        from toporlogy.optimization.lengthscale import GreyLevel
        return GreyLevel(_field_source(ctx, source))

    def conduction_connectivity(ctx: BuildContext, sink: str | None = None, kmin_ratio: float = 1e-3,
                                penal: float = 3.0, aggregate: str = "pnorm", p: float = 8.0,
                                eta: float | None = 0.7, reference: Any = "solid") -> Any:
        """One body: virtual heat conduction from every interface to the sink (default the first); upper default
        20. Interfaces: connectivity.interfaces, else one per support and load."""
        from toporlogy.optimization.connectivity import ConductionConnectivity
        r = ConductionConnectivity(ctx.domain, _interfaces(ctx), sink, kmin_ratio=kmin_ratio, penal=penal,
                                   aggregate=aggregate, p=p, eta=eta, reference=reference)
        r.config_upper = 20.0
        return r

    def coupling_compliance(ctx: BuildContext, held: str | None = None, aggregate: str = "pnorm", p: float = 8.0,
                            eta: float | None = 0.7, reference: Any = "first") -> Any:
        """One body: auxiliary cases with the held interface clamped and unit forces on the others; upper default
        2.5 (x first value)."""
        from toporlogy.optimization.connectivity import CouplingCompliance
        r = CouplingCompliance(ctx.domain, _interfaces(ctx), held, nu=ctx.material["nu"], aggregate=aggregate, p=p,
                               eta=eta, reference=reference)
        r.config_upper = 2.5
        return r

    def frequency(ctx: BuildContext, k: int = 1, E_unit: float = 1.0, rho_unit: float = 1.0,
                  length_unit: float = 1.0) -> Any:
        """f_k in Hz (the model's E, rho and length units given in SI multiples)."""
        from toporlogy.optimization.dynamics import Frequency
        _need_modal(ctx, "frequency")
        return Frequency(k, E_unit, rho_unit, length_unit)

    def eigenvalue(ctx: BuildContext, k: int = 1) -> Any:
        """lambda_k = omega_k^2 in model units (only differentiable while simple)."""
        from toporlogy.optimization.dynamics import Eigenvalue
        _need_modal(ctx, "eigenvalue")
        return Eigenvalue(k)

    def min_eigenvalue_ks(ctx: BuildContext, n_modes: int = 3, rho: float = 50.0, scale: float | None = None,
                          kind: str = "ks") -> Any:
        """Smooth lower bound of min(lambda_1 .. lambda_n) (KS or p-norm); maximise it with type maximize."""
        from toporlogy.optimization.dynamics import MinEigenvalueKS
        _need_modal(ctx, "min_eigenvalue_ks")
        return MinEigenvalueKS(n_modes, rho, scale, kind)

    def maximize(ctx: BuildContext, response: Any) -> Any:
        """-response (the study minimises), e.g. objective {type: maximize, response: min_eigenvalue_ks}."""
        from toporlogy.optimization.dynamics import Negated
        return Negated(response)

    def min_frequency(ctx: BuildContext, f_min: float, n_modes: int = 1, rho: float = 50.0, E_unit: float = 1.0,
                      rho_unit: float = 1.0, length_unit: float = 1.0) -> Any:
        """Constraint f_1 >= f_min (Hz) as 1 - lambda_low / lambda_min <= 0 (upper 0, scale 1 by default)."""
        from toporlogy.optimization.dynamics import FrequencyMargin
        _need_modal(ctx, "min_frequency")
        r = FrequencyMargin(f_min, n_modes, rho, E_unit, rho_unit, length_unit)
        r.config_upper = 0.0
        r.config_scale = 1.0
        return r

    # ---------------------------------------------------------------- stress, buckling, displacement, algebra
    def _von_mises(ctx: BuildContext, sigma_lim: float | None, form: str, q: float, penal: float, exclude: Any,
                   cases: list | None, criterion: str | None = None) -> Any:
        """The local stress field: VonMises (isotropic, plane stress / 3-D, no criterion: the original path) or
        FailureIndex (any criterion, orthotropic or plane strain). The limit is sigma_lim or material.strength."""
        from toporlogy.core.material import Strength
        from toporlogy.optimization.strength import make_criterion
        from toporlogy.optimization.stress import FailureIndex, VonMises
        mask = None if not exclude else _region_mask(exclude, ctx.domain)
        mm = material_model(ctx)
        if sigma_lim is not None and mm.strength is not None:
            raise ValueError("give sigma_lim or material.strength, not both")
        if criterion is None and mm.elasticity is None:
            if sigma_lim is None:
                if not isinstance(mm.strength, Strength):
                    raise ValueError("stress: give sigma_lim or material.strength")
                sigma_lim = mm.strength.allowable_tension
            return VonMises(ctx.domain, sigma_lim, nu=ctx.material["nu"], E0=ctx.material["E0"], form=form, q=q,
                            penal=penal, exclude=mask, cases=cases)
        kind = criterion or ("tsai_wu" if mm.orthotropic else "von_mises")
        strength = mm.strength
        if strength is None:
            if sigma_lim is None or kind not in ("von_mises", "tresca"):
                raise ValueError(f"stress criterion {kind}: needs material.strength"
                                 + ("" if kind in ("von_mises", "tresca") else " (sigma_lim is not enough)"))
            strength = Strength(float(sigma_lim))
        return FailureIndex(ctx.domain, make_criterion(kind, strength, mm.R), D=mm.elasticity, nu=ctx.material["nu"],
                            szz_row=mm.szz_row, E0=ctx.material["E0"], form=form, q=q, penal=penal, exclude=mask,
                            cases=cases)

    def stress(ctx: BuildContext, sigma_lim: float | None = None, aggregate: str = "pnorm", parameter: float = 8.0,
               q: float = 0.5, adaptive: float | None = 0.5, exclude: Any = None, cases: list | None = None,
               criterion: str | None = None) -> Any:
        """Largest relaxed failure index (centroid, solid stress x^q), aggregated: max | pnorm | pmean | ks |
        ks_lower | induced_exponential | induced_power with its parameter. The index is von Mises / sigma_lim, or
        with criterion (von_mises | tresca | drucker_prager | raghava | tsai_wu) the criterion's index against
        material.strength (sigma_lim null = the material's allowable). adaptive: damping of the scale that ties the
        aggregate to the true maximum (null = none). exclude: shapes whose elements are left out (point loads,
        supports). Upper default 1."""
        from toporlogy.optimization.local import Aggregated
        r = Aggregated(_von_mises(ctx, sigma_lim, "ratio", q, 3.0, exclude, cases, criterion), aggregate, parameter,
                       adaptive, name="stress")
        r.config_upper = 1.0
        r.config_scale = 1.0
        return r

    def stress_augmented_lagrangian(ctx: BuildContext, objective: Any, sigma_lim: float | None = None,
                                    form: str = "polynomial", mu0: float = 10.0, alpha: float = 1.1,
                                    mu_max: float = 1.0e4, penal: float = 3.0, q: float = 0.5, exclude: Any = None,
                                    cases: list | None = None, reference: Any = "first",
                                    criterion: str | None = None) -> Any:
        """objective / reference + augmented Lagrangian of the local stress constraints (one per element and
        case; form polynomial | margin). Use it as the objective with schedule multiplier_update. sigma_lim and
        criterion as for stress."""
        from toporlogy.optimization.local import AugmentedLagrangian
        return AugmentedLagrangian(objective, _von_mises(ctx, sigma_lim, form, q, penal, exclude, cases, criterion),
                                   mu0, alpha, mu_max, reference, name="stress_al")

    def buckling(ctx: BuildContext, load_factor: float | None = None, n_modes: int = 6, rho: float = 50.0,
                 case: int = 0, penal_g: float = 3.0, scale: float | None = None,
                 preload_case: int | None = None, exclude: Any = None) -> Any:
        """KS upper bound of 1 / lambda_1 (linear buckling of load case `case`; preload_case: a case whose load
        stays fixed while `case` is scaled); with load_factor the constraint lambda_1 >= load_factor is this
        response <= 1 / load_factor (default upper and scale). exclude: shapes whose elements carry no stress
        stiffness (ends of clamped patches, re-entrant corners, point loads: singular stresses localise the mode
        there). Plain elasticity physics with the direct solver only."""
        from toporlogy.optimization.buckling import BucklingKS
        mask = None if not exclude else _region_mask(exclude, ctx.domain)
        r = BucklingKS(n_modes, rho, case, penal_g, scale, E0=ctx.material["E0"], preload_case=preload_case,
                       exclude=mask)
        if load_factor is not None:
            if not load_factor > 0:
                raise ValueError("load_factor > 0 is required")
            r.config_upper = 1.0 / float(load_factor)
            r.config_scale = float(load_factor)
        return r

    def displacement(ctx: BuildContext, where: Any, direction: Any, case: int = 0, name: str = "displacement") -> Any:
        """Mean displacement of the selected nodes along `direction` (axis or vector) in load case `case`:
        mechanism output (objective) or deflection bound (constraint)."""
        from toporlogy.optimization.compose import Displacement
        d = ctx.domain
        nodes = np.asarray(checked_selector(ctx, where).select(d), np.int64)
        e = _vector(direction, d.dim)
        dofs = np.concatenate([d.dim * nodes + k for k in range(d.dim) if e[k] != 0.0])
        w = np.concatenate([np.full(nodes.size, e[k] / nodes.size) for k in range(d.dim) if e[k] != 0.0])
        return Displacement(dofs, w, case, name)

    def weighted_sum(ctx: BuildContext, responses: Any, weights: list | None = None, references: Any = None) -> Any:
        """sum_i w_i r_i / ref_i; references: null, numbers, or "first" (each value at the first evaluation)."""
        from toporlogy.optimization.compose import WeightedSum
        return WeightedSum(responses, weights, references)

    def end_compliance(ctx: BuildContext, weights: list | None = None) -> Any:
        """Weighted end compliance sum_c w_c F_c.u_c at the converged large-displacement equilibrium (physics
        nonlinear_elasticity); weights null = the load-case weights."""
        from toporlogy.fem.nonlinear import NonlinearElasticity
        from toporlogy.optimization.nonlinear import EndCompliance
        if not isinstance(ctx.physics, NonlinearElasticity):
            raise ValueError("end_compliance needs physics nonlinear_elasticity (use compliance for linear physics)")
        return EndCompliance(_case_weights(ctx, weights))

    register("response", "end_compliance", end_compliance)

    def end_strain_energy(ctx: BuildContext, weights: list | None = None) -> Any:
        """Weighted stored energy sum_c w_c W(u_c) at the converged non-linear equilibrium (physics
        nonlinear_elasticity, hyperelastic material); weights null = 1 per case (docs/olcumler/nonlineer_cozucu.md)."""
        from toporlogy.fem.nonlinear import NonlinearElasticity
        from toporlogy.optimization.nonlinear import EndStrainEnergy
        if not isinstance(ctx.physics, NonlinearElasticity):
            raise ValueError("end_strain_energy needs physics nonlinear_elasticity")
        return EndStrainEnergy(weights)

    register("response", "end_strain_energy", end_strain_energy)

    def transient_thermal_compliance(ctx: BuildContext) -> Any:
        """Time-averaged thermal compliance (1/t_N) sum_n w_n q.T_n (trapezoid) of physics transient_heat
        (optimization.transient, docs/olcumler/gecici_study_onkayit.md)."""
        from toporlogy.fem.transient_heat import TransientHeatPhysics
        from toporlogy.optimization.transient import TransientThermalCompliance
        if not isinstance(ctx.physics, TransientHeatPhysics):
            raise ValueError("transient_thermal_compliance needs physics transient_heat")
        return TransientThermalCompliance()

    def dynamic_compliance(ctx: BuildContext) -> Any:
        """Time-averaged dynamic compliance (1/t_N) sum_n w_n f(t_n).u_n (trapezoid) of physics dynamics_newmark
        (optimization.transient, docs/olcumler/gecici_study_onkayit.md)."""
        from toporlogy.fem.dynamics_newmark import NewmarkPhysics
        from toporlogy.optimization.transient import DynamicCompliance
        if not isinstance(ctx.physics, NewmarkPhysics):
            raise ValueError("dynamic_compliance needs physics dynamics_newmark")
        return DynamicCompliance()

    register("response", "transient_thermal_compliance", transient_thermal_compliance)
    register("response", "dynamic_compliance", dynamic_compliance)

    def centre_of_mass(ctx: BuildContext, axis: str = "x") -> Any:
        """Coordinate of the centre of mass of the physical density along axis x | y | z (length unit; material rho
        as the density of solid, x̄ linear). No state solve."""
        from toporlogy.optimization.mass_responses import CentreOfMass
        return CentreOfMass(axis, float(ctx.material.get("rho", 1.0)))

    def inertia(ctx: BuildContext, i: str = "z", j: str | None = None) -> Any:
        """Component I_ij of the inertia tensor about the centre of mass (mass·length²; j null = i). No state
        solve."""
        from toporlogy.optimization.mass_responses import InertiaComponent
        return InertiaComponent(i, j, float(ctx.material.get("rho", 1.0)))

    def scenario_compliance(ctx: BuildContext, measure: str = "mean", k: float = 0.0, alpha: float = 0.95,
                            scale: float = 1.0) -> Any:
        """Risk measure of the compliances of all load cases, each case one scenario of an uncertain load: measure
        mean | mean_std (mean + k std) | cvar (level alpha); one factorisation for every scenario
        (optimization.uq, docs/olcumler/laminat_uq.md)."""
        from toporlogy.optimization.uq import ScenarioCompliance
        return ScenarioCompliance(measure, k, alpha, scale)

    def compliance_case(ctx: BuildContext, case: int = 0) -> Any:
        """Compliance F_c.u_c of one load case (0-based): objective or epsilon-constraint of a trade-off between load
        cases (optimization.pareto)."""
        from toporlogy.optimization.pareto import CaseCompliance
        if not 0 <= int(case) < int(ctx.n_cases):
            raise ValueError(f"case {case}: the config has {ctx.n_cases} load case(s)")
        return CaseCompliance(int(case))

    register("response", "scenario_compliance", scenario_compliance)
    register("response", "compliance_case", compliance_case)
    def effective_stiffness(ctx: BuildContext, i: int = 0, j: int = 0, negate: bool = False) -> Any:
        """Component C_ij (Voigt) of the design homogenised as a periodic cell (the whole box; fem.homogenization);
        negate: -C_ij, for a lower bound. No state solve."""
        from toporlogy.optimization.homogenization_responses import EffectiveStiffness
        return EffectiveStiffness(i, j, float(ctx.material["nu"]), negate)

    def effective_poisson(ctx: BuildContext, convention: str | None = None, load: int = 0,
                          transverse: int = 1) -> Any:
        """Effective Poisson ratio of the periodic cell (fem.homogenization). No state solve. convention (required):
        uniaxial = -eps_transverse/eps_load under uniaxial stress along load (2-D and 3-D); stiffness_ratio = the
        legacy C[transverse, load]/C[load, load] (C12/C11; 2-D)."""
        from toporlogy.optimization.homogenization_responses import POISSON_CONVENTIONS, EffectivePoisson
        if convention is None:
            raise ValueError("effective_poisson needs 'convention': uniaxial (-eps_transverse/eps_load under uniaxial "
                             "stress) or stiffness_ratio (the legacy C12/C11, which is nu21 of an uncoupled cell); "
                             f"choose one of {' | '.join(POISSON_CONVENTIONS)}")
        return EffectivePoisson(convention, float(ctx.material["nu"]), load=load, transverse=transverse)

    register("response", "effective_stiffness", effective_stiffness)
    register("response", "effective_poisson", effective_poisson)
    register("response", "centre_of_mass", centre_of_mass)
    register("response", "inertia", inertia)
    register("response", "stress", stress, nested={"exclude": "shape[]"}, units={"sigma_lim": "force/length^2"})
    register("response", "stress_augmented_lagrangian", stress_augmented_lagrangian,
             nested={"objective": "response", "exclude": "shape[]"}, types={"reference": None},
             units={"sigma_lim": "force/length^2"})
    register("response", "buckling", buckling)
    register("response", "displacement", displacement, nested={"where": "selector"}, types={"direction": None})
    register("response", "weighted_sum", weighted_sum, nested={"responses": "response[]"},
             types={"references": None})
    register("response", "robust_compliance", robust_compliance, types={"realisations": list})
    register("response", "dilated_volume", dilated_volume)
    register("response", "local_volume", local_volume, types={"outside": None},
             units={"t_max": length, "radius": "elements"})
    register("response", "grey_level", grey_level)
    register("response", "conduction_connectivity", conduction_connectivity, types={"reference": None})
    register("response", "coupling_compliance", coupling_compliance, types={"reference": None})
    register("response", "frequency", frequency)
    register("response", "eigenvalue", eigenvalue)
    register("response", "min_eigenvalue_ks", min_eigenvalue_ks)
    register("response", "maximize", maximize, nested={"response": "response"})
    register("response", "min_frequency", min_frequency, units={"f_min": "Hz"})

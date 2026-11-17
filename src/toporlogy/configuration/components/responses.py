"""Component constructors: responses (moved from toporlogy.registry).
"""
from __future__ import annotations

from typing import Any

from toporlogy.configuration.helpers import _case_weights, _field_source, _interfaces, _robust_set
from toporlogy.configuration.registry import BuildContext, register


def register_responses() -> None:
    """Register the responses components (moved from registry._register_features)."""
    length = "length"

    # ---------------------------------------------------------------- responses
    def _need_modal(ctx: BuildContext, what: str) -> None:
        from toporlogy.fem.modal import Modal
        if not isinstance(ctx.physics, Modal):
            raise ValueError(f"{what} needs physics modal")

    def robust_compliance(ctx: BuildContext, realisations: list = ("eroded", "intermediate", "dilated"),
                          aggregate: str = "ks", rho: float = 50.0, p: float = 16.0, scale: float | None = None,
                          weights: list | None = None) -> Any:
        """Worst-case compliance over the eroded / intermediate / dilated realisations (one solve each); thresholds
        from the robust block; weights null = load-case weights."""
        from toporlogy.robust import RobustCompliance
        return RobustCompliance(_robust_set(ctx), weights=_case_weights(ctx, weights), realisations=realisations,
                                aggregate=aggregate, rho=rho, p=p, scale=scale)

    def dilated_volume(ctx: BuildContext, weighted: bool = False) -> Any:
        """Volume of the dilated realisation (robust formulation); robust_continuation keeps the blueprint at the
        constraint's initial bound."""
        from toporlogy.robust import DilatedVolume
        return DilatedVolume(_robust_set(ctx), weighted)

    def local_volume(ctx: BuildContext, t_max: float | None = None, radius: float | None = None,
                     member: str | None = None, p: float = 16.0, aggregate: str = "pmean", adaptive: float = 0.0,
                     outside: Any = "void", source: str = "xphys") -> Any:
        """Maximum member thickness: local volume fraction in a ball <= alpha. t_max (length) sets radius and the
        constraint bound alpha (max_thickness_settings); or radius (elements) with an explicit upper. source:
        xphys | dilated | intermediate | eroded."""
        from toporlogy.lengthscale import LocalVolume, max_thickness_settings
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
        from toporlogy.lengthscale import GreyLevel
        return GreyLevel(_field_source(ctx, source))

    def conduction_connectivity(ctx: BuildContext, sink: str | None = None, kmin_ratio: float = 1e-3,
                                penal: float = 3.0, aggregate: str = "pnorm", p: float = 8.0,
                                eta: float | None = 0.7, reference: Any = "solid") -> Any:
        """One body: virtual heat conduction from every interface to the sink (default the first); upper default
        20. Interfaces: connectivity.interfaces, else one per support and load."""
        from toporlogy.connectivity import ConductionConnectivity
        r = ConductionConnectivity(ctx.domain, _interfaces(ctx), sink, kmin_ratio=kmin_ratio, penal=penal,
                                   aggregate=aggregate, p=p, eta=eta, reference=reference)
        r.config_upper = 20.0
        return r

    def coupling_compliance(ctx: BuildContext, held: str | None = None, aggregate: str = "pnorm", p: float = 8.0,
                            eta: float | None = 0.7, reference: Any = "first") -> Any:
        """One body: auxiliary cases with the held interface clamped and unit forces on the others; upper default
        2.5 (x first value)."""
        from toporlogy.connectivity import CouplingCompliance
        r = CouplingCompliance(ctx.domain, _interfaces(ctx), held, nu=ctx.material["nu"], aggregate=aggregate, p=p,
                               eta=eta, reference=reference)
        r.config_upper = 2.5
        return r

    def frequency(ctx: BuildContext, k: int = 1, E_unit: float = 1.0, rho_unit: float = 1.0,
                  length_unit: float = 1.0) -> Any:
        """f_k in Hz (the model's E, rho and length units given in SI multiples)."""
        from toporlogy.dynamics import Frequency
        _need_modal(ctx, "frequency")
        return Frequency(k, E_unit, rho_unit, length_unit)

    def eigenvalue(ctx: BuildContext, k: int = 1) -> Any:
        """lambda_k = omega_k^2 in model units (only differentiable while simple)."""
        from toporlogy.dynamics import Eigenvalue
        _need_modal(ctx, "eigenvalue")
        return Eigenvalue(k)

    def min_eigenvalue_ks(ctx: BuildContext, n_modes: int = 3, rho: float = 50.0, scale: float | None = None,
                          kind: str = "ks") -> Any:
        """Smooth lower bound of min(lambda_1 .. lambda_n) (KS or p-norm); maximise it with type maximize."""
        from toporlogy.dynamics import MinEigenvalueKS
        _need_modal(ctx, "min_eigenvalue_ks")
        return MinEigenvalueKS(n_modes, rho, scale, kind)

    def maximize(ctx: BuildContext, response: Any) -> Any:
        """-response (the study minimises), e.g. objective {type: maximize, response: min_eigenvalue_ks}."""
        from toporlogy.dynamics import Negated
        return Negated(response)

    def min_frequency(ctx: BuildContext, f_min: float, n_modes: int = 1, rho: float = 50.0, E_unit: float = 1.0,
                      rho_unit: float = 1.0, length_unit: float = 1.0) -> Any:
        """Constraint f_1 >= f_min (Hz) as 1 - lambda_low / lambda_min <= 0 (upper 0, scale 1 by default)."""
        from toporlogy.dynamics import FrequencyMargin
        _need_modal(ctx, "min_frequency")
        r = FrequencyMargin(f_min, n_modes, rho, E_unit, rho_unit, length_unit)
        r.config_upper, r.config_scale = 0.0, 1.0
        return r

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

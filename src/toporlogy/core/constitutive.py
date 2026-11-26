"""Constitutive protocol and the material laws of the solid physics (02_MATERIAL_DESIGN §2, 04 §3).

One call per batch of material points::

    model.evaluate(kinematics, params, committed_state, request)
        -> MaterialResponse(stress, tangent, energy, trial_state, dissipation, diagnostics)

Contracts (``docs/olcumler/nonlineer_cozucu_onkayit.md`` §1):

* Kinematics are explicit. ``small``: ε = sym ∇u, the stress is the Cauchy stress σ and the tangent dσ/dε.
  ``finite``: the displacement gradient H = F − I (Total Lagrangian), the stress is the second Piola–Kirchhoff S and
  the tangent ℂ = dS/dE (E the Green–Lagrange strain). No model is asked for a measure it does not define; other
  measures (PK1, Cauchy, Kirchhoff) follow from :mod:`toporlogy.core.tensor`.
* Voigt order of :mod:`toporlogy.core.tensor`: 2-D ``[xx, yy, xy]``, 3-D ``[xx, yy, zz, yz, xz, xy]``; stresses
  with the physical shear component, strains with the engineering shear (so ``stress = tangent @ strain`` for a
  linear law). 2-D kinematics carry ``plane``: ``"strain"`` (ε_zz = 0, F_zz = 1) or ``"stress"`` (σ_zz = 0; only
  models that can solve it accept it).
* Unit modulus: every law here is the unit-modulus law; the physics multiplies stress, tangent, energy and
  dissipation by the element modulus E_e. For J2 the yield stress and hardening are given relative to E = 1, so the
  scaling is exact and the internal variables do not depend on E_e.
* State: ``committed_state`` is a mapping of arrays (one row per point) that a model must not modify; it is passed
  read-only (``writeable = False``) by :class:`StateStore`. ``trial_state`` is a new mapping. The caller commits a
  trial only after a converged load step (:meth:`StateStore.commit`) and drops it otherwise (rollback). Stateless
  models return ``trial_state = None``.
* Energy is optional (``request.energy``): defined for the hyperelastic laws (conservative) and as stored energy
  (elastic + hardening) for J2, whose dissipation increment is returned separately.
* Failures: :class:`InadmissibleState` (outside the model's domain: J ≤ 0, no plane-stress solution; code
  ``invalid_deformation``) and :class:`ReturnMappingFailure` (local return mapping not converged; code
  ``return_mapping``). Both derive from :class:`ConstitutiveFailure`, which the general Newton solver
  (:mod:`toporlogy.solvers.newton`) turns into a failed iteration with that code; it does not know the material.

Models: :class:`SaintVenantKirchhoff` (finite, any unit D; per-point D by index), :class:`NeoHooke` (finite,
compressible, isotropic; plane strain, plane stress by a local Newton on S₃₃ = 0, 3-D), :class:`LinearElastic`
(small strain, any unit D), :class:`J2Plasticity` (small strain, associative von Mises plasticity with isotropic
hardening, linear plus optional Voce saturation; radial return with the consistent algorithmic tangent; 3-D and
plane strain; plane stress is refused — it needs a local condensation that is not implemented). J2 is a
small-strain model: no large-deformation support is claimed. Optional protocol method of a path-dependent model,
``evaluate_vjp(kin, params, committed, stress_bar, state_bar, dissipation_bar) -> (grad_bar, committed_bar)``: the
vector-Jacobian product of one update (J2: :meth:`J2Plasticity.evaluate_vjp`, closed form); the design-sensitivity
path adjoint of :mod:`toporlogy.fem.nonlinear` needs it.

Oracles: :func:`energy_oracle` differentiates a torch energy ψ(E_voigt) twice (autograd) for the stress and tangent
of an energy-based model; :func:`svk_energy_torch` and :func:`neo_hooke_energy_torch` are written from the energy
alone (not from the analytic stress). :func:`drive_material_point` integrates a mixed strain/stress-controlled
material point path with the model's tangent (the reference of the FE plasticity tests).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.linear_elastic import isotropic_moduli
from toporlogy.references import cite

__all__ = ["KINEMATICS", "ConstitutiveFailure", "InadmissibleState", "ReturnMappingFailure", "Kinematics",
           "MaterialRequest", "MaterialResponse", "ConstitutiveModel", "StateStore", "SaintVenantKirchhoff", "NeoHooke",
           "LinearElastic",
           "J2Plasticity", "neo_hooke_response", "energy_oracle", "svk_energy_torch", "neo_hooke_energy_torch",
           "MaterialPointHistory", "drive_material_point"]

KINEMATICS = ("small", "finite")
PLANES = ("3d", "stress", "strain")


# ----------------------------------------------------------------------------------------------- failures
class ConstitutiveFailure(ArithmeticError):
    """A material point evaluation that has no admissible answer; ``code`` names the failure kind."""

    code = "constitutive"


class InadmissibleState(ConstitutiveFailure):
    """A state outside the model's domain (inverted element det F ≤ 0, no plane-stress solution)."""

    code = "invalid_deformation"


class ReturnMappingFailure(ConstitutiveFailure):
    """The local return mapping did not converge (or produced a non-finite increment)."""

    code = "return_mapping"


# ----------------------------------------------------------------------------------------------- protocol types
@dataclass(frozen=True)
class Kinematics:
    """Displacement gradient ``grad`` (n, dim, dim), ∂u/∂X, of n points; ``measure`` small | finite; ``plane`` 3d (dim
    3) or stress | strain (dim 2)."""

    grad: np.ndarray
    measure: str
    plane: str = "3d"

    def __post_init__(self) -> None:
        if self.measure not in KINEMATICS:
            raise ValueError(f"kinematics measure: {' | '.join(KINEMATICS)}, got {self.measure!r}")
        if self.grad.ndim != 3 or self.grad.shape[1] != self.grad.shape[2] or self.grad.shape[1] not in (2, 3):
            raise ValueError(f"grad must be (n, dim, dim) with dim 2 or 3, got {self.grad.shape}")
        if (self.dim == 3) != (self.plane == "3d") or self.plane not in PLANES:
            raise ValueError(f"plane {self.plane!r} does not fit dim {self.dim} (3d in 3-D; stress | strain in 2-D)")

    @property
    def dim(self) -> int:
        return int(self.grad.shape[1])

    @property
    def n(self) -> int:
        return int(self.grad.shape[0])


@dataclass(frozen=True)
class MaterialRequest:
    """What to compute: ``energy`` (if the model defines one), ``tangent``; ``tangent_kind`` algorithmic (the
    consistent tangent of the update, default) or continuum (the rate tangent; for diagnostics only)."""

    energy: bool = False
    tangent: bool = True
    tangent_kind: str = "algorithmic"


_DEFAULT_REQUEST = MaterialRequest()


@dataclass
class MaterialResponse:
    """Stress (n, nv), tangent (n, nv, nv) or None, energy (n,) or None, trial state (mapping) or None (stateless),
    dissipation increment (n,) or None (not defined), diagnostics (name → (n,) array)."""

    stress: np.ndarray
    tangent: np.ndarray | None
    energy: np.ndarray | None = None
    trial_state: dict[str, np.ndarray] | None = None
    dissipation: np.ndarray | None = None
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


class ConstitutiveModel(Protocol):
    """The interface every law implements (module notes)."""

    name: str
    kinematics: str              # small | finite
    conservative: bool           # an energy whose gradient is the stress exists and is path independent
    state_fields: tuple[str, ...]

    def initial_state(self, n: int) -> dict[str, np.ndarray]: ...

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest) -> MaterialResponse: ...


def _readonly(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in state.items():
        a = np.array(v, copy=True)
        a.setflags(write=False)
        out[k] = a
    return out


class StateStore:
    """Committed internal variables of ``n`` points (read-only arrays) and the trial of the current iteration.

    :meth:`set_trial` keeps the trial of the last evaluation; :meth:`commit` makes it the committed state (only after a
    converged load step); :meth:`rollback` drops it. :meth:`checkpoint` / :meth:`restore` copy the committed state."""

    def __init__(self, model: Any, n: int) -> None:
        self.fields: tuple[str, ...] = tuple(model.state_fields)
        self.committed: dict[str, np.ndarray] = _readonly(model.initial_state(n))
        self.trial: dict[str, np.ndarray] | None = None
        self.n_commits = 0
        self.n_rollbacks = 0

    def set_trial(self, trial: Mapping[str, np.ndarray] | None) -> None:
        self.trial = None if trial is None else {k: np.asarray(v) for k, v in trial.items()}

    def commit(self) -> None:
        if self.trial is not None:
            self.committed = _readonly(self.trial)
        self.trial = None
        self.n_commits += 1

    def rollback(self) -> None:
        self.trial = None
        self.n_rollbacks += 1

    def checkpoint(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.committed.items()}

    def restore(self, cp: Mapping[str, np.ndarray]) -> None:
        self.committed = _readonly(cp)
        self.trial = None


# ----------------------------------------------------------------------------------------------- finite strain
# Every strain measure is formed from the displacement gradient H = F − I without subtracting I: E = ½(H + Hᵀ + HᵀH)
# (core.tensor.green_lagrange), J − 1 from the invariants of H (core.tensor.jacobian_minus_one). Forming ½(FᵀF − I)
# loses about log10(1/|H|) digits (measured: relative error of f_int 1.4e-6 at |H| = 1e-8 in 3-D), which leaves a
# residual floor no convergence test can pass at small loads.
_VA = np.array([p[0] for p in T.VOIGT_3D])
_VB = np.array([p[1] for p in T.VOIGT_3D])
_VDELTA = (_VA == _VB).astype(float)


@cite("feng_zhang_khandelwal_2025", note="regularised neo-Hooke energy, Eqs. (9)-(10)")
def neo_hooke_response(E: np.ndarray, Jm1: np.ndarray, kappa: float, mu: float,
                       energy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """S (n, 3, 3), ℂ = 2 ∂S/∂C in Voigt form (n, 6, 6; engineering shear) and optionally ψ (n,) of
    ψ = ½κ(J − 1)² + ½μ(Ī₁ − 3), Ī₁ = J^(−2/3) tr C, at Green strains E (n, 3, 3) with J − 1 = ``Jm1``
    (C = I + 2E). Derived here:

        S = κ J (J − 1) C⁻¹ + 2μ J^(−2/3) C⁻¹ dev E        (μ J^(−2/3) (I − ⅓ I₁ C⁻¹) written without cancellation)
        ℂ = κ [(2J − 1) J C⁻¹⊗C⁻¹ − 2J(J − 1) 𝕀] + μ J^(−2/3) [−⅔ (I⊗C⁻¹ + C⁻¹⊗I) + (2/9) I₁ C⁻¹⊗C⁻¹ + ⅔ I₁ 𝕀]

    with 𝕀_ijkl = ½(C⁻¹_ik C⁻¹_jl + C⁻¹_il C⁻¹_jk). Raises :class:`InadmissibleState` for J ≤ 0."""
    J = 1.0 + Jm1
    if not np.all(J > 0):
        raise InadmissibleState("det F <= 0")
    eye = np.eye(3)
    C = eye + 2.0 * E
    Ci = np.linalg.inv(C)
    trE = np.trace(E, axis1=1, axis2=2)
    dev = E - (trE / 3.0)[:, None, None] * eye
    I1 = 3.0 + 2.0 * trE
    Jm = J ** (-2.0 / 3.0)
    iso_s = Ci @ dev
    S = (kappa * J * Jm1)[:, None, None] * Ci + (mu * Jm)[:, None, None] * (iso_s + np.swapaxes(iso_s, 1, 2))
    a, c = _VA, _VB                                    # Voigt pair (i, j) = (a[I], c[I]); (k, l) = (a[J], c[J])
    civ = Ci[:, a, c]                                  # (n, 6)
    CiCi = civ[:, :, None] * civ[:, None, :]
    sym = 0.5 * (Ci[:, a[:, None], a[None, :]] * Ci[:, c[:, None], c[None, :]]
                 + Ci[:, a[:, None], c[None, :]] * Ci[:, c[:, None], a[None, :]])
    ICi = _VDELTA[None, :, None] * civ[:, None, :]
    CiI = civ[:, :, None] * _VDELTA[None, None, :]
    b = (slice(None), None, None)
    vol = kappa * ((2.0 * J - 1.0) * J)[b] * CiCi - (2.0 * kappa * J * Jm1)[b] * sym
    iso = (mu * Jm)[b] * (-2.0 / 3.0 * (ICi + CiI) + (2.0 / 9.0) * I1[b] * CiCi + (2.0 / 3.0) * I1[b] * sym)
    psi = None
    if energy:                                       # Ī₁ − 3 = J^(−2/3)·2 tr E + 3 (J^(−2/3) − 1)
        psi = 0.5 * kappa * Jm1 ** 2 + 0.5 * mu * (Jm * 2.0 * trE + 3.0 * np.expm1(-2.0 / 3.0 * np.log1p(Jm1)))
    return S, vol + iso, psi


def _check_kin(model: Any, kin: Kinematics) -> None:
    if kin.measure != model.kinematics:
        raise ValueError(f"{model.name} is a {model.kinematics}-strain model, got {kin.measure} kinematics")


class SaintVenantKirchhoff:
    """S = D E (Voigt, engineering shear) with a unit-modulus D (nv, nv), or the distinct matrices (m, nv, nv) of a
    per-point field with ``params["index"]`` (n,) picking each point's matrix; ψ = ½ Eᵀ D E. In 2-D the given D
    already holds the plane condition (the 2-D model with the plane-stress constants: S₃₃ = 0 is imposed on the linear
    law, not on a 3-D SVK body). Physical limits: SVK softens without bound in compression (S → 0 as a stretch → 0
    for uniaxial compression, the energy does not grow as J → 0) and loses ellipticity there; it is a moderate-strain
    model."""

    name = "svk"
    kinematics = "finite"
    conservative = True
    state_fields: tuple[str, ...] = ()

    def __init__(self, D: np.ndarray) -> None:
        self.D = np.asarray(D, float)
        if self.D.ndim not in (2, 3) or self.D.shape[-1] != self.D.shape[-2] or self.D.shape[-1] not in (3, 6):
            raise ValueError(f"D must be (nv, nv) or (m, nv, nv) with nv 3 or 6, got {self.D.shape}")

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {}

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None = None,
                 committed: Mapping[str, np.ndarray] | None = None,
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        _check_kin(self, kin)
        H = kin.grad
        Ev = T.strain_to_voigt(T.green_lagrange(H), rtol=None)          # symmetric by construction
        if self.D.ndim == 2:
            S = Ev @ self.D
            C = np.broadcast_to(self.D, (len(H),) + self.D.shape)
        else:
            which = None if params is None else params.get("index")
            assert which is not None, "per-point D needs params['index']"
            C = self.D[which]
            S = (C @ Ev[:, :, None])[:, :, 0]                              # D symmetric: S = D E
        psi = 0.5 * np.einsum("nv,nv->n", Ev, S) if request.energy else None
        return MaterialResponse(S, C, psi)


class NeoHooke:
    """Compressible neo-Hooke ψ = ½κ(J − 1)² + ½μ(Ī₁ − 3) (:func:`neo_hooke_response`), unit modulus:
    κ = 1/(3(1 − 2ν)), μ = 1/(2(1 + ν)). Plane strain: C₃₃ = 1. Plane stress: C₃₃ is solved at every point from
    S₃₃ = 0 (scalar Newton) and the tangent is condensed, ℂ_ps = ℂ_αβγδ − ℂ_αβ33 ℂ_33γδ / ℂ_3333 (only σ_zz:
    correct for this isotropic law, where the out-of-plane shear couplings vanish; not for an anisotropic one)."""

    name = "neo_hooke"
    kinematics = "finite"
    conservative = True
    state_fields: tuple[str, ...] = ()

    def __init__(self, nu: float) -> None:
        m = isotropic_moduli(E=1.0, nu=nu)                              # unit-modulus K and G
        self.nu = float(nu)
        self.kappa, self.mu = m.K, m.G

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {}

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None = None,
                 committed: Mapping[str, np.ndarray] | None = None,
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        _check_kin(self, kin)
        H = kin.grad
        energy = request.energy
        Eg = T.green_lagrange(H)
        a = T.jacobian_minus_one(H)
        if kin.dim == 3:
            S3, C6, psi = neo_hooke_response(Eg, a, self.kappa, self.mu, energy)
            return MaterialResponse(S3[:, _VA, _VB], C6, psi)
        E3 = np.zeros((len(H), 3, 3))
        E3[:, :2, :2] = Eg
        Jm1 = self._plane_stress(E3, a) if kin.plane == "stress" else a
        S3, V, psi = neo_hooke_response(E3, Jm1, self.kappa, self.mu, energy)
        ip = [0, 1, 5]
        Cv = V[:, ip][:, :, ip]
        if kin.plane == "stress":
            Cv = Cv - V[:, ip, 2][:, :, None] * V[:, 2, ip][:, None, :] / V[:, 2, 2][:, None, None]
        diag = {"E33": E3[:, 2, 2].copy()} if kin.plane == "stress" else {}
        return MaterialResponse(np.stack([S3[:, 0, 0], S3[:, 1, 1], S3[:, 0, 1]], axis=1), Cv, psi, diagnostics=diag)

    def _plane_stress(self, E3: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Solve S₃₃ = 0 for E₃₃ at every point (scalar Newton, dS₃₃/dE₃₃ = ℂ₃₃₃₃); sets ``E3[:, 2, 2]`` and returns
        J − 1 = (det F₂ − 1) s + (s − 1), s = √(1 + 2E₃₃)."""
        e = np.zeros(len(E3))

        def jm1(e: np.ndarray) -> np.ndarray:
            s = np.sqrt(1.0 + 2.0 * e)
            return a * s + 2.0 * e / (s + 1.0)

        k, m = self.kappa, self.mu
        tr2 = E3[:, 0, 0] + E3[:, 1, 1]
        scale = (k + m) * np.abs(E3[:, :2, :2]).max(axis=(1, 2))
        for _ in range(60):
            # S₃₃ and ℂ₃₃₃₃ of neo_hooke_response in closed form (C⁻¹₃₃ = 1/C₃₃, no out-of-plane shear)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                j1 = jm1(e)
                J = 1.0 + j1
                ci = 1.0 / (1.0 + 2.0 * e)
                trE = tr2 + e
                I1 = 3.0 + 2.0 * trE
                Jm = J ** (-2.0 / 3.0)
                r = k * J * j1 * ci + 2.0 * m * Jm * (e - trE / 3.0) * ci
                T33 = k * ((2.0 * J - 1.0) * J - 2.0 * J * j1) * ci ** 2 \
                    + m * Jm * (-4.0 / 3.0 * ci + (2.0 / 9.0 + 2.0 / 3.0) * I1 * ci ** 2)
                step = r / T33
            if not (np.all(J > 0) and np.all(np.isfinite(step))):      # a trial state of extreme distortion
                raise InadmissibleState("plane stress: no admissible S33 = 0 solution")
            if np.all((np.abs(r) <= 1e-14 * scale) | (np.abs(step) <= 1e-16 * (1 + np.abs(e)))):
                E3[:, 2, 2] = e
                return j1
            e = np.maximum(e - step, 0.5 * (e - 0.5))                  # keeps 1 + 2e > 0 (halfway to e = −½)
        raise InadmissibleState("plane stress: S33 = 0 not reached")


# ----------------------------------------------------------------------------------------------- small strain
def _small_strain_voigt(H: np.ndarray) -> np.ndarray:
    """Engineering-shear Voigt strain of ε = sym H, (n, nv)."""
    return T.strain_to_voigt(0.5 * (H + np.swapaxes(H, 1, 2)), rtol=None)


class LinearElastic:
    """Small-strain linear elasticity σ = D ε (unit D (nv, nv) or (m, nv, nv) with ``params["index"]``), ψ = ½εᵀDε.
    The D carries the plane condition (as :class:`SaintVenantKirchhoff`)."""

    name = "linear"
    kinematics = "small"
    conservative = True
    state_fields: tuple[str, ...] = ()

    def __init__(self, D: np.ndarray) -> None:
        self._svk = SaintVenantKirchhoff(D)
        self.D = self._svk.D

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {}

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None = None,
                 committed: Mapping[str, np.ndarray] | None = None,
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        _check_kin(self, kin)
        ev = _small_strain_voigt(kin.grad)
        if self.D.ndim == 2:
            S = ev @ self.D
            C = np.broadcast_to(self.D, (kin.n,) + self.D.shape)
        else:
            which = None if params is None else params.get("index")
            assert which is not None, "per-point D needs params['index']"
            C = self.D[which]
            S = (C @ ev[:, :, None])[:, :, 0]
        psi = 0.5 * np.einsum("nv,nv->n", ev, S) if request.energy else None
        return MaterialResponse(S, C, psi)


_I6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])            # symmetric identity, Voigt (engineering-shear strain)
_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_IDEV6 = _I6 - np.outer(_ONE6, _ONE6) / 3.0
_IP2 = [0, 1, 5]                                         # plane strain: [xx, yy, xy] of the 3-D Voigt


@cite("simo_hughes_1998", note="radial return and consistent tangent of J2 with isotropic hardening, Box 3.1-3.2 "
                               "(as derived here; the book was not read in this session)")
class J2Plasticity:
    """Small-strain associative von Mises (J2) plasticity with isotropic hardening, unit E (module notes).

    Yield function f = q − κ(α), q = √(3/2) ‖s‖, s = dev σ; hardening κ(α) = σ_y + Hα + (σ_∞ − σ_y)(1 − e^(−δα))
    (``saturation_stress`` σ_∞ None: linear). Elastic law σ = K tr(ε − ε_p) 1 + 2μ (ε − ε_p)_dev, K = 1/(3(1 − 2ν)),
    μ = 1/(2(1 + ν)). Backward-Euler radial return: trial s* = 2μ(dev ε − ε_p), q* = √(3/2)‖s*‖; if
    f* = q* − κ(α_n) > 0, Δγ solves g(Δγ) = q* − 3μΔγ − κ(α_n + Δγ) = 0 (closed form for linear hardening, scalar
    Newton otherwise, tolerance ``local_tol`` · σ_y, at most ``local_max_iter`` iterations), n̂ = s*/‖s*‖,
    s = (1 − 3μΔγ/q*) s*, ε_p ← ε_p + √(3/2) Δγ n̂, α ← α + Δγ. Algorithmic tangent

        C = K 1⊗1 + 2μ θ I_dev − 2μ θ̄ n̂⊗n̂,   θ = 1 − 3μΔγ/q*,   θ̄ = 3μ/(3μ + κ'(α_{n+1})) − (1 − θ);

    ``tangent_kind="continuum"`` returns θ = 1 (the rate tangent; not consistent with the update). State
    ``plastic_strain`` (n, 3, 3) and ``alpha`` (n,). Energy: stored ψ = ½ ε_e:C:ε_e + ψ_h(α), ψ_h = ½Hα² +
    (σ_∞ − σ_y)(α − (1 − e^(−δα))/δ); dissipation increment σ_y Δα (≥ 0). Diagnostics: ``yield_residual`` f at the
    returned state, ``plastic`` (1/0), ``local_iterations``, ``plastic_work`` σ_{n+1}:Δε_p, ``sigma_zz`` (plane
    strain)."""

    name = "j2"
    kinematics = "small"
    conservative = False
    state_fields = ("plastic_strain", "alpha")

    def __init__(self, nu: float, yield_stress: float, hardening: float = 0.0, saturation_stress: float | None = None,
                 saturation_rate: float = 0.0, local_tol: float = 1e-13, local_max_iter: int = 50) -> None:
        m = isotropic_moduli(E=1.0, nu=nu)
        self.nu, self.K, self.mu = float(nu), m.K, m.G
        self.sy, self.H = float(yield_stress), float(hardening)
        self.s_inf = None if saturation_stress is None else float(saturation_stress)
        self.delta = float(saturation_rate)
        if not (self.sy > 0 and self.H >= 0):
            raise ValueError("J2: yield_stress > 0 and hardening >= 0 are required")
        if self.s_inf is not None and not (self.s_inf >= self.sy and self.delta > 0):
            raise ValueError("J2: saturation_stress >= yield_stress and saturation_rate > 0 are required")
        if self.H == 0 and self.s_inf is None:
            pass                                                         # perfect plasticity: admissible here
        self.local_tol, self.local_max_iter = float(local_tol), int(local_max_iter)
        self.Ce6 = self.K * np.outer(_ONE6, _ONE6) + 2.0 * self.mu * _IDEV6

    # ---------------------------------------------------------------- hardening
    def kappa(self, a: np.ndarray) -> np.ndarray:
        k = self.sy + self.H * a
        if self.s_inf is not None:
            k = k + (self.s_inf - self.sy) * (-np.expm1(-self.delta * a))
        return k

    def dkappa(self, a: np.ndarray) -> np.ndarray:
        d = np.full_like(np.asarray(a, float), self.H)
        if self.s_inf is not None:
            d = d + (self.s_inf - self.sy) * self.delta * np.exp(-self.delta * a)
        return d

    def hardening_energy(self, a: np.ndarray) -> np.ndarray:
        e = 0.5 * self.H * a ** 2
        if self.s_inf is not None:
            e = e + (self.s_inf - self.sy) * (a + np.expm1(-self.delta * a) / self.delta)
        return e

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"plastic_strain": np.zeros((n, 3, 3)), "alpha": np.zeros(n)}

    # ---------------------------------------------------------------- update
    def _strain3(self, kin: Kinematics) -> np.ndarray:
        if kin.dim == 2 and kin.plane != "strain":
            raise ValueError("J2 plasticity in 2-D: plane strain only (plane stress needs a local condensation that "
                             "is not implemented)")
        eps = 0.5 * (kin.grad + np.swapaxes(kin.grad, 1, 2))
        if kin.dim == 3:
            return eps
        e3 = np.zeros((kin.n, 3, 3))
        e3[:, :2, :2] = eps
        return e3

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray], request: MaterialRequest = _DEFAULT_REQUEST
               ) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray],
                                                     dict[str, np.ndarray], np.ndarray | None]:
        """Return mapping at 3-D strains ``eps`` (n, 3, 3): σ (n, 3, 3), tangent (n, 6, 6) or None, trial state,
        diagnostics, stored energy or None."""
        ep_n = np.asarray(committed["plastic_strain"])
        a_n = np.asarray(committed["alpha"])
        n = len(eps)
        eye = np.eye(3)
        tr = np.trace(eps, axis1=1, axis2=2)
        dev = eps - (tr / 3.0)[:, None, None] * eye
        s_tr = 2.0 * self.mu * (dev - ep_n)
        nrm = np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))
        q_tr = math.sqrt(1.5) * nrm
        k_n = self.kappa(a_n)
        f_tr = q_tr - k_n
        plastic = f_tr > 0.0
        dg = np.zeros(n)
        its = np.zeros(n)
        if plastic.any():
            qp, ap = q_tr[plastic], a_n[plastic]
            if self.s_inf is None:
                dg[plastic] = f_tr[plastic] / (3.0 * self.mu + self.H)
                its[plastic] = 1
            else:
                x = f_tr[plastic] / (3.0 * self.mu + self.dkappa(ap))
                ok = np.zeros(len(x), bool)
                for _k in range(self.local_max_iter):
                    g = qp - 3.0 * self.mu * x - self.kappa(ap + x)
                    ok = np.abs(g) <= self.local_tol * self.sy
                    its[np.flatnonzero(plastic)[~ok]] += 1
                    if ok.all():
                        break
                    x = x + g / (3.0 * self.mu + self.dkappa(ap + x))
                else:
                    g = qp - 3.0 * self.mu * x - self.kappa(ap + x)
                    ok = np.abs(g) <= self.local_tol * self.sy
                if not (ok.all() and np.all(np.isfinite(x)) and np.all(x >= 0)):
                    raise ReturnMappingFailure(f"J2 return mapping: {int((~ok).sum())} points not converged in "
                                               f"{self.local_max_iter} local iterations")
                dg[plastic] = x
        nhat = np.zeros_like(s_tr)
        with np.errstate(invalid="ignore", divide="ignore"):
            nhat[plastic] = s_tr[plastic] / nrm[plastic][:, None, None]
        theta = np.ones(n)
        theta[plastic] = 1.0 - 3.0 * self.mu * dg[plastic] / q_tr[plastic]
        s = theta[:, None, None] * s_tr
        dep = math.sqrt(1.5) * dg[:, None, None] * nhat
        ep = ep_n + dep
        a = a_n + dg
        p = self.K * tr
        sig = s + p[:, None, None] * eye
        q = theta * q_tr
        yres = q - self.kappa(a)
        C = None
        if request.tangent:
            C = np.broadcast_to(self.Ce6, (n, 6, 6)).copy()
            if plastic.any():
                nv = nhat[plastic][:, _VA, _VB]                            # tensor components (not doubled)
                dk = self.dkappa(a[plastic])
                th = theta[plastic] if request.tangent_kind == "algorithmic" else np.ones(int(plastic.sum()))
                if request.tangent_kind not in ("algorithmic", "continuum"):
                    raise ValueError("tangent_kind: algorithmic | continuum")
                thb = 3.0 * self.mu / (3.0 * self.mu + dk) - (1.0 - th)
                C[plastic] = (self.K * np.outer(_ONE6, _ONE6)[None] + 2.0 * self.mu * th[:, None, None] * _IDEV6[None]
                              - 2.0 * self.mu * thb[:, None, None] * nv[:, :, None] * nv[:, None, :])
        psi = None
        if request.energy:
            ee = eps - ep
            psi = 0.5 * np.einsum("nij,nij->n", sig, ee) + self.hardening_energy(a)
        diag = {"yield_residual": np.where(plastic, yres, np.minimum(f_tr, 0.0)), "plastic": plastic.astype(float),
                "local_iterations": its, "plastic_work": np.einsum("nij,nij->n", sig, dep), "delta_alpha": dg}
        return sig, C, {"plastic_strain": ep, "alpha": a}, diag, psi

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None,
                 committed: Mapping[str, np.ndarray], request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        _check_kin(self, kin)
        eps = self._strain3(kin)
        sig, C6, trial, diag, psi = self.update(eps, committed, request)
        if kin.dim == 2:
            diag["sigma_zz"] = sig[:, 2, 2].copy()
            S = np.stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 0, 1]], axis=1)
            C = None if C6 is None else C6[:, _IP2][:, :, _IP2]
        else:
            S, C = sig[:, _VA, _VB], C6
        return MaterialResponse(S, C, psi, trial, self.sy * diag["delta_alpha"], diag)

    # ---------------------------------------------------------------- vector-Jacobian product of the update
    def update_vjp(self, eps: np.ndarray, committed: Mapping[str, np.ndarray], sig_bar: np.ndarray | None = None,
                   ep_bar: np.ndarray | None = None, alpha_bar: np.ndarray | None = None,
                   dalpha_bar: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cotangents (ε̄, ε̄_p^n, ᾱ_n) of the return mapping (ε, ε_p^n, α_n) → (σ, ε_p, α, Δα) of :meth:`update`
        for the output cotangents σ̄ (n, 3, 3), ε̄_p (n, 3, 3), ᾱ (n,), Δᾱ (n,) (None = 0). Tensors are contracted
        with ``:``; the plastic set is the one of :meth:`update` at these inputs (one-sided derivative; a point with
        f* = 0 exactly is taken as elastic). Derived here (``docs/olcumler/plastik_duyarlilik_onkayit.md``): with
        s* = 2μ(dev ε − ε_p^n), r = ‖s*‖, n̂ = s*/r, s = s* − √6 μ Δγ n̂, ε_p = ε_p^n + √(3/2) Δγ n̂, α = α_n + Δγ and
        dΔγ = (√(3/2) n̂:ds* − κ' dα_n)/(3μ + κ'), κ' = κ'(α_n + Δγ), dn̂ = (ds* − n̂ (n̂:ds*))/r:

            c = −√6 μ σ̄:n̂ + √(3/2) ε̄_p:n̂ + ᾱ + Δᾱ ,   T̄ = Δγ (−√6 μ σ̄ + √(3/2) ε̄_p) ,
            s̄* = σ̄ + (T̄ − (n̂:T̄) n̂)/r + c √(3/2) n̂/(3μ + κ')       (elastic points: s̄* = σ̄) ,
            ε̄ = K tr(σ̄) 1 + 2μ dev s̄* ,   ε̄_p^n = ε̄_p − 2μ s̄* ,   ᾱ_n = ᾱ − c κ'/(3μ + κ')  (elastic: ᾱ) ."""
        n = len(eps)
        z3 = np.zeros((n, 3, 3))
        sb = z3 if sig_bar is None else np.asarray(sig_bar, float)
        eb = z3 if ep_bar is None else np.asarray(ep_bar, float)
        ab = np.zeros(n) if alpha_bar is None else np.asarray(alpha_bar, float)
        db = np.zeros(n) if dalpha_bar is None else np.asarray(dalpha_bar, float)
        _sig, _C, trial, diag, _psi = self.update(eps, committed, MaterialRequest(tangent=False))
        plastic = diag["plastic"] > 0.5
        eye = np.eye(3)
        s_bar = sb.copy()
        a_n_bar = ab.copy()
        if plastic.any():
            ep_n = np.asarray(committed["plastic_strain"])[plastic]
            e = eps[plastic]
            dev = e - (np.trace(e, axis1=1, axis2=2) / 3.0)[:, None, None] * eye
            s_tr = 2.0 * self.mu * (dev - ep_n)
            r = np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))
            nh = s_tr / r[:, None, None]
            dg = diag["delta_alpha"][plastic]
            dk = self.dkappa(trial["alpha"][plastic])
            den = 3.0 * self.mu + dk
            sp_, ep_ = sb[plastic], eb[plastic]
            r6, r15 = math.sqrt(6.0), math.sqrt(1.5)
            c = (-r6 * self.mu * np.einsum("nij,nij->n", sp_, nh) + r15 * np.einsum("nij,nij->n", ep_, nh)
                 + ab[plastic] + db[plastic])
            Tb = dg[:, None, None] * (-r6 * self.mu * sp_ + r15 * ep_)
            PT = Tb - np.einsum("nij,nij->n", nh, Tb)[:, None, None] * nh
            s_bar[plastic] = sp_ + PT / r[:, None, None] + (c * r15 / den)[:, None, None] * nh
            a_n_bar[plastic] = ab[plastic] - c * dk / den
        tr_s = np.trace(sb, axis1=1, axis2=2)
        dev_sbar = s_bar - (np.trace(s_bar, axis1=1, axis2=2) / 3.0)[:, None, None] * eye
        eps_bar = (self.K * tr_s)[:, None, None] * eye + 2.0 * self.mu * dev_sbar
        ep_n_bar = eb - 2.0 * self.mu * s_bar
        return eps_bar, ep_n_bar, a_n_bar

    def evaluate_vjp(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                     stress_bar: np.ndarray | None = None, state_bar: Mapping[str, np.ndarray] | None = None,
                     dissipation_bar: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Vector-Jacobian product of :meth:`evaluate` in its own conventions: cotangents of the Voigt ``stress``
        (n, nv), of the trial state (``plastic_strain`` (n, 3, 3), ``alpha`` (n,)) and of the ``dissipation`` σ_y Δα
        (n,) → cotangents of the displacement gradient ``grad`` (n, dim, dim) and of the committed state (same
        fields). Used by the path adjoint of :class:`toporlogy.fem.nonlinear.NonlinearElasticity`."""
        _check_kin(self, kin)
        eps = self._strain3(kin)
        n, dim = kin.n, kin.dim
        sig_bar = None
        if stress_bar is not None:
            Sb = np.asarray(stress_bar, float)
            sig_bar = np.zeros((n, 3, 3))
            pairs = T.VOIGT_3D if dim == 3 else ((0, 0), (1, 1), (0, 1))
            for v, (i, j) in enumerate(pairs):
                if i == j:
                    sig_bar[:, i, i] = Sb[:, v]
                else:                                  # σ symmetric: S̄_v dσ_ij = ½S̄_v (dσ_ij + dσ_ji)
                    sig_bar[:, i, j] = sig_bar[:, j, i] = 0.5 * Sb[:, v]
        sb = {} if state_bar is None else state_bar
        db = None if dissipation_bar is None else self.sy * np.asarray(dissipation_bar, float)
        eps_bar, ep_n_bar, a_n_bar = self.update_vjp(eps, committed, sig_bar, sb.get("plastic_strain"),
                                                     sb.get("alpha"), db)
        H_bar = 0.5 * (eps_bar + np.swapaxes(eps_bar, 1, 2))[:, :dim, :dim]   # ε = sym H (embedded in 3-D)
        return H_bar, {"plastic_strain": ep_n_bar, "alpha": a_n_bar}


# ----------------------------------------------------------------------------------------------- AD oracle
def _eye_voigt_strain(Ev: Any) -> Any:
    import torch
    E = torch.zeros((3, 3), dtype=Ev.dtype)
    for v, (i, j) in enumerate(T.VOIGT_3D):
        f = 1.0 if i == j else 0.5
        E = E.index_put((torch.tensor([i]), torch.tensor([j])), Ev[v:v + 1] * f)
        if i != j:
            E = E.index_put((torch.tensor([j]), torch.tensor([i])), Ev[v:v + 1] * f)
    return E


def svk_energy_torch(D6: np.ndarray) -> Callable[[Any], Any]:
    """ψ(E_voigt) = ½ Eᵀ D E as a torch function of the 3-D engineering-shear Voigt Green strain (6,)."""
    import torch
    Dt = torch.as_tensor(np.asarray(D6, float))

    def psi(Ev: Any) -> Any:
        return 0.5 * Ev @ (Dt @ Ev)
    return psi


def neo_hooke_energy_torch(kappa: float, mu: float) -> Callable[[Any], Any]:
    """ψ = ½κ(J − 1)² + ½μ(J^(−2/3) tr C − 3), C = I + 2E, J = √det C, as a torch function of the 3-D
    engineering-shear Voigt Green strain (6,) (written from the energy, independently of the analytic stress)."""
    import torch

    def psi(Ev: Any) -> Any:
        C = torch.eye(3, dtype=Ev.dtype) + 2.0 * _eye_voigt_strain(Ev)
        J = torch.sqrt(torch.linalg.det(C))
        return 0.5 * kappa * (J - 1.0) ** 2 + 0.5 * mu * (J ** (-2.0 / 3.0) * torch.trace(C) - 3.0)
    return psi


def energy_oracle(psi: Callable[[Any], Any], Ev: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(ψ, ∂ψ/∂E_v, ∂²ψ/∂E_v²) of a torch energy at one engineering-shear Voigt strain ``Ev`` (6,), float64
    autograd. With engineering shear, ∂ψ/∂E_v is the Voigt stress (physical shear) and the Hessian the Voigt
    tangent."""
    import torch
    x = torch.tensor(np.asarray(Ev, float), dtype=torch.float64, requires_grad=True)
    val = psi(x)
    (g,) = torch.autograd.grad(val, x, create_graph=True)
    Hs = torch.stack([torch.autograd.grad(g[i], x, retain_graph=True)[0] for i in range(len(g))])
    return float(val.detach()), g.detach().numpy(), Hs.detach().numpy()


# ----------------------------------------------------------------------------------------------- material point
@dataclass
class MaterialPointHistory:
    """Converged path of :func:`drive_material_point`: strains and stresses (k, 6) (3-D Voigt), states per step,
    energies, dissipation increments, local Newton iterations per step."""

    strain: np.ndarray
    stress: np.ndarray
    states: list[dict[str, np.ndarray]]
    energy: np.ndarray
    dissipation: np.ndarray
    plastic_work: np.ndarray
    yield_residual: np.ndarray
    iterations: list[int]


def drive_material_point(model: Any, strain: Sequence[Sequence[float]], stress: Sequence[Sequence[float]],
                         control: Sequence[bool], tol: float = 1e-14, max_iter: int = 25) -> MaterialPointHistory:
    """Integrate a small-strain 3-D material point along a mixed path. ``control`` (6,) marks the Voigt components
    whose STRAIN is prescribed (True) — the others have their STRESS prescribed. ``strain`` / ``stress`` (k, 6):
    targets per step (only the controlled entries of each are read). The unknown strains are found per step by Newton
    with the model's tangent (the elastic one ``Ce6`` in the first iteration when the model has it;
    ‖σ_free − target‖∞ ≤ ``tol``); the state is committed after each converged step."""
    ctrl = np.asarray(control, bool)
    free = ~ctrl
    eps_t = np.asarray(strain, float)
    sig_t = np.asarray(stress, float)
    store = StateStore(model, 1)
    e = np.zeros(6)
    out_e, out_s, states, en, dis, wp, yr, its = [], [], [], [], [], [], [], []
    for k in range(len(eps_t)):
        e = e.copy()
        e[ctrl] = eps_t[k, ctrl]
        for it in range(max_iter + 1):
            H = T.voigt_to_strain(e)[None]
            r = model.evaluate(Kinematics(H, "small"), None, store.committed, MaterialRequest(energy=True))
            res = r.stress[0, free] - sig_t[k, free]
            if not free.any() or np.abs(res).max() <= tol:
                break
            if it == max_iter:
                raise ReturnMappingFailure(f"material point step {k}: mixed control not converged ({res})")
            assert r.tangent is not None
            # first iteration of a step: the elastic tangent when the model has one (the start point lies on the
            # yield surface after a plastic step; its plastic tangent sends an unloading step far into reverse yield)
            Ce = getattr(model, "Ce6", None)
            Cff = (Ce if (it == 0 and Ce is not None) else r.tangent[0])[np.ix_(free, free)]
            e[free] -= np.linalg.solve(Cff, res)
        store.set_trial(r.trial_state)
        store.commit()
        out_e.append(e.copy())
        out_s.append(r.stress[0].copy())
        states.append(store.checkpoint())
        en.append(float(r.energy[0]) if r.energy is not None else np.nan)
        dis.append(float(r.dissipation[0]) if r.dissipation is not None else 0.0)
        wp.append(float(r.diagnostics.get("plastic_work", np.zeros(1))[0]))
        yr.append(float(r.diagnostics.get("yield_residual", np.zeros(1))[0]))
        its.append(it)
    return MaterialPointHistory(np.array(out_e), np.array(out_s), states, np.array(en), np.array(dis), np.array(wp),
                                np.array(yr), its)

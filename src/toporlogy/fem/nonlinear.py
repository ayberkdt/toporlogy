"""Geometrically non-linear static elasticity (large displacements) on the regular grid, Total Lagrangian.

Kinematics and balance (reference configuration X, displacement u, element-wise):

    F = I + ∇_X u ,   E = ½ (Fᵀ F − I) ,   S = ∂ψ/∂E (second Piola–Kirchhoff),
    f_int,e = ∫ Bᵀ S dV ,   K_T,e = ∫ (Bᵀ ℂ B + Gᵀ S G) dV ,

with B the non-linear strain matrix (δE = B δu, built from F) and the geometric term Gᵀ S G = (∇N S ∇Nᵀ) ⊗ I.
Equilibrium R(u) = f_int(u) − λ F = 0 for the dead load F (does not follow the deformation), solved by Newton–Raphson
with the consistent tangent K_T. Elements: Q4 (2-D, plane stress or plane strain, thickness t) and H8 (3-D), full
2 × 2(× 2) Gauss rule, the same grid, node and DOF order as :class:`toporlogy.fem.elastic.Elasticity`; element moduli
E_e scale the element's internal force and tangent (the same ``E_active`` that ``Elasticity`` takes).

Materials (unit modulus, multiplied by E_e):

* ``"svk"``: St. Venant–Kirchhoff, S = D E in Voigt form with the same unit-modulus D as ``Elasticity`` (isotropic
  plane stress / plane strain / 3-D from ``nu``, or any D given as ``elasticity``). In 2-D plane stress this is the
  2-D model with the plane-stress constants (S₃₃ = 0 is imposed on the linear law, not on a 3-D SVK body).
* ``"neo_hooke"``: compressible neo-Hooke ψ = ½κ(J − 1)² + ½μ(Ī₁ − 3), Ī₁ = tr(J^(−2/3) C), κ = E/(3(1 − 2ν)),
  μ = E/(2(1 + ν)) (the form of Feng, Zhang & Khandelwal 2025, Eqs. 9–10). Plane strain: C₃₃ = 1. Plane stress:
  C₃₃ is solved at every Gauss point from S₃₃ = 0 (scalar Newton) and the tangent is condensed,
  ℂ_ps = ℂ_αβγδ − ℂ_αβ33 ℂ_33γδ / ℂ_3333.

Both reduce to the linear elasticity of ``Elasticity`` for small displacements (same D).

Low-density elements (``energy_interpolation``). Void elements of a density design have a tiny modulus and are
dragged along by the solid; under large displacements they invert and Newton fails. The remedy is the energy
interpolation of Wang, Lazarov, Sigmund & Jensen (2014), as written by Feng, Zhang & Khandelwal (2025, Eqs. 11–14):
each element's energy is ψ_NL(γu) − ψ_L(γu) + ψ_L(u), i.e.

    f_int,e = E_e [γ f_NL(γ u_e) + (1 − γ²) K0 u_e] ,   K_T,e = E_e [γ² K_NL(γ u_e) + (1 − γ²) K0] ,

with K0 the small-displacement element matrix (the same as ``Elasticity``'s) and γ = exp(β ρ̃)/(exp(β c) + exp(β ρ̃))
a smoothed step: solid elements (γ → 1) are non-linear, void ones (γ → 0) linear. The physics sees the moduli only,
not the density, so ρ̃ = (E_e / E0)^(1/p) (``density_exponent`` p, default 3: the SIMP density when E_min = 0 and
frac = 1). The cutoff c starts at ``cutoff`` (0.1) and, when Newton fails at the smallest load step, is raised by
``cutoff_step`` (0.02) up to ``cutoff_max`` (1.0) and the case is solved again from zero load (the adaptive rule the
same source describes); cutoff values that move no element's γ by more than 1e-3 are skipped (they would repeat the
same failed solve). Every new solve starts again at ``cutoff``. The cutoff used is in :attr:`last_reports`.
Its dependence on the design is part of the residual's design derivative (∂γ/∂E_e), so the adjoint is consistent
for the cutoff of the solve. When the cutoff is raised during a design evaluation, the response is a different
(more linear) model: the objective is then discontinuous in the design. [Wang et al. 2014 not read; the scheme
as read in the secondary source.]

Newton–Raphson (per load case; cases are solved independently — no superposition):

* load factor λ from 0 to 1 in ``load_steps`` equal increments; a failed increment is halved (at most down to
  ``min_step``), after two successful increments in a row it grows by 1.5 (at most to 1/load_steps);
* convergence of an increment: ‖R_free‖₂ ≤ ``rtol`` · ‖λ F_free‖₂, or backward error
  η = ‖R_free‖ / (‖K_T‖₁ ‖u_free‖ + ‖λ F_free‖) ≤ :data:`ETA_TOL` — the definition of ``checked_residual``. The
  residual is formed from element forces K_e-sized times u_e, so its rounding floor is ~eps ‖K‖ ‖u‖, which exceeds
  rtol ‖λF‖ when ‖K‖ ‖u‖ ≫ ‖F‖ (slender members, high stiffness contrast; measured 25 Nov 2026 on a cantilever
  L/H = 50: relative residual floor 2.7e-9 at λ = 2e-4);
* an increment fails on ``max_iter`` iterations, a non-finite residual, a residual growing above ``1e6`` × the
  first one, an inverted element (det F ≤ 0 with neo-Hooke), or a linear solve that fails its residual check;
* when the smallest increment fails (and the cutoff cannot be raised), :class:`NonlinearConvergenceError` is raised
  with the load factor reached and the residual history. There is no silent partial result.

:meth:`NonlinearElasticity.solve` is differentiable (``torch.autograd.Function``): the backward pass assembles the
tangent at the converged state and solves ONE adjoint system per load case, K_T λ = ∂J/∂U (K_T is symmetric:
hyperelastic material, dead load), and returns ∂J/∂E_e = −λ_eᵀ ∂f_int,e/∂E_e. No self-adjoint shortcut: for the end
compliance Fᵀu the adjoint λ is not u.

Not provided (load control only): limit points, snap-through and post-buckling (no arc-length), follower loads,
prescribed non-zero displacements, plasticity, contact, incompatible-mode elements (Q4IM, H8IM), anisotropic
neo-Hooke, iterative / multigrid solvers.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from toporlogy.core import tensor as T
from toporlogy.core.domain import Domain
from toporlogy.core.linear_elastic import isotropic_moduli
from toporlogy.fem.elastic import checked_residual, matrix_norm1
from toporlogy.core.material import unique_matrices
from toporlogy.fem.elements import H8_CORNER, Q4_CORNER, D_iso, D_plane_strain, D_plane_stress, indexed_matmul
from toporlogy.fem.mechanisms import SolveResidualError, check_moduli, check_supports, fixed_dof_rows
from toporlogy.references import cite
from toporlogy.solvers.direct import DirectSolver

__all__ = ["NonlinearElasticity", "NonlinearConvergenceError", "EnergyInterpolation", "CaseReport", "StepRecord",
           "MATERIALS", "ETA_TOL"]

MATERIALS = ("svk", "neo_hooke")
ETA_TOL = 1e-13            # backward error of the residual accepted as converged (rounding floor)
_G = 1.0 / math.sqrt(3.0)


class NonlinearConvergenceError(RuntimeError):
    """Newton–Raphson did not reach equilibrium at the full load (see the module notes for the rule)."""

    code = "nonlinear_convergence"


class _Inadmissible(ArithmeticError):
    """An element state outside the material's domain (inverted element, no plane-stress solution)."""


@cite("wang_lazarov_sigmund_jensen_2014", "feng_zhang_khandelwal_2025", "zhang_alberdi_khandelwal_2018",
      note="energy interpolation and its adaptive cutoff, as written in Feng et al. 2025, Eqs. (11)-(14) and the "
           "remark on p. 11; the 2014 and 2018 papers themselves were not read")
@dataclass(frozen=True)
class EnergyInterpolation:
    """Parameters of the energy interpolation (module notes): γ = exp(β ρ̃)/(exp(β c) + exp(β ρ̃)),
    ρ̃ = (E_e / E0)^(1/p). Defaults β = 120, c0 = 0.1, Δc = 0.02, c ≤ 1 as in Feng, Zhang & Khandelwal (2025)."""

    E0: float = 1.0
    beta: float = 120.0
    cutoff: float = 0.1
    cutoff_step: float = 0.02
    cutoff_max: float = 1.0
    density_exponent: float = 3.0

    def __post_init__(self) -> None:
        if not (self.E0 > 0 and self.beta > 0 and self.density_exponent > 0):
            raise ValueError("energy interpolation: E0, beta and density_exponent must be positive")
        if not (0.0 <= self.cutoff <= self.cutoff_max) or self.cutoff_step < 0:
            raise ValueError("energy interpolation: 0 <= cutoff <= cutoff_max and cutoff_step >= 0 are required")

    def gamma(self, E: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
        """(γ, dγ/dE) per element for moduli E (> 0) and the given cutoff."""
        p = self.density_exponent
        rho = (np.asarray(E, float) / self.E0) ** (1.0 / p)
        z = self.beta * (rho - cutoff)
        g = 0.5 * (1.0 + np.tanh(0.5 * z))                    # = exp(βρ)/(exp(βc)+exp(βρ)), overflow-free
        dg = self.beta * g * (1.0 - g) * rho / (p * np.asarray(E, float))
        return g, dg


@dataclass
class StepRecord:
    """One load increment: target load factor, accepted or not, why it failed, relative residual ‖R‖/‖λF‖ and
    backward error η per iteration (the first entries before any update)."""

    load_factor: float
    converged: bool
    residuals: list[float]
    reason: str = ""
    backward_errors: list[float] = field(default_factory=list)


@dataclass
class CaseReport:
    """Newton history of one load case."""

    steps: list[StepRecord] = field(default_factory=list)
    cutoff: float | None = None            # energy-interpolation cutoff of the returned solution
    relative_residual: float = 0.0         # ‖R‖/‖F‖ at the returned solution
    backward_error: float = 0.0            # η at the returned solution
    cuts: int = 0                          # failed increments (halvings)
    restarts: int = 0                      # restarts from zero load with a raised cutoff

    @property
    def iterations(self) -> int:
        return sum(len(s.residuals) - 1 for s in self.steps)


# ----------------------------------------------------------------------------------------------- materials
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

    with 𝕀_ijkl = ½(C⁻¹_ik C⁻¹_jl + C⁻¹_il C⁻¹_jk). Raises :class:`_Inadmissible` for J ≤ 0."""
    J = 1.0 + Jm1
    if not np.all(J > 0):
        raise _Inadmissible("det F <= 0")
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


class _Material:
    """Unit-modulus material law in the element's Voigt form: (S (n, nv), ℂ (n, nv, nv), ψ (n,) | None) from the
    displacement gradient H (n, dim, dim)."""

    def __init__(self, kind: str, dim: int, D: np.ndarray, nu: float | None, plane: str) -> None:
        # D: one unit D (nv, nv), or the distinct matrices (m, nv, nv) of a per-element field (``which`` picks)
        self.kind, self.dim, self.D, self.plane = kind, dim, D, plane
        if kind == "neo_hooke":
            assert nu is not None
            m = isotropic_moduli(E=1.0, nu=nu)                          # unit-modulus K and G
            self.kappa, self.mu = m.K, m.G

    def __call__(self, H: np.ndarray, energy: bool = False,
                 which: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        dim = self.dim
        Eg = T.green_lagrange(H)
        if self.kind == "svk":
            Ev = T.strain_to_voigt(Eg, rtol=None)                   # symmetric by construction
            if self.D.ndim == 2:
                S = Ev @ self.D
                C = np.broadcast_to(self.D, (len(H),) + self.D.shape)
            else:
                assert which is not None
                C = self.D[which]
                S = (C @ Ev[:, :, None])[:, :, 0]                      # D symmetric: S = D E
            psi = 0.5 * np.einsum("nv,nv->n", Ev, S) if energy else None
            return S, C, psi
        a = T.jacobian_minus_one(H)
        if dim == 3:
            S3, C6, psi = neo_hooke_response(Eg, a, self.kappa, self.mu, energy)
            return S3[:, _VA, _VB], C6, psi
        E3 = np.zeros((len(H), 3, 3))
        E3[:, :2, :2] = Eg
        Jm1 = self._plane_stress(E3, a) if self.plane == "stress" else a
        S3, V, psi = neo_hooke_response(E3, Jm1, self.kappa, self.mu, energy)
        ip = [0, 1, 5]
        Cv = V[:, ip][:, :, ip]
        if self.plane == "stress":
            Cv = Cv - V[:, ip, 2][:, :, None] * V[:, 2, ip][:, None, :] / V[:, 2, 2][:, None, None]
        return np.stack([S3[:, 0, 0], S3[:, 1, 1], S3[:, 0, 1]], axis=1), Cv, psi

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
                raise _Inadmissible("plane stress: no admissible S33 = 0 solution")
            if np.all((np.abs(r) <= 1e-14 * scale) | (np.abs(step) <= 1e-16 * (1 + np.abs(e)))):
                E3[:, 2, 2] = e
                return j1
            e = np.maximum(e - step, 0.5 * (e - 0.5))                  # keeps 1 + 2e > 0 (halfway to e = −½)
        raise _Inadmissible("plane stress: S33 = 0 not reached")


# ----------------------------------------------------------------------------------------------- element geometry
def _shape_gradients(dim: int, h: float) -> np.ndarray:
    """(ngp, dim, nnode) ∂N/∂X at the Gauss points of an element of edge h (corner order of elements.py)."""
    corner = Q4_CORNER if dim == 2 else H8_CORNER
    pts = [np.array(p, float) for p in np.array(np.meshgrid(*([[-_G, _G]] * dim), indexing="ij")).reshape(dim, -1).T]
    out = []
    for p in pts:
        a = 1.0 + corner * p                                        # (nnode, dim)
        g = np.empty((dim, len(corner)))
        for j in range(dim):
            others = np.prod(np.delete(a, j, axis=1), axis=1)
            g[j] = corner[:, j] * others / 2 ** dim
        out.append(g * (2.0 / h))
    return np.asarray(out)


# ----------------------------------------------------------------------------------------------- physics
class NonlinearElasticity:
    """Static geometrically non-linear elasticity (Total Lagrangian) on a :class:`Domain` with fixed DOFs.

    Parameters
    ----------
    domain : Domain
    fixed : sequence of int
        Box DOF numbers with zero displacement (homogeneous supports only).
    nu : float
        Poisson's ratio (isotropic D, and κ, μ of neo-Hooke).
    material : {"svk", "neo_hooke"}
    plane : {"stress", "strain"}
        2-D only.
    element : {"Q4", "H8"}, optional
        Default Q4 in 2-D, H8 in 3-D.
    elasticity : array, optional
        Unit-modulus D for ``"svk"`` (orthotropic, rotated, plane strain;
        :func:`toporlogy.core.material.unit_elasticity`), or one per active element (n_active, nv, nv;
        :func:`toporlogy.core.material.unit_elasticity_field`); ``nu`` and ``plane`` are then not used. Not
        accepted with ``"neo_hooke"``.
    energy_interpolation : EnergyInterpolation, optional
        Low-density remedy (module notes). None = every element fully non-linear.
    load_steps, max_iter, rtol, min_step : Newton–Raphson controls (module notes).
    warm_start : bool
        Start each case from the previous converged displacement at the full load (one increment), falling back to
        load stepping from zero when that fails. Off by default: the result then depends on the design only, not on
        the evaluation history.
    solver : object with ``factor(A_upper)`` / ``solve(B)``, optional
        Default :class:`DirectSolver` (symmetric; the tangent of a stable equilibrium is positive definite).
    """

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray, nu: float = 0.3, material: str = "svk",
                 plane: str = "stress", element: str | None = None, elasticity: np.ndarray | None = None,
                 energy_interpolation: EnergyInterpolation | None = None, load_steps: int = 10, max_iter: int = 30,
                 rtol: float = 1e-9, min_step: float = 1e-4, warm_start: bool = False, solver: Any = None) -> None:
        dim = domain.dim
        if material not in MATERIALS:
            raise ValueError(f"material: {' | '.join(MATERIALS)}, got {material!r}")
        if plane not in ("stress", "strain"):
            raise ValueError("plane: stress | strain")
        if dim == 3 and plane != "stress":
            raise ValueError("plane applies to 2-D domains only")
        self.element = element or ("H8" if dim == 3 else "Q4")
        if self.element != ("H8" if dim == 3 else "Q4"):
            raise ValueError(f"non-linear elasticity supports {'H8' if dim == 3 else 'Q4'} in {dim}-D, "
                             f"not {self.element!r}")
        if not (load_steps >= 1 and max_iter >= 1 and rtol > 0 and 0 < min_step <= 1.0 / load_steps):
            raise ValueError("load_steps >= 1, max_iter >= 1, rtol > 0, 0 < min_step <= 1/load_steps are required")
        self.domain, self.material, self.plane = domain, material, plane if dim == 2 else "stress"
        if elasticity is not None:
            if material != "svk":
                raise ValueError("elasticity (a general D) is accepted with material svk only")
            self.nu: float | None = None
            self.D = np.asarray(elasticity, float).copy()
            nv = 6 if dim == 3 else 3
            if self.D.shape != (nv, nv) and self.D.shape != (domain.n_active, nv, nv):
                raise ValueError(f"elasticity must be {nv} x {nv} or one per active element "
                                 f"({domain.n_active}, {nv}, {nv}), got {self.D.shape}")
        else:
            self.nu = float(nu)
            self.D = D_iso(nu) if dim == 3 else (D_plane_strain(nu) if plane == "strain" else D_plane_stress(nu))
        # per-element D: the distinct matrices and each element's index (as Elasticity.ke_index); None = one D
        self.d_index: np.ndarray | None = None
        if self.D.ndim == 3:
            Du, self.d_index = unique_matrices(self.D)
            self._law = _Material(material, dim, Du, self.nu, self.plane)
        else:
            self._law = _Material(material, dim, self.D, self.nu, self.plane)
        self.interpolation = energy_interpolation
        self.load_steps, self.max_iter, self.rtol, self.min_step = int(load_steps), int(max_iter), float(rtol), \
            float(min_step)
        self.warm_start = bool(warm_start)
        self.geom = domain.h if dim == 3 else domain.thickness       # as Elasticity (used by responses / tests)
        self._dN = _shape_gradients(dim, domain.h)                     # (ngp, dim, nnode)
        self._w = (domain.h / 2.0) ** dim * (1.0 if dim == 3 else domain.thickness)
        self.elements = domain.active_elements
        self.edof = domain.edof[self.elements]
        fixed_arr = np.unique(np.asarray(fixed, np.int64))
        if fixed_arr.size and (fixed_arr.min() < 0 or fixed_arr.max() >= domain.ndof):
            raise ValueError("fixed DOF out of range")
        self.fixed = fixed_arr
        self.free = np.setdiff1d(domain.active_dofs, fixed_arr)
        self.fixed.setflags(write=False)
        self.free.setflags(write=False)
        if self.free.size == 0:
            raise ValueError("no free DOF")
        nd, nf, nl = domain.ndof, len(self.free), self.edof.shape[1]
        gmap = np.full(nd, -1, np.int64)
        gmap[self.free] = np.arange(nf)
        r = gmap[np.repeat(self.edof, nl, axis=1).ravel()]
        c = gmap[np.tile(self.edof, (1, nl)).ravel()]
        keep = (r >= 0) & (c >= 0) & (r <= c)
        self._sel = np.flatnonzero(keep)
        key = r[keep] * nf + c[keep]
        ukey, inv = np.unique(key, return_inverse=True)
        self._inv = inv.astype(np.int64)
        self._P = sp.csr_matrix((np.zeros(len(ukey)), (ukey // nf, ukey % nf)), shape=(nf, nf))
        self.solver = solver if solver is not None else DirectSolver()
        m = 1 if self.d_index is None else len(self._law.D)
        K0 = self._unit_state(np.zeros((m, nl)), np.ones(m), need_K=True, which=np.arange(m))[1]
        assert K0 is not None
        # small-displacement element matrix; per-element D: the distinct matrices (m, nloc, nloc), K0[d_index[e]]
        self.K0: np.ndarray = K0[0] if self.d_index is None else K0
        self._kinematics_checked = False
        self.n_solves = 0
        self.n_adjoint_solves = 0
        self.last_reports: list[CaseReport] = []
        self._warm: np.ndarray | None = None

    # ------------------------------------------------------------------ sizes and checks
    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def dim(self) -> int:
        return self.domain.dim

    def check_kinematics(self) -> None:
        """Raise :class:`~toporlogy.fem.mechanisms.MechanismError` if the supports leave a zero-energy motion of the
        undeformed structure (checked once, before the first solve)."""
        check_supports(self.domain, self.elements, fixed_dof_rows(self.fixed, self.dim))
        self._kinematics_checked = True

    # ------------------------------------------------------------------ element level
    def _unit_state(self, Ue: np.ndarray, gam: np.ndarray, need_K: bool, energy: bool = False,
                    which: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Unit-modulus non-linear element force f_NL(γ u_e) (n, nloc), tangent K_NL(γ u_e) (n, nloc, nloc) and
        energy ψ_NL(γ u_e) (n,) of the elements' displacements ``Ue`` (n, nloc) scaled by ``gam`` (n,).
        ``which``: per-element D, index of each row's D (default: the active elements' :attr:`d_index`)."""
        if which is None:
            which = self.d_index
        dim = self.dim
        n, nl = Ue.shape
        nn = nl // dim
        U3 = (gam[:, None] * Ue).reshape(n, nn, dim)
        f = np.zeros((n, nl))
        K = np.zeros((n, nl, nl)) if need_K else None
        psi = np.zeros(n) if energy else None
        eye = np.eye(dim)
        pairs = T.voigt_pairs(dim)
        Ut = np.swapaxes(U3, 1, 2)                                     # (n, dim, nn)
        for dN in self._dN:                                            # (dim, nn)
            Hg = Ut @ dN.T
            F = eye + Hg
            S, C, ps = self._law(Hg, energy, which)
            B = np.zeros((n, len(pairs), nn, dim))
            for v, (j, k) in enumerate(pairs):
                B[:, v] = dN[k][None, :, None] * F[:, :, j][:, None, :]
                if j != k:
                    B[:, v] += dN[j][None, :, None] * F[:, :, k][:, None, :]
            B2 = B.reshape(n, len(pairs), nl)
            Bt = np.swapaxes(B2, 1, 2)
            f += self._w * (Bt @ S[:, :, None])[:, :, 0]
            if psi is not None and ps is not None:
                psi += self._w * ps
            if K is not None:
                K += self._w * (Bt @ (C @ B2))
                St = np.zeros((n, dim, dim))
                for v, (j, k) in enumerate(pairs):
                    St[:, j, k] = St[:, k, j] = S[:, v]
                Gm = self._w * (dN.T @ St @ dN)                        # (n, nn, nn)
                K4 = K.reshape(n, nn, dim, nn, dim)
                for i in range(dim):
                    K4[:, :, i, :, i] += Gm
        return f, K, psi

    def _k0_times(self, Ue: np.ndarray) -> np.ndarray:
        """(n_active, nloc) K0_e u_e (K0 symmetric)."""
        if self.d_index is None:
            return Ue @ self.K0
        return indexed_matmul(self.K0, self.d_index, Ue[:, :, None])[:, :, 0]

    def _gamma(self, E: np.ndarray, cutoff: float | None) -> tuple[np.ndarray, np.ndarray]:
        if self.interpolation is None:
            return np.ones(len(E)), np.zeros(len(E))
        return self.interpolation.gamma(E, self.interpolation.cutoff if cutoff is None else cutoff)

    def element_state(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None, need_K: bool = True,
                      need_dE: bool = False) -> dict[str, np.ndarray]:
        """Element internal forces ``f`` (na, nloc), tangents ``K`` (na, nloc, nloc) and, with ``need_dE``, the design
        derivative ``dfdE`` = ∂f_int,e/∂E_e (na, nloc) at the full displacement vector ``u`` (ndof,) and moduli
        ``E`` (na,) (energy interpolation with ``cutoff``, default its initial value)."""
        Ue = u[self.edof]
        gam, dgam = self._gamma(E, cutoff)
        interp = self.interpolation is not None
        fN, KN, _ = self._unit_state(Ue, gam, need_K or (need_dE and interp))
        out: dict[str, np.ndarray] = {}
        if interp:
            k0u = self._k0_times(Ue)
            g2 = gam ** 2
            f_unit = gam[:, None] * fN + (1.0 - g2)[:, None] * k0u
            if need_K:
                assert KN is not None
                K0 = self.K0 if self.d_index is None else self.K0[self.d_index]
                out["K"] = E[:, None, None] * (g2[:, None, None] * KN + (1.0 - g2)[:, None, None] * K0)
            if need_dE:
                assert KN is not None
                dfg = fN + gam[:, None] * np.einsum("epq,eq->ep", KN, Ue) - 2.0 * gam[:, None] * k0u
                out["dfdE"] = f_unit + (E * dgam)[:, None] * dfg
        else:
            f_unit = fN
            if need_K:
                assert KN is not None
                out["K"] = E[:, None, None] * KN
            if need_dE:
                out["dfdE"] = f_unit
        out["f"] = E[:, None] * f_unit
        return out

    def strain_energy(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None) -> float:
        """Total stored energy Σ_e E_e [ψ_NL(γu) − ψ_L(γu) + ψ_L(u)] (its u-gradient is f_int)."""
        Ue = u[self.edof]
        gam, _ = self._gamma(np.asarray(E, float), cutoff)
        _f, _K, psi = self._unit_state(Ue, gam, need_K=False, energy=True)
        assert psi is not None
        if self.d_index is None:
            lin = 0.5 * np.einsum("ep,pq,eq->e", Ue, self.K0, Ue)
        else:
            lin = 0.5 * np.einsum("ep,ep->e", Ue, self._k0_times(Ue))
        return float(np.sum(np.asarray(E, float) * (psi + (1.0 - gam ** 2) * lin)))

    # ------------------------------------------------------------------ assembly
    def _assemble_vec(self, fe: np.ndarray) -> np.ndarray:
        return np.bincount(self.edof.ravel(), weights=fe.ravel(), minlength=self.domain.ndof)

    def _assemble_mat(self, Ke: np.ndarray) -> sp.csr_matrix:
        A = self._P.copy()
        A.data = np.bincount(self._inv, weights=Ke.reshape(len(Ke), -1).ravel()[self._sel], minlength=self._P.nnz)
        return A

    def internal_force(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None) -> np.ndarray:
        """Assembled f_int(u) (ndof,), fixed DOFs included (their entries are reactions)."""
        return self._assemble_vec(self.element_state(u, np.asarray(E, float), cutoff, need_K=False)["f"])

    def tangent(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None) -> sp.csr_matrix:
        """Full symmetric tangent K_T(u) on the free DOFs (nf × nf, ordered as :attr:`free`)."""
        A = self._assemble_mat(self.element_state(u, np.asarray(E, float), cutoff)["K"])
        return (A + sp.triu(A, k=1).T).tocsr()

    # ------------------------------------------------------------------ Newton–Raphson
    def _newton(self, E: np.ndarray, fl: np.ndarray, u0: np.ndarray, cutoff: float | None,
                lam: float) -> tuple[bool, np.ndarray, StepRecord, float, float]:
        u = u0.copy()
        fr = self.free
        nfl = float(np.linalg.norm(fl[fr]))
        rec = StepRecord(lam, False, [])
        first = None
        for it in range(self.max_iter + 1):
            try:
                st = self.element_state(u, E, cutoff)
            except _Inadmissible as e:
                rec.reason = f"inadmissible state ({e})"
                return False, u0, rec, np.inf, np.inf
            R = self._assemble_vec(st["f"])[fr] - fl[fr]
            nr = float(np.linalg.norm(R))
            A = self._assemble_mat(st["K"])
            k1 = matrix_norm1(A)
            rel = nr / nfl
            eta = nr / (k1 * float(np.linalg.norm(u[fr])) + nfl)
            rec.residuals.append(rel)
            rec.backward_errors.append(eta)
            if not math.isfinite(nr):
                rec.reason = "non-finite residual"
                return False, u0, rec, rel, eta
            if rel <= self.rtol or eta <= ETA_TOL:
                rec.converged = True
                return True, u, rec, rel, eta
            first = nr if first is None else first
            if it == self.max_iter:
                rec.reason = f"no convergence in {self.max_iter} iterations"
                return False, u0, rec, rel, eta
            if nr > 1e6 * first:
                rec.reason = "residual diverging"
                return False, u0, rec, rel, eta
            if it >= 6 and rel > 0.5 * rec.residuals[-6]:
                rec.reason = "residual stagnating (less than halved over 5 iterations)"
                return False, u0, rec, rel, eta
            try:
                self.solver.factor(A)
                du = np.asarray(self.solver.solve(R)).reshape(R.shape)
                checked_residual(A, du, R, k1)
            except (SolveResidualError, RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                rec.reason = f"tangent solve failed ({type(e).__name__}: {e})"
                return False, u0, rec, rel, eta
            u[fr] -= du
        rec.reason = "unreachable"                                      # pragma: no cover
        return False, u0, rec, np.inf, np.inf

    def _solve_case(self, E: np.ndarray, f: np.ndarray, u_start: np.ndarray | None) -> tuple[np.ndarray, CaseReport]:
        rep = CaseReport()
        nd = self.domain.ndof
        if not np.any(f[self.free]):
            return np.zeros(nd), rep
        it = self.interpolation
        cutoff = None if it is None else it.cutoff
        rep.cutoff = cutoff
        if u_start is not None:                                        # warm start: one increment at full load
            ok, u, rec, rel, eta = self._newton(E, f, u_start, cutoff, 1.0)
            rep.steps.append(rec)
            if ok:
                rep.relative_residual, rep.backward_error = rel, eta
                return u, rep
        while True:
            lam, dlam, u, streak = 0.0, 1.0 / self.load_steps, np.zeros(nd), 0
            failed = False
            while lam < 1.0:
                target = min(1.0, lam + dlam)
                if 1.0 - target < 1e-12:
                    target = 1.0
                ok, unew, rec, rel, eta = self._newton(E, target * f, u, cutoff, target)
                rep.steps.append(rec)
                if ok:
                    lam, u, streak = target, unew, streak + 1
                    if streak >= 2:
                        dlam = min(1.5 * dlam, 1.0 / self.load_steps)
                    if lam == 1.0:
                        rep.relative_residual, rep.backward_error = rel, eta
                    continue
                streak = 0
                rep.cuts += 1
                dlam *= 0.5
                if dlam < self.min_step:
                    failed = True
                    break
            if not failed:
                return u, rep
            nxt = self._next_cutoff(E, cutoff)
            if nxt is not None:
                cutoff = nxt
                rep.cutoff = cutoff
                rep.restarts += 1
                continue
            last = rep.steps[-1]
            hist = ", ".join(f"{r:.2e}" for r in last.residuals)
            raise NonlinearConvergenceError(
                f"Newton-Raphson failed: equilibrium reached up to load factor {lam:.6g} of 1; the increment to "
                f"{last.load_factor:.6g} failed after halving to {dlam:.3g} < min_step {self.min_step:g} "
                f"({last.reason}; relative residuals {hist})"
                + ("" if cutoff is None else f"; energy-interpolation cutoff {cutoff:g}"))

    def _next_cutoff(self, E: np.ndarray, cutoff: float | None) -> float | None:
        """Next cutoff of the adaptive rule that changes some element's γ by more than 1e-3 (raising it where no γ
        moves would repeat the same failed solve), or None when the rule is exhausted."""
        it = self.interpolation
        if it is None or cutoff is None or it.cutoff_step <= 0:
            return None
        g0 = it.gamma(E, cutoff)[0]
        c = cutoff
        while c + it.cutoff_step <= it.cutoff_max + 1e-12:
            c = min(c + it.cutoff_step, it.cutoff_max)
            if np.abs(it.gamma(E, c)[0] - g0).max() > 1e-3:
                return c
        return None

    def solve_numpy(self, E_active: np.ndarray, F: np.ndarray) -> np.ndarray:
        """U (ndof, n_cases) at moduli ``E_active`` (na,) and dead loads ``F`` (ndof,) or (ndof, n_cases); no autograd.
        Raises :class:`NonlinearConvergenceError` when a case does not reach the full load."""
        E = check_moduli(E_active, self.n_elements)
        if not self._kinematics_checked:
            self.check_kinematics()
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        if F2.shape[0] != self.domain.ndof:
            raise ValueError(f"loads have {F2.shape[0]} rows, the domain {self.domain.ndof} DOFs")
        U = np.zeros(F2.shape)
        reports = []
        warm = self._warm if (self.warm_start and self._warm is not None and self._warm.shape == F2.shape) else None
        for c in range(F2.shape[1]):
            U[:, c], rep = self._solve_case(E, F2[:, c], None if warm is None else warm[:, c])
            reports.append(rep)
        self.last_reports = reports
        self.n_solves += 1
        if self.warm_start:
            self._warm = U.copy()
        return U

    def cutoffs(self) -> list[float | None]:
        """Energy-interpolation cutoff of each case of the last solve."""
        return [r.cutoff for r in self.last_reports]

    def adjoint_gradient(self, E_active: np.ndarray, U: np.ndarray, G: np.ndarray,
                         cutoffs: Sequence[float | None] | None = None) -> np.ndarray:
        """∂J/∂E (na,) for J with ∂J/∂U = ``G`` (ndof, n_cases) at the converged ``U`` (one adjoint solve per case
        with a nonzero ``G`` on the free DOFs)."""
        E = np.asarray(E_active, float)
        fr = self.free
        dE = np.zeros(self.n_elements)
        for c in range(U.shape[1]):
            g = G[fr, c]
            if not np.any(g):
                continue
            cut = None if cutoffs is None else cutoffs[c]
            st = self.element_state(U[:, c], E, cut, need_K=True, need_dE=True)
            A = self._assemble_mat(st["K"])
            self.solver.factor(A)
            x = np.asarray(self.solver.solve(g)).reshape(g.shape)
            checked_residual(A, x, g, matrix_norm1(A))
            lam = np.zeros(self.domain.ndof)
            lam[fr] = x
            dE -= np.einsum("ep,ep->e", lam[self.edof], st["dfdE"])
            self.n_adjoint_solves += 1
        return dE

    # ------------------------------------------------------------------ autograd
    def solve(self, E_active: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        """Differentiable ``U(E)``: (ndof, n_cases) float64 tensor; ``E_active`` (na,) tensor."""
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        return _NonlinearSolve.apply(E_active, self, F2)


class _NonlinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E_active: torch.Tensor, model: NonlinearElasticity, F: np.ndarray) -> torch.Tensor:
        E_np = E_active.detach().cpu().double().numpy().copy()
        U = model.solve_numpy(E_np, F)
        ctx.model, ctx.U, ctx.E, ctx.cutoffs = model, U, E_np, model.cutoffs()
        ctx.device, ctx.dtype = E_active.device, E_active.dtype
        return torch.from_numpy(U).to(device=E_active.device, dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, gU: torch.Tensor) -> tuple[Any, None, None]:
        G = gU.detach().cpu().double().numpy()
        dE = ctx.model.adjoint_gradient(ctx.E, ctx.U, G, ctx.cutoffs)
        return torch.from_numpy(dE).to(device=ctx.device, dtype=ctx.dtype), None, None

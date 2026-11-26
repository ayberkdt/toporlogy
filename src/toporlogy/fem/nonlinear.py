"""Non-linear static solid mechanics on the regular grid: Total Lagrangian large displacements and small-strain
inelastic materials, through the constitutive protocol of :mod:`toporlogy.core.constitutive`.

Kinematics and balance (reference configuration X, displacement u, element-wise):

    finite:  F = I + ∇_X u ,   E = ½ (Fᵀ F − I) ,   S = ∂ψ/∂E (second Piola–Kirchhoff),
             f_int,e = ∫ Bᵀ S dV ,   K_T,e = ∫ (Bᵀ ℂ B + Gᵀ S G) dV ,
    small:   ε = sym ∇u ,   f_int,e = ∫ B₀ᵀ σ dV ,   K_T,e = ∫ B₀ᵀ C_alg B₀ dV ,

with B the non-linear strain matrix (δE = B δu, built from F; B₀ the linear one, F = I) and the geometric term
Gᵀ S G = (∇N S ∇Nᵀ) ⊗ I. Equilibrium R(u) = f_int(u) − λ F = 0 for the dead load F (does not follow the
deformation), solved by Newton–Raphson (:mod:`toporlogy.solvers.newton`) with the consistent tangent. Elements: Q4
(2-D, plane stress or plane strain, thickness t) and H8 (3-D), full 2 × 2(× 2) Gauss rule, the same grid, node and
DOF order as :class:`toporlogy.fem.elastic.Elasticity`; element moduli E_e scale the element's internal force and
tangent (the same ``E_active`` that ``Elasticity`` takes).

Materials (unit modulus, multiplied by E_e; :mod:`toporlogy.core.constitutive`):

* ``"svk"``: :class:`~toporlogy.core.constitutive.SaintVenantKirchhoff`, S = D E in Voigt form with the same
  unit-modulus D as ``Elasticity`` (isotropic plane stress / plane strain / 3-D from ``nu``, or any D given as
  ``elasticity``).
* ``"neo_hooke"``: :class:`~toporlogy.core.constitutive.NeoHooke` (compressible, κ = E/(3(1 − 2ν)),
  μ = E/(2(1 + ν)); Feng, Zhang & Khandelwal 2025, Eqs. 9–10). Plane strain: C₃₃ = 1. Plane stress: C₃₃ from
  S₃₃ = 0 at every Gauss point, condensed tangent.
* any model object of the protocol (``material=``): small-strain ones (:class:`LinearElastic`,
  :class:`J2Plasticity`) use the linear kinematics; a model with internal variables (J2) keeps a committed state
  per Gauss point and load case, commits it only after a converged load increment and rolls back on failure.
  Path-dependent models: no design sensitivity (``solve`` refuses), no energy interpolation, no warm start.

Both hyperelastic laws reduce to the linear elasticity of ``Elasticity`` for small displacements (same D).

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
same failed solve). With ``policy="adaptive"`` (default) every new solve starts again at ``cutoff``; with
``policy="monotone"`` it starts at the highest cutoff any earlier solve of this physics reached (the model changes
only upward, and only between evaluations that the caller can see). Every change of the returned cutoffs between
two solves increments :attr:`NonlinearElasticity.model_epoch` and is logged in :attr:`model_log`: an optimizer must
not reuse its history (moving asymptotes, multipliers) across an epoch change. ``cutoff_step = 0`` fixes the cutoff.
Its dependence on the design is part of the residual's design derivative (∂γ/∂E_e), so the adjoint is consistent
for the cutoff of the solve. When the cutoff is raised during a design evaluation, the response is a different
(more linear) model: the objective is then discontinuous in the design and no classical derivative exists across
the switch. [Wang et al. 2014 not read; the scheme as read in the secondary source.]

Newton–Raphson (per load case; cases are solved independently — no superposition, no shared state):

* load factor λ from 0 to 1 in ``load_steps`` equal increments (:func:`toporlogy.solvers.continuation.load_control`);
  a failed increment is halved (at most down to ``min_step``), after two successful increments in a row it grows by
  1.5 (at most to 1/load_steps);
* convergence of an increment: ‖R_free‖₂ ≤ ``rtol`` · ‖λ F_free‖₂, or backward error
  η = ‖R_free‖ / (‖K_T‖₁ ‖u_free‖ + ‖λ F_free‖) ≤ :data:`ETA_TOL` — the definition of ``checked_residual``. The
  residual is formed from element forces K_e-sized times u_e, so its rounding floor is ~eps ‖K‖ ‖u‖, which exceeds
  rtol ‖λF‖ when ‖K‖ ‖u‖ ≫ ‖F‖ (slender members, high stiffness contrast; measured 25 Nov 2026 on a cantilever
  L/H = 50: relative residual floor 2.7e-9 at λ = 2e-4);
* an increment fails with a code (:data:`toporlogy.solvers.newton.CODES`): ``max_iterations``, ``non_finite``,
  ``divergence`` (residual above 1e6 × the first one), ``stagnation``, ``invalid_deformation`` (det F ≤ 0 with
  neo-Hooke, no plane-stress solution), ``return_mapping``, ``singular_tangent`` (a linear solve that fails its
  residual check), ``line_search`` (with ``line_search="backtracking"``);
* when the smallest increment fails (and the cutoff cannot be raised), :class:`NonlinearConvergenceError`
  (code ``step_limit``, ``cause`` the code of the last failed increment) is raised with the load factor reached and
  the residual history. There is no silent partial result.

Path following beyond limit points: :meth:`NonlinearElasticity.continuation_problem` gives the residual / tangent /
load of one case to :func:`toporlogy.solvers.continuation.arc_length` and ``displacement_control`` (with an
indefinite direct solver).

:meth:`NonlinearElasticity.solve` is differentiable (``torch.autograd.Function``): the backward pass assembles the
tangent at the converged state and solves ONE adjoint system per load case, K_T λ = ∂J/∂U (K_T is symmetric:
hyperelastic material, dead load), and returns ∂J/∂E_e = −λ_eᵀ ∂f_int,e/∂E_e. No self-adjoint shortcut: for the end
compliance Fᵀu the adjoint λ is not u.

Not provided: follower loads, prescribed non-zero displacements, contact, incompatible-mode elements (Q4IM, H8IM),
anisotropic neo-Hooke, finite-strain plasticity, design sensitivities of path-dependent models, iterative /
multigrid solvers.
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
from toporlogy.core.constitutive import Kinematics, NeoHooke, MaterialRequest, SaintVenantKirchhoff, StateStore
from toporlogy.core.domain import Domain
from toporlogy.core.material import unique_matrices
from toporlogy.fem.elastic import checked_residual, matrix_norm1
from toporlogy.fem.elements import H8_CORNER, Q4_CORNER, D_iso, D_plane_strain, D_plane_stress, indexed_matmul
from toporlogy.fem.mechanisms import SolveResidualError, check_moduli, check_supports, fixed_dof_rows
from toporlogy.references import cite
from toporlogy.solvers.continuation import IndefiniteDirectSolver, load_control
from toporlogy.solvers.direct import DirectSolver
from toporlogy.solvers.newton import LinearSolveFailure, Linearization, NewtonSettings, StepRecord, newton

__all__ = ["NonlinearElasticity", "NonlinearConvergenceError", "EnergyInterpolation", "CaseReport", "MATERIALS",
           "ETA_TOL", "CUTOFF_POLICIES"]

MATERIALS = ("svk", "neo_hooke")
CUTOFF_POLICIES = ("adaptive", "monotone")
ETA_TOL = 1e-13            # backward error of the residual accepted as converged (rounding floor)
_G = 1.0 / math.sqrt(3.0)


class NonlinearConvergenceError(RuntimeError):
    """Newton–Raphson did not reach equilibrium at the full load (see the module notes for the rule). ``code`` is
    ``step_limit``; ``cause`` the failure code of the last increment; ``load_factor`` the load factor reached."""

    code = "step_limit"

    def __init__(self, message: str, cause: str = "", load_factor: float = 0.0) -> None:
        super().__init__(message)
        self.cause, self.load_factor = cause, load_factor


@cite("wang_lazarov_sigmund_jensen_2014", "feng_zhang_khandelwal_2025", "zhang_alberdi_khandelwal_2018",
      note="energy interpolation and its adaptive cutoff, as written in Feng et al. 2025, Eqs. (11)-(14) and the "
           "remark on p. 11; the 2014 and 2018 papers themselves were not read")
@dataclass(frozen=True)
class EnergyInterpolation:
    """Parameters of the energy interpolation (module notes): γ = exp(β ρ̃)/(exp(β c) + exp(β ρ̃)),
    ρ̃ = (E_e / E0)^(1/p). Defaults β = 120, c0 = 0.1, Δc = 0.02, c ≤ 1 as in Feng, Zhang & Khandelwal (2025).
    ``policy``: adaptive | monotone (module notes)."""

    E0: float = 1.0
    beta: float = 120.0
    cutoff: float = 0.1
    cutoff_step: float = 0.02
    cutoff_max: float = 1.0
    density_exponent: float = 3.0
    policy: str = "adaptive"

    def __post_init__(self) -> None:
        if not (self.E0 > 0 and self.beta > 0 and self.density_exponent > 0):
            raise ValueError("energy interpolation: E0, beta and density_exponent must be positive")
        if not (0.0 <= self.cutoff <= self.cutoff_max) or self.cutoff_step < 0:
            raise ValueError("energy interpolation: 0 <= cutoff <= cutoff_max and cutoff_step >= 0 are required")
        if self.policy not in CUTOFF_POLICIES:
            raise ValueError(f"energy interpolation policy: {' | '.join(CUTOFF_POLICIES)}, got {self.policy!r}")

    def gamma(self, E: np.ndarray, cutoff: float,
              density: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(γ, dγ/dE) per element for moduli E (> 0) and the given cutoff. ``density`` = (ρ, dρ/dE): the real
        density of the design (``NonlinearElasticity.density_map``); default ρ̃ = (E/E0)^(1/p)."""
        if density is None:
            p = self.density_exponent
            rho = (np.asarray(E, float) / self.E0) ** (1.0 / p)
            drho = rho / (p * np.asarray(E, float))
        else:
            rho, drho = (np.asarray(a, float) for a in density)
        z = self.beta * (rho - cutoff)
        g = 0.5 * (1.0 + np.tanh(0.5 * z))                    # = exp(βρ)/(exp(βc)+exp(βρ)), overflow-free
        dg = self.beta * g * (1.0 - g) * drho
        return g, dg


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


# ----------------------------------------------------------------------------------------------- Newton problems
class _CaseProblem:
    """One load case at load factor λ for :func:`toporlogy.solvers.newton.newton`: unknowns u_free."""

    def __init__(self, phys: NonlinearElasticity, E: np.ndarray, fl: np.ndarray, cutoff: float | None,
                 store: StateStore | None, need_energy: bool) -> None:
        self.phys, self.E, self.fl, self.cutoff, self.store = phys, E, fl, cutoff, store
        self.need_energy = need_energy
        self.conservative = bool(getattr(phys.model, "conservative", False)) and store is None

    def full(self, x: np.ndarray) -> np.ndarray:
        u = np.zeros(self.phys.domain.ndof)
        u[self.phys.free] = x
        return u

    def linearize(self, x: np.ndarray) -> Linearization:
        p = self.phys
        u = self.full(x)
        st = p.element_state(u, self.E, self.cutoff, state=None if self.store is None else self.store.committed,
                             tangent_kind=p.tangent_kind)
        R = p._assemble_vec(st["f"])[p.free] - self.fl[p.free]
        A = p._assemble_mat(st["K"])
        en = None
        if self.need_energy and self.conservative:
            en = p.strain_energy(u, self.E, self.cutoff) - float(self.fl @ u)
        return Linearization(R, A, matrix_norm1(A), en, st)

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray:
        A = lin.matrix
        try:
            self.phys.solver.factor(A)
            du = np.asarray(self.phys.solver.solve(rhs)).reshape(rhs.shape)
            checked_residual(A, du, rhs, lin.norm1)
        except (SolveResidualError, RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            raise LinearSolveFailure(f"{type(e).__name__}: {e}") from e
        return du


class _ContinuationProblem:
    """One load case R(x, λ) = f_int(u) − λ F for :mod:`toporlogy.solvers.continuation` (indefinite solver)."""

    def __init__(self, phys: NonlinearElasticity, E: np.ndarray, F: np.ndarray, cutoff: float | None,
                 solver: Any) -> None:
        self.phys, self.E, self.F, self.cutoff = phys, E, F, cutoff
        self.solver = solver if solver is not None else IndefiniteDirectSolver()
        self.load = F[phys.free].copy()
        self.store = None if not phys.model.state_fields else StateStore(phys.model, phys.n_elements * phys.ngp)
        self._factored: Any = None

    def full(self, x: np.ndarray) -> np.ndarray:
        u = np.zeros(self.phys.domain.ndof)
        u[self.phys.free] = x
        return u

    def linearize(self, x: np.ndarray, lam: float) -> Linearization:
        p = self.phys
        u = self.full(x)
        st = p.element_state(u, self.E, self.cutoff, state=None if self.store is None else self.store.committed)
        R = p._assemble_vec(st["f"])[p.free] - lam * self.load
        A = p._assemble_mat(st["K"])
        return Linearization(R, A, matrix_norm1(A), None, st)

    def _factor(self, lin: Linearization) -> None:
        if self._factored is not lin:
            self.solver.factor(lin.matrix)
            self._factored = lin

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray:
        try:
            self._factor(lin)
            return np.asarray(self.solver.solve(rhs))
        except LinearSolveFailure:
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            raise LinearSolveFailure(f"{type(e).__name__}: {e}") from e

    def det_sign(self, lin: Linearization) -> int | None:
        f = getattr(self.solver, "det_sign", None)
        if f is None:
            return None
        self._factor(lin)
        return int(f())

    def commit(self, x: np.ndarray, lam: float) -> None:
        if self.store is not None:
            st = self.linearize(x, lam).aux
            self.store.set_trial(st["trial"])
            self.store.commit()


# ----------------------------------------------------------------------------------------------- physics
class NonlinearElasticity:
    """Static non-linear solid (Total Lagrangian, or small strain with an inelastic law) on a :class:`Domain` with
    fixed DOFs.

    Parameters
    ----------
    domain : Domain
    fixed : sequence of int
        Box DOF numbers with zero displacement (homogeneous supports only).
    nu : float
        Poisson's ratio (isotropic D, and κ, μ of neo-Hooke).
    material : {"svk", "neo_hooke"} or a constitutive model
        A model object of :mod:`toporlogy.core.constitutive` (unit modulus; e.g. ``LinearElastic(D)``,
        ``J2Plasticity(...)``); ``nu`` is then not used and ``elasticity`` is not accepted.
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
        Low-density remedy (module notes). None = every element fully non-linear. Finite-strain models only.
    load_steps, max_iter, rtol, min_step : Newton–Raphson controls (module notes).
    line_search : {"none", "backtracking"}; merit : {"residual", "energy"}
        Globalisation of each Newton step (:mod:`toporlogy.solvers.newton`); default the plain full step.
    log_energy : bool
        Record the stored energy of every converged increment in its step record (one extra element pass).
    tangent_kind : {"algorithmic", "continuum"}
        Tangent of a path-dependent model in Newton: the consistent algorithmic one (default) or the continuum
        (rate) tangent (a negative control: not consistent with the update, Newton loses its quadratic rate).
    warm_start : bool
        Start each case from the previous converged displacement at the full load (one increment), falling back to
        load stepping from zero when that fails. Off by default: the result then depends on the design only, not on
        the evaluation history. Refused for path-dependent models.
    solver : object with ``factor(A_upper)`` / ``solve(B)``, optional
        Default :class:`DirectSolver` (symmetric; the tangent of a stable equilibrium is positive definite).
    """

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray, nu: float = 0.3, material: Any = "svk",
                 plane: str = "stress", element: str | None = None, elasticity: np.ndarray | None = None,
                 energy_interpolation: EnergyInterpolation | None = None, load_steps: int = 10, max_iter: int = 30,
                 rtol: float = 1e-9, min_step: float = 1e-4, warm_start: bool = False, solver: Any = None,
                 line_search: str = "none", merit: str = "residual", log_energy: bool = False,
                 tangent_kind: str = "algorithmic") -> None:
        dim = domain.dim
        named = isinstance(material, str)
        if named and material not in MATERIALS:
            raise ValueError(f"material: {' | '.join(MATERIALS)} or a constitutive model, got {material!r}")
        if not named and not (hasattr(material, "evaluate") and hasattr(material, "kinematics")):
            raise ValueError(f"material: {' | '.join(MATERIALS)} or a constitutive model, got {material!r}")
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
        self.domain, self.plane = domain, plane if dim == 2 else "stress"
        self.material = material if named else str(getattr(material, "name", type(material).__name__))
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
            self.nu = float(nu) if named else None
            self.D = D_iso(nu) if dim == 3 else (D_plane_strain(nu) if plane == "strain" else D_plane_stress(nu))
        # per-element D: the distinct matrices and each element's index (as Elasticity.ke_index); None = one D
        self.d_index: np.ndarray | None = None
        if self.D.ndim == 3:
            Du, self.d_index = unique_matrices(self.D)
            self.model: Any = SaintVenantKirchhoff(Du)
        elif not named:
            self.model = material
        elif material == "svk":
            self.model = SaintVenantKirchhoff(self.D)
        else:
            assert self.nu is not None
            self.model = NeoHooke(self.nu)
        self.measure = str(self.model.kinematics)
        self._plane_kin = "3d" if dim == 3 else self.plane
        self.stateful = bool(self.model.state_fields)
        if self.measure == "small" and energy_interpolation is not None:
            raise ValueError("energy_interpolation applies to finite-strain models only")
        if self.stateful and warm_start:
            raise ValueError("warm_start is refused for path-dependent models (the state depends on the path)")
        self.interpolation = energy_interpolation
        # E (active elements) -> (rho, drho/dE) of the real density; set by Study from its SIMP / RAMP
        # interpolation (docs/olcumler/gercek_yogunluk_onkayit.md). None: rho = (E/E0)^(1/p).
        self.density_map: Any = None
        self.load_steps, self.max_iter, self.rtol, self.min_step = int(load_steps), int(max_iter), float(rtol), \
            float(min_step)
        self.newton_settings = NewtonSettings(max_iter=self.max_iter, rtol=self.rtol, eta_tol=ETA_TOL,
                                              line_search=line_search, merit=merit)
        self.log_energy = bool(log_energy)
        if tangent_kind not in ("algorithmic", "continuum"):
            raise ValueError("tangent_kind: algorithmic | continuum")
        self.tangent_kind = tangent_kind
        self.warm_start = bool(warm_start)
        self.geom = domain.h if dim == 3 else domain.thickness       # as Elasticity (used by responses / tests)
        self._dN = _shape_gradients(dim, domain.h)                     # (ngp, dim, nnode)
        self.ngp = len(self._dN)
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
        m = 1 if self.d_index is None else len(self.model.D)
        st0 = None if not self.stateful else StateStore(self.model, m * self.ngp).committed
        K0 = self._unit_state(np.zeros((m, nl)), np.ones(m), need_K=True, which=np.arange(m), state=st0)[1]
        assert K0 is not None
        # small-displacement element matrix; per-element D: the distinct matrices (m, nloc, nloc), K0[d_index[e]]
        self.K0: np.ndarray = K0[0] if self.d_index is None else K0
        self._kinematics_checked = False
        self.n_solves = 0
        self.n_adjoint_solves = 0
        self.last_reports: list[CaseReport] = []
        self.last_states: list[dict[str, np.ndarray]] = []
        self._warm: np.ndarray | None = None
        self.model_epoch = 0
        self.model_log: list[tuple[int, list[float | None], list[float | None]]] = []
        self._cutoff_floor: float | None = None
        self._last_cutoffs: list[float | None] | None = None

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
                    which: np.ndarray | None = None, state: dict[str, np.ndarray] | None = None,
                    tangent_kind: str = "algorithmic", extra: dict[str, Any] | None = None
                    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Unit-modulus non-linear element force f_NL(γ u_e) (n, nloc), tangent K_NL(γ u_e) (n, nloc, nloc) and
        energy ψ_NL(γ u_e) (n,) of the elements' displacements ``Ue`` (n, nloc) scaled by ``gam`` (n,).
        ``which``: per-element D, index of each row's D (default: the active elements' :attr:`d_index`).
        ``state``: committed internal variables of the n elements' Gauss points (rows Gauss-point major, g·n + e);
        the trial state, dissipation and diagnostics go to ``extra`` when given."""
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
        finite = self.measure == "finite"
        params = None if which is None else {"index": which}
        req = MaterialRequest(energy=energy, tangent=True, tangent_kind=tangent_kind)
        trials: list[dict[str, np.ndarray]] = []
        diss = np.zeros(n) if self.stateful else None
        for g, dN in enumerate(self._dN):                              # (dim, nn)
            Hg = Ut @ dN.T
            F = eye + Hg if finite else np.broadcast_to(eye, Hg.shape)
            committed = {} if state is None else {k: v[g * n:(g + 1) * n] for k, v in state.items()}
            r = self.model.evaluate(Kinematics(Hg, self.measure, self._plane_kin), params, committed, req)
            S, C, ps = r.stress, r.tangent, r.energy
            if r.trial_state is not None:
                trials.append(r.trial_state)
            if diss is not None and r.dissipation is not None:
                diss += self._w * r.dissipation
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
                assert C is not None
                K += self._w * (Bt @ (C @ B2))
                if finite:
                    St = np.zeros((n, dim, dim))
                    for v, (j, k) in enumerate(pairs):
                        St[:, j, k] = St[:, k, j] = S[:, v]
                    Gm = self._w * (dN.T @ St @ dN)                    # (n, nn, nn)
                    K4 = K.reshape(n, nn, dim, nn, dim)
                    for i in range(dim):
                        K4[:, :, i, :, i] += Gm
        if extra is not None:
            if trials:
                extra["trial"] = {k: np.concatenate([t[k] for t in trials], axis=0) for k in trials[0]}
            extra["dissipation"] = diss
        return f, K, psi

    def _k0_times(self, Ue: np.ndarray) -> np.ndarray:
        """(n_active, nloc) K0_e u_e (K0 symmetric)."""
        if self.d_index is None:
            return Ue @ self.K0
        return indexed_matmul(self.K0, self.d_index, Ue[:, :, None])[:, :, 0]

    def _gamma(self, E: np.ndarray, cutoff: float | None) -> tuple[np.ndarray, np.ndarray]:
        if self.interpolation is None:
            return np.ones(len(E)), np.zeros(len(E))
        return self.interpolation.gamma(E, self.interpolation.cutoff if cutoff is None else cutoff,
                                        self._density(E))

    def _density(self, E: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        return None if self.density_map is None else self.density_map(np.asarray(E, float))

    def element_state(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None, need_K: bool = True,
                      need_dE: bool = False, state: dict[str, np.ndarray] | None = None,
                      tangent_kind: str = "algorithmic") -> dict[str, Any]:
        """Element internal forces ``f`` (na, nloc), tangents ``K`` (na, nloc, nloc) and, with ``need_dE``, the design
        derivative ``dfdE`` = ∂f_int,e/∂E_e (na, nloc) at the full displacement vector ``u`` (ndof,) and moduli
        ``E`` (na,) (energy interpolation with ``cutoff``, default its initial value). Path-dependent models: the
        committed ``state`` (default the initial one), and the ``trial`` state and per-element ``dissipation``
        increment E_e Σ w σ_y Δα in the output."""
        Ue = u[self.edof]
        gam, dgam = self._gamma(E, cutoff)
        interp = self.interpolation is not None
        extra: dict[str, Any] | None = None
        if self.stateful:
            if state is None:
                state = StateStore(self.model, self.n_elements * self.ngp).committed
            extra = {}
        fN, KN, _ = self._unit_state(Ue, gam, need_K or (need_dE and interp), state=state, tangent_kind=tangent_kind,
                                     extra=extra)
        out: dict[str, Any] = {}
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
        if extra is not None:
            out["trial"] = extra.get("trial")
            out["dissipation"] = E * extra["dissipation"]
        return out

    def strain_energy(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None) -> float:
        """Total stored energy Σ_e E_e [ψ_NL(γu) − ψ_L(γu) + ψ_L(u)] (its u-gradient is f_int). Hyperelastic models
        only (a path-dependent model's stored energy needs its state: :meth:`element_state`)."""
        return float(np.sum(self.element_energies(u, E, cutoff)[0]))

    def element_energies(self, u: np.ndarray, E: np.ndarray,
                         cutoff: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(W_e, ∂W_e/∂E_e) per element at fixed ``u``: W_e = E_e [ψ_NL(γu) + (1 − γ²) ψ_L(u)] and its explicit
        design derivative [ψ_NL + (1 − γ²)ψ_L] + E_e γ'(E_e) [f_NL(γu)·u_e − 2γ ψ_L(u)] (γ' = 0 without
        interpolation)."""
        if self.stateful:
            raise ValueError("strain_energy: path-dependent model (the stored energy depends on the state)")
        Ue = u[self.edof]
        E = np.asarray(E, float)
        gam, dgam = self._gamma(E, cutoff)
        fN, _K, psi = self._unit_state(Ue, gam, need_K=False, energy=True)
        assert psi is not None
        if self.d_index is None:
            lin = 0.5 * np.einsum("ep,pq,eq->e", Ue, self.K0, Ue)
        else:
            lin = 0.5 * np.einsum("ep,ep->e", Ue, self._k0_times(Ue))
        w_unit = psi + (1.0 - gam ** 2) * lin
        dw = w_unit + E * dgam * (np.einsum("ep,ep->e", fN, Ue) - 2.0 * gam * lin)
        return E * w_unit, dw

    def min_det_f(self, u: np.ndarray) -> float:
        """Smallest det F over the Gauss points of the active elements at the full displacement ``u`` (1 for
        small-strain models: no deformation measure of the current configuration is used)."""
        if self.measure != "finite":
            return 1.0
        Ue = u[self.edof]
        n, nl = Ue.shape
        Ut = np.swapaxes(Ue.reshape(n, nl // self.dim, self.dim), 1, 2)
        return float(min(np.min(1.0 + T.jacobian_minus_one(Ut @ dN.T)) for dN in self._dN))

    # ------------------------------------------------------------------ assembly
    def _assemble_vec(self, fe: np.ndarray) -> np.ndarray:
        return np.bincount(self.edof.ravel(), weights=fe.ravel(), minlength=self.domain.ndof)

    def _assemble_mat(self, Ke: np.ndarray) -> sp.csr_matrix:
        A = self._P.copy()
        A.data = np.bincount(self._inv, weights=Ke.reshape(len(Ke), -1).ravel()[self._sel], minlength=self._P.nnz)
        return A

    def internal_force(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None,
                       state: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Assembled f_int(u) (ndof,), fixed DOFs included (their entries are reactions)."""
        return self._assemble_vec(self.element_state(u, np.asarray(E, float), cutoff, need_K=False, state=state)["f"])

    def tangent(self, u: np.ndarray, E: np.ndarray, cutoff: float | None = None,
                state: dict[str, np.ndarray] | None = None, tangent_kind: str = "algorithmic") -> sp.csr_matrix:
        """Full symmetric tangent K_T(u) on the free DOFs (nf × nf, ordered as :attr:`free`)."""
        A = self._assemble_mat(self.element_state(u, np.asarray(E, float), cutoff, state=state,
                                                  tangent_kind=tangent_kind)["K"])
        return (A + sp.triu(A, k=1).T).tocsr()

    # ------------------------------------------------------------------ Newton–Raphson
    def _newton(self, E: np.ndarray, fl: np.ndarray, u0: np.ndarray, cutoff: float | None, lam: float,
                store: StateStore | None = None,
                ref: np.ndarray | None = None) -> tuple[bool, np.ndarray, StepRecord, float, float]:
        fr = self.free
        prob = _CaseProblem(self, E, fl, cutoff, store, self.newton_settings.merit == "energy")
        scale = float(np.linalg.norm(fl[fr]))
        if scale == 0.0 and ref is not None:            # unloaded to λ = 0: relative to the load of the path
            scale = float(np.linalg.norm(ref[fr]))
        out = newton(prob, u0[fr], scale, self.newton_settings, lam)
        rec = out.record
        if not out.converged:
            if store is not None:
                store.rollback()
            return False, u0, rec, out.relative_residual, out.backward_error
        u = u0.copy()
        u[fr] = out.x
        assert out.lin is not None
        st = out.lin.aux
        if self.measure == "finite":
            rec.min_det_f = self.min_det_f(u)
        if store is not None:
            store.set_trial(st["trial"])
            rec.dissipation = float(np.sum(st["dissipation"]))
        if self.log_energy and not self.stateful:
            rec.energy = self.strain_energy(u, E, cutoff)
        return True, u, rec, out.relative_residual, out.backward_error

    def _start_cutoff(self) -> float | None:
        it = self.interpolation
        if it is None:
            return None
        if it.policy == "monotone" and self._cutoff_floor is not None:
            return max(it.cutoff, self._cutoff_floor)
        return it.cutoff

    def _solve_case(self, E: np.ndarray, f: np.ndarray, u_start: np.ndarray | None,
                    factors: Sequence[float] | None = None,
                    store: StateStore | None = None) -> tuple[np.ndarray, CaseReport, list[np.ndarray]]:
        rep = CaseReport()
        nd = self.domain.ndof
        path = [1.0] if factors is None else [float(v) for v in factors]
        if not np.any(f[self.free]):
            return np.zeros(nd), rep, [np.zeros(nd) for _ in path]
        cutoff = self._start_cutoff()
        rep.cutoff = cutoff
        if u_start is not None:                                        # warm start: one increment at full load
            ok, u, rec, rel, eta = self._newton(E, f, u_start, cutoff, 1.0)
            rep.steps.append(rec)
            if ok:
                rep.relative_residual, rep.backward_error = rel, eta
                return u, rep, [u]
        while True:
            u, lam, failed, snaps = np.zeros(nd), 0.0, False, []
            res = None

            def attempt(t: float, x: np.ndarray, c: float | None = cutoff
                        ) -> tuple[bool, np.ndarray, StepRecord, float, float]:
                return self._newton(E, t * f, x, c, t, store)
            for end in path:
                res = load_control(attempt, u, lam, end,
                                   1.0 / self.load_steps, self.min_step,
                                   commit=None if store is None else (lambda _l, _x: store.commit()))
                rep.steps.extend(res.records)
                rep.cuts += res.cuts
                u, lam = res.x, res.lam
                if not res.ok:
                    failed = True
                    break
                snaps.append(u)
            assert res is not None
            if not failed:
                rep.relative_residual, rep.backward_error = res.last
                return u, rep, snaps
            nxt = self._next_cutoff(E, cutoff)
            if nxt is not None:
                cutoff = nxt
                rep.cutoff = cutoff
                rep.restarts += 1
                continue
            last = rep.steps[-1]
            hist = ", ".join(f"{r:.2e}" for r in last.residuals)
            raise NonlinearConvergenceError(
                f"Newton-Raphson failed: equilibrium reached up to load factor {lam:.6g} of {path[-1]:g}; the "
                f"increment to {last.load_factor:.6g} failed after halving to {res.dlam:.3g} < min_step "
                f"{self.min_step:g} ({last.reason}; relative residuals {hist})"
                + ("" if cutoff is None else f"; energy-interpolation cutoff {cutoff:g}"), last.code, lam)

    def _next_cutoff(self, E: np.ndarray, cutoff: float | None) -> float | None:
        """Next cutoff of the adaptive rule that changes some element's γ by more than 1e-3 (raising it where no γ
        moves would repeat the same failed solve), or None when the rule is exhausted."""
        it = self.interpolation
        if it is None or cutoff is None or it.cutoff_step <= 0:
            return None
        dens = self._density(E)
        g0 = it.gamma(E, cutoff, dens)[0]
        c = cutoff
        while c + it.cutoff_step <= it.cutoff_max + 1e-12:
            c = min(c + it.cutoff_step, it.cutoff_max)
            if np.abs(it.gamma(E, c, dens)[0] - g0).max() > 1e-3:
                return c
        return None

    def _note_model(self) -> None:
        """Model-change bookkeeping after a solve (module notes): epoch, log, monotone floor."""
        cuts = self.cutoffs()
        if self._last_cutoffs is not None and cuts != self._last_cutoffs:
            self.model_epoch += 1
            self.model_log.append((self.n_solves, list(self._last_cutoffs), list(cuts)))
        self._last_cutoffs = list(cuts)
        vals = [c for c in cuts if c is not None]
        if vals and self.interpolation is not None and self.interpolation.policy == "monotone":
            top = max(vals)
            self._cutoff_floor = top if self._cutoff_floor is None else max(self._cutoff_floor, top)

    def reset_model(self) -> None:
        """Forget the monotone cutoff floor and the last cutoffs (a new optimisation)."""
        self._cutoff_floor, self._last_cutoffs = None, None

    def _check_loads(self, E_active: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        E = check_moduli(E_active, self.n_elements)
        if not self._kinematics_checked:
            self.check_kinematics()
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        if F2.shape[0] != self.domain.ndof:
            raise ValueError(f"loads have {F2.shape[0]} rows, the domain {self.domain.ndof} DOFs")
        return E, F2

    def solve_numpy(self, E_active: np.ndarray, F: np.ndarray) -> np.ndarray:
        """U (ndof, n_cases) at moduli ``E_active`` (na,) and dead loads ``F`` (ndof,) or (ndof, n_cases); no autograd.
        Raises :class:`NonlinearConvergenceError` when a case does not reach the full load. Path-dependent models:
        proportional loading from the initial state per case; the final states are in :attr:`last_states`."""
        E, F2 = self._check_loads(E_active, F)
        U = np.zeros(F2.shape)
        reports, states = [], []
        warm = self._warm if (self.warm_start and self._warm is not None and self._warm.shape == F2.shape) else None
        for c in range(F2.shape[1]):
            store = StateStore(self.model, self.n_elements * self.ngp) if self.stateful else None
            U[:, c], rep, _ = self._solve_case(E, F2[:, c], None if warm is None else warm[:, c], store=store)
            reports.append(rep)
            if store is not None:
                states.append(store.checkpoint())
        self.last_reports = reports
        self.last_states = states
        self.n_solves += 1
        self._note_model()
        if self.warm_start:
            self._warm = U.copy()
        return U

    def solve_history(self, E_active: np.ndarray, F: np.ndarray, factors: Sequence[float],
                      checkpoint: dict[str, Any] | None = None) -> tuple[np.ndarray, CaseReport, dict[str, Any]]:
        """One load case along the piecewise-linear load-factor path 0 → factors[0] → factors[1] → … (loading and
        unloading). Returns U at every factor (ndof, k), the report and a checkpoint (``u``, ``lam``, ``state``)
        from which a later call continues (``checkpoint=``; the path then starts at its ``lam``)."""
        E, F2 = self._check_loads(E_active, F)
        if F2.shape[1] != 1:
            raise ValueError("solve_history: one load case")
        f = F2[:, 0]
        store = StateStore(self.model, self.n_elements * self.ngp) if self.stateful else None
        u0: np.ndarray = np.zeros(self.domain.ndof)
        lam0 = 0.0
        if checkpoint is not None:
            u0, lam0 = np.asarray(checkpoint["u"], float).copy(), float(checkpoint["lam"])
            if store is not None:
                store.restore(checkpoint["state"])
        rep = CaseReport()
        snaps, u, lam = [], u0, lam0
        for end in factors:
            res = load_control(lambda t, x: self._newton(E, t * f, x, None, t, store, f), u, lam, float(end),
                               1.0 / self.load_steps, self.min_step,
                               commit=None if store is None else (lambda _l, _x: store.commit()))
            rep.steps.extend(res.records)
            rep.cuts += res.cuts
            u, lam = res.x, res.lam
            if not res.ok:
                last = rep.steps[-1]
                raise NonlinearConvergenceError(f"Newton-Raphson failed on the load path at load factor {lam:.6g} "
                                                f"(target {end:g}; {last.reason})", last.code, lam)
            rep.relative_residual, rep.backward_error = res.last
            snaps.append(u)
        cp = {"u": u.copy(), "lam": lam, "state": None if store is None else store.checkpoint()}
        self.n_solves += 1
        return np.column_stack(snaps), rep, cp

    def continuation_problem(self, E_active: np.ndarray, F: np.ndarray, cutoff: float | None = None,
                             solver: Any = None) -> _ContinuationProblem:
        """R(x, λ) = f_int(u) − λF of one case on the free DOFs for :mod:`toporlogy.solvers.continuation` (x =
        u_free; default solver :class:`~toporlogy.solvers.continuation.IndefiniteDirectSolver`)."""
        E, F2 = self._check_loads(E_active, F)
        if F2.shape[1] != 1:
            raise ValueError("continuation_problem: one load case")
        return _ContinuationProblem(self, E, F2[:, 0], cutoff if cutoff is not None else self._start_cutoff(), solver)

    def cutoffs(self) -> list[float | None]:
        """Energy-interpolation cutoff of each case of the last solve."""
        return [r.cutoff for r in self.last_reports]

    def adjoint_gradient(self, E_active: np.ndarray, U: np.ndarray, G: np.ndarray,
                         cutoffs: Sequence[float | None] | None = None) -> np.ndarray:
        """∂J/∂E (na,) for J with ∂J/∂U = ``G`` (ndof, n_cases) at the converged ``U`` (one adjoint solve per case
        with a nonzero ``G`` on the free DOFs)."""
        if self.stateful:
            raise NotImplementedError("design sensitivities of path-dependent models are not implemented (the last "
                                      "tangent alone does not give the history gradient)")
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
        if self.stateful:
            raise NotImplementedError("NonlinearElasticity.solve: no design sensitivity for a path-dependent model; "
                                      "use solve_numpy / solve_history")
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

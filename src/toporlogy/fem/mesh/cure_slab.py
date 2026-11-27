"""Coupled thermo-chemical cure of a resin slab and its one-way residual stress (WP16 TS04 stage 2, WP17 H04/H05
levels 2-3; pre-registration ``docs/foundations/prereg/TS7_cure_slab.md``, notes ``docs/foundations/CURE_SLAB.md``).

Heat equation of a neat resin with the cure exotherm (constant ρ, c_p, k; c_p the sensible heat, the reaction heat
enters once, as the source; ΔH in J per kg of resin, exothermic positive)::

    ρ c_p dT/dt = ∇·(k ∇T) + ρ ΔH dα/dt,        dα/dt = f(α, T)  (:mod:`toporlogy.core.cure`).

* Space: the conduction matrix and the Robin facets of :class:`~toporlogy.fem.mesh.problem.HeatProblem`, the consistent
  capacity ρ c_p ∫NNᵀ of :func:`~toporlogy.fem.mesh.kernels.mass`; Dirichlet temperatures on node sets. Geometry is
  whatever mesh is given; the verified use is a strip ``rectangle(1, n, Lx, L)`` with y through the thickness.
* Time: backward Euler. The cure state α is carried at the quadrature points and advanced with
  ``kinetics.advance(α_n, T_n, T_{n+1}, Δt)`` (backward Euler, rate at the end temperature, bounded, no clipping) at the
  interpolated temperature T_q = Σ N_a(x_q) T_a.
* Coupling: **monolithic Newton on the nodal temperatures**, α eliminated at every quadrature point. Residual
  R(T) = (M/Δt + K) T − (M/Δt) T_n − S(T) − f(t_{n+1}), S_a = Σ_q N_a w_q ρ ΔH (α_{n+1,q} − α_{n,q})/Δt; Jacobian
  J = M/Δt + K − Σ_q N_a N_b w_q ρ ΔH/Δt dα_{n+1}/dT, with dα_{n+1}/dT = Δt f_T/(1 − Δt f_α) (implicit function theorem
  on x − α_n − Δt f(x, T) = 0). Converged when ‖R_free‖_∞ ≤ rtol·scale + noise (scale: the largest ∞-norm over the free
  rows of the inertia, conduction, source and load terms; noise: 8 ε_mach times the rounding magnitude of the terms),
  at most ``max_iter`` iterations. A step that does not converge, leaves the contract of the kinetics (T ≤ 0, α outside
  [0, 1]) or fails in the linear solve raises :class:`~toporlogy.fem.mesh.history.StepRejected`; (T, α, t, history) are
  committed only after convergence, so a rejected step changes nothing.
* Energy: every step reports stored heat ρ c_p Σ ∫ΔT, reaction heat ρ ΔH Σ ∫Δα, and the heat entering through the Robin
  facets (backward Euler: flux at t_{n+1}) and through the Dirichlet nodes (the nodal reactions); the budget
  stored − reaction − boundary equals Δt Σ_free R (the Newton residual).
* Mechanics (:func:`residual_stress`): the stored step history is replayed through
  :class:`~toporlogy.fem.mesh.history.HistoryProblem` with :class:`~toporlogy.core.cure.CuringThermoElasticPoint` (same
  kinetics, one sub-step, same quadrature points and the same interpolation of T, so the point model reproduces the
  thermal α), plane strain. One-way chain T → α → σ: the deformation does not act on the heat.

Not implemented: temperature- or cure-dependent k, c_p, ρ; vitrification-controlled kinetics; viscoelastic relaxation
during cure; finite strain; release (de-moulding) events; contact; 3-D meshes are not excluded by the thermal part but
are unverified. Units: seconds, kelvin, metres, J, kg, Pa. All test data are synthetic."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from toporlogy.core.constitutive import ConstitutiveFailure, TimeStep
from toporlogy.core.cure import ChemicalShrinkage, CuringThermoElasticPoint, CureKinetics, ReactionHeat
from toporlogy.core.properties import IsotropicModuliOfT, ThermalExpansion
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh._newton import newton_free
from toporlogy.fem.mesh._validation import Scalar, max_iterations, positive, time_function
from toporlogy.fem.mesh.history import HistoryProblem, StepRejected
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import HeatProblem

__all__ = ["CureSlab", "CureSlabStep", "ThermalStepResult", "MechanicalResult", "residual_stress"]

_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class CureSlabStep:
    """One accepted step of the history: nodal temperatures before and after, time and step (s)."""

    t: float
    dt: float
    T_prev: np.ndarray
    T_next: np.ndarray


@dataclass
class ThermalStepResult:
    """Outcome of an accepted thermal step. Energies in J per unit out-of-plane thickness (2-D) of the whole mesh."""

    t: float                          # end time of the step
    dt: float
    T: np.ndarray                     # nodal temperatures at t + dt
    alpha: np.ndarray                 # cure state at the quadrature points (ne, nq)
    iterations: int
    residual: float                   # final ‖R_free‖_∞ (W)
    stored: float                     # ρ c_p ∫(T_{n+1} − T_n) dV
    reaction: float                   # ρ ΔH ∫(α_{n+1} − α_n) dV
    robin_heat: float                 # heat entering through the Robin facets during the step (flux at t_{n+1})
    dirichlet_heat: float             # heat supplied by the Dirichlet reservoirs during the step
    energy_residual: float            # stored − reaction − robin_heat − dirichlet_heat (= Δt Σ_free R)
    residual_history: tuple[float, ...] = field(default_factory=tuple)


@dataclass
class MechanicalResult:
    """Residual-stress solve: stress (ne, nq, 3) [xx, yy, xy] after the last step, the cure state the point model
    carried (ne, nq), displacements, and (if kept) the stress after every step."""

    stress: np.ndarray
    alpha: np.ndarray
    u: np.ndarray
    stress_history: list[np.ndarray]
    iterations: list[int]
    problem: HistoryProblem


class CureSlab:
    """Transient heat conduction with the cure exotherm (module notes).

    Parameters
    ----------
    mesh : Mesh
        2-D (verified use) mesh; for the strip ``rectangle(1, n, Lx, L)`` y is the thickness direction.
    kinetics : CureKinetics
        dα/dt = f(α, T) (:mod:`toporlogy.core.cure`), used with one backward-Euler sub-step per step.
    reaction : ReactionHeat
        ΔH in J/kg of resin.
    rho, cp, conductivity : float
        Density (kg/m^3, > 0), sensible heat capacity (J/(kg K), > 0) and conductivity (W/(m K), >= 0), constant.
    T0 : float | (n_nodes,) array
        Initial temperature in kelvin (> 0).
    alpha0 : float | (ne, nq) array
        Initial conversion in [0, 1].
    thickness : float
        Out-of-plane thickness of a 2-D mesh (m); the energies refer to it.
    """

    def __init__(self, mesh: Mesh, kinetics: CureKinetics, reaction: ReactionHeat, rho: float, cp: float,
                 conductivity: float, T0: Any, alpha0: Any = 0.0, thickness: float = 1.0,
                 degree: int | None = None) -> None:
        if not isinstance(kinetics, CureKinetics):
            raise ValueError("kinetics must be a CureKinetics")
        if not isinstance(reaction, ReactionHeat):
            raise ValueError("reaction must be a ReactionHeat")
        self.mesh, self.kinetics, self.reaction = mesh, kinetics, reaction
        self.rho = positive(rho, "rho")
        self.cp = positive(cp, "cp")
        self.conductivity = positive(conductivity, "conductivity", allow_zero=True)
        self.thickness = positive(thickness, "thickness")
        self.heat = HeatProblem(mesh, kappa=self.conductivity, thickness=self.thickness, degree=degree)
        self.degree = self.heat.degree
        g = self.heat.geom
        self._N = g.N                                                   # (nq, nloc)
        self._w = g.dV * (self.thickness if mesh.dim == 2 else 1.0)     # (ne, nq) quadrature weights
        self.cells = mesh.cells
        self.n_nodes = mesh.n_nodes
        self.shape_q = self._w.shape
        self._M = sp.csr_matrix(self.heat.pattern.assemble(kn.mass(g, self.rho * self.cp, ncomp=1,
                                                                   thickness=self.thickness)))
        self._active = np.unique(self.heat.cell_dofs)
        T0a = np.broadcast_to(np.asarray(T0, dtype=float), (self.n_nodes,)).copy()
        if not np.all(np.isfinite(T0a)) or np.any(T0a <= 0.0):
            raise ValueError("T0 must be finite and > 0 K")
        a0 = np.broadcast_to(np.asarray(alpha0, dtype=float), self.shape_q).copy()
        if not np.all(np.isfinite(a0)) or np.any(a0 < 0.0) or np.any(a0 > 1.0):
            raise ValueError("alpha0 must lie in [0, 1]")
        self.alpha0_field = a0.copy()
        self.alpha0_field.setflags(write=False)
        self._T, self._alpha, self._t = T0a, a0, 0.0
        self._T.setflags(write=False)
        self._alpha.setflags(write=False)
        self._robin: list[tuple[np.ndarray, Callable[[float], float]]] = []
        self._dirichlet: list[tuple[np.ndarray, Callable[[float], float]]] = []
        self.history: list[CureSlabStep] = []

    # ------------------------------------------------------------------ state
    @property
    def t(self) -> float:
        return self._t

    @property
    def T(self) -> np.ndarray:
        """Committed nodal temperatures (read-only)."""
        return self._T

    @property
    def alpha(self) -> np.ndarray:
        """Committed cure state at the quadrature points (ne, nq) (read-only)."""
        return self._alpha

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights w_q (ne, nq) in m^2 (2-D, thickness included): ∫ g dV = Σ w_q g_q."""
        return self._w

    @property
    def K(self) -> sp.csr_matrix:
        """Conduction plus Robin matrix (W/K)."""
        return self.heat.K

    @property
    def M(self) -> sp.csr_matrix:
        """Consistent capacity ρ c_p ∫NNᵀ (J/K)."""
        return self._M

    def quadrature_temperature(self, T: np.ndarray) -> np.ndarray:
        """T_q = Σ_a N_a(x_q) T_a at the quadrature points, (ne, nq)."""
        return np.einsum("qa,ea->eq", self._N, np.asarray(T, dtype=float)[self.cells])

    # ------------------------------------------------------------------ boundaries
    def add_robin(self, facets: np.ndarray, h: float, T_inf: Scalar) -> None:
        """Convection h (T − T∞(t)) on ``facets`` ((cell, local facet) pairs), h in W/(m^2 K) > 0, T∞ in K (a number
        or a function of time evaluated at the END of each step: backward Euler)."""
        unit = self.heat.robin(facets, positive(h, "h"), 1.0)                 # h ∫ N dS (load for T∞ = 1)
        self._robin.append((unit, time_function(T_inf, "T_inf")))

    def add_dirichlet(self, nodes: np.ndarray, value: Scalar) -> None:
        """Prescribed temperature (K, a number or a function of time evaluated at the END of each step) on nodes."""
        nodes = np.unique(np.asarray(nodes, np.int64).ravel())
        if len(nodes) and (nodes.min() < 0 or nodes.max() >= self.n_nodes):
            raise ValueError("Dirichlet node out of range")
        for prev, _ in self._dirichlet:
            if np.intersect1d(prev, nodes).size:
                raise ValueError("a node already carries a Dirichlet temperature")
        self._dirichlet.append((nodes, time_function(value, "value")))

    def _dirichlet_values(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if not self._dirichlet:
            return np.zeros(0, np.int64), np.zeros(0)
        idx = np.concatenate([n for n, _ in self._dirichlet])
        val = np.concatenate([np.full(len(n), float(f(t))) for n, f in self._dirichlet])
        if not np.all(np.isfinite(val)) or np.any(val <= 0.0):
            raise ValueError("Dirichlet temperatures must be finite and > 0 K")
        return idx, val

    def _load(self, t: float) -> np.ndarray:
        f = np.zeros(self.n_nodes)
        for unit, fn in self._robin:
            Ti = float(fn(t))
            if not math.isfinite(Ti) or Ti <= 0.0:
                raise ValueError(f"ambient temperature must be finite and > 0 K, got {Ti!r} at t = {t}")
            f += Ti * unit
        return f

    # ------------------------------------------------------------------ residual and Jacobian
    def _terms(self, T: np.ndarray, dt: float, load: np.ndarray, jac: bool
               ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, sp.csr_matrix | None]:
        Tq = self.quadrature_temperature(T)
        Tq0 = self.quadrature_temperature(self._T)
        a1 = np.asarray(self.kinetics.advance(self._alpha, Tq0, Tq, dt), dtype=float)
        coef = self.rho * self.reaction.delta_h / dt
        S = np.bincount(self.cells.ravel(), weights=np.einsum("qa,eq->ea", self._N, coef * self._w * (a1 - self._alpha)
                                                              ).ravel(), minlength=self.n_nodes)
        inertia = self._M @ (T - self._T) / dt
        cond = self.K @ T
        R = inertia + cond - load - S
        J = None
        if jac:
            fT = np.asarray(self.kinetics.drate_dT(a1, Tq), dtype=float)
            fa = np.asarray(self.kinetics.drate_dalpha(a1, Tq), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                dadT = dt * fT / (1.0 - dt * fa)
            Ke = np.einsum("qa,qb,eq->eab", self._N, self._N, coef * self._w * dadT)
            Jsrc = self.heat.pattern.assemble(Ke)
            J = sp.csr_matrix(self._M / dt + self.K - Jsrc)
        return R, {"inertia": inertia, "cond": cond, "S": S, "load": load}, a1, J

    def residual_jacobian(self, T_trial: np.ndarray, dt: float, t_next: float) -> tuple[np.ndarray, sp.csr_matrix]:
        """Residual R (W, all nodes: at a Dirichlet node it is the heat supplied by the reservoir) and Jacobian
        J = dR/dT (W/K) at a trial temperature field for the step to ``t_next`` from the committed state. Nothing is
        modified."""
        dt = positive(dt, "dt")
        R, _, _, J = self._terms(np.asarray(T_trial, dtype=float), dt, self._load(t_next), True)
        assert J is not None
        return R, J

    # ------------------------------------------------------------------ the step
    def step(self, dt: float, rtol: float = 1e-12, max_iter: int = 25) -> ThermalStepResult:
        """Advance to t + dt by backward Euler with the monolithic Newton iteration of the module notes. Commits on
        convergence only; raises :class:`StepRejected` (nothing changed) when the iteration does not converge or leaves
        the contract of the kinetics; ``ValueError`` (nothing changed) for invalid arguments."""
        dt = positive(dt, "dt")
        rtol = positive(rtol, "rtol")
        max_iter = max_iterations(max_iter)
        t1 = self._t + dt
        load = self._load(t1)
        D, Dv = self._dirichlet_values(t1)
        free = np.setdiff1d(self._active, D)
        T = self._T.copy()
        T[D] = Dv
        Tn = self._T
        absA = abs(self.K) + abs(self._M) / dt

        def evaluate(T: np.ndarray) -> tuple[np.ndarray, float, float, tuple[dict[str, np.ndarray], np.ndarray]]:
            R, parts, a1, _ = self._terms(T, dt, load, False)
            scale = max((float(np.abs(v[free]).max()) if len(free) else 0.0) for v in parts.values())
            noise = 8.0 * _EPS * float(np.max((absA @ (np.abs(T) + np.abs(Tn)) + np.abs(parts["S"]))[free])
                                       ) if len(free) else 0.0
            return R, scale, noise, (parts, a1)

        def jacobian(T: np.ndarray) -> sp.csr_matrix:
            J = self._terms(T, dt, load, True)[3]
            assert J is not None
            return J

        try:
            r = newton_free(T, free, evaluate, jacobian, rtol, max_iter)
        except (ConstitutiveFailure, ValueError, FloatingPointError, RuntimeError) as e:
            raise StepRejected(f"step to t = {t1}: {e}") from e
        except StepRejected as e:
            raise StepRejected(f"step to t = {t1}: {e}") from e
        T, R, (parts, a1), it, rhist = r.x, r.R, r.payload, r.iterations, r.history
        rmax = rhist[-1]
        # accepted: energies, then commit
        stored = float(np.sum(self._M @ (T - Tn)))
        reaction = float(self.rho * self.reaction.delta_h * np.sum(self._w * (a1 - self._alpha)))
        robin_heat = float(dt * (load.sum() - parts["cond"].sum()))
        dirichlet_heat = float(dt * R[D].sum()) if len(D) else 0.0
        a_new = a1.copy()
        T_new = T.copy()
        a_new.setflags(write=False)
        T_new.setflags(write=False)
        self.history.append(CureSlabStep(self._t, dt, Tn, T_new))
        self._T, self._alpha, self._t = T_new, a_new, t1
        return ThermalStepResult(t1, dt, T_new, a_new, it, rmax, stored, reaction, robin_heat, dirichlet_heat,
                                 stored - reaction - robin_heat - dirichlet_heat, tuple(rhist))

    def advance_to(self, t_end: float, dt: float, rtol: float = 1e-12, max_iter: int = 25) -> list[ThermalStepResult]:
        """Uniform steps of ``dt`` up to ``t_end`` (the span must be a whole number of steps within 1e-9 dt)."""
        dt = positive(dt, "dt")
        span = float(t_end) - self._t
        n = int(round(span / dt))
        if n < 1 or abs(n * dt - span) > 1e-9 * dt:
            raise ValueError(f"t_end - t = {span} is not a positive whole number of steps of {dt}")
        return [self.step(dt, rtol, max_iter) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------- mechanics
def residual_stress(slab: CureSlab, moduli_uncured: IsotropicModuliOfT, moduli_cured: IsotropicModuliOfT,
                    expansion: ThermalExpansion, shrinkage: ChemicalShrinkage, constraint: str = "in_plane",
                    tol: float = 1e-10, max_iter: int = 25, keep_history: bool = True,
                    force_scale: float = 0.0) -> MechanicalResult:
    """One-way mechanical solve of the thermal history of ``slab`` (plane strain section of the plate).

    The accepted steps of ``slab.history`` are replayed through :class:`HistoryProblem` with
    :class:`CuringThermoElasticPoint` (the slab's kinetics, one sub-step, the slab's initial conversion, which must be
    uniform), the nodal temperatures interpolated to the quadrature points by the slab's own formula.

    ``constraint``: ``"in_plane"`` fixes u_x = 0 at every node (the plate held in-plane; plus u_y = 0 at the first node
    against rigid translation in y); ``"free"`` uses rigid-body supports only (u_x, u_y = 0 at the first node and
    u_y = 0 at the node with the same y and the largest x). For a through-thickness gradient the free strip is a
    finite block (not an infinite plate).

    ``force_scale`` (N, >= 0): :class:`HistoryProblem` stops when the free-DOF residual is below ``tol`` times
    the largest of the external force and the internal forces; for a state that is (almost) stress free those
    internal forces are at the level of rounding and the relative test cannot be met. A reference force of this size
    is applied to a SUPPORTED
    DOF (the support carries it: no effect on the stresses) so that the criterion has a physical force scale; choose it
    of the order of the internal forces of the section (stress × cross-section). Default 0: no reference force."""
    if slab.mesh.dim != 2:
        raise ValueError("residual_stress: 2-D (plane strain) meshes only")
    if constraint not in ("in_plane", "free"):
        raise ValueError(f"constraint must be 'in_plane' or 'free', got {constraint!r}")
    if not slab.history:
        raise ValueError("the slab has no accepted step")
    a0 = slab.alpha0_field
    if not np.all(a0 == a0.flat[0]):
        raise ValueError("residual_stress needs a uniform initial conversion")
    model = CuringThermoElasticPoint(slab.kinetics, moduli_uncured, moduli_cured, expansion, shrinkage,
                                     alpha0=float(a0.flat[0]), substeps=1)
    prob = HistoryProblem(slab.mesh, model, thickness=slab.thickness, plane="strain", degree=slab.degree)
    pts = slab.mesh.points
    first = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    if constraint == "in_plane":
        used = np.unique(slab.mesh.cells)
        dofs = np.unique(np.concatenate([2 * used, [2 * first + 1]]))
    else:
        row = np.flatnonzero(np.abs(pts[:, 1] - pts[first, 1]) <= 1e-9 * max(float(np.ptp(pts)), 1e-300))
        other = int(row[np.argmax(pts[row, 0])])
        if other == first:
            raise ValueError("free supports need two nodes on the lowest row")
        dofs = np.array([2 * first, 2 * first + 1, 2 * other + 1])
    values = np.zeros(len(dofs))
    if not (math.isfinite(force_scale) and force_scale >= 0.0):
        raise ValueError(f"force_scale must be finite and >= 0, got {force_scale!r}")
    f_ext = np.zeros(prob.ndof)
    f_ext[2 * first + 1] = force_scale
    hist: list[np.ndarray] = []
    its: list[int] = []
    res = None
    for h in slab.history:
        ts = TimeStep(h.t, h.dt, temperature=slab.quadrature_temperature(h.T_next).ravel(),
                      temperature_prev=slab.quadrature_temperature(h.T_prev).ravel())
        res = prob.step(ts, f_ext=f_ext, prescribed=(dofs.tolist(), values.tolist()), tol=tol, max_iter=max_iter)
        its.append(res.iterations)
        if keep_history:
            hist.append(res.stress)
    assert res is not None
    return MechanicalResult(res.stress, res.diagnostics["alpha"], res.u, hist, its, prob)

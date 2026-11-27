"""Nonlinear transient and steady heat conduction with temperature-dependent conductivity and heat capacity on a
general :class:`~toporlogy.fem.mesh.mesh.Mesh` (WP17 H04;
pre-registration ``docs/foundations/prereg/NH_nonlinear_heat.md``,
notes ``docs/foundations/NONLINEAR_HEAT.md``)::

    dh(T)/dt = div(k(T) grad T) + Q,      h(T) = ∫_{T_ref}^{T} C(T') dT',      C = ρ c_p  [J/(m^3 K)].

``k`` [W/(m K)] and ``C`` are :class:`~toporlogy.core.properties.PropertyFunction` laws of the absolute temperature
(``positive=True`` required; the validity interval is enforced by the law itself, so a temperature outside it is an
error, never a clamp). They are evaluated, with their derivatives, at the quadrature points, ``T_q = Σ_a N_a(x_q) T_a``.

* Time: backward Euler, boundary data and source at ``t_{n+1}``. **Enthalpy form (default, conservative)**: the inertia
  term is ``Σ_q w_q N_a [H(T_q^{n+1}) − H(T_q^n)]/Δt`` with the exact property integral
  ``H(T^{n+1}) − H(T^n) = C.integral(T^n, T^{n+1})``. Its sum over the nodes is the exact enthalpy change of the
  interpolated field, so the per-step energy balance closes to rounding for any Δt and the increments telescope to
  ``H(T^N) − H(T^0)``. The **capacity form** (``form="capacity"``, a negative control) uses ``C(T^{n+1}) (T^{n+1} −
  T^n)/Δt``, which does not conserve the enthalpy (the error per step is ``-½ C' ΔT²`` per volume for linear C).
* Newton: exact tangent ``J_ab = Σ_q w_q [C(T_q) N_a N_b/Δt + k(T_q) ∇N_a·∇N_b + k'(T_q) (∇N_a·∇T_q) N_b] + K_R``
  (capacity form: ``C + C'(T_q)(T_q − T_q^n)`` in place of ``C``). Converged when ``‖R_free‖_∞ ≤ rtol·scale + noise``
  (scale: largest ∞-norm over the free rows of the inertia, conduction and load terms; noise: 8 ε_mach times the
  rounding magnitude of the terms), at most ``max_iter`` iterations. No line search. A step that does not converge,
  leaves a property's validity interval, meets a non-positive k or C, or fails in the linear solve raises
  :class:`~toporlogy.fem.mesh.history.StepRejected`; (T, t) are committed only after convergence, so a rejected step
  changes nothing.
* Energy (J per unit out-of-plane thickness in 2-D, per step): ``stored`` (exact enthalpy difference), ``robin_heat`` =
  Δt(Σ f_robin − Σ K_R T), ``flux_heat``, ``source_heat``, ``dirichlet_heat`` (heat supplied by the Dirichlet
  reservoirs,
  Δt Σ R at those nodes); ``energy_residual = stored − robin − flux − source − dirichlet = Δt Σ_free R``.
* Boundaries: Dirichlet nodes (value or ``value(t)`` per node set), inward flux on facets (``q`` or ``q(t)``, spatially
  uniform), Robin facets (``h``, ``T_inf`` or ``T_inf(t)``) through the facet kernels of
  :class:`~toporlogy.fem.mesh.problem.HeatProblem`; volumetric source ``Q(x, t)`` (callable of points (n, dim) and time
  returning (n,), or a number).

Isotropic scalar k only; no phase change or latent heat (put it in C(T) as an apparent capacity only if the table
resolves it); no T-dependent density (C is the volumetric capacity of the reference configuration). Units: seconds,
kelvin, metres, J, W. All test data are synthetic."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from toporlogy.core.properties import PropertyFunction
from toporlogy.fem.mesh._newton import newton_free
from toporlogy.fem.mesh._validation import max_iterations, positive, time_function
from toporlogy.fem.mesh.history import StepRejected
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import HeatProblem

__all__ = ["NonlinearHeat", "HeatStepResult", "SteadyResult"]

_EPS = float(np.finfo(float).eps)
Scalar = float | Callable[[float], Any]
CONDUCTIVITY_UNITS = ("W/(m K)", "W/(m*K)", "W m^-1 K^-1")
CAPACITY_UNITS = ("J/(m^3 K)", "J/(m^3*K)", "J m^-3 K^-1")


@dataclass
class HeatStepResult:
    """Outcome of an accepted step. Energies in J per unit out-of-plane thickness (2-D) of the whole mesh."""

    t: float                          # end time of the step
    dt: float
    T: np.ndarray                     # nodal temperatures at t
    iterations: int
    residual: float                   # final ‖R_free‖_∞ (W)
    stored: float                     # Σ_q w_q [H(T_q^{n+1}) − H(T_q^n)]: the exact enthalpy change
    stored_scheme: float              # what the scheme's inertia term stores (= stored in the enthalpy form)
    source_heat: float                # Δt Σ f_source
    flux_heat: float                  # Δt Σ f_flux (inward positive)
    robin_heat: float                 # Δt (Σ f_robin − Σ K_R T): heat entering through the Robin facets
    dirichlet_heat: float             # Δt Σ_D R: heat supplied by the Dirichlet reservoirs
    energy_residual: float            # stored − source − flux − robin − dirichlet
    energy_scale: float               # largest absolute value among the five heat terms (J)
    residual_history: tuple[float, ...] = field(default_factory=tuple)


@dataclass
class SteadyResult:
    """Outcome of an accepted steady solve at time ``t``."""

    t: float
    T: np.ndarray
    iterations: int
    residual: float                   # final ‖R_free‖_∞ (W)
    dirichlet_heat: float             # heat per unit time supplied by the Dirichlet reservoirs (W)
    residual_history: tuple[float, ...] = field(default_factory=tuple)


def _check_property(f: Any, name: str, units: tuple[str, ...]) -> PropertyFunction:
    if not isinstance(f, PropertyFunction):
        raise ValueError(f"{name} must be a core.properties.PropertyFunction, got {type(f).__name__}")
    if not f.positive:
        raise ValueError(f"{name} must be built with positive=True (k > 0 and ρ c_p > 0 are required)")
    if f.units not in units:
        raise ValueError(f"{name}: units must be one of {units}, got {f.units!r}")
    return f


class NonlinearHeat:
    """Heat conduction with k(T) and ρ c_p(T) (module notes).

    Parameters
    ----------
    mesh : Mesh
    conductivity : PropertyFunction
        k(T), W/(m K), ``positive=True``.
    capacity : PropertyFunction
        ρ c_p(T), J/(m^3 K), ``positive=True`` (enthalpy ``h = ∫ C dT``).
    T0 : float | (n_nodes,) array
        Initial temperature in kelvin; the properties must be valid at its interpolation.
    form : {"enthalpy", "capacity"}
        Inertia term; ``"capacity"`` is the non-conservative negative control.
    thickness : float
        Out-of-plane thickness of a 2-D mesh (m); the energies refer to it.
    degree : int, optional
        Quadrature degree (default ``2 · p + 2``, p the element degree, so that k(T), C(T) polynomial of low order
        and smooth sources are integrated accurately).
    """

    def __init__(self, mesh: Mesh, conductivity: PropertyFunction, capacity: PropertyFunction, T0: Any,
                 form: str = "enthalpy", thickness: float = 1.0, degree: int | None = None) -> None:
        self.k = _check_property(conductivity, "conductivity", CONDUCTIVITY_UNITS)
        self.C = _check_property(capacity, "capacity", CAPACITY_UNITS)
        if form not in ("enthalpy", "capacity"):
            raise ValueError(f"form must be 'enthalpy' or 'capacity', got {form!r}")
        self.form = form
        self.mesh = mesh
        self.thickness = positive(thickness, "thickness")
        deg = 2 * mesh.ref.degree + 2 if degree is None else int(degree)
        # the conduction of this HeatProblem is zero (kappa = 0): it supplies the geometry, the facet kernels and
        # the (linear) Robin matrix, which is K of this object
        self.heat = HeatProblem(mesh, kappa=0.0, thickness=self.thickness, degree=deg)
        g = self.heat.geom
        self._N = g.N                                                     # (nq, nloc)
        self._dNdx = g.dNdx                                               # (ne, nq, nloc, dim)
        self._x = g.x                                                     # (ne, nq, dim)
        self._w = g.dV * (self.thickness if mesh.dim == 2 else 1.0)       # (ne, nq)
        self.cells = mesh.cells
        self.n_nodes = mesh.n_nodes
        self._active = np.unique(self.heat.cell_dofs)
        T0a = np.broadcast_to(np.asarray(T0, dtype=float), (self.n_nodes,)).copy()
        if not np.all(np.isfinite(T0a)) or np.any(T0a <= 0.0):
            raise ValueError("T0 must be finite and > 0 K")
        self._T, self._t = T0a, 0.0
        self._check_nodes(T0a)                                           # validity of T0 (ValueError)
        self._property_values(self.quadrature_temperature(T0a))
        self._T.setflags(write=False)
        self._robin: list[tuple[np.ndarray, Callable[[float], float]]] = []
        self._flux: list[tuple[np.ndarray, Callable[[float], float]]] = []
        self._dirichlet: list[tuple[np.ndarray, Callable[[float], Any]]] = []
        self._source: Callable[[np.ndarray, float], Any] | None = None

    # ------------------------------------------------------------------ state
    @property
    def t(self) -> float:
        return self._t

    @property
    def T(self) -> np.ndarray:
        """Committed nodal temperatures (read-only)."""
        return self._T

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights (ne, nq), thickness included: ∫ g dV = Σ w_q g_q."""
        return self._w

    @property
    def points(self) -> np.ndarray:
        """Quadrature points (ne, nq, dim)."""
        return self._x

    @property
    def K_robin(self) -> sp.csr_matrix:
        """Robin matrix Σ h ∫ N Nᵀ dS (W/K); the linear part of the conduction matrix."""
        return self.heat.K

    def quadrature_temperature(self, T: np.ndarray) -> np.ndarray:
        """T_q = Σ_a N_a(x_q) T_a at the quadrature points, (ne, nq)."""
        return np.einsum("qa,ea->eq", self._N, np.asarray(T, dtype=float)[self.cells])

    def enthalpy_change(self, T_from: np.ndarray, T_to: np.ndarray) -> float:
        """Exact enthalpy change ``Σ_q w_q ∫_{T_q(from)}^{T_q(to)} C dT`` (J per unit thickness) of the interpolated
        nodal fields."""
        return float(np.sum(self._w * np.asarray(self.C.integral(self.quadrature_temperature(T_from),
                                                                 self.quadrature_temperature(T_to)))))

    # ------------------------------------------------------------------ boundaries and source
    def add_dirichlet(self, nodes: np.ndarray, value: Scalar) -> None:
        """Prescribed temperature (K) on nodes: a number, or a function of time (END of each step) returning a number or
        one value per node (in the sorted order of the unique nodes)."""
        nodes = np.unique(np.asarray(nodes, np.int64).ravel())
        if len(nodes) and not np.isin(nodes, self._active).all():
            raise ValueError("Dirichlet node out of range or belonging to no cell")
        for prev, _ in self._dirichlet:
            if np.intersect1d(prev, nodes).size:
                raise ValueError("a node already carries a Dirichlet temperature")
        self._dirichlet.append((nodes, time_function(value, "value")))

    def add_robin(self, facets: np.ndarray, h: float, T_inf: Scalar) -> None:
        """Convection ``h (T − T∞(t))`` on facets, h in W/(m^2 K) > 0, T∞ in K (a number or a function of time)."""
        unit = self.heat.robin(facets, positive(h, "h"), 1.0)           # h ∫ N dS (the load for T∞ = 1)
        self._robin.append((unit, time_function(T_inf, "T_inf")))

    def add_flux(self, facets: np.ndarray, q: Scalar) -> None:
        """Inward heat flux q (W/m^2, positive into the body; a number or a function of time) on facets."""
        self._flux.append((self.heat.flux(facets, 1.0), time_function(q, "q")))

    def set_source(self, Q: Any) -> None:
        """Volumetric heat source Q (W/m^3): a number, or ``Q(x, t)`` with x of shape (n, dim) returning (n,)."""
        if callable(Q):
            self._source = Q
        else:
            c = float(Q)
            if not math.isfinite(c):
                raise ValueError(f"source must be finite, got {Q!r}")
            self._source = lambda x, t: np.full(len(x), c)

    def _dirichlet_values(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if not self._dirichlet:
            return np.zeros(0, np.int64), np.zeros(0)
        idx = np.concatenate([n for n, _ in self._dirichlet])
        val = np.concatenate([np.broadcast_to(np.asarray(f(t), dtype=float), (len(n),)) for n, f in self._dirichlet])
        if not np.all(np.isfinite(val)) or np.any(val <= 0.0):
            raise ValueError("Dirichlet temperatures must be finite and > 0 K")
        return idx, val

    def _loads(self, t: float) -> dict[str, np.ndarray]:
        robin = np.zeros(self.n_nodes)
        for unit, fn in self._robin:
            Ti = float(fn(t))
            if not math.isfinite(Ti) or Ti <= 0.0:
                raise ValueError(f"ambient temperature must be finite and > 0 K, got {Ti!r} at t = {t}")
            robin += Ti * unit
        flux = np.zeros(self.n_nodes)
        for unit, fn in self._flux:
            q = float(fn(t))
            if not math.isfinite(q):
                raise ValueError(f"flux must be finite, got {q!r} at t = {t}")
            flux += q * unit
        source: np.ndarray = np.zeros(self.n_nodes)
        if self._source is not None:
            ne, nq, dim = self._x.shape
            Q = np.asarray(self._source(self._x.reshape(-1, dim), t), dtype=float).reshape(ne, nq)
            if not np.all(np.isfinite(Q)):
                raise ValueError(f"source is not finite at t = {t}")
            source = np.asarray(np.bincount(self.cells.ravel(),
                                            weights=np.einsum("qa,eq->ea", self._N, Q * self._w).ravel(),
                                            minlength=self.n_nodes), dtype=float)
        return {"robin": robin, "flux": flux, "source": source}

    # ------------------------------------------------------------------ residual and Jacobian
    def _check_nodes(self, T: np.ndarray) -> None:
        """The validity interval of both laws must also hold at the nodal temperatures (the quadrature points of a
        linear element lie strictly inside the nodal range, so a node slightly outside would pass unseen)."""
        Ta = np.asarray(T, dtype=float)[self._active]
        self.k.value(Ta)
        self.C.value(Ta)

    def _property_values(self, Tq: np.ndarray, derivatives: bool = False) -> tuple[np.ndarray, ...]:
        """(k, C) at the points, with (k', C') if requested; ValueError outside the validity interval or if k or C is
        not finite and > 0."""
        k = np.asarray(self.k.value(Tq), dtype=float)
        C = np.asarray(self.C.value(Tq), dtype=float)
        if not (np.all(np.isfinite(k)) and np.all(k > 0.0)):
            raise ValueError(f"conductivity is not finite and > 0 (min {float(np.min(k)):.6g})")
        if not (np.all(np.isfinite(C)) and np.all(C > 0.0)):
            raise ValueError(f"heat capacity is not finite and > 0 (min {float(np.min(C)):.6g})")
        if not derivatives:
            return k, C
        return k, C, np.asarray(self.k.derivative(Tq), dtype=float), np.asarray(self.C.derivative(Tq), dtype=float)

    def _terms(self, T: np.ndarray, Tn: np.ndarray | None, dt: float | None, loads: np.ndarray, jac: bool
               ) -> tuple[np.ndarray, dict[str, np.ndarray], sp.csr_matrix | None]:
        """Residual R = inertia + conduction + K_R T − loads (W; ``dt``/``Tn`` None: steady, no inertia)."""
        self._check_nodes(T)
        Tq = self.quadrature_temperature(T)
        Te = T[self.cells]
        gradT = np.einsum("eqnj,en->eqj", self._dNdx, Te)
        k, C, dk, dC = self._property_values(Tq, True)
        wk = self._w * k
        cond_e = np.einsum("eqaj,eqj,eq->ea", self._dNdx, gradT, wk)
        cond = np.bincount(self.cells.ravel(), weights=cond_e.ravel(), minlength=self.n_nodes)
        cond_abs = np.bincount(self.cells.ravel(), weights=np.abs(cond_e).ravel(), minlength=self.n_nodes)
        absd = np.abs(self._dNdx)
        Tabs = np.abs(Te)
        mag_e = np.einsum("eab,eb->ea", np.einsum("eqai,eqbi,eq->eab", absd, absd, wk), Tabs)    # |K(T)| |T|
        KR = self.heat.K
        robin_out = KR @ T
        inertia: np.ndarray = np.zeros(self.n_nodes)
        inertia_abs: np.ndarray = np.zeros(self.n_nodes)
        Ceff = None
        if dt is not None:
            assert Tn is not None
            Tq0 = self.quadrature_temperature(Tn)
            if self.form == "enthalpy":
                dH = np.asarray(self.C.integral(Tq0, Tq), dtype=float)
                Ceff = C
            else:
                dH = C * (Tq - Tq0)
                Ceff = C + dC * (Tq - Tq0)
            Tn_abs = np.abs(Tn[self.cells])
            mag_e = mag_e + np.einsum("eab,eb->ea", np.einsum("qa,qb,eq->eab", self._N, self._N, self._w * C / dt),
                                      Tabs + Tn_abs)                                           # |M(T)/dt| (|T| + |T_n|)
            ine = np.einsum("qa,eq->ea", self._N, self._w * dH / dt)
            inertia = np.asarray(np.bincount(self.cells.ravel(), weights=ine.ravel(), minlength=self.n_nodes),
                                 dtype=float)
            inertia_abs = np.asarray(np.bincount(self.cells.ravel(), weights=np.abs(ine).ravel(),
                                                 minlength=self.n_nodes), dtype=float)
        R = inertia + cond + robin_out - loads
        J = None
        if jac:
            Ke = np.einsum("eqai,eqbi,eq->eab", self._dNdx, self._dNdx, wk)
            Ke = Ke + np.einsum("eqaj,eqj,qb,eq->eab", self._dNdx, gradT, self._N, self._w * dk)
            if Ceff is not None:
                Ke = Ke + np.einsum("qa,qb,eq->eab", self._N, self._N, self._w * Ceff / dt)
            J = sp.csr_matrix(self.heat.pattern.assemble(Ke) + KR)
        return R, {"inertia": inertia, "cond": cond, "robin_out": robin_out, "loads": loads, "inertia_abs": inertia_abs,
                   "cond_abs": cond_abs,
                   "mag": np.bincount(self.cells.ravel(), weights=mag_e.ravel(), minlength=self.n_nodes)}, J

    def residual_jacobian(self, T_trial: np.ndarray, dt: float, t_next: float) -> tuple[np.ndarray, sp.csr_matrix]:
        """Residual R (W, all nodes; at a Dirichlet node the heat supplied by the reservoir) and Jacobian J = dR/dT
        (W/K) at a trial temperature field for the step to ``t_next`` from the committed state. Nothing is modified."""
        dt = positive(dt, "dt")
        ld = self._loads(t_next)
        loads = ld["robin"] + ld["flux"] + ld["source"]
        R, _, J = self._terms(np.asarray(T_trial, dtype=float), self._T, dt, loads, True)
        assert J is not None
        return R, J

    # ------------------------------------------------------------------ Newton
    def _newton(self, T: np.ndarray, Tn: np.ndarray | None, dt: float | None, loads: np.ndarray, free: np.ndarray,
                rtol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int, list[float]]:
        absK = abs(self.heat.K)

        def evaluate(T: np.ndarray) -> tuple[np.ndarray, float, float, dict[str, np.ndarray]]:
            R, parts, _ = self._terms(T, Tn, dt, loads, False)
            scale = max((float(np.abs(v[free]).max()) if len(free) else 0.0)
                        for v in (parts["inertia"], parts["cond"] + parts["robin_out"], parts["loads"]))
            mag = parts["inertia_abs"] + parts["cond_abs"] + parts["mag"] + absK @ np.abs(T) + np.abs(loads)
            noise = 8.0 * _EPS * float(np.max(mag[free])) if len(free) else 0.0
            return R, scale, noise, parts

        def jacobian(T: np.ndarray) -> sp.csr_matrix:
            J = self._terms(T, Tn, dt, loads, True)[2]
            assert J is not None
            return J

        r = newton_free(T, free, evaluate, jacobian, rtol, max_iter)
        return r.x, r.R, r.payload, r.iterations, r.history

    @staticmethod
    def _check_args(rtol: float, max_iter: Any) -> tuple[float, int]:
        return positive(rtol, "rtol"), max_iterations(max_iter)

    # ------------------------------------------------------------------ the step
    def step(self, dt: float, rtol: float = 1e-12, max_iter: int = 25) -> HeatStepResult:
        """Advance to t + dt by backward Euler with the Newton iteration of the module notes. Commits on convergence
        only; raises :class:`StepRejected` (nothing changed) when the iteration does not converge or a property is
        invalid; ``ValueError`` (nothing changed) for invalid arguments or boundary data."""
        dt = positive(dt, "dt")
        rtol, max_iter = self._check_args(rtol, max_iter)
        t1 = self._t + dt
        ld = self._loads(t1)
        loads = ld["robin"] + ld["flux"] + ld["source"]
        D, Dv = self._dirichlet_values(t1)
        free = np.setdiff1d(self._active, D)
        T = self._T.copy()
        T[D] = Dv
        Tn = self._T
        try:
            T, R, parts, it, rhist = self._newton(T, Tn, dt, loads, free, rtol, max_iter)
            stored = float(np.sum(self._w * np.asarray(self.C.integral(self.quadrature_temperature(Tn),
                                                                       self.quadrature_temperature(T)))))
        except StepRejected as e:
            raise StepRejected(f"step to t = {t1}: {e}") from e
        except (ValueError, FloatingPointError, RuntimeError) as e:
            raise StepRejected(f"step to t = {t1}: {e}") from e
        stored_scheme = float(dt * parts["inertia"].sum())
        source_heat = float(dt * ld["source"].sum())
        flux_heat = float(dt * ld["flux"].sum())
        robin_heat = float(dt * (ld["robin"].sum() - parts["robin_out"].sum()))
        dirichlet_heat = float(dt * R[D].sum()) if len(D) else 0.0
        resid = stored - source_heat - flux_heat - robin_heat - dirichlet_heat
        scale = max(abs(stored), abs(source_heat), abs(flux_heat), abs(robin_heat), abs(dirichlet_heat))
        T_new = T.copy()
        T_new.setflags(write=False)
        self._T, self._t = T_new, t1
        return HeatStepResult(t1, dt, T_new, it, rhist[-1], stored, stored_scheme, source_heat, flux_heat, robin_heat,
                              dirichlet_heat, resid, scale, tuple(rhist))

    def advance_to(self, t_end: float, dt: float, rtol: float = 1e-12, max_iter: int = 25) -> list[HeatStepResult]:
        """Uniform steps of ``dt`` up to ``t_end`` (the span must be a whole number of steps within 1e-9 dt)."""
        dt = positive(dt, "dt")
        span = float(t_end) - self._t
        n = int(round(span / dt))
        if n < 1 or abs(n * dt - span) > 1e-9 * dt:
            raise ValueError(f"t_end - t = {span} is not a positive whole number of steps of {dt}")
        return [self.step(dt, rtol, max_iter) for _ in range(n)]

    def steady(self, rtol: float = 1e-12, max_iter: int = 25) -> SteadyResult:
        """Steady state ``div(k(T) grad T) + Q = 0`` with the boundary data and the source evaluated at the current
        time ``t`` (which is not advanced), started from the committed field. Commits T on convergence only. Needs
        at least one Dirichlet node or Robin facet (else the temperature is free up to a constant: ``ValueError``)."""
        rtol, max_iter = self._check_args(rtol, max_iter)
        if not self._dirichlet and not self._robin:
            raise ValueError("a steady problem needs a Dirichlet node or a Robin facet (else T is free up to a "
                             "constant)")
        ld = self._loads(self._t)
        loads = ld["robin"] + ld["flux"] + ld["source"]
        D, Dv = self._dirichlet_values(self._t)
        free = np.setdiff1d(self._active, D)
        T = self._T.copy()
        T[D] = Dv
        try:
            T, R, _, it, rhist = self._newton(T, None, None, loads, free, rtol, max_iter)
        except StepRejected as e:
            raise StepRejected(f"steady solve at t = {self._t}: {e}") from e
        except (ValueError, FloatingPointError, RuntimeError) as e:
            raise StepRejected(f"steady solve at t = {self._t}: {e}") from e
        T_new = T.copy()
        T_new.setflags(write=False)
        self._T = T_new
        return SteadyResult(self._t, T_new, it, rhist[-1], float(R[D].sum()) if len(D) else 0.0, tuple(rhist))

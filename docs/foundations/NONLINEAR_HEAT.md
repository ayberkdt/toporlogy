# Nonlinear heat conduction k(T), rho c_p(T) on a general mesh

Module `toporlogy.fem.mesh.nonlinear_heat` (`NonlinearHeat`), record `prereg/NH_nonlinear_heat.md`, tests
`tests/toporlogy/test_nonlinear_heat.py`. WP17 H04, WP23 H3 (nonlinear heat MMS), H5 (thermal energy balance),
H6 (smooth time convergence). All numbers in the tests are **synthetic**; none is material data.

## Model

    d h(T)/dt = div( k(T) grad T ) + Q,        h(T) = integral_{T_ref}^{T} C(T') dT',    C = rho c_p  [J/(m^3 K)]

`k` [W/(m K)] and `C` are `core.properties.PropertyFunction` laws of the absolute temperature. Both must be built with
`positive=True`; the units are checked; the validity interval of the law is enforced at the quadrature points **and** at
the nodal temperatures (a node outside the interval is an error although the interior Gauss points of a linear element
might still be inside); there is no clamp. Isotropic scalar k; no latent heat apart from an apparent capacity in C(T);
the volumetric capacity is that of the reference configuration (no T-dependent density).

Boundaries: Dirichlet nodes (number or `value(t)`, per node set), inward flux on facets (`q`, `q(t)`), Robin facets
(`h`, `T_inf(t)`; the facet kernels of `HeatProblem`), volumetric source `Q(x, t)`. All time-dependent data are
evaluated at the END of the step (backward Euler).

## Scheme and the choice of the inertia term

Weak form at the quadrature points, `T_q = sum_a N_a(x_q) T_a`:

    R_a = sum_q w_q N_a [H(T_q^{n+1}) - H(T_q^n)]/dt  +  sum_q w_q k(T_q) grad N_a . grad T_q  +  (K_R T)_a  -  f_a.

`H(T^{n+1}) - H(T^n) = C.integral(T^n, T^{n+1})` is the exact property integral (no cancellation of two large
enthalpies). **This enthalpy form is the default and the one to use.** Reason: summing the nodal equations over all
nodes, `sum_a N_a = 1` and `sum_a grad N_a = 0` leave exactly the enthalpy change of the interpolated field on the
left, so the heat balance of every step closes to rounding for any `dt` and the increments telescope,
`sum_n stored_n = int [H(T^N) - H(T^0)]`. The alternative `C(T^{n+1}) (T^{n+1} - T^n)/dt` (`form="capacity"`, kept as a
negative control) is first-order consistent but not conservative: for `C = C0 (1 + gamma (T - T_ref))` it stores
`-(C0 gamma/2) int (T^{n+1}-T^n)^2 dV` more than the enthalpy it was given (derived in the prereg, verified to 8e-10
relative in NH5; about 1 to 2 percent of the stored energy for 5 K steps and gamma = 0.005).

Newton tangent, exact, with the property derivatives: enthalpy form

    J_ab = sum_q w_q [ C(T_q) N_a N_b / dt + k(T_q) grad N_a . grad N_b + k'(T_q) (grad N_a . grad T_q) N_b ] + (K_R)_ab,

capacity form with `C + C'(T_q)(T_q - T_q^n)` in place of `C`. Checked against the central finite difference of the
residual (3e-11 relative; with `k'` removed the same check shows 1.6e-2). Convergence: `||R_free||_inf <= rtol * scale +
noise`, `rtol = 1e-12`, `scale` the largest inf-norm over the free rows of the inertia, conduction (including Robin) and
load terms, `noise = 8 eps_mach` times the gross rounding magnitude of the terms (element contributions in absolute
value, `|K(T)||T|`, `|M/dt|(|T| + |T_n|)`, loads); at most 25 iterations. There is no line search: a step that
does not converge, leaves a validity interval or meets a non-positive k or C raises `StepRejected` and (T, t) are
committed only after convergence, so a rejected step changes nothing (checked bitwise).

## Energy report

Per step (J per unit out-of-plane thickness for 2-D meshes): `stored` (exact enthalpy change), `stored_scheme` (what
the inertia term of the chosen form stores), `source_heat`, `flux_heat`, `robin_heat` (positive when the ambient is
hotter than the surface), `dirichlet_heat` (heat supplied by the Dirichlet reservoirs, `dt * R` at those nodes) and

    energy_residual = stored - source - flux - robin - dirichlet  (= dt * sum of the free residuals),   energy_scale = largest term.

Enthalpy form: `|energy_residual| / energy_scale` is at the Newton tolerance (2e-13 on the bar of NH4a; checked against
an independent balance computed from the nodal field with the test's own Gauss quadrature and enthalpy polynomial).
Capacity form: the imbalance is of order 1e-2 of the energy scale and equals the closed form above.

## Verification summary (details and measured values in the prereg record)

| ID | Claim | Result |
|---|---|---|
| NH1 | Newton Jacobian equals central FD (both forms, Q4 and T6; Pchip / piecewise-linear and polynomial laws); negative control without k' detected | pass (<= 3.4e-11; control 1.6e-2) |
| NH2 | spatial order with the time-consistent manufactured source (exact solution solves the backward-Euler scheme) | Q4 and T6 pass; T3 and Q9 least-squares windows **failed** on the coarsest level (xfail, ledger); finest-pair orders 1.95 / 1.99 / 3.00 / 2.99 (post hoc) |
| NH3 | temporal first order, continuous sympy source, Q9 16 x 16 | slope 0.978 (spatial floor 4.8 percent of the smallest time error) |
| NH4 | per-step energy balance to rounding, telescoping, all boundary kinds on a T6 mesh | pass (2.4e-13 of the energy scale) |
| NH5 | capacity form imbalance equals the closed form | pass (8e-10 relative; 1.0 to 2.0 percent of the energy scale) |
| NH6 | k, C constant: equals `HeatProblem` (steady) and a dense backward Euler from `HeatProblem.K` and `kernels.mass` (transient, regression) | pass (1e-9 window) |
| NH7 | steady 1-D k(T) against the Kirchhoff closed form (Dirichlet, flux, Robin) | prereg slope test **failed** (xfail): nodal values are exact to 6.3e-13 K at every mesh; post hoc proof and check |
| NH8 | refusals and rollback | pass |
| NH9 | Newton iterations <= 6 | pass (3 per step) |

Finding (NH7): for linear elements in 1-D, `k(T_h) T_h'` is the derivative of `Phi(T_h(x))`, so with exact quadrature
the element flux is `Phi(T_{i+1}) - Phi(T_i)` and the discrete equations force `Phi(T_i)` to be linear in `i`: the
nodal values are those of the Kirchhoff solution on any mesh (up to Newton and rounding error). The preregistered
O(h^2) expectation was wrong; the strong statement is verified post hoc.

## Limits and what is not done

* Scalar isotropic k; no anisotropic or state-dependent (cure, crystallinity) laws; no latent heat or phase change.
* Backward Euler only (first order, unconditionally stable for the linearised problem; no explicit scheme, no step-size
  control, no adaptivity). Newton without line search or damping: strongly nonlinear laws may need small steps, and
  the failure is a clean `StepRejected`.
* The Newton loop is structurally the one of `cure_slab.CureSlab` but is not shared code (that module was out of
  scope for this slice); the two property-free helpers `_positive` and `_time_function` are imported from it.
* No coupling with the mechanics (`HistoryProblem`): the temperature history of an accepted run can be fed to
  it exactly as `cure_slab.residual_stress` does. Thermomechanical feedback (dissipation, thermoelastic coupling) is
  not implemented.
* Temperature in kelvin only; Celsius is refused by the property laws.

# NH_nonlinear_heat — nonlinear transient heat conduction k(T), rho c_p(T) on a general mesh (preregistration)

Preregistered 2026-11-27 08:24 +0300 (output of `date "+%Y-%m-%d %H:%M %z"`), before any test of this slice was run. Base: `9ecc143`.
New module `src/toporlogy/fem/mesh/nonlinear_heat.py`, tests `tests/toporlogy/test_nonlinear_heat.py`, docs
`docs/foundations/NONLINEAR_HEAT.md`. Plan rows: WP17 H04 (temperature-dependent thermal FE: enthalpy form, consistent
tangent with property derivatives), 23_ACCEPTANCE_CASES H3 (nonlinear heat MMS), H5 (thermal energy source and BC heat
balance), H6 (smooth time convergence). WP11 P03 thermal properties (k(T), rho c_p(T), enthalpy). All numbers are
**synthetic**; none is material data. Units: s, K (absolute), m, J, W, kg.

## What is built (the design statements the checks test)

* Equation: `dh(T)/dt = div(k(T) grad T) + Q`, `h(T) = int_{Tref}^{T} C(T') dT'`, `C = rho c_p` [J/(m^3 K)] and
  `k` [W/(m K)] given as `core.properties.PropertyFunction` (`positive=True` required, units checked, validity interval
  enforced by the property function itself: a temperature outside it raises `ValueError`, no clamp).
* Space: any mesh of `fem.mesh.mesh`; T at the quadrature points, `T_q = sum_a N_a(x_q) T_a`; k and C (and k', C') are
  evaluated at `T_q`. Dirichlet (nodes, value(t)), inward flux (facets, q(t)), Robin (facets, h, T_inf(t)) via the
  `HeatProblem` facet kernels, volumetric source `Q(x, t)`.
* Time: backward Euler, boundary data and source at `t_{n+1}`. **Enthalpy form (default)**:
  `R_a = sum_q w_q N_a [H(T_q^{n+1}) - H(T_q^n)]/dt + sum_q w_q k(T_q) grad N_a . grad T_q + (K_R T)_a - f_a`, with the
  exact property integral `H(T^{n+1}) - H(T^n) = C.integral(T^n, T^{n+1})`. **Capacity form (negative control, option
  `form="capacity"`)**: `sum_q w_q N_a C(T_q^{n+1}) (T_q^{n+1} - T_q^n)/dt`.
* Newton tangent (exact): enthalpy form `J_ab = sum_q w_q [ C(T_q) N_a N_b/dt + k(T_q) gradN_a.gradN_b +
  k'(T_q) (gradN_a . grad T_q) N_b ] + (K_R)_ab`; capacity form replaces `C` by `C + C'(T_q)(T_q - T_q^n)`.
  Convergence: `||R_free||_inf <= rtol*scale + noise`, rtol = 1e-12, scale = largest inf-norm over the free rows of the
  inertia, conduction (incl. Robin), load terms, noise = 8 eps_mach times the rounding magnitude of the terms; at most
  25 iterations, else `StepRejected`. Commit of (T, t) only after convergence.

## Derivations (sign, reaction, normalisation)

* **Energy per step** (J per unit out-of-plane thickness in 2-D). Nodal equation at node a:
  `[inertia]_a + [cond]_a + [K_R T]_a = f_robin,a + f_flux,a + f_src,a + Q_a`, `Q_a` the heat supplied by a Dirichlet
  reservoir (`Q_a = R_a` at a Dirichlet node). Sum over all nodes: `sum_a N_a = 1` and `sum_a grad N_a = 0`, so the
  nonlinear conduction term sums to zero and the inertia term sums to `sum_q w_q [H(T_q^{n+1}) - H(T_q^n)]/dt`. Per step
  `stored := sum_q w_q [H(T_q^{n+1}) - H(T_q^n)]` (the exact enthalpy difference of the interpolated field, the
  quadrature sum of the property integral) and
  `stored - robin - flux - source - dirichlet = dt sum_free R` with `robin = dt (sum f_robin - sum K_R T)` (positive when
  the ambient is hotter than the surface), `flux = dt sum f_flux` (inward flux positive), `source = dt sum f_src`,
  `dirichlet = dt sum_D R_a`. In the enthalpy form `sum_free R = 0` to the Newton tolerance, so
  `energy_residual := stored - robin - flux - source - dirichlet` is at rounding level, every step, for every
  `dt`. Telescoping: `sum_n stored_n = sum_q w_q [H(T_q^N) - H(T_q^0)]` exactly (the enthalpy form conserves the
  total enthalpy; the sum of the per-step increments is path independent).
* **Capacity form imbalance** (closed form for linear `C(T) = C0 (1 + gamma (T - Tref))`): `H(T1) - H(T0) =
  C(T1)(T1 - T0) - (C0 gamma/2)(T1 - T0)^2`. The capacity-form scheme conserves its own stored energy
  `sum w C(T_q^{n+1}) dT_q`, so, with no Dirichlet node, `energy_residual_capacity = stored_exact - stored_scheme =
  -(C0 gamma / 2) sum_q w_q (T_q^{n+1} - T_q^n)^2`, **negative for gamma > 0** (the scheme stores more than the
  enthalpy it was given, i.e. creates energy), of order gamma dT/2 relative to the stored energy.
* **Kirchhoff transform** (steady 1-D, source-free): with `Phi(T) = int_{Tref}^{T} k dT'` and
  `k = k0 (1 + beta theta)`, `theta = T - Tref`: `Phi = k0 (theta + beta theta^2/2)`, `k T_x = const`, so `Phi(T(x))`
  is linear in x. Inversion `theta = (-1 + sqrt(1 + 2 beta Phi/k0))/beta`. Dirichlet both ends: `Phi(x) = Phi_L +
  (Phi_R - Phi_L) x/L`. Dirichlet left + inward flux q at x = L: heat flux vector `-k T_x e_x`, inward at the right end
  means `flux . n = -q` with `n = e_x`, so `k T_x = q`, `Phi(x) = Phi_L + q x`. Robin right with inward flux
  `q = h (T_inf - T_R)`: `Phi(theta_R) = Phi_L + h (T_inf - T_R) L`, a scalar equation (`scipy.optimize.brentq`).
* **Manufactured solution** (unit square, Dirichlet on all four sides from the exact solution):
  `T(x,y,t) = 300 + 20x + 10y + 15 g(t) sin(pi x) sin(2 pi y)`, `g(t) = 1 + 0.5 sin(30 t)`, `T_end = 0.1`.
  `k = k0 (1 + beta (T - 300))`, `C = C0 (1 + gamma (T - 300))`, k0 = 1, beta = 0.01, C0 = 2, gamma = 0.005,
  both `Polynomial` laws, `T_ref = 300`, validity (250, 450) K, positive. Continuous source (sympy in the test):
  `Q = C(T) T_t - d_x(k T_x) - d_y(k T_y)`. **Time-consistent source** for the pure spatial test: step n+1 gets
  `Q_n+1(x) = [H(T(x,t_{n+1})) - H(T(x,t_n))]/dt - div(k(T(x,t_{n+1})) grad T(x,t_{n+1}))` with
  `H(T) = C0 [(T-300) + gamma (T-300)^2 / 2]`; the exact solution then satisfies the backward-Euler scheme
  *exactly* at every step (the nodal values of the exact field are the only error source: initial interpolation and
  spatial discretisation), so the spatial order is measured without time error.

## Checks

Independent references: closed forms above, sympy for the source, dense `numpy` / `scipy.optimize.brentq` /
`numpy.polynomial.legendre` written in the test. Errors are measured with `kernels.geometry` quadrature points of the
mesh (infrastructure verified elsewhere) at rule degree 8, and the exact field is evaluated in the test. Checks marked
*regression* call shared kernels.

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| NH1 | The Newton Jacobian equals the central finite difference of the residual (all nodes, both forms, k = Pchip table / C = `PiecewiseLinear` table for the enthalpy form, and Polynomial k, C for the capacity form; Q4 and T6 meshes, random temperature state in the validity box, dt = 0.01) | central FD of the production residual, step h = 1e-3 K, column by column (regression of the residual, independent of the tangent code) | `max abs(J - J_fd) <= 1e-6 max abs(J)`. **Negative control**: with `k'` replaced by 0 (a subclass whose `_derivative` returns zeros) the mismatch is `>= 1e-3 max abs(J)` |
| NH2 | Spatial order (time-consistent source), `dt = 0.0125` (8 steps), relative L2 error `norm(T_h - T)/norm(T)` at `t = 0.1` over `n = 2,4,8,16` (T3, Q4) and `n = 2,4,8` (T6, Q9): least-squares slope of `log e` against `log h` | exact field | T3, Q4: slope in [1.85, 2.2]; T6, Q9: slope in [2.8, 3.3]; errors strictly decreasing |
| NH3 | Temporal first order: continuous source, Q9 mesh `n = 8`, `dt = 0.0125, 0.00625, 0.003125, 0.0015625`, same error norm at `t = 0.1`: slope of `log e` against `log dt` | exact field | slope in [0.9, 1.15]; and the spatial floor of the same mesh (NH2 Q9 `n = 8`) is `<= 0.1` times the error at the smallest `dt` (else the slope is not a time slope) |
| NH4 | Energy balance, enthalpy form, to rounding. (a) 1-D bar: `rectangle(8,1,1.0,0.1)` Q4, flux at x = 0 (`q = 50 (1+t)`), Robin at x = 1 (`h = 20`, `T_inf = 320 + 10 t`), source `Q = 100 (1+x^2)(1+t)`, no Dirichlet node; T0 = 300; k, C as above; 6 steps `dt = 0.05`; independent balance `stored_ind - dt w [q + h (T_inf - T_R)] - dt w int Q dx` with `w = 0.1`, `stored_ind` from the test's own Gauss quadrature of the nodal field and its own `H`, `T_R` the nodal temperature at x = 1. (b) 2-D, T6 mesh with all four boundary kinds, Pchip k, PiecewiseLinear C, 5 steps: the reported `energy_residual`. (c) telescoping: `sum stored_n` against `sum_q w [H(T^N) - H(T^0)]` of the test | the test's own sums | (a), (b): `abs(imbalance) <= 1e-9 energy_scale` at every step (`energy_scale` = largest heat term of the step); (a) also the reported `stored` agrees with `stored_ind` to 1e-11 relative; (c) 1e-11 relative |
| NH5 | Capacity form negative control, same problem as NH4(a) with `form="capacity"`: the independent imbalance equals `-(C0 gamma/2) int w (T^{n+1} - T^n)^2` (test's own quadrature) | closed form above | agreement to 1e-6 relative of the predicted value; and `abs(imbalance) >= 1e-3 energy_scale` at every step (so the control is not vacuous) |
| NH6 | Linear limit: constant `k`, `C` (`Constant` laws): steady problem equals `HeatProblem.solve` (Dirichlet + Robin + body source, Q4 and T3, `rectangle(6,4)`), transient equals dense backward Euler `(M/dt + K) T1 = M/dt T0 + f` built from `HeatProblem.K` and `kernels.mass` (regression) | `HeatProblem` | `max abs(T - T_ref) <= 1e-9 (1 + max abs(T_ref))` for the steady solution and for the state after 5 steps |
| NH7 | Steady 1-D with k(T) against the Kirchhoff closed form: `rectangle(nx,1,1,0.1)` Q4, `nx = 4,8,16,32`, k0 = 1, beta = 0.01, T_L = 300; (a) T_R = 400 Dirichlet, (b) flux q = 100 at x = 1, (c) Robin h = 1, T_inf = 420 at x = 1 | closed forms (brentq in (c)) | max nodal error: slope over the four levels `>= 1.7`; error at `nx = 32` `<= 0.1 K` (= 1e-3 of the 100 K range); Newton iterations of the steady solve `<= 10` |
| NH8 | Refusals and rollback. (a) property without `positive=True`, wrong units, a non-PropertyFunction: `ValueError`; (b) T0 outside the validity interval: `ValueError`; (c) a Dirichlet value outside the validity interval at step time: `StepRejected`, and `T`, `t` unchanged (equal arrays); a later valid step after a rejection equals, bitwise, the same step of a solver that never saw the failure; (d) `max_iter = 1` on a nonlinear step: `StepRejected`, state unchanged; (e) a property law that returns a non-positive value at run time: `StepRejected`; (f) invalid arguments (dt <= 0, nan, Dirichlet T <= 0, steady problem with no Dirichlet node and no Robin facet): `ValueError` | exceptions and state comparison | exact |
| NH9 | Newton efficiency (consequence of the exact tangent): at every step of the NH2 problem Q4 `n = 8`, iterations `<= 6` | counts | strict |

Expected failure modes: NH2 at the quadratic elements could be limited by the initial interpolation (it is of the
same order `h^{p+1}`, so it does not change the slope); NH7 slope could exceed 2 by nodal superconvergence (the
threshold is a lower bound).

Run: `python -m pytest tests/toporlogy/test_nonlinear_heat.py -q -p no:randomly` with `TMP`/`TEMP` and `--basetemp`
under `D:/Masaustu/Topology Optimization/.claude/tmp-toporlogy/NH`. Largest case: Q9/T6 mesh 8 x 8 and Q4 16 x 16,
8 Newton steps each; seconds.

First result (append after run): see below.

# Temperature-dependent properties and the thermoelastic material point

Module `toporlogy.core.properties`. Pre-registration and first results: `prereg/P_properties.md`; tests:
`tests/toporlogy/test_properties.py` (PP1-PP17). Every number in the tests and in the examples below is
**synthetic**; none is material data.

## Conventions

* Temperature is absolute, in kelvin. A temperature argument is never a Celsius value and never a difference. A
  difference ΔT = T − T_ref is in kelvin as well, and only appears inside the integrals (thermal strain, enthalpy,
  polynomial in T − T_ref). Degrees Celsius are refused as `temperature_units` (an absolute 20 °C and a 20 K difference
  must not be confused); the constructor accepts only `"K"`.
* Evaluation needs T finite, T > 0 and inside the validity interval of the property. The extrapolation policy is
  `"error"` and is the only one implemented: outside the interval `ValueError`, no clamp, no linear extension.
* Units are documented strings (`units`, `source`; `source="synthetic"` for invented numbers). They are not parsed
  dimensionally: the existing machinery `configuration/units.py` is a config-value parser (length, mass, time, force,
  pressure, K / delta_degC / degC); it knows neither J, W, Pa·s, 1/s nor parenthesised units such as `J/(kg K)`, and
  extending it was not a clean small change. The consumers check the string against a short accepted set: moduli
  `Pa | kPa | MPa | GPa` (the inputs of one `IsotropicModuliOfT` must share one unit), ν `1 | - | dimensionless`,
  CTE `1/K | K^-1`, c_p `J/(kg K) | J/(kg*K) | J kg^-1 K^-1`. Stress and stiffness come out in the unit of the moduli.
  The unit-checker extension of WP11 P05 (Pa·s, 1/s, activation energy J/mol, W/(m K), absolute vs differential K)
  is NOT done.
* Voigt order [xx, yy, zz, yz, xz, xy]; stress physical shear, strain engineering shear; tension positive.

## Property functions

| class | law | derivative | integral | smoothness |
|---|---|---|---|---|
| `Constant(c, units, ...)` | c | 0 | c (T1 − T0) | C-infinity |
| `PiecewiseLinear(T, y, ...)` | linear table, knots strictly increasing | segment slope; at an interior knot the right (default) or left (`side=`) slope; `nonsmooth(T)` is True there | exact (sum of trapezoids) | C0 |
| `Pchip(T, y, ...)` | scipy `PchipInterpolator`; monotone on every interval where the data are monotone, never leaves [min y, max y] | analytic | exact antiderivative of the cubics | C1 (second derivative jumps at knots) |
| `Polynomial(c, T_ref, domain, ...)` | Σ c_k (T − T_ref)^k | exact | exact | C-infinity |

Common: `value(T)` (also `f(T)`), `derivative(T)`, `integral(T0, T1)` (negative for T1 < T0), `domain` (validity
interval; the table laws use their first and last knot, `Constant` defaults to every T > 0, `Polynomial` requires a
finite one), `units`, `source`, `positive=True` (rejects a law that is not strictly positive on its domain: exact for the
tables, Pchip and polynomials; end points and real critical points). Scalars and arrays of any shape are accepted.

At a discontinuity of the physical property (a phase transition) use two laws on two domains; this slice does not
implement a combined piecewise-in-domain law.

## Isotropic moduli of temperature

`IsotropicModuliOfT(E=f, nu=g)` takes **exactly two** functions among E, nu, K, G (any of the six pairs). The other
moduli follow from K(T), G(T) (WP04 E01): E = 9KG/(3K+G), ν = (3K − 2G)/(2(3K+G)), λ = K − 2G/3, and the derivatives by
the chain rule through the pair-to-(K, G) conversion and ∂E/∂K = 9G²/(3K+G)², ∂E/∂G = 27K²/(3K+G)²,
∂ν/∂K = 9G/(2(3K+G)²), ∂ν/∂G = −9K/(2(3K+G)²). API: `moduli(T)`, `dmoduli(T)` (E, nu, K, G, lam),
`E/nu/K/G/lam(T)`, `value(name, T)`, `derivative(name, T)`, `stiffness(T)` = K 1⊗1 + 2G I_dev (…, 6, 6) and
`dstiffness(T)`.

Refused: three or four moduli (redundant data are not averaged; a consistency check against measurement uncertainty is
not implemented), `lam`, `M`, `mu` (use G), non-function inputs, non-overlapping validity intervals, mixed or unknown
units, and any temperature where K ≤ 0 or G ≤ 0 (also E ≤ 0 or ν outside (−1, 1/2) for the inputs that carry them; ν = 1/2
and ν = −1 are limits, not inputs; near-incompressibility needs a mixed model, K is never capped). The constructor samples
257 temperatures of the common interval (a sampled, not certified, check); every evaluation checks exactly the
temperatures asked.

## Thermal strain, enthalpy

`ThermalExpansion(alpha, T_ref)`: ε_th(T) = ∫_{T_ref}^{T} α dT with α the **instantaneous** (differential) CTE.
`strain(T)`, `alpha(T)`, `secant(T)` = ε_th/(T − T_ref) (limit α(T_ref) at T = T_ref, used within 1e-9 K of it).
`ThermalExpansion.from_secant(alpha_sec, T_ref)` takes a secant CTE referred to the same T_ref: ε_th = α_sec ΔT and
α = α_sec + ΔT dα_sec/dT. For α = a + b ΔT: ε_th = aΔT + ½bΔT², α_sec = a + ½bΔT; using α_sec as if instantaneous gives
aΔT + ¼bΔT² (negative control PP9: −3.1 % at ΔT = 80 K, b = 2e-8 1/K²). `T_ref` is the stress-free temperature, it
must lie in the CTE validity interval. `ThermalExpansionTensor({"xx": f, ...}, T_ref, rotation=R)`: symmetric 3×3
ε_th from per-component instantaneous CTEs in the material axes (missing components are zero), `rotation` = proper
orthogonal matrix whose columns are the material axes in global coordinates (ε = R ε_mat Rᵀ). `Enthalpy(cp, T_ref).H(T)`
= ∫ c_p dT in J/kg for a c_p declared `positive=True`; latent heat is not included.

## ThermoElasticPoint

`ThermoElasticPoint(moduli, alpha, T_ref)`: σ = C(T) : (ε − ε_th(T) 1) with C(T) = K(T) 1⊗1 + 2G(T) I_dev. History
contract (`physical = True`, `kinematics = "small"`, `conservative = False`, no state): the temperature comes from
`MaterialRequest.step` (`require_step(..., temperature=True)`; scalar or per point (n,)); dt is not used. Tangent =
C(T) at the step temperature (exact for the discrete update). Energy and dissipation are None; the diagnostic
`elastic_strain_energy` = ½ e:σ with e = ε − ε_th is not a Helmholtz free energy (the T dependence of C is not
accounted for). The temperature is prescribed: this is a one-way coupling, not a two-way thermodynamic model (no
entropy or heat terms).

Verified (see the prereg for measured values): free expansion gives zero stress (E(T) alone produces none); a bar fully
constrained axially with free lateral faces gives σ = −E(T) ε_th, ε_lateral = (1+ν) ε_th; 3-D full constraint gives
σ = −3K(T) ε_th 1; closed-form moduli, all six pairs against mpmath, PCHIP/table/polynomial integrals against `quad`.

## Not implemented

* Finite thermal strain (a multiplicative thermal stretch F = F_e F_th is not derived or implemented; additive thermal
  strain is a small-strain statement only).
* Anisotropic C(T) and its interpolation (convex combination, log-Euclidean or Cholesky); only isotropic moduli of T.
* Dependence of a property on time, state (cure, crystallinity, age) or rate; only T.
* Log or positive-preserving interpolation of tabulated quantities; thermal conductivity k(T), density ρ(T) (and a
  mass-conservation model), latent heat, coupled thermomechanics.
* Fitting or uncertainty checks of redundant moduli data; the dimensional unit checker for J, W, Pa·s, 1/s, J/mol.
* Material-card schema/YAML and FEM hookup of the thermoelastic point (WP11 P05 / WP17); `NonlinearElasticity`
  refuses `physical = True` models.

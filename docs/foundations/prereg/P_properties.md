# P_properties — temperature-dependent property functions, thermal strain, thermoelastic point (preregistration)

Preregistered 2026-11-27 01:27 +0300 (output of `date "+%Y-%m-%d %H:%M %z"`), before any test of this slice was
run. Base SHA `25d8df6`. Modules: `core/properties.py` (new), tests `tests/toporlogy/test_properties.py`, docs
`docs/foundations/TEMPERATURE_PROPERTIES.md`. Plan rows: WP11 P03/P04/P05/P06 and acceptance P3, P4, P6 (WP23);
moduli formulas E01 (WP04). Everything here uses **synthetic** numbers; none is material data.

Units machinery inspected: `configuration/units.py` is a dimensional-analysis parser (length, mass, time, force,
pressure, K / delta_degC / degC) for config values; it has no J, W, Pa·s, 1/s, mol, and no parenthesised units
("J/(kg K)"). Extending it cleanly is not a small change, so **units stay documented strings** in the property
objects (non-empty, checked only for the temperature axis: kelvin only, Celsius spellings rejected). The unit
checker extension of WP11 P05 is not done in this slice.

Conventions: temperature absolute in kelvin; ΔT in kelvin; tension positive; Voigt [xx, yy, zz, yz, xz, xy], stress
physical shear, strain engineering shear.

References (independent of production code): closed forms derived below; `scipy.integrate.quad` (epsrel 1e-13) on
the production `value`; central finite differences written in the test (step h and tolerance given per check);
dense numpy 6x6 stiffness written in the test; mpmath 1.3.0 at 50 digits for derivative chains of the moduli pairs.

| ID | Hypothesis (domain) | Reference | Metric / threshold |
|---|---|---|---|
| PP1 | Synthetic E(T) = 3e9 − 1e7 (T − 300) Pa on [280, 350] K, ν = 0.35: K = E/(3(1−2ν)) = E/0.9, G = E/(2(1+ν)) = E/2.7, λ = K − 2G/3, and dK/dT = −1e7/0.9, dG/dT = −1e7/2.7, dE/dT = −1e7, dν/dT = 0 | closed forms; central FD (h = 1e-2 K) of the test's own E→K,G conversion | values rel ≤ 1e-13; derivatives rel ≤ 1e-12 vs closed form and ≤ 1e-7 vs FD |
| PP2 | Endpoints T = 280 and 350 K evaluate exactly (E = 3.2e9, 2.5e9); T just outside (279.999, 350.001), NaN, ±inf, 0, negative raise ValueError; batch (n,) input gives (n,) output | closed form; exact | rel ≤ 1e-13; exact raise |
| PP3 | All six pairs (E,ν), (E,K), (E,G), (K,G), (K,ν), (G,ν) with smooth non-linear synthetic functions (quadratic polynomials) return E, ν, K, G, λ and their dT-derivatives by the chain rule | mpmath at 50 digits: the test converts the pair to K, G with its own formulas and differentiates with `mp.diff` | rel ≤ 1e-10 (values and derivatives) |
| PP4 | The given pair is reproduced (round trip): the input E(T), ν(T) are returned unchanged by `E(T)`, `nu(T)` | the input functions | rel ≤ 1e-13 |
| PP5 | `PiecewiseLinear`: exact at knots (equality), linear between (closed form), derivative = slope in the interior, documented one-sided value at knots (`side`), integral exact | closed-form trapezoid sums written in the test; slopes | equality at knots; rel ≤ 1e-13 |
| PP6 | `Pchip` of monotone data is monotone on 2001 points; passes through knots; stays within [min y, max y]; integral equals `quad` of the value; derivative matches FD | monotone check on the test grid; `quad` epsrel 1e-13; FD h = 1e-6 K | monotone: min diff ≥ −1e-12 · range; integral rel ≤ 1e-10; derivative rel ≤ 1e-6 |
| PP7 | `Polynomial` in (T − T_ref): value, derivative, integral exact | Horner by hand in the test; integral closed form and `quad` | rel ≤ 1e-13 |
| PP8 | Thermal strain with α(T) = a + b (T − T_ref), a = 1.2e-5 1/K, b = 2e-8 1/K²: ε_th = aΔT + ½bΔT² for ΔT in [−60, +80] K (heating and cooling); secant α_sec = ε_th/ΔT = a + ½bΔT, limit at T = T_ref equals a (value a, finite); `from_secant` round trip returns the same ε_th and α | closed form | rel ≤ 1e-13 |
| PP9 | **Negative control**: using α_sec(T) as if it were instantaneous in the integral changes ε_th by ¼bΔT² (b ≠ 0); the test measures that this wrong result differs from the closed form by more than 1e-3 relative at ΔT = 80 K, and that b = 0 gives no difference | closed form | wrong/right − 1 > 1e-3 at b ≠ 0; ≤ 1e-14 at b = 0 |
| PP10 | Enthalpy H(T) = ∫ c_p dT for linear c_p, `PiecewiseLinear` c_p and `Pchip` c_p equals closed form / `quad`; non-positive c_p rejected | closed form; `quad` | rel ≤ 1e-10 |
| PP11 | Tensor CTE: symmetric 3x3 ε_th(T) = ∫ α_ij dT from per-component functions; rotation of the material axes by 90 degrees about z swaps xx and yy; a rotated tensor is symmetric | closed form; direct R ε Rᵀ in the test | rel ≤ 1e-13 |
| PP12 | `ThermoElasticPoint` free expansion (all stress components controlled to 0, T history from T_ref through a heating ramp): stress = 0 and ε = ε_th(T) I, with E(T) varying 3.2e9 → 2.5e9 Pa (E alone produces no stress) | closed form ε_th | stress abs ≤ 1e-5 Pa (driver tol 1e-6 Pa); strain rel ≤ 1e-12 |
| PP13 | Uniaxial bar, ε_xx = 0 controlled, other stresses 0: σ_xx = −E(T) ε_th(T), σ_yy = σ_zz = 0, ε_yy = ε_zz = (1+ν) ε_th | closed form | σ_xx rel ≤ 1e-9; strains rel ≤ 1e-9 |
| PP14 | 3-D fully constrained (all strains 0): σ = −3K(T) ε_th(T) I, shear 0 | closed form | rel ≤ 1e-12 |
| PP15 | Tangent returned equals the test's dense K 1⊗1 + 2G I_dev at the evaluation T, and equals the FD derivative dσ/dε at fixed T | dense numpy in the test; FD h = 1e-7 | rel ≤ 1e-12 vs dense; ≤ 1e-6 vs FD |
| PP16 | Contract: `physical = True`, `kinematics = "small"`, state empty; no step or a step without temperature raises ValueError; temperature outside the property domain raises; energy and dissipation are None | exact | exact |
| PP17 | P6 failures are explicit ValueError: non-strictly-increasing / duplicate / NaN knots, length mismatch, < 2 knots, non-positive value with `positive=True` (table, Pchip, polynomial dipping negative inside the domain), extrapolation policy other than "error", empty or non-string units, Celsius temperature units, inverted or non-positive domain; moduli: three functions, one function, unsupported keys (lam, M, mu), (E,G) with E ≥ 3G at some T (K ≤ 0), ν reaching 0.5 or −1, E ≤ 0 | exact | exact |

Expected limiting behaviour: PP9 shows the instantaneous/secant mix-up is detectable; PP12 shows E(T) alone does not
produce stress. Failure mode to watch: ε_th used with the wrong sign or secant CTE integrated twice; K, G admissibility
checked only at sampled temperatures (constructor samples 257 points; evaluation checks exactly at every T asked).

Run: `python -m pytest tests/toporlogy/test_properties.py -q -p no:randomly -p no:cacheprovider` (TMP/TEMP and basetemp
under `.claude/tmp-toporlogy/P_properties`). Resource cap: seconds, small arrays.

First result (append after run): see below.

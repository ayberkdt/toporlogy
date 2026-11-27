# Fractional viscoelasticity at the material point

Status: WP12 V05 slice (not required for P0). Code: `src/toporlogy/core/fractional.py`. Tests:
`tests/toporlogy/test_fractional.py`. Preregistration and first results: `prereg/VF_fractional.md`; ledger entry:
`ledger_additions/VF_fractional.json`. Conventions as `RHEOLOGY.md` (SI, e^{iωt}, ω in rad/s, G(t) is not 1/J(t)). All
numbers are **synthetic**; no real polymer is claimed to be validated. Scope: linear, isothermal, small strain, one shear
kernel at a material point. **Not here:** a fractional constitutive model in the history contract (no
`update/evaluate`), temperature shift of the fractional kernel, bulk fractional kernel, fractional FE. The bridge to the
existing 3-D model is the Prony approximation below, which is a `GeneralizedMaxwell`.

## Models

Springpot of order α ∈ (0, 1]: operational modulus c s^α (Caputo derivative), relaxation c t^{-α}/Γ(1−α); c = E τ^α.
With u = t/τ, E_α = E_{α,1}, G0 = G∞ + G1, λ = G∞/G0 (derivations in the preregistration, from the Laplace pair
L{t^{β−1}E_{α,β}(−μt^α)} = s^{α−β}/(s^α+μ), Podlubny 1999, written from memory and **checked** by a numerical inverse
Laplace transform, VF5/VF6):

| model | G*(s) | relaxation G(t) | creep J(t) |
|---|---|---|---|
| fractional Maxwell (E, τ, α) | E x/(1+x), x = (sτ)^α | E E_α(−u^α) | (1/E)(1 + u^α/Γ(1+α)) |
| fractional Kelvin–Voigt | E(1 + x) | E(1 + u^{-α}/Γ(1−α)), α < 1, t > 0; α = 1 refused (impulse) | (1/E) u^α E_{α,α+1}(−u^α) = (1/E)(1 − E_α(−u^α)) |
| fractional Zener (G∞, G1, τ, α) | G∞ + G1 x/(1+x) | G∞ + G1 E_α(−u^α) | 1/G0 + (G1/G0²) u^α E_{α,α+1}(−λu^α) |

α = 1 gives Maxwell, Kelvin–Voigt and the standard linear solid of `core.viscoelastic` (VF7, relative 1e-13). Ramp-hold
superposition uses Φ(t) = ∫₀^t G = G∞ t + G1 t E_{α,2}(−u^α).

Complex modulus (`fractional_zener_modulus`): with u_ω = (ωτ)^α, c = cos(πα/2), s = sin(πα/2),
G'' = G1 u_ω s/(1 + 2u_ω c + u_ω²) ≥ 0 (passive: the denominator is ≥ 1 because c ≥ 0), G' = G∞ + G1(u_ω c + u_ω²)/(…).
The loss peak is at **ωτ = 1 for every α** with G''_max = (G1/2) tan(πα/4); G'' is symmetric under ωτ → 1/(ωτ); low/high
frequency slopes ±α in log–log. The evaluation switches to u → 1/u for u > 1 (no overflow). Checked in VF8: G'' ≥ 0 on
[1e-8, 1e8]/τ for α ∈ {0.3, 0.5, 0.8, 0.99, 1} against an independent numpy/mpmath complex-power evaluation, peak value,
peak location (2e-3 on a grid), symmetry, slopes, plateau moduli, and tan δ → tan(πα/2) for the fractional Maxwell.

## Mittag-Leffler function

`mittag_leffler(z, alpha, beta=1)`: real z ≤ 0, 0 < α ≤ 1, β > 0 (positive z is refused, α = 1 only for integer β).

* x^{1/α} ≤ 5: the power series (largest term ≈ e^{x^{1/α}} ≤ 150, so ≤ 2 digits are lost).
* otherwise the Hankel contour deformed onto the negative real axis (poles at |arg t| = π/α > π are off the principal
  sheet for α < 1):
  E_{α,β}(−x) = (1/(πα)) ∫₀^∞ e^{-u^{1/α}} u^{(1−β)/α} [u sin πβ + x sin π(β−α)] / (u² + 2xu cos πα + x²) du, β < α+1.
  β = 1 gives a smooth bounded integrand. β ≥ α+1 by E_{α,β+α}(z) = (E_{α,β}(z) − 1/Γ(β))/z from β' ∈ (1−α, 1] (only
  for x > 1: no cancellation). Integration by `scipy.integrate.quad`, relative 1e-12, split at u = x, cut at
  u^{1/α} = 45.
* x ≥ 1e8: asymptotic expansion −Σ z^{-k}/Γ(β−αk), six terms (**outside the validated range**, not exercised by the
  tests).

Measured (VF1–VF3, first run, reference mpmath ≥ 30 digits: α = 1 exp, α = 1/2 e^{x²}erfc x, series / asymptotic series /
`mp.quad` of the same integral elsewhere, itself cross-validated in VF4 against the series, the asymptotic expansion and
`invertlaplace`) on x ∈ [0, 1000], 24 points: worst relative error 1.2e-13 (α = 0.3), 2.6e-14 (0.5), 1.5e-13 (0.8), 0
(1.0); stress α = 0.1, 0.95, 0.99: 2.9e-15, 3.3e-13, 2.7e-13; E_{α,β} for β ∈ {α, 1.5, 1+α, 2, 2.7}, α ∈ {0.3, 0.5,
0.8}: worst 3.4e-12. Thresholds were 1e-10, 1e-9, 1e-8. Not validated: α < 0.1, x > 1e3 (except by construction),
non-integer β at α = 1 (refused).

## Memory approximation: Prony bridge and error budget

A power-law memory needs many exponentials. `prony_error_budget(G∞, G1, τ, α, [N…])` fits non-negative Prony series
G∞' + Σ g_i e^{-t/τ_i} (g_i ≥ 0) to the fractional Zener relaxation modulus over a window (default
[1e-4 τ, 1e4 τ], i.e. 8 decades): τ_i log-spaced on [t_lo, 10 t_hi] (nested for N − 1 = 4, 8, 16, …), rows at 801 log-spaced
times weighted by 1/G(t) (relative error), `scipy.optimize.nnls`. The result (`FractionalPronyFit`) carries the
`PronySeries`, `rel_error_time` e_t = max |G_N − G|/G over 4001 other log-spaced times of the window, and
`rel_error_modulus` e_ω = max |G*_N − G*|/|G*| over ω ∈ [10/t_hi, 0.1/t_lo]. `to_generalized_maxwell(bulk)` returns the 3-D
`GeneralizedMaxwell` (the fitted series is the shear kernel). **Nothing is claimed outside the window**: beyond t_hi the
Prony relaxation goes to a constant while the fractional kernel keeps decaying as t^{-α}.

Error budget, synthetic G∞ = 2e5, G1 = 8e5 Pa, τ = 1 s, window [1e-4, 1e4] s (first run, 2026-11-27):

| α | N = 5 | 9 | 17 | 33 | 65 |
|---|---|---|---|---|---|
| 0.3, e_t | 1.07e-1 | 1.08e-2 | 1.88e-4 | 8.89e-5 | 8.98e-5 |
| 0.3, e_ω | 1.25e-1 | 1.76e-2 | 3.95e-4 | 3.56e-4 | 3.38e-4 |
| 0.5, e_t | 1.39e-1 | 2.56e-2 | 5.39e-4 | 1.46e-5 | 1.38e-5 |
| 0.5, e_ω | 1.76e-1 | 3.59e-2 | 9.26e-4 | 5.14e-5 | 4.63e-5 |
| 0.8, e_t | 2.75e-1 | 1.17e-1 | 4.98e-3 | 1.91e-5 | 3.50e-7 |
| 0.8, e_ω | 2.74e-1 | 1.34e-1 | 5.55e-3 | 2.34e-5 | 1.08e-6 |

Reading: error drops by roughly two decades between 9 and 17 branches (9 decades of τ, i.e. 2 branches per decade) and
reaches 1e-4–1e-5 at 4 branches per decade, then **stalls**. The preregistered claim "strictly decreasing in N" held for
α = 0.5 and 0.8 and **failed for α = 0.3** (8.89e-5 → 8.98e-5, +1 %): the maximum of the error sits at the short-time
window edge t = t_lo, which is also the smallest branch time, and the NNLS optimum minimises the weighted L2 residual
(which does decrease, 2.78e-4 → 2.62e-4, one diagnostic run), not the max norm. The test is kept as `xfail(strict)`;
extending τ_i below t_lo is the obvious (not tested) remedy and would be a separate preregistration. The thresholds
e_t(17) ≤ 5e-2, e_t(33) ≤ 5e-3, e_t(65) ≤ 5e-4 and e_ω(65) ≤ 5e-2 held for all three orders.

**Time-domain use (VF10).** A ramp-hold shear drive (γ0 = 1e-3 in 1 ms, hold to 1e3 s, α = 0.5) of the fitted
`GeneralizedMaxwell` (exact piecewise-linear update of `core.viscoelastic`) against the independent hereditary integral
σ = (γ0/t_r)[Φ(t) − Φ(t − t_r)] (Φ from the mpmath E_{α,2}): the stress error stays within γ0 e_t(N) G(t − t_r) (the bound
that follows from e_t, since the arguments of G stay inside the window); measured largest ratio error/bound 0.97 (N = 17),
0.079 (N = 33), 0.031 (N = 65), largest absolute stress error 0.20, 1.1e-3 and 3.9e-4 Pa on stresses of order 10²–10³ Pa.
So e_t is a usable error budget for a time-domain simulation, but only for arguments of G inside the window; a drive whose
histories reach back beyond t_hi (or load steps shorter than t_lo) is outside it.

## Conclusions and limits

* The Mittag-Leffler function, the three fractional models, creep/relaxation closed forms and the complex modulus agree with
  independent references to ≈ 1e-12 and better; the checks above are the evidence, nothing more.
* The fractional kernel is causal and passive (G'' ≥ 0, G(t) completely monotone for α ≤ 1); the Prony fit keeps that
  structure (non-negative amplitudes), so the resulting `GeneralizedMaxwell` is thermodynamically admissible by
  construction of `PronySeries`.
* A real use needs fitted (G∞, G1, τ, α) from data and a window that covers the load history; the fractional order α
  and the window decide N (about 4 branches per decade for e_t ≈ 1e-4–1e-5 here). No real material is validated.

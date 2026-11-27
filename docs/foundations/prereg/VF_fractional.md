# VF — Fractional viscoelasticity at the material point (preregistration)

Preregistered 2026-11-27 06:00 +0300 (pasted from `date "+%Y-%m-%d %H:%M %z"` in the command that stamped this file), before
any numerical test of this slice was executed. Base SHA `1cd3a8f` (branch `worktree-agent-aa15946aaed9ed84e`). Module
`src/toporlogy/core/fractional.py` (new), tests `tests/toporlogy/test_fractional.py`, doc
`docs/foundations/FRACTIONAL_VISCOELASTICITY.md`. Plan: `12_VISCOELASTICITY_DMA.md` V05 ("Fractional viscoelasticity:
Mittag-Leffler exact benchmark, causal/passive kernel, memory approximation and error budget; not required for P0").
All parameters are **synthetic**; nothing here is data of a real polymer. Conventions as in `core/viscoelastic.py`: SI
(Pa, s), e^{iωt}, ω in rad/s, relaxation modulus G(t) (stress under γ0 H(t)) distinct from creep compliance J(t) (strain
under σ0 H(t)), t ≥ 0.

## Models (derivations; sign and normalisation fixed here, not assumed)

Springpot (Scott-Blair element) of order α ∈ (0, 1]: σ = c D^α ε (Caputo), operational modulus c s^α, relaxation
c t^{-α}/Γ(1−α). Three elements, written with a relaxation time τ (so that c = E τ^α):

* **Fractional Maxwell** (spring E in series with a springpot): G*(s) = E (sτ)^α/(1+(sτ)^α).
* **Fractional Kelvin–Voigt** (spring E parallel to a springpot): G*(s) = E (1 + (sτ)^α).
* **Fractional Zener** (spring G∞ parallel to a fractional Maxwell branch G1, τ, α):
  G*(s) = G∞ + G1 (sτ)^α/(1+(sτ)^α), G0 = G∞ + G1. The frequency response is s = iω (e^{iωt}). α = 1 is the standard
  linear solid of `sls_*`.

Operational modulus and Laplace transforms (zero initial state, causal): Ĝ(s) = G*(s)/s, Ĵ(s) = 1/(s G*(s))
(so that (s Ĝ)(s Ĵ) = 1).

Laplace pair (Podlubny 1999, eq. 1.80, **from memory; checked independently by the numerical inverse Laplace
transform below**): L{t^{β−1} E_{α,β}(−μ t^α)} = s^{α−β}/(s^α + μ), Re s > μ^{1/α}.
Mittag-Leffler function E_{α,β}(z) = Σ_{k≥0} z^k / Γ(αk+β), E_{α,1} = E_α. Identity E_{α,1}(z) = 1 + z E_{α,α+1}(z).

Closed forms derived from that pair (u = t/τ, λ = G∞/G0):

1. Fractional Maxwell: Ĝ = E s^{α−1}/(s^α + τ^{-α}) ⇒ **G(t) = E E_α(−u^α)** (fluid, G(∞) = 0).
   Ĵ = (1+(sτ)^α)/(E s (sτ)^α) = (1/E)(1/s + τ^{-α} s^{-α-1}) ⇒ **J(t) = (1/E)[1 + u^α/Γ(1+α)]**.
2. Fractional Kelvin–Voigt: Ĵ = 1/(s E (1+(sτ)^α)) = (1/E) τ^{-α} s^{-1}/(s^α + τ^{-α}) ⇒
   J = (1/E) τ^{-α} t^α E_{α,α+1}(−u^α) (β = α+1, μ = τ^{-α}); by the identity this is
   **J(t) = (1/E)[1 − E_α(−u^α)]**. Relaxation: Ĝ = E/s + E τ^α s^{α−1} ⇒ **G(t) = E + E u^{-α}/Γ(1−α)** for
   0 < α < 1 (finite for t > 0, singular at 0+); α = 1 is the ideal Kelvin–Voigt (impulsive, **refused**, as
   `kelvin_voigt_relaxation`).
3. Fractional Zener: s Ĝ = [G∞ + G0 x]/(1+x), x = (sτ)^α, so **G(t) = G∞ + G1 E_α(−u^α)**
   (Ĝ = G∞/s + G1 s^{α−1}/(s^α+τ^{-α})). Ĵ = (1/s)(1+x)/(G∞+G0 x) = (1/(s G0))[1 + (1−λ)/(x+λ)]; with
   μ = λ τ^{-α}, β = α+1 and the identity: **J(t) = 1/G∞ − (G1/(G∞ G0)) E_α(−λ u^α)**. At α = 1 it is
   `sls_creep` with τ_c = τ/λ = τ G0/G∞ (checked).
4. ∫₀^t E_α(−(s/τ)^α) ds = t E_{α,2}(−u^α) (term by term: ∫ s^{αk}/Γ(αk+1) ds = t^{αk+1}/Γ(αk+2)). Hence for a
   ramp-hold shear strain γ(t) = γ0 min(t, t_r)/t_r the fractional Zener stress is (Boltzmann superposition)
   σ(t) = (γ0/t_r)[Φ(t) − Φ(t − t_r)H(t − t_r)], Φ(t) = G∞ t + G1 t E_{α,2}(−u^α).

Complex modulus, u_ω = (ωτ)^α, c = cos(πα/2) ≥ 0, s_α = sin(πα/2) ≥ 0; (iωτ)^α = u_ω e^{iπα/2} (principal branch),
z = (iωτ)^α, z/(1+z) = z(1+z̄)/|1+z|², |1+z|² = 1 + 2 u_ω c + u_ω² ≥ 1:
G'' = G1 u_ω s_α / (1 + 2 u_ω c + u_ω²) ≥ 0, G' = G∞ + G1 (u_ω c + u_ω²)/(1 + 2 u_ω c + u_ω²).
The map u → 1/u leaves u/(1+2uc+u²) invariant, so G''(ω) = G''(1/(τ²ω)) (symmetric in log ω about ω = 1/τ), and
d/du [u/(1+2uc+u²)] = (1−u²)/(1+2uc+u²)² ⇒ **the loss peak is at ωτ = 1 for every α, with
G''_max = G1 s_α/(2+2c) = (G1/2) tan(πα/4)** (α = 1: G1/2). Limits: ω → 0: G' → G∞, G'' ≃ G1 s_α (ωτ)^α;
ω → ∞: G' → G0, G'' ≃ G1 s_α (ωτ)^{-α}. Fractional Maxwell (G∞ = 0) at low ω: tan δ = G''/G' → s_α/c = tan(πα/2).

## Mittag-Leffler evaluation (method)

For z = −x ≤ 0, 0 < α < 1: (i) **series** Σ (−x)^k/Γ(αk+β) (terms via exp(k ln x − lnΓ(αk+β))) when x^{1/α} ≤ 5
(the largest term is ≈ exp(x^{1/α}), so the cancellation is ≤ e^5 ≈ 150); (ii) otherwise the **Hankel contour
deformed onto the negative real axis**: from E_{α,β}(z) = (1/2πi)∫_Ha e^t t^{α−β}/(t^α − z) dt (the poles
t^α = −x sit at |arg t| = π/α > π, off the principal sheet for α < 1) the two banks t = r e^{±iπ} of the cut give,
for β < α+1,
E_{α,β}(−x) = (1/π)∫₀^∞ e^{-r} r^{α−β} [r^α sin πβ + x sin π(β−α)] / (r^{2α} + 2x r^α cos πα + x²) dr
= (1/(πα))∫₀^∞ e^{-u^{1/α}} u^{(1−β)/α} [u sin πβ + x sin π(β−α)] / (u² + 2xu cos πα + x²) du (u = r^α).
β = 1: E_α(−x) = (x sin πα/(πα))∫₀^∞ e^{-u^{1/α}}/(u²+2xu cos πα + x²) du, smooth and bounded (large x gives
1/(Γ(1−α)x)). (iii) β ≥ α+1: reduce to β' = β − mα ∈ (1−α, 1] and recur upward
E_{α,β+α}(z) = (E_{α,β}(z) − 1/Γ(β))/z (used only for x > 1: no cancellation). α = 1: exp (β = 1), integer β by the
same recurrence. Integration by `scipy.integrate.quad` (relative 1e-13, split at u = x), upper limit u^{1/α} = 45.
**This formula is derived here, not taken from a table**; its independent confirmation is VF4 (high-precision
series, asymptotic expansion, mpmath `invertlaplace`).

## Prony bridge

Target: G(t) = G∞ + G1 E_α(−(t/τ)^α), synthetic G∞ = 2e5 Pa, G1 = 8e5 Pa, τ = 1 s. Window of validity
t ∈ [t_lo, t_hi] = [1e-4, 1e4] s. Branch times τ_i log-spaced on [1e-4, 1e5] s with N points (N−1 = 4, 8, 16, 32, 64,
so the grids are **nested**: N = 5, 9, 17, 33, 65). Fit: non-negative least squares (`scipy.optimize.nnls`) of
G∞' + Σ g_i e^{-t_j/τ_i} on 801 log-spaced t_j in the window, rows weighted by 1/G(t_j). Error metrics on independent
grids (not the fit points): e_t = max over 4001 log-spaced t in the window of |G_N − G|/G; e_ω = max over 1201
log-spaced ω ∈ [1e-3, 1e3] rad/s of |G*_N − G*|/|G*|. Since E_α(−t^α) is completely monotone for 0 < α ≤ 1, its
exact relaxation spectrum is positive and the non-negativity constraint does not exclude a good solution. The result
is a `PronySeries` and, with a bulk series, a `GeneralizedMaxwell`.

## Checks

| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| VF1 | `mittag_leffler(−x, α)` (β = 1) for α ∈ {0.3, 0.5, 0.8, 1.0}, x ∈ {0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1, 1.6, 2, 3, 4, 5, 8, 12, 20, 30, 50, 100, 200, 300, 500, 700, 1000} | mpmath, ≥ 30 digits: α = 1 exp; α = 1/2 e^{x²}erfc(x) (exact closed form); otherwise a series at raised precision when x^{1/α} ≤ 150, the asymptotic expansion Σ_{k=1}^{n}(−1)^{k+1}x^{-k}/Γ(β−αk) (n = 20, x ≥ 300), else mp.quad of the integral above (40 digits; cross-validated in VF4) | relative error ≤ 1e-10 at every point |
| VF2 | `mittag_leffler(−x, α, β)` for α ∈ {0.3, 0.5, 0.8}, β ∈ {α, 1.5, 1+α, 2, 2.7}, same x grid | same reference with that β | relative ≤ 1e-9 |
| VF3 | stress α ∈ {0.1, 0.95, 0.99}, β = 1, same x grid | same reference | relative ≤ 1e-8 |
| VF4 | the reference itself is right: (a) high-precision series vs mp.quad integral where x^{1/α} ≤ 150 (α = 0.3: x = 3, 4; α = 0.8: x = 20, 50); (b) asymptotic vs mp.quad at x = 300 (α = 0.3, 0.8; β = 1 and 1.5); (c) mpmath `invertlaplace` (talbot, 30 digits) of s^{α−1}/(s^α+1) at t = 0.5, 2, 20 vs the reference E_α(−t^α) (α = 0.3, 0.8) | three independent routes | (a), (b) relative ≤ 1e-20; (c) relative ≤ 1e-10 |
| VF5 | fractional Zener relaxation G(t) = G∞ + G1 E_α(−u^α), α ∈ {0.3, 0.5, 0.8}, u ∈ {1e-2, 0.1, 1, 10, 100} (G∞ = 2e5, G1 = 8e5 Pa, τ = 1 s) | `mpmath.invertlaplace` (talbot, dps 30) of Ĝ = G*(s)/s with mp complex power | relative ≤ 1e-8 |
| VF6 | creep and the other two models: fractional Zener J (α ∈ {0.3, 0.5, 0.8}), fractional KV J, fractional Maxwell J, fractional KV G, fractional Maxwell G, same u grid | `invertlaplace` of Ĵ = 1/(s G*(s)) resp. G*(s)/s; fractional KV relaxation at α = 1 raises ValueError | relative ≤ 1e-8; exact raise |
| VF7 | α = 1 recovers `core.viscoelastic`: Zener G, J vs `sls_relaxation`, `sls_creep`; KV J vs `kelvin_voigt_creep(E, η = Eτ)`; Maxwell G, J vs `maxwell_relaxation`, `maxwell_creep(E, η = Eτ)`; complex modulus vs `complex_modulus(PronySeries(G∞,[G1],[τ]))`. Continuity: α = 0.99 and 0.999: max_t \|G_α − G_1\|/G1 and max_ω \|G*_α − G*_1\|/G1 ≤ 2(1−α) (the α-derivative of z/(1+z), z = (iωτ)^α, is at most ≈ π/4 = 0.785 at ωτ = 1, an estimate made here, not measured; that of E_α(−t^α) assumed O(1)) | the existing module (a regression of a limit, labelled so) | α = 1: relative ≤ 1e-13; continuity ≤ 2(1−α) |
| VF8 | passivity and peak: G'' ≥ 0 for α ∈ {0.3, 0.5, 0.8, 0.99, 1} on 401 log-spaced ω ∈ [1e-8, 1e8]/τ, for the production function and for an independent complex-power evaluation G∞ + G1 z/(1+z), z = (1j ω τ)**α (numpy) and mp complex at s = iω; the two agree; peak at ωτ = 1 with G''_max = (G1/2) tan(πα/4) (exact at ω = 1/τ) and the argmax over a 20001-point log grid on [1e-3, 1e3] within 2e-3 relative of 1/τ; symmetry G''(ω) = G''(1/(τ²ω)); low-frequency slope G''/(G1 s_α (ωτ)^α) → 1 at ωτ = 1e-30; fractional Maxwell tan δ → tan(πα/2) at ωτ = 1e-30; G' → G∞ and G0 at ωτ = 1e-30, 1e30 | formulas derived above, numpy/mpmath complex arithmetic | G''/G1 ≥ −1e-14 (production and independent); agreement relative ≤ 1e-12; peak value relative ≤ 1e-12; argmax ≤ 2e-3; slope and tan δ relative ≤ 1e-6; limits relative ≤ 1e-6 |
| VF9 | Prony bridge, α ∈ {0.3, 0.5, 0.8}, nested N = 5, 9, 17, 33, 65: e_t strictly decreasing with N; e_t(17) ≤ 5e-2, e_t(33) ≤ 5e-3, e_t(65) ≤ 5e-4; e_ω(65) ≤ 5e-2 (e_ω reported for all N, no other constraint); all amplitudes ≥ 0; the production G used in e_t agrees with the mpmath reference at 40 grid points to relative 1e-9 | mpmath reference of G | as stated |
| VF10 | time-domain drive of the fitted `GeneralizedMaxwell` (shear = fit, bulk = elastic 1e6 Pa, α = 0.5, N ∈ {17, 33, 65}): pure shear γ_xy, ramp to γ0 = 1e-3 in t_r = 1e-3 s (one exact step), then hold with steps to t = 1e3 s; σ_xy at 25 log-spaced hold times t_k ∈ [1e-2, 1e3] s vs σ_ref = (γ0/t_r)[Φ(t) − Φ(t−t_r)] with Φ from item 4 and the mpmath E_{α,2}. Since σ_GM − σ_ref = (γ0/t_r)∫(G_N − G) over arguments in [t−t_r, t] ⊂ window, the reported error bounds it | independent mpmath Φ | \|σ_GM − σ_ref\| ≤ 1.01 γ0 e_t(N) G(t_k − t_r) + 1e-9 γ0 G0 at every t_k |
| VF11 | negative controls and input validation: (a) a naive double-precision power series at α = 1, x = 50 has absolute error > 1 (why the integral is needed); (b) the harness detects a perturbed order: `mittag_leffler(−x, α(1+1e-3))` differs from the reference by relative > 1e-6 for some x of the VF1 grid (α = 0.5, 0.8); (c) the N = 5 fit has e_t > 100 · e_t(65); (d) ValueError for α ∉ (0, 1], z > 0, NaN, E ≤ 0, τ ≤ 0, t < 0, window with t_lo ≥ t_hi, N < 2 | — | as stated |

Run: `python -m pytest tests/toporlogy/test_fractional.py -q -p no:randomly -p no:cacheprovider` (TMP/TEMP under
`D:/Masaustu/Topology Optimization/.claude/tmp-toporlogy/VF`; resource cap: a few minutes, < 500 MB).
Expected failure modes: near α = 1 (0.99) the narrow Lorentzian of the integral may cost digits (VF3); the Prony
monotonicity might fail at the noise floor of NNLS (N = 65). A failed check is kept as xfail(strict) with the measured
value, never loosened.

## First result

(appended after the first run)

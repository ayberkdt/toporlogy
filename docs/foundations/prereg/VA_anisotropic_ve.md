# VA_anisotropic_ve — Anisotropic linear viscoelasticity at the material point (preregistration)

Preregistered 2026-11-27 09:01 +0300 (clock output), before any new numerical test was run. Base SHA `36c020f` (ff-merged; worktree
branch `worktree-agent-a6f428f0aa8711bc1`).
Module: `core/viscoelastic_aniso.py` (`AnisotropicMaxwell`, `isotropic_stiffness`); document
`docs/foundations/ANISOTROPIC_VISCOELASTICITY.md`; tests `tests/toporlogy/test_viscoelastic_aniso.py`. WP12 V05.
All parameters in tests are **synthetic** (not material data). Production code was written before this record; no
test of it had been executed (only an import).

## Model and derived expectations (sign, normalisation and reaction derived here, not assumed)

Voigt order [xx, yy, zz, yz, xz, xy]; stress physical shear, strain engineering shear; tension positive. Mandel map
m = (1,1,1,√2,√2,√2): ε_M = tensor_to_mandel(ε) (shear √2 ε_ij), σ_M = m∘σ_V, M = diag(m) C_V diag(m). Then
σ_V = C_V ε_V ⇔ σ_M = M ε_M (ε_V = m∘ε_M), and σ_M·ε_M = σ:ε.

σ = C∞ε + Σ s_i, ṡ_i = −s_i/τ_i + C_i ε̇. Strain linear on a step (Δt, Δε): s(u) = e^{−au}s₀ + h(u) C Δε,
a = Δt/τ, h(u) = (1−e^{−au})/a, u∈[0,1]; end of step: s₁ = r s₀ + φ CΔε, r = e^{−a}, φ = (1−e^{−a})/a (exact).
Algorithmic tangent dσ/dε = C∞ + Σφ_iC_i (Voigt). Step strain at t=0+ (a step of Δt=0, φ=1) then hold (Δε=0):
s_i(t) = C_iΔε e^{−t/τ_i}, hence **C(t) = C∞ + Σ C_i e^{−t/τ_i}** and σ(t) = C(t) ε₀ (reaction = stress of the held
strain mode k, column k of C(t)ε₀ in Voigt with ε_V = [ε_xx, ε_yy, ε_zz, 2ε_yz, 2ε_xz, 2ε_xy]).

Energy ψ = ½ε_Mᵀ M∞ ε_M + Σ ½ s_Mᵀ M_i⁺ s_M (M_i⁺ pseudo-inverse of the Mandel matrix on its range; a branch tensor
has null space where eigenvalue ≤ 1e-12 λ_max; an eigenvalue < −1e-12 λ_max is refused). With s in range:
ψ̇_i = sᵀM⁺ṡ = −sᵀM⁺s/τ + s·ε̇, so the dissipation rate is D = Σ s_iᵀM_i⁺s_i/τ_i and work = Δψ + ∫D. Exact step
integral (a∫₀¹ Q(s(u),s(u)) du with Q(x,y)=xᵀM⁺y, Q(s₀,CΔε)=s₀·Δε for s₀ in range, Q(CΔε,CΔε)=ΔεᵀCΔε), q = 1−e^{−a}:
**D_step = ½q(2−q) s₀ᵀM⁺s₀ + aφ² s₀·Δε + g3(a) ΔεᵀMΔε**, g3 = (a−q−q²/2)/a² (same g3 as `GeneralizedMaxwell`).
Isotropic C_i = 3K P_vol + 2G P_dev has M⁺ = P_vol/(3K) + P_dev/(2G) on its range, so ψ, D reduce to the formulas of
`GeneralizedMaxwell` (s:s/(4G), p²/(2K), s:s/(2Gτ), p²/(Kτ)).

DMA (e^{iωt}): C*(ω) = C∞ + Σ C_i iωτ_i/(1+iωτ_i) = C∞ + Σ C_i (x²+ix)/(1+x²), x = ωτ_i. For ε = ε₀ sin ωt:
σ = M′e₀ sin ωt + M″e₀ cos ωt (M′ = Re, M″ = Im of the Mandel C*, e₀ Mandel vector), work per period
W = ∮σ·ε̇ dt = π e₀ᵀM″e₀ ≥ 0 iff M″ ⪰ 0 (passivity; for isotropic shear amplitude γ₀: π G″γ₀²). M″ = Σ M_i x/(1+x²) is
PSD for PSD M_i (each term has a nonnegative scalar factor); M′ − M∞ = Σ M_i x²/(1+x²) is PSD; a single branch peaks at
x=1 with value M_i/2.

## Checks

| ID | Hypothesis | Reference (independence) | Metric / threshold |
|---|---|---|---|
| VA1 | C_i = 2G P_dev (shear-only, rank 5), 3K P_vol (bulk-only, rank 1) with independent τ, and combined (K_j, G_j same τ) reproduce `GeneralizedMaxwell` on a random 40-step history (strain scale 1e-3, dt log-uniform in [1e-3, 1e2] s, one dt = 0 step; seeds 0–2). Exercises the pseudo-inverse on a singular range | `GeneralizedMaxwell` (a separately written production kernel; a cross-implementation check, **not** an independent oracle) | stress, tangent: max abs diff / max abs ref ≤ 1e-13; ψ ≤ 1e-12 of max ψ; D ≤ 1e-11 of max D. (The task says "bitwise or 1e-14"; the summation order differs so bitwise is not expected; 1e-13 relative to the peak is the registered threshold, fixed before the run.) |
| VA2 | Orthotropic (`Orthotropic.stiffness()` scaled, plus one rank-deficient branch) step strain in each of the 6 Mandel modes and a random strain: σ(t) = C(t)ε₀ at 12 times in [1e-3, 1e4] s | closed form derived above, evaluated in the test in Voigt with its own ε_V (no Mandel) | max abs err / max abs σ(0+) ≤ 1e-12 |
| VA3 | Objectivity: model(rotation=R) under ε(t) equals R σ_model(rotation=None)(RᵀεR) Rᵀ for stress, tangent (rotated by test einsum), ψ and D invariant; isotropic model unaffected by R | test-side einsum rotation of tensors (does not call `rotate4`) | rel to peak ≤ 1e-12 (stress, tangent, ψ, D) |
| VA4 | Irregular multiaxial history (12 knots at irregular times in [0, 50] s, 3 branches τ in [0.3, 40], one rank-deficient, orthotropic + rotation) matches the hereditary solution | `solve_ivp` DOP853 (rtol 1e-12, atol 1e-14·scale) of ṡ = −s/τ + C_V ε̇_V per linear segment, in Voigt (no Mandel) | max abs err / max abs σ ≤ 1e-9 at the knots |
| VA5 | Algorithmic tangent equals the central finite-difference Jacobian of the update stress (h = 1e-6 in each Voigt strain component, engineering shear) at a state with history; `tangent_kind="continuum"` equals C∞+ΣC_i exactly | central FD of the same update (independent of the tangent formula) | max abs err / max abs C ≤ 1e-8; continuum equality ≤ 1e-14 rel |
| VA6 | ψ ≥ 0 and D ≥ 0 on 5 random models (SPD C∞, random PSD branches incl. rank-deficient), 60 steps, dt log-uniform [1e-4, 1e3]; D equals the 10-point Gauss–Legendre integral of the rate Σ sᵀM⁺s/τ along the exact branch stress for histories with dt ≤ 4 τ_min | GL quadrature in the test with its own `numpy.linalg.pinv` of the test-built Mandel matrix | min ψ ≥ −1e-12 max ψ; min D ≥ −1e-11 max D; GL: abs diff ≤ 1e-10 max D |
| VA7 | Energy closure: (i) W_exact − (ψ_N − ψ_0) − ΣD, with W_exact the 10-pt GL integral of σ·ε̇ along the exact stress of each step (dt ≤ 4τ_min), ≤ 1e-10·max(W, ΣD, ψ_N); (ii) trapezoid work W_h = Σ ½(σ_n+σ_{n+1})·Δε on a smooth multiaxial sinusoid, T = 2 s, N = 40, 80, 160 steps: residual R_h = W_h − Δψ − ΣD shrinks with ratio abs(R_h/R_{h/2}) ≥ 3.2 for both refinements and abs(R_160) ≤ 1e-3 W | GL work (test side, exact path from the ODE solution), trapezoid (second-order rule, error O(dt²)) | as stated |
| VA8 | Refused with ValueError: branch tensor with eigenvalue −1e-3 λ_max (Mandel), asymmetric tensor, wrong shape, NaN, τ ≤ 0 / inf, C∞ non-SPD, singular C∞ without `allow_fluid`, singular C∞ with `allow_fluid` but C∞+ΣC_i singular, rotation not orthonormal and reflection, missing `request.step`. Accepted: PSD branch with a zero eigenvalue, zero tensor, fluid C∞ with `allow_fluid=True` (`is_fluid` True). The classification is by the Mandel spectrum (a Voigt matrix that is Mandel-PSD but has a negative Voigt eigenvalue, if one exists in the test set, is accepted) | — | exact |
| VA9 | Restart and rollback bitwise: 12 steps in one run vs 6 + new model instance restored from copied committed state; a discarded wild trial leaves the committed arrays unchanged (bytes) and the next evaluation equals the run without the trial; `drive_material_point` with `times` for a strain-controlled path equals the manual loop | the uninterrupted run (regression) | `np.array_equal` |
| VA10 | DMA passivity: for 3 models (one rotated, one with rank-deficient branches) over 400 ω in [1e-6, 1e6] rad/s: λ_min(M″) ≥ −1e-12 λ_max(M″), λ_min(M′−M∞) ≥ −1e-12 λ_max(M′), M*(0) = C∞ (≤1e-14 rel), M*(ω→∞) = C₀ (1e-10 rel at ωτ_min = 1e12); single branch: max M″ at ωτ = 1 equals M_i/2; steady-state cycle: starting from the analytic periodic state s_i(0) = M_i e₀ x/(1+x²), ε = ε₀ sin ωt, 2000 steps per period, ΣD over the period equals π e₀ᵀM″e₀ | closed forms above; the cycle work uses production D (the ψ-return is by periodicity) | spectra as stated; cycle: rel err ≤ 1e-4 |
| VA11 | Contract: `physical`, `kinematics`, `conservative`, `state_fields`, `initial_state` shapes; `evaluate` with `Kinematics` equals `update` (regression); `committed` arrays (read-only) are not written; 2-D plane strain runs, plane stress refused | contract | exact |
| VA12 | Stability: single branch, dt/τ in {1e-10, 1e-3, 1, 50, 1e6}; φ from mpmath (50 digits); stress after a strain jump over dt equals C∞ε + φ C₁ Δε; s after a hold step with dt/τ = 1e6 is exactly 0; no NaN/inf; ψ, D finite and D ≥ 0 | mpmath | rel ≤ 1e-13 |
| NC1 | Negative control: a branch tensor with one eigenvalue −0.2 λ_max (Mandel) is refused (VA8) and, simulated by dense ODE (`solve_ivp`, 30 cycles, ω = 2, τ = 0.5, strain amplitude along the negative eigenvector), has negative work in the last cycle equal to π e₀ᵀM″e₀ (< 0); the PSD control of the same set-up has positive work equal to the same formula | `solve_ivp` of ṡ = −s/τ + Cε̇ with W as an extra state (Voigt, no Mandel); this also validates the cycle-work formula independently of the production code | sign exact; value rel err ≤ 1e-6 |

Seeds fixed in the test file (numpy `default_rng`). dtype float64, numpy backend. No mesh. Expected failure modes:
(a) a Voigt-vs-Mandel mix-up shows in VA2/VA4 as a shear-coupling error of order 1; (b) the pseudo-inverse on a
rank-deficient branch would show in VA1 (energy) and VA6; (c) an orientation convention error (R vs Rᵀ) shows in VA3.

Run: `TMP/TEMP=D:/Masaustu/Topology Optimization/.claude/tmp-toporlogy/VA`, `python -m pytest
tests/toporlogy/test_viscoelastic_aniso.py -q -p no:randomly -p no:cacheprovider --basetemp=<TMP>/bt`. Resource cap:
one process, < 200 MB.

## First result (2026-11-27 09:07 +0300, clock output; run at 09:04 +0300 on base 36c020f + production module)

All 20 collected tests PASS on the first run (`python -m pytest tests/toporlogy/test_viscoelastic_aniso.py -q -p no:randomly`).
No threshold changed and no test-set-up fix was needed. Measured values (worst over the parametrisation):

| ID | Measured | Threshold |
|---|---|---|
| VA1 (6 cases: separate/combined x seeds 0-2) | stress 3.12e-16, tangent 3.31e-16, psi 7.68e-16, D 4.17e-16 (relative to peak) | 1e-13, 1e-13, 1e-12, 1e-11 |
| VA2 | 3.64e-16 | 1e-12 |
| VA3 | stress 4.61e-16, tangent 1.63e-15, psi 3.62e-16, D 7.57e-16; isotropic 6.31e-16 | 1e-12 |
| VA4 | 5.32e-12 | 1e-9 |
| VA5 | FD 7.11e-14; continuum 0.00e+00 | 1e-8; 1e-14 |
| VA6 | psi, D positive in all 300 steps (smallest min/max ratios 1.19e-02, 1.36e-05, from the re-run with the recording fixed); D vs GL 5.28e-14 | >= -1e-12, >= -1e-11; 1e-10 |
| VA7(i) | 7.34e-16 | 1e-10 |
| VA7(ii) | residuals N=40/80/160: -68.731, -17.222, -4.308; ratios 3.991, 3.998; abs(R160)/W 2.22e-04 | ratios >= 3.2; <= 1e-3 |
| VA8 | all refusals/acceptances as registered; negative branch lambda_min/lambda_max = -1.000e-03 | exact |
| VA9 | restart, rollback, drive_material_point restart bitwise equal | exact |
| VA10 | passivity: smallest eig(M'')/max 2.52e-04, eig(M'-Minf)/max 8.91e-14 (both >= 0); cycle dissipation rel err 1.44e-06 | >= -1e-12; 1e-4 |
| VA11 | contract holds | exact |
| VA12 | 1.35e-16 | 1e-13 |
| NC1 | W_bad = -314.1592654 (formula -314.1592654), W_ok = 314.1592654 (formula 314.1592654) | sign exact; 1e-6 |

Notes. The first run recorded VA6 and VA10 minima through `min(0, ...)`, which printed 0.0; the recording was changed
(test instrumentation only, no assertion or threshold) and the suite re-run (also 20 passed) to obtain the true
minima shown above. VA2 drives the six unit Voigt strain directions (engineering shear component 1e-3), not unit
Mandel vectors; both span the same six directions. VA1 passed at ~1e-16, far inside 1e-13, so the cross-implementation
agreement is at rounding level. Lint (`ruff`) and `mypy` on the module are clean. Raw record:
`.claude/tmp-toporlogy/VA/record.json`, `record2.json` (not committed).
Limitations: see ANISOTROPIC_VISCOELASTICITY.md; follow-ups (anisotropic temperature shift, per-point orientation,
state VJP/JVP) need separate preregistrations.

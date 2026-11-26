# MM — Constituent-to-composite micromechanics (preregistration)

Preregistered 2026-11-27 02:35 +0300 (clock), before any new test was executed. Base SHA `a2b50a5`. Module
`core/micromechanics.py` (new), tests `tests/toporlogy/test_micromechanics.py`, record `docs/foundations/MICROMECHANICS.md`.
Plan sources: `18_COMPOSITE_MICROMECHANICS.md` (MM01), `04_ELASTIC_MODULI.md` (E05), `23_ACCEPTANCE_CASES.md` (E7, MM1, MM2,
MM3, MM4).

## Sources: what was read, what is derived

No literature text was read in this session. `src/toporlogy/references` holds entries for Hashin–Shtrikman 1963,
Bendsøe–Sigmund 1999, Voigt 1889 and Reuss 1929, all marked "text not read". `design/interp.py` (read) carries a derivation
of the HS bounds for a void phase and of the SIMP exponent. Everything below is **derived here** (Sections 1–4) or **checked
against an independent solution in the tests**; where a classical value is quoted from memory (Eshelby's sphere components,
Mackenzie's dilute pore result) it is labelled so and is not the only evidence.

## Conventions

Phases are isotropic with bulk K and shear G (3-D), or with the plane bulk modulus κ = λ + μ (2-D, written K below) and
G = μ (2-D; plane stress: κ = E/(2(1−ν))). Void: K = G = 0. Valid phase: K > 0 and G > 0, or the void. Fractions f_i ∈ [0, 1],
Σ f_i = 1 (tolerance 1e-9 on the sum, none on the range). Tensors in Mandel form for the algebra, Voigt [xx, yy, zz, yz, xz, xy]
(stress physical shear, strain engineering shear) at the interface.

## 1. Voigt, Reuss, and the shift form of the Hashin–Shtrikman bounds

Voigt: K_V = Σ f K_i, G_V = Σ f G_i (uniform strain). Reuss: 1/K_R = Σ f/K_i, 1/G_R = Σ f/G_i (uniform stress); zero if a void
phase has f > 0. Both are rigorous for any microstructure (Loewner order, C_R ≤ C_eff ≤ C_V).

Comparison-medium form. Let C0 = (K0, G0) be an isotropic comparison medium and C* its "constraint" tensor, the one with
P0 = S0 C0⁻¹ (Hill polarisation tensor of a sphere, resp. circle) and C* = P0⁻¹ − C0 = C0 (S0⁻¹ − I). With the Eshelby sphere
S0 = α P_V + β P_D (Section 2), α = 3K0/(3K0+4G0), β = 6(K0+2G0)/(5(3K0+4G0)):

* bulk shift K* = K0 (1/α − 1) = 4G0/3 (3-D); 2-D circle: α = κ0/(κ0+G0), K* = G0.
* shear shift ζ = G0 (1/β − 1) = G0 (9K0 + 8G0) / (6 (K0 + 2G0)) (3-D); 2-D: ζ = G0 κ0 / (κ0 + 2 G0).

The Hashin–Shtrikman–Walpole bounds read

    K_HS = [ Σ f_i / (K_i + K*) ]⁻¹ − K*,      G_HS = [ Σ f_i / (G_i + ζ) ]⁻¹ − ζ,

upper bound with (K0, G0) = (max K_i, max G_i), lower with (min K_i, min G_i), over the phases with f_i > 0. **Phase-ordering
assumption**: the upper bound needs C0 ≥ C_i for every phase in the Loewner order, i.e. both K0 ≥ K_i and G0 ≥ G_i; so K0 and G0
are the largest values present, and they need not belong to the same phase. If the stiffer-K phase is not the stiffer-G phase
(crossing) the bound is still valid but is attained by no known microstructure; it is not the HS formula with one phase as
reference. Void: lower bound K = G = 0 whenever the void fraction is positive (the term f_void/(0 + 0) is +∞ and the shifts are 0);
upper bound finite. Statement of validity: statistical isotropy of the composite, two or more linear elastic isotropic phases, no
claim for anisotropic composites. I did not prove validity of the shift form here (that is the Hashin–Shtrikman variational
principle, text not read); it is **checked**: ordered pairs against the textbook two-phase form written in the test; the bulk
bound against Hashin's coated-sphere assemblage solved from the Lamé solution in the test (both coating orders, also for a crossed
pair, where both realisable values must lie inside); the void limit against `design/interp.py`; ordering below.

Void limit is a RAMP form. With phase 2 void and x the solid fraction, [x/(a+s) + (1−x)/s]⁻¹ − s = a x / (1 + (a/s)(1−x)) for a solid
modulus a and shift s: K_HS = K x/(1 + q_K (1−x)) with q_K = K/K*, G_HS = G x/(1 + q_G (1−x)) with q_G = G/ζ. (3-D: q_K = 3K/(4G),
q_G = 6(K+2G)/(9K+8G); 2-D: q_K = κ/G, q_G = (κ+2G)/κ.)

Ordering (E7). t ↦ g(t) = [Σ f_i/(a_i + t)]⁻¹ − t has g'(t) = Σ f w²/(Σ f w)² − 1 ≥ 0 (w_i = 1/(a_i + t), Cauchy–Schwarz), so g is
nondecreasing, g(0) = Reuss, g(∞) = Voigt. Shifts: 0 ≤ K*(G_min) ≤ K*(G_max) ≤ ∞, and ζ(K,G) is increasing in K (∂ζ/∂K = 5G²/(3(K+2G)²) > 0)
and in G (numerator (9K²+16KG+16G²)/(K+2G)² > 0); hence **K_R ≤ K_HS− ≤ K_HS+ ≤ K_V and G_R ≤ G_HS− ≤ G_HS+ ≤ G_V for every admissible
input, crossed or not.**

## 2. Eshelby sphere, dilute and Mori–Tanaka (MM2)

S (strain of an ellipsoidal inclusion = S × eigenstrain) for a sphere in an isotropic matrix: S = α P_V + β P_D in Mandel form,
P_V = (1/3) e eᵀ with e = (1,1,1,0,0,0), P_D = I − P_V, α = (1+ν)/(3(1−ν)) = 3K/(3K+4G), β = 2(4−5ν)/(15(1−ν)). α is derived: a
dilating inclusion (eigenstrain ε* I) has uniform interior strain A I and exterior u = B/r², continuity gives B = A a³, radial
stress continuity 3K(A − ε*) = −4G A, hence A = α ε*. β is quoted: Eshelby's components S1111 = (7−5ν)/(15(1−ν)), S1122 =
(5ν−1)/(15(1−ν)), S1212 = (4−5ν)/(15(1−ν)) (from memory, text not read) give P_D eigenvalue S1111 − S1122 = (8−10ν)/(15(1−ν)) = β;
the test builds S from these components independently and compares.

Dilute: ε_inclusion = A_dil ε̄, A_dil = [I + S C_m⁻¹ (C_i − C_m)]⁻¹; C_eff = C_m + f (C_i − C_m) A_dil. For isotropic phases, per
part: K_dil = K_m + f ΔK/(1 + a_K), G_dil = G_m + f ΔG/(1 + a_G) with a_K = ΔK/(K_m + K*_m), a_G = ΔG/(G_m + ζ_m) (using β/G_m =
1/(G_m + ζ_m)). Mori–Tanaka: A_MT = A_dil [(1−f) I + f A_dil]⁻¹ (average strain of the matrix equals the far-field strain
closure), C_eff = C_m + f (C_i − C_m) A_MT, i.e. K_MT = K_m + f ΔK/(1 + (1−f) a_K), same for G.
Implementation: 6×6 Mandel matrices, `np.linalg.solve`, no scalar shortcut; the closed forms above are the reference.

(a) MT − dilute: ΔK_MT − ΔK_dil relative to ΔK_dil is exactly f a_K / (1 + (1−f) a_K): O(f).
(b) **MT is the HS form with the matrix as comparison medium.** With x_m = G_m + ζ_m, x_i = G_i + ζ_m: [ (1−f)/x_m + f/x_i ]⁻¹ − ζ_m =
G_m + f (G_i − G_m)/(1 + (1−f)(G_i − G_m)/x_m) (algebra: both equal x_m x_i/((1−f)x_i + f x_m) − ζ_m); identically for K. So MT with matrix
= the phase that has the larger K and the larger G (stiff matrix, soft spherical inclusions) **is the HS upper bound**, and MT
with the softer matrix and stiffer inclusions is the HS lower bound. **The brief's wording ("MT equals the HS lower bound for a
stiff-matrix/soft-inclusion pair") is the other way round; the derivation says upper. The test is written on the derived
statement, and also asserts that MT(stiff matrix, soft inclusion) differs from the HS lower bound by more than 1 % at f = 0.3.
This is recorded as a failed prediction of the brief's wording, not as a code or test failure.**

## 3. Interpolation admissibility (MMI1)

Scalar interpolation C(x) = r(x) C_solid, solid/void mixture, x = solid fraction, ν fixed. Admissible iff r(x) K ≤ K_HS(x) and r(x) G ≤
G_HS(x) for all x ∈ [0,1]. By Section 1, K_HS(x)/K = x/(1 + q_K (1−x)) and G_HS(x)/G = x/(1 + q_G (1−x)); with q = max(q_K, q_G):

* RAMP r = x/(1 + q_r (1−x)) ≤ x/(1 + q (1−x)) for all x iff q_r ≥ q (exact, both directions).
* SIMP r = x^p: x^p ≤ x/(1+q(1−x)) ⟺ h(x) = x^{p−1}(1 + q − q x) ≤ 1. h(1) = 1, h'(x) = x^{p−2}[(p−1)(1+q) − q p x]; for p−1 ≥ q the root
  x0 = (p−1)(1+q)/(q p) ≥ 1, so h is increasing on [0,1] and h ≤ 1: sufficient; h'(1) = p − 1 − q < 0 for p−1 < q: h > 1 just below x = 1:
  necessary. **p_min = 1 + q = max(1 + K/K*, 1 + G/ζ).**
* Closed forms. 2-D plane stress (κ = E/(2(1−ν)), G = E/(2(1+ν))): 1 + κ/G = 2/(1−ν), 1 + G/ζ = 2 + 2G/κ = 4/(1+ν): p_min = max(2/(1−ν), 4/(1+ν)).
  3-D: p_min = max(3(1−ν)/(2(1−2ν)), 15(1−ν)/(7−5ν)). In 2-D p_min ≥ 3 with equality only at ν = 1/3 (the two branches cross where
  2(1+ν) = 4(1−ν)). This reproduces the condition attributed to Bendsøe & Sigmund (1999), which I did not read; the derivation is mine.
  RAMP: q_min = p_min − 1. Plane strain differs through κ = λ + μ.
* x ↦ the grid check uses the general `hs_bounds` with phases (void, solid) over 2001 points including both endpoints.

## 4. Fraction conversion (MM3)

v_i volume fractions, ρ_i densities. Mixture density ρ = Σ v_i ρ_i; mass fraction w_i = v_i ρ_i/ρ; inverse v_i = (w_i/ρ_i)/Σ_j(w_j/ρ_j),
requires ρ_i > 0 for every phase (a void has w = 0 and its volume is not recoverable from mass fractions: refused). Hand
numbers: v = (0.6, 0.3, 0.1), ρ = (1800, 1200, 0) kg/m³ → ρ = 1080 + 360 = 1440, w = (0.75, 0.25, 0).

## 5. Interpolation of two isotropic solids (MMI2)

x = fraction of phase B. Schemes: voigt, reuss, hs_upper, hs_lower (Section 1, with two phases). With S_K(x) = (1−x)/(K_A+K*) + x/(K_B+K*),
K(x) = 1/S_K − K*, dK/dx = −S_K'/S_K² with S_K' = 1/(K_B+K*) − 1/(K_A+K*);
Voigt: dK/dx = K_B − K_A; Reuss: dK/dx = K² (1/K_A − 1/K_B). Positivity: K, G > 0 on [0,1] for K_A, G_A, K_B, G_B > 0 (harmonic/arithmetic
means of positives minus a shift that is below the minimum of each a_i + shift). E = 9KG/(3K+G), ν = (3K−2G)/(2(3K+G)). Legacy path: Voigt
with phase A = r × B (same ν) equals SIMP with p = 1 and emin_ratio r.

## Test table

| ID | Hypothesis | Reference (independent of production code) | Threshold |
|---|---|---|---|
| MM1a | Identical phases (K,G) and pure endpoints (f = 0, 1): Voigt, Reuss, HS±, dilute, MT return the phase moduli; 2-D and 3-D | the input | rel 1e-13 |
| MM1b | Ordered pair: HS± equal the textbook two-phase form (K1 + f2/(1/(K2−K1) + 3f1/(3K1+4G1)) and the G form with ζ) | closed forms written in the test | rel 1e-12 |
| MM1c | Void: Voigt K = xK, Reuss = 0, HS− = 0 for f_void > 0; HS+ K and G equal `design.interp.hashin_shtrikman_bounds` (cross-check with existing code); HS+ K equals the hollow-sphere Lamé solution | Lamé 3-unknown linear solve in the test | 1e-14 / exact / 1e-12 / 1e-10 |
| MM1d | Bulk bound vs coated spheres: ordered pair, stiff coating = K_HS+, soft coating = K_HS−; crossed pair (K1 > K2, G1 < G2): both coating orders lie in [K_HS−, K_HS+] | Lamé solve in the test | 1e-10 equality; inside with margin ≥ −1e-12 K |
| MM2a | Eshelby sphere S from the code equals S built from S1111, S1122, S1212 (remembered); S_vol eigenvalue = 3K/(3K+4G) | components in the test | abs 1e-14 |
| MM2b | Tensor dilute and MT (matrix solves) equal the scalar closed forms of Section 2 for 5 pairs incl. void and ×1e3 inclusions; C_eff symmetric, `symmetry_class` isotropic, positive definite | closed forms in the test | rel 1e-12 |
| MM2c | MT − dilute increment ratio equals f a/(1+(1−f)a) for f = 1e-3, 5e-4 (K and G), and is O(f) | closed form | rel 1e-9 |
| MM2d | MT = HS+ (stiff matrix, soft spheres) and MT = HS− (soft matrix, stiff spheres), K and G; MT(stiff matrix) differs from HS− by > 1 % at f = 0.3 (brief's wording false) | HS from `hs_bounds` (own implementation, Section 1 vs Section 2 identity) and the textbook form | rel 1e-12; > 1e-2 |
| MM2e | MT stays inside [HS−, HS+] for contrast E_i/E_m ∈ {1e-9, 1e-3, 1e3}, f = 0.3 | HS bounds | margin ≥ −1e-12 |
| MM3 | Fractions: hand numbers of Section 4, round trip, density; invalid input refused (sum ≠ 1, negative, NaN, ρ ≤ 0 in the inverse, void in the inverse) | hand arithmetic | 1e-14 |
| MM4a | Layered periodic cell (2-D plane stress, 3 layers values E = 1, 7, 3, same ν, normal to y): `homogenize_elasticity` equals the exact layered average | Backus form, dense numpy written in the test | rel 1e-9 |
| MM4b | Same cell and random two-phase cells (24 × 24, E ∈ {1, 10}, exactly 50 %, seeds 0–4): C_R ≤ C_eff ≤ C_V in the Loewner order | Voigt/Reuss from the phase matrices | min eigenvalue ≥ −1e-10 max abs C |
| MM4c | Random cells: κ_cell = (C11+C22+2C12)/4 and μ_cell = (C11+C22−2C12+4C33)/8 (isotropic parts; equal κ, μ for an isotropic C) lie in the 2-D HS bounds of the two phases. **Measured, not a theorem** (finite sample, only statistically isotropic); a failure is a finding about the cell, not about the formula | `hs_bounds(dim=2)` | inside, no tolerance |
| E7 | 200 random admissible inputs (2–3 phases, 2-D/3-D, log-uniform K, G in [1e-3, 1e3], ≥ 20 crossed): K_R ≤ K_HS− ≤ K_HS+ ≤ K_V and G likewise; invalid fractions and phases rejected (ValueError) | ordering of Section 1 | rel 1e-12 |
| MMI1 | SIMP p_min and RAMP q_min of Section 3 equal the closed forms for 2-D ν ∈ {0, 0.1, 0.3, 1/3, 0.45} and 3-D ν ∈ {0, 0.2, 0.3, 0.45}; p = p_min + 0.02 has no violation on the grid, p = p_min − 0.05 violates (near x = 1); 2-D p_min ≥ 3, = 3 at ν = 1/3; agrees with `design.interp.simp_admissible_penal` (cross-check) | closed forms | rel 1e-12; grid margin ≥ −1e-12 |
| MMI2 | Two-solid interpolation: endpoints exact, K,G,E > 0 on a grid (also crossed pair), ordering Reuss ≤ HS− ≤ HS+ ≤ Voigt, dK/dx, dG/dx equal analytic Voigt/Reuss derivatives and a central difference (h = 1e-6) for all four schemes; Voigt with equal ν equals `SIMP(p=1, emin_ratio=r)` | closed forms, finite difference | endpoints 1e-14; derivative 1e-6 rel of the scale |

Run: `python -m pytest tests/toporlogy/test_micromechanics.py -q -p no:randomly`. First results are appended below.

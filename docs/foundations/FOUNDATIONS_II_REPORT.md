# Foundations II — integration report (WP24 draft)

Branch `foundations/round-3` from `f1ff694` (main after PR #23). Every number below is copied from a
preregistration record in `docs/foundations/prereg/` (first-result sections) or a log in `docs/foundations/logs/`;
the record is named in each row. Synthetic parameters throughout; no real material is claimed. Live tracker:
`STATUS.md` / `status.json`.

## What was built and verified

| Slice | Content | Verified scope (record) | Failed preregistered checks |
|---|---|---|---|
| P01 contract | `TimeStep`, `MaterialRequest.step`, history evaluate helper, timed/restartable material-point driver; FE refuses physical models | CT1–CT8 (`P01_contract.md`) | — |
| E04 Poisson | `poisson_uniaxial`; `effective_poisson` needs `convention` | PO1–PO6; historical auxetic "ν12 = −0.95" is ν21, ν12 = −0.84 (`E04_poisson.md`) | — |
| L solver | backward/forward error report, inertia, Krylov with structural checks, gauge solve, refinement, transpose solve | L1–L5 (`L_solver.md`) | L2c: SPD path does not detect an indefinite matrix (answer residual-checked) |
| P properties | E(T)/ν(T)/K(T)/G(T) from two functions, validity box, ε_th = ∫α dT, thermoelastic point | PP1–PP17 (`P_properties.md`) | — |
| V viscoelastic | Maxwell/KV/SLS/Burgers, 3-D generalized Maxwell (independent K, G spectra), WLF/Arrhenius, DMA | V1–V10, C1 (`V_viscoelastic.md`) | — |
| C creep | Norton, Norton–Bailey, Arrhenius, adaptive long holds | C2–C7 (`C_creep.md`) | C3g (n = 20 relaxation prediction), C6a (global error of local step control) |
| R rate | physical J2, Perzyna, Eyring, Drucker–Prager | R1–R7 (`R_rate.md`) | R3a (1-ulp steady state) |
| T spectral | f(A) with Fréchet derivative/JVP/VJP at repeated eigenvalues; VRH, A_U, Zener | TS1–TS9, MO1–4, E5 (`T_spectral.md`) | — |
| TK tensor | cofactor/det derivative without inverse; rotation log near π | TK1–TK5; **fixed an O(π − θ) axis error** of the existing near-π branch (`TK_cofactor_rotation.md`) | — |
| TS cure | n-th order/Kamal–Sourour (bounded implicit), DiBenedetto, reaction heat, shrinkage, Avrami/Nakamura | TS1–TS6, TP3–TP4 (`TS_cure.md`) | — |
| MM micromechanics | Voigt/Reuss, HS, Eshelby sphere (from memory, cross-checked), Mori–Tanaka, SIMP/RAMP HS admissibility | MM1–MM4, E7, MMI1–2 (`MM_micromechanics.md`) | wording: MT with a stiff matrix is the HS **upper** bound |
| D identification | Prony (NNLS, free times), Norton (log-linear, nonlinear), identifiability SVD, data records with units and validity box | D4–D7; D8 NOT VERIFIED (`D_identification.md`) | D5a covariance at κ = 8e8 |
| H history FE | small-strain history solid on a general mesh, commit-after-convergence, rollback, reactions | H0, H1, H3–H5, H6a–e (`H_history.md`, `H6_physical_models_fe.md`) | H2 (1-D series reference wrong in 3-D) |
| X mixed | K-based u–p (`MixedKG`, valid at ν = 0), inf–sup, locking, spurious modes | X1–X4 (`X_mixed.md`); **λ-based solve now refuses incompatible incompressible data** (`XF_mixed_compatibility.md`) | X2e plain-MINRES control |
| N nonlinear | objectivity, plane stress, patches, truss, Hencky via spectral log | N1–N5 (`N_nonlinear.md`); R07 still present (measured) | — |

## Optimisation and robustness

| Study | Result (record) |
|---|---|
| VO: viscoelastic topology optimisation, exact discrete history adjoint | forward vs history FE 6e-14; gradient vs FD 5.5e-10; proportional spectra ⇒ ∇J = (J/c_el)∇c_el to 8.5e-14, non-proportional breaks it (0.146); detached history wrong by 378 % (`VO_viscoelastic_topology.md`) |
| VO6 service time vs instantaneous design | at convergence 0.20 % gain at t_s, 0.18 % loss at 0⁺ — a real but small trade-off; the 40-iteration difference was non-convergence (post hoc) |
| VR min–max over service times, one constant load | VR1 failed: reduces to the last time; min–max 0.28 % worse (`VR_viscoelastic_robust.md`) |
| VU robust (mean + 2 std over 9 spectrum samples) | weak pass; the uncertainty (±8.6 %) is nearly design-independent (`VU_viscoelastic_uncertain.md`) |
| KS KKT-residual stop | KS2/KS3 failed: KKT-stationary designs still differ by 0.26–0.33 %: different local optima, not the stop rule; warm start from the nominal design improves it by 0.015 % (`KS_kkt_stop.md`) |
| VM min–max over a two-load history | min–max design J_A = 3.68e-5, J_B = 3.67e-5 (balanced to 0.075 %); single-phase designs reach 903 and 2008 on the other phase (`VM_minmax_switching.md`) |

Conclusion for the optimisation side: the gradients and formulations are exact and verified; the benefit of
time-aware, robust or min–max design depends on the problem — negligible for one constant load on this material,
decisive when the critical time depends on the design. Design comparisons below ~0.5 % need multi-start.

## Regression evidence

* Baseline marker at f1ff694: 8 passed (bitwise); collection 2584 tests (`logs/`).
* Fast suite at 79623a7: 2747 passed, 55 xfailed, 2 failed — both snapshot tests, artefacts of regenerating the
  snapshots in the working tree during the run (pass after regeneration). Final clean-worktree run: see below.
* ruff and mypy clean at each integration commit (mypy: 170 files).

## Not done / open

YAML and `Study` integration of the physical history models and of the viscoelastic optimisation; FE coupling of
creep / plasticity with the design density (only linear viscoelasticity is design-coupled); WP07 external mesh I/O
(meshio absent), WP09 adaptivity (in progress at the time of writing), WP19 shell drilling (memory-blocked), WP05 L06
saddle-point preconditioners and benchmarks; real material data (D8, R12). Remote push and PR need the user's approval.

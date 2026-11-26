# Bilimsel kataloglar ve uyumluluk matrisi (üretilmiş dosya, elle düzenleme)

Şema sürümü 1.1.0. Kaynak: `src/toporlogy/configuration/catalog.py` (ve `_catalog_benchmarks.py`); yeniden üretim: `python -m toporlogy.configuration.catalog --write docs`. Bileşen adları registry'den, kaynak anahtarları `toporlogy.references`'tan gelir; `tests/toporlogy/test_catalog.py` ikisini ve bu dosyanın güncelliğini denetler.

Durumlar: **SUPPORTED** hızlı pakette bir referansa ya da sonlu farka karşı geçen test var; **EXPERIMENTAL** koşuyor, denetim kısmi; **UNSUPPORTED** reddediliyor (`enforced`: bileşen, katalog ya da ilk değerlendirme); **NOT VERIFIED** kurulum kabul ediyor, bilinen test yok. Registry'de olmak desteklenmek değildir.

## 1. Elemanlar

| Eleman | Boyut | Düğüm × SD | İnterpolasyon | Quadrature | Fizik | Kilitlenme | Bozulma | Kanıt |
|:---|---:|:---|:---|:---|:---|:---|:---|:---|
| Q4 | 2 | 4 × 2 | bilinear | 2 x 2 Gauss | elasticity, modal, nonlinear_elasticity, tensor_elasticity | shear-locks in bending (parasitic shear) | regular grid, undistorted unit square/cube elements only (no Jacobian varies) | test_fem_mms.py::test_manufactured_solution_converges_at_second_order<br>test_skfem_crosscheck.py::test_x1_stiffness<br>test_independent_vv.py::test_top88_fixed_design_objective_and_sensitivity |
| Q4IM | 2 | 4 × 2 | bilinear + 2 incompatible modes per component (Q6), statically condensed | 2 x 2 Gauss | elasticity, modal | removes bending shear locking on undistorted squares | regular grid, undistorted unit square/cube elements only (no Jacobian varies) | test_fem_mms_extended.py::test_m1_q4im_converges_at_second_order<br>test_buckling_physics.py::test_gradient_q4im_with_modes_in_kg |
| H8 | 3 | 8 × 3 | trilinear | 2 x 2 x 2 Gauss | elasticity, elasticity_gmg, modal, nonlinear_elasticity, tensor_elasticity | shear-locks in bending | regular grid, undistorted unit square/cube elements only (no Jacobian varies) | test_fem_mms.py::test_manufactured_solution_converges_at_second_order<br>test_regression.py::test_topo3d_box_no_projection<br>test_gmg.py::test_gmg_matches_direct_masked_domain |
| H8IM | 3 | 8 × 3 | trilinear + 9 Wilson incompatible modes, statically condensed | 2 x 2 x 2 Gauss | elasticity, elasticity_gmg, modal | removes bending shear locking on undistorted cubes | regular grid, undistorted unit square/cube elements only (no Jacobian varies) | test_regression.py::test_topo3d_h8im<br>test_buckling.py::test_euler_column_3d_h8im<br>test_gmg.py::test_gmg_matches_direct_masked_domain |
| T10 | 3 | 10 × 3 | quadratic | 4-point rule | post:reverify | none expected for quadratic tetrahedra (not measured here) | general gmsh meshes; inverted or degenerate Jacobians are refused | test_post_verify.py::test_t10_patch_uniaxial_tension<br>test_t10_convergence.py::test_mesh_convergence_needs_three_sizes |
| mesh:Q4 | 2 | 4 × 2 | bilinear (isoparametric) | 2 x 2 Gauss |  | shear-locks on elongated elements (R = 0.29 at aspect 5); volumetric locking in plane strain (L2 error 0.94 at nu = 0.4999, n = 16) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured | test_mesh_elements.py::test_e4_completeness<br>test_mesh_assembly.py::test_g2_affine_patch_test<br>test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution<br>test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh |
| mesh:Q9 | 2 | 9 × 2 | biquadratic Lagrange (isoparametric) | 3 x 3 Gauss |  | no shear locking measured (R >= 0.9997, aspect up to 5); partial volumetric locking (L2 error 0.013 at nu = 0.4999 vs 3.5e-4 at 0.3) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured | test_mesh_elements.py::test_e4_completeness<br>test_mesh_assembly.py::test_g2_affine_patch_test<br>test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution<br>test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh |
| mesh:T3 | 2 | 3 × 2 | linear | centroid (exact for B^T D B) |  | locks in bending and volumetrically (L2 error 0.97 at nu = 0.4999) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured | test_mesh_elements.py::test_e4_completeness<br>test_mesh_assembly.py::test_g2_affine_patch_test<br>test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution<br>test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh |
| mesh:T6 | 2 | 6 × 2 | quadratic | 3-point (degree 2) |  | partial volumetric locking (L2 error 0.018 at nu = 0.4999) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured | test_mesh_elements.py::test_e4_completeness<br>test_mesh_assembly.py::test_g2_affine_patch_test<br>test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution<br>test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh |
| mesh:H8 | 3 | 8 × 3 | trilinear (isoparametric) | 2 x 2 x 2 Gauss |  | shear-locks on elongated elements (R = 0.24 at aspect 5); volumetric locking | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured | test_mesh_elements.py::test_e4_completeness<br>test_mesh_assembly.py::test_g2_affine_patch_test<br>test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution<br>test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh |
| mesh:T4 | 3 | 4 × 3 | linear | centroid (exact for B^T D B) |  | locks volumetrically (L2 error 0.996 at nu = 0.4999, n = 6) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured | test_mesh_elements.py::test_e4_completeness<br>test_mesh_assembly.py::test_g2_affine_patch_test<br>test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution<br>test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh |
| mesh:T10 | 3 | 10 × 3 | quadratic (gmsh node order) | 4-point Keast (degree 2) |  | curved T10 fails the patch test with the 4-point rule (1.7e-4); a degree-3 rule passes | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured | test_mesh_elements.py::test_e4_completeness<br>test_mesh_assembly.py::test_g2_affine_patch_test<br>test_mesh_verification.py::test_v2_scikit_fem_stiffness_load_and_solution<br>test_mesh_verification.py::test_v3_mms_convergence_on_distorted_mesh |
| mesh:TH-T6T3 | 2 | 6 × 2 | quadratic displacement + continuous linear pressure (Taylor-Hood u-p, sigma = 2 mu eps - p I, p = -lambda div u) | 3-point (degree 2) |  | locking-free near and at nu = 0.5 (L1c, L1e); inf-sup constant bounded under refinement (L2) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured; plane strain and 3-D only (plane stress refused), lambda = 0 refused | test_mesh_locking.py::test_l1c_taylor_hood_does_not_lock<br>test_mesh_locking.py::test_l1e_taylor_hood_converges_near_and_at_incompressibility<br>test_mesh_locking.py::test_l2_taylor_hood_inf_sup_is_bounded |
| mesh:TH-Q9Q4 | 2 | 9 × 2 | biquadratic displacement + continuous linear pressure (Taylor-Hood u-p, sigma = 2 mu eps - p I, p = -lambda div u) | 3 x 3 Gauss |  | locking-free near and at nu = 0.5 (L1c, L1e); inf-sup constant bounded under refinement (L2) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured; plane strain and 3-D only (plane stress refused), lambda = 0 refused | test_mesh_locking.py::test_l1c_taylor_hood_does_not_lock<br>test_mesh_locking.py::test_l1e_taylor_hood_converges_near_and_at_incompressibility<br>test_mesh_locking.py::test_l2_taylor_hood_inf_sup_is_bounded |
| mesh:TH-T10T4 | 3 | 10 × 3 | quadratic displacement + continuous linear pressure (Taylor-Hood u-p, sigma = 2 mu eps - p I, p = -lambda div u) | 4-point Keast (degree 2) |  | locking-free near and at nu = 0.5 (L1c, L1e); inf-sup constant bounded under refinement (L2) | general meshes (toporlogy.fem.mesh); signed detJ checked at every quadrature point and node, inverted or degenerate cells refused; distorted-mesh patch test and MMS rates measured; plane strain and 3-D only (plane stress refused), lambda = 0 refused | test_mesh_locking.py::test_l1c_taylor_hood_does_not_lock<br>test_mesh_locking.py::test_l1e_taylor_hood_converges_near_and_at_incompressibility<br>test_mesh_locking.py::test_l2_taylor_hood_inf_sup_is_bounded |

## 2. Fizik ve çözücüler

| Fizik | Rezidü | Matris | Çözüm | Cihaz | Boyut | Eleman | Mesnet | Yakınsama | Tasarım girişi |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| elasticity | K(E) u = f (linear, small strain) | symmetric positive definite on the free DOFs (refuses floating parts) | direct (solver: direct) | cpu | 2, 3 | Q4, Q4IM, H8, H8IM | fixed DOFs; oblique rollers and springs (ConstrainedElasticity); per-case supports (PerCaseSupports) | direct solve; relative residual <= 1e-6 or backward error <= 1e-14 accepted, else SolveResidualError | E_active (n_active,) scales one unit-modulus D per element (D may differ per element) |
| elasticity_gmg | K(E) u = f (linear, small strain) | symmetric positive definite | geometric multigrid preconditioned CG; direct below direct_below DOFs | cpu, cuda | 3 | H8, H8IM | axis-aligned fixed DOFs, one support set | PCG to tol (default 1e-8); fail-closed without fallback, logged fallback to the direct solver | E_active scales ONE unit D (per-element D refused) |
| modal | K(E) phi = lambda M(x) phi (generalised eigenproblem) | K SPD on the free DOFs, M SPD; repeated eigenvalues detected (KS aggregation) | shift-invert eigensolver (sigma) with the direct solver | cpu | 2, 3 | Q4, Q4IM, H8, H8IM | axis-aligned fixed DOFs, one support set; no static state u | eigensolver tolerance (not recorded per run) | E_active scales K; mass from the density (linear or two-range) |
| nonlinear_elasticity | r(u; E) = f_int(u) - f = 0 (total Lagrangian, SVK or neo-Hooke; API: small-strain models of core.constitutive incl. J2 plasticity, no sensitivity) | symmetric; positive definite only on the stable branch | general Newton (solvers.newton) with load steps; arc length / displacement control through continuation_problem (API); one adjoint solve with K_T per case | cpu | 2, 3 | Q4, H8 | axis-aligned fixed DOFs, one support set | Newton residual and error contraction; NonlinearConvergenceError with the history | E_active through an energy interpolation; general D with material svk only |
| tensor_elasticity | K(C) u = f (linear, small strain) | symmetric positive definite on the free DOFs (every C_e a convex combination of SPD tensors) | direct (solver: direct) | cpu | 2, 3 | Q4, H8 | axis-aligned fixed DOFs, one support set | direct solve; relative residual <= 1e-6 or backward error <= 1e-14 accepted, else SolveResidualError | per-element elasticity tensor C_e (MaterialField of interpolation tensor_phases); the scalar grid the study passes is only checked against the field |

| Çözücü | Registry'de | Yöntem | Arka uçlar | Durum | Not |
|:---|:---|:---|:---|:---|:---|
| direct | evet | sparse Cholesky / LU of the upper triangle | SuperLU (scipy), PARDISO (pypardiso) | SUPPORTED | SuperLU always available; PARDISO optional (fast extra), VERIFIED only where the pardiso tests ran (backends.yml) |
| gmg | hayır | geometric multigrid PCG (inside physics elasticity_gmg) | torch CPU, torch CUDA | EXPERIMENTAL | CPU path tested in the fast suite; CUDA path NOT VERIFIED without a GPU |

## 3. Benchmark kataloğu

| Kimlik | Tür | Test | Nicelik | Referans | Tolerans | Süre | Düzey |
|:---|:---|:---|:---|:---|:---|:---|:---|
| mms_q4_h8 | mms | test_fem_mms.py::test_manufactured_solution_converges_at_second_order | nodal error, observed order (Q4, H8, H8IM) | manufactured solution (docs/gelistirme_plani.md V1) | rates in [1.8, 2.25]; finest e < 1e-3 / 1e-2 / 1.2e-2 | fast | solution verification |
| mms_wrong_constant | mms | test_fem_mms.py::test_manufactured_solution_detects_a_wrong_material_constant | order with a wrong material constant (negative control) | manufactured solution | last rate < 1.0, error > 5x | fast | solution verification |
| mms_q4im | mms | test_fem_mms_extended.py::test_m1_q4im_converges_at_second_order | Q4IM nodal error order | manufactured solution (dogrulama_turu M1) | rates in (1.8, 2.25); finest <= 1e-3 | fast | solution verification |
| mms_orientation_field | mms | test_fem_mms_extended.py::test_m2_orientation_field_converges_at_second_order | orthotropic orientation field order | manufactured solution (M2) | rates in (1.8, 2.25); finest <= 5e-3 / 5e-2 | fast | solution verification |
| mms_heat | mms | test_fem_mms_extended.py::test_m3_heat_conduction_converges_at_second_order | conduction order | manufactured solution (M3) | rates in (1.8, 2.25) | fast | solution verification |
| mms_nonlinear | mms | test_fem_mms_extended.py::test_m4_nonlinear_manufactured_solution_converges_at_second_order | SVK / neo-Hooke order | manufactured solution (M4) | rates in (1.8, 2.25) | fast (3-D cases slow) | solution verification |
| skfem_stiffness | external_solver | test_skfem_crosscheck.py::test_x1_stiffness | stiffness matrix | scikit-fem | relative <= 1e-12 | fast | code verification |
| skfem_mass | external_solver | test_skfem_crosscheck.py::test_x2_mass | consistent mass | scikit-fem | relative <= 1e-12 | fast | code verification |
| skfem_conduction | external_solver | test_skfem_crosscheck.py::test_x3_conduction | conduction matrix | scikit-fem | relative <= 1e-12 | fast | code verification |
| skfem_cantilever | external_solver | test_skfem_crosscheck.py::test_x4_cantilever_solution | cantilever displacement | scikit-fem | relative <= 1e-10 | fast | code verification |
| top88_fixed_design | internal_oracle | test_independent_vv.py::test_top88_fixed_design_objective_and_sensitivity | objective, sensitivity | published_codes.py (Top88 port) | c 1e-9, dc 1e-8 | fast | code verification |
| top88_history | internal_oracle | test_independent_vv.py::test_top88_history_first_50_iterations | compliance history, 50 iterations | published_codes.py (Top88 port) | c 1e-6, x 1e-5 | fast | code verification |
| top3d_fixed_design | internal_oracle | test_independent_vv.py::test_top3d_fixed_design_objective_and_sensitivity | objective, sensitivity | published_codes.py (Top3D port) | c 1e-9, dc 1e-8 | fast | code verification |
| topo3d_regression | internal_oracle | test_regression.py::test_topo3d_box_no_projection | compliance history, density | topo_oracles.topo3d (frozen driver) | c 1e-6, x 1e-6 | fast | code verification |
| cantilever_exact | analytic | test_independent_vv.py::test_cantilever_q4_converges_to_the_exact_solution | compliance bound and tip error order | exact polynomial cantilever solution | order >= 1.8; tip <= 2e-2 | fast | solution verification |
| slip_square_modes | analytic | test_independent_vv.py::test_slip_square_q4_eigenvalues_converge_with_repeated_pairs | eigenvalues with repeated pairs | analytic slip-wall modes | order >= 1.8; gaps <= 1e-8 | fast | solution verification |
| independent_K | independent_same_author | test_independent_matrices.py::test_k1_k2_reduced_stiffness | reduced stiffness (8 element/material variants) | _independent_fe.py | relative <= 1e-12 | fast | code verification |
| independent_buckling | independent_same_author | test_independent_buckling.py::test_load_factors_equal_the_dense_independent_solution | first six load factors | _independent_fe.py (dense eigh) | |dmu| <= 1e-9 |mu1| | fast | code verification |
| independent_stress | independent_same_author | test_independent_stress.py::test_s2_von_mises_index | von Mises index | _independent_fe.py | relative <= 1e-12 | fast | code verification |
| timoshenko_tip | analytic | test_timoshenko_cantilever.py::test_h4_tip_deflection | tip deflection | timoshenko_goodier_1970 | Q4 <= 2e-2, Q4IM <= 2e-3 | fast | solution verification |
| timoshenko_order_q4 | analytic | test_timoshenko_cantilever.py::test_h2_q4_converges_at_second_order | nodal error order (strict xfail) | timoshenko_goodier_1970 | rates in [1.8, 2.25] | fast | solution verification |
| fs2019_q4_column | published | test_fs2019_column.py::test_published_q4_error_20x2 | buckling error Q4 20x2 | ferrari_sigmund_2019 (Fig. 1 table) | |e - 0.119| <= 5e-4 | fast | code verification |
| fs2019_q6_column | published | test_fs2019_column.py::test_published_q6_error_20x2 | buckling error Q4IM vs published Q6 (strict xfail) | ferrari_sigmund_2019 | |e + 6.6e-3| <= 5e-5 | fast | code verification |
| euler_column_orthotropic | analytic | test_buckling_physics.py::test_orthotropic_column_buckles_with_its_axial_modulus | column buckling load, refinement ratios | Euler column | err <= 5e-3; ratios >= 2.5 | fast | solution verification |
| euler_bernoulli_modal_2d | analytic | test_modal.py::test_cantilever_vs_euler_bernoulli_2d | first frequency ratio | Euler-Bernoulli beam | 0.98 < r < 1.0, monotone | fast | solution verification |
| heat_series_bar | analytic | test_heat.py::test_bar_series_resistance_exact | end temperature | series resistance | relative <= 1e-10 | fast | code verification |
| nl_uniaxial | analytic | test_nonlinear.py::test_v2_uniaxial_tension_closed_form | stretches under dead load | closed form | <= 1e-9 | fast | code verification |
| nl_elastica | independent_same_author | test_nonlinear.py::test_v5_large_deflection_cantilever_elastica | tip displacement | _elastica.py | v <= 5e-3, u <= 1e-2 at n = 16 | slow | solution verification |
| deng_pnorm_after_mma | published | test_published_values.py::test_deng_cantilever_published_value_is_after_the_first_mma_step | p-norm after one MMA step (post hoc) | Deng, Vulimiri & To 2021 §6.2 | relative <= 5e-3 | fast | code verification |
| topbuck250_wall | published | test_published_values.py::test_topbuck250_wall_solid_blf_against_published_value | solid wall load factor (strict xfail) | ferrari_sigmund_guest_2021 §5.2 | |lambda - 3.15| <= 0.0051 | slow | code verification |
| t10_mesh_convergence | mms | test_t10_convergence.py::test_the_same_cantilever_converges_on_fine_meshes_and_not_on_coarse_ones | three-grid GCI verdict | celik_2008 | CONVERGED on fine, NOT CONVERGED on coarse | slow | solution verification |
| taylor_t1 | gradient | test_taylor_remainder.py::test_t1_taylor_orders | Taylor remainder order of 11 responses | docs/olcumler/dogrulama_turu.md T | two consecutive p in (1.8, 2.2) | fast | code verification |
| gradients_compliance | gradient | test_gradients.py::test_compliance_single_case | compliance gradient | five-point finite difference | relative <= 1e-6 | fast | code verification |
| constitutive_ad_oracle | internal_oracle | test_constitutive.py::test_p1_energy_stress_tangent_match_the_ad_oracle | SVK / neo-Hooke energy, PK2 stress, tangent (3-D, plane strain, plane stress) | torch autograd of the energy (core.constitutive.energy_oracle) | psi, S <= 1e-11; C <= 1e-10 relative | fast | code verification |
| j2_uniaxial_cyclic | analytic | test_constitutive.py::test_j2_cyclic_uniaxial_closed_form_and_elastic_unloading | J2 uniaxial stress, cyclic, isotropic hardening | closed form (1-D isotropic hardening, leg by leg) | stress <= 1e-10 sigma_y; alpha <= 1e-10 eps_y | fast | code verification |
| j2_consistent_tangent | gradient | test_constitutive.py::test_j4_consistent_tangent_matches_finite_difference | J2 algorithmic tangent (linear and Voce hardening) | central finite difference of the return map | relative <= 1e-6 | fast | code verification |
| j2_fe_homogeneous_bar | internal_oracle | test_nonlinear_solvers.py::test_j7_homogeneous_bar_matches_the_material_point | FE J2 bar (Q4 plane strain, H8): loading and unloading strain | mixed-control material point driver (core.constitutive.drive_material_point) | relative <= 1e-9 | fast | code verification |
| truss_snap_through | analytic | test_nonlinear_solvers.py::test_a1_a3_arc_length_traces_the_truss_snap_through | arc-length path of the shallow two-bar truss through both limit points | closed form P(w) (tests/toporlogy/_truss.py) | |lambda - P(w)/P_ref| <= 1e-9; det K sign 100 % | fast | code verification |
| fe_arc_length_vs_load_control | internal_oracle | test_nonlinear_solvers.py::test_a5_fe_arc_length_matches_load_control | arc-length path of a stocky cantilever (SVK) | load control at the same load factors | relative <= 1e-7 | fast | code verification |
| end_strain_energy_taylor | gradient | test_nonlinear_solvers.py::test_t1_t2_end_strain_energy_taylor_remainder | end strain energy gradient (fixed cutoff) | Taylor remainder order | two consecutive orders in [1.8, 2.2]; x1.01 control <= 1.3 | fast | code verification |

Deney verisiyle doğrulama (validation) yok: her satır code/solution verification. Aynı yazarın ikinci uygulaması (`independent_same_author`) dış ekip doğrulaması sayılmaz.

## 4. Malzeme veri kataloğu

Yalnız sentetik kayıt var; kaynağı, sayfası ve lisansı bu çalışmada okunmamış gerçek malzeme değeri girilmez. YAML: `material: {catalog: <kimlik>}` (SI kayıt `units` bölümü ister ve ona çevrilir; `docs/olcumler/malzeme_katalog_onkayit.md`).

| Kimlik | Tür | Aile | Değerler | Birim | Kaynak | Lisans | Not |
|:---|:---|:---|:---|:---|:---|:---|:---|
| synthetic_unit_isotropic | synthetic | isotropic | E0 = 1, nu = 0.3, rho = 1 | dimensionless | schema default (configuration/schema.py _MATERIAL) | — | not a real material; the default of every config |
| synthetic_auxetic_isotropic | synthetic | isotropic | E0 = 1, nu = -0.5, rho = 1 | dimensionless | chosen for tests (docs/olcumler/malzeme_katalog_onkayit.md C4) | — | not a real material: a negative Poisson ratio inside (-1, 1/2) |
| synthetic_orthotropic_ply | synthetic | orthotropic | E1 = 40, E2 = 9, E3 = 9, G12 = 4, G13 = 4, G23 = 3.2, nu12 = 0.28, nu13 = 0.28, nu23 = 0.4, rho = 1 | dimensionless | the README physics example (round numbers) | — | not a real material: ratios of the order of a fibre ply |
| synthetic_orthotropic_high_nu23 | synthetic | orthotropic | E1 = 10, E2 = 1, E3 = 1, G12 = 0.5, G13 = 0.5, G23 = 0.3, nu12 = 0.05, nu13 = 0.05, nu23 = 0.65, rho = 1 | dimensionless | chosen for tests (docs/olcumler/malzeme_katalog_onkayit.md C4) | — | not a real material: positive definite with nu23 > 1/2 (anisotropic bounds differ from isotropic) |
| synthetic_si_isotropic | synthetic | isotropic | E0 = 2e+11, nu = 0.3, rho = 8000 | SI | chosen for tests (docs/olcumler/malzeme_katalog_onkayit.md C6) | — | not a real material: round numbers of the order of a steel, in Pa and kg/m^3 |

## 5. Uyumluluk: fizik × yanıt

| Yanıt | elasticity | elasticity_gmg | modal | nonlinear_elasticity | tensor_elasticity |
|:---|:---|:---|:---|:---|:---|
| buckling | SUPPORTED | UNSUPPORTED (catalog) | UNSUPPORTED (catalog) | UNSUPPORTED (catalog) | NOT VERIFIED |
| centre_of_mass | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| compliance | SUPPORTED | SUPPORTED | UNSUPPORTED (catalog) | EXPERIMENTAL | SUPPORTED |
| conduction_connectivity | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| cost | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| coupling_compliance | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| dilated_volume | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| displacement | SUPPORTED | NOT VERIFIED | UNSUPPORTED (catalog) | SUPPORTED | NOT VERIFIED |
| eigenvalue | UNSUPPORTED (component) | UNSUPPORTED (component) | SUPPORTED | UNSUPPORTED (component) | NOT VERIFIED |
| end_compliance | UNSUPPORTED (component) | UNSUPPORTED (component) | UNSUPPORTED (component) | SUPPORTED | NOT VERIFIED |
| end_strain_energy | UNSUPPORTED (component) | UNSUPPORTED (component) | UNSUPPORTED (component) | SUPPORTED | UNSUPPORTED (component) |
| frequency | UNSUPPORTED (component) | UNSUPPORTED (component) | SUPPORTED | UNSUPPORTED (component) | NOT VERIFIED |
| grey_level | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| inertia | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| local_volume | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| mass | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| maximize | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| min_eigenvalue_ks | UNSUPPORTED (component) | UNSUPPORTED (component) | SUPPORTED | UNSUPPORTED (component) | NOT VERIFIED |
| min_frequency | UNSUPPORTED (component) | UNSUPPORTED (component) | SUPPORTED | UNSUPPORTED (component) | NOT VERIFIED |
| phase_volume | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| robust_compliance | SUPPORTED | NOT VERIFIED | UNSUPPORTED (catalog) | NOT VERIFIED | NOT VERIFIED |
| stress | SUPPORTED | NOT VERIFIED | UNSUPPORTED (catalog) | NOT VERIFIED | NOT VERIFIED |
| stress_augmented_lagrangian | SUPPORTED | NOT VERIFIED | UNSUPPORTED (catalog) | NOT VERIFIED | NOT VERIFIED |
| volume | SUPPORTED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| weighted_sum | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |

Gerekçe ve kanıt (yalnız boş olmayanlar):

- `buckling` × `elasticity`: SUPPORTED [test_buckling.py::test_gradient_buckling_ks; test_independent_buckling.py::test_load_factors_equal_the_dense_independent_solution]
- `buckling` × `elasticity_gmg`: UNSUPPORTED — buckling needs plain Elasticity (direct solver)
- `buckling` × `modal`: UNSUPPORTED — needs the static state u; physics modal solves an eigenproblem only
- `buckling` × `nonlinear_elasticity`: UNSUPPORTED — linear buckling about a linear prestress; not defined on the non-linear state
- `centre_of_mass` × `elasticity`: SUPPORTED — tested with physics elasticity [test_mass_properties.py::test_mp7_mp8_study_responses_value_and_gradient]
- `centre_of_mass` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `centre_of_mass` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `centre_of_mass` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `centre_of_mass` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `compliance` × `elasticity`: SUPPORTED [test_gradients.py::test_compliance_single_case; test_independent_vv.py::test_top88_fixed_design_objective_and_sensitivity]
- `compliance` × `elasticity_gmg`: SUPPORTED [test_gmg.py::test_gradients; test_gmg.py::test_study_regression_against_direct]
- `compliance` × `modal`: UNSUPPORTED — needs the static state u; physics modal solves an eigenproblem only
- `compliance` × `nonlinear_elasticity`: EXPERIMENTAL — same number as end_compliance by construction (EndCompliance subclasses Compliance); only end_compliance is tested [test_nonlinear.py::test_v6_end_compliance_gradient_finite_difference]
- `compliance` × `tensor_elasticity`: SUPPORTED [test_material_design.py::test_k8b_scalar_limit_matches_multiphase_simp; test_material_design.py::test_g1_compliance_gradient_finite_difference]
- `conduction_connectivity` × `elasticity`: SUPPORTED — tested with physics elasticity [test_connectivity.py::test_gradcheck_conduction_connectivity]
- `conduction_connectivity` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `conduction_connectivity` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `conduction_connectivity` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `conduction_connectivity` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `cost` × `elasticity`: SUPPORTED — tested with physics elasticity [test_multi_material.py::test_v4_gradients_finite_difference]
- `cost` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `cost` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `cost` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `cost` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `coupling_compliance` × `elasticity`: SUPPORTED — tested with physics elasticity [test_connectivity.py::test_gradcheck_coupling_compliance]
- `coupling_compliance` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `coupling_compliance` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `coupling_compliance` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `coupling_compliance` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `dilated_volume` × `elasticity`: SUPPORTED — tested with physics elasticity [test_robust.py::test_gradcheck_robust_compliance_and_dilated_volume]
- `dilated_volume` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `dilated_volume` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `dilated_volume` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `dilated_volume` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `displacement` × `elasticity`: SUPPORTED [test_gradients.py::test_non_self_adjoint_displacement; test_compose.py::test_displacement_value_and_gradient]
- `displacement` × `modal`: UNSUPPORTED — needs the static state u; physics modal solves an eigenproblem only
- `displacement` × `nonlinear_elasticity`: SUPPORTED [test_nonlinear.py::test_v5_large_deflection_cantilever_elastica; test_nonlinear.py::test_v6_end_compliance_gradient_finite_difference]
- `eigenvalue` × `elasticity`: UNSUPPORTED — needs physics modal
- `eigenvalue` × `elasticity_gmg`: UNSUPPORTED — needs physics modal
- `eigenvalue` × `modal`: SUPPORTED [test_modal.py::test_response_gradients; test_modal.py::test_eigenpairs_match_dense]
- `eigenvalue` × `nonlinear_elasticity`: UNSUPPORTED — needs physics modal
- `end_compliance` × `elasticity`: UNSUPPORTED — needs physics nonlinear_elasticity
- `end_compliance` × `elasticity_gmg`: UNSUPPORTED — needs physics nonlinear_elasticity
- `end_compliance` × `modal`: UNSUPPORTED — needs physics nonlinear_elasticity
- `end_compliance` × `nonlinear_elasticity`: SUPPORTED [test_nonlinear.py::test_v6_end_compliance_gradient_finite_difference; test_nonlinear_orientation.py::test_n4_end_compliance_gradient_with_a_field]
- `end_strain_energy` × `elasticity`: UNSUPPORTED — needs physics nonlinear_elasticity
- `end_strain_energy` × `elasticity_gmg`: UNSUPPORTED — needs physics nonlinear_elasticity
- `end_strain_energy` × `modal`: UNSUPPORTED — needs physics nonlinear_elasticity
- `end_strain_energy` × `nonlinear_elasticity`: SUPPORTED — hyperelastic materials only (svk, neo_hooke); a stateful law is refused [test_nonlinear_solvers.py::test_t1_t2_end_strain_energy_taylor_remainder]
- `end_strain_energy` × `tensor_elasticity`: UNSUPPORTED — needs physics nonlinear_elasticity
- `frequency` × `elasticity`: UNSUPPORTED — needs physics modal
- `frequency` × `elasticity_gmg`: UNSUPPORTED — needs physics modal
- `frequency` × `modal`: SUPPORTED [test_modal.py::test_response_gradients; test_modal.py::test_eigenpairs_match_dense]
- `frequency` × `nonlinear_elasticity`: UNSUPPORTED — needs physics modal
- `grey_level` × `elasticity`: SUPPORTED — tested with physics elasticity [test_lengthscale.py::test_gradcheck_local_volume_and_grey]
- `grey_level` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `grey_level` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `grey_level` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `grey_level` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `inertia` × `elasticity`: SUPPORTED — tested with physics elasticity [test_mass_properties.py::test_mp7_mp8_study_responses_value_and_gradient]
- `inertia` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `inertia` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `inertia` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `inertia` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `local_volume` × `elasticity`: SUPPORTED — tested with physics elasticity [test_lengthscale.py::test_gradcheck_local_volume_and_grey]
- `local_volume` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `local_volume` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `local_volume` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `local_volume` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `mass` × `elasticity`: SUPPORTED — tested with physics elasticity [test_multi_material.py::test_v4_gradients_finite_difference]
- `mass` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `mass` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `mass` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `mass` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `maximize` × `elasticity`: NOT VERIFIED — composite: the weakest status of its parts
- `maximize` × `elasticity_gmg`: NOT VERIFIED — composite: the weakest status of its parts
- `maximize` × `modal`: NOT VERIFIED — composite: the weakest status of its parts
- `maximize` × `nonlinear_elasticity`: NOT VERIFIED — composite: the weakest status of its parts
- `maximize` × `tensor_elasticity`: NOT VERIFIED — composite: the weakest status of its parts
- `min_eigenvalue_ks` × `elasticity`: UNSUPPORTED — needs physics modal
- `min_eigenvalue_ks` × `elasticity_gmg`: UNSUPPORTED — needs physics modal
- `min_eigenvalue_ks` × `modal`: SUPPORTED [test_modal.py::test_response_gradients; test_modal.py::test_eigenpairs_match_dense]
- `min_eigenvalue_ks` × `nonlinear_elasticity`: UNSUPPORTED — needs physics modal
- `min_frequency` × `elasticity`: UNSUPPORTED — needs physics modal
- `min_frequency` × `elasticity_gmg`: UNSUPPORTED — needs physics modal
- `min_frequency` × `modal`: SUPPORTED [test_modal.py::test_response_gradients; test_modal.py::test_eigenpairs_match_dense]
- `min_frequency` × `nonlinear_elasticity`: UNSUPPORTED — needs physics modal
- `phase_volume` × `elasticity`: SUPPORTED — tested with physics elasticity [test_multi_material.py::test_v4_gradients_finite_difference]
- `phase_volume` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `phase_volume` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `phase_volume` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `phase_volume` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `robust_compliance` × `elasticity`: SUPPORTED [test_robust.py::test_gradcheck_robust_compliance_and_dilated_volume]
- `robust_compliance` × `modal`: UNSUPPORTED — needs the static state u; physics modal solves an eigenproblem only
- `stress` × `elasticity`: SUPPORTED [test_stress.py::test_gradient_aggregated_stress; test_independent_stress.py::test_s2_von_mises_index]
- `stress` × `modal`: UNSUPPORTED — needs the static state u; physics modal solves an eigenproblem only
- `stress` × `nonlinear_elasticity`: NOT VERIFIED — builds; the small-strain stress of the stress response on a large-displacement state is not a verified measure (risk)
- `stress_augmented_lagrangian` × `elasticity`: SUPPORTED [test_stress.py::test_gradient_augmented_lagrangian]
- `stress_augmented_lagrangian` × `modal`: UNSUPPORTED — needs the static state u; physics modal solves an eigenproblem only
- `volume` × `elasticity`: SUPPORTED — tested with physics elasticity [test_gradients.py::test_volume_weighted_and_unweighted]
- `volume` × `elasticity_gmg`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `volume` × `modal`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `volume` × `nonlinear_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `volume` × `tensor_elasticity`: NOT VERIFIED — does not read the study's physics (read from the code); combination not tested
- `weighted_sum` × `elasticity`: NOT VERIFIED — composite: the weakest status of its parts
- `weighted_sum` × `elasticity_gmg`: NOT VERIFIED — composite: the weakest status of its parts
- `weighted_sum` × `modal`: NOT VERIFIED — composite: the weakest status of its parts
- `weighted_sum` × `nonlinear_elasticity`: NOT VERIFIED — composite: the weakest status of its parts
- `weighted_sum` × `tensor_elasticity`: NOT VERIFIED — composite: the weakest status of its parts

## 6. Diğer kurallar (malzeme, eleman, mesnet, faz, cihaz)

| Kural | Koşul | Durum | Uygulama | Gerekçe | Kanıt |
|:---|:---|:---|:---|:---|:---|
| gmg_2d | physics = elasticity_gmg, dim = 2 | UNSUPPORTED | component | elasticity_gmg is 3-D only | — |
| gmg_material_field | physics = elasticity_gmg, material = orthotropic_field | UNSUPPORTED | component | matrix-free kernels assume one element matrix | — |
| gmg_per_case_supports | physics = elasticity_gmg, supports = per_case | UNSUPPORTED | component | one support set only | — |
| gmg_oblique_supports | physics = elasticity_gmg, supports = oblique_or_spring | UNSUPPORTED | component | axis-aligned fixed DOFs only | — |
| modal_per_case_supports | physics = modal, supports = per_case | UNSUPPORTED | component | one support set only | — |
| nonlinear_per_case_supports | physics = nonlinear_elasticity, supports = per_case | UNSUPPORTED | component | one support set only | — |
| buckling_oblique_supports | response = buckling, supports = oblique_or_spring | UNSUPPORTED | catalog | LinearBuckling assumes axis-aligned fixed DOFs (ConstrainedElasticity gave 'wrong M dimensions' at the first evaluation before this rule) | — |
| buckling_per_case_supports | response = buckling, supports = per_case | UNSUPPORTED | catalog | LinearBuckling needs one support set | — |
| plane_strain_3d | material = plane_strain, dim = 3 | UNSUPPORTED | component | plane applies to 2-D only | — |
| hashin_shtrikman_orthotropic | interpolation = hashin_shtrikman, material = orthotropic | UNSUPPORTED | component | isotropic bound | — |
| phases_responses | material = phases, response = other_than_compliance_phase_volume_mass_cost | UNSUPPORTED | component | only Compliance, PhaseVolume, Mass, Cost verified with several phases | — |
| phases_oc | material = phases, optimizer = oc | UNSUPPORTED | component | OC assumes one positive-gradient constraint | — |
| phases_links | material = phases, link = other_than_embed_density_filter_projection_passive | UNSUPPORTED | component | only Embed, DensityFilter, HeavisideProjection, Passive verified | — |
| phases_elasticity | material = phases, physics = elasticity | SUPPORTED | — | — | test_multi_material.py::test_v4_gradients_finite_difference<br>test_multi_material.py::test_v3_two_material_bar_closed_form |
| phases_gmg | material = phases, physics = elasticity_gmg | NOT VERIFIED | — | — | — |
| phases_modal | material = phases, physics = modal | NOT VERIFIED | — | builds; the mass interpolation of several phases is not tested | — |
| phases_nonlinear | material = phases, physics = nonlinear_elasticity | NOT VERIFIED | — | builds with compliance; end_compliance is refused with phases | — |
| orthotropic_elasticity | material = orthotropic_field, physics = elasticity | SUPPORTED | — | — | test_fem_mms_extended.py::test_m2_orientation_field_converges_at_second_order<br>test_independent_matrices.py::test_k1_k2_reduced_stiffness |
| orthotropic_buckling | material = orthotropic_field, response = buckling | SUPPORTED | — | — | test_element_orientation.py::test_p3c_buckling_gradient_q4im<br>test_buckling_physics.py::test_orthotropic_column_buckles_with_its_axial_modulus |
| orthotropic_modal | material = orthotropic, physics = modal | NOT VERIFIED | — | builds; modal tests use isotropic material | — |
| orthotropic_gmg_uniform | material = orthotropic, physics = elasticity_gmg | NOT VERIFIED | — | one rotated D is accepted; not tested | — |
| orthotropic_nonlinear | material = orthotropic_field, physics = nonlinear_elasticity | SUPPORTED | — | material svk only (neo-Hooke with a general D is refused) | test_nonlinear_orientation.py::test_n3_consistent_tangent_with_a_field<br>test_nonlinear_orientation.py::test_n4_end_compliance_gradient_with_a_field |
| nonlinear_incompatible_modes | physics = nonlinear_elasticity, element = Q4IM|H8IM | UNSUPPORTED | component | incompatible-mode elements not implemented for the non-linear physics | — |
| plane_strain_elasticity | material = plane_strain, physics = elasticity | EXPERIMENTAL | — | D checked against D_plane_strain; no solution benchmark in plane strain | test_material_config.py::test_plane_strain |
| plane_strain_modal | material = plane_strain, physics = modal | NOT VERIFIED | — | — | — |
| finalize_2d | post = finalize, dim = 2 | UNSUPPORTED | component | finalize needs a 3-D domain | — |
| cuda_gmg | physics = elasticity_gmg, device = cuda | NOT VERIFIED | — | no GPU in CI or in this record | — |
| mesh_grid_equivalence | element = mesh:Q4|mesh:H8, mesh = from_domain | SUPPORTED | — | the general mesh of a Domain reproduces the grid path (K <= 1e-14, u <= 1e-12) | test_mesh_assembly.py::test_g1_from_domain_q4_equals_grid_path<br>test_mesh_assembly.py::test_g1_from_domain_h8_equals_grid_path |
| mesh_material_field | element = mesh:*, material = per_cell_D | SUPPORTED | — | — | test_mesh_assembly.py::test_g9_two_material_bar<br>test_mesh_assembly.py::test_g8_pattern_reuse |
| mesh_nearly_incompressible_displacement_only | element = mesh:Q4|mesh:T3|mesh:H8|mesh:T4, material = nu_near_0.5 | EXPERIMENTAL | — | runs, but locks (L2 error 0.94-0.996 at nu = 0.4999): use the Taylor-Hood elements | test_mesh_locking.py::test_l1b_displacement_elements_lock |
| mesh_incompressible_mixed | element = mesh:TH-*, material = nu_0.5 | SUPPORTED | — | lambda = inf exact; the zero-mean pressure gauge is added when the constant pressure is in the null space | test_mesh_locking.py::test_l1e_taylor_hood_converges_near_and_at_incompressibility |
| tensor_phases_elasticity | interpolation = tensor_phases, physics = tensor_elasticity | SUPPORTED | — | phases with their own tensors, rules simplex and recursive | test_material_design.py::test_k7_independent_assembly<br>test_material_design.py::test_k6_series_parallel_bar |
| angle_elasticity | link = embed.angle, physics = tensor_elasticity | SUPPORTED | — | density + fibre angle, 2-D, one phase | test_material_design.py::test_a5_off_axis_tension_and_gradient<br>test_material_design.py::test_a8_yaml_example_runs |
| tensor_responses | physics = tensor_elasticity, response = other_than_compliance_volume_phase_volume_mass_cost | UNSUPPORTED | catalog | only Compliance, Volume, PhaseVolume, Mass, Cost verified with tensor_elasticity | — |
| tensor_interpolation | physics = tensor_elasticity, interpolation = other_than_tensor_phases | UNSUPPORTED | catalog | tensor_elasticity reads the material field of interpolation tensor_phases | — |
| tensor_phases_physics | interpolation = tensor_phases, physics = other_than_tensor_elasticity | UNSUPPORTED | component | the scalar-modulus physics cannot take the element tensors | — |
| tensor_incompatible_modes | physics = tensor_elasticity, element = Q4IM|H8IM | UNSUPPORTED | component | condensed incompatible modes are not linear in D | — |
| tensor_per_case_supports | physics = tensor_elasticity, supports = per_case | UNSUPPORTED | component | one support set only | — |
| tensor_oblique_supports | physics = tensor_elasticity, supports = oblique_or_spring | UNSUPPORTED | component | axis-aligned fixed DOFs only | — |
| tensor_post | physics = tensor_elasticity, post = other_than_phase_report | UNSUPPORTED | catalog | the other post steps assume a scalar modulus or an isotropic material | — |
| angle_3d | link = embed.angle, dim = 3 | UNSUPPORTED | component | 3-D orientation is not implemented | — |
| angle_phases | link = embed.angle, interpolation = tensor_phases with several phases | UNSUPPORTED | component | Study takes several design fields only through the plain embed link | — |
| angle_interpolation | link = embed.angle, interpolation = other_than_tensor_phases | UNSUPPORTED | component | the angle turns the phase tensors of tensor_phases | — |
| angle_isotropic | link = embed.angle, material = isotropic phase | UNSUPPORTED | component | the material does not depend on the angle (zero gradient) | — |
| angle_oc | link = embed.angle, optimizer = oc | UNSUPPORTED | catalog | OC assumes a constraint gradient positive in every variable; the angle has none | — |

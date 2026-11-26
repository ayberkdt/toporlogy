# Risk kapsam matrisi: test katmanı × kütüphane alanı (üretilmiş dosya)

Kaynak: `tests/toporlogy/_vv_layers.py` (dosya → katman sınıflaması elle, dosya → alan içe aktarmalardan). Yeniden üretim: `python tests/toporlogy/_vv_layers.py --write`. Hücre: o katmandaki ve o alanı içe aktaran test dosyası sayısı / test fonksiyonu sayısı. Bu bir harita, başarı ölçütü değil: parametreli test bir sayılır, dolu hücre denetimin gücünü söylemez, boş hücre (—) o çiftte test olmadığını söyler.

| Katman | core | geometry | boundary | design | fem | solvers | optimization | optimizers | configuration | execution | post | references |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U unit / property | 14/194 | 3/30 | — | 7/102 | 9/117 | 3/43 | 15/191 | 3/63 | 3/52 | 2/14 | 4/44 | — |
| MP material point | 16/204 | — | — | 8/125 | 11/153 | 1/25 | 13/162 | 5/90 | 7/96 | 1/13 | 4/55 | — |
| EL element patch / quadrature | 7/80 | — | — | 1/10 | 10/109 | 1/14 | 4/52 | 1/14 | 1/17 | — | 3/40 | — |
| AS assembly / BC / reactions | 5/110 | 1/23 | 5/62 | 2/58 | 8/126 | 6/78 | 3/81 | 2/58 | 1/33 | — | 1/20 | — |
| AN analytical / MMS | 17/211 | — | — | 5/93 | 22/252 | 1/25 | 17/232 | 5/99 | 4/74 | 1/13 | 2/20 | — |
| IS independent solver / implementation | 8/95 | — | — | 2/58 | 11/134 | 2/37 | 5/83 | 2/58 | 2/47 | — | — | — |
| OG optimization / KKT / gradient | 14/178 | — | — | 7/114 | 16/205 | 5/73 | 24/257 | 15/195 | 7/103 | 2/30 | 2/30 | — |
| E2E end-to-end geometry | 2/38 | 5/39 | — | 2/13 | 2/38 | — | 7/61 | 1/21 | 4/38 | 1/10 | 10/67 | — |
| IC integration / compatibility | 5/51 | 2/34 | — | 4/64 | 5/58 | — | 6/83 | 3/56 | 14/153 | 9/85 | 1/8 | 1/11 |
| PF performance | — | — | 1/11 | — | 2/23 | 2/23 | — | — | — | — | — | — |

Boş hücreler (açık risk adayı; her biri bir risk değildir, örn. `references` için çoğu katman anlamsız):

- **U** (unit / property): boundary, references
- **MP** (material point): geometry, boundary, references
- **EL** (element patch / quadrature): geometry, boundary, execution, references
- **AS** (assembly / BC / reactions): execution, references
- **AN** (analytical / MMS): geometry, boundary, references
- **IS** (independent solver / implementation): geometry, boundary, execution, post, references
- **OG** (optimization / KKT / gradient): geometry, boundary, references
- **E2E** (end-to-end geometry): boundary, solvers, references
- **IC** (integration / compatibility): boundary, solvers
- **PF** (performance): core, geometry, design, optimization, optimizers, configuration, execution, post, references

Katman sınıflaması (dosya → katmanlar):

- `test_aggregates.py`: U, OG
- `test_anisotropy.py`: U, MP
- `test_api_surface.py`: IC
- `test_architecture.py`: IC
- `test_auxetic_cell.py`: OG, E2E
- `test_baseline.py`: IC
- `test_baseline_policy.py`: IC
- `test_bc.py`: AS
- `test_benchmark.py`: OG
- `test_buckling.py`: EL, AN, OG
- `test_buckling_physics.py`: AN, OG
- `test_catalog.py`: IC
- `test_catalog_ledger.py`: IC
- `test_cli.py`: IC
- `test_closed_forms.py`: U, AN
- `test_compatibility.py`: IC
- `test_compose.py`: U, OG
- `test_config.py`: IC
- `test_connectivity.py`: U, OG, E2E
- `test_constitutive.py`: U, MP, AN
- `test_core.py`: U, EL, AS
- `test_deriving_methods.py`: OG, IC
- `test_element_orientation.py`: MP, EL, OG, E2E
- `test_fault_injection.py`: IC
- `test_fem_mms.py`: AN
- `test_fem_mms_extended.py`: AN
- `test_frame_invariance.py`: U, MP
- `test_fs2019_column.py`: AN, IS
- `test_gcmma.py`: OG
- `test_geometry.py`: U, E2E
- `test_gmg.py`: IS, OG, PF
- `test_gradients.py`: OG
- `test_heat.py`: EL, AN, OG
- `test_homogenization.py`: AN, OG
- `test_identification.py`: MP, OG
- `test_implicit.py`: U, OG
- `test_independent_buckling.py`: IS
- `test_independent_matrices.py`: IS, EL
- `test_independent_stress.py`: IS, MP
- `test_independent_vv.py`: IS, AN
- `test_invariants.py`: U
- `test_kinematic_contract.py`: AS
- `test_kkt_estimate.py`: OG
- `test_laminate.py`: U, MP, AN
- `test_lengthscale.py`: U, OG
- `test_linear_elastic.py`: U, MP
- `test_manufacturing.py`: U, OG
- `test_mass_properties.py`: U, AN, OG
- `test_material.py`: MP, EL
- `test_material_catalog.py`: MP, IC
- `test_material_config.py`: MP, IC
- `test_material_design.py`: U, MP, AS, AN, IS, OG, IC
- `test_mechanisms.py`: AS
- `test_merged_physics.py`: IC
- `test_mesh_assembly.py`: EL, AS
- `test_mesh_elements.py`: U, EL
- `test_mesh_locking.py`: AN, IS
- `test_mesh_verification.py`: IS, AN
- `test_method_components.py`: IC
- `test_mma.py`: OG
- `test_modal.py`: EL, AN, OG
- `test_mode_tracking.py`: U, AN
- `test_multi_material.py`: MP, AN, OG
- `test_newmark.py`: AN, OG
- `test_nonlinear.py`: MP, AN, OG
- `test_nonlinear_orientation.py`: MP, OG
- `test_nonlinear_solvers.py`: MP, AN, IS, OG, AS
- `test_onebody.py`: OG, E2E
- `test_ordering.py`: AS, PF
- `test_pareto.py`: AN, OG
- `test_pde_filter.py`: U, OG
- `test_post_cantilever.py`: E2E
- `test_post_checks.py`: E2E
- `test_post_pipeline.py`: E2E
- `test_post_sdf.py`: U, E2E
- `test_post_verify.py`: EL, E2E
- `test_published_values.py`: IS, AN
- `test_registry_features.py`: IC
- `test_regression.py`: IS, OG
- `test_research_examples.py`: IC
- `test_residual.py`: AS
- `test_result_contract.py`: IC, OG
- `test_robust.py`: OG
- `test_rom.py`: IS, AN
- `test_sdf_contract.py`: E2E
- `test_simpl.py`: OG
- `test_skfem_crosscheck.py`: IS, EL
- `test_strength.py`: MP
- `test_stress.py`: AN, OG
- `test_stress_lbracket.py`: OG, E2E
- `test_stress_recovery.py`: EL, AN
- `test_stress_singularity.py`: AN
- `test_stress_zero_density.py`: U, OG
- `test_study_options.py`: IC
- `test_sweep_connectivity.py`: E2E
- `test_sweep_post.py`: E2E
- `test_sweep_resume.py`: IC
- `test_symmetric_ties.py`: E2E
- `test_symmetry.py`: U, E2E
- `test_t10_convergence.py`: AN, E2E
- `test_taylor_remainder.py`: OG
- `test_tensor.py`: U
- `test_thermoelastic.py`: AN, AS, OG
- `test_timoshenko_cantilever.py`: AN
- `test_transient_heat.py`: AN, OG
- `test_tsai_wu_admissibility.py`: MP
- `test_units.py`: U, IC
- `test_uq.py`: U, AN, OG
- `test_vtu.py`: U, IC

# Risk kapsam matrisi: test katmanı × kütüphane alanı (üretilmiş dosya)

Kaynak: `tests/toporlogy/_vv_layers.py` (dosya → katman sınıflaması elle, dosya → alan içe aktarmalardan). Yeniden üretim: `python tests/toporlogy/_vv_layers.py --write`. Hücre: o katmandaki ve o alanı içe aktaran test dosyası sayısı / test fonksiyonu sayısı. Bu bir harita, başarı ölçütü değil: parametreli test bir sayılır, dolu hücre denetimin gücünü söylemez, boş hücre (—) o çiftte test olmadığını söyler.

| Katman | core | geometry | boundary | design | fem | solvers | optimization | optimizers | configuration | execution | post | references |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U unit / property | 38/594 | 3/30 | — | 12/148 | 21/282 | 6/96 | 22/285 | 5/83 | 7/86 | 3/19 | 5/68 | — |
| MP material point | 38/537 | — | — | 9/138 | 22/241 | 2/44 | 20/235 | 6/103 | 7/97 | 1/13 | 4/55 | — |
| EL element patch / quadrature | 13/162 | — | — | 3/32 | 19/235 | 3/53 | 6/74 | 3/36 | 3/40 | 2/23 | 3/40 | — |
| AS assembly / BC / reactions | 11/157 | 1/23 | 7/77 | 2/59 | 16/200 | 7/98 | 4/90 | 2/59 | 3/49 | — | 1/20 | — |
| AN analytical / MMS | 47/657 | — | — | 9/143 | 48/567 | 7/131 | 29/352 | 8/131 | 12/163 | 3/36 | 3/44 | — |
| IS independent solver / implementation | 26/305 | — | 1/14 | 5/104 | 33/421 | 9/157 | 13/170 | 4/86 | 6/99 | 2/23 | 1/24 | — |
| OG optimization / KKT / gradient | 32/349 | — | 3/29 | 19/214 | 42/431 | 6/87 | 54/510 | 26/292 | 26/269 | 8/80 | 3/54 | — |
| E2E end-to-end geometry | 4/56 | 5/39 | — | 2/13 | 7/85 | — | 12/108 | 1/21 | 5/51 | 2/23 | 11/91 | — |
| IC integration / compatibility | 15/148 | 2/34 | 2/15 | 11/120 | 19/177 | — | 23/210 | 9/107 | 30/270 | 17/141 | 1/8 | 1/11 |
| PF performance | — | — | 2/25 | 1/14 | 3/37 | 3/37 | 3/19 | 1/14 | 1/14 | 1/1 | — | — |

Boş hücreler (açık risk adayı; her biri bir risk değildir, örn. `references` için çoğu katman anlamsız):

- **U** (unit / property): boundary, references
- **MP** (material point): geometry, boundary, references
- **EL** (element patch / quadrature): geometry, boundary, references
- **AS** (assembly / BC / reactions): execution, references
- **AN** (analytical / MMS): geometry, boundary, references
- **IS** (independent solver / implementation): geometry, references
- **OG** (optimization / KKT / gradient): geometry, references
- **E2E** (end-to-end geometry): boundary, solvers, references
- **IC** (integration / compatibility): solvers
- **PF** (performance): core, geometry, post, references

Katman sınıflaması (dosya → katmanlar):

- `test_adaptivity.py`: EL, AN, IS
- `test_adiabatic.py`: U, MP, AN
- `test_aggregates.py`: U, OG
- `test_anisotropy.py`: U, MP
- `test_api_surface.py`: IC
- `test_architecture.py`: IC
- `test_auxetic_cell.py`: OG, E2E
- `test_baseline.py`: IC
- `test_baseline_policy.py`: IC
- `test_bc.py`: AS
- `test_benchmark.py`: OG
- `test_bifurcation.py`: U, AN, IS
- `test_buckling.py`: EL, AN, OG
- `test_buckling_geometry.py`: U, AN, IS, OG, E2E
- `test_buckling_physics.py`: AN, OG
- `test_catalog.py`: IC
- `test_catalog_ledger.py`: IC
- `test_checkpoint.py`: OG, IC
- `test_checkpoint_adjoint.py`: OG, PF
- `test_cli.py`: IC
- `test_cli_threads.py`: IC, PF
- `test_closed_forms.py`: U, AN
- `test_compatibility.py`: IC
- `test_compose.py`: U, OG
- `test_config.py`: IC
- `test_connectivity.py`: U, OG, E2E
- `test_constitutive.py`: U, MP, AN
- `test_core.py`: U, EL, AS
- `test_creep.py`: U, MP, AN
- `test_creep_plasticity.py`: U, MP, AN
- `test_creep_temperature.py`: MP, AN, IS, OG
- `test_creep_topology.py`: MP, AN, IS, OG
- `test_cure.py`: U, MP, AN
- `test_cure_slab.py`: MP, AS, AN, IS
- `test_curved_t10_degree.py`: EL, AN
- `test_deriving_methods.py`: OG, IC
- `test_element_orientation.py`: MP, EL, OG, E2E
- `test_fault_injection.py`: IC
- `test_fem_mms.py`: AN
- `test_fem_mms_extended.py`: AN
- `test_fem_rates_investigation.py`: EL, AN
- `test_foundations_examples.py`: IC
- `test_fractional.py`: U, MP, AN, IS
- `test_frame_invariance.py`: U, MP
- `test_fs2019_column.py`: AN, IS
- `test_gcmma.py`: OG
- `test_geometry.py`: U, E2E
- `test_gmg.py`: IS, OG, PF
- `test_gradients.py`: OG
- `test_grid_refusals.py`: U, IC
- `test_heat.py`: EL, AN, OG
- `test_history_fem.py`: MP, AS, AN, IS
- `test_history_more_models.py`: MP, AS, IS
- `test_history_physical.py`: MP, AS, IS
- `test_history_time_convergence.py`: AS, AN
- `test_history_yaml.py`: IC, E2E, OG
- `test_history_yaml_temperature.py`: IC, OG
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
- `test_kkt_stop.py`: OG, IC
- `test_laminate.py`: U, MP, AN
- `test_lengthscale.py`: U, OG
- `test_linear_elastic.py`: U, MP
- `test_manufacturing.py`: U, OG
- `test_mass_properties.py`: U, AN, OG
- `test_material.py`: MP, EL
- `test_material_cards.py`: U, MP, IC
- `test_material_catalog.py`: MP, IC
- `test_material_config.py`: MP, IC
- `test_material_contract.py`: U, MP, IC
- `test_material_design.py`: U, MP, AS, AN, IS, OG, IC
- `test_material_identification.py`: U, MP, OG
- `test_mechanisms.py`: AS
- `test_merged_physics.py`: IC
- `test_mesh_assembly.py`: EL, AS
- `test_mesh_elements.py`: U, EL
- `test_mesh_foundations.py`: U, EL, AS, IS
- `test_mesh_locking.py`: AN, IS
- `test_mesh_topology.py`: EL, OG
- `test_mesh_verification.py`: IS, AN
- `test_mesh_yaml.py`: IC, OG
- `test_mesh_yaml_comb.py`: AS, IC, OG
- `test_mesh_yaml_ext.py`: EL, AN, IS, IC, OG
- `test_method_components.py`: IC
- `test_micromechanics.py`: U, AN, IS
- `test_mixed_kg.py`: EL, AS, AN, IS
- `test_mma.py`: OG
- `test_modal.py`: EL, AN, OG
- `test_mode_tracking.py`: U, AN
- `test_model_epoch.py`: OG, IC
- `test_moduli_foundations.py`: U, AN
- `test_multi_material.py`: MP, AN, OG
- `test_multistart.py`: OG, IC
- `test_newmark.py`: AN, OG
- `test_nonlinear.py`: MP, AN, OG
- `test_nonlinear_foundations.py`: MP, EL, AN, IS
- `test_nonlinear_orientation.py`: MP, OG
- `test_nonlinear_solvers.py`: MP, AN, IS, OG, AS
- `test_onebody.py`: OG, E2E
- `test_ordering.py`: AS, PF
- `test_pareto.py`: AN, OG
- `test_pde_filter.py`: U, OG
- `test_periodic_filter.py`: U, OG
- `test_phase_angle.py`: OG, IC
- `test_plastic_sensitivity.py`: MP, OG, IC
- `test_plate.py`: U, AN, OG
- `test_poisson_convention.py`: U, AN, OG, IC
- `test_post_cantilever.py`: E2E
- `test_post_checks.py`: E2E
- `test_post_pipeline.py`: E2E
- `test_post_sdf.py`: U, E2E
- `test_post_verify.py`: EL, E2E
- `test_properties.py`: U, MP, AN
- `test_published_values.py`: IS, AN
- `test_real_density.py`: U, OG
- `test_registry_features.py`: IC
- `test_regression.py`: IS, OG
- `test_research_examples.py`: IC
- `test_residual.py`: AS
- `test_result_contract.py`: IC, OG
- `test_robust.py`: OG
- `test_rom.py`: IS, AN
- `test_saddle.py`: U, AN, IS
- `test_sdf_contract.py`: E2E
- `test_shell.py`: U, AN, OG
- `test_shell_drilling.py`: AN, OG
- `test_shell_reference.py`: IS, AN
- `test_shell_study.py`: OG, IC
- `test_shell_yaml.py`: IC, OG
- `test_shell_yaml_comb.py`: AS, IC, OG
- `test_shell_yaml_ext.py`: EL, AN, IS, IC, OG
- `test_simpl.py`: OG
- `test_skfem_crosscheck.py`: IS, EL
- `test_solver_diagnostics.py`: U, AN, IS
- `test_spectral.py`: U, AN, IS
- `test_speedups.py`: IS, OG, PF
- `test_start_design.py`: U, IC
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
- `test_tensor_cofactor.py`: U, AN, IS
- `test_thermoelastic.py`: AN, AS, OG
- `test_thermoelastic_study.py`: AN, OG, IC
- `test_thermoplastic.py`: U, MP, AN
- `test_timoshenko_cantilever.py`: AN
- `test_transient_heat.py`: AN, OG
- `test_transient_study.py`: AN, IS, OG, IC
- `test_tsai_wu_admissibility.py`: MP
- `test_units.py`: U, IC
- `test_uq.py`: U, AN, OG
- `test_viscoelastic.py`: U, MP, AN
- `test_viscoelastic_minmax_switch.py`: OG, E2E
- `test_viscoelastic_robust.py`: OG, E2E
- `test_viscoelastic_temperature.py`: MP, AN, IS, OG
- `test_viscoelastic_topology.py`: MP, AN, IS, OG
- `test_viscoelastic_uncertain.py`: OG, AN, E2E
- `test_viscoplastic.py`: U, MP, AN
- `test_vtu.py`: U, IC
- `test_yaml_research_responses.py`: OG, IC

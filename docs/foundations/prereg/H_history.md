# H — History-dependent FE on a general mesh (WP17 minimal slice; preregistration)

Preregistered 2026-11-27 01:40 +0300 (clock), before the module and its tests exist. Base SHA `f49b623`. New module
`src/toporlogy/fem/mesh/history.py`.

## Design

`HistoryProblem(mesh, model, thickness=1.0, plane="strain", degree=None)`: small-strain solid on a general mesh
(`fem.mesh.Mesh`, any element of `fem.mesh.reference`) with one material state per quadrature point
(`StateStore(model, n_cells · n_q)`). The model is used **as given** (physical parameters; no element-modulus
scaling); any model with the small-strain history contract (`update` or `evaluate` with `Kinematics("small")`).
2-D: plane strain only (the model's own plane-strain embedding).

`step(step: TimeStep, f_ext, prescribed=(dofs, values), tol=1e-10, max_iter=25) -> StepResult`: Newton on the free
DOFs with the material's algorithmic tangent, K = Σ Bᵀ C B dV (assembled with the mesh pattern, factorised with
structure "general", since a history tangent need not be symmetric), f_int = Σ Bᵀ σ dV; convergence
‖r_free‖∞ ≤ tol · max(‖f_ext‖∞, ‖f_int‖∞, tiny). Every iterate evaluates the model on the *committed* state;
the trial state of the converged iterate is committed once, after convergence. On non-convergence (or a
`ConstitutiveFailure`) nothing is committed, the displacement returns to the last accepted one and
`StepRejected` is raised. Reactions = f_int − f_ext at the prescribed DOFs (from the original residual). The
accepted time grid is recorded.

## Checks

| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| H0 | linear isotropic (stateless `LinearElastic` with physical D) on a distorted Q4 mesh under an affine prescribed displacement: displacement field exact and the reaction sum equals the analytic traction resultant | affine patch solution (analytic) | rel 1e-10 |
| H1 | homogeneous H8 bar (2×1×1 cells), roller supports (x = 0 face u_x = 0, one node line u_y, one node u_z), prescribed axial displacement history (ramp, hold, unload) with the legacy `J2Plasticity` (unit modulus, plastic in the ramp): every quadrature point's stress equals `drive_material_point` with ε_xx prescribed and the other five stresses zero | material-point driver (same constitutive kernel, independent global algebra: partially independent, labelled so) | abs 1e-9 · σ_y on all components, every step |
| H2 | two-material series bar (2 H8 cells along x, J2 with different yield stresses 1e-3 and 2e-3, unit modulus), end force ramp beyond the first yield: end displacement = Σ L_i ε_xx,i with ε_xx,i from `drive_material_point` under stress control σ_xx = F/A for each material | material point per material (partially independent) | rel 1e-8 |
| H3 | temperature: `ThermoElasticPoint` (core.properties, synthetic E(T), α) on an H8 bar fixed axially at both ends with free lateral faces, uniform T step from T_ref: axial stress −E(T) ε_th at every point, reaction = −E(T) ε_th · A | closed form (independent) | rel 1e-10 |
| H4 | rollback: a step forced to fail (max_iter = 0 on a plastic step) raises `StepRejected`, leaves the committed state and displacement bitwise unchanged; repeating it with the normal settings gives the same result as a run that never failed | uninterrupted run | `np.array_equal` |
| H5 | `HistoryProblem` refuses a finite-strain model, plane stress in 2-D, a prescribed DOF outside the mesh, a model without the small-strain contract | — | ValueError |

Run: `python -m pytest tests/toporlogy/test_history_fem.py -q -p no:randomly`. First result appended below.

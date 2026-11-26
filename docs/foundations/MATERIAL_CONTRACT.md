# Material contract: physical time, temperature and parameters

Status: P01 slice (WP11). Code: `src/toporlogy/core/constitutive.py`. Preregistration: `prereg/P01_contract.md`.

## What existed

`ConstitutiveModel.evaluate(kin, params, committed, request) -> MaterialResponse` with `Kinematics` (displacement
gradient, small | finite, plane), `StateStore` (read-only committed state, trial, commit, rollback, checkpoint), and
the unit-modulus convention: the FE physics multiplies stress, tangent, energy and dissipation by the element
modulus E_e. For `J2Plasticity` the yield stress and hardening are relative to E = 1, so they scale with E_e. That
convention stays as the **legacy normalised route**; it is not a physical material model.

## What this slice adds

* `TimeStep(t, dt, temperature=None, temperature_prev=None)`: one increment of physical time from t_n to t_n + dt,
  dt ≥ 0 (0 is an instantaneous probe), temperatures absolute in kelvin (> 0). Time is in seconds or one declared
  consistent unit; a load factor, an arc-length parameter or an optimisation iteration is never a time.
* `MaterialRequest.step`: the time step of the evaluation. Rate-independent laws ignore it; a model that needs it
  calls `require_step(model, request, temperature=...)`, which raises when it is missing.
* Physical models set `physical = True` and take moduli, stresses, viscosities and times in physical units (SI in
  material records: Pa, Pa·s, s, K, J/mol). `NonlinearElasticity` refuses such a model, because it scales every law
  by E_e and would scale the yield stress, viscosity and relaxation times with the design modulus. FE coupling of
  physical history models is WP17 and is not implemented here.
* `HistoryUpdate` and `small_strain_history_evaluate`: a 3-D small-strain history model implements
  `update(eps (n, 3, 3), committed, request) -> HistoryUpdate` and gets `evaluate` (Voigt packing, plane-strain
  embedding, plane stress refused) from the shared helper.
* `drive_material_point(..., times, temperatures, t0, temperature0, store, strain0)`: the mixed strain/stress
  driver with a physical time and temperature schedule, and a restart from a committed `StateStore`. The committed
  state is never changed inside the Newton loop of a step; one commit per converged step.

## Rules for new models

1. State is everything the update needs from the past (including the previous strain when the update is
   incremental). No hidden module state.
2. The tangent returned is the algorithmic tangent of the discrete update that was actually performed.
3. Energy and dissipation are returned only where they are defined; otherwise None (never zeros).
4. No clipping of physical parameters, no silent extrapolation of property tables, no regularisation that changes the
   problem. Invalid input raises.
5. Every physical law states its units; temperatures that enter an Arrhenius or WLF law are absolute (K).

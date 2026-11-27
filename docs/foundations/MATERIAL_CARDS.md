# Material cards (WP11 P05)

A material card is a plain dict (YAML-ready) that `toporlogy.optimization.material_cards.load_material_card` validates and
turns into one physical material model, together with its provenance, its units and a validity box. Pre-registration and
results: `prereg/P05_material_cards.md`; tests: `tests/toporlogy/test_material_cards.py`; examples:
`examples/foundations/materials/*.yaml` (one per model family, **all synthetic**: invented numbers, no real material, no
engineering claim).

The module lives in `optimization/` and not in `core/` because it reuses `ValidityBox` and `UNIT_FACTORS` of
`optimization/material_identification.py` and `tests/toporlogy/test_architecture.py` forbids `core` importing
`optimization`. Moving those two objects into `core` would allow moving the module with them.

```python
import yaml
from toporlogy.optimization.material_cards import load_material_card

card = load_material_card(yaml.safe_load(open("examples/foundations/materials/perzyna.yaml")))
card.model          # the built Perzyna model (SI)
card.provenance     # {'source': ..., 'synthetic': True, 'date': '2026-11-27', 'license': ...}
card.validity       # ValidityBox over the declared coordinates, SI
card.check(stress=3.0e7, rate=1.0e-2)    # raises ValidityBoxError outside the closed box (SI arguments)
card.to_dict()      # the card in its declared units; load_material_card(card.to_dict()) is the same card
```

## Schema
A card has exactly the keys `name`, `model`, `units`, `parameters`, `provenance`, `validity`. Unknown or missing keys, wrong
types and invalid values raise `ValueError` whose message starts with the dotted path of the offending key
(`parameters.elasticity.E: must be > 0, got -2e+09`).

**model**: `thermoelastic`, `generalized_maxwell`, `norton`, `norton_bailey`, `physical_j2`, `perzyna`, `drucker_prager`,
`curing_thermoelastic`.

**units**: explicit, one entry per kind the card uses, from the unit table of `material_identification.UNIT_FACTORS`:
`stress` Pa|kPa|MPa|GPa, `time` s|min|h, `temperature` K, `rate` 1/s|1/min|1/h. Numbers are given in the declared unit and are
multiplied once by the table factor to SI before the model is built; `to_dict` returns them as declared. Celsius and
Fahrenheit are refused as a unit, as a unit key, as a negative temperature, and as a string (`"25C"`): an offset is not a
factor. Not selectable (fixed by the model): dimensionless numbers, activation energy J/mol, CTE 1/K, the WLF constant C2 in K
(a temperature difference). A kind that is used but not declared is an error at the parameter that needs it.

**YAML pitfall**: YAML 1.1 reads `5.0e6` (exponent without sign) as a string. Write `5.0e+6`; the error message says so.

**elasticity** (all models with a modulus): exactly two of `E`, `nu`, `K`, `G`. A third is refused, and the message says whether
it is consistent with the pair (redundant) or not (redundant and inconsistent). `thermoelastic` and `curing_thermoelastic`
take each of them, and `alpha`, as a number (constant over `validity.temperature`) or a table `{T: [K...], values: [...]}`
(piecewise linear; its temperatures must cover `validity.temperature`). The other models take numbers.

**parameters** per model (SI kind in brackets):

| model | parameters |
|---|---|
| thermoelastic | `elasticity`, `alpha` [1/K], `T_ref` [K] |
| generalized_maxwell | `shear`, `bulk`: `{inf, amplitudes, taus}` [stress, stress, time]; optional `shift`: `{type: wlf, T_ref, C1, C2}` or `{type: arrhenius, T_ref, Ea}` |
| norton | `elasticity`, `A` [rate], `sigma_ref`, `n`; optional `arrhenius: {Q [J/mol], T_ref}` |
| norton_bailey | `elasticity`, `B`, `sigma_ref`, `n`, `m`, `t_ref` [time]; optional `hardening: time \| strain` |
| physical_j2 | `elasticity`, `yield_stress`; optional `hardening`, `saturation_stress`, `saturation_rate` |
| perzyna | `elasticity`, `yield_stress`, `hardening`, `tau0` [time], `sigma_ref`, `m` |
| drucker_prager | `elasticity`, `a`, `b0`; optional `hardening`, `a_flow` |
| curing_thermoelastic | `kinetics` (`nth_order: {A [rate], Q, n}` or `kamal_sourour: {k1, k2: {A, Q}, m, n}`), `moduli_uncured`, `moduli_cured`, `alpha`, `T_ref`, `beta`; optional `alpha_gel`, `alpha0`, `substeps` |

The model constructors of `core` do the physics checks (`a < 3` for Drucker-Prager, saturation consistency, ...); their
messages are re-raised with the path `parameters`.

**provenance**: `source` (non-empty), `synthetic` (a real boolean), `date` (YYYY-MM-DD), and `license`, which is required when
`synthetic` is false. A card with `synthetic: false` must name a real source (the word "synthetic" is refused) and a license.
The module checks that provenance is stated, never that it is true: loading a card with `synthetic: false` proves nothing about
the data. Optional fields of the plan (grade, test method, version, uncertainty, batch) are **not implemented**.

**validity**: closed ranges `[lo, hi]` for `temperature` [K, lo >= 0], `time` [>= 0], `strain_rate` [>= 0, magnitude; box
coordinate `rate`] and `stress` [signed]. Required: thermoelastic: temperature; generalized_maxwell: time (and temperature
with a `shift`); norton: stress (and temperature with `arrhenius`); norton_bailey: stress, time; physical_j2: stress; perzyna:
stress, strain_rate; drucker_prager: stress; curing_thermoelastic: temperature, time. `card.check(temperature=, time=, rate=,
stress=)` takes SI values (scalars or arrays) and raises `ValidityBoxError` (a `ValueError`) outside; an undeclared coordinate
raises `ValueError`. The box is not enforced by the models themselves: the caller (or a driver) calls `check`, except that
the property functions of the temperature-dependent models refuse a temperature outside the validity range by themselves.

## What the unit checks catch (and what they do not)
When a `validity.stress` range is declared, two checks look for a mislabelled stress unit. With A = max(|lo|, |hi|):

* **U1** A must not exceed the smallest Young modulus of the card (it would be a strain above 100 %).
  Catches: a modulus labelled with a unit 10^k too small while the range is right.
* **U2** each strength scale (`yield_stress`, `b0`, `sigma_ref`) divided by A must lie in [1e-3, 1e3].
  Catches: a strength labelled with a unit 10^k too small (or large) while the range and the modulus are right.

Measured in `test_material_cards.py` (MC6): E written 2000 with unit Pa (truly MPa) is caught by U1; yield written 20 with unit
Pa is caught by U2; a card whose every stress number is MPa labelled Pa is **self-consistent and accepted**, and its stress is
exactly 1e-6 of the right one; a card without a stress range (thermoelastic) has no check at all. A unit label is checked
against the other numbers of the same card, not against the world.

## Limits
Cards describe the models that exist; there is no card for a model that is not in `core`. The 1-D/3-D material-point driver
is `core.constitutive.drive_material_point`; the FEM hook-up is not part of this slice. No card in the repository carries real
data; a real card needs source, license and date and is the data owner's responsibility.

# P05_material_cards — material cards: schema, provenance, units, validity box (preregistration)

Preregistered 2026-11-27 03:26 +0300 (clock output), before any new test was run. Base SHA `6c45d19`. Module:
`src/toporlogy/optimization/material_cards.py` (`load_material_card`, `MaterialCard`); docs:
`docs/foundations/MATERIAL_CARDS.md`; examples: `examples/foundations/materials/*.yaml`; tests:
`tests/toporlogy/test_material_cards.py`.

**Deviation from the task text, stated up front.** The task asks for `core/material_cards.py` and for reuse of
`ValidityBox` / `UNIT_FACTORS` of `optimization/material_identification.py`. `tests/toporlogy/test_architecture.py` forbids
`core` (layer "base") importing `optimization`, and duplicating those objects is forbidden by the task. The module is
therefore placed in `optimization/`, which may import both. Moving `ValidityBox`/`UNIT_FACTORS` into `core` and the
module after them is an integrator decision; no test depends on the module path except the import line.

All numbers in cards and tests are **synthetic** (no real material data, no real-engineering claim).

## Schema (what is tested)
Top level exactly: `name`, `model`, `units`, `parameters`, `provenance`, `validity`. `units` keys among `stress`
(Pa, kPa, MPa, GPa), `time` (s, min, h), `temperature` (K only), `rate` (1/s, 1/min, 1/h): the `UNIT_FACTORS` table of
material_identification. A kind used by a parameter or validity entry must be declared; fixed-unit quantities
(dimensionless, activation energy J/mol, CTE 1/K) are not selectable. Values are numbers in the declared unit and are
converted once, by the table factor, to SI before the model is built. Isotropic elasticity = exactly two of E, nu, K, G.
Provenance: `source`, `synthetic` (bool), `date` (YYYY-MM-DD), `license` (required iff synthetic is false).
Validity keys: `temperature` [K], `time`, `strain_rate` (box coordinate `rate`), `stress`; each `[lo, hi]`, lo <= hi,
converted to SI, closed interval. Required per model: thermoelastic: temperature; generalized_maxwell: time (+ temperature
with a shift); norton: stress (+ temperature with arrhenius); norton_bailey: stress, time; physical_j2: stress; perzyna:
stress, rate; drucker_prager: stress; curing_thermoelastic: temperature, time.

Two unit sanity checks run when a stress range is declared (they are the only mislabel detectors; they see nothing else):
**U1** stress range upper bound <= smallest Young modulus of the card (strain <= 100 %); **U2** each strength scale
(yield_stress, b0, sigma_ref) divided by the stress range upper bound lies in [1e-3, 1e3].

## Tests
| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| MC1 | each of the 8 example cards: `load(card.to_dict()).to_dict() == card.to_dict()`, also through `yaml.safe_dump`/`safe_load` | the dict itself | exact equality |
| MC2 | model built from a card = model built directly with hand-converted SI parameters, same stress on a 5-step material-point path (`drive_material_point`, times and temperatures where the model needs them) | **regression** (same kernel); the independent part is the hand-written SI parameters | `np.array_equal` |
| MC2b | independent closed forms. Uniaxial STRAIN (all six strains controlled, eps_xx = e, rest 0, tension positive): sigma_xx = (K + 4G/3) e − 3K eps_th, sigma_yy = sigma_zz = (K − 2G/3) e − 3K eps_th (derivation: elastic strain diag(e − eps_th, −eps_th, −eps_th), lambda = K − 2G/3, 3 lambda + 2 mu = 3K), eps_th = alpha (T − T_ref) for constant alpha. thermoelastic card: E(T) = np.interp of the card table, nu constant, K = E/(3(1−2nu)), G = E/(2(1+nu)). physical_j2 card with e < sigma_y/(2G) (q = 2G e < sigma_y, elastic): sigma_xx = (K + 4G/3) e | closed forms written in the test from the card numbers | max abs error / max abs sigma <= 1e-12 |
| MC3 | R01–R38 below each raise `ValueError` whose message contains the stated path of the offending key | none needed | exact (ValueError + substring) |
| MC4 | units: a card in MPa / min / 1/min with factor-converted numbers builds the same model (same stress path) as the card in Pa / s / 1/s; `to_dict` keeps the declared units; validity converted | the Pa card | max abs error / max abs sigma <= 1e-12 (conversion rounding) |
| MC5 | validity box: `check` passes inside and at both closed ends, raises `ValidityBoxError` (a ValueError) just outside for temperature, time, rate, stress (SI coordinates, also when declared in MPa/min); an undeclared coordinate raises ValueError | the declared range | exact |
| MC6 | negative control, see below | closed ratio 1e-6 | exact refusal; stress ratio within 1e-12 |
| MC7 | every example YAML under examples/foundations/materials loads, has `synthetic: true`, and the set of `model` values is exactly the 8 families | the file list | exact |

### MC3 refusal cases (mutation of the base physical_j2 card, substring expected in the message)
Base: units {stress Pa, time s, temperature K, rate 1/s}; E 2.0e9, nu 0.35, yield_stress 2.0e7, hardening 1.0e8;
validity.stress [0, 3.0e7].

R01 extra top-level key `color`: "color". R02 no provenance: "provenance". R03 no validity: "validity". R04 model "foo": "model".
R05 unknown parameter `yeild2`: "parameters.yeild2". R06 no yield_stress: "parameters.yield_stress". R07 units key `C`:
"units.C" and "Celsius". R08 units.temperature "degC": "units.temperature" and "Celsius". R09 units.temperature "C": same.
R10 units.stress "psi": "units.stress". R11 units.stress "s": "units.stress". R12 units without stress:
"parameters.elasticity.E". R13 validity.temperature [-20, 80] (thermoelastic card): "validity.temperature". R14 validity.stress
[5, 1]: "validity.stress". R15 validity.stress [0]: "validity.stress". R16 validity key `pressure`: "validity.pressure".
R17 E < 0: "parameters.elasticity.E". R18 yield_stress < 0: "parameters.yield_stress". R19 hardening < 0: "parameters.hardening".
R20 E, nu and the consistent G: "parameters.elasticity" and "redundant". R21 E, nu and an inconsistent G: "redundant" and
"inconsistent". R22 only E: "parameters.elasticity". R23 nu = 0.5: "parameters.elasticity.nu". R24 yield_stress NaN:
"parameters.yield_stress". R25 yield_stress True: same. R26 yield_stress "2e7" (string): same. R27 synthetic false without
source: "provenance.source". R28 synthetic false with source, no license: "provenance.license". R29 synthetic "false" (string):
"provenance.synthetic". R30 no date: "provenance.date". R31 date "27/11/2026": "provenance.date". R32 synthetic false with
source "synthetic": "provenance.source". R33 empty name: "name". R34 thermoelastic T_ref = −10: "parameters.T_ref". R35 thermoelastic
alpha table with a negative knot: "parameters.alpha.T". R36 generalized_maxwell negative tau: "parameters.shear.taus".
R37 norton n = 0: "parameters.n"; norton arrhenius Q < 0: "parameters.arrhenius.Q". R38 thermoelastic T_ref given as the string
"25C": "parameters.T_ref"; drucker_prager a = 3: "parameters".

### MC6 negative control, and exactly which check catches what
Base: yield 2.0e7, E 2.0e9, hardening 1.0e8, range [0, 3.0e7], all in Pa.
(a) modulus mislabeled only (E written 2000 with unit Pa, truly MPa; yield and range correct in Pa): **U1** refuses
(range upper 3e7 > E = 2000 Pa); U2 would not (2e7/3e7 = 0.67).
(b) strength mislabeled only (yield written 20, truly MPa; E and range in Pa): **U2** refuses (20/3e7 = 6.7e-7 < 1e-3);
U1 would not.
(c) everything mislabeled consistently (E 2000, yield 20, hardening 100, range [0, 30], all "Pa", truly MPa): **no check
can catch it**, the card is self-consistent. Prediction: accepted, and the stress of its model is exactly 1e-6 of the stress
of the correct card (ratio within 1e-12). This is the stated limit: a unit label is checked against the other numbers of the
same card, not against the physical world.
(d) thermoelastic card (no stress range required, none declared) with E mislabeled: accepted, no check applies.

Run: `python -m pytest tests/toporlogy/test_material_cards.py -q -p no:randomly` (TMP/TEMP and --basetemp under
`.claude/tmp-toporlogy/P05`). First result appended below after the run.

## Amendment (added before the first run)

Amendment written 2026-11-27 03:31 +0300 (clock output), before the first run: two checks were added to the test file that the table above
does not list. **MC2c** (regression, bitwise): a `curing_thermoelastic` card with `kamal_sourour` kinetics builds the same model
as the directly constructed `KamalSourour` one. **MC8**: `load_material_card` does not mutate its input dict (equality with a
deep copy). No threshold or earlier prediction was changed.

## First result (2026-11-27 03:34 +0300, clock output)

Run on base 6c45d19 + the P05 files: `python -m pytest tests/toporlogy/test_material_cards.py -q -p no:randomly` -> **69 passed**, no check failed, no threshold changed. Breakdown: MC1 8, MC7 1, MC2 8 (bitwise equal), MC2c 1, MC2b 2 (closed forms, <= 1e-12 relative), MC3 40 refusal cases (R01-R38 with R37a/b, R38a/b; every message contained the stated path), MC3 control 1, MC4 2, MC5 1, MC6 a-d 4, MC8 1.

Test set-up fix before this passing run (an earlier attempt with `-x` stopped at the first example): the example YAMLs wrote exponents without a sign (`5.0e6`), which YAML 1.1 reads as a string; the loader rejected it ("expected a number"). Fixed in the example files (`5.0e+6`) and a hint added to the error message. No threshold or prediction changed.

Measured in MC6: (a) U1 refused, (b) U2 refused, (c) accepted with stress ratio within 1e-12 of 1e-6, (d) accepted: as predicted.
Related existing tests run: `test_architecture.py`, `test_api_surface.py` (11 passed; the module adds no name to the package namespace).
Limitations: the box is checked on request, not by the models; real material data is not covered (all synthetic); the grade/test-method/uncertainty fields of the plan are not implemented.

## Integration note (2026-11-27 04:35 +0300)

After the HY merge the example folder holds 10 cards (two isothermal variants used by the history YAML examples).
The MC7 test had implemented "covers exactly the 8 families" as a file count; it now checks the set of families
(the preregistered wording) and that a variant's file name starts with its family. No threshold involved.

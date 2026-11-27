# HT — Temperature histories in the history YAML path (preregistration)

Preregistered 2026-11-27 06:50 +0300 (clock), before the change and the tests. Base `43ba79d`.
Change: optional top-level `temperature: {table: [[t, T], ...]}` (t in the config's time unit, T absolute in K; the
first point at t = 0, strictly increasing t, the last point at or after the last step time; piecewise linear, no
extrapolation). A material card with a WLF/Arrhenius shift (generalized_maxwell) or an Arrhenius factor (norton) is
accepted only with a temperature schedule; a schedule without such a card is refused. The step temperatures are the
table interpolated at the step times, T_0 = T(0); they go to `ViscoelasticStructure(temperatures=, temperature0=)` /
`CreepStructure(arrhenius=, temperatures=, temperature0=)` (VT, CT). The card's temperature validity (if declared) is
checked. Isothermal configs are unchanged.

| ID | Hypothesis | Threshold |
|---|---|---|
| HT1 | a YAML config with a shifted generalized_maxwell card and a temperature table builds a structure whose objective and gradient equal the same problem built with the API (VT) | bitwise |
| HT2 | the same for a norton card with an Arrhenius factor (CT) | bitwise |
| HT3 | refusals with key paths: shifted card without temperature; temperature without a shift/Arrhenius card; table not starting at 0, not increasing, ending before the last step, T ≤ 0, wrong shape; temperature outside the card's validity | ConfigError naming the path |
| HT4 | resolve is idempotent with a temperature block; every existing HY test passes unchanged | equal / pass |

Run: `python -m pytest tests/toporlogy/test_history_yaml_temperature.py tests/toporlogy/test_history_yaml.py -q -p no:randomly`.

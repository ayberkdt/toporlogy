# Hata enjeksiyonu (08 §2): bozuk girdi açık bir hatayla durmalı

Test: `tests/toporlogy/test_fault_injection.py`. Etiketler: **[Ö]** ölçüldü, **[O]** koddan okundu.

## 0. Ön kayıt (2026-11-26 00:27 +0300, testlerin hiçbir koşusundan önce)

Beklenti her durumda aynı: işlem **durur** (sessiz başarı yok), kabul edilen türden bir istisna yükseltir ve mesajı
(büyük/küçük harf duyarsız) suçluyu adlandıran düzenli ifadeye uyar. Başka türden bir istisna (`AttributeError`,
`IndexError`, çıplak `LinAlgError`) ya da suçluyu adlandırmayan mesaj **FAIL** sayılır. Ön kayıtlı beklentiyi
tutmayan durum strict xfail olarak kalır, gözlenen davranış §1'e ve testin `reason` metnine yazılır; beklenti
sonradan değiştirilmez.

| Grup | Durum (test id) | Kabul edilen tür | Mesaj deseni |
|:---|:---|:---|:---|
| NaN/Inf | `nan_modulus_solve` (E'de NaN, `Elasticity.solve_numpy`) | ValueError | `finite\|nan` |
| | `inf_load_solve` (yükte Inf) | ValueError | `finite\|inf` |
| | `nan_force_config` (YAML'da `force: [nan, -1]`) | ConfigError | `finite\|nan` |
| | `inf_E0_config` | ConfigError | `E0` |
| | `nan_design_evaluate` (`Study.evaluate`, x'te NaN) | ValueError | `finite\|nan` |
| SPD olmayan malzeme | `negative_E0_config`, `zero_E0_config` | ConfigError | `E0` |
| | `nu_half_3d` (ν = 0,5, 3-B; D tekil) | ValueError | `nu\|poisson\|positive definite` |
| | `nu_above_half_config` (ν = 0,7), `nu_below_minus_one_config` (ν = −1,2) | ConfigError | aynı |
| | `orthotropic_not_spd_config` (ν12 = 2,5) | ConfigError | `positive definite\|nu12\|admissib` |
| | `negative_modulus_solve` (bir elemanda E = −1) | ValueError | `modul\|positive\|E` |
| Ters/bozuk eleman | `negative_h_domain`, `negative_h_config` | ValueError / ConfigError | `h > 0` / `h` |
| | `negative_thickness_domain` (2-B kalınlık −1) | ValueError | `thickness` |
| Tekil sınır koşulu | `no_supports_config` | ConfigError | `support` |
| | `one_node_support_config` (2-B, tek düğüm ankastre: dönme serbest) | ConfigError | `rigid\|mechanism\|singular\|rotation` |
| | `zero_load_config` (bütün yükler sıfır) | ConfigError | `load\|zero` |
| Yanlış şekil | `modulus_wrong_length_solve`, `load_wrong_rows_solve` | ValueError | `shape\|length\|active` / `shape\|rows\|dof` |
| | `domain_shape_1d_config`, `force_3_components_2d` | ConfigError | `shape` / `component` |
| Yanlış birim/değer | `h_with_unit_text` (`h: "1 mm"`) | ConfigError | `number` |
| | `negative_rmin`, `volume_upper_above_one` (hacim üst sınırı 50), `negative_max_iter`, `negative_tol` | ConfigError | parametre adı |
| Bozuk girdi | `yaml_syntax_error`, `yaml_top_level_list`, `yaml_unknown_top_key`, `missing_file` | ConfigError | `cannot parse` / `mapping` / anahtar adı / `not found` |
| | `yaml_duplicate_key` (aynı anahtar iki kez; PyYAML sessizce sonuncuyu alır) | ConfigError | `duplicate` |
| Bayat koşu klasörü | `rerun_into_finished_folder` (bitmiş klasöre ikinci koşu) | CLI çıkış kodu 2 | — |

Gerekçe: 08 §2 "fail-closed error messages". Birim sistemi yok (README: "any consistent set"), bu yüzden "yanlış
birim" burada yalnız birim yazılmış metin ve fiziksel olarak anlamsız değerlerle sınanır; boyut analizi (kütle/ağırlık,
basınç/kuvvet) 08 §4'ün işi, bu dilimde yok.

Kapsam dışı (bu dilimde): çözücü ıraksaması (GMG `maxit`, Newton) ve bayat önbellek/checkpoint (checkpoint/restart
yok, 08 §4). Bunlar NOT VERIFIED.

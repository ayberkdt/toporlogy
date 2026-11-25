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

## 1. İlk koşu (2026-11-26 00:30 +0300, `b5ba393` üstünde) [Ö]

Komut: `python -m pytest tests/toporlogy/test_fault_injection.py -q -p no:randomly`. Sonuç: 33 durumdan 24'ü ön
kayıtlı beklentiyi tuttu (PASS), 9'u tutmadı (FAIL → strict xfail; gözlenen davranış testteki `FAILED_FIRST_RUN`
sözlüğünde, aynı metin aşağıda). Beklentilerin hiçbiri değiştirilmedi.

| Durum | Sonuç | Gözlenen |
|:---|:---|:---|
| `inf_load_solve` | FAIL | çözümden sonra `SolveResidualError` (RuntimeError) "load case 0: the solution is not finite": mesaj açık, tür ve yer ön kayıttaki gibi değil (girdi denetimi yok) |
| `nan_force_config` | FAIL | sessizce kabul: `force: [nan, -1]` ile Setup kuruluyor, NaN yük vektörüne giriyor |
| `inf_E0_config` | FAIL | "elastic constants do not give a positive definite stiffness (check the Poisson ratios)": E0'ı adlandırmıyor, ν'ye yönlendiriyor |
| `negative_E0_config`, `zero_E0_config` | FAIL | "material: E1 > 0 is required": izotrop malzemede anahtar E0, mesaj E1 diyor |
| `negative_thickness_domain` | FAIL | sessizce kabul: `Domain(thickness=-1)` ve `Elasticity` kuruluyor (negatif rijitlik) |
| `load_wrong_rows_solve` | FAIL | `IndexError` (`Elasticity._solve_free`): F'nin satır sayısı denetlenmiyor |
| `volume_upper_above_one` | FAIL | sessizce kabul: hacim oranı üst sınırı 50; kısıt hiç etkin olmaz |
| `yaml_duplicate_key` | FAIL | yinelenen anahtar yakalanmıyor (PyYAML sonuncuyu alır). Testin YAML'ında kısıt yoktu, kurulum ilgisiz bir hatada durdu ("optimizer oc needs at least one constraint"); kısıt eklenmiş keşif koşusunda (ön kayıtsız) ikinci `domain` [4, 4] sessizce kullanıldı |
| diğer 24 durum | PASS | NaN modül, NaN tasarım, ν = 0,5 / 0,7 / −1,2, SPD olmayan ortotrop, negatif E elemanı, negatif h, desteksiz ve tek düğümlü mesnet, sıfır yük, yanlış uzunlukta E, 1-B şekil, 3 bileşenli 2-B kuvvet, birimli metin, negatif rmin / max_iter / tol, YAML sözdizimi, liste kök, bilinmeyen anahtar, eksik dosya, bitmiş klasöre ikinci koşu (çıkış 2) |

Test tasarımındaki hata (benim): `yaml_duplicate_key` YAML'ı kısıt içermiyordu; bulgu (yinelenen anahtarın
yakalanmaması) bundan etkilenmiyor ama ön kayıtlı durumun kendisi ilgisiz bir hatayla durdu. Düzeltme ayrı, yeni bir
ön kayıtlı durum olarak yazılmalı; bu dilimde yapılmadı.

Çözüm koşulu (her biri için): ilgili girdi denetimi eklendiğinde strict xfail XPASS olur ve test kırmızıya döner;
o zaman `FAILED_FIRST_RUN` girdisi silinir. Düzeltmeler bu dilimin kapsamında değil (malzeme denetimi
`core/material.py`, tensor-core oturumunun dosyası).

## 2. Düzeltmeler (2026-11-26, `integrate/research-round-1`) [Ö]

Ön kayıtlı beklentiler değişmedi; yedi durum kendi beklentisini artık tutuyor ve strict xfail'den çıktı (kütükten
de silindi). Komut: `python -m pytest tests/toporlogy/test_fault_injection.py -q -p no:randomly` → 31 passed, 2 xfailed.

| Durum | Düzeltme | Yer |
|:---|:---|:---|
| `nan_force_config` | her yük durumunun yük vektörü sonlu olmalı: `ConfigError` "load case has a load that is not finite (NaN or Inf)" | `configuration/build.py` |
| `inf_E0_config`, `negative_E0_config`, `zero_E0_config` | izotrop yolda E0 sonlu ve > 0: "E0 must be a finite number > 0" | `core/material.py` `elasticity_from_config` |
| `negative_thickness_domain` | `Domain`: "thickness > 0 is required (finite)" | `core/domain.py` |
| `volume_upper_above_one` | `volume` kısıtının üst sınırı (0, 1] (parçanın hacim oranı) | `configuration/build.py` |
| `yaml_duplicate_key` | YAML yükleyicisi (SafeLoader alt sınıfı) ve JSON okuyucu yinelenen anahtarı reddeder: "cannot parse …: duplicate key" | `configuration/resolve.py` |

Kalan iki strict xfail (`inf_load_solve`, `load_wrong_rows_solve`) `fem/elastic.py`'nin girdi denetimini ister; bu
dosya o sırada hızlandırma oturumunda olduğundan sonraya bırakıldı. `yaml_duplicate_key` durumunun YAML'ındaki
eksik kısıt artık önemsiz: hata okuma aşamasında, kısıt denetiminden önce geliyor.

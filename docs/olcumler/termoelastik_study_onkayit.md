# Termoelastisite `Study` fiziği olarak — ön kayıt

Ön kayıt 2026-11-26 11:29 +0300, kod yazılmadan önce. `fem.thermoelastic.ThermoElastic` (TE1–TE9, `termoelastisite_onkayit.md`)
şimdiye kadar yalnız Python'dan kullanılıyordu; `Study` ve YAML bağlantısı yoktu.

Kapsam (bu tur): **verilen düzgün sıcaklık artışı** ΔT (ısı çözümü yok, sıcaklık tasarıma bağlı değil). Isıl yük
tasarıma bağlıdır: F_th(E) = Σ_e E_e G ΔT. Tasarıma bağlı sıcaklık (iletkenliği yoğunluktan) bu turda yok.

Değişiklik:
* `fem.thermoelastic.ThermoElasticPhysics(te, T)`: `solve(E, F)` → U, her yük durumu için u_c = K(E)⁻¹ (F_c + F_th(E));
  `total_load(E, F)` → F + F_th(E) (autograd E üzerinden hem K'dan hem yükten).
* `Compliance`: fizik `total_load` sunuyorsa c = Σ_c w_c F_tot,cᵀ U_c (termoelastik uyumun olağan tanımı);
  sunmuyorsa bugünkü F ᵀ U, değişmeden.
* YAML: `physics: {type: thermoelastic, alpha: …, delta_T: …, element: null}`; ν, düzlem ve D malzemeden.
  Tek mesnet takımı, yalnız x / y / z sıfır yer değiştirmesi; Q4 / H8; eleman başına D alanı: açık hata.

| # | Kontrol | Eşik |
|:---|:---|:---|
| TS1 | ΔT = 0: Study uyumu ve gradyanı = `elasticity` fiziğiyle aynı problem | ≤ 1e-12 göreli |
| TS2 | ΔT ≠ 0: Study değeri = doğrudan `ThermoElastic.displacement` ile F_totᵀu | ≤ 1e-12 göreli |
| TS3 | ΔT ≠ 0 ve mekanik yük: Study gradyanı merkezi farkla (2-B ve 3-B) | ≤ 1e-6 göreli |
| TS4 | YAML örneği kurulur ve 5 MMA iterasyonu koşar | sonlu değerler |
| TS5 | birden çok mesnet takımı, eğik mesnet, Q4IM, eleman başına D | açık hata |
| TS6 | elastik örnekler | `pytest -m baseline` bit düzeyinde; çözülmüş yapılandırma anlık görüntüsü değişmez (yeni anahtar yalnız yeni fizikte) |

## Sonuç (ilk ölçüm, 2026-11-26)

`tests/toporlogy/test_thermoelastic_study.py` (10 test); hepsi ilk koşuda geçti. Kontrol sırasında TS5'in "eleman
başına D" durumunun yanlış nedenle düştüğü görüldü (eksik ortotropik malzeme: test kurulum hatası); geçerli açı alanıyla
yeniden kuruldu ve her durum beklenen iletiyle eşleştirildi (`match=`).

| # | Durum | Ölçülen |
|:---|:---|:---|
| TS1 | PASS | değer 5,6e-14, gradyan 6,7e-14 göreli (12 × 6 kiriş) |
| TS2 | PASS | 0 (c = 378,1; α = 1e-2, ΔT = 50) |
| TS3 | PASS | 2-B 4,5e-8, 3-B 1,2e-9 göreli |
| TS4 | PASS | 5 MMA iterasyonu, sonlu |
| TS5 | PASS | iki mesnet takımı, eğik mesnet, Q4IM, açı alanı (eleman başına D), eksik α: beklenen iletiyle açık hata |
| TS6 | PASS | baseline 8 passed bit düzeyinde; çözülmüş yapılandırma anlık görüntüsü değişmedi |

Ek değişiklik (ön kayıtta yoktu): `ScenarioCompliance` yükü doğrudan `ctx.F`'den okuyordu; termoelastik fizikle ısıl
yükü atlayacaktı. Artık `Compliance().per_case` kullanıyor (diğer fiziklerde aynı işlemler). `test_uq.py` ve
`test_yaml_research_responses.py` bu değişiklikten sonra geçti. Katalogda `compliance × thermoelastic` SUPPORTED,
diğer yanıtlar NOT VERIFIED.

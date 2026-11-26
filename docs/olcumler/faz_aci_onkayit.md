# Çok fazlı tensör karışımı + fiber açısı bloğu: ön kayıt

Yazıldı: 2026-11-26 12:32:58 +0300 (`date`), kodun ilk satırından önce. Taban: `cc5d5b9` (`research/round-2`).
Önceki tur: `malzeme_tasarimi.md` (§8 ilk madde: "birden çok faz + açı" geçiş koşulu).

## 0. Bugünkü durum (bu oturumda okundu)

* `design/orientation.py`: `OrientedEmbed` tek yoğunluk bloğu (`n_design`) + açı bloğu taşıyor; `Embed`'den türüyor
  ama `n_fields` hep 1.
* `optimization/study.py`: `_check_phases` ilk bağlantının tür adı tam olarak `"Embed"` değilse reddediyor;
  `_phase_start` yalnız-katı-hacim durumunda `np.full(self.n_var, τ)` döndürüyor (açı bloğu olsaydı açıyı da τ
  yapardı) ve `default_start` çok fazlıda `_reduce`'u çağırmıyor.
* `configuration/components/core.py` `embed`: `n != 1` fazla açı bloğunu reddediyor (kural `angle_phases`,
  UNSUPPORTED, `enforced="component"`).

## 1. Formülasyon (kararlar, koddan önce)

* Tasarım vektörü: `[yoğunluk bloğu (m · n_design, alan-öncelikli, `Embed(n_fields=m)` ile aynı) | açı bloğu
  (n_design ya da 1)]`. Blok adı `density` kalır (m = 1'de düzen, `describe()` ve vektör bit düzeyinde aynı).
* Tek açı alanı, `rotate = True` olan bütün fazları aynı θ ile döndürür (`TensorPhases.phase_stiffness` bugünkü
  davranışı). Faz başına ayrı açı bu dilimde YOK (geçiş koşulu olarak yazılacak).
* `density=False` ile birden çok faz reddedilir (hangi fazın tam dolu olduğu tanımsız): yeni UNSUPPORTED kural
  `angle_phases_no_density`, bileşende ve `OrientedEmbed`'de fail-closed.
* `Study._check_phases`: ilk bağlantı `Embed` ya da `OrientedEmbed` olabilir; geri kalan liste (`DensityFilter`,
  `HeavisideProjection`, `Passive`) ve yanıt / optimizasyon kısıtları değişmez.
* `Study.default_start` çok fazlıda: `_phase_start` yalnız yoğunluk alanlarını (m · n_design) döndürür, sonra
  ilk bağlantının `reduce`'u (varsa) açı bloğunu ekler. Düz `Embed`'de `reduce` yok, n_var = m · n_design: sayılar
  değişmez.

## 2. Kontroller ve eşikler

Göreli hata = max|a − b| / max|b| (dizi üzerinde). Izgara aksi yazılmadıkça 12 × 6, sol kenar ankastre, uç yükü.

| # | Kontrol | Eşik |
|:---|:---|:---|
| MA0 | Değişiklik öncesi / sonrası parmak izi (betik `faz_aci_parmakizi.py`, öncesi `faz_aci_oncesi.json` bu commit'te): tek fazlı `OrientedEmbed` (element + global açı, density true / false) ileri geçiş ızgarası, açı alanı, `reduce`, `layout.describe()`; tek fazlı açı `Study.evaluate` (f0, df0, g, dg); çok fazlı düz `Embed` `Study.default_start` ve `evaluate` (`simplex`, `recursive`; PhaseVolume ve yalnız katı hacim başlangıçları) | bit düzeyinde aynı (sha256) |
| MA1a | İndirgeme, `recursive`: fazlar [izotrop A, fiber], t_A ≡ 0, açı + yoğunluk filtresi; `Study.evaluate` = tek fazlı [fiber] açı Study'si (aynı fiber yoğunluğu ve açı vektörü) | f0 ve PhaseVolume(fiber) değeri göreli ≤ 1e-12; fiber yoğunluk bloğu + açı bloğu gradyanları göreli ≤ 1e-10 |
| MA1b | İndirgeme, `simplex`: fazlar [fiber, fiber kopyası], t_1 ≡ 0; aynı karşılaştırma | MA1a ile aynı |
| MA2 | Merkezi FD, h = 1e-6, `Study.evaluate` gradyanı; amaç Compliance, kısıtlar PhaseVolume (her faz) ve Mass; zincir `OrientedEmbed` + `DensityFilter(1,5)` + `Passive` (bir keep sütunu); üç yön: yalnız yoğunluk, yalnız açı, ikisi (N(0, 1)); durumlar: `simplex`-element (rmin 1,5), `recursive`-element (rmin 1,5), `simplex`-global, üç faz (fiber, izotrop, `rotate = false` ortotrop) `simplex`-element | her yanıt ve yönde \|FD − g·d\| ≤ 1e-6 · \|FD\| |
| MA2b | Taylor kalanı, `simplex`-element, h ∈ {1e-3, 5e-4, 2,5e-4, 1,25e-4}, amaç | log-log eğim ≥ 1,9 |
| MA3 | Başlangıç vektörü: `Study.default_start` uzunluğu m · n_design + n_açı; yoğunluk kısmı `phase_start(m, n_design, PhaseVolume sınırları)` ile, açı kısmı `s_of(start)` ile bit düzeyinde aynı; yalnız katı hacim sınırıyla yoğunluk kısmı τ = 1 − (1 − V)^(1/m), açı kısmı `s_of(start)` | bit düzeyinde |
| MA4 | YAML örneği `examples/research/fiber_matrix_angle_cantilever2d.yaml` (40 × 20, matris izotrop + fiber ortotrop, `simplex`, element açısı rmin 2, iki PhaseVolume kısıtı), `run.max_iter=20`, MMA | (a) bütün geçmiş değerleri sonlu; (b) son amaç < ilk amaç; (c) her kısıt ≤ üst · (1 + 1e-2) |
| MA5 | Fail-closed: (a) `density: false` + 2 faz → `ConfigError` (YAML) ve `ValueError` (`OrientedEmbed`); (b) açı + 2 faz + `oc` → reddedilir; (c) açı + 2 faz + `volume` yanıtı → reddedilir; (d) `OrientedEmbed` ilk bağlantı değilse → reddedilir | dört durumda hata |
| MA6 | Katalog: `angle_phases` SUPPORTED, kanıt MA2 ve MA4 test kimlikleri; yeni `angle_phases_no_density` UNSUPPORTED + prob; `test_catalog.py`, `test_catalog_ledger.py`, `test_compatibility.py` | geçer |
| MA7 | `pytest -m baseline` | 8 passed, bit düzeyinde |
| MA8 | `test_material_design.py`, `test_element_orientation.py`, `test_merged_physics.py`, `test_multi_material.py` (`-m "not slow"`): değişiklikten önceki koşuyla aynı passed / xfailed sayıları, failed 0 | aynı sayılar |
| MA9 | Ölçüm (eşiksiz): MA4 örneği 60 iterasyon; son uyum, faz hacim kesirleri, gri payı (`phase_report`), açı tutarlılığı en küçük ρ̃ | kayıt |

Başarısız bir kontrol, ölçülen değeriyle strict xfail olur; eşik gevşetilmez. İlk koşudan sonra eklenen kontroller
"post hoc" etiketi taşır.

## 3. Anlık görüntü tahminleri

* `tests/toporlogy/baseline/api_surface.json`: değişmez (yeni genel sınıf özniteliği yok; `n_fields` örnek
  özniteliği).
* `tests/toporlogy/baseline/list.txt` (bileşen listesi): değişmez (`embed` parametreleri aynı).
* `test_units.py` anlık görüntüsü: değişmez.
* Çözümlenmiş yapılandırma anlık görüntüsü: değişmez (yeni YAML anahtarı yok).
* `docs/vv_katalog.*`: `angle_phases` satırı UNSUPPORTED → SUPPORTED (kanıtlı), yeni `angle_phases_no_density`
  satırı, `angle_elasticity` gerekçe metni ("one or several phases"). `_vv_layers` çıktısına yeni test dosyası.

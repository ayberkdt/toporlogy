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

---

## 4. Sonuçlar (2026-11-26 13:21 +0300, ön kayıttan sonra yazıldı)

Kod ve testler: commit `4574272`. Ham ölçüm: `faz_aci.json` (betik `faz_aci.py`, `4574272` üzerinde, temiz ağaç;
eşzamanlı test koşusuyla aynı anda, süre gürültülü). Testler: `tests/toporlogy/test_phase_angle.py`.

| # | Sonuç | Ölçülen / not |
|:---|:---|:---|
| MA0 | PASS | `faz_aci_sonrasi.json` (`4574272`, `src` temiz) ile `faz_aci_oncesi.json` (`cc5d5b9`): 72/72 sha256 aynı |
| MA1a (`recursive`) | PASS | f0, g, df0, dg farkları 0 (bit düzeyinde) |
| MA1b (`simplex`) | PASS | f0, g, dg 0; df0 5,4e-16 |
| MA2 | **FAIL** (1/12) | `simplex`-global, yalnız yoğunluk yönü, uyum: \|FD − g·d\| / \|FD\| = **2,43e-6** (> 1e-6; strict xfail). Öteki 11 durumda en büyük 4,4e-7 (`simplex`-element, açı yönü) |
| MA2 post hoc | PASS | Richardson FD (h = 1e-3, 5e-4), aynı eşik: 12 durumda en büyük 1,8e-7 |
| MA2b | PASS | eğim 2,008 |
| MA3 | PASS | |
| MA4 | PASS | 20 iterasyon (testte) |
| MA5a–d | PASS | |
| MA6 | PASS | `test_catalog.py`, `test_compatibility.py` geçti; `test_catalog_ledger.py` ayrı koşuda 7 passed (§4.2) |
| MA7 | PASS | `pytest -m baseline`: 8 passed (359 s), listelenen her örnekte en büyük göreli fark 0,00e+00 |
| MA8 | PASS | dört dosya (`-m "not slow"`): önce (`b01cd7e`, ayrı çalışma ağacı) 99 passed, 5 deselected, 3 xfailed; sonra (`4574272`) aynı: 99 / 5 / 3, failed 0 |
| MA9 | ölçüm | 60 iterasyon (`max_iter`, yakınsamadı): uyum 3830,6 → 180,4; PhaseVolume matris 0,24998 (≤ 0,25), fiber 0,19998 (≤ 0,2); boşluk 0,550; gri payı 0,458 (max η < 0,9, projeksiyonsuz); en küçük açı tutarlılığı ρ̃ = 0,158; 2400 değişken; 12,6 s |

### 4.1 MA2: ön kayıtlı FD adımı yuvarlama bölgesinde

Başarısız durumda FD adım taraması (oturum içi teşhis betiği, kayıt dışı; aynı x ve yön): h = 1e-3: 7,7e-6;
5e-4: 1,9e-6; 1e-4: 7,2e-8; 1e-5: 3,2e-7; 1e-6: 2,4e-6; 1e-7: 8,9e-6. h < 1e-4'te hata büyüyor: kesme değil
yuvarlama (f0 = 904, FD farkı ~3e-10 mutlak, çözücü yuvarlamasının mertebesi). Richardson (1e-3, 5e-4) 6,1e-9.
Gradyanın yanlış olduğuna işaret yok; eşik gevşetilmedi. Bu yeni bir bulgu değil: `test_multi_material.py` V4 aynı
nedenle Richardson kullanıyor. Yeni ön kayıtta FD adımı h = 1e-4 ya da Richardson seçilmeli.

### 4.2 Test ortamı

`test_catalog_ledger.py` ilk koşuda (11 dosyalık toplu koşu) 3 hata verdi: dosyanın kendi `--collect-only` alt
süreci 3221225794 (0xC0000142, Windows süreç başlatma / DLL başlatma hatası) koduyla çıktı; test iddiası değil. Tek
başına yeniden koşuda 7 passed. Toplu koşunun geri kalanı: 247 passed, 5 xfailed (beşi de beklenen strict xfail:
`test_material_design` G2 ×2 ve A2 6 × 6, `test_units` U8, bu turun MA2).

### 4.3 Anlık görüntü tahminleri

* `api_surface.json`, `list.txt`, `test_units.py` anlık görüntüsü: değişmedi (tahmin doğru; `list.txt` yalnız satır
  sonu farkı, git değişiklik görmüyor).
* Çözümlenmiş yapılandırma anlık görüntüsü (`tests/toporlogy/baseline/resolved_configs.json`): **tahmin yanlış**.
  Yeni YAML örneği anlık görüntüye bir girdi ekledi (14 → 15; var olan 14 girdi değişmedi, betikle karşılaştırıldı).
  Ön kayıtta örneğin bu dosyaya girdiğini hesaba katmadım.
* `vv_katalog.*`: tahmin edildiği gibi (`angle_phases` SUPPORTED, `angle_phases_no_density` yeni,
  `angle_elasticity` metni). `vv_kutugu.*`: yeni strict xfail girdisi (MA2). `vv_risk_kapsam.md`: yeni dosya.

### 4.4 Kapsam dışı / geçiş koşulları

* Faz başına ayrı açı yok: tek açı alanı `rotate = true` fazlarını birlikte döndürür.
* `density: false` birden çok fazla reddediliyor (`angle_phases_no_density`).
* `run.start` (pertürbe başlangıç) açı bloğu ve çok fazla hâlâ reddediliyor (`perturbed_start`, bu oturumun dosyası
  değil).
* `Study.evaluate_density` açı bloğuyla desteklenmiyor (önceki turdan; değişmedi).
* Tam takım (~2 saat) koşmadım.

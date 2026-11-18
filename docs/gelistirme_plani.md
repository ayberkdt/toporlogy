# toporlogy — envanter ve geliştirme planı (17 Kasım 2026)

**Taban:** `p0-result-consistency` dalı, `8bcaeeb` (altında `main` = `ff1db52`).
**Kaynaklar:** modül modül kod okuması, `docs/mimari_plan.md`, `docs/olcumler/*`, test paketi ve 17 Kasım 2026 tarihli dış
mimari inceleme (7 belge; bulgularından kodla doğrulananlar §3'te).
**Etiketler** (`mimari_plan.md` ile aynı): **[Ö]** ölçüldü (komut ya da dosya verilmiş), **[O]** koddan okundu
(dosya:satır), **[K]** karar önerisi, **[D]** doğrulanmadı.

Bu belge `mimari_plan.md`'nin yerine geçmez: o belge yöntem kararlarını ve aşamaları (A0–D) tutar, bu belge kodun
bugünkü durumunu, bulunan açıkları ve sonraki işlerin sırasını tutar.

### Durum (17 Kasım 2026, `g0-safety-net` dalı)

| Aşama | Durum | Not |
|---|---|---|
| A-01–A-05 (§3.1) | tamam | `3e3a1cc` … `8bcaeeb` |
| G0 güvenlik ağı | tamam | baseline (7 örnek, bit düzeyinde), API yüzeyi + `list`, katman kuralı + import sırası testleri |
| G1 doğruluk düzeltmeleri | tamam | B-02 ve B-06 plandaki gibi G7'ye kaldı |
| G2 `registry.py`'yi böl | tamam | `configuration/` paketi; `registry.py` uyumluluk modülü (aşağıda) |
| G3 `config.py`'yi böl | tamam | `configuration/{schema,resolve,build}.py`, `execution/{runner,runs}.py`; `config.py` uyumluluk modülü |
| G4 `bc.py`'yi böl | tamam | `boundary/{part,selectors,supports,constrained,loads,checks,cases}.py`; `bc.py` uyumluluk modülü |
| G5 katman paketleri | kısmen | `core/`, `design/`, `optimization/` tamam; `execution/` (`record`, `cli`) yapılmadı |
| G6–G9 | başlamadı | K1 kararı verildi: sınırlı (K1-a) |
| K5 lisans | kapandı | tescilli, tüm hakları saklı (`LICENSE`) |

G1 bulgularının kapandığı commit'ler: B-01 `aeb3ac3`; B-03 `6dca762`; B-04 `a6b0bc6`; B-05 `cf83e40`; B-07 `54ff90f`,
`d648f7a`; B-08 `7994862`; B-09 `fbb0c9c`; B-10 `b19ec1c`; B-11 `a4fe49d`; B-12 ve B-13 `9fba67b`; B-14 `3d98811`;
B-15 `6881fab`; B-16 `a7f3fd0`; B-17 `9fe2d0f`; B-18 `5033fd9`. B-19 ölçüldü, etkisi yok (`419f444`: çözüm ve
iterasyon sayısı değişmiyor; kod değişmedi, regresyon testi eklendi).

Uygulama sırasında plandan sapan iki ölçüm: B-04'te L-BFGS-B'nin başarı bayrağı tek başına yakınsama göstermedi
(249 dual çözümde "başarısız" bitenlerin kalıntısı ≤ 1,1e-8, "başarılı" bitenlerinki ≤ 4,1e-8); uyarı dual kalıntısına
bağlandı. B-08'de öngörülen boş `argmax` hatası yerine yanıltıcı bir "bodies [1] cannot be reached" iletisi gözlendi.

G2 (K1-a ile): `registry.py`'nin 1854 satırı, kaynak metinden üretilerek `configuration/` altına taşındı:
`registry.py` (mekanizma), `helpers.py`, `components/{core,geometry,boundary,design,physics,responses,schedules}.py`,
`post_steps.py`, `boundary_mapping.py`. `toporlogy.registry` eski adların hepsini aynı nesnelerle yeniden dışa aktarıyor;
kayıt sırası korundu. Planın G2 tablosundan iki sapma: son işlem adımları `configuration/post_steps.py`'de, BC eşlemesi
`configuration/boundary_mapping.py`'de duruyor (K1-a'da `verification/` paketi açılmıyor).

G3: `config.py`'nin 1053 satırı aynı yöntemle taşındı: `configuration/schema.py` (şema belgesi, varsayılanlar),
`configuration/resolve.py` (okuma, çözümleme, geçersiz kılma, tarama), `configuration/build.py` (`Setup`, `build`),
`execution/runner.py` (`run_setup`), `execution/runs.py` (`write_resolved`, `status_of`, `write_sweep_csv`,
`read_run`). Plandan sapma: `status_of` `runner` yerine `runs`'ta, çünkü `read_run` onu kullanıyor ve `runner` →
`runs` → `runner` döngüsü oluşuyordu. Katman testine bir `compat` katmanı eklendi: `toporlogy.config` ve
`toporlogy.registry` her şeyi yeniden dışa aktarabilir, paketin içinde yalnız yürütme katmanı onları içe aktarabilir.

G4: `bc.py`'nin 1912 satırı `boundary/` paketine taşındı: `part.py` (`PartInfo`, `part_info` ve önbelleği),
`selectors.py`, `supports.py`, `constrained.py` (`ConstraintSet`, `ConstrainedElasticity`, `PerCaseSupports`),
`loads.py`, `checks.py` (rapor ve tablo yardımcıları), `cases.py` (`LoadCase`, `Combination`, `BoundaryConditions`).
`toporlogy.boundary` `bc`'nin açık adlarını aynen sunuyor; `bc.py` özel adlar dahil hepsini yeniden dışa aktarıyor.
Paketin içindeki 16 içe aktarma `toporlogy.boundary`'ye taşındı. Plandan sapma: plandaki `assembly.py` ve
`validation.py` ayrı modül olmadı, çünkü birleştirme ve kontroller `BoundaryConditions` sınıfının yöntemleri; sınıfın
dışındaki yardımcılar `checks.py`'de.

G5 (iki adım): modüller bölünmeden `git mv` ile taşındı: `domain` → `core/`; `chain`, `interp`, `manufacturing` →
`design/`; `responses`, `study`, `robust`, `lengthscale`, `connectivity`, `dynamics`, `checks` → `optimization/`.
Eski adlar `sys.modules` takma adı: `toporlogy.study` ile `toporlogy.optimization.study` aynı modül nesnesi (özel adlar,
modül durumu ve monkeypatch ortak). Paketin içindeki içe aktarmalar yeni yollara çevrildi; katman testi eski adları
`compat` katmanında sayıyor, bu yüzden eski yolda kalan bir iç içe aktarma testi düşürür. Kalan adım: `record` ve
`cli` → `execution/`.

Yeni açık bulgu (B-20, 18 Kasım 2026): yavaş test `test_connectivity.py::test_split_scenario_all_measures[3]`
("coupling" senaryosu) başarısız: tasarım tek gövde yerine iki gövde (209 + 7 hücre, "1 floating body without
interface"). Aynı sonuç `main`'de (`ff1db52`, bu planın bütün değişikliklerinden önce) ve G5'ten önceki `e57444c`'de
aynı sayılarla ölçüldü; değişikliklerden kaynaklanmıyor. Test normal pakette koşmadığı için daha önce görülmedi.
Nedeni araştırılmadı.

B-21 (18 Kasım 2026, dış incelemeden, düzeltildi): A-02'nin kapısı, onarılmış tasarımın değerlendirilemeyen kısıtlarını
(`feasible=None`) geçer sayıyordu. `lbracket_bore3d`'de onarılmış bir yoğunlukta amaç ve üç kısıtın hiçbiri
değerlendirilemiyor (robust yanıtlar ve aşındırılmış alan zincire ihtiyaç duyuyor), yani onarım hacmi ne kadar artırırsa
artırsın kapı geçiyordu. Artık geçmek için bütün kısıtların değerlendirilip sağlanması gerekiyor; değerlendirilemeyenler
özette adıyla yazılıyor, düz hacim oranı (önce/sonra) bilgi olarak raporlanıyor. Aynı incelemenin öteki maddeleri: B-02
(sonlandırma ölçütü, G7), `fem.modal → optimization.dynamics` istisnası (izleniyor) ve CI (G9'un ilk parçası eklendi:
`.github/workflows/tests.yml`).

---

## 1. Özet

Kütüphane yöntem açısından olgun: 87 bileşen YAML'dan kurulabiliyor, çekirdek iki eski sürücüye karşı 1e-6 içinde
regresyon testli, gradyanlar sonlu farkla ≤ 1e-6 doğrulanıyor, matris tutmayan GMG çözücüsü PARDISO'dan 7,2× hızlı
[Ö `docs/olcumler/A2.md`]. Sorunlar üç grupta toplanıyor:

1. **Doğruluk ve rapor açıkları** (§3): beşi `p0-result-consistency` dalında kapatıldı; 19 açık bulgu var, en
   önemlileri yakınsama bayrağının kısıt uygunluğunu içermemesi, β'nın ardışık `run()` çağrıları arasında taşınması,
   MMA alt probleminin başarısız çözümü sessizce kabul etmesi ve bağlantılılık bilgisindeki çift sayım.
2. **Yapısal borç** (§4): üç dosya (`registry.py` 1854, `bc.py` 1881, `config.py` 1053 satır) birden çok sorumluluk
   taşıyor; beş kavram iki ayrı uygulamayla duruyor (`Shape`, `Cylinder`, STL okuma/yazma, `local_thickness`,
   `Symmetry`).
3. **Yetenek ve kanıt boşlukları** (§6): bağımsız referans yok (regresyon kâhinleri deponun kendi portları), T10 ağ
   yakınsaması gösterilmedi, 3-D gerilme kısıtı taşınmadı, onarılmış tasarımlar hacim kısıtını aşıyor.

Önerilen sıra (§5): önce güvenlik ağı (G0), sonra doğruluk düzeltmeleri (G1), sonra davranışı değiştirmeyen taşımalar
(G2–G5), en son davranış ve arayüz değişiklikleri (G6–G9). Her adım ayrı commit ve aynı kabul kapısından geçer.

---

## 2. Elimizde olan

### 2.1 Büyüklük

**Kodun yarısından fazlası optimizasyon, yapılandırma ve son işlem katmanında; yapılandırmanın tamamı iki dosyada.**
[Ö: `src/toporlogy` altında satır sayımı, 17 Kasım 2026]

| Katman | Dosya | Satır | Modüller |
|---|---:|---:|---|
| optimizasyon | 13 | 3256 | `study`, `responses`, `robust`, `lengthscale`, `connectivity`, `dynamics`, `checks`, `optim/` |
| yapılandırma | 2 | 2907 | `registry` (1854), `config` (1053) |
| son işlem | 12 | 2752 | `post/` |
| sınır koşulları | 1 | 1881 | `bc` |
| çözücü | 7 | 1456 | `solvers/` (doğrudan, PARDISO yükleyici, GMG) |
| fizik | 6 | 1181 | `fem/` (elastisite, GMG elastisite, ısı, modal, elemanlar) |
| tasarım | 3 | 1004 | `chain`, `interp`, `manufacturing` |
| geometri | 1 | 836 | `geometry` |
| kaynakça | 10 | 820 | `references/` |
| yürütme | 2 | 607 | `cli`, `record` |
| çekirdek | 1 | 192 | `domain` |
| paket | 2 | 41 | `__init__`, `__main__` |
| **toplam** | **60** | **16 933** | |

### 2.2 Yetenekler

Her satırın kanıtı `dosya:satır` olarak verildi [O]. "Test" sütunu `tests/toporlogy` içinde sınıf adı ya da YAML adıyla
yapılan aramaya dayanır.

**Izgara ve geometri**

| Yetenek | Uygulama | Test | Belgelenmiş sınır |
|---|---|---|---|
| Düzenli 2-D/3-D ızgara, kısmi hacim (`frac`), `keep`/`void` maskeleri | `domain.py:31-192` | `test_core.py`, eski voksel sürücüsüne karşı 1e-14 | kalınlık yalnız 2-D (`domain.py:47`) |
| CSG: 11 şekil (kutu, küre, daire, silindir, kapsül, yarı uzay, çokgen, prizma, yarık, delik…), `|`, `&`, `-` | `geometry.py:83-356` | `test_geometry.py:18`, `:38` | Boole işlemlerinde mesafe tam değil, sınır (`geometry.py:7-8`); prizma yalnız x/y/z boyunca |
| Vokselleştirme (yüzeye yakın alt örnekleme, en büyük bileşen) | `geometry.py:376-447` | `test_geometry.py:47-71` | — |
| STL okuma/yazma, ağ hacmi, sarım sayısı, ışınla tam aralık | `geometry.py:465-655` | `test_geometry.py:90-115` | STL düzeni "bellekten" yazıldı (`geometry.py:463`) |
| Delik bulma, çatal birleştirme | `geometry.py:670-818` | `test_geometry.py:125` | yalnız eksene hizalı (`geometry.py:34`, `:749`) |

**Sınır koşulları** (`bc.py`)

| Yetenek | Uygulama | Test | Belgelenmiş sınır |
|---|---|---|---|
| 18 seçici (düzlem, kutu, silindir yüzü, delik, küre, şekil, en yakın, düğüm no; birleşim/kesişim/fark) | `bc.py:274-591`, kayıt `registry.py:1089-1099` | `test_bc.py:34-52`, `test_registry_features.py` | hepsi düğüm tabanlı |
| 5 mesnet: sabit (eğik yön dahil), kayar, simetri, pim deliği, yay | `bc.py:595-886` | `test_bc.py:141-234` | eğik ve yay mesnetleri `fixed_dofs`'a girmez (`bc.py:1616`) |
| 10 yük: düğüm, nokta, yüzey kuvveti, çekme, basınç, rulman, uzak kuvvet/moment, tork, hacim kuvveti | `bc.py:1017-1416` | `test_bc.py:70-303` | rulman yükünün ticari tanımları okunmadı (`bc.py:39`, `:46`) |
| Yük durumları, durum başına mesnet, kombinasyonlar, bileşke ve rijit-gövde kontrolü | `bc.py:1418-1880` | `test_bc.py:264-290` | kombinasyonlarda durumlar mesnetleri paylaşmalı (`bc.py:1428`) |

**Tasarım zinciri ve üretim kısıtları**

| Yetenek | Uygulama | Test | Belgelenmiş sınır |
|---|---|---|---|
| `Embed` ► `DensityFilter` (seyrek / konvolüsyon) ► `HeavisideProjection` ► `Passive` | `chain.py:32-189` | `test_gradients.py:87-132`, top88/topo3d'ye karşı < 1e-14 | konvolüsyon yolu CPU'da test edilmiyor |
| SIMP, RAMP, iki aralıklı SIMP | `interp.py:13-40`, `dynamics.py:87` | `test_gradients.py:142`, `test_modal.py:193` | RAMP deneysel, kaynağı okunmadı (`interp.py:29-30`) |
| AM filtresi (Langelaar), döküm yönü, simetri/ekstrüzyon/desen bağlama, grup ortalaması | `manufacturing.py:154-742` | `test_manufacturing.py` (20 test; eski AM filtresine karşı ≤ 1e-12) | sarkma ≤ 45°; AM ve döküm sırası değiştirilemez (`manufacturing.py:755`) |

**Fizik ve çözücüler**

| Yetenek | Uygulama | Test | Belgelenmiş sınır |
|---|---|---|---|
| Elastisite H8, H8IM, Q4; autograd uyumlu çözüm, kendine eşlenik kısayol | `fem/elastic.py`, `fem/elements.py:131-139` | `test_core.py:152-196`, `test_gradients.py:222-301` | H8 bir-iki eleman kalınlıkta eğilmede kilitlenir (`fem/elements.py:3-4`) |
| Isı iletimi (bağlantılılık için) | `fem/heat.py:87` | `test_heat.py` (tam çözümlere 1e-10) | — |
| Modal analiz, küme tespiti, tekrarlı özdeğer uyarısı | `fem/modal.py:109` | `test_modal.py` | tekrarlı özdeğer türevlenemez (`fem/modal.py:25`) |
| Doğrudan çözücü: PARDISO / SuperLU | `solvers/direct.py:29-40` | `test_core.py:196` | PARDISO torch'tan önce yüklenmeli (`__init__.py:9-16`) |
| GMG-PCG: V/W/K döngüsü, kara kutu aktarım, ada deflasyonu, CUDA karma hassasiyet | `solvers/gmg*.py`, `fem/elastic_gmg.py:42` | `test_gmg.py` (doğrudana karşı ≤ 1e-8) | yalnız 3-D, H8/H8IM; `coarse`, `pin_rel`, tek başına `pcg` test edilmiyor |

**Yanıtlar, optimizasyon, bağlantılılık**

| Yetenek | Uygulama | Test | Belgelenmiş sınır |
|---|---|---|---|
| 13 yanıt: uyum, hacim, robust uyum, genişletilmiş hacim, yerel hacim, gri düzey, iletim/bağlama bağlantılılığı, özfrekans, özdeğer, KS en küçük özdeğer, `maximize`, en küçük frekans | `registry.py:686-687`, `:1404-1415` | `test_gradients.py`, `test_robust.py`, `test_modal.py`, `test_lengthscale.py` | `FrequencyMargin` test edilmiyor |
| OC (tek kısıt), MMA (çok kısıt), β sürekliliği, robust süreklilik | `optim/`, `robust.py:420` | `test_mma.py` (eski MMA'ya karşı < 1e-6), `test_regression.py` | MMA GCMMA değil, genel yakınsama garantisi yok (`optim/mma.py:28-31`) |
| Robust biçimlendirme (aşınmış/ara/genişlemiş), en küçük boyut kuralları, sınır biçimlendirmesi | `robust.py` | `test_robust.py` (7) | boyut garantisi serbest yüzlerde ±1 eleman (`robust.py:44-50`) |
| Bağlantılılık analizi (PASS/FRAGILE/FAIL), en kısa bağ, köprü onarımı, tek gövde planı | `connectivity.py` | `test_connectivity.py` (19), `test_onebody.py` | köprü sonrası yeniden analiz yok (`connectivity.py:777-779`) |

**Son işlem** (`post/`)

| Yetenek | Uygulama | Test | Belgelenmiş sınır |
|---|---|---|---|
| `finalize`: ince ızgara, hacim eşlemeli eşik, SDF, adaya göre temizlik, uyarlamalı açma/kapama, arayüzleri tam yeniden yerleştirme, ağ çıkarma, özellik kilitli düzgünleştirme | `post/pipeline.py:139-296` | `test_post_pipeline.py` (ΔV < %0,5, delik < %1, Hausdorff < 0,6 h_f) | keskin dış kenarda SDF hatası ~0,7 h (`post/sdf.py:17`) |
| Üretim kontrolleri: su geçirmezlik, gövde/boşluk sayısı, et kalınlığı (ışın ve iç küre), sarkma, iç köşe yarıçapı | `post/checks.py:247-296` | `test_post_checks.py` | baskı servisi değerleri varsayılan değil (`post/checks.py:17-19`) |
| Dışa aktarma: STL, PLY, OBJ, JSON rapor, işleme payı varyantı | `post/export.py` | `test_post_checks.py` | STEP yazılmaz (`post/export.py:3-5`) |
| T10 yeniden analiz (gmsh), voksel referansı, BC eşdeğerlik etiketi | `post/verify.py:102-361`, `registry.py:1716-1786` | `test_post_verify.py`, `test_registry_features.py` | destek köşeleri ve yük yamaları tekil (`post/verify.py:13`) |

**Yapılandırma ve çalıştırma**

| Yetenek | Uygulama | Test |
|---|---|---|
| Kayıt: 13 tür, 87 bileşen, tembel `"modül:ad"` hedefleri, birimli parametreler, iç içe bileşenler, yakın ad önerisi | `registry.py:1-443` | `test_config.py`, `test_registry_features.py` |
| Şema, doğrulama, kurulum, geçersiz kılma (`--set`), tarama (`--sweep`), koşu klasörü | `config.py` | `test_config.py` (16), `test_cli.py` (8) |
| CLI: `run`, `validate`, `show`, `compare`, `list`; çıkış kodları 0/1/2 | `cli.py:501-540` | `test_cli.py` |
| Koşu kaydı: `setup.json`, `history.jsonl`, `result.npz`, anlık görüntüler, git sürümü | `record.py` | `test_core.py:277` |

### 2.3 Kanıt altyapısı

- **Test paketi:** 25 dosyada 236 test fonksiyonu [Ö]. Yavaş olanlar 10; iki tarama modülü
  (`test_sweep_connectivity.py`, `test_sweep_post.py`) modül düzeyindeki `pytestmark` ile bütünüyle `-m slow` arkasında,
  bu yüzden docstring'lerinde anılan "hızlı alt küme" de normal koşuda çalışmıyor [O `test_sweep_post.py:34`].
- **Regresyon kâhinleri** (`tests/toporlogy/topo_oracles/`): eski `topo3d` (7 senaryo) ve `top88` portlarına karşı uyum
  geçmişi ve x̄ ≤ 1e-6 [O `test_regression.py:29-30`]; eski MMA'ya karşı iterasyon başına < 1e-6
  [O `test_mma.py:124-141`]. `stress_topo.py` kâhininin gerilme optimizasyonunu hiçbir test çağırmıyor; `topo3d.py:456`
ondan yalnız `mma_multi`'yi alıyor [O].
- **Gradyan kontrolleri:** her bağlantı ve yanıt için yönlü merkezi fark, bağıl hata ≤ 1e-6 (`checks.py:13-36`).
- **Ölçüm kayıtları:** A0 GMG kapısı kaldı (2,2×, 5× gerekiyordu); A2 geçti (çözümde 7,2×, 10⁶ elemanda yineleme başına
  ~11,9 s; sessiz makinede ölçülmedi); bağlantılılık taraması 14 senaryo; son işlem taraması 12 giriş
  [Ö `docs/olcumler/`].
- **Gerçekten test edilmeyen kod** (sınıf adı ve YAML adıyla arandı, dolaylı çağrılar ayıklandı) [Ö]: `bc.Where`,
  `geometry.shape_from_function`, `robust.filter_profile`, `dynamics.FrequencyMargin`, CPU'da konvolüsyon filtresi,
  GMG `coarse` ve `pin_rel` seçenekleri, tek başına `solvers.gmg.pcg`.

### 2.4 Depo ve paketleme

- Sürüm 0.2.0, Python ≥ 3.11; bağımlılıklar numpy, scipy, torch ≥ 2.2, pyyaml; ek paketler `plot`, `post` (pyvista,
  vtk, numba), `fast` (pypardiso), `mesh` (gmsh), `dev` [O `pyproject.toml:6-18`].
- `.gitignore` beyaz liste: yayımlanan depo yalnız `src/toporlogy`, `tests/toporlogy`, `examples`, `docs` içindeki
  belirli dosyalar ve kök dosyalardan oluşuyor; izlenen dosya sayısı 112 [Ö `git ls-files`]. `docs/` altında yalnız
  `mimari_plan`, `registry`, `olcumler` ve `assets` izinli, bu belge eklenirse listeye girmesi gerekir.
- LICENSE, CHANGELOG ve CI yapılandırması yok; README lisansın olmadığını açıkça yazıyor (`README.md:124`).
- `pytest` `testpaths = ["tests"]`: yerel makinede git'e girmeyen `tests/conftest.py` (aerokit'i içe aktarıyor) ve
  `tests/test_bracket_*.py` dosyaları da toplanıyor [O `pyproject.toml:29`].
- Statik denetim: ruff `src` ve `tests` için temiz; mypy `src/toporlogy` için 31 dosyada 230 hata [Ö, 17 Kasım 2026].

---

## 3. Doğrulanmış açıklar

### 3.1 Kapatılanlar (`p0-result-consistency`)

**Sonucu ya da raporu yanlış gösteren beş açık kapatıldı; her biri önce başarısız olan bir testle sabitlendi.**

| No | Açık | Düzeltme | Commit |
|---|---|---|---|
| A-01 | Özet, kısıtları son analiz edilen iterasyondan, amacı dönen tasarımdan okuyordu (MBB 16×6, 3 iterasyon: hacim 0,499768 raporlandı, dönen tasarım 0,500001) | kısıtlar `Result.constraint_values`'dan | `3e3a1cc` |
| A-02 | Köprü onarımı `Result.xphys`'i değiştiriyor, `x`, amaç ve `U` eski tasarımda kalıyordu; kısıtı bozan onarım geçmiş sayılıyordu | onarım ayrı varyant, `Study.evaluate_density` ile yeniden değerlendirme; `split` örneğinde hacim 0,128 → 0,332 (sınır 0,15), kapı artık başarısız | `3a4a31d` |
| A-03 | `reverify`'ın BC yaklaştırmaları yalnız serbest metin notuydu | `bc_mapping`: `equivalent` / `approximate` / `unsupported` | `9c4e3e7` |
| A-04 | `post` numba'yı modül düzeyinde içe aktarıyor, hiçbir ek paket onu listelemiyordu | `post` ek paketine `numba>=0.59` | `6cd0705` |
| A-05 | Yavaş son işlem testleri yalnız su geçirmezliği kontrol ediyordu | ön kayıttaki bütün ölçütler; kayıtlı veride 7/12 geçiyor, bilinen 5 başarısızlık katı xfail | `8bcaeeb` |

### 3.2 Açık bulgular

**On dokuz bulgu kodda ya da kayıtta doğrulandı; dördü sonucun yorumunu doğrudan etkiliyor (P1).** Öncelik
`mimari_plan.md` dışında tanımlandı: P0 G0'ı engelliyor, P1 sonucu ya da raporu yanıltabiliyor, P2 belirli bir
kullanımda hata veriyor, P3 bakım ve hijyen.

| No | Öncelik | Bulgu | Kanıt [O] | Önerilen düzeltme |
|---|---|---|---|---|
| B-01 | P0 | Normal pakette 3 test kırık. `test_volume_weighted_and_unweighted`: iki kısıt da `"volume"` adında, `Study` benzersiz ad istiyor. `test_connectivity_block_ties_constraint_interfaces`: B-05'i yakalıyor. `test_pardiso_equals_superlu`: pypardiso import edilemeyince atlanmak yerine başarısız oluyor | `test_gradients.py:214`, `study.py:134-136`; `test_registry_features.py:298`; `test_core.py:206`, `tests/toporlogy/conftest.py:1-6` | birinciye `name=`, üçüncüye `ModuleNotFoundError` durumunda `skip`; ikinci B-05 ile |
| B-02 | P1 | `Result.converged` yalnız tasarım değişimine ve süreklilik takvimine bakıyor; kısıt ihlali ya da KKT kalıntısı hesaba girmiyor | `study.py:287-292` | `TerminationReason` + ayrı `feasible` alanı (G7); `converged` adı korunur, belgelenir |
| B-03 | P1 | β ardışık `run()` çağrıları arasında taşınıyor: `BetaContinuation` β'yı yalnız kurucuda β₀'a ayarlıyor, `run()` takvimi ve optimizer'ı sıfırlamıyor; ikinci koşu önceki β ve sayaçla başlıyor | `optim/schedule.py:29-35`, `study.py:246-264` | `Schedule.reset()` ve `run()` başında takvim + optimizer sıfırlama |
| B-04 | P1 | MMA alt probleminin duali L-BFGS-B ile çözülüyor, `res.success` okunmuyor; yakınsamayan dual sessizce adım üretiyor | `optim/_mma_core.py:97-99` | başarısızlıkta uyarı ve geçmişe bayrak; isteğe bağlı fail-closed |
| B-05 | P1 | Bağlantılılık bilgisinde çift sayım: `one_body` arayüz hücrelerini bağ maskesine katıyor, `config` aynı hücreleri `frozen_interface_cells` olarak bir daha sayıyor. `split` testinde CLI "84 + 36" yazıyor, eklenen hücre 60 [Ö] | `connectivity.py` `one_body` (bağ maskesine arayüz ekleme), `config.py:662-677`, `cli.py:365` | `ties` yalnız yeni bağ hücrelerini saysın |
| B-06 | P2 | Son β adımı tam `max_iter`'de olursa döngü `continue` ile biter ve `converged=True` döner; son β'da hiç iterasyon yapılmamıştır | `optim/schedule.py:49`, `study.py:281-292` | B-02'deki `TerminationReason` ile ayırt et |
| B-07 | P2 | OC, U gerektiren bir kısıtla ilk adımda hata veriyor (`constraint_fn=None`); yapılandırma bunu önceden reddetmiyor | `study.py:252-254`, `optim/oc.py:45-46` | `build_setup` aşamasında `ConfigError` |
| B-08 | P2 | `bridge`, ilk arayüzde katı hücre yoksa `argmax` boş diziyle çöküyor | `connectivity.py:798`, `:816` | açık hata iletisi ya da kökü başka arayüzden seç |
| B-09 | P2 | Yük listesi boş bir yük durumunda `BoundaryConditions.summary` `max([])` ile çöküyor; kurucu boş listeyi reddetmiyor | `bc.py:1821-1823` | kurucuda reddet ya da özette boş durumu ayrı yaz |
| B-10 | P2 | `PerCaseSupports` yalnız `solve` sağlıyor; `_LoadSolve` `solve_numpy`, `_solve_free`, `factor` istiyor. Durum başına mesnetle tasarıma bağlı hacim kuvveti çalışmaz | `bc.py:889-911`, `:1355-1368` | eksik yöntemleri ekle ya da birleşimi açıkça reddet |
| B-11 | P2 | Mesnet kümeleri nesne kimliğiyle (`id`) ayırt ediliyor; eşit ama ayrı kurulmuş mesnetler farklı küme sayılıyor ve kombinasyon hata veriyor | `bc.py:1544-1550`, `:1571` | içerik karşılaştırması |
| B-12 | P2 | `geometry.read_stl` dosyayı ancak boyutu tam 84 + 50n ise ikili sayıyor; sonunda fazladan bayt olan ikili dosya ASCII ayrıştırıcıya düşüp hata veriyor | `geometry.py:470-476` | başlıkta "solid" + ASCII denetimi; G6'da tek okuyucu |
| B-13 | P2 | `Union` ve `Intersection` şekillerin boyutunu karşılaştırmıyor (`Difference` karşılaştırıyor) | `geometry.py:316-346` | aynı denetim |
| B-14 | P2 | İstisna olursa `move` kısaltılmış değerinde kalıyor (geri yükleme `try/finally` içinde değil) | `study.py:294-295` | `try/finally` |
| B-15 | P2 | Başlangıç tasarımı ilk `Volume` kısıtından kuruluyor ama ağırlıklı hacim (`weighted=True`) için de ağırlıksız sayım kullanılıyor | `study.py:236-237`, `domain.py:179` | ağırlıklı hacimde `frac` ile başlangıç |
| B-16 | P2 | `docs/olcumler/robustness.md` T10 kontrollerinin "kayıtlarda olmadığını" yazıyor; `robustness.json` r1 ve r2 için iki boyutta T10 koşusu içeriyor. r1 yakınsamadı (oran 0,972 → 0,865, %12,4). r2'nin %0,9 değişimi yakınsama göstermiyor: iki ağın eleman sayısı 140 550 ve 143 062, ağ incelmemiş [Ö] | `robustness.md:268`, `:277`; `robustness.json` → `post` | rapor §4.2 ve §5'i veriye göre düzelt; T10 kontrolünde eleman sayısı oranını da kaydet |
| B-17 | P3 | Projeksiyon bağlantısı yoksa `BetaContinuation.beta` NaN dönüyor ve `history.jsonl`'a standart dışı `NaN` belirteci yazılıyor | `optim/schedule.py:38-39`, `record.py:51-54` | projeksiyonsuz takvimde `beta` alanını yazma |
| B-18 | P3 | `RemoteMoment.apply` nesnenin `F` ve `point` alanlarının üzerine yazıyor | `bc.py:1066-1074` | yerel değişkenle hesapla |
| B-19 | P3 | GMG ada deflasyonu düğüme komşu elemanların en büyük etiketini veriyor; iki bileşenin paylaştığı düğüm yalnız birine sayılıyor. Etkisi ölçülmedi [D] | `solvers/gmg_deflation.py:56-57` | önce ölçüm: paylaşılan düğümlü iki gövdeli test |

### 3.3 Dış incelemenin geri kalan maddeleri

Dış incelemenin A-06–A-18 maddeleri kodla karşılaştırıldı. Hepsi aşağıda yer aldı ya da bilinçli olarak bırakıldı:

- A-06, A-07, A-08 (büyük dosyalar), A-09 ve A-12 (çift geometri/STL), A-10 (iki `Symmetry`), A-11 (`Volume` adı),
  A-15 (bağlamlardaki `Any`), A-16 (GMG fizik sınıfı), A-17 (iki `local_thickness`) → §4 ve §5.
- A-13 → B-02. A-18 (CI, lisans) → G9.
- A-14 (FRAGILE kararının geçmiş sayılması) bir hata değil, `registry.py:1457`'deki açıklamaya göre bilinçli
  politika; değiştirilmedi. Robust modda katı PASS istemek bir seçenek olarak G7'de kalır.

---

## 4. Yapısal borç

### 4.1 Büyük dosyalar

**Üç dosya birden çok değişiklik nedeni taşıyor; sınırları satır aralığı olarak belli, bu yüzden bölme mekanik.**
[O, 17 Kasım 2026 satır numaraları]

| Dosya | Bölüm | Satır |
|---|---|---|
| `registry.py` | kayıt mekanizması (`Param`, `Entry`, `Registry`, `BuildContext`) | 1–443 |
| | ortak yardımcılar | 445–553 |
| | çekirdek bileşen kurucuları (`_register_core`) | 555–706 |
| | yardımcılar (`SpreadForce`, `find_bore`) | 709–869 |
| | özellik bileşen kurucuları (`_register_features`) | 872–1512 |
| | son işlem adımları (`finalize`, `checks`, `export`, `reverify`) | 1526–1713 |
| | BC çevirisi (`bc_mapping`, `bc_callables`, `selector_callable`) | 1716–1841 |
| `config.py` | şema ve varsayılanlar | 1–117 |
| | okuma, çözümleme, doğrulama | 121–430 |
| | kurulum (`Setup`, `build`, `build_setup`) | 431–785 |
| | geçersiz kılma ve tarama | 786–890 |
| | koşu (`write_resolved`, `status_of`, `run_setup`, `write_sweep_csv`, `read_run`) | 892–1053 |
| `bc.py` | seçiciler | 274–591 |
| | mesnetler ve kısıtlı elastisite | 595–911 |
| | yükler | 914–1416 |
| | yük durumları, kombinasyonlar, `BoundaryConditions`, kontroller | 1418–1880 |

### 4.2 İkili kavramlar

**Beş kavramın iki ayrı uygulaması var; ikisi aynı adla farklı anlam taşıyor (`Cylinder`, `local_thickness`).** [O]

| Kavram | Birinci | İkinci | Fark |
|---|---|---|---|
| Şekil tabanı | `geometry.Shape` (`geometry.py:83`): `sdf`, `bbox`, `inside`, `dim`, Boole işleçleri | `post.primitives.Shape` (`primitives.py:36`): `sdf`, `grad`, `project`, `offset`, yalnız 3-D | ortak alt küme yalnız `sdf` (içerisi negatif) |
| Silindir | `geometry.Cylinder(p0, p1, radius)`: kapalı uçlu, 3-D | `post.primitives.Cylinder(center, axis, radius, length=None)`: `None` sonsuz | parametre ve varsayılan uç davranışı |
| STL okuma | `geometry.read_stl`: ikili + ASCII, `merge_tol` × köşegen | `post.export.read_stl`: yalnız ikili, yalnız tam eşit köşeler birleşir | biçim ve birleştirme kuralı |
| Yerel kalınlık | `lengthscale.local_thickness`: bool alan, EDT düzeyleri, hücre biriminde dizi | `post.checks.local_thickness`: SDF, 13 yönde sırt, uzunluk biriminde sözlük | algoritma, birim, dönüş türü |
| Simetri | `bc.Symmetry`: `Roller` döndüren fonksiyon (mesnet) | `manufacturing.Symmetry`: `Tying` bağlantısı (tasarım değişkeni) | aynı ad, ilişkisiz işlev |

### 4.3 Bağımlılık yönü

İç içe aktarma grafiği AST ile çıkarıldı [Ö]. Fizik, çözücü ve geometri katmanından yapılandırmaya ya da CLI'ye
giden bağımlılık yok. Katman kuralını çiğneyen kenarlar:

- yanıtlar → optimizasyon döngüsü: `dynamics` → `study`, `robust` → `study`;
- fizik ⇄ yanıt: `fem.modal` → `dynamics` (tembel), `dynamics` → `fem.modal` (modül düzeyi);
- sınır koşulları → fizik: `bc` → `fem.elastic` (kısıtlı elastisite için);
- takvim → zincir: `optim.schedule` → `chain`.

`registry` ve `config` bütün özellik modüllerini tembel içe aktarıyor; `registry.py:697` ve `:1506-1512`'de beş hedef
metin olarak yazılmış (`"toporlogy.optim.mma:MMA"`, `"toporlogy.registry:finalize_step"` …), taşımada bunlar da
güncellenmeli.

### 4.4 Arayüz ve durum

- **Çözücü sözleşmesi iki türlü:** `Elasticity` çözücüden `factor(A)` / `solve(B)` bekliyor (`fem/elastic.py:96`,
  `:102`); `GMGSolver` `update(E, fixed)` / `solve(E, F)` sunuyor (`solvers/gmg.py:273`, `:531`). Bu yüzden GMG
  `Elasticity`'ye takılamıyor, ayrı `ElasticityGMG` fizik sınıfı olarak duruyor.
- **Bağlam nesneleri:** `EvalContext` 11 alanın 3'ü `Any` (`responses.py:21-34`); `BuildContext` 19 alanın 10'u `Any`
  (`registry.py:292-313`).
- **Paylaşılan durum:** `pardiso._STATE`, `REGISTRY` tekil nesnesi, `BetaContinuation`'ın paylaşılan zincirdeki β'yı
  değiştirmesi (B-03'ün nedeni), çözücü önbellekleri (`Elasticity._token`, `ElasticityGMG._prev`).
- **Sonuç nesneleri:** `Result` değiştirilebilir bir dataclass; onarılmış tasarım bugün son işlem durumunda
  (`registry.py` `_design`) sonuca bağlı bir sözlükle taşınıyor, kalıcı bir tür değil.

---

## 5. Göç planı

### 5.1 İlkeler

1. **Taşıma ile davranış ayrı commit'te.** Taşıma commit'i sayısal sonucu değiştirmez; değiştiren her commit nedenini
   ve baseline farkını mesajında yazar.
2. **Eski yollar çalışmaya devam eder.** `toporlogy.bc`, `toporlogy.registry`, `toporlogy.config` gibi her eski modül,
   yeni yerden yeniden dışa aktaran ince bir modül olarak kalır. `DeprecationWarning` ancak G8'de ve CHANGELOG'da
   ilan edilerek eklenir.
3. **Her commit aynı kabul kapısından geçer:**
   - baseline karşılaştırması (G0, `-m baseline`, ~3 dakika) bit düzeyinde geçer;
   - `pytest tests/toporlogy -m "not slow"` 0 hata ile biter;
   - ruff temiz kalır, mypy hata sayısı artmaz (bugün 230);
   - `python -m toporlogy list` çıktısı değişmez;
   - API yüzeyi anlık görüntüsündeki her ad eski yolundan içe aktarılabilir;
   - içe aktarma kuralı testi yeni yasak kenar bulmaz.
4. **İçe aktarma sırası korunur:** `__init__.py` PARDISO'yu torch'tan önce yükler; taşınan hiçbir modül paket
   içe aktarılırken torch'u bundan önce yüklememeli. Bu, G0'daki duman testine bağlanır.
5. **Dal:** her aşama `p0-result-consistency` üzerine kendi dalında; `main`'e alma kararı aşama sonunda verilir.

### 5.2 Aşamalar

#### G0 — Güvenlik ağı (davranış değişmez)

Amaç: sonraki her adımın "hiçbir şey değişmedi" iddiasını bir komuta bağlamak.

1. **B-01'deki üç testi düzelt** (yalnız test değişikliği). PARDISO testi pypardiso yoksa atlanır, torch PARDISO'dan
   önce yüklendiyse yine başarısız olur (testin asıl amacı bu).
2. **Baseline aracı:** `tests/toporlogy/baseline/` altında, yedi örnek YAML'ı `run.max_iter=10` ile koşturup her koşu
   için amaç, kısıt değerleri, iterasyon sayısı, `x` ve `xphys`'in SHA-256 özeti, `summary.json` anahtarları ve STL
   özetlerini JSON'a yazan bir betik; bu JSON'a karşı bit düzeyinde karşılaştıran, `baseline` işaretli bir test
   (CPU, SuperLU; gerekçe §5.3).
3. **API yüzeyi anlık görüntüsü:** her `toporlogy.*` modülünün açık adlarının listesi ve her adın tanımlandığı nesne.
   Taşımadan sonra eski yoldan alınan ad yeni yerdeki nesnenin aynısı olmalı.
4. **`toporlogy list` anlık görüntüsü** (bileşen adları, parametreler, birimler, sıra).
5. **İçe aktarma kuralı testi:** AST ile iç grafiği çıkarır; §4.3'teki bugünkü kenarlar izinli listede, yeni kenar
   eklenirse test başarısız olur. Liste aşamalar ilerledikçe daraltılır.
6. **İçe aktarma sırası duman testi:** temiz bir alt süreçte `import toporlogy` sonrasında PARDISO durumu ve
   `sys.modules` sırası.

Kabul: baseline aracı iki ayrı süreçte aynı çıktıyı verir; normal paket 0 hata.

#### G1 — Doğruluk düzeltmeleri (davranış değişir, taşıma yok)

B-02…B-15, her biri önce başarısız olan bir testle, ayrı commit olarak; düzeltmeler bugünkü dosyalarda yapılır ki diff
okunur kalsın ve taşıma sonradan saf olsun. Sıra: B-05 (G0'daki testle birlikte), B-03, B-04, B-07, B-14, B-15, B-08,
B-09, B-10, B-11, B-12, B-13, ardından P3 hijyeni B-17 ve B-18. B-02 ve B-06 sonuç türüne dokunduğu için G7'ye
kalır; B-16 belge düzeltmesi olarak ayrı commit; B-19 önce ölçülür (paylaşılan düğümlü iki gövdeli test), etki
çıkarsa düzeltilir, çıkmazsa ölçüm kaydı ve bir regresyon testi olarak kalır. Sayısal sonucu değiştiren commit (B-03, B-04, B-15 olası) baseline'ı yeniden üretir ve farkı mesajında
listeler.

#### G2 — `registry.py`'yi böl (saf taşıma)

| Hedef | İçerik (bugünkü satırlar) |
|---|---|
| `configuration/registry.py` | kayıt mekanizması (1–443) |
| `configuration/components/core.py` | çekirdek kurucular (555–706) |
| `configuration/components/{geometry,boundary,design,physics,responses,schedules}.py` | özellik kurucuları (872–1512), alan başına |
| `configuration/post_steps.py` | son işlem adımları (1526–1713) |
| `verification/boundary_mapping.py` | `bc_mapping`, `bc_callables`, `selector_callable` (1716–1841) |

Kayıt sırası korunur (CLI `list` çıktısı sıraya bağlı). `toporlogy.registry` bütün eski adları ve PEP 562
`__getattr__`'ı yeniden dışa aktarır. Metin hedefleri yeni yollara güncellenir.

#### G3 — `config.py`'yi böl (saf taşıma)

`configuration/schema.py` (şema, varsayılanlar), `configuration/resolve.py` (okuma, çözümleme, geçersiz kılma, tarama
ayrıştırma), `configuration/build.py` (`Setup`, `build`, `build_setup`); `execution/runner.py` (`run_setup`,
`status_of`), `execution/runs.py` (`write_resolved`, `read_run`, `write_sweep_csv`). `toporlogy.config` eski adları
yeniden dışa aktarır.

#### G4 — `bc.py`'yi böl (saf taşıma)

`boundary/selectors.py`, `boundary/supports.py`, `boundary/constrained.py` (`ConstrainedElasticity`,
`PerCaseSupports`), `boundary/loads.py`, `boundary/cases.py` (`LoadCase`, `Combination`, `BoundaryConditions`),
`boundary/checks.py` (bileşke, rijit-gövde, özet). `bc` → `fem.elastic` bağımlılığı yalnız `constrained.py`'de kalır.

#### G5 — Katman paketleri (saf taşıma, kapsamı K1 kararına bağlı)

İki seçenek [K]:

- **K1-a, sınırlı (önerilen):** yerleşik ve tutarlı paketler (`fem/`, `solvers/`, `optim/`, `post/`, `references/`)
  adlarını korur; yalnız düz modüller gruplanır: `domain` → `core/`, `chain`, `interp`, `manufacturing` → `design/`,
  `study`, `responses`, `robust`, `lengthscale`, `connectivity`, `dynamics`, `checks` → `optimization/`, `record`,
  `cli` → `execution/`. Gerekçe: `fem` → `physics` gibi yeniden adlandırmalar yalnız ad değiştirir, her biri bir ince
  modül daha ekler.
- **K1-b, tam:** dış incelemedeki hedef ağacın tamamı (`physics/`, `optimization/algorithms/`, `postprocessing/`,
  `verification/` …).

Her alt paket ayrı commit. §4.3'teki ihlallerden `dynamics`/`robust` → `study` ve `fem.modal` ⇄ `dynamics`, ortak
türler `optimization/types.py`'ye çekilerek kaldırılır; bu da saf taşımadır.

#### G6 — Geometri ve IO'yu birleştir (davranış değişebilir)

1. Ortak `ImplicitSolid` sözleşmesi: `sdf(points)`, `dim`, `bounds()`. `geometry.Shape` ve `post.primitives.Shape` bunu
   uygular.
2. Tek kanonik `Cylinder`; öteki adaptör olarak kalır, dönüşüm testleri (eksen, yarıçap, uç, SDF değerleri).
3. Tek STL okuyucu/yazıcı (`geometry/io/stl.py`): ikili + ASCII, açık birleştirme toleransı, B-12 düzeltilmiş.
4. İki `local_thickness` açık adlar alır (`voxel_local_thickness`, `surface_ray_thickness`); eski adlar takma ad.

Kabul: son işlem testleri ve baseline değişmez ya da fark açıklanır.

#### G7 — Sonuç ve durum arayüzü

- `TerminationReason` (`DESIGN_CHANGE`, `MAX_ITER`, `SCHEDULE_LAST_STEP`, `SOLVER_FAILURE`), `Result.feasible`,
  isteğe bağlı MMA KKT kalıntısı (B-02, B-06).
- `RunStatus` sınıflandırması: `status_of`'un metin çıktısı korunur, yanına makinece okunur alan eklenir.
- `DesignVariant`: A-02'deki onarılmış tasarım durumunun kalıcı türü (`source`, `xphys`, `changes`, `evaluation`).
- `OptimizationResult` = `Result` takma adı; `Result` ad olarak kalır.
- Seçenek: robust modda bağlantılılık kapısının katı PASS istemesi (A-14).

#### G8 — Adlandırma ve şema sürümü

- Takma adlar: `SymmetrySupport` / `SymmetryTying`, `VolumeFraction`, `EvaluationContext`; eski adlar kalır.
- YAML `schema_version: 2`: açık birimli alan adları (`youngs_modulus`, `element_size`, `design_change_tolerance` …).
  Takma adları yalnız normalize aşaması çözer; aynı alan eski ve yeni adla birlikte verilirse hata.
- `toporlogy migrate-config` komutu; sürüm 1 dosyaları okunmaya devam eder.
- Kabul: yedi örneğin sürüm 1 ve sürüm 2 hâlleri aynı normalize çıktıyı ve aynı baseline'ı verir.
- Bu aşamanın gerekliliği K4 kararına bağlı.

#### G9 — Sürüm altyapısı

- GitHub Actions: Linux + Windows, Python 3.11 ve 3.12, `pytest tests/toporlogy -m "not slow"`, ruff, mypy (artış
  yasak), temiz ortamda `pip install .` ve `.[post,mesh,dev]`; GPU ve yavaş testler elle tetiklenen ayrı iş.
- İçe aktarma kuralı testi CI'da zorunlu; izinli liste boşalmış olmalı.
- CHANGELOG, sürüm 0.3.0, lisans (K5).

### 5.3 Determinizm ölçümü

Baseline'ın bit düzeyinde mi toleransla mı karşılaştıracağı ölçümle belirlendi: yedi örnek iki ayrı süreçte
`run.max_iter=10` ile koşturuldu, `x` ve `xphys`'in SHA-256 özeti, amaç, kısıt değerleri ve STL dosyalarının özeti
karşılaştırıldı [Ö, 17 Kasım 2026, `8bcaeeb`].

**Yedi örneğin yedisi iki süreçte bit düzeyinde aynı çıktı verdi**; `lbracket_bore3d`'nin iki STL dosyası da
(finalize + export) aynı. Bütün koşular 10 iterasyonda "not converged" durumuyla bitti (beklenen). İki geçişin toplam
süresi 373 s, tek geçiş ~3 dakika; en uzunları `lbracket_bore3d` (71–87 s) ve `symmetric_am3d` (48–51 s).

Ölçümün sınırı: PARDISO yüklü değildi (doğrudan çözücü SuperLU), CPU'da, tek makinede. PARDISO'nun çok iş parçacıklı
çözümü ve CUDA GMG için determinizm ölçülmedi [D]. Karar [K]: baseline CPU + SuperLU'da bit düzeyinde karşılaştırılır
(`DirectSolver("splu")` zorlanarak); PARDISO ve GPU koşuları baseline'a girmez. Tek geçiş 3 dakika sürdüğü için
baseline testi normal pakette değil, ayrı bir `baseline` işaretiyle her taşıma commit'inde koşulur.

### 5.4 Büyüklük

Süre tahmini verilmedi; aşamaların büyüklüğü değişen satır ve commit sayısıyla ölçülür.

| Aşama | Tür | Tahmini commit | Dokunulan kod |
|---|---|---:|---|
| G0 | test altyapısı | 4–6 | yalnız `tests/` |
| G1 | davranış | 15–16 | küçük, dağınık |
| G2 | taşıma | 3–4 | `registry.py` (1854 satır) |
| G3 | taşıma | 2 | `config.py` (1053) |
| G4 | taşıma | 2–3 | `bc.py` (1881) |
| G5 | taşıma | 4–8 (K1'e bağlı) | 13–30 modül |
| G6 | davranış | 4 | `geometry.py`, `post/primitives.py`, `post/export.py`, `post/checks.py`, `lengthscale.py` |
| G7 | arayüz | 3–4 | `study.py`, `config.py`/`execution`, `registry`/`configuration` |
| G8 | arayüz | 3–5 | `configuration/` |
| G9 | altyapı | 2–3 | `.github/`, kök dosyalar |

---

## 6. Yetenek ve kanıt yol haritası

Göçten bağımsız, ama G0'ın güvenlik ağından ve G7'nin sonuç türlerinden yararlanır. Öncelik sırasıyla:

| No | İş | Neden | Bağımlılık |
|---|---|---|---|
| V1 | Bağımsız referanslar: DTU top88 MATLAB, Top3D, analitik kiriş ve NAFEMS elastisite | bugünkü kâhinler deponun kendi portları (`mimari_plan.md` §4); 1e-11 uyum regresyondur, bağımsız doğrulama değil | G0 |
| V2 | T10 ağ yakınsaması: gerçekten inceltilen en az üç ağ, eleman sayısı oranı kaydı | r1 yakınsamadı, r2'nin ağı incelmedi (B-16) | gmsh kurulu ortam |
| V3 | Son işlem başarısızlıklarının nedeni: r2–r4 Hausdorff (tek yönlü, son yüzey → ham), p05 kesin hacim | nedenleri araştırılmadı (`robustness.md` §4.1) | — |
| V4 | Sonlandırma ölçütleri: KKT kalıntısı, kısıt uygunluğu, `TerminationReason` | B-02 | G7 |
| F1 | 3-D gerilme kısıtı (P-norm / KS, gevşetme, adjoint) | `mimari_plan.md` A3 açığı; `topo_oracles/stress_topo.py`'deki eski gerilme sürücüsü regresyon kâhini olarak hazır, testlerde kullanılmıyor | G1 |
| F2 | Hacim bütçesine duyarlı bağlantılılık: onarımdan sonra yeniden optimizasyon ya da bütçeli köprü | köprü onarımı hacmi 1,07–2,61 × vf'ye çıkarıyor (`robustness.md:179`), `split` örneğinde 0,128 → 0,332 | A-02 |
| F3 | Robust en küçük boşluk: iç köşelerde 2,69 (hedef 4) | `mimari_plan.md` §4 | — |
| F4 | Çözücü sözleşmesini birleştirme: GMG'yi `Elasticity`'ye takılabilir çözücü yapmak; sessiz makinede A2 ölçümü | §4.4; A2 açık maddeleri | G5 |
| F5 | Yeni fizik: burkulma, termoelastisite | `mimari_plan.md` A4 | F4 önerilir |
| F6 | CAD: B-rep / STEP | kapsam dışı (`post/export.py:5`); ayrı araştırma projesi | — |

`mimari_plan.md` §5'teki "yapılmayacaklar" (yapısız ağda optimizasyon, level-set/BESO, MPI) geçerliliğini korur.

---

## 7. Riskler

| Risk | Dayanak | Önlem |
|---|---|---|
| Taşıma sırasında torch'un PARDISO'dan önce yüklenmesi (Windows'ta süreç çöker) | `__init__.py:9-16` ölçülmüş çökme | G0'daki duman testi; taşınan modül paket yüklenirken torch'u erken içe aktarmamalı |
| Kayıt sırasının değişmesi, CLI `list` çıktısının ve hata önerilerinin kayması | kayıt sırası çağrı sırasına bağlı | `list` anlık görüntüsü |
| İnce yeniden dışa aktarım modüllerinin döngüsel içe aktarma yaratması | `registry`/`config` her şeyi tembel içe aktarıyor | içe aktarma kuralı testi + temiz alt süreçte içe aktarma |
| Saf taşıma sanılan bir değişikliğin sayıyı değiştirmesi | — | baseline karşılaştırması; davranış commit'i ayrı |
| Kullanıcı betiklerinin ve `studies/` altındaki çalışmaların kırılması | eski yollar | eski yollar en az bir sürüm boyunca korunur |
| Bit düzeyi karşılaştırmanın başka bir makinede ya da PARDISO ile yanlış alarm vermesi | CPU + SuperLU'da ölçüldü (7/7 aynı); PARDISO ve CUDA ölçülmedi | baseline SuperLU'ya zorlanır; başka makinede ilk koşuda baseline yeniden üretilir |

## 8. Açık kararlar

| No | Karar | Seçenekler | Öneri |
|---|---|---|---|
| K1 | Katman paketlerinin derinliği (G5) | sınırlı / tam | sınırlı |
| K2 | `p0-result-consistency`'nin `main`'e alınma zamanı | şimdi / G0 sonrası / göç sonunda | G0 sonrası |
| K3 | Eski yolların kullanımdan kaldırılma süresi | bir sürüm / 1.0'a kadar | 1.0'a kadar |
| K4 | YAML `schema_version: 2` (G8) gerekli mi | evet / takma adlarla yetin | önce kullanım ihtiyacına bakılmalı |
| K5 | Lisans | — | depo sahibinin kararı |
| K6 | `pytest testpaths` `tests/toporlogy` ile sınırlansın mı | evet / yerel düzen korunsun | CI komutunda açık yol; `testpaths` yerel kullanıma bırakılır |

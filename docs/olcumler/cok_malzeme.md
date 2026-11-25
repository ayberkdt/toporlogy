# Çok malzemeli yoğunluk tabanlı optimizasyon — doğrulama kaydı (25 Kasım 2026)

Dal: `claude/multi-material-2b0745` (koordinasyon notunda `feat/multi-material`; taban `integrate/parallel-physics`
= d566290). Etiketler `literatur_2026-11-21.md` ile aynı: **[Ö]** ölçüldü, **[O]** koddan okundu, **[K]** kaynaktan
okundu, **[H]** hafızadan, doğrulanmadı.

## 0. Ön kayıt (2026-11-25 17:56:53 +0300, ilk koşudan önce yazıldı)

Bu bölüm yazıldığında çok malzeme kodu yoktu; aşağıdaki hiçbir test koşulmamıştı. Sonradan eklenen her şey
"post hoc" diye işaretlenir. Tutmayan ön kayıt testte `xfail(strict=True)` olur ve gerekçesi docstring'e yazılır.

### Seçilen interpolasyon (gerekçe §1'de)

m katı faz + boşluk, fazlar E'ye göre artan sırada (E_1 ≤ … ≤ E_m), eleman başına m kutu değişkeni t_j ∈ [0, 1]:

    E^(0) = E_min,   E^(j) = E^(j−1) + t_j^p (E_j − E^(j−1)),   E = E^(m) · frac

Hacim kesirleri (cezasız, "yukarıdan çubuk kırma"): η_m = t_m, η_j = t_j Π_{k>j}(1 − t_k), η_boşluk = Π_k (1 − t_k);
Σ η = 1 yapı gereği. Bu, Gangl ve ark. (2026, arXiv:2605.11994, §6.1) yinelemesinin t_j = η̃_j / S_j yazımıdır.

### Ön kayıtlı kontroller ve toleranslar

| # | Kontrol | Tolerans |
|:--|:--|:--|
| V1a | Tek faz: `MultiPhaseSIMP` E alanı = `SIMP` E alanı (rastgele x̄, iç nokta ve 0/1) | bit düzeyinde (`torch.equal`) |
| V1b | Tek faz: 30×10 MBB, OC, 20 adım; f0 geçmişi ve son x̄ | bit düzeyinde |
| V1c | Tek faz YAML (`material.phases` bir faz + `multiphase_simp`) = varsayılan YAML; son x̄ ve amaç | bit düzeyinde |
| V2a | İki faz aynı E: t_1 = 0 iken E = SIMP(t_2), t_2 = 0 iken E = SIMP(t_1) | ≤ 1e-14 göreli |
| V2b | İki faz aynı E: 0/1 tasarımda fazların yer değiştirmesi uyumu değiştirmez | ≤ 1e-14 göreli |
| V2c | İki faz aynı E, katı hacim ≤ 0,5, 60×20 MBB, rmin 1,5, projeksiyonsuz, MMA 150 adım: uyum, tek malzeme MMA uyumuna göre | ≤ %3 göreli |
| V3a | Paralel iki malzemeli çubuk (ν = 0,3, iki katman, tutarlı düğüm kuvvetleri): FE uyumu = P²L / (E_A A_A + E_B A_B) | ≤ 1e-10 göreli |
| V3b | Seri iki malzemeli çubuk (ν = 0): FE uyumu = P² (L_A/E_A + L_B/E_B) / A | ≤ 1e-10 göreli |
| V4 | Sonlu fark gradyanı: m = 3, kutu olmayan alan (delik, keep, void, kısmi frac), filtre + projeksiyon (β = 2) + passive; uyum, faz hacmi, katı hacmi, kütle, maliyet | ≤ 1e-6 göreli (test_gradients kuralı; uyumda h = 1e-3 Richardson) |
| V5 | Gangl ve ark. 2026 §6.1 2-B kıyası: kaynakta uyum sayısı yok → NOT VERIFIED; indirgenmiş çözünürlükte (96×32) yalnız çalışırlık: üç faz hacim kısıtı feasibility_tol 1e-3 içinde, sonuç raporlanır, kıyaslanmaz | kısıtlar ≤ 1e-3 göreli ihlal |
| V6 | m ≥ 2 iken OC, SiMPL, `Volume`, desteklenmeyen zincir bağı açık hata (ValueError / ConfigError) | hata yükseltilir |
| V7 | YAML uçtan uca (2-B, m = 2): `summary.json` faz hacimleri = `phase_volume` yanıt değerleri; 3-B faz yüzeyi (voksel sınırı STL) kapalı ve sardığı hacim = voksel sayısı · h³ | ≤ 1e-12 göreli |

Ayrıca: `test_baseline.py` bit düzeyi kayıtları değişmeden geçmeli; `python -m pytest -m "not slow" tests/toporlogy`,
ruff ve mypy temiz.

## 1. İnterpolasyon seçimi ve gerekçe

Okunan kaynak yalnız Gangl, Keith, Kim, Lazarov & Surowiec (2026), arXiv:2605.11994, HTML v2. Metni bu oturumun bir
alt ajanı okudu; formülleri ve sayıları o aktardı [K: alt ajan aktarımı; metni kendim görmedim]. Aktarılanlar:

- §2 Eq. (1b)–(1c): değişkenler simplekste (Σ η_i = 1, η_i ≥ 0).
- §6.1 yineleme: E^(j) = E^(j−1)[1 − (η̃_j/S_j)^p] + E_j (η̃_j/S_j)^p, S_j = Σ_{i≤j} η̃_i, p = 3, boşluk E = 1e-6.
  Yinelemenin başlangıç değeri E^(1) metinde yazmıyor.
- İyileştirici ayna inişi; MMA ve OC metinde geçmiyor.

Diğer adaylar ve durumları:

- DMO, Stegmann & Lund (2005), IJNME 62(14):2009–2027, doi:10.1002/nme.1259. Künye Gangl 2026 kaynakçasından
  [K: künye]; şema ve ağırlıklarının toplamının 1 olmadığı [H].
- Hvejsel & Lund (2011), SMO 43(6):811–825, doi:10.1007/s00158-011-0625-z. Künye aynı yerden [K: künye]; eleman
  başına doğrusal kısıtlı SIMP/RAMP genellemesi [H].
- Sigmund & Torquato (1997), üç fazlı "extended SIMP" E = ρ^p [y^p E_1 + (1 − y^p) E_2] [H].
- Zuo & Saitou (2017), ordered SIMP: eleman başına tek değişken [H].

Gangl 2026 yinelemesini seçtim. Gerekçeler:

1. Metni okunan tek kaynak bu.
2. t_j = η_j / S_j dönüşümü simpleksi kutuya taşıyor (S_m = 1, S_{j−1} = S_j (1 − t_j)). Böylece yalnız kutu
   sınırı kalıyor: MMA ve GCMMA değişmeden çalışıyor, eleman başına simpleks kısıtı gerekmiyor [O].
3. m = 1'de değiştirilmiş SIMP'in işlemlerine aynı sırayla iniyor. V1 bunu bit düzeyinde ölçtü [Ö].
4. Σ η = 1 yapı gereği sağlanıyor; faz hacmi, kütle ve maliyet η'da doğrusal [O].

Makaleden farklar [O]:

- Filtre ve projeksiyon t alanlarına uygulanıyor; makale η'yı filtreliyor.
- E_min = emin_ratio · E_m.
- Fazlar E'ye göre artan sırada olmalı. Azalan sırada karışım, doğrusal karışımdan daha sert çıkıp kayırılır
  (`design/phases.py` notları).
- Keep elemanları en sert faza gidiyor, çünkü `Passive` her alanı 1 yapıyor.

## 2. Ne eklendi [O]

| Dosya | Değişiklik |
|:---|:---|
| `design/phases.py` (yeni) | `Phase`, `MultiPhaseSIMP` (`fractions`, `solid`), `phase_fractions`, `phase_labels`, `phase_report`, `phase_start` |
| `design/chain.py` | `Embed(domain, n_fields=1)`: m > 1'de alan öncelikli m·n_design vektörü (m, *shape) yapıyor ve `n_in` bildiriyor. `DensityFilter` (m, *shape) girdide her alanı ayrı süzüyor. `HeavisideProjection` ve `Passive` zaten öğe bazlı ve yayınımlı; yalnız docstring değişti |
| `optimization/responses.py` | `PhaseVolume(phases=None, ad ya da liste; weighted)`, `Mass`, `Cost`; `Volume` çok alanda açık hata veriyor |
| `optimization/study.py` | Çok fazda denetim: alan sayısı, izinli bağlar (Embed, DensityFilter, HeavisideProjection, Passive), OC ve SiMPL reddi, izinli yanıtlar (Compliance, PhaseVolume, Mass, Cost). Faz hacmi sınırlarından tekdüze başlangıç; `settings()["phases"]` |
| `configuration/schema.py` | `material.phases` parametresi ve şema metni |
| `configuration/components/core.py` | `embed` kurucusu faz sayısını okuyor; `multiphase_simp`; `phase_volume`, `mass`, `cost` yanıtları |
| `configuration/components/design.py` | `phase_report` son işlem adımı: `phases.npz` (etiket, kesirler), `phases.json`, 3-B'de faz başına STL |
| `configuration/build.py` (ortak, eklemeli) | `material.phases` başka interpolasyonla verilirse ConfigError; m > 1'de `phase_report` dışı son işlem ConfigError |
| `execution/runner.py` (hiçbir listede yok; 8 satır eklendi) | m > 1'de `summary.json`'a `phases` ve fazlı gri payı yazılıyor |
| `post/export.py` | `voxel_surface`, `export_phase_surfaces` |
| `references/_topopt_methods.py` | `gangl_2026` kaydı |
| `tests/toporlogy/test_multi_material.py` (yeni), `docs/olcumler/cok_malzeme.py` (yeni) | Kontroller ve JSON üreticisi |

## 3. Sonuçlar [Ö]

Kaynak: `PYTHONPATH=src python docs/olcumler/cok_malzeme.py` (2026-11-25T18:05:09+0300, taban d566290 + çalışma
ağacı) → `docs/olcumler/cok_malzeme.json`. `python -m pytest tests/toporlogy/test_multi_material.py -s` de aynı
değerleri basıyor.

**Ön kayıtlı 13 kontrolün hepsi ilk koşuda tuttu. Yayımlanmış kıyas değeri yok (V5).**

| Kontrol | Ne | Ön kayıtlı tolerans | Ölçülen | Sonuç |
|:---|:---|:---|:---|:---|
| V1a | Tek faz E alanı = SIMP | bit düzeyi | torch.equal doğru (2 tasarım × 2 parametre) | PASS |
| V1b | Tek faz MBB 30×10, OC 20 adım | bit düzeyi | f0 geçmişi ve son x̄ aynı | PASS |
| V1c | Tek faz YAML = varsayılan YAML | bit düzeyi | son x̄ ve amaç aynı | PASS |
| V2a | Aynı E, bir alan 0 → SIMP | ≤ 1e-14 | 0 | PASS |
| V2b | Aynı E, faz takası | ≤ 1e-14 | 0 | PASS |
| V2c | Aynı E, iki faz MMA optimumu / tek malzeme | ≤ %3 | %1,9 (214,66 / 210,66) | PASS |
| V3a | Paralel çubuk (ν = 0,3) | ≤ 1e-10 | 3,1e-15 | PASS |
| V3b | Seri çubuk (ν = 0) | ≤ 1e-10 | 8,3e-16 | PASS |
| V4 | Sonlu fark gradyanı (6 yanıt) | ≤ 1e-6 | en büyük 1,6e-7 (maliyet) | PASS |
| V5 | Gangl 2026 §6.1, 96×32: kısıt ihlali | ≤ 1e-3 | 0; uyum 0,00681 (kıyas yok) | PASS · NOT VERIFIED |
| V6 | OC, SiMPL, Volume, uyumsuz zincir reddi | hata | 9 durumun hepsi hata verdi | PASS |
| V7a | YAML özet faz hacmi = yanıt değeri | ≤ 1e-12 | 1,4e-16 | PASS |
| V7b | Voksel STL kapalı, hacim = n·h³ | ≤ 1e-12 | kapalı; 0 | PASS |

Notlar:

- V2c: iki fazlı çözüm %1,9 daha kötü, ön kayıtlı %3'ün içinde. Aynı E'de iki alanlı parametreleme
  E = E_0 − (E_0 − E_min)(1 − t_1^p)(1 − t_2^p) tek malzemenin s^p'sinden farklı. Bu yüzden aynı yerel optimum
  beklenmiyordu.
- V4 ayrıntısı: uyum 3,2e-9 (h = 1e-3 Richardson), faz hacmi 2,2e-8, katı hacmi 6,1e-8, kütle 6,3e-8, maliyet
  1,6e-7, ağırlıklı a+c hacmi 7,1e-9.
- V5: 200 adımda max_iter'de durdu. Faz hacimleri 0,05999 / 0,11999 / 0,12000; gri payı 0,33 (projeksiyon yok).
  Makalede bu problem için uyum sayısı yok: alt ajan metinde, başlıklarda ve tabloda aradı, bulamadı. Bu yüzden
  **NOT VERIFIED**. Şekil 3'ün eksen işaretlerinden değer okunmadı.
- V5'in makaleden farkları: 96×32 çözünürlük (makalede h = 2^-8); konik filtre R = 0,06 (makalede PDE filtresi,
  ε = 0,06/(2√3); R = 2√3 ε ilişkisi [H]); disk yükü içindeki düğümlere eşit bölündü; iyileştirici MMA.
- Post hoc test düzeltmesi: V6'nın son adımı ilk koşuda yanlış nedenle hata verdi (`export` adımı önce `finalize`
  istiyor). Test `measure_lengthscale` ile yeniden yazıldı; tolerans ve beklenen davranış aynı kaldı.
- Post hoc: `test_phase_start_and_labels` ön kayıtta yoktu.

## 4. Yapılmayanlar ve sınırlar

- **Farklı ν ya da anizotropi.** Her fazın kendi D matrisi eleman başına D ister; bu iş A dalında. Ona bağlı kod
  yazılmadı. **Birleştirmeden sonra** `MultiPhaseSIMP`, eleman başına D karışımına genişletilebilir.
- Çok malzemeli SiMPL (simpleks gizli değişkeni, Gangl 2026 §3–5) uygulanmadı. m > 1'de `SiMPL` ve `OC` açık hata
  veriyor.
- m > 1'de şunlar desteklenmiyor: imalat ve simetri bağları, Helmholtz filtresi, robust, connectivity, gerilme,
  burkulma ve modal yanıtlar, `phase_report` dışındaki son işlem adımları (finalize ve export tek yoğunluk alanı
  okuyor). Hepsi açık hata veriyor; sessizce yanlış sonuç veren yol yok.
- Keep bölgesine faz seçilemiyor; keep her zaman en sert faz.
- Faz yüzeyleri yalnız 3-B'de ve voksel sınırı olarak yazılıyor: düzgünleştirme yok, 2-B'de STL yok.
- `execution/runs.py` çalıştırma listesindeki `grey_fraction`, `result.npz`'deki bütün alanları düzleştirerek
  hesaplanıyor ve fazlı kuralı kullanmıyor. Yalnız listeleme etkileniyor; `summary.json` doğru.
- Yayımlanmış bir çok malzeme değeriyle sayısal kıyas yapılamadı (V5: NOT VERIFIED).

## 5. README için önerilen metin

> **Çok malzeme (aynı ν, farklı E, yoğunluk ve maliyet).** `material.phases: [{name, E, density, cost}]` E'ye göre
> artan sırada verilir ve `interpolation: multiphase_simp` seçilir. Eleman başına faz sayısı kadar tasarım alanı
> açılır. Interpolasyon, Gangl ve ark. (2026, arXiv:2605.11994, §6.1) yinelemeli SIMP'inin kutu değişkenli
> yazımıdır. Yanıtlar: `phase_volume` (bir fazın ya da katının hacim kesri), `mass`, `cost`, `compliance`.
> İyileştirici MMA ya da GCMMA olmalı; OC ve SiMPL açık hatayla reddedilir. `phase_report` son işlem adımı baskın
> faz etiketini ve faz başına hacim, kütle ve maliyeti yazar; 3-B'de faz başına voksel STL de çıkarır. Tek faz
> verilince sonuç bugünkü SIMP yoluyla bit düzeyinde aynıdır. Doğrulama: seri ve paralel iki malzemeli çubukta
> kapalı formla fark 1e-14 düzeyinde, gradyanlarda sonlu farkla fark ≤ 1e-6. Yayımlanmış bir çok malzeme değeriyle
> kıyas yapılmadı (`docs/olcumler/cok_malzeme.md`). Fazlara ayrı ν ya da anizotropi verilemiyor.

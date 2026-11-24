# toporlogy — mimari ve yol haritası (sürüm 2, 14 Kasım 2026)

Amaç: deney yapmaya açık, kendi kontrolümüzde, bağımsız bir yoğunluk tabanlı topoloji optimizasyonu kütüphanesi.
Yeni bir amaç, kısıt, filtre, fizik ya da optimizer eklemek bir dosya ve bir test kadar kısa sürmeli; büyük 3-D
problemler bir dizüstünde makul sürede koşmalı.

Etiketler: **[Ö]** ölçüldü (dosya/komut verilmiş), **[O]** koddan okundu, **[L]** literatür, bu çalışmada okunmadı,
**[K]** karar.

---

## 1. Kararlar

| # | Karar | Durum | Gerekçe / kanıt |
|---|---|---|---|
| D1 | Dizi altyapısı torch; türevler autograd, doğrusal çözüm tek özel adjoint | uygulandı | zincir ve yanıtlar yalnız ileri hesapla yazılıyor; her parça sonlu farkla test ediliyor (`tests/toporlogy/test_gradients.py`) [Ö] |
| D2 | Tam sınır kutusu üzerinde düzenli ızgara + maske + kısmi hacim | uygulandı | eski voksel sürücüsüyle aynı numaralama; maskeli alanlarda regresyon ≤ 1e-11 [Ö] |
| D3 | Hızlı çözücü: matris tutmayan GMG-PCG | uygulandı | A0'da kaldı; A2'de K-döngüsü + ada deflasyonu ile kapı geçti: 49k'da PARDISO'dan 7,2×, 1e6 rastgele alan 361 iterasyonda 9,9e-9 (`docs/olcumler/A2.md`, yüklü makinede) [Ö] |
| D4 | Paket yeri `src/toporlogy/` | uygulandı | `@cite` ve kaynakça altyapısı ortak |
| D5 | git deposu | uygulandı | yayımlanan depo yalnız kütüphaneyi içerir (`.gitignore` beyaz liste) |

## 2. Katmanlar ve modüller

```
Study (bildirimsel kurulum: configuration/ (şema, çözümleme, kurulum, kayıt, bileşenler); koşu ve CLI: execution/)
  ├── Domain           core/domain.py        ızgara, frac, keep/void, numaralama;  geometry.py  CSG / STL / delik bulma
  ├── Sınır koşulları  boundary/             seçiciler, mesnetler, yükler, yük durumları, rapor
  ├── Chain            design/chain.py       Embed ► DensityFilter | HelmholtzFilter (pde_filter.py) ► [AM / Casting] ► Projection ► Passive
  │                    design/manufacturing.py  simetri / ekstrüzyon / desen bağlama, AM, döküm yönü
  ├── Interpolation    design/interp.py      SIMP, RAMP;  optimization/dynamics.py  iki aralıklı kütle/rijitlik
  ├── Physics          fem/elastic.py        H8, H8IM, Q4 + autograd çözüm;  fem/heat.py;  fem/modal.py
  │     └── Solvers    solvers/direct.py (PARDISO/SuperLU),  solvers/gmg.py (A2)
  ├── Responses        optimization/         responses, robust, lengthscale, connectivity, dynamics,
  │                                          stress, buckling (fem/buckling.py), compose
  ├── Yöntem araçları  optimization/         aggregates (toplama), local (toplanmış / artırılmış Lagrange), benchmark (0/1)
  ├── Optimizer        optimizers/           oc.py, mma.py, gcmma.py, simpl.py, schedule.py
  ├── Recorder         execution/record.py   koşu klasörü, git sürümü, geçmiş, anlık görüntüler
  └── Post             post/                 SDF, düzenleme, pürüzsüzleştirme, kontroller, T10 doğrulama, dışa aktarma
```

Her modülün, sınıfın ve yapılandırma bileşeninin tek adı vardır: eski yollar için takma ad modülü ya da ikinci ad
tutulmaz (18 Kasım 2026 kararı; `tests/toporlogy/test_architecture.py` bunu denetler).

Kurallar: yanıt yazarı yalnız ileri hesabı yazar; her bağlantı ve yanıt `optimization.diagnostics.gradcheck` testinden geçer (bağıl
hata ≤ 1e-6, float64); eski sürücüler yalnız `tests/toporlogy/topo_oracles/` altında, regresyon kâhini olarak durur.

## 3. Aşamalar

| Aşama | İçerik | Durum | Kanıt |
|---|---|---|---|
| A0 | Çözücü kapısı (GMG prototipi) | ölçüldü: **kaldı** (rastgele yüksek kontrast), geçti (tekdüze) | `docs/olcumler/A0.md`, `A0.json` |
| A1 | Çekirdek iskelet (CPU, doğrudan çözücü) | tamam | eski 3-D sürücüyle 7 senaryoda ≤ 3,4e-11, top88 ile 1,35e-11; gradyanlar ≤ 1e-6; 52 test |
| A2 | Hızlı çözücü entegrasyonu | tamam | gerçek tasarımlar 8–20 CG iterasyonu; 1e6 elemanlı optimizasyon ~12 s/iterasyon; tolerans sözleşmesi birebir, varsayılan fail-closed |
| A3 | Özelliklerin 3-D taşınması | büyük kısmı tamam | robust min boyut, max kalınlık, AM, simetri, döküm; gerilme 21 Kasım'da yeniden yazıldı (2-B ve 3-B, toplanmış ve artırılmış Lagrange; eski sürücüyle karşılaştırılmadı) |
| A4 | Yeni fizik ve amaçlar | özfrekans, ısı iletimi, doğrusal burkulma, yer değiştirme amacı (mekanizma) tamam | termoelastik, çok malzeme, doğrusal olmayanlık yok |
| A6 | Yöntem türetme altyapısı (21 Kasım 2026) | tamam | toplama araçları, yerel alan soyutlaması, yanıt cebiri, 0/1 karşılaştırma; `docs/deriving_methods.md`, `tests/toporlogy/test_deriving_methods.py` |
| A5 | Deney düzeni | tamam | YAML/JSON, registry (tüm özellik modülleri), CLI, tarama; `validate` sınır koşulu kontrolünü çalıştırır |
| B | Sınır koşulları + geometri | tamam | bileşke kontrolleri 1e-13; rijit-gövde modu tespiti; rulman yükü gerçek yüzeyden |
| C | Bağlantılılık | tamam | düz uyum TO'su ayrışıyor; dondurulmuş bağ aşındırma testini geçiyor |
| D | Üretim öncesi son işlem | tamam | arayüz yüzeyleri birebir (delik yarıçapı 1634 noktada tam), hacim ±%0,44, min üçgen açısı 22–28°; T10 ağ yakınsaması açık |

## 4. Bilinen açıklar (ajan raporlarından, ölçülü)

* **Bağlantılılık:** ısı iletimi ve bağlama kısıtları eşik 0,3/0,5/0,7'de tek gövde veriyor ama bir elemanlık
  aşındırmada 1–2 hücrelik boyunlardan kopuyor (FRAGILE). Aşındırmaya dayanıklı tek seçenek dondurulmuş en kısa bağ
  (uyum ×1,10–1,16). η-aşındırılmış alan kullanılmazsa iki kısıt da gri bir zarla kandırılıyor. Öneri: bağ + η'lı
  iletim kısıtı + tespit kapısı + `bridge`.
* **Robust minimum boşluk:** 90×30 örnekte iç köşelerde 2,69 (hedef 4); küçük 3-D örnekte β = 32'de OC çöküşü.
  Olası çare: yüksek β'da hareket sınırını daraltan bir takvim (`Schedule`'ın optimizer'a erişimi gerekir).
* **MMA:** genel yakınsama garantisi yok; kısıtsız iç optimumda 2-döngü görüldü. `GCMMA` (21 Kasım) bunu gideriyor
  (kalan salınım kabul toleransı düzeyinde), bedeli iterasyon başına ek çözümler (MBB'de 2 kat, gerilmede 6 kat).
* **Özfrekans:** çoklu özdeğerde yalnız KS yumuşatma; LOBPCG/multigrid özdeğer çözücüsü yok.
* **Sınır koşulları:** delik bulma yalnız eksen hizalı; pim mesnedi ve tork düğüm tabanlı (rulman yükü gibi gerçek
  yüzeyden örneklenmiyor); tasarıma bağlı öz ağırlıkta düşük yoğunluk sorunu yalnız belgelendi.
* **Performans:** CPU'da float64 `conv3d` yavaş → filtre CPU'da seyrek matrisle; yük altında çok iş parçacıklı
  MKL/torch 10–25× yavaşlıyor.
* **Gerilme:** kısıt gri tasarımda ve gevşetilmiş biçimde sağlanıyor; eşiklenmiş tasarımda L-brakette %2,5–71
  aşım ölçüldü (`docs/olcumler/yontemler.md` §3). Sınıra doğru gerilme ve cisme oturan doğrulama açık.
* **Burkulma:** yalnız düz `Elasticity`; mod sayısı sabit, küme kesilirse uyarı yok; katıdaki yerel sahte modlara
  karşı önlem yok.
* **Yeni yöntemlerin kaynakları:** hiçbir makalenin tam metni okunamadı (`docs/literatur_2026-11-21.md` §0).
* **Bağımsız referanslar:** top88 uyumu depodaki Python portuna karşı; DTU MATLAB, Top3D, NAFEMS ve geniş veri kümesi karşılaştırmaları açık (V&V yol haritası).

## 5. Yapılmayacaklar (şimdilik)

Yapısız (tet) ağda optimizasyon (tet yalnız doğrulamada), level-set/BESO, ticari yazılımların kapalı yöntemlerinin
taklidi, dağıtık (MPI) çalışma.

## 6. Riskler

| Risk | Önlem |
|---|---|
| GMG yüksek kontrastta yavaş yakınsar | A2: gerçek geçmişlerde ölçüm, güçlü kaba uzay, doğrudan çözücüye otomatik düşme |
| autograd belleği büyük ızgarada şişer | zincirde elle `backward` (gradcheck'li), ara tensörleri tutmayan çözüm |
| float32 GPU sonuçları float64'ten sapar | karma hassasiyet yalnız ön koşulda; regresyon her iki cihazda |
| Okunmamış literatür | `[L]` öğeler kaynağı okunmadan uygulanmaz ya da docstring'de "okunmadı" yazılır |

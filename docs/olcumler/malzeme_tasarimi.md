# Malzeme tasarımı (tensör karışımı, fiber açısı, malzeme alanı): sonuçlar

Ön kayıt: `malzeme_tasarimi_onkayit.md` (26 Kasım 2026 02:42 +0300, commit `c594c46`, kodun ilk satırından önce).
Kod ve testler: commit `1a0c490`. Ham ölçüm: `malzeme_tasarimi.json` (betik `malzeme_tasarimi.py`, 03:39 +0300,
`1a0c490` üzerinde, temiz ağaç). Figürler: `malzeme_tasarimi_figur.py` → `malzeme_tasarimi_yonlu_modul.png`,
`malzeme_tasarimi_tarama.png` (datawiz: ikisi de 0 HATA). Parmak izi: `malzeme_tasarimi_parmakizi.py`.
Testler: `tests/toporlogy/test_material_design.py` (test adları kontrol kimliklerini taşır).

Kapsam: 02 §1 (malzeme alanı, adlandırılmış bloklar), §3 (çok fazlı tensör karışımı, fiber açısı), §4 (referans
problemler). §2 (konstitutif protokol) nonlinear-solvers oturumunun; 3-B yön parametrizasyonu yok (fail-closed).

## 1. Durum

| # | Kontrol | Sonuç | Ölçülen / not |
|:---|:---|:---|:---|
| K1 | Saf faz geri kazanımı (2-B, 3-B, iki kural) | PASS | ≤ 1e-14 |
| K2 | Pozitif tanımlılık, 2000 örnek | PASS | λ_min ≥ (1 − 1e-12) · alt sınır |
| K3a | Özdeş fazlar, p = 1 | PASS | ≤ 1e-14 |
| K3b | `recursive`, C_i = E_i D = `MultiPhaseSIMP` · D | PASS | ≤ 1e-14 |
| K4 | Permütasyon, `simplex` | PASS | ≤ 1e-14; `recursive` sıra bağımlılığı (ölçüm): **0,41** göreli |
| K5 | Simplex korunumu (filtre + projeksiyon) | PASS | |
| K6a–d | Paralel / seri çubuk, Voigt / Reuss | PASS | ≤ 1e-12, iki kural |
| K7 | Bağımsız montaj (`_independent_fe`), 2-B ve 3-B | PASS | K ≤ 1e-13, U ≤ 1e-10 |
| K8a | frac bir kez | PASS | ≤ 1e-15 |
| K8b | Skaler limit = `MultiPhaseSIMP` + `Elasticity` | PASS | eşikler tuttu; ilk kod denemesinde (duman betiği, kayıt dışı) 1,3e-12 |
| A1 | Düzlem içi döndürme = çekirdek ve bağımsız yol | PASS | ≤ 1e-13 |
| A2 | Periyot: 3 × 3 (triklinik, kübik π/2, ortotrop ≠ π/2), `angle_period` | PASS | |
| A2 (6 × 6) | Triklinik 3-B tensör z etrafında π-periyodik | **FAIL** | **0,206** (strict xfail; §3.1) |
| A3 | dC/dθ autograd = analitik; FD | PASS | |
| A4a | Sarma: ±85° dama tahtası → 90° ± 5° | PASS | en büyük sapma 0,66° |
| A4b | Doğrudan θ süzgeci (karşılaştırma) | ölçüm | en büyük sapma 87,1°, iç elemanlarda ≥ 86,3° |
| A4c | Düzgün alan korunur | PASS | |
| A5a | Eğik çekme E_x(FE) = kapalı form, 0…90° | PASS | ≤ 3,7e-14 |
| A5b | dc/dθ = kapalı formun türevi | PASS | ≤ 1,7e-13; 0° ve 90°'de \|g\|/c ≤ 1,3e-14 |
| A6 | 1-B tarama referansı, global θ, MMA 3 başlangıç | **FAIL** | açı ≤ 1° (geçiyor), c_opt / c_min = 1,00084 / 1,00029 / 1,00029 (strict xfail; §3.2) |
| A7 | Eleman başına θ ≤ en iyi düzgün açı | PASS | c 91,60 → 43,83 (%52 düşüş), en küçük tutarlılık ρ̃ = 0,27 |
| A8 | YAML örneği koşar | PASS | |
| G1 | FD yönlü türev (faz, açı, yoğunluk + açı, 3-B) | PASS | ≤ 1e-6 |
| G2 | Taylor eğimi, h ∈ [1,25e-2; 1e-1] | **FAIL** (2/4) | `recursive` 1,31, yoğunluk + açı 1,55 (strict xfail; §3.3); `simplex` 1,93, açı 2,02 geçiyor |
| G2 post hoc | h ∈ [1,25e-4; 1e-3] | PASS | eğim 1,99–2,00 (dört durum) |
| G3 | Sıfır türevler (saf izotrop faz, keep) | PASS | |
| G4 | Taban doğrusallığı KE(C) (Q4, H8) | PASS | ≤ 1e-14 |
| G5 | Kütle / maliyet türevi | PASS | ≤ 1e-8 |
| E1 | `pytest -m baseline` | PASS | 8 örnek, en büyük göreli fark 0 |
| E2 | Parmak izi (`SIMP`, `RAMP`, `HashinShtrikman`, `MultiPhaseSIMP`, `phase_fractions`, `phase_start`) | PASS | 12/12 bit düzeyinde aynı |
| E3 | Tek izotrop faz tensör yolu = SIMP + `Elasticity` (MBB 30 × 10) | PASS | uyum ≤ 1e-10, gradyan ≤ 1e-8, 10 MMA adımı ≤ 1e-6 |
| E4 | Sabit açı alanı = mevcut ortotrop `angle_field` yolu | PASS | uyum ≤ 1e-10, U ≤ 1e-9 |
| U1 | 12 yeni UNSUPPORTED kuralı fail-closed | PASS | `test_compatibility.py` probları |
| U2 | Katalog belgeleri, `test_catalog.py` | PASS | |
| M1 | `MaterialField` yalnız aktif eleman, istenmeyen alan yok | PASS | |

## 2. Seçilen formülasyon ve alternatifler

* **Fazlar** kendi `ElasticTensor`'larıyla (`core.material_field.PhaseMaterial`). Kesirler mevcut kutu alanları ve
  çubuk kırma ile; filtre / projeksiyon t'ye. İki kural da SPD tensörlerin konveks bileşimi: `simplex` (fazlara
  simetrik, varsayılan) ve `recursive` (mevcut `MultiPhaseSIMP`'in tensör genellemesi, sıra bağımlı: K4 ölçümü
  0,41). İkisi de ceza kuralı, homojenleştirme değil; izotrop HS sınırları anizotrop C'ye uygulanmadı. p > 1'de
  özdeş fazlar yalnız saf durumlarda tek malzemeye eşit (p = 1'de her yerde, K3a).
* **Karşılaştırma** (ölçüm, eşiksiz; 48 × 24 konsol, P1 + P3, faz hacmi 0,15 + 0,20, MMA 80 adım, projeksiyonsuz):
  `simplex` uyum 331,97, gri payı 0,43; `recursive` uyum 285,22, gri payı 0,45; ikisi de uygun (feasible). 80
  adımda projeksiyon olmadan iki kural da gri; bu tek koşudan kural seçimi için genel bir sonuç çıkmaz.
* **Açı süzgeci**: çift açı temsili (cos 2θ, sin 2θ) mevcut `DensityFilter` ile; 2-B'de yönelim tensörünü süzmekle
  özdeş olduğu için ayrı yönelim tensörü yolu eklenmedi. Doğrudan θ süzgeci ±85° alanını 0°'ye çekiyor (A4b: 87°
  sapma); çift açı süzgeci 0,66°. Tutarlılık ρ̃ → 0'da yön tanımsız ve türev ∝ 1/ρ̃ (A7 sonunda en küçük ρ̃ 0,27).
  Süzülen açı tasarım dışı elemanlarda `keep_angle`'a döner (`Passive`'in açı karşılığı).
* **Fizik**: `fem/tensor_elastic.TensorElasticity`, `Elasticity`'nin eleman başına matris kipindeki alt sınıfı;
  Q4 / H8'de KE(C) = Σ_a c_a KE(S_a), dJ/dc = −s λᵀ KE(S_a) u. Mevcut `fem/*.py` değişmedi. Q4IM / H8IM yoğuşturma
  nedeniyle D'de doğrusal değil: reddediliyor.
* **Study adaptörü** (Study değişmedi): `TensorPhases.forward` bilgi amaçlı skaler ızgarayı döndürür ve
  `MaterialField`'ı yayımlar; fizik onu bir kez alır, ızgaranın alanla aynı olduğunu denetler (`StaleFieldError`).
  Açı, `embed` bağlantısının ikinci bloğu (`OrientedEmbed`); yoğunluk zincirin geri kalanından olduğu gibi geçer
  (filtre, projeksiyon, β takvimi, passive çalışır). `Study.evaluate_density` açı bloğuyla desteklenmiyor
  (fail-closed: bayat alan hatası).
* **Adlandırılmış bloklar**: `design.orientation.DesignLayout` (ad, konum, boyut, sınır, birim; `split` / `join`).

## 3. Başarısız ön kayıtlı kontroller

### 3.1 A2 (6 × 6): ön kaydın periyot iddiası 3-B için yanlış

Ön kayıt §0, dördüncü mertebe tensörün R = −I altında değişmediğinden "C(θ + π) = C(θ) her malzemede" diyordu. 3-B'de
z etrafında π dönüşü diag(−1, −1, 1)'dir, −I değil: tek sayıda z indisi taşıyan bileşenlerin (Voigt C14, C15, C24,
C25, C34, C35, C46, C56) işaretini çevirir. Ölçülen göreli fark 0,206 (triklinik, tohum 2). Doğru ifade: 2-B
tasarımın döndürdüğü indirgenmiş 3 × 3 matris yalnız dört düzlem içi indisli bileşenlerden oluştuğu için her
malzemede π-periyodik (A2'nin bu kısmı geçiyor); 3-B tensör yalnız z iki katlı malzeme ekseni ise π-periyodik (post
hoc test: ortotrop eksen 3 = z'de geçiyor, triklinikte tam olarak tek z indisli bileşenler işaret değiştiriyor).
Komut metnindeki "π yalnız ortotropik simetride" ifadesi de tam değil: 2-B'de her malzeme, 3-B'de z etrafında
monoklinik simetri yeter. Modül belgeleri düzeltildi. Etki: 2-B açı tasarımı doğru; 3-B yön tasarımı eklenirse
periyot malzemenin z eksenine göre ayrıca belirlenmeli.

### 3.2 A6: MMA tek değişkende salınıyor

Tarama (1° adım) periyotta tek yerel minimum gösteriyor, 0°'de (problem y-orta düzlemine göre simetrik; c(θ) =
c(−θ)). 1° çözünürlükte bu, düzgün açılı problemin global minimumudur; daha ince ölçekte iddia yok. MMA üç
başlangıçta da açıyı 1° içine getiriyor (0,999°, −0,582°, +0,582°), ama tam minimumda durmuyor: 0°'den başlasa bile
±1,6° arasında sabit adımla (değişim 9e-3) salınıyor, son uyum tarama minimumundan %0,084 / %0,029 büyük. Eşik
(1 + 1e-9) tutmadı. Gradyan doğru (A5b, G1). Neden olarak MMA'nın tek değişkenli davranışı düşünülüyor; doğrulanmadı.
`optimizers/` perf oturumunun dosyası. Çözüm koşulu: asimptot uyarlaması ya da hareket sınırı azaltımıyla yeni ön
kayıt.

### 3.3 G2: Taylor adım aralığı asimptotik bölgenin dışında

Ön kayıtlı h ∈ [1,25e-2; 1e-1] aralığında kalan, `recursive` faz alanlarında (eğim 1,31) ve yoğunluk + açı
bloğunda (1,55) h²'ye düşmüyor. Aynı yönde h ≤ 1e-3'te eğim 1,996 ve 1,993; FD (G1) 1e-6 içinde. Gradyan doğru;
eşik seçimi büyük adımlı (yön vektörü N(0, 1), 216 / 144 değişken, açı bloğunda h = 0,1 ≈ 0,3 rad). Post hoc test
(küçük adımlar) dört durumda geçiyor ve ayrı etiketli.

### 3.4 İlk koşuda düzeltilen test hataları (eşik değişmedi)

`Domain.n_elements` diye bir öznitelik yok (testte `n_active` kullanıldı); `torch.as_tensor(..., requires_grad=)`
desteklenmiyor (duman betiğinde). İkisi de test kodunda, kütüphanede değil. Kütüphanede ilk koşudan sonra davranış
değişikliği yok (yalnız belge düzeltmesi, §3.1).

## 4. Bellek ve zaman

`malzeme_tasarimi.json` → `timing` (120 × 60, değerlendirme + gradyan, medyan 5 koşu): tek torch iş parçacığında
SIMP yolu 0,194 s, tensör yolu 0,211 s (oran 1,09). Varsayılan 6 iş parçacığında 0,376 s / 2,158 s (oran 5,7): bu
makinede torch'un çok iş parçacıklı küçük tensör işlemleri tek iş parçacığından ~30 kat yavaş (7200 × 3 × 3 çarpım:
6 iş parçacığı 26 ms, 1 iş parçacığı 0,8 ms). Bu koşu hızlı paketle eşzamanlıydı, 6 iş parçacıklı sayılar gürültülü.
Malzeme alanı 576 kB, eleman matrisleri 3,7 MB (n_active × 8 × 8 × 8 B). Eski yol değişmedi (E2 bit düzeyinde), ek
maliyet yok.

## 5. Testler ve araçlar

`1a0c490` üzerinde (bu makine, diğer koşularla eşzamanlı):

* `pytest tests/toporlogy -m "not slow" -p no:randomly`: 1216 passed, 4 skipped, 69 deselected, 25 xfailed, 1 failed
  (3198 s). Tek hata `test_ledger_entries_are_complete`: koşu bu dosya yazılmadan önce başladı, kütük girdisinin
  kayıt yolu henüz yoktu; dosya eklendikten sonra `test_catalog_ledger.py` 8 passed. Taban `10daa03`: 1149 passed,
  22 xfailed (koordinasyon notundan; bu oturumda tabanı yeniden koşmadım).
* Yavaş: A6 strict xfail (beklenen), A7 passed.
* `pytest -m baseline`: 8 passed, her örnekte en büyük göreli fark 0,00e+00 (E1 PASS).
* `ruff check src tests`: temiz. `python -m mypy`: 116 dosyada sorun yok.
* Yeni test dosyası 60 test (4 strict xfail, 2 yavaş: A6, A7).

## 6. Uyumluluk matrisi (katalog satırları)

SUPPORTED: `tensor_phases` × `tensor_elasticity`; `embed.angle` × `tensor_elasticity` (2-B, tek faz).
UNSUPPORTED (fail-closed): `tensor_elasticity` ile Compliance / Volume / PhaseVolume / Mass / Cost dışındaki
yanıtlar, `tensor_phases` dışı interpolasyon, `phase_report` dışı son işlem; `tensor_phases` ile başka fizik;
Q4IM / H8IM; durum başına mesnet; eğik mesnet / yay; açı bloğu: 3-B, birden çok faz, `tensor_phases` dışı
interpolasyon, izotrop faz, OC.

## 7. Kaynaklar

Bu dilimde dış kaynak okunmadı; formüller (konveks karışım, çift açı süzgeci, eğik çekme modülü, π-periyot) burada
türetildi ve testlerle sınandı. DMO (Stegmann & Lund) ve sürekli fiber açısı optimizasyonu literatürü okunmadı;
benzerlik iddiası yok.

## 8. Ertelenen işler / geçiş koşulları

* Birden çok faz + açı: `Study._check_phases` yalnız `Embed` + liste bağlantılarını kabul ediyor (`optimization/
  study.py`, perf oturumunun dosyası). Gereken değişiklik: `OrientedEmbed` ve açı bloğu için kabul.
* 3-B yön parametrizasyonu (dönme vektörü ya da iki birim vektör, periyot §3.1'e göre).
* Gerilme / burkulma / modal yanıtları tensör alanıyla (eleman başına D zaten `TensorElasticity.D`'de).
* MMA tek değişkenli salınımı (A6) ve torch iş parçacığı yükü (§4): perf.
* Konstitutif protokol (§2) ile birleşme: `MaterialField` small-strain C taşıyor; tanjant / durum değişkeni yok.

## Birleştirmeden sonra: A6 ve GCMMA (koordinatör, 2026-11-26, post hoc)

A6'daki ±1,6° salınım, `optimizers/mma.py`'de belgelenmiş düz MMA sınırlamasıdır: iç optimumlu ve etkin kısıtın
dokunmadığı bir değişken, asimptotlar x'ten asymin = 0,01 uzakta tutulunca 2-döngüde kalabilir (13 Kasım ölçümü,
`test_mma.py`). Aynı problem GCMMA ile (move 0,1, 30 iterasyon): başlangıç 0°, ±45° için açı 9,5e-5°, 3,3e-4°,
3,0e-4°; c / c_min − 1 = 7,4e-12, 9,0e-11, 7,8e-11 (A6'nın iki ölçütü de tutuyor). Post hoc test
`test_a6_posthoc_gcmma_reaches_the_sweep_minimum` (yavaş). A6 strict xfail olarak kalır (MMA ile ön kayıtlıydı).

# Gerilme recovery ve tekillik tanısı: ön kayıt

Yazıldı: 26 Kasım 2026 00:24 +0300 (`date`), ilk koşudan önce. Taban: `main` = `1d1f027`. Bu dosya koşudan sonra
değiştirilmez; sonuçlar ve sapmalar `gerilme.md`'ye yazılır. Eşik gevşetilmez: tutmayan kontrol strict xfail olur
ve ölçülen değer kaydedilir; sonradan eklenen her ölçüm "post-hoc" diye etiketlenir.

Kapsam: `06_STRESS_GEOMETRY.md`'nin erken dilimi (recovery çıktı türleri, tekillik tanısı, L-bracket gerilme
deneyinin yeniden üretimi). Burkulma, geometri aktarımı, T10 / gövdeye uyumlu ağ bu turda yok (gmsh kurulu değil).

Kod: `src/toporlogy/optimization/stress_recovery.py`, `stress_singularity.py`. Testler:
`tests/toporlogy/test_stress_recovery.py`, `test_stress_singularity.py`. L-bracket: `docs/olcumler/gerilme.py` →
`gerilme.json`.

## 1. Tanımlar

Çıktı türleri (ayrı nesneler; hiçbiri "kesin gerilme" değildir):

* `integration_point`: 2 × 2 (× 2) Gauss noktalarında σ = E D (B u + B_α α); uyumsuz modlu elemanda modlar dahil.
* `centroid`: eleman merkezinde.
* `integration_point_max`: eleman başına, Gauss noktalarındaki skaler ölçütün (ör. von Mises) en büyüğü.
* `extrapolated`: Gauss değerlerinden iki/üç doğrusal dışdeğerleme, eleman köşelerinde (elemanlar arası süreksiz).
* `nodal_averaged`: köşe değerlerinin düğümde aritmetik ortalaması; farklı malzemeler (D, E) ayrı tutulur.
* `patch_recovered`: iç düğüm yamasındaki 2^dim eleman merkezine iki/üç doğrusal uydurma (Zienkiewicz-Zhu SPR
  fikri; kaynak metin bu turda okunmadı), sınır düğümleri iç yamalardan; yamalar malzeme sınırını aşmaz.

## 2. Analitik ve bağımsız kontroller

| # | Kontrol | Eşik |
|:---|:---|:---|
| T1 | Tek eksenli çekme, Q4, Q4IM, H8, H8IM; beş çıktı türünün hepsi | max \|σ − tam\| ≤ 1e-9 σ₀ |
| B0 | Saf eğilme (ν = 0,3), Q4IM ve H8IM (kare/küp eleman): beş çıktı türü | ≤ 1e-8 σ_max |
| I1 | Paralel iki malzeme (ν = 0, E₂ = 4 E₁, aynı εx): `nodal_averaged` ve `patch_recovered` ayrık | ≤ 1e-9 σ_max |
| I2 | I1'de ayrım kapatılınca arayüz düğümündeki hata | \|E₂ − E₁\| / (2 E₂) = 0,375 ± 1e-9 |
| O1 | Gauss ve merkez gerilmesi `_independent_fe`'ye karşı; Q4, H8; izotrop ve eleman başına dönük ortotrop; rastgele U | ≤ 1e-12 göreli |
| O2 | Σ Gauss enerji yoğunluğu × ağırlık = ½ uᵀKu; Q4, Q4IM, H8, H8IM; eleman başına rastgele E | ≤ 1e-12 göreli |
| F1 | `failure` ölçütü olarak von Mises kriteri = von Mises / izin verilen | ≤ 1e-12 |
| F2 | Kapalı başarısız: 2-B'de düzlem gerilme olduğu bilinmeyen D ile von Mises; NaN yer değiştirme | hata fırlatır |

## 3. Yakınsama mertebesi (saf eğilme, ν = 0,3)

Q4: H = 1, L = 2, n_y = 8, 16, 32, 64 (kare eleman). H8: H = 1, L = 2, W = 0,5, n_y = 4, 8, 16. Sol uç simetri
(u_x = 0), sağ uçta doğrusal çekme dağılımının tutarlı düğüm yükleri.

Hata e = bölgedeki en büyük \|σ_h − σ_tam\| (Voigt bileşeni) / σ_max. Bölge **R**: x ≤ L/2 (yüklü uçtan en az H
uzakta; yüklü uçta tutarlı yük ile eleman sütununun iç kuvvetleri köşede uyuşmaz, `test_timoshenko_cantilever`'da
görülen köşe etkisiyle aynı tür). Bölge **R'** (yalnız 2-B): H/4 ≤ x ≤ L/2 ve H/4 ≤ y ≤ 3H/4. Mertebe
p = log₂(e_kaba / e_ince), en ince iki ağdan.

Gerekçe (teorik beklenti, koşudan önce): bilineer elemanda merkez σ O(h²); Gauss noktasında εy parçalı sabit ve
parazit kayma γ ∝ (x − x_merkez) O(h); köşeye dışdeğerleme O(h); düğüm ortalaması içeride kayma ve εy hatasını
götürür (O(h²)) ama serbest yüzeyde götürmez (O(h)); yama uydurması merkez değerlerini kullandığı için sınırda da
O(h²).

| # | Çıktı türü, bölge | Q4 eşiği | H8 eşiği |
|:---|:---|:---|:---|
| C1 | `centroid`, R | p ≥ 1,8 | p ≥ 1,6 |
| C2 | `integration_point`, R | 0,8 ≤ p ≤ 1,2 | 0,8 ≤ p ≤ 1,25 |
| C3 | `extrapolated`, R | 0,8 ≤ p ≤ 1,2 | 0,8 ≤ p ≤ 1,25 |
| C4 | `nodal_averaged`, R | 0,8 ≤ p ≤ 1,2 | 0,8 ≤ p ≤ 1,25 |
| C5 | `nodal_averaged`, R' | p ≥ 1,8 | — |
| C6 | `patch_recovered`, R | p ≥ 1,8 | p ≥ 1,6 |

## 4. Tekillik tanısı

Karar kuralı (h kesin azalan, tepe değerleri p_i; son üç ağ; Δ₁ = p_{n−1} − p_{n−2}, Δ₂ = p_n − p_{n−1};
gürültü tabanı g = 1e-8 \|p_n\|):

1. üçten az düzey → `NOT VERIFIED`;
2. Δ₁ > g, Δ₂ > g ve Δ₂ ≥ 0,8 Δ₁ → `singular — max stress not converged` (artış yavaşlamıyor);
3. \|Δ₂\| ≤ 0,02 \|p_n\| ve (\|Δ₂\| < 0,8 \|Δ₁\| ya da \|Δ₂\| ≤ g) → `converged`;
4. diğer → `not converged`.

Kural sonlu bir ağ dizisinden tanı koyar, kanıt değildir: h^q ile yakınsayan bir nicelikte q < 0,32 ise yanlışlıkla
"singular" der.

Problemler (2-B, Q4 düzlem gerilme, E = 1, ν = 0,3, t = 1, ölçüt merkez von Mises; bölge tekil noktanın çevresinde
yarı genişliği 0,25 olan kutu, parçayla kesişimi; n = 16, 32, 64, 128):

* **P** noktasal yük: birim kare, alt kenar ankastre, P = 1 üst orta düğümde aşağı.
* **L** içbükey köşe: birim kareden sağ üst çeyrek çıkarılmış; düşey kolun üst kenarı ankastre; yatay kolun sağ
  ucunda düzgün yayılı kayma, toplam 1.
* **C** ankastre kenar: 2 × 1 konsol (n_x = 2 n_y), sol kenar ankastre, sağ uçta düzgün yayılı kayma, toplam 1;
  bölge (0, 1) köşesi çevresinde.
* **D** yayılı yük (negatif kontrol): P problemi, aynı toplam yük 0,25 genişliğinde ayak izine yayılı.
* **S** saf eğilme (negatif kontrol): bölüm 3'ün Q4 problemi, bölge R, n_y = 8 … 64.

| # | Kontrol | Eşik |
|:---|:---|:---|
| S1 | P, L, C: bölge tepe durumu | `singular — max stress not converged` |
| S2 | P, L, C: en ince ağda tepe konumunun tekil noktaya uzaklığı | ≤ 2h |
| S3 | P: son çiftte yerel eğim −d ln p / d ln h (beklenen 1: σ ∝ 1/r) | 0,8 – 1,2 |
| S4 | L: aynı eğim (Williams mod I, 270°: 1 − 0,5445 = 0,4555; mod II ve düzenli terimler payı için pay) | 0,30 – 0,55 |
| S5 | P, L, C: bölgenin alan ortalaması von Mises, son çiftte göreli değişim | ≤ 3 % |
| S6 | D: bölge tepe durumu | `converged` |
| S7 | S: n_y = 8…64 `converged`; n_y = 8…32 `not converged` (büyüyen ama yavaşlayan tepe tekil sayılmaz) | aynen |

C'nin eğimi ve basamaklı (voksel) yuvarlatmalı L köşesi yalnız raporlanır, eşik yoktur. Gövdeye uyumlu pürüzsüz
yuvarlatma bu ızgarada kurulamaz: NOT VERIFIED.

## 5. L-bracket gerilme deneyinin yeniden üretimi

`examples/research/stress_lbracket2d.yaml` (toplulaştırılmış) ve `stress_al_lbracket2d.yaml` (augmented
Lagrangian) **değiştirilmeden** koşulur. Tarihsel değerler (`yontemler.md` §3; Linux, NumPy 2.5.3, torch 2.15):
gri tasarımda en büyük ham katı oran 1,131 / 1,103; 0/1 tasarımda 1,715 / 1,025. Bu ortam farklıdır (Windows,
NumPy 2.2.6, torch 2.5.1) ve optimizasyon konveks değildir.

Ham oran = merkez von Mises(E0 D B u/h) / σ_lim, yalnız dışlanmamış ve x̄ > 0,5 elemanlarda (tarihsel tanım).
Relaxed oran = x̄^0,5 × ham, bütün dışlanmamış elemanlarda. İkisi ayrı yazılır ve **döndürülen** tasarımdan
(`Result.xphys`) hesaplanır; 0/1 tasarım x̄ > 0,5 eşiği, aynı ağda yeniden analiz.

| # | Kontrol | Eşik |
|:---|:---|:---|
| K1 | Tarihsel dört oranın yeniden üretimi | her biri \|yeni − tarihsel\| ≤ 0,05 |
| K2 | Döndürülen 0/1 tasarımda merkez gerilmeleri `_independent_fe.centroid_stresses` ile | ≤ 1e-12 göreli |
| K3 | Dolu L-bracket'in en büyük merkez oranı | 0,49 ± 0,005 |
| K4 | Sabit geometri (dolu; iki 0/1 tasarım) her eleman k × k bölünerek (k = 1, 2, 4, 8; aynı noktasal yükler, aynı mesnet, aynı fiziksel dışlama kutusu): dışlama dışındaki katı tepe von Mises durumu | üçü de `singular — max stress not converged` |

Rapor (eşiksiz): her çıktı türüyle tepe oranı, tepe konumu, dışlanan elemanlardaki değer, hacim, M_nd, iterasyon,
durum çözümü sayısı, optimizerin gördüğü kısıt değeri.

Tek problemden yöntemler arası genelleme yapılmaz. Kısıtı gri tasarımda sağlayan bir sonuç, ham ya da 0/1 oranı
1'i aşıyorsa "gerilme açısından uygun" sayılmaz.

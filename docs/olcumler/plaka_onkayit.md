# Mindlin–Reissner plaka elemanı (düzenli ızgara, Q4) — ön kayıt

Ön kayıt 2026-11-26 13:49 +0300, kod yazılmadan önce. Yeni modül `fem/plate.py`: 2-B ızgarada eğilme plakası, düğüm başına
(w, βx, βy); γ = ∇w − β, κ = [∂βx/∂x, ∂βy/∂y, ∂βx/∂y + ∂βy/∂x]. Eğilme D_b = E t³ / (12 (1 − ν²)) [[1, ν, 0], [ν, 1, 0],
[0, 0, (1 − ν)/2]] ile 2 × 2 Gauss; kesme k_s G t I (k_s = 5/6) ile **tek nokta** (seçici indirgenmiş integrasyon,
kesme kilitlenmesine karşı). Eleman matrisi birim modüllü; eleman modülü E_e ile ölçeklenir (`GridSPD`, autograd).
Düzgün basınç q: eleman başına her düğüme q h² / 4 (bilineer N için ∫N dA). Bu turda Study / YAML bağlantısı yok.

**Karşılaştırma çözümü (burada türetildi, bu oturumda kaynaktan okunmadı).** Kenarları sert basit mesnetli (w = 0 ve
teğet dönme 0) a × a kare plaka, düzgün q. Kirchhoff çözümü Navier serisi: q'nun çift sinüs açılımı tek m, n için
q_mn = 16 q / (π² m n); D∇⁴w_K = q ⇒ w_K,mn = q_mn / (D λ²), λ = π² (m² + n²) / a². Mindlin denklemlerinde
(Q = k_s G t (∇w − β), ∇·Q + q = 0, ∇·M = Q) sert basit mesnetli plaka için w = w_K − (D / (k_s G t)) ∇² w_K bağıntısı
sağlanır (Navier alanları w ~ sin sin, βx ~ cos sin, βy ~ sin cos sınır koşullarını tam karşılar); sinüs modunda
−∇² = λ olduğundan w_mn = q_mn [1 / (D λ²) + 1 / (k_s G t λ)]. Seri m, n ≤ 399 (tek) ile kesilir; kesme hatası
testte bir sonraki terim bloğuyla ölçülüp kaydedilir.

| # | Kontrol | Eşik |
|:---|:---|:---|
| PL1 | Rijit modlar (w = 1; w = x, βx = 1; w = y, βy = 1) enerjisiz: ‖KE r‖ | ≤ 1e-12 ‖KE‖ |
| PL2 | Sabit eğrilik yaması: düğümlerde w = x²/2, βx = x, βy = 0 ⇒ eleman enerjisi = D11 h² / 2 (tek nokta kesmede γ = 0) | ≤ 1e-12 göreli |
| PL3 | Navier (Mindlin), a/t = 10 ve 100, n = 8, 16, 32 eleman/kenar: merkez çökmesi hatası | n = 32'de ≤ %1; 16 → 32 gözlenen mertebe ≥ 1,5 |
| PL4 | Negatif kontrol: kesme 2 × 2 tam integrasyon, a/t = 1000, n = 16 | merkez çökmesi ≤ 0,5 × Navier (kilitlenme görünür) |
| PL5 | İnce sınır: a/t = 1000, n = 32, seçici integrasyon, Kirchhoff Navier'e göre | ≤ %2 |
| PL6 | Tasarım gradyanı: c = Fᵀw, E_e = 1e-6 + x_e³ (rastgele x), merkezi fark | ≤ 1e-6 göreli |
| PL7 | 3-B alan, t ≤ 0, ν ∉ (−1, 0,5) | açık hata |

## Sonuç (ilk ölçüm, 2026-11-26)

`tests/toporlogy/test_plate.py`; yedi kontrol ilk koşuda geçti.

| # | Durum | Ölçülen |
|:---|:---|:---|
| PL1 | PASS | üç rijit mod ≤ 1e-12 ‖KE‖ |
| PL2 | PASS | eleman enerjisi = D11 h² / 2, ≤ 1e-12 |
| PL3 | PASS | a/t = 10: 3,6e-3 / 8,9e-4 / 2,2e-4 (n = 8 / 16 / 32), mertebe 2,00; a/t = 100: 5,1e-3 / 1,3e-3 / 3,1e-4, mertebe 2,01 |
| PL4 | PASS | tam kesme integrasyonu: merkez çökmesi 0,0018 × Navier (kilitlenme) |
| PL5 | PASS | a/t = 1000, n = 32: Kirchhoff Navier'e göre 3,1e-4 |
| PL6 | PASS (sınıra yakın) | 8,8e-7 göreli (eşik 1e-6; adım 1e-6 merkezi fark) |
| PL7 | PASS | 3-B alan, t = 0, ν = 0,5 için açık hata |

Navier serisinin kesilmesi (eşiksiz kayıt ölçümü): 399 → 799 terimde değişim 1,2e-9 (a/t = 10), 1,3e-11 (100),
4,3e-13 (1000); karşılaştırma hatalarının çok altında.

## Ek ön kayıt: plaka `Study` fiziği olarak (PQ1–PQ4)

Ön kayıt 2026-11-26 13:58 +0300, kod yazılmadan önce. `fem.plate.PlatePhysics(plate)`: `solve(E, F)` → U (3 nn, n_cases);
`state_dofs` = 3 nn. `Study` yük satırlarını `physics.state_dofs` varsa ona, yoksa `domain.ndof`'a göre denetler
(bugünkü fiziklerin hiçbiri `state_dofs` tanımlamıyor). YAML bağlantısı (plaka DOF'larında yük ve mesnet) bu turda yok.

| # | Kontrol | Eşik |
|:---|:---|:---|
| PQ1 | Study uyumu = doğrudan `MindlinPlate.solve` ile Fᵀw (SIMP + yoğunluk filtresi) | ≤ 1e-12 göreli |
| PQ2 | Study gradyanı merkezi farkla | ≤ 1e-6 göreli |
| PQ3 | 24 × 24 sert basit mesnetli plaka, düzgün basınç, hacim ≤ 0,4, 30 MMA iterasyonu | son uyum < ilk uyum; hacim ≤ 0,4 (1 + 1e-3) |
| PQ4 | elastik çalışmalar etkilenmez; yanlış satırlı yük | `pytest -m baseline` bit düzeyinde; açık hata `state_dofs` sayısını söyler |

### PQ1–PQ4 sonucu (ilk ölçüm)

Hepsi ilk koşuda geçti (`test_plate.py::test_pq1_pq2_…`, `::test_pq3_…`, `::test_pq4_…`). PQ1 ≤ 1e-12 (eşik
geçildi; değer ayrıca yazdırılmadı); PQ2 2,6e-7 göreli; PQ3 24 × 24 plaka, uyum 3,62e4 → 1,58e4 (30 iterasyon), hacim
0,3997; PQ4 baseline 8 passed bit düzeyinde, yanlış satırlı yük "48 state DOFs" iletisiyle reddedildi.

## Ek ön kayıt: plakanın YAML bağlantısı (PY1–PY4)

Ön kayıt 2026-11-26 15:19 +0300, kod yazılmadan önce. `physics: {type: plate, thickness, edge: simply_supported | clamped,
pressure: q, force: P, at: <seçici>}`. Geçici ısı gibi `load_cases` almaz (`NO_LOAD_PHYSICS`); `supports` plakanın
mesnet düğümlerini seçer (her bileşeni sabit `fixed` mesnet). `edge: simply_supported` bu düğümlerde w = 0 (yumuşak
basit mesnet), `clamped` w = βx = βy = 0. Yük fizikten gelir: düzgün basınç q ve/veya `at` düğümlerine bölünen toplam
düşey kuvvet P; `build` fizik kurulduktan sonra Study'nin yükünü `physics.state_loads` ile değiştirir.

| # | Kontrol | Eşik |
|:---|:---|:---|
| PY1 | YAML plaka çalışması = aynı mesnet DOF'ları ve yükle Python'da kurulan `PlatePhysics` çalışması: değer ve gradyan | ≤ 1e-12 göreli |
| PY2 | YAML: 16 × 16, kenarlar `clamped`, merkezde P, hacim ≤ 0,4, 15 MMA iterasyonu | son uyum < ilk uyum; hacim ≤ 0,4 (1 + 1e-3) |
| PY3 | `load_cases` verilmesi, kısmi bileşenli mesnet, ne q ne P, t ≤ 0, 3-B alan, `at`'sız P | açık hata |
| PY4 | var olan yapılandırmalar | `pytest -m baseline` bit düzeyinde; çözülmüş yapılandırma anlık görüntüsü değişmez |

### PY1–PY4 sonucu

İlk koşuda iki test kurulum hatasıyla düştü (ölçüm değil): PY1'in Python karşılaştırma çalışması `emin_ratio` 1e-9 ile
kurulmuştu, YAML'ınki 1e-6 (iki çalışma aynı problem değildi; ölçülen fark 7,0e-6); PY3'ün "kısmi mesnet" durumu
geçersiz roller tanımıyla (`axis` yerine `normal`) yanlış nedenle düşüyordu. İkisi düzeltildi, eşikler aynı.

| # | Durum | Ölçülen |
|:---|:---|:---|
| PY1 | PASS | değer ve gradyan farkı 0 |
| PY2 | PASS | 16 × 16 ankastre plaka, merkezde P: uyum 1,21e5 → 1,45e4 (15 iterasyon), hacim 0,3998 |
| PY3 | PASS | `load_cases`, kısmi mesnet, yüksüz, t = 0, 3-B, `at`'sız P: beklenen iletiyle açık hata |
| PY4 | PASS | baseline 8 passed bit düzeyinde; çözülmüş yapılandırma anlık görüntüsü değişmedi |

Katalog: `plate` fiziği kaydı, `compliance × plate` SUPPORTED; uyumluluk sondası plaka için kendi yüküyle kuruluyor
(`test_compatibility.py::physics_for`).

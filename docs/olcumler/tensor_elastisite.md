# Tensör çekirdeği, lineer elastik aile ve Poisson analizi — doğrulama kaydı

Görev: araştırma komutu `01_TENSOR_ELASTICITY.md` (00_MASTER kurallarıyla). Dal `claude/tensor-core-f4b40b`, taban
`main` = `1d1f027`. Etiketler `malzeme_modelleri.md` ile aynı: **[Ö]** ölçüldü, **[O]** koddan okundu, **[H]**
hafızadan, doğrulanmadı.

## 0. Ön kayıt (2026-11-26 00:29 +0300, `date` çıktısı; yeni kodun ilk satırından ve ilk test koşusundan önce)

Tutmayan ön kayıtlı bir kontrol gevşetilmez: test `xfail(strict=True)` olarak kalır, ölçülen değer ve neden
docstring'e ve bu kayda yazılır. Ön kayıttan sonra eklenen her kontrol "post hoc" diye etiketlenir.

### 0.1 Tasarım kararları (toleransları etkilediği için burada)

- Tek çekirdek: `core/tensor.py` (2./4. mertebe cebir, Voigt/Mandel, rotasyon, invariantlar, kinematik, gerilme
  ölçüleri), `core/linear_elastic.py` (izotropik → genel anizotropik aile, SPD / strong ellipticity, düzlem
  indirgemeleri), `core/anisotropy.py` (yönlü modüller, Poisson ekstremumları, tarama, duyarlılık). Mevcut
  kopyalar (`core/material.py::_tensor`, `rotate_stiffness`, `fem/elements.py::D_*`, `optimization/strength.py`
  içindeki Voigt→tensör ve invariantlar, `fem/nonlinear.py::_green/_voigt/_det_minus_one`,
  `design/interp.py::_bulk_shear`, `post/verify.py::von_mises`) bu çekirdeğe bağlanır.
- Eski yolun aritmetiği korunur: `D_iso`, `D_plane_stress`, `D_plane_strain`, von Mises ve Tresca aynı ifade
  sırasıyla yazılır. Bu yüzden R1'de bu girdiler **bit düzeyinde** aynı olmalı.
- Çekirdek işlevleri hem NumPy hem Torch dizisi alır (aynı kod, ad alanı girdiden seçilir); ikinci bir backend
  yazılmaz. Torch modülü yalnız Torch girdisi geldiğinde içe aktarılır.
- Voigt sırası `[xx, yy, zz, yz, xz, xy]`; gerilme fiziksel kayma bileşeniyle, şekil değiştirme mühendislik kaymasıyla
  (γ = 2ε). Rijitlik Voigt matrisi C_ab = C_ijkl (çarpansız); esneklik S_ab = w_a w_b S_ijkl (w = 1, 1, 1, 2, 2, 2).
  Mandel: m = (1, 1, 1, √2, √2, √2), M_C = diag(m) C diag(m), M_S = diag(1/m) S diag(1/m) = M_C⁻¹. Özdeğer ve koşul
  sayısı yalnız Mandel gösteriminde fiziksel anlam taşır.
- Simetri politikası: giriş matrisi `max|A − Aᵀ| ≤ 1e-12 · max|A|` değilse **hata**; sessiz simetrize yok. Bu eşiğin
  altında kalan yuvarlama asimetrisi (ör. ters alma sonrası) belgelenmiş biçimde ortalanır.
- SPD politikası: simetri kontrolü + Mandel matrisinde Cholesky; ek olarak ölçekli en küçük özdeğer
  λ_min/λ_max raporlanır ve `≤ 64 · eps` ise "sayısal olarak tekil" diye reddedilir.
- Global ekstremum garantisi yoksa sonuç `sampled` / `refined` etiketi taşır.

### 0.2 Tolerans gerekçesi

Float64, birim mertebesinde iyi koşullu girdiler (rastgele örneklerde Mandel koşul sayısı κ ≤ 1e4 ile üretilir):
cebirsel özdeşlikler için göreli 1e-13 (≈ 450 eps; birkaç einsum toplamı), ters alma içerenlerde κ ile ölçekli
`1e-12 · κ` alt sınır 1e-12. Salt indeks kopyası / 2 ile çarpma-bölme gibi kesin işlemler: **tam eşitlik**.
Türevler: merkezi fark, adım h = 1e-6 · ölçek, hedef göreli 1e-6 (kesme hatası ~h², yuvarlama ~eps/h ≈ 1e-10);
sınıra yakın koşullarda `1e-6 · κ`. Taylor testi: birinci mertebe düzeltilmiş kalanın eğimi ≥ 1.9 (beklenen 2).

### 0.3 Kontroller

| # | Karşılaştırma | Referans / bağımsız taraf | Tolerans |
|:--|:---|:---|---:|
| T1 | Voigt gerilme ↔ tensör, Voigt şekil değiştirme ↔ tensör gidiş-dönüş, 2-B ve 3-B, 200 rastgele simetrik | kendisi | tam eşitlik |
| T1b | Mandel ↔ tensör gidiş-dönüş | kendisi | ≤ 1e-15 · max\|x\| |
| T1c | Simetrik olmayan tensör (P = F S) `stress_to_voigt`'a verilince hata; 1e-14 göreli asimetri kabul | politika | hata / kabul |
| T2 | Enerji eşdeğerliği σ:ε = σ_vᵀ ε_v = σ_Mᵀ ε_M, 200 durum | tensör çift daraltma | ≤ 1e-13 göreli |
| T3 | 120 rastgele SPD + 6 deterministik C (izotropik, kübik, TI, ortotropik, monoklinik, triklinik): Voigt C → tensör → Voigt | kendisi | tam eşitlik |
| T3b | Aynı örneklerde S_voigt = C_voigt⁻¹ iken tensör S:C = 𝕀ˢ; M_S = M_C⁻¹ | 𝕀ˢ, birim matris | ≤ 1e-12 · κ |
| T4 | Voigt'ten kurulan tensörün küçük ve büyük simetrisi | tanım | tam eşitlik |
| T4b | 1e-6 göreli asimetrik C reddedilir; 1e-14 göreli kabul edilir (negatif kontrol) | politika | hata / kabul |
| T5 | 100 rastgele R (QR): ortonormallik, det = +1; yansıma (det −1) ve 1e-8 bozulmuş R `check_rotation`'da hata | tanım | ≤ 1e-14 / hata |
| T5b | rotate4 ile R sonra Rᵀ gidiş-dönüş; döndürülmüş enerji ε':C':ε' = ε:C:ε; Mandel özdeğerleri değişmez | kendisi | ≤ 1e-12 göreli |
| T5c | İzotropik C her R altında değişmez | kendisi | ≤ 1e-13 · \|C\| |
| T5d | Dönme vektörü ↔ R (exp/log) gidiş-dönüş, açı ∈ (0, π − 1e-3) | kendisi | ≤ 1e-12 |
| T5e | 2-B: ortotropik C(θ) = C(θ + π); C(θ) ≠ C(θ + π/2) (kare simetrisi yokken, negatif kontrol) | periyodiklik | ≤ 1e-13 / > 1e-3 |
| T6 | 𝕀ˢ:A = sym A; P_vol, P_dev idempotent, P_vol:P_dev = 0; tr(dev A) = 0; izotropik C = 3K P_vol + 2G P_dev | tanım | ≤ 1e-13 göreli |
| T7 | I1, I2, I3 = özdeğerlerin simetrik fonksiyonları; J2 = ½ s:s, J3 = det s; Voigt von Mises² = 3 J2 | özdeğerler | ≤ 1e-12 göreli (ölçek \|A\|^k) |
| T7b | Lode açısı (cos 3θ = (3√3/2) J3 / J2^{3/2}, θ ∈ [0, π/3]): tek eksenli çekme 0, basma π/3, saf kayma π/6; hidrostatik (J2 = 0) NaN | kapalı form | 1e-6 rad / 1e-12 / NaN |
| T8 | Ayrık özdeğerlerde dλ_i/dA = n_i ⊗ n_i (boşluk ≥ 0.1 \|A\|), 50 örnek | merkezi fark | ≤ 1e-6 göreli |
| T8b | Tekrarlı özdeğerde türev NaN + `repeated` bayrağı | politika | davranış |
| T9 | Düzeltilmiş von Mises √(q + e²) − √q ∈ [0, e], 1000 durum | kesin eşitsizlik | e (1 + 1e-12) |
| K1 | 100 rastgele F (J > 0, \|H\| ~ 0.3): C = FᵀF, B = FFᵀ; polar R özgün ortogonal, U SPD, F = RU = VR | tanım | ≤ 1e-13 · \|F\| |
| K1b | Green–Lagrange(H) = ½(C − I); Euler–Almansi = F⁻ᵀ E F⁻¹; Hencky: exp(2 ln U) = C, ln V = R ln U Rᵀ | tanım | ≤ 1e-12 göreli |
| K1c | \|H\| = 1e-6: \|E − sym H\| ≤ ½\|H\|² (kesin sınır, Frobenius) | eşitsizlik | sınır |
| K1d | \|H\| = 1e-10: H biçimli Green–Lagrange ve J − 1, `fractions.Fraction` ile kesin değere karşı (10 örnek); F biçimli ½(FᵀF − I) hatası yalnız raporlanır | kesin aritmetik | ≤ 1e-14 göreli |
| K2 | det F < 0 → `InvertedDeformationError`; det F = 0 → `SingularDeformationError`; κ(F) > 1e8 → `near_singular` sınıfı | politika | davranış |
| S1 | SVK ψ(E): P = F S, ∂ψ/∂F'ye karşı | merkezi fark | ≤ 1e-6 göreli |
| S1b | İş eşleniği P:Ḟ = S:Ė = τ:d = J σ:d (d = sym(Ḟ F⁻¹)), 100 rastgele durum | kendisi | ≤ 1e-12 göreli |
| S1c | PK2 → Cauchy → PK2 ve PK2 → PK1 → PK2 gidiş-dönüş; σ simetrik | kendisi | ≤ 1e-12 göreli |
| L1 | İzotropik (E, ν), (E, G), (K, G), (λ, μ), (E, K), (M, G) çiftlerinden bütün modüllere ve geri; ν ∈ {−0.99, −0.5, −0.3, 0, 0.3, 0.49} | kapalı formlar | ≤ 1e-13 göreli |
| L1b | ν = 0.5, ν = −1, ν = 0.6, ν = −1.2, E ≤ 0, G ≤ 0 reddedilir; ν = −0.3 (λ < 0) kabul edilir | politika | hata / kabul |
| L1c | `isotropic_stiffness(E, ν)` = 2μ 𝕀ˢ + λ I⊗I (Lamé tensör biçimi) | bağımsız ifade | ≤ 1e-14 göreli |
| L2 | Ortotropik: mühendislik sabitleri → S → sabitler gidiş-dönüş; ν21 = ν12 E2/E1 (reciprocity) | kendisi | ≤ 1e-13 göreli |
| L2b | E1/E2 = 10, ν12 = 2 SPD kabul; ν12² > E1/E2 (iki eksenli Schur koşulu ihlali) reddedilir; ν_ij < −1 olan SPD örnek kabul edilir | SPD analitik | kabul / hata |
| L3 | Transvers izotropik (5 sabit): eksen etrafında rastgele açılarla C değişmez; C66 = (C11 − C12)/2; dik eksen etrafında döndürme değiştirir (negatif kontrol) | tanım | ≤ 1e-13 · \|C\| / > 1e-3 |
| L4 | Genel anizotropik (21): C ya da S'den kurulum aynı kanonik C; belirsiz (bir negatif Mandel özdeğeri) ve simetrik olmayan C reddedilir, simetrize edilmez | politika | ≤ 1e-12 · κ / hata |
| L5 | Simetri sınıfı (verilen çerçevede): izotropik, kübik, TI, ortotropik, triklinik örnekleri doğru adlandırılır; döndürülmüş izotropik yine izotropik | tanım | doğru sınıf |
| L6 | Akustik tensör Q(n): izotropikte özdeğerler μ, μ, λ + 2μ, örneklenen bütün n'lerde | kapalı form | ≤ 1e-13 göreli |
| L6b | 120 rastgele SPD'de örneklenmiş min özdeğer > 0 (PD ⇒ SE) | teori | > 0 |
| L6c | μ = 1, λ = −1.5: PD reddedilir, örneklenmiş SE min = min(μ, λ + 2μ) = 0.5; λ + 2μ < 0 örneğinde örneklenmiş min < 0 | kapalı form | ≤ 1e-12 / < 0 |
| L7 | Düzlem gerilme: esneklik biçimi = C üzerinden Schur tümleyeni (σzz, σyz, σxz = 0), 120 rastgele C | blok ters özdeşliği | ≤ 1e-12 · κ |
| L7b | İzotropik → `D_plane_stress`; 3-B referans: rastgele anizotropik C, rastgele düzlem içi ε, düzlem dışı ε'ler σ_o = 0'dan 6 × 6 çözümle: σ_p = Q ε_p ve geri kazanım matrisi | 3-B çözüm | ≤ 1e-12 göreli |
| L7c | Yalnız σzz yoğuşturma: z etrafında döndürülmüş ortotropikte tamla aynı; genel döndürmede fark > 1e-3 göreli (eski yaklaşımın yetersizliği, negatif kontrol) | tam Schur | ≤ 1e-13 / > 1e-3 |
| L7d | Düzlem şekil değiştirme: Q = C_pp; σzz, σyz, σxz geri kazanımı 3-B C ε'ye eşit | 3-B | ≤ 1e-13 göreli |
| L7e | dQ_plane_stress / dC analitik, merkezi farka karşı | merkezi fark | ≤ 1e-6 göreli |
| P1 | İzotropikte E(n) = E, ν(n, m) = ν, G(n, m) = G, β(n) = 1/(3K), 2000 Fibonacci yönü | kapalı form | ≤ 1e-12 göreli |
| P2 | E(n), ν(n, m), G(n, m), β(n) tensör formülleri, n'yi 1. eksen yapan döndürülmüş Voigt S'nin S'11, −S'12/S'11, 1/S'66'sına karşı (50 malzeme × 20 yön) | bağımsız yol | ≤ 1e-11 göreli |
| P3 | Malzeme ve yönler aynı Q ile döndürülünce değerler değişmez | kendisi | ≤ 1e-12 göreli |
| P4 | Sabit n'de m ⊥ n üzerinden ν ekstremumu (2 × 2 özproblem) | 3600 noktalı çember taraması; sınırlı skaler rafine | taramaya ≤ 1e-5 göreli, rafineye ≤ 1e-10 |
| P5 | n küresi: Fibonacci N = 2000 ve 8000 + çoklu başlangıçlı rafine; iki N'nin rafine sonucu; rafine ≥ örneklenmiş (maks için) | kendisi | ≤ 1e-8 göreli |
| P6 | İzotropik tarama ν ∈ {−0.99, −0.9, −0.5, 0, 0.2, 0.3, 0.45, 0.49, 0.499, 0.4999}, E sabit ve G sabit: Mandel κ(C) = max(3K, 2G)/min(3K, 2G) | kapalı form | ≤ 1e-10 göreli |
| P6b | Aynı tarama: K/G, λ, μ, M, dalga hızları (ρ = 1), birim ε'de enerji, float32/float64 S = C⁻¹ gidiş-dönüş hatası | ölçüm | yalnız kayıt |
| P7 | Sınırsız ν ortotropik SPD ailesi: E1 = k², E2 = E3 = 1, ν12 = ν13 = ±0.5k, ν23 = 0, G = 1, k ∈ {1, 10, 100, 1000}: analitik PD koşulu (ν12² < E1/2) ve Cholesky kabul; ν12 = 0.5k sınırsız büyür | analitik | kabul |
| P8 | Kübik aile (Ting & Chen 2005'in abstract'ında adı geçen sınıf; tam metin okunamadı): PD sınırına yaklaşan parametrede örneklenmiş/rafine max\|ν\| | — | **yalnız ölçüm, eşik yok** |
| P9 | dC/dE, dC/dν (izotropik) analitik; dS = −S dC S; 2-B E(θ) açı türevi | merkezi fark | ≤ 1e-7 / 1e-7 / 1e-6 göreli |
| P9b | Taylor kalanı (6 yarılama): birinci mertebe düzeltilmiş kalanın eğimi | teori | ≥ 1.9 |
| P9c | ν = 0.4999 ve ν = −0.999'da aynı FD kontrolü | merkezi fark | ≤ 1e-6 · κ |
| P10 | Cholesky parametrizasyonu θ → C: 1000 rastgele θ (büyük değerler dahil) SPD; θ → C → θ | kendisi | Cholesky geçer / ≤ 1e-12 |
| P10b | Mühendislik sabiti parametrizasyonuna karşı: rastgele adımlarda SPD dışına çıkma oranı | ölçüm | yalnız kayıt |
| P11 | Genel HS üs fonksiyonu = eski `simp_admissible_penal` (düzlem gerilme, 3-B), ν ızgarası | eski kod | ≤ 1e-14 göreli |
| P11b | x ∈ (0, 1] 10⁴ noktada, 2-B düzlem gerilme, 2-B düzlem şekil değiştirme, 3-B: p* ile x^p K ≤ K_HS(1 + 1e-12) ve x^p G ≤ G_HS(1 + 1e-12); 0.95 p* ile ihlal | HS sınırları | eşitsizlik |
| P11c | Düzlem şekil değiştirmede p* ≠ düzlem gerilme p* (ν = 0.3: 3.5 ve 40/13) | kapalı form | ≤ 1e-14 göreli |
| R1 | `tensor_elastisite_parmakizi.py`: taban `1d1f027`'de kaydedilen 63 dizi (`tensor_elastisite_oncesi.json`, değerler + SHA-256) | eski kod | `D_iso`, `D_plane_*`, `ke_*`, ölçütler, von Mises/Tresca, `nonlinear` yardımcıları **bit düzeyinde**; geri kalanı ≤ 1e-15 göreli |
| R2 | `pytest -m "not slow"`: tabanda geçen hiçbir test kırılmaz; `-m baseline` fark 0; ruff, mypy temiz | taban | aynı |

P8'in neden eşiksiz olduğu: Ting & Chen (2005), QJMAM 58(1) 73–82, https://doi.org/10.1093/qjmamj/hbh021. Bu
oturumda yalnız yayıncı sayfasındaki abstract okundu [Ö]: anizotropik malzemede, şekil değiştirme enerjisi pozitif
tanımlıyken Poisson oranının keyfi büyük pozitif ya da negatif olabileceği ve bunun kübik malzemeler için fiziksel
olarak gerçekçi olduğu söyleniyor. Hangi sabit birleşiminin ve hangi (n, m) yönlerinin sınırsızlığı verdiği tam
metinde ve okunamadı. Bu yüzden kübik aile ön kayıtlı bir geçti/kaldı ölçütü değil; geçti/kaldı ölçütü analitik
olarak doğrulanabilen P7 ailesidir. P7 bir fiziksel malzeme kataloğu verisi değildir.

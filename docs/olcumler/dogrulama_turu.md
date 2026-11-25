# Doğrulama turu: bağımsız karşılaştırmalar, üretilmiş çözümler, Taylor testi, değişmezler (25 Kasım 2026)

Amaç: README'nin "Known limitations" bölümündeki "stress and buckling have not been compared with independent codes"
maddesini ve `gelistirme_plani.md` V1'in kalan kısmını kapatmak; MMS doğrulamasını yeni fiziklere (eleman başına
ortotrop D, ısı iletimi, geometrik doğrusal olmama) genişletmek; gradyan doğrulamasını adım boyuna bağlı olmayan bir
ölçüte (Taylor kalıntısı) taşımak; doğrudan çözücünün iç içe bölme sırasını SPD olduğu bilinen iki yola daha
(burkulma ön yükü, σ ≤ 0 modal analiz) uygulamak.

Etiketler `literatur_2026-11-21.md` ile aynı: **[Ö]** ölçüldü, **[O]** koddan okundu, **[H]** hafızadan.

"Bağımsız" burada şunu söyler: beklenen değer testin içinde, kütüphanenin eleman, malzeme, montaj, gerilme ve ölçüt
yordamları kullanılmadan, tanımdan yazılan NumPy/SciPy koduyla hesaplanır (kendi şekil fonksiyonları, kendi Gauss
kuralı, kendi döngüyle montaj, yoğun özdeğer çözümü). Başka bir ekibin kodu değildir; aynı yazarın ikinci, ayrı
yazılmış uygulamasıdır. Bu sınır README'de de böyle yazılır.

## 0. Ön kayıt (2026-11-25 21:05 +0300, yeni testlerin hiçbir koşusundan önce)

Eşikler kuramdan: doğrusal elemanın düğüm hatası O(h²); yoğun ile seyrek doğrudan çözümün farkı yuvarlama düzeyinde
(κ ≈ 1e3–1e5 küçük problemlerde → 1e-9 pay); düzgün bir fonksiyonun Taylor kalıntısı O(h²).

### B — Bağımsız burkulma (`tests/toporlogy/test_independent_buckling.py`)

K ve K_G testte kurulur: düğüm şekil fonksiyonları, 2 × 2(× 2) Gauss, σ = E_e D B u her Gauss noktasında,
K_G,e = ∫ ∇Nᵀ σ ∇N dV ⊗ I, düz döngüyle montaj; −K_G φ = μ K φ yoğun `scipy.linalg.eigh` ile.

| # | Kontrol | Eşik |
|:---|:---|:---|
| B1 | Q4 düzlem gerilme, 12 × 4, rastgele E ∈ [0.2, 1], sol ankastre, uçta basma + yanal yük: en büyük altı μ | \|Δμ\| ≤ 1e-9 · \|μ₁\| |
| B2 | H8, 6 × 3 × 3, aynı kurgu | B1 |
| B3 | Q4, eleman başına ortotrop D (rastgele açılar) | B1 |
| B4 | H8, eleman başına ortotrop D (z ekseni çevresinde rastgele açılar) | B1 |
| B5 | Kütüphanenin modları bağımsız özdeğer denklemini sağlıyor; K-normlu | ‖(−K_G − μK)φ‖ / (\|μ\| ‖Kφ‖) ≤ 1e-7; \|φᵀKφ − 1\| ≤ 1e-9 |
| B6 | Ön yük (ikinci yük durumunun K_G'si sabit): μ, (K + K_G0) ile yoğun çözüme karşı | B1 |

### S — Bağımsız gerilme ve ölçüt (`tests/toporlogy/test_independent_stress.py`)

Ağırlık merkezinde B testte türetilir; ortotrop eksenlere dönüş tensör olarak (Rᵀ σ R), Voigt dönüşüm matrisi
kullanılmadan; Tsai–Wu ve von Mises tanımdan.

| # | Kontrol | Eşik |
|:---|:---|:---|
| S1 | `centroid_stress`, Q4 ve H8, rastgele U | max fark ≤ 1e-12 · max \|σ\| |
| S2 | `FailureIndex` von Mises (eps = 0), izotrop, Q4 ve H8 | ≤ 1e-12 göreli |
| S3 | `FailureIndex` Tsai–Wu, eleman başına yön alanı, Q4 (düzlem gerilme) ve H8 | ≤ 1e-12 göreli |
| S4 | Q4 düzlem şekil değiştirme (`szz_row`), Tsai–Wu, yön alanı | ≤ 1e-12 göreli |

### M — Üretilmiş çözümler (`tests/toporlogy/test_fem_mms_extended.py`)

`test_fem_mms.py` ile aynı yöntem: sınırda sıfır olan düzgün alan, gereken hacim kuvveti testte, tutarlı düğüm
yükleri testte (3 noktalı Gauss). Değişken katsayılı ve doğrusal olmayan durumlarda hacim kuvveti b = −Div P, analitik
P(X)'in merkezi farklarıyla (δ = 1e-5) hesaplanır; bu farkın hatası (~1e-10) ölçülen FE hatalarının çok altında.
Oranlar ardışık ızgara çiftleri arasında log₂(e_h / e_{h/2}).

| # | Kontrol | Eşik |
|:---|:---|:---|
| M1 | Q4IM, h = 1/8, 1/16, 1/32 | her oran ∈ [1.8, 2.25]; en ince hata ≤ 1e-3 |
| M2a | Q4, ortotrop, açı θ(x) = 30° + 40° x/L + 20° y/H (eleman merkezinde örneklenen D) | oran ∈ [1.8, 2.25]; en ince ≤ 5e-3 |
| M2b | Q4IM, M2a ile aynı | M2a |
| M2c | H8, ortotrop, z çevresinde θ(x, y, z), h = 1/4, 1/8, 1/16 | oran ∈ [1.8, 2.25]; en ince ≤ 5e-2 |
| M2d | Negatif kontrol: FE'de sabit açı θ(merkez) | son oran < 1.0 |
| M3a | Isı iletimi (Poisson), Q4, h = 1/8, 1/16, 1/32 | oran ∈ [1.8, 2.25]; en ince ≤ 1e-3 |
| M3b | Isı iletimi, H8, h = 1/4, 1/8, 1/16 | oran ∈ [1.8, 2.25]; en ince ≤ 1e-2 |
| M3c | Negatif kontrol: iletkenlik 1.1 kat | son oran < 1.0 |
| M4a | SVK düzlem gerilme Q4, max \|∇u\| = 0.2 | oran ∈ [1.8, 2.25]; Newton yakınsar |
| M4b | SVK H8 | M4a |
| M4c | neo-Hooke düzlem şekil değiştirme Q4 | M4a |
| M4d | neo-Hooke H8 | M4a |
| M4e | Negatif kontrol: aynı yüklerle doğrusal `Elasticity` | en ince hata ≥ 10 × doğrusal olmayanın en ince hatası |

### T — Taylor kalıntısı (`tests/toporlogy/test_taylor_remainder.py`)

r(h) = \|f(x + h d) − f(x) − h ∇f·d\|, h_k = 0.05 · 2⁻ᵏ, k = 0 … 6; x ∈ [0.3, 0.7], d rastgele, max \|d\| = 1;
p_k = log₂(r_k / r_{k+1}). Doğru bir gradyanda p → 2, %1 yanlış bir gradyanda küçük h'de p → 1.

| # | Kontrol | Eşik |
|:---|:---|:---|
| T1 | Her yanıt için (aşağıdaki liste): ardışık iki p_k ∈ [1.8, 2.2] | geçer |
| T2 | Aynı yanıtlar, ∇f·d %1 bozulmuş: son iki p_k | ≤ 1.3 |

Yanıtlar: 2-B ve 3-B uyum; yer değiştirme (öz-eşlenik olmayan); von Mises p-normu; Tsai–Wu (yön alanı); burkulma KS;
özfrekans; iletim bağlantılılığı; çok fazlı uyum, faz hacmi, maliyet; doğrusal olmayan son uyum; GMG ile 3-B uyum.

### I — Değişmezler (`tests/toporlogy/test_frame_invariance.py`)

| # | Kontrol | Eşik |
|:---|:---|:---|
| I1 | 2-B problemi 90° döndür (ızgara, yükler, mesnetler), fiber açısı θ → θ + 90°: uyum, Tsai–Wu alanı (yerleri değişmiş), burkulma μ₁; Q4 ve Q4IM | uyum ve alan ≤ 1e-10, μ₁ ≤ 1e-8 göreli |
| I2 | 2-B ayna x → −x, θ → −θ | uyum ≤ 1e-10 |
| I3 | 3-B, z çevresinde 90°, eksen alanı döndürülmüş; H8 ve H8IM | uyum ≤ 1e-10 |
| I4 | Doğrusal olmayan nesnellik: rastgele şekil değiştirmiş duruma büyük rijit dönme Q bindir: W aynı, düğüm iç kuvveti Q ile döner; SVK (tek D ve yön alanı) ve neo-Hooke, 2-B (gerilme, şekil değiştirme) ve 3-B | W ≤ 1e-12, f ≤ 1e-11 göreli |
| I5 | Saf rijit hareket (1 rad dönme + öteleme): iç kuvvet ve enerji sıfır | ‖f‖ ≤ 1e-12 · ‖K0‖ · ‖u‖, W ≤ 1e-12 · ½uᵀK0u |

### O — Çözücü sıralaması, iki yeni yol (`tests/toporlogy/test_ordering.py` ekleri)

`cozucu_siralama.md`'deki iç içe bölme sırası şimdiye dek yalnız `Elasticity.factor` ve ısı iletiminde kullanılıyor.
İki yol daha SPD sistem çarpanlarına ayırıyor [O]: burkulmada ön yüklü K + K_G0 (`fem/buckling.py`, SuperLU
varsayılanı COLAMD; ön yük tek başına burkulmuyorsa SPD) ve σ ≤ 0 modal analizde K − σM (`fem/modal.py`).
PDE filtresi de SPD, ama bir koşuda bir kez çarpanlarına ayrılıyor; bu turda değişmez.

| # | Kontrol | Eşik |
|:---|:---|:---|
| O8 | Ön yüklü burkulma: iç içe bölme ile eski yol | μ ≤ 1e-9 göreli |
| O9 | Modal (σ = 0 ve σ < 0): iç içe bölme ile eski yol | özdeğerler ≤ 1e-9 göreli; σ > 0 eski yolda kalır |
| O10 | Hızlı paket değişmeden geçer; örnekler sayısal kipte REL_TOL içinde | geçer |

Süreler kayıt içindir, eşik değildir.

## 1. Sonuçlar

(İlk koşudan sonra doldurulacak.)

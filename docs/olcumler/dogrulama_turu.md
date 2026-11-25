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

## 1. Sonuçlar [Ö]

Değerler testlerin `-s` çıktısından (`python -m pytest <dosya> -s`). Ön kayıttan sapan her şey §2'de.

**Özet: B, S, M, I ve O'nun bütün kontrolleri ve T1 (11/11 yanıt) tuttu; T2 11 yanıtın 5'inde tutmadı. T2'nin
tutmaması gradyan hatası değil, negatif kontrolün tasarım hatası (§1.4). Bağımsız uygulamayla farklar yuvarlama
düzeyinde (≤ 4e-13), yakınsama oranları 1,97–2,05.**

### 1.1 B — burkulma

| # | Durum | Ölçülen | Eşik | Sonuç |
|:---|:---|---:|---:|:---|
| B1 | Q4 | 1,1e-13 | 1e-9 | PASS |
| B2 | H8 | 1,6e-14 | 1e-9 | PASS |
| B3 | Q4, yön alanı | 1,6e-13 | 1e-9 | PASS |
| B4 | H8, yön alanı | 1,5e-14 | 1e-9 | PASS |
| B5 | mod kalıntısı Q4 / H8 / Q4 alan / H8 alan | 5,5e-13 / 1,0e-13 / 7,3e-12 / 7,8e-14 | 1e-7 | PASS |
| B5 | \|φᵀKφ − 1\| (en büyüğü) | 8,7e-14 | 1e-9 | PASS |
| B6 | ön yük Q4 / H8 / Q4 alan / H8 alan | 2,2e-13 / 9,4e-15 / 4,0e-13 / 2,8e-14 | 1e-9 | PASS (§2.1) |

Ön yükün kütüphanedeki K_G0'ı, testin kendi K_G0'ıyla 3,5e-14 içinde aynı (2-B, ayıklama sırasında ölçüldü).

### 1.2 S — gerilme ve ölçüt

| # | Durum | Ölçülen | Eşik | Sonuç |
|:---|:---|---:|---:|:---|
| S1 | `centroid_stress` Q4 / H8 | 2,8e-16 / 3,1e-16 | 1e-12 | PASS |
| S2 | von Mises Q4 / H8 | 3,2e-16 / 2,9e-16 | 1e-12 | PASS |
| S3 | Tsai–Wu yön alanı Q4 / H8 | 1,1e-15 / 2,9e-16 | 1e-12 | PASS |
| S4 | Tsai–Wu düzlem şekil değiştirme Q4 | 6,5e-16 | 1e-12 | PASS |

### 1.3 M — üretilmiş çözümler

| # | Durum | Hatalar (kaba → ince) | Oranlar | Sonuç |
|:---|:---|:---|:---|:---|
| M1 | Q4IM | 9,4e-3 · 2,4e-3 · 6,0e-4 | 1,98 · 2,00 | PASS |
| M2a | Q4, yön alanı | 6,9e-3 · 1,7e-3 · 4,3e-4 | 2,01 · 2,00 | PASS |
| M2b | Q4IM, yön alanı | 1,0e-2 · 2,5e-3 · 6,3e-4 | 2,01 · 2,00 | PASS |
| M2c | H8, yön alanı | 1,1e-1 · 2,9e-2 · 7,1e-3 | 1,97 · 2,02 | PASS |
| M2d | sabit açı (negatif kontrol) | 0,353 · 0,352 · 0,353 | 0,01 · −0,00 | PASS (yakınsamıyor) |
| M3a | ısı Q4 | 5,1e-3 · 1,3e-3 · 3,2e-4 | 2,00 · 2,00 | PASS |
| M3b | ısı H8 | 1,1e-1 · 2,6e-2 · 6,4e-3 | 2,05 · 2,01 | PASS |
| M3c | iletkenlik × 1,1 (negatif kontrol) | 0,086 · 0,090 · 0,091 | −0,06 · −0,01 | PASS (yakınsamıyor) |
| M4a | SVK Q4 | 7,7e-3 · 1,9e-3 · 4,8e-4 | 2,01 · 2,00 | PASS |
| M4b | SVK H8 | 1,1e-1 · 2,6e-2 · 6,4e-3 | 2,02 · 2,01 | PASS (`slow`, ~60 s) |
| M4c | neo-Hooke Q4 düzlem şekil değiştirme | 8,2e-3 · 2,1e-3 · 5,1e-4 | 2,01 · 2,00 | PASS |
| M4d | neo-Hooke H8 | 1,0e-1 · 2,6e-2 · 6,3e-3 | 2,00 · 2,02 | PASS (`slow`, ~60 s) |
| M4e | doğrusal çözücü, aynı yükler (h = 1/32) | SVK 6,7e-2 ↔ 4,8e-4; neo-Hooke 3,9e-2 ↔ 5,1e-4 | ≥ 10× | PASS (138×, 77×) |

Testin kendi iki yardımcısı ayrıca sınandı: enerji tabanlı dönüşüm ↔ tensör dönüşümü (1e-13) ve neo-Hooke
P = ∂ψ/∂F (merkezi fark, 1e-7).

### 1.4 T — Taylor kalıntısı

On bir yanıtın hepsinde T1 tuttu: p_k 1,93 … 2,04'ten 2,00'ye gidiyor (`test_taylor_remainder.py -s`).

| Yanıt | T1 (ilk → son p_k) | T2 son iki p_k* | T2 |
|:---|:---|:---|:---|
| uyum 2-B | 1,98 → 2,00 | 1,26 · 1,15 | PASS |
| uyum 3-B | 1,97 → 2,00 | 1,25 · 1,15 | PASS |
| yer değiştirme (öz-eşlenik değil) | 1,99 → 2,00 | 1,42 · 1,27 | FAIL (strict xfail) |
| von Mises p-normu | 1,93 → 2,00 | 1,36 · 1,22 | FAIL (strict xfail) |
| Tsai–Wu, yön alanı | 2,04 → 2,00 | −1,16 · 0,53 | PASS |
| burkulma KS | 2,02 → 2,00 | 4,14 · −0,53 | FAIL (strict xfail) |
| özfrekans | 2,01 → 2,00 | 1,20 · 1,11 | PASS |
| iletim bağlantılılığı | 2,04 → 2,00 | 1,47 · 1,31 | FAIL (strict xfail) |
| çok faz: uyum / yumuşak faz hacmi / maliyet | 2,00 → 2,00 (üçü de) | 1,66 · 1,50 / −0,06 · 0,67 / 0,76 · 0,89 | uyum FAIL (strict xfail) |
| doğrusal olmayan son uyum | 2,01 → 2,00 | −2,85 · 0,38 | PASS |
| GMG 3-B uyum | 1,97 → 2,00 | 1,25 · 1,15 | PASS |

T2'nin neden tutmadığı: %1'lik gradyan hatası h · 0,01 · \|∇f·d\| ancak h < 0,01 \|∇f·d\| / c iken (c: eğrilik
terimi) kalıntıya baskın çıkar. Beş yanıtta bu sınır en küçük adımın (7,8e-4) altında; p* 1'e doğru iniyor ama son
iki adımda 1,3'ün altına inmiyor. Burkulma KS'de %1'lik hata işaretli kalıntıyı h ≈ 1,6e-3 yakınında sıfırlıyor (p* = 4,14,
sonra −0,53). Ölçüt h aralığına bakılmadan yazıldı; ön kaydın hatası budur. Gradyanın kendisi T1'de doğrulanıyor.
Sonradan bir "en küçük h'de bozuk kalıntı ≥ 10 × doğru kalıntı" ölçütü denendi, ölçülen oranlar 2,9–9,5 çıktı.
Eşiği ölçüme göre seçmek ön kaydı boşa çıkaracağı için o ölçüt eklenmedi.

### 1.5 I — değişmezler

| # | Ölçülen (en büyük) | Eşik | Sonuç |
|:---|:---|---:|:---|
| I1 uyum Q4 / Q4IM | 1,9e-15 / 2,1e-16 | 1e-10 | PASS |
| I1 Tsai–Wu alanı Q4 / Q4IM | 4,5e-15 / 6,4e-15 | 1e-10 | PASS |
| I1 μ₁ Q4 / Q4IM | 4,4e-15 / 6,5e-16 | 1e-8 | PASS |
| I2 ayna Q4 / Q4IM | 1,1e-15 / 6,6e-15 | 1e-10 | PASS |
| I3 H8 / H8IM | 4,5e-15 / 1,0e-15 | 1e-10 | PASS |
| I4 W / f, 9 durum (SVK tek D ve alan, neo-Hooke; düzlem gerilme, şekil değiştirme, 3-B) | 1,2e-15 / 3,7e-15 | 1e-12 / 1e-11 | PASS |
| I5 ‖f‖ / W, aynı 9 durum | ≤ 5e-16 (sınır ≥ 3,8e-12) / ≤ 2,6e-18 (sınır ≥ 3,7e-14) | — | PASS |

I4 ve I5 enerji interpolasyonu olmadan koşuldu: interpolasyonun doğrusal terimi (1 − γ²) K0 u düşük yoğunluklu
elemanlarda nesnel değil. Bu bir hata değil, modelin bilinen sınırı; o kipte nesnellik sınanmadı.

### 1.6 O — çözücü sırası

`docs/olcumler/dogrulama_turu.py`, değişiklikten önce ve sonra birer kez (`dogrulama_turu.json`); test koşusu yokken,
SuperLU.

| Problem | Serbest DOF | Önce | Sonra | Hızlanma | Özdeğer farkı |
|:---|---:|---:|---:|---:|---:|
| Burkulma + ön yük, 2-B 400 × 100 | 80 200 | 23,2 s | 17,4 s | 1,33× | 8,7e-15 |
| Burkulma + ön yük, 3-B 40 × 10 × 10 | 13 530 | 7,13 s | 4,74 s | 1,51× | 8,9e-16 |
| Modal σ = 0, 3-B 40 × 20 × 20 | 52 920 | 48,2 s | 18,5 s | 2,61× | 4,1e-15 |
| Modal σ < 0, 3-B 24 × 12 × 12 | 12 168 | 2,82 s | 2,02 s | 1,40× | 1,7e-15 |
| Modal σ > 0 (yol değişmedi) | 12 168 | 2,52 s | 2,53 s | 1,00× | 0 |

Burkulmada süre bütün `modes` çağrısı (Lanczos dahil); ön yüklü matrisin çarpanlara ayrılması onun bir parçası.
Testler: O8 (2-B, 3-B) ≤ 1,8e-15; O9 (σ = 0, σ < 0) ≤ 2,3e-15; σ > 0 bit düzeyinde aynı ve eski sırayla.

## 2. Ön kayıttan sapmalar ve test kurgusu düzeltmeleri

1. **B6:** ilk koşuda ön yük sabit −0,3 idi ve tek başına burkuluyordu (μ₀ = 4,3; kütüphane bu durumda hata verir).
   Ön yük, tek başına μ₀ = 0,4 verecek şekilde ölçeklendi (K_G yükte doğrusal). Eşik değişmedi.
2. **T, yer değiştirme:** ilk sürüm yer değiştirmeyi birim yükün noktasında ve yönünde ölçüyordu; değeri uyuma
   eşit, yani öz-eşlenik çıktı (f ve gradyan uyumla aynı). Ölçüm sağ üst köşeye ve x yönüne alındı. T2 bu durumda
   da tutmadı.
3. **T, kurgu (alt ajanın raporu, test dosyasında da yazılı):** doğrusal olmayan durumda ilk yük (0,05) Newton'u
   kesim değerini yükseltmeye zorladı (f tasarımda süreksiz); yük 5e-4'e indirildi, her değerlendirmede kesim 0,1
   kaldı. Gerilme yanıtlarında uyarlamalı ölçek kapatıldı (`adaptive: None`; durumlu, f çağrı geçmişine bağlı olurdu).
   Çok fazda sert faz hacmi kısıtı çıkarıldı: x'te doğrusal, kalıntısı yuvarlama düzeyinde, mertebesi anlamsız.
4. **M4b, M4d** `slow` işaretli (her biri ~60 s); sonuçları yukarıda, ilk koşudan.

## 3. Ek ön kayıt (2026-11-25 22:20 +0300, `test_independent_matrices.py`'nin ilk koşusundan önce)

README "the Q4 comparison is not independent (`ke_q4` comes from top88)" diyor. `test_modal.py`'nin kütle matrisi
karşılaştırması ise depo dışındaki eleman kitaplığına (aerokit) bağlı; bu makinede kurulu değil, test atlanıyor
(NOT VERIFIED). İkisi `_independent_fe` ile kapatılır: rijitlik ve kütle tanımdan, uyumsuz modlar
(1 − ξ², [−1, 1] üzerinde, her bileşen için) 2 × 2(× 2) Gauss ile ve statik yoğunlaştırmayla.

| # | Kontrol | Eşik |
|:---|:---|:---|
| K1 | İndirgenmiş K, Q4 ve H8, izotrop ve eleman başına ortotrop, rastgele E, 2-B kalınlık ≠ 1, h ≠ 1 | max fark ≤ 1e-12 · max \|K\| |
| K2 | Aynısı Q4IM ve H8IM (bağımsız yoğunlaştırma) | K1 |
| K3 | Tutarlı kütle matrisi, 2-B ve 3-B, rastgele ρ | K1 |
| K4 | Modal: en küçük altı λ, yoğun `scipy.linalg.eigh(K, M)`'ye karşı, Q4 ve H8IM | ≤ 1e-9 göreli |
| K5 | Isı iletim matrisi (`assemble_conduction`), 2-B ve 3-B | K1 |

### 3.1 Sonuçlar (ilk koşu) [Ö]

| # | Durum | Ölçülen | Eşik | Sonuç |
|:---|:---|---:|---:|:---|
| K1 | Q4 / H8 / Q4 alan / H8 alan | 4,6e-16 / 2,6e-16 / 3,3e-16 / 1,7e-16 | 1e-12 | PASS |
| K2 | Q4IM / H8IM / Q4IM alan / H8IM alan | 2,5e-16 / 2,9e-16 / 2,7e-16 / 2,0e-16 | 1e-12 | PASS |
| K3 | kütle 2-B / 3-B | 3,0e-16 / 2,3e-16 | 1e-12 | PASS |
| K4 | modal Q4 / H8IM | 2,2e-15 / 3,2e-14 | 1e-9 | PASS |
| K5 | iletim 2-B / 3-B | 2,3e-16 / 2,0e-16 | 1e-12 | PASS |

Uyumsuz modlu elemanlar (Wilson'ın Q6'sı ve dokuz modlu H8IM) bağımsız statik yoğunlaştırmayla aynı; kütle
matrisi, depo dışı eleman kitaplığı olmadan da doğrulanıyor.

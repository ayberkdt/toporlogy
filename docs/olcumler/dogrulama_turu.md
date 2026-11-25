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

T2'nin neden tutmadığı: %1'lik gradyan hatası h · 0,01 · \|∇f·d\|, doğru kalıntı c h²; ikisi h* = 0,01 \|∇f·d\| / c'de
eşit ve bozuk kalıntı ancak h ≪ h* iken birinci mertebe. İlk yazımda h*'ın en küçük adımın (7,8e-4) altında olduğunu
sanmıştım; T3'ün ölçümü (§6) h* = 1,1e-3 … 1,2e-2 verdi: sabit adımlar h*'a fazla yakın bitiyor, p* 1'e doğru iniyor
ama son iki adımda 1,3'ün altına inmiyor. Burkulma KS'de %1'lik hata işaretli kalıntıyı h ≈ 1,6e-3 yakınında sıfırlıyor (p* = 4,14,
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

## 4. Ek ön kayıt: optimizer'dan bağımsız KKT ölçüsü (2026-11-25 22:22 +0300, kod yazılmadan önce; ilk yazımda saat yanlışlıkla 22:35 yazılmıştı, commit `cd66b09` 22:22)

`Result.kkt_residual` bugün optimizer'ın son adımının çarpanlarını kullanıyor [O: `study.py` `_kkt_residual`]: bir
önceki iterasyona ait bir tahmin, çarpan vermeyen optimizer'da (SiMPL) yok. Ek: `optimization.diagnostics.kkt_estimate`
dönen tasarımda çarpanları kendisi bulur. Etkin kısıtlar (g_j ≥ −`active_tol`) için λ ≥ 0, sınırda olmayan
değişkenlerde ‖∇f + Σ λ_j ∇g_j‖ en küçük olacak şekilde NNLS ile; etkin olmayanlara λ = 0. Ölçü, sınırlarda
izdüşürülmüş durağanlık (göreli, max \|∇f\|'e bölünmüş) ile kısıt ihlalinin (max(g, 0)) büyüğüdür. `Study.run` bunu
`Result.stationarity` olarak her optimizer için verir (kısıt varsa).

| # | Kontrol | Eşik |
|:---|:---|:---|
| P1 | Kapalı çözümlü problem: min Σ (x_i − a_i)², Σ x_i ≤ V, 0 ≤ x ≤ 1 (bazı a_i > 1, bazıları < 0): çözümde | ölçü ≤ 1e-12; λ = λ* ≤ 1e-10 göreli |
| P2 | Aynı problem, çözümden 1e-3 uzak bir nokta | ölçü ≥ 1e-4 |
| P3 | Etkin olmayan kısıt: λ = 0, ölçü = izdüşürülmüş gradyanın göreli büyüklüğü | tam (1e-14) |
| P4 | MBB (`test_result_contract.py`'deki 30 × 10) MMA, OC, GCMMA, SiMPL ile yakınsamış tasarım | ölçü < 0,1 × bir iterasyonluk tasarımınki |

### 4.1 Sonuçlar (ilk koşu; GCMMA ve SiMPL ilk koşuda test dosyasındaki içe aktarma hatasıyla düştü, düzeltilip koşuldu) [Ö]

| # | Ölçülen | Eşik | Sonuç |
|:---|:---|:---|:---|
| P1 | ölçü 1,4e-16; λ = 1,875 (kapalı çözüm 1,875) | 1e-12 / 1e-10 | PASS |
| P2 | 3,75e-3 | ≥ 1e-4 | PASS |
| P3 | λ = 0, ölçü = 1 tam | 1e-14 | PASS |
| P4 MMA | 1 iterasyon 1,000 → 57 iterasyon 1,94e-4 (optimizer çarpanlarıyla `kkt_residual` 1,81e-4) | < 0,1 × | PASS |
| P4 OC | 0,791 → 179 iterasyon 1,71e-2 (`kkt_residual` 2,08e-2) | < 0,1 × | PASS |
| P4 GCMMA | 1,000 → 52 iterasyon 3,62e-5 (`kkt_residual` 3,43e-5) | < 0,1 × | PASS |
| P4 SiMPL | 0,808 → 41 iterasyon 1,17e-2 (`kkt_residual` yok: çarpan vermiyor) | < 0,1 × | PASS |

Bir iterasyonluk tasarımda ölçünün tam 1 çıkması: hacim kısıtı henüz etkin değil (g < −1e-3), λ = 0 ve iç
noktalarda izdüşürülmüş gradyan gradyanın kendisi.

## 5. Ek ön kayıt (2026-11-25 22:46 +0300, commit `72a632e`, yeni testlerin ve ölçümlerin ilk koşusundan önce; ilk yazımda saat yanlışlıkla 22:58)

### 5.1 T3 — eşiğe göre adımlanan negatif kontrol, T4 — burkulma KS'de işaret iptali

T2 sabit h aralığı kullanıyordu; %1'lik hata h · 0,01 · \|g\| (g = ∇f·d), doğru kalıntı c h² (c = r₃ / h₃², T1'in
asimptotik bölgesinden). İkisi h* = 0,01 \|g\| / c'de eşitlenir. T3 adımları h*'a göre seçer: h_j = h* · 2⁻ʲ,
j = 3 … 6.

| # | Kontrol | Eşik |
|:---|:---|:---|
| T3a | Her yanıt: %1 bozuk gradyanla ardışık üç mertebe | hepsi ∈ [0,8, 1,2] |
| T3b | Aynı adımlarda doğru kalıntı / bozuk kalıntı | ≤ 0,2 |
| T4 | Burkulma KS: işaretli kalıntı s(h) = f(x + hd) − f − h g'nin eğriliği c_s = s(h₃) / h₃² ile 0,01 g aynı işaretli ve iptal noktası h_c = 0,01 g / c_s, T2'deki sıçramanın bulunduğu [h₆, h₄] = [7,8e-4, 3,1e-3] aralığında | geçer |

### 5.2 O11, O12 — PDE filtresi ve GMG kaba ızgarası (karar kuralı)

İkisi de SPD (I + r² L; kaba ızgara K). Ölçüm: aynı matris, MMD (bugünkü) ve iç içe bölme sırası, çarpanlara ayırma
süresi. **Karar kuralı:** temsilî boyutta (PDE filtresi 3-B 60 × 30 × 30; GMG kaba ızgarası `direct_below` = 4000
elemana karşılık gelen en büyük kaba ızgara) hızlanma ≥ 1,3× ise sıra uygulanır ve sonuç eski yolla ≤ 1e-12 göreli
aynı olmalı; değilse değişmez ve ölçüm kayda geçer.

## 6. §5'in sonuçları (ilk koşu) [Ö]

| # | Ölçülen | Eşik | Sonuç |
|:---|:---|:---|:---|
| T3a | 13 nicelik (11 yanıt; çok fazda üçü): h* = 1,1e-3 … 1,2e-2; bozuk gradyanla mertebeler 0,90 … 1,08 | ∈ [0,8, 1,2] | PASS (hepsi) |
| T3b | doğru / bozuk kalıntı, en büyüğü (h*/8'de) | ≤ 0,2 | PASS (0,14) |
| T4 | c_s = 5,6e-3 ve 0,01 g = 7,6e-6 aynı işaretli; h_c = 1,36e-3 | [7,8e-4, 3,1e-3] | PASS |
| O11 | PDE filtresi 3-B 60 × 30 × 30 (54 000 bilinmeyen): MMD 5,20 s, iç içe bölme 4,84 s, 1,07×; çözüm farkı 7,6e-15 | ≥ 1,3× ise uygula | uygulanmadı |
| O12 | GMG kaba ızgarası 10 × 10 × 10 (3630 serbest DOF): MMD 0,466 s, iç içe bölme 0,092 s, 5,1×; çözüm farkı 0 | ≥ 1,3× ise uygula; sonuç ≤ 1e-12 | uygulandı, PASS |

Ölçümler `docs/olcumler/dogrulama_turu_o11_o12.py` → `dogrulama_turu_o11_o12.json` (en iyi 3 / 5 tekrar).

Ön kayıttan iki sapma:

1. **O12'nin kapsamı.** Ön kayıt "`direct_below` = 4000 elemana karşılık gelen en büyük kaba ızgara" diyordu. Kodda
   kaba ızgaranın sınırı `max_coarse_dof` = 4000 DOF [O: `elastic_gmg.py:145-151`]. Ayrıca GMG'nin varsayılan kaba
   çözümü cihazda yoğun Cholesky (`coarse="dense"`), SuperLU değil [O: `gmg.py:215`, `:334-337`]. Değişiklik yalnız
   tanı amaçlı `coarse="direct"` yolunu etkiliyor (o yolda kaba ızgara `amg_max_dof` = 60 000 DOF'a kadar
   büyüyebilir); varsayılan GMG yolu değişmedi. Ölçüm 4000 DOF sınırının altındaki 10 × 10 × 10 ızgarada yapıldı.
2. **O12'de farkın tam 0 çıkması** büyük olasılıkla V-döngüsünün düşük duyarlıklı veri türünden (kaba çözüm
   sonucu o türe yuvarlanıyor); doğrulanmadı. Testte iki yolun gerçekten koştuğu (`_order` var / yok) ayrıca sınanıyor.

T3 ile T2'nin beş xfail'i açıklanmış oldu: gradyanlar doğru (T1), negatif kontrol de h*'ın altında doğru çalışıyor
(T3); T2'nin sabit adım aralığı bu yanıtlar için h*'a fazla yakın bitiyordu. T2'nin xfail'leri ön kaydın sonucu olarak
yerinde kalıyor.

## 7. Ek ön kayıt: Timoshenko konsolu (2026-11-25 23:05 +0300, ilk koşudan önce)

Kapalı çözüm (Timoshenko & Goodier, *Theory of Elasticity*, konsol ucunda parabolik kesme; düzlem gerilme) [H:
formül hafızadan; test önce kendi tutarlılığını sınar]. Uzunluk L = 4D, y ∈ [−D/2, D/2], I = D³/12:

u_x = −P y / (6EI) · [(6L − 3x) x + (2 + ν)(y² − D²/4)],
u_y = P / (6EI) · [3ν y² (L − x) + (4 + 5ν) D² x / 4 + (3L − x) x²],
σ_xx = −P (L − x) y / I, σ_yy = 0, τ_xy = P (D²/4 − y²) / (2I).

FE problemi: x = 0 ve x = L yüzlerinde tam çekiş kuvvetleri (tutarlı düğüm yükleri, 3 noktalı Gauss), üst ve alt
yüz yüksüz, rijit hareket üç sıfır DOF ile tutulur; hata, FE − kapalı çözüm farkından en küçük kareler rijit hareketi
çıkarıldıktan sonra, max düğüm hatası / max \|u\|. Izgara (4n, n), n = 4, 8, 16 eleman derinlikte.

| # | Kontrol | Eşik |
|:---|:---|:---|
| H1 | Formülün tutarlılığı: u'dan gerilme = σ formülü, Div σ = 0, üst/alt yüz yüksüz, uçta ∫τ dy = P (merkezi farklar, δ = 1e-5) | ≤ 1e-7 göreli |
| H2 | Q4: hata oranları | ∈ [1,8, 2,25] |
| H3 | Q4IM: en ince hata ≤ 0,5 × Q4'ünki; oranlar ≥ 1,8 (hata < 1e-10 değilse) | geçer |
| H4 | Uç çökmesi u_y(L, 0), en ince ağ | Q4 ≤ 2e-2, Q4IM ≤ 2e-3 göreli |

### 7.1 Sonuçlar (ilk koşu) [Ö]

| # | Ölçülen | Eşik | Sonuç |
|:---|:---|:---|:---|
| H1 | u'dan gerilme 2,8e-10; denge 1,8e-11; üst/alt yüz 0; uç bileşkesi 1,1e-16 | ≤ 1e-7 | PASS: hafızadan yazılan formül tutarlı |
| H2 | Q4 hataları 4,2e-3 · 1,2e-3 · 3,4e-4; oranlar 1,78 · 1,82 | ∈ [1,8, 2,25] | FAIL (strict xfail) |
| H3 | Q4IM hataları 1,8e-4 · 7,3e-5 · 2,5e-5 (Q4'ün 1/13'ü); oranlar 1,34 · 1,52 | ≤ 0,5 × Q4 ve oran ≥ 1,8 | FAIL (strict xfail): doğruluk kısmı tutuyor, oran tutmuyor |
| H4 | uç çökmesi: Q4 1,8e-4, Q4IM 2,3e-6 | 2e-2 / 2e-3 | PASS |

Tanı amaçlı ek koşu (ölçüt değil): n = 32 ve 64'te Q4 oranları 1,85 · 1,86, Q4IM oranları 1,63 · 1,70. Oranlar
2'ye alttan, yavaş yaklaşıyor. Maksimum normda h² \|ln h\| davranışı bunu açıklayabilir; doğrulanmadı. Sınırda sıfır
yer değiştirmeli MMS problemlerinde (§1.3) oranlar ilk üç ağda 2,00'dı. Buradaki fark, problemin tamamen çekiş
yüklü olması (üç noktada rijit hareket tutuluyor) ve hatanın uçlarda yoğunlaşması olabilir; bu da sınanmadı. Bu
yüzden üç ağlık oran ölçütü bu problem için fazla sıkıydı; çözümün doğruluğu (H1, H4) ise ön kayıtla tutuyor.

### 7.2 H2/H3'ün nedeni (tanı, post hoc) [Ö]

En büyük düğüm hatası her ağda yüklü bir uç yüzün köşesinde: Q4'te x = 0, y = ±D/2; Q4IM'de x = L, y = −D/2. Köşeden
uzakta oranlar ikinci mertebe:

| Eleman | Maksimum (H2/H3) | RMS, bütün düğümler | Maksimum, açıklığın orta yarısı |
|:---|:---|:---|:---|
| Q4 | 1,78 · 1,82 | 2,06 · 2,05 | 2,06 · 2,04 |
| Q4IM | 1,34 · 1,52 | 2,14 · 2,13 | 2,00 · 2,01 |

Yani H2/H3'ün tutmaması çözücünün genel doğruluğuyla ilgili değil, yüklü köşelerdeki yerel bir etki. Q4IM'de etkinin
daha güçlü olması için ilk aday, yüklü kenar boyunca sıfır olmayan uyumsuz modlara düşen yük payının ihmal edilmesiydi.
**Sınandı ve çürüdü** [Ö]: bağımsız uygulamada (`_independent_fe`, yoğun çözüm) o yük payı da yoğunlaştırılınca hata
büyüyor (n = 4 … 32: 4,1e-3 → 1,0e-3, oranlar 0,42 · 0,71 · 0,85). Yalnız düğüm yükleriyle aynı bağımsız çözüm
kütüphanenin Q4IM sonucunu dört ağda da aynı basamaklarla veriyor (1,850e-4 · 7,321e-5 · 2,547e-5 · 8,212e-6).
Kütüphanenin seçimi (modlara yük yok) bu problemde doğru olan; köşe etkisinin nedeni, yeri dışında, açıklanmadı. Tanı `test_h5_..._post_hoc` olarak testte
(ön kayıtlı değil, bu ölçümden sonra yazıldı; eşik H2'nin aralığı).

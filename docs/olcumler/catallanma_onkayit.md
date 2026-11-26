# Çatallanma tespiti ve dal değiştirme (`solvers/continuation.py`) — ön kayıt

Ön kayıt 2026-11-26 12:40 +0300 (`date`), taban `cc5d5b9` (`research/round-2`), kod yazılmadan ve hiçbir kontrol
koşulmadan önce. Parmak izi çapası `catallanma_parmakizi.py` bu tabanda koşulup `catallanma_parmakizi_oncesi.json`'a
yazıldı (aynı commit). Gerekçe: `nonlineer_cozucu.md` A6b — sığ kemerde simetrik limit noktasından önce antisimetrik
çatallanma var, yay boyu birincil yolda kalıyor, kararsız noktaları kabul ediyor; dal değiştirme yok.

## 1. Yöntem (uygulanacak olan)

R(x, λ) = f_int(x) − λ q = 0, K = ∂R/∂x simetrik.

- **Spektrum** (`TangentSpectrum`): her kabul edilen noktada K'nin özdeğerleri. `dense`: tam simetrik matrisin
  bütün özdeğerleri (numpy `eigh`), indeks = negatif özdeğer sayısı (Sylvester ataleti, kesin). `sparse`: sıfıra en
  yakın k özdeğer (shift-invert `eigsh`, σ = 0); indeks bu pencerede negatif olanların sayısı (yerel indeks: pencere
  dışındaki negatif özdeğerleri görmez, belgelenecek). FE problemi teğeti üst üçgen saklıyor
  (`IndefiniteDirectSolver.factor(A_upper)`): tam matris A + triu(A, 1)ᵀ.
- **Tespit**: ardışık iki yakınsamış nokta arasında indeks değişimi; spektrum yoksa `det_sign` işaret değişimi
  (yalnız tek sayıda geçişi görür).
- **Konum**: A ile B arasındaki kiriş d = (x_B − x_A, λ_B − λ_A) boyunca t ∈ [0, 1]; her t'de R = 0 ve
  d_xᵀ(x − x_A) = t d_xᵀd_x kısıtıyla (λ bilinmeyen) sınırlı Newton (yer değiştirme kontrolünün genelleştirilmesi:
  δx = δx_R + δλ δx_q, δλ = (w − d_xᵀ(x + δx_R))/(d_xᵀ δx_q)). Geçen özdeğer μ(t) (A'da sıralı spektrumun
  j = min(m_A, m_B)'inci elemanı, sonra özvektör örtüşmesiyle izlenen) için Illinois (değiştirilmiş regula falsi);
  dur: |μ| ≤ 1e-10 · max(|μ(0)|, |μ(1)|) ya da t aralığı ≤ 1e-14, en çok 60 değerlendirme.
- **Sınıflama** (yük yönü testi): φ geçen özvektör (‖φ‖ = 1). |φᵀq|/‖q‖ ≤ 1e-6 → çatallanma, aksi hâlde limit
  noktası. Ek bilgi (sınıflamaya girmiyor): λ_A, λ_B < λ_c ya da > λ_c ise Δλ işaret değiştiriyor.
- **Dal değiştirme** (yalnız çatallanmada): φ'nin işareti en büyük bileşeni pozitif olacak şekilde sabitlenir.
  Hiperdüzlem kısıtı φᵀ(x − x_c) = ξ ile (λ bilinmeyen) sınırlı Newton, tahmin (x_c + ξφ, λ_c). Simetrik birincil
  yolda φᵀ(x − x_c) ≡ 0 olduğundan birincil yol bu hiperdüzlemi kesmez. Ardından yay boyu bu noktadan, ilk tahmin
  yönü (x_s − x_c, λ_s − λ_c) ile (x_c'den uzaklaşan) devam eder.
- Spektrum verilmediğinde `arc_length` / `displacement_control` aritmetiği değişmez (B1).

## 2. Kapalı formlar (bu kayıtta türetildi)

### S: eksenel yaylı rijit olmayan çubuk + taban dönme yayı (simetrik çatallanma)

Taban (0, 0) mafsallı; uç düğüm (u, v), başlangıçta (0, L). X = u, Y = L + v, ℓ = √(X² + Y²), θ = atan2(X, Y).
U = ½ k_a (ℓ − L)² + ½ k_r θ²; düşey ölü yük λ, q = (0, −1). Gradyan ∇U = k_a(ℓ − L)∇ℓ + k_r θ ∇θ; Hessian
H = k_a ∇ℓ∇ℓᵀ + k_a(ℓ − L)∇²ℓ + k_r ∇θ∇θᵀ + k_r θ ∇²θ, ∇ℓ = (X, Y)/ℓ, ∇²ℓ = [[Y², −XY], [−XY, X²]]/ℓ³,
∇θ = (Y, −X)/ℓ², ∇²θ = [[−2XY, X² − Y²], [X² − Y², 2XY]]/ℓ⁴.

Birincil yol u = 0: ℓ_u = 0, ℓ_v = 1, θ = 0 → R_v = k_a(ℓ − L) + λ = 0 → ℓ = L − λ/k_a, v = −λ/k_a. Teğet:
K_uv = 0 (ℓ_u = θ = θ_v = ℓ_uv = 0), K_uu = k_a(ℓ − L)/ℓ + k_r/ℓ² = −λ/ℓ + k_r/ℓ². K_uu = 0 ⇔ λ ℓ = k_r ⇔
λ²/k_a − Lλ + k_r = 0 → **λ_c = k_a (L − √(L² − 4k_r/k_a)) / 2** (k_a → ∞: k_r/L). Kritik vektör φ = (1, 0),
φᵀq = 0 → çatallanma.

İkincil yol (ℓ, θ koordinatlarında, θ ≠ 0; (u, v) ↔ (ℓ, θ) ℓ > 0'da difeomorfizma, denge iki koordinatta aynı):
Π = U + λ ℓ cos θ → ∂_ℓ: k_a(ℓ − L) + λ cos θ = 0; ∂_θ: k_r θ − λ ℓ sin θ = 0. m(θ) = k_r θ / sin θ ile λℓ = m ve
ℓ = L − λ cos θ/k_a → (cos θ/k_a) λ² − Lλ + m = 0 → **λ_S(θ) = k_a (L − √(L² − 4 m cos θ / k_a)) / (2 cos θ)** (θ → 0'da
λ_c'ye giden kök). Parametreler: L = 1, k_a = 100, k_r = 1 → λ_c = 50 (1 − √0,96).

### A: polinom yay modeli (asimetrik çatallanma)

U = ½ k_a e² + ½ k_r u² + (β/6) u³, e = v + u²/(2L); q = (0, −1). R_u = k_a e u/L + k_r u + (β/2)u², R_v = k_a e + λ.
Teğet K_uu = k_a (u/L)² + k_a e/L + k_r + βu, K_uv = k_a u/L, K_vv = k_a. Denge: k_a e = −λ →
R_u = u (k_r − λ/L) + (β/2) u² = 0. Birincil u = 0 (K = diag(k_r − λ/L, k_a)): **λ_c = k_r L**, φ = (1, 0), φᵀq = 0.
İkincil: **λ_A(u) = L (k_r + β u / 2)** (u'da doğrusal: transkritik), v = −λ/k_a − u²/(2L). Parametreler L = 1,
k_a = 100, k_r = 1, β = 3 → λ_c = 1, λ_A = 1 + 1,5u.

### T: iki çubuklu kafes (limit noktası)

`tests/toporlogy/_truss.py` (türetme orada): P(w) = EA w (2h − w)(h − w)/L³, limit noktaları
w₁,₂ = h (1 ∓ 1/√3), λ = P/P_max = ±1; simetrik yolda K_uv = 0, kritik vektör φ = (0, 1), |φᵀq|/‖q‖ = 1.

## 3. Kontroller ve eşikler

| # | Kontrol | Eşik |
|:---|:---|:---|
| M0 | S ve A modellerinin teğeti = merkezi FD (h = 1e-6), 5 rastgele nokta | ≤ 1e-6 bağıl |
| S1 | S: yay boyu (ds = 0,01, rtol 1e-12, η_tol 0) λ = 1,5'e; tek kritik aralık; konum λ_c | \|λ − λ_c\|/λ_c ≤ 1e-8; \|v + λ_c/k_a\| ≤ 1e-8 L; \|u\| ≤ 1e-12 |
| S2 | S: sınıflama | çatallanma; \|φᵀq\|/‖q‖ ≤ 1e-8 |
| S3 | S: dal değiştirme ξ = 0,02 L, yay boyu \|u\| ≥ 0,3 L'ye | (a) her noktada yakınsama ölçütü yeniden hesapla sağlanıyor (‖R‖/‖q‖ ≤ 1e-12); (b) \|u\| ≥ 0,5 ξ her noktada ve \|u\| monoton artıyor, son \|u\| ≥ 0,3 L (max_steps 500 içinde); (c) \|λ − λ_S(θ)\| ≤ 1e-9 λ_c her noktada |
| A1 | A: aynı yol λ = 1,5'e; konum | \|λ − 1\| ≤ 1e-8; sınıflama çatallanma, \|φᵀq\|/‖q‖ ≤ 1e-8 |
| A2 | A: dal değiştirme ξ = +0,02 ve −0,02, \|u\| ≥ 0,2'ye | (a) S3a gibi; (b) S3b gibi (0,2 ile); (c) \|λ − λ_A(u)\| ≤ 1e-9 her noktada; (d) +ξ'de bütün noktalarda λ > λ_c, −ξ'de λ < λ_c |
| T1 | Kafes: yay boyu (A1 ayarları, ds = 0,005) w = 2,5h'ye; kritik aralıklar | tam iki aralık; ikisi de limit noktası (\|φᵀq\|/‖q‖ ≥ 0,5) |
| T2 | Kafes: konum | \|λ₁ − 1\| ≤ 1e-8, \|λ₂ + 1\| ≤ 1e-8; \|w − w₁,₂\| ≤ 1e-6 h |
| T3 | Kafes: yer değiştirme kontrolü yolu (v hedefleri, 50 adım, 2,5h'ye) üzerinde aynı konum | T2 eşikleri |
| T4 | Kafes, spektrumsuz: `det_sign` ile bulunan aralıklar spektrumlu indeksle aynı | birebir aynı indeksler |
| T5 | `switch_branch` limit noktasını reddediyor | `ValueError` |
| FC1 | FE kolon (Q4, SVK düzlem gerilme, ν = 0, E = 1, t = 1, h = 1; b = 4, L = 80, 4 × 80 eleman; taban ankastre, üstte birim basma bileşkesi, köşe düğümleri yarım ağırlık; `test_buckling._column` kurgusu): yay boyu, ilk kritik nokta; aynı ağ `LinearBuckling` λ_lin | \|λ_nl/λ_lin − 1\| ≤ 1e-2 |
| FC2 | FC1'in farkı inceliğe göre: d = \|λ_nl/λ_lin − 1\|, L = 80 ve L = 160 (4 × 160). Burkulma gerinimi ε_c = π²/(48 (L/b)²) ∝ (b/L)²; ön burkulma etkisi O(ε_c) ise oran ≈ 4 | d(80)/d(160) ∈ [2, 8] |
| FC3 | FC1 sınıflama; kritik vektörün kolon eksenine göre antisimetrik payı (ayna x → b − x: u_x(Mx) = u_x(x), u_y(Mx) = −u_y(x) antisimetrik) | çatallanma; \|φᵀq\|/‖q‖ ≤ 1e-8; antisimetrik pay ≥ 0,999 |
| FC4 | FC1, `dense` ↔ `sparse` (k = 6) | yol boyunca indeks dizileri aynı; konumlanan λ bağıl farkı ≤ 1e-9 |
| FB1 | FC1 kolonunda dal değiştirme: ξ, ξ · max\|φ_x(üst)\| = 1e-3 L olacak şekilde; yay boyu üst yanal yer değiştirme ≥ 0,05 L'ye | her noktada yakınsama ölçütü yeniden hesapla (rtol 1e-10 ya da η ≤ 1e-14) |
| FB2 | FB1: birincil yoldan ayrılma | üst yanal yer değiştirme her noktada monoton artıyor ve 200 adımda 0,05 L'ye ulaşıyor; birincil yolun kritik noktaya kadarki noktalarında \|u_x(üst)\| ≤ 1e-8 L |
| FB3 | FB1: ikincil yolda λ ≥ λ_c (1 − 1e-6) her noktada (beklenti: kararlı simetrik çatallanma — elastika; burada türetilmedi, beklenti olarak kayıtlı) | eşik |
| FA1 (yavaş) | A6b kemeri (`test_nonlinear_solvers._arch`, SVK, ds = 0,25, rtol 1e-12), yay boyu tepe çökmesi 3'e: ilk kritik nokta | çatallanma (\|φᵀq\|/‖q‖ ≤ 1e-8), φ'nin kemer eksenine (x = 20) göre antisimetrik payı ≥ 0,999, λ_b/λ_lim ∈ [0,99, 1) |
| FA2 (yavaş) | FA1 yolunda ikinci kritik nokta | limit noktası (\|φᵀq\|/‖q‖ ≥ 1e-3); λ_lim ≥ (1 − 1e-10) · yolun ayrık max λ'sı ve ≤ (1 + 1e-3) · ayrık max |
| FA3 (yavaş) | FA1 çatallanmasında dal değiştirme, ξ = 0,05, yay boyu 40 adım | her noktada yakınsama ölçütü (FB1 gibi); son noktada antisimetrik bileşen normu ≥ 10 ξ; birincil yolun FA1'e kadarki noktalarında ≤ 1e-8 |
| B1 | Spektrum verilmeden `arc_length` / `displacement_control` (kafes yay boyu, kafes yer değiştirme kontrolü, A5 FE konsolu) | `catallanma_parmakizi.py --check`: 3/3 SHA-256 bit düzeyinde eşit |
| B2 | Spektrum verildiğinde yol (λ, x her noktada) spektrumsuzla aynı (kafes, S) | bit düzeyinde eşit |
| B3 | Mevcut testler: `test_nonlinear_solvers.py` (yavaşlar dahil), `pytest -m baseline` | değişmeden geçiyor (strict xfail'lar xfail kalıyor); baseline 8 passed, bit düzeyinde |
| C1 | Sözleşme: indeks değişimi olmayan aralıkta `locate_critical`, `dense` n > `dense_limit`, bilinmeyen yöntem | `ValueError` |
| C2 | `stop_at_critical=True` (kafes) | yol w₁'den sonraki ilk noktada duruyor; son iki noktanın indeksi 0 ve 1 |

Ölçüm (eşiksiz): konum değerlendirme sayısı, konumdaki \|μ\|, dal değiştirme Newton iterasyonları, kolonda d/ε_c.

Başarısız kontrol strict xfail olur (ölçülen değerle), eşik gevşetilmez; ilk koşudan sonra eklenen kontrol "post hoc"
etiketlidir.

## 4. Sonuç (2026-11-26)

Testler: `tests/toporlogy/test_bifurcation.py` (değerler `-s` çıktısından; son kodda `-m "slow or not slow"`: 22 passed,
4 xfailed, 105 s). Kod: `solvers/continuation.py` (`TangentSpectrum`, `Spectrum`, `full_tangent`, `sparse_inertia`,
`find_critical`, `locate_critical`, `CriticalSettings`, `CriticalPoint`, `switch_branch`, `BranchSwitch`;
`arc_length(..., spectrum, start_tangent, stop_at_critical)`, `displacement_control(..., spectrum)`). `newton.py`
değişmedi. Ortam: Python 3.12.1, numpy 2.2.6, scipy 1.14.1, Windows 11, CPU.

### Özet

| Grup | PASS | FAIL (strict xfail, ölçülen değerle) | Not |
|:---|:---|:---|:---|
| M | M0 | — | |
| S simetrik | S1, S2, S3 | — | |
| A asimetrik | A1, A2 post hoc (ds = 0,0137) | A1, A2 (±) ön kayıtlı ds = 0,01 | yol tam tekil noktaya düşüyor |
| T limit | T1, T2, T3, T4, T5 | — | |
| FC / FB FE kolon | FC1, FC2, FC3, FB1, FB2, FB3; FC4 indeks maddesi (post hoc) | FC4 yük maddesi (1,4e-9 > 1e-9) | |
| FA kemer (yavaş) | FA1, FA2, FA3 | — | A6b'nin çatallanması bulunuyor ve dal değiştiriliyor |
| B, C | B1 (3/3), B2, B3, C1, C2 | — | |

### Ölçülen değerler

- **M0** teğet ↔ merkezi FD: S 7,5e-11, A 1,7e-11 (eşik 1e-6).
- **S1/S2** 3 yol noktası, aralık [1]; λ_c = 1,010205144336429 (kapalı form 1,010205144336440), bağıl 1,1e-14;
  \|v + λ_c/k_a\| 1,2e-16 L, \|u\| = 0; 4 değerlendirme, \|μ\| 9,1e-15; çatallanma, φᵀq/‖q‖ = 0; indeks 0 → 1.
- **S3** dal değiştirme 2 Newton iterasyonu (u = 0,0200, λ = 1,01027247); 30 nokta, 0 ret; en kötü bağıl kalıntı
  1,6e-14; \|u\| 0,0200 → 0,3050 monoton; λ 1,010272 … 1,026556 (kararlı simetrik: artıyor); \|λ − λ_S(θ)\|/λ_c ≤ 1,6e-13.
- **A1/A2 FAIL (ön kayıtlı ds = 0,01)**: birincil yolda \|dx/dλ\| = 1/k_a = 0,01, ilk adım Δλ = 1 ve kabul edilen
  nokta tam λ = 1,0 = λ_c (doğrulandı: kabul noktaları λ = 0,0 ve 1,0, orada K = diag(0, 100)). Sonraki tahmin çözümü
  K⁻¹q tekil (`LinAlgError`), 11 ret, `ContinuationError` step_limit / singular_tangent. Yerleştirmenin değil
  `arc_length`'in kusuru: tam tekil kabul noktasından tahmin üretemiyor. **Post hoc** (aynı eşikler, ds = 0,0137):
  A1 aralık [0], λ = 1,000000000000000 (hata 0), 1 değerlendirme (μ(t) doğrusal), çatallanma, φᵀq = 0; A2 +ξ: başlangıç
  u = +0,02, λ = 1,03, 20 nokta, kalıntı ≤ 4,9e-16, λ 1,0300 … 1,3123 (> λ_c), \|λ − λ_A(u)\| ≤ 1,6e-15; −ξ: u = −0,02,
  λ = 0,97, λ 0,6868 … 0,9700 (< λ_c), ≤ 5,6e-16.
- **T1/T2** aralıklar [8, 31]; iki limit noktası, φᵀq/‖q‖ = −1,000 (φ = (0, 1), q = (0, −P̂)); λ = +1 ve −1 basılan
  12 basamakta tam (hata < 5e-13); w/h = 0,422649731 ve 1,577350269 (kapalı form 1 ∓ 1/√3 = 0,4226497308 /
  1,5773502692); 5'er değerlendirme.
- **T3** yer değiştirme kontrolü yolunda aynı aralıklar ve aynı değerler. **T4** det işaretiyle [8, 31] = indeksle
  [8, 31]. **T5**, **C1** `ValueError`. **C2** yol 10 noktada duruyor (son w/h 0,4500 > w₁/h 0,4226), son iki indeks
  [0, 1].
- **FC1/FC3** (4 × 80, sparse spektrum) 14 nokta, aralık [10]; λ_nl = 2,1158066780e-3, λ_lin = 2,1169285613e-3,
  **λ_nl/λ_lin − 1 = −5,300e-4** (= −1,031 ε_c, ε_c = π²/(48 · 20²)); 2 değerlendirme; çatallanma, φᵀq/‖q‖ = 2,2e-16;
  antisimetrik pay 1,000000000000.
- **FC2** d(80) = 5,2996e-4, d(160) = 1,3313e-4, **oran 3,981** (beklenen 4); d/ε_c = 1,031 ve 1,036: ön burkulma
  düzeltmesi bu SVK kolonda ≈ −ε_c.
- **FC4** indeks dizileri aynı (dense eigh ↔ sparse LDLᵀ: 11 × 0, 3 × 1). **Yük maddesi FAIL**: yerleşen λ bağıl
  farkı 1,435e-9 (eşik 1e-9). İki konum da 2 değerlendirmede ilk koşudan sonra eklenen özdeğer doğruluk tabanında
  (§5, madde 2) duruyor; bu tabanda yük ~1e-9 çözünürlükte. Post hoc: yalnız indeks maddesi (geçiyor).
- **FB** ξ = 0,7730 (ξ · max\|φ_x(üst)\| = 1e-3 L); dal değiştirme 1 iterasyon; 20 nokta, 0 ret; en kötü bağıl kalıntı
  3,3e-11 (rtol 1e-10); üst yanal yer değiştirme 1,00e-3 L → 0,0501 L monoton; λ/λ_c 1,00000035 … 1,00076372 (FB3: hep
  ≥ 1 − 1e-6, artıyor); birincil yolda üst yanal ≤ 1,1e-14 L; son noktada x − x_c'nin antisimetrik payı 0,999225.
- **FA** (yavaş, dense) 83 nokta, aralıklar [60, 65]. FA1: çatallanma λ_b = 4,3827388938e-3 (tepe çökmesi 2,2724),
  φᵀq/‖q‖ = −4,2e-9, antisimetrik pay 1,000000000. FA2: limit noktası λ_lim = 4,3989643824e-3 (w 2,4556),
  φᵀq/‖q‖ = −0,139; λ_lim / ayrık max − 1 = +2,60e-5; **λ_b/λ_lim = 0,996312**. FA3: 41 nokta, λ 3,78458e-3 …
  4,38272e-3 (ikincil dal düşüyor: kararsız), en kötü bağıl kalıntı 8,3e-14, son antisimetrik norm 8,242 (≥ 0,5);
  birincil yolda ≤ 9,3e-13. Not: `nonlineer_cozucu.md` A6b'deki "0,9972 λ_max" ilk det K < 0 ayrık noktasının oranıydı
  (tanım farklı); burada iki nokta da yerleştirildi.
- **B1** `catallanma_parmakizi.py --check` son kodda: 3/3 bit düzeyinde eşit (kafes yay boyu 51 nokta, kafes yer
  değiştirme kontrolü 51, FE konsol 14). **B2** kafes ve S: spektrumlu ve spektrumsuz yollar bit düzeyinde aynı.
  **B3** `test_nonlinear_solvers.py` (yavaşlar dahil): 28 passed, 6 xfailed; `pytest -m baseline`: 8 passed, bütün
  örneklerde en büyük bağıl fark 0,00e+00.

## 5. İlk koşudan sonra değişenler (eşikler değişmedi)

1. **Dense spektrum maliyeti**: 4 × 80 kolonda tek `numpy.linalg.eigh` (800 × 800) bu makinede 20 s (aynı anda çok
   oturum koşarken; tek iş parçacığıyla 0,34 s, `OPENBLAS_NUM_THREADS=1`) — OpenBLAS iş parçacığı aşırı aboneliği; ilk
   FE testi pytest zaman aşımına (900 s) takıldı. Kütüphane iş parçacığı ayarına dokunmuyor.
2. **Konum durma ölçütü**: ilk koşuda dense konum μ = −7,9e-17'ye indi ama \|μ\| ≤ 1e-10 · max(\|μ(0)\|, \|μ(1)\|)
   (≈ 1e-17) hiç sağlanmadı, t aralığı ölçütüyle 16 değerlendirmede durdu (yük doğrusal burkulmadan −5,30e-4, sonuçla
   aynı). Ölçüte geriye kararlı özdeğer çözücüsünün mutlak doğruluğu eklendi: \|μ\| ≤ max(…, 16 ε ‖K‖₁). FC4 yük
   maddesinin başarısızlığı bu tabandan.
3. **Sparse spektrum** (§1'deki tanımdan sapma): indeks artık pencere sayımı değil, SuperLU'nun simetrik modda
   köşegen pivotlu ayrışımından (P K Pᵀ = L D Lᵀ, U = D Lᵀ) Sylvester ataleti — kesin; köşegen dışı pivot alınırsa
   `LinearSolveFailure`. Pencere (sıfıra en yakın k özdeğer) sıralı spektrumda ardışık olduğundan küresel konumu
   (`Spectrum.offset`) biliniyor; konumda dense ile aynı konumsal kural (j = min(m_A, m_B)) kullanılıyor, özvektör
   izleme kaldırıldı. Post hoc sözleşme testi: rastgele belirsiz matrislerde (n = 50, 300; 21 ve 139 negatif) ve
   kaydırılmış FE kolon teğetlerinde atalet = `eigvalsh` sayımı, pencere değerleri küresel konumlarında ≤ 1e-12.
   Kolon testleri (FC1–FC3, FB) sparse spektrumla, FC4 ve FA dense ile koşuyor.
4. Test kodu: `_mirror_share` sıfır vektörde 0/0 uyarısı veriyordu (yalnız norm kullanılıyordu), korundu.

## 6. Kapsam ve sınırlar

- Yol bağımlı problemler (`store` taşıyan, ör. J2) için `locate_critical` ve `switch_branch` reddediyor: kabul
  noktaları arasındaki deneme noktaları yanlış durumla değerlendirilirdi.
- Çok katlı kritik nokta (`CriticalPoint.multiplicity` > 1) raporlanıyor ama dal değiştirme yalnız konumsal j'inci
  vektör boyunca; çok katlı dallar (ör. kare kesitli 3-B kolon) sınanmadı.
- `arc_length` tam tekil bir kabul noktasından ilerleyemiyor (A1/A2); önceki artış yönünü tahmin olarak kullanmak
  sonraki iş.
- Elastika ikincil yolunun kapalı formla karşılaştırması yapılmadı (FB3 yalnız kararlılık beklentisi).

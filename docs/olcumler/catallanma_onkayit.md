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

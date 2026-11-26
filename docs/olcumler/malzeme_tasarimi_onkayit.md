# Malzeme tasarımı (tensör karışımı, fiber açısı, malzeme alanı): ön kayıt

Yazıldı: 26 Kasım 2026 02:42 +0300 (`date`), kodun ilk satırından ve ilk koşudan önce. Taban:
`integrate/research-round-1` = `10daa03`. Ortam: Python 3.12.1, NumPy 2.2.6, PyTorch 2.5.1+cu121, SciPy 1.14.1,
Windows 11. Bu dosya koşudan sonra değiştirilmez; sonuçlar ve sapmalar `malzeme_tasarimi.md`'ye yazılır. Eşik
gevşetilmez: tutmayan kontrol strict xfail olur ve ölçülen değer kaydedilir; sonradan eklenen her ölçüm "post hoc"
diye etiketlenir.

Kapsam: `02_MATERIAL_DESIGN.md` §1 (malzeme alanı, adlandırılmış tasarım blokları), §3 (çok fazlı tensör
interpolasyonu, fiber açısı tasarım bloğu), §4 (referans problemler). §2 (konstitutif protokol) bu oturumun değil
(nonlinear-solvers). 3-B yön parametrizasyonu bu dilimde yok (fail-closed).

## 0. Seçilen formülasyon (koşudan önce)

* Fazlar: her fazın kendi `ElasticTensor`'u (`core.linear_elastic`), kütle yoğunluğu, maliyeti, isteğe bağlı
  iletkenliği. 2-B'de faz tensörü önce düzlem gerilme (ya da düzlem şekil değiştirme) ile 3 × 3'e indirgenir, karışım
  ve döndürme 3 × 3 üzerinde yapılır (z ekseni etrafında döndürme σzz, σyz, σxz'yi kendi aralarında taşıdığı için
  indirgeme ile değişmeli; A1 bunu sınar).
* Faz kesirleri: mevcut kutu alanları t ∈ [0, 1]^m ve çubuk kırma (stick-breaking) η = `phase_fractions(t)`
  (void + m faz, Σ η = 1). Filtre ve projeksiyon t'ye uygulanır (mevcut zincir).
* İki karışım kuralı, ikisi de pozitif tanımlı tensörlerin konveks bileşimi (enerji pozitifliği yapı gereği):
  * `simplex`: C = (1 − Σ_i η_i^p) C_void + Σ_i η_i^p C_i, C_void = ε · (1/m) Σ_i C_i. Fazların permütasyonuna
    simetrik. Σ η_i^p ≤ 1 olduğu için ağırlıklar negatif olmaz.
  * `recursive`: C⁽⁰⁾ = C_void = ε C_m, C⁽ʲ⁾ = C⁽ʲ⁻¹⁾ + t_j^p (C_j − C⁽ʲ⁻¹⁾); C_i = E_i D olduğunda mevcut
    `MultiPhaseSIMP` × D'nin aynısı. Sıra bağımlı (K4'te ölçülür, eşiksiz).
  * Her iki kural da ceza (penalization) kuralıdır, fiziksel homojenleştirme değildir; izotrop HS sınırları
    anizotrop C'ye uygulanmaz. Gri eleman fiziksel bir karışım iddiası taşımaz.
* Fiber açısı: 2-B eleman başına θ (ya da tek global θ). Tasarım değişkeni s ∈ [0, 1], θ = θ_lo + (θ_hi − θ_lo) s.
  Süzgeç: çift açı temsili (cos kθ, sin kθ), k = 2π / periyot, iki bileşen mevcut `DensityFilter` ile süzülür,
  θ̃ = atan2(S̃, C̃) / k. 2-B'de n ⊗ n = ½ (I + [[cos 2θ, sin 2θ], [sin 2θ, −cos 2θ]]) olduğundan k = 2 için bu,
  yönelim tensörünü süzüp baş eksenini almakla aynıdır; ayrı bir yönelim tensörü yolu eklenmedi. Süzülmüş vektörün
  boyu ρ̃ = |(C̃, S̃)| ∈ [0, 1] "tutarlılık" tanısı olarak raporlanır; ρ̃ → 0'da yön belirsizdir ve atan2 türevi
  ıraksar (kayıtta en küçük ρ̃ yazılır).
* Periyot (koşudan önce, kaynağına bakılarak düzeltilmiş ifade): dördüncü mertebeden her elastisite tensörü R = −I
  altında değişmez, dolayısıyla düzlem içi döndürmede C(θ + π) = C(θ) yalnız ortotrop değil **her** malzemede
  geçerlidir (komut metnindeki "π yalnız ortotropik simetride" ifadesi elastisite için doğru değil; A2 bunu sınar).
  Simetriye bağlı olan daha kısa periyottur: düzlem içi dörtgen (tetragonal / kübik) simetride π/2, izotropta açı
  anlamsız. `angle_period` sayısal olarak π ya da π/2 döndürür, izotrop fazda reddeder.
* Fizik: `TensorElasticity` (`fem/tensor_elastic.py`, yeni), `Elasticity`'nin eleman başına matris yolunun alt
  sınıfı; mevcut `fem/*.py` değişmez. Q4 ve H8'de eleman matrisi D'de doğrusaldır: KE(C) = Σ_a c_a KE(S_a)
  (S_a simetrik taban, 2-B'de 6, 3-B'de 21). Uyumsuz modlu elemanlar (Q4IM, H8IM) yoğuşturma nedeniyle D'de doğrusal
  değil: fail-closed. dJ/dc_{e,a} = −s λ_eᵀ KE(S_a) u_e.
* `Study`'ye adaptör (Study değişmez): interpolasyon `MaterialField`'ı (aktif elemanlarda C, istenirse kütle
  yoğunluğu, iletkenlik) üretir ve yayımlar, `TensorElasticity.solve` onu tüketir; bayat alan (aynı değerlendirmeye
  ait olmayan) hata fırlatır. Açı bloğu `embed` bağlantısında (`angle:` parametresi) tasarım vektörünün ikinci
  bloğudur; yoğunluk bloğu zincirin geri kalanından (filtre, projeksiyon, passive) olduğu gibi geçer.

Ortak malzemeler (testlerde): P1 izotrop E = 0,3, ν = 0,2; P2 izotrop E = 0,6, ν = 0,4; P3 ortotrop E1 = 1,
E2 = E3 = 0,2, G12 = G13 = 0,1, G23 = 0,07, ν12 = ν13 = 0,3, ν23 = 0,35, z etrafında 30°. ε = 1e-6.

## 1. Karışım (K)

| # | Kontrol | Eşik |
|:---|:---|:---|
| K1 | Saf faz: her köşe (void, P1, P2, P3) ve iki kural, 2-B ve 3-B: C = C_i | max\|C − C_i\| / max\|C_i\| ≤ 1e-14 |
| K2 | Pozitif tanımlılık: 2000 rastgele t ∈ [0,1]³ (tohum 0), iki kural, 2-B ve 3-B | λ_min(C(t)) ≥ (1 − 1e-12) · min(λ_min(C_void), λ_min(C_i)) |
| K3a | Özdeş fazlar, p = 1, üç faz = P3, iki kural: C(t) = C_void + s (C − C_void), s = 1 − η_void | ≤ 1e-14 göreli |
| K3b | `recursive`, p = 3, C_i = E_i D (E = 0,3; 0,6; 1): C(t) = E_MultiPhaseSIMP(t) · D | ≤ 1e-14 göreli |
| K4 | Permütasyon, `simplex`: fazlar ve η birlikte permüte | ≤ 1e-14 göreli; `recursive` yalnız ölçülür |
| K5 | Simplex korunumu: Embed(m = 3) → DensityFilter(2,5) → HeavisideProjection(β = 8) → Passive, rastgele x | η ≥ −1e-15, \|Σ η − 1\| ≤ 1e-14, t ∈ [−1e-15, 1 + 1e-15] |
| K6a | Paralel çubuk 8 × 4, iki faz A (E = 1, ν = 0), B (E = 0,25, ν = 0) katmanları yük boyunca; sağ kenara düzgün şekil değiştirme çözümünün kapalı form düğüm kuvvetleri | u_x(L) = εL her düğümde ve E_eff = E_Voigt, ≤ 1e-12 göreli |
| K6b | K6a, iki faz ν = 0,3 | ≤ 1e-12 |
| K6c | Seri çubuk (katmanlar yüke dik), düzgün çekme, ν = 0 | E_eff = E_Reuss ≤ 1e-12 |
| K6d | K6c, ν_A / E_A = ν_B / E_B (ν_A = 0,4, E_A = 1; ν_B = 0,1, E_B = 0,25) | ≤ 1e-12 |
| K7 | Bağımsız montaj (`_independent_fe`): 2-B 6 × 4 (h = 0,5, t = 0,7), 3-B 3 × 2 × 2 (h = 0,5); P1, P2, P3 (3-B'de P3 genel eksenlerle); t ∈ [0,05; 1] rastgele (tohum 1); iki kural; C testte formülden ayrıca hesaplanır | ‖K_lib − K_ind‖_max / ‖K_ind‖_max ≤ 1e-13; U ≤ 1e-10 göreli |
| K8a | frac bir kez: frac_e = 0,5 ⇒ C_e = 0,5 C(t_e) | ≤ 1e-15 göreli |
| K8b | Skaler limit, fizik düzeyi: 24 × 12 konsol, t ∈ [0,2; 1] rastgele; `MultiPhaseSIMP` + `Elasticity` ile `recursive` + `TensorElasticity` (C_i = E_i D) | uyum ≤ 1e-10, U ≤ 1e-9, dc/dt ≤ 1e-8 (max-norm göreli) |

K8b toleransı: KE taban birleşiminin yuvarlaması ~1e-15, bu ızgarada κ(K) ≲ 1e6 (t^p ≥ 0,008), ileri hata
≲ κ · eps ≈ 1e-10.

## 2. Açı (A)

| # | Kontrol | Eşik |
|:---|:---|:---|
| A1 | 3 × 3 düzlem içi döndürme (torch) = `plane_stress(rotate_stiffness(C6, R_z))` ve `_independent_fe` (`rotate` + `plane_stress`); P3, 13 açı | ≤ 1e-13 göreli |
| A2 | Periyot: rastgele SPD 6 × 6 (triklinik, tohum 2) C(θ + π) = C(θ) (3 × 3 ve 6 × 6); kübik (C11 = 1, C12 = 0,4, C44 = 0,2) C(θ + π/2) = C(θ); P3'te θ = 10° için ‖C(θ + π/2) − C(θ)‖ / ‖C‖ | ≤ 1e-14; ≤ 1e-14; ≥ 1e-3. `angle_period`: P3 ve triklinik π, kübik π/2, izotrop hata |
| A3 | dC/dθ: autograd = analitik (dR = R J) | ≤ 1e-13 göreli; merkezi FD (h = 1e-6) ≤ 1e-8 |
| A4a | Sarma: 10 × 10 dama tahtası ±85° (fiziksel olarak hepsi 90° ± 5°), rmin = 2,5, çift açı süzgeci | her elemanda \|wrap_π(θ̃ − 90°)\| ≤ 5° + 1e-12 |
| A4b | A4a'da doğrudan θ süzgeci (karşılaştırma) | ölçülür (iç elemanlarda ~90° sapma beklenir) |
| A4c | Düzgün alan θ ∈ {−80°, 0°, 45°, 89,9°}: θ̃ = θ (mod π) | ≤ 1e-12 rad |
| A5a | Eğik çekme, 8 × 4 levha, düzgün θ ∈ {0, 15, …, 90}°, P3'ün düzlem gerilme hali, sol kenar u_x = 0, sol alt köşe u_y = 0, sağ kenar düzgün çekme: E_x(FE) = σ² V / c ile kapalı form 1/E_x = c⁴/E1 + (1/G12 − 2ν12/E1) s²c² + s⁴/E2 | ≤ 1e-12 göreli |
| A5b | dc/dθ (global açı bloğu, FE üzerinden autograd) = σ² V d(1/E_x)/dθ (kapalı formun türevi) | ≤ 1e-8 göreli; 0° ve 90°'de mutlak ≤ 1e-10 c |
| A6 | 1-B tarama referansı: 40 × 20 konsol, yoğunluk 1 (tasarlanmaz), malzeme E1 = 1, E2 = 0,1, G12 = 0,05, ν12 = 0,3 (düzlem gerilme; E3 = 0,1, G13 = 0,05, G23 = 0,04, ν13 = ν23 = 0,3), sağ kenar ortasında aşağı yük. Tarama θ ∈ [−90°, 90°), 1° adım. MMA ile global θ, başlangıç 0°, 45°, −45°, 30 adım | her başlangıçta \|wrap_π(θ_opt − θ_argmin)\| ≤ 1° ve c_opt ≤ c_min,tarama (1 + 1e-9). "Global minimum" yalnız tarama periyotta tek yerel minimum gösterirse söylenir |
| A7 | A6 problemi, eleman başına θ (rmin = 2, çift açı), başlangıç taramanın en iyi düzgün açısı, MMA 40 adım | c_eleman ≤ c_düzgün,opt |
| A8 | YAML örneği (`examples/research/fiber_angle_cantilever2d.yaml`: yoğunluk + eleman açısı) | koşar, son uyum < ilk uyum, hacim ≤ sınır (1 + 1e-3) |

## 3. Gradyan (G)

| # | Kontrol | Eşik |
|:---|:---|:---|
| G1 | Uyumun yönlü türevi, merkezi FD (h = 1e-6), 12 × 6: (a) faz alanları m = 3, iki kural, filtre ile; (b) eleman açıları, çift açı süzgeci ile; (c) yoğunluk + açı birlikte; (d) 3-B faz alanları 4 × 3 × 3 | göreli ≤ 1e-6 |
| G2 | Taylor kalanı ‖f(x + hd) − f(x) − h gᵀd‖, h ∈ {1e-1, 5e-2, 2,5e-2, 1,25e-2}, G1 (a)–(c) | log-log eğimi ≥ 1,9 |
| G3 | Sıfır türevler (ayrı mutlak norm): izotrop fazın saf olduğu elemanlarda dc/dθ; keep elemanlarında | ‖g‖_∞ ≤ 1e-13 · max\|g\| |
| G4 | Taban doğrusallığı: KE(C) tabandan = `element_matrix_from_D(C)` (Q4, H8; P3, rastgele SPD) | ≤ 1e-14 göreli |
| G5 | Kütle ve maliyet yanıtının dt türevi, merkezi FD | ≤ 1e-8 göreli |

## 4. Eski yol ve uyumluluk (E, U, M)

| # | Kontrol | Eşik |
|:---|:---|:---|
| E1 | `pytest -m baseline` (örnekler bit düzeyinde) | geçer |
| E2 | `MultiPhaseSIMP`, `phase_fractions`, `phase_start`, `SIMP`, `RAMP`, `HashinShtrikman` çıktılarının parmak izi (SHA-256, sabit girdiler), değişiklikten önce alınır (`malzeme_tasarimi_oncesi.json`) | sonra aynı |
| E3 | Tek izotrop faz (C = E0 D(ν)) tensör yolu = SIMP + `Elasticity`, 2-B MBB (`examples/mbb2d.yaml` ızgarası küçültülmüş, 30 × 10), aynı x'te | uyum ≤ 1e-10, gradyan ≤ 1e-8; 10 MMA adımının amaç geçmişi ≤ 1e-6 göreli |
| E4 | Sabit açı alanlı tensör SIMP = mevcut ortotrop `angle_field` yolu (`unit_elasticity_field` + SIMP + `Elasticity`) | uyum ≤ 1e-10, U ≤ 1e-9 |
| U1 | Her yeni UNSUPPORTED kuralı yapılandırmada hata verir (`test_compatibility.py` probları) | geçer |
| U2 | Katalog belgeleri yeniden üretilir, `test_catalog.py` geçer | geçer |
| M1 | `MaterialField` yalnız aktif elemanlarda: inaktif elemanlı alanda C (n_active, k, k); kütle / iletkenlik istenmedikçe yok | şekil ve None denetimi |

Ölçüm (eşiksiz, karşılaştırma): iki fazlı konsol (48 × 24, P1 + P3, MMA 80 adım, faz hacim kısıtları) iki
kuralla: uyum, gri payı, faz hacimleri, süre. Tensör yolunun değerlendirme süresi SIMP yoluna göre (aynı ızgara).

Başarısızlık tanımı: eşiği aşan her kontrol FAIL; koşulamayan NOT VERIFIED. FAIL olan kontrol strict xfail olarak
testte kalır, ölçülen değer sonuç dosyasına yazılır.

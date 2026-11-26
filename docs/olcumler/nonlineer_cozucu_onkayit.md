# Nonlineer çözücüler, konstitutif protokol ve J2 plastisite: ön kayıt

Yazıldı: 26 Kasım 2026 02:51 +0300 (`date`), yeni kontrollerin ilk koşusundan önce. Taban:
`integrate/research-round-1` = `10daa03`; bu dalda yalnız parmak izi çapası (`e9379c0`,
`nonlineer_cozucu_parmakizi.py` + `nonlineer_cozucu_oncesi.json`, tabanda yazıldı) önce geldi. Bu dosya koşudan
sonra değiştirilmez; sonuçlar ve sapmalar `nonlineer_cozucu.md`'ye yazılır. Eşik gevşetilmez: tutmayan kontrol
strict xfail olur ve ölçülen değer kaydedilir; sonradan eklenen her ölçüm "post hoc" diye etiketlenir.

Kapsam: `04_NONLINEAR_SOLVERS.md` §1 (genel Newton, yük/yer değiştirme kontrolü, yay boyu), §2 (enerji
interpolasyonu riski), §3'ün J2 maddesi, §5'in ilgili kabulleri; `02_MATERIAL_DESIGN.md` §2 (konstitutif
protokol). Kapsam dışı (bu turda yok, sonuç notunda "ertelendi"): sonlu şekil değiştirme düzlem gerilme genel
anizotropi, neredeyse sıkıştırılamaz neo-Hooke, yol bağımlı adjoint (J2 tasarım gradyanı), izleyen yük,
kinematik pekleşme, sonlu şekil değiştirme plastisitesi, dal değiştirme (bifurcation branch switching).

Kod (planlanan): `src/toporlogy/core/constitutive.py` (protokol, SVK, neo-Hooke, küçük şekil değiştirme doğrusal
elastik, J2, malzeme noktası sürücüsü, AD kahini), `src/toporlogy/solvers/newton.py` (genel Newton, hata kodları,
line search, adım günlüğü), `src/toporlogy/solvers/continuation.py` (yük kontrolü, yer değiştirme kontrolü, yay
boyu), `src/toporlogy/fem/nonlinear.py` (protokole taşıma, küçük şekil değiştirme kinematiği, durum yönetimi),
`src/toporlogy/optimization/nonlinear.py` (yeni yanıt `EndStrainEnergy`). Testler:
`tests/toporlogy/test_constitutive.py`, `tests/toporlogy/test_nonlinear_solvers.py`. Ölçüm betiği
`docs/olcumler/nonlineer_cozucu.py` → `nonlineer_cozucu.json`.

## 1. Sözleşmeler (uygulama bunlara uyar; testler bunları sınar)

* `evaluate(kinematics, params, committed_state, request) -> Response(stress, tangent, energy, trial_state,
  dissipation, diagnostics)`. Kinematik `small` (ε = sym ∇u, gerilme Cauchy σ, tanjant dσ/dε) ya da `finite`
  (H = F − I, gerilme PK2 S, tanjant dS/dE); Voigt sırası `core.tensor` (2-B `[xx, yy, xy]`, 3-B `[xx, yy, zz, yz,
  xz, xy]`), gerilmede fiziksel kayma, şekil değiştirmede mühendislik kayması. Enerji her modelde zorunlu değil.
* `committed_state` salt okunur verilir (`writeable = False`); `trial_state` yeni dizilerdir. Commit yalnız
  yakınsamış adımda; başarısız iterasyon / adım bölünmesi trial'ı atar (rollback). Yük durumları arasında durum
  paylaşılmaz (her durum ilk durumdan başlar).
* Malzeme birim modüllüdür ve eleman modülü E_e ile çarpılır (J2'de σ_y ve H da birim E'ye göre verilir:
  homojen ölçekleme, iç değişkenler E_e'den bağımsız).
* Hata kodları (ayrı): `singular_tangent`, `divergence`, `stagnation`, `max_iterations`, `non_finite`,
  `invalid_deformation`, `return_mapping`, `line_search`, `step_limit`, `complex_roots` (yay boyu kısıtının reel
  kökü yok). Yalnız küçük güncelleme başarı sayılmaz: yakınsama kalıntı (bağıl ya da geri hata η) ile.
* Yay boyu seçimi: Crisfield silindirik yay boyu (ψ = 0; ψ > 0 küresel aile, seçenek). Gerekçe: kısıt yalnız yer
  değiştirme artışında olduğu için yük ile yer değiştirmenin birim uyumsuzluğu ölçekleme gerektirmez; kuadratik
  kısıt iki kök verir ve kök seçimi (önceki artışla en küçük açı) ile işaret seçimi (öngörücüde
  Δu_öncekiᵀ δu_q) açıkça uygulanır ve günlüğe yazılır; reel kök yoksa adım reddedilir ve yay boyu yarılanır. Riks
  / Ramm normal düzlem (doğrusal kısıt, kök sorunu yok) alternatifi değerlendirildi; normal düzlemin keskin
  eğrilikli yolda kesişmemesi de adım reddi gerektirir ve işaret seçimi sorunu aynıdır. Dal değiştirme garantisi
  YOK: simetrik bir problemde çatallanma noktası geçilir, ikincil dal izlenmez.
* Yer değiştirme kontrolü: bilinmeyen yük çarpanı λ, kısıt u_c(x) = hedef (Batoz–Dhatt tipi); bu bir öngörülmüş
  yer değiştirme (reaksiyonlu Dirichlet) DEĞİLDİR. Raporlanan yük λ F'dir.

## 2. Kontroller ve eşikler

### P: protokole taşıma (bit düzeyinde davranış)

| # | Kontrol | Eşik |
|:---|:---|:---|
| P0 | `python docs/olcumler/nonlineer_cozucu_parmakizi.py`: 13 yapılandırma (SVK / neo-Hooke × Q4 düzlem gerilme, Q4 düzlem gerinim, H8 × enerji interpolasyonu açık/kapalı + eleman başına D alanı, iki yük durumu, başarısız çözüm dahil) | 13/13 bit düzeyinde eşit, çıkış kodu 0 |
| P0b | Var olan `test_nonlinear.py`, `test_nonlinear_orientation.py`, `test_fem_mms_extended.py` M4 değişmeden geçer (V4 neo-Hooke strict xfail kalır); `pytest -m baseline` geçer | geçer |
| P1 | Malzeme noktasında AD kahini (torch autograd, ψ(E) ayrı yazılmış): SVK ve neo-Hooke, 3-B ve düzlem gerinim, 20 rastgele E (\|H\| ≤ 0,3, J ≥ 0,5), ν ∈ {−0,3, 0, 0,3, 0,45}; neo-Hooke düzlem gerilme: 3-B kahin tanjantının σ_zz, σ_yz, σ_xz yoğuşturması, modelin E₃₃'ünde | ψ, S: max\|Δ\|/max\|ref\| ≤ 1e-11; ℂ ≤ 1e-10 |
| P2 | Rijit dönme nesnelliği (finite): ψ(QF) = ψ(F), S(QF) = S(F), 3-B, 20 rastgele Q | bağıl ≤ 1e-12 |
| P3 | Trial committed'ı değiştirmez: J2 `evaluate` salt okunur committed ile hata vermez; committed baytları (SHA-256) önce/sonra eşit | eşit |
| P4 | Küçük şekil değiştirme doğrusal elastik model FE'de (Q4 düzlem gerilme, H8) `Elasticity` ile aynı U | bağıl ≤ 1e-12 |
| P5 | Elastik sınır: σ_y = 1e30 J2 FE çözümü P4'ün doğrusal çözümüne eşit | bağıl ≤ 1e-12 |

Tolerans gerekçesi (P1): ψ, S, ℂ birkaç yüz işlemle C⁻¹ (cond(C) ≤ ~6 bu örneklemde) üzerinden hesaplanır;
yuvarlama ~1e-13 beklenir, ×100 pay.

### J: J2 plastisite (küçük şekil değiştirme, izotropik pekleşme)

Parametreler (birim E = 1): ν = 0,3, σ_y = 1e-3, doğrusal pekleşme H = 0,05; Voce eki σ_∞ = 1,5e-3, δ = 500
(κ(α) = σ_y + Hα + (σ_∞ − σ_y)(1 − e^(−δα))). ε_y = σ_y/E.

| # | Kontrol | Eşik |
|:---|:---|:---|
| J1 | Tek eksenli gerilme (σ₂₂ = σ₃₃ = kaymalar = 0, karışık kontrollü malzeme noktası sürücüsü, yerel tol. 1e-14), çekme ve basma ayrı, 0 → ±10 ε_y, 50 adım; kapalı form σ = σ_y + E_t(\|ε\| − ε_y), E_t = EH/(E + H) | max\|σ₁₁ − σ_ref\| ≤ 1e-10 σ_y; max\|α − α_ref\| ≤ 1e-10 ε_y |
| J2 | Çevrimsel: 0 → +5ε_y → −5ε_y → +8ε_y, bacak başına 40 adım; referans: testte ayrı yazılmış 1-B izotropik pekleşme kapalı formu (bacak bacak kırılma noktaları); elastik boşaltmada Δσ = E Δε ve α değişmez | J1 eşikleri; boşaltmada α bit düzeyinde aynı |
| J3 | Orantılı karışık yol (sapma + hacimsel, ε = t ε̂, ‖dev ε̂‖ ile t = 1'de 10 ε_y eşdeğer): q = 3μ(σ_y + H ē)/(3μ + H), p = K tr ε | max\|Δq\|, \|Δp\| ≤ 1e-11 σ_y |
| J4 | Tutarlı algoritmik tanjant ↔ dönüş haritasının merkezi farkı (adım 1e-6 · max\|ε\|), 20 rastgele durum (yaklaşık yarısı plastik), doğrusal ve Voce; akma yüzeyine bağıl 1e-6'dan yakın noktalar hariç | max\|C − C_FD\| ≤ 1e-6 max\|C\| |
| J4b | Negatif kontrol: süreklilik (continuum) tanjantı FD'den en az bir plastik noktada (Δα ≥ ε_y) farklı | ≥ 1e-3 max\|C\| |
| J5 | Akma yüzeyi: plastik adımlarda \|f\| = \|q − κ(α)\|; elastik adımlarda f ≤ 0 | ≤ 1e-12 σ_y; f ≤ 0 |
| J6 | Dissipasyon: her artışta D = σ_y Δα ≥ 0 ve plastik iş σ_{n+1}:Δε_p ≥ 0; ayrık enerji özdeşliği Σ σ_{n+1}:Δε = ψ_e(son) + Σ ½Δε_e:C:Δε_e + Σ σ_{n+1}:Δε_p (geri Euler için cebirsel özdeşlik) | D, W_p ≥ −1e-14 σ_y ε_y; özdeşlik bağıl ≤ 1e-12 (Σ\|σ:Δε\|'ye göre) |
| J7 | FE homojen çubuk (H8 4×1×1 ve Q4 düzlem gerinim 4×1; simetri mesnetleri; uç yüzde düzgün çekme, σ = 2σ_y'ye 10 adım, sıfıra 10 adım): uç ε₁₁ ve kalıcı ε, malzeme noktası sürücüsüyle (aynı kontrol: σ₁₁ öngörülü, diğerleri σ = 0; düzlem gerinimde ε₃₃ = 0) | bağıl ≤ 1e-9; Gauss noktaları arası α farkı ≤ 1e-12 bağıl |
| J8 | FE konsol plastik (Q4 düzlem gerinim 20 × 5, h = 1, uç yükü kökte akma; 10 adım yükleme, 10 adım tam boşaltma): (a) hepsi yakınsar; (b) toplam Newton iterasyonu algoritmik tanjantla, süreklilik tanjantının 1/1,5'inden az ya da süreklilik tanjantı yakınsamaz; (c) kalıcı yer değiştirme max\|u_res\| ≥ 1e-3 max\|u\|; (d) her adımda Σ w σ_y Δα ≥ 0; (e) ayrık global enerji özdeşliği Σ λ_{n+1} Fᵀ Δu_n = Σ_gp w [ψ_e(son) + Σ ½Δε_e:C:Δε_e + Σ σ:Δε_p] | (a)–(d) ifade edildiği gibi; (e) bağıl ≤ 1e-8 |
| J9 | Rollback/checkpoint: (a) zorlanmış başarısız artıştan sonra committed baytları eşit; (b) 0 → λ₁ çöz, checkpoint al, yeniden başlatıp λ₂'ye: kesintisiz koşuyla aynı adım dizisinde U ve durum bit düzeyinde aynı; (c) iki yük durumlu çözümde ikinci durum tek başına çözülmüşle bit düzeyinde aynı | eşit |

### N: genel Newton

| # | Kontrol | Eşik |
|:---|:---|:---|
| N1 | Her hata kodu kurgulanmış bir problemde üretilir ve adım kaydında görünür: tekil tanjant (K = 0), ıraksama (ters işaretli tanjant), duraklama, iterasyon sınırı, sonlu olmayan kalıntı, ters eleman (neo-Hooke, tek adımda büyük basma), return-mapping (Voce, yerel iterasyon sınırı 1), adım sınırı (yük kontrolü min_step'te), karmaşık kök (yay boyu) | kod tam eşleşir |
| N2 | Line search: R(x) = atan(x), x₀ = 2: tam Newton başarısız (kod ∈ {max_iterations, stagnation, divergence}); emniyetli geri izleme (½‖R‖² Armijo) ile \|x\| ≤ 1e-14, ≤ 15 iterasyon | ifade edildiği gibi |
| N2b | Enerji merit'i yalnız korunumlu model ve iniş yönünde; belirsiz (indefinite) tanjantta kalıntı merit'ine düşer ve günlüğe yazılır | davranış |
| N3 | Adım günlüğü: her adımda λ, kalıntı, güncelleme normu, line search adımları, iterasyon, kod, min det F (finite), enerji (korunumlu), dissipasyon (durumlu), rollback bilgisi alanları dolu | dolu |

### A: yay boyu ve yer değiştirme kontrolü

Kafes: simetrik iki çubuk (von Mises kafesi), yarı açıklık a = 1, yükseklik h = 0,1, EA = 1, Green şekil
değiştirmeli SVK çubuk, tepe düğümü 2 serbestlik (u, w), düşey ölü yük λ P̂, P̂ = P_max. Analitik:
P(w) = EA w(2h − w)(h − w)/L³, L = √(a² + h²); limit noktaları w₁,₂ = h(1 ∓ 1/√3), P(w₁) = −P(w₂) = P_max =
2EAh³/(3√3 L³); yanal rijitlik K_uu = EA(2a² + w² − 2hw)/L³ > 0 (h < √2 a: çatallanma yok). Yay boyu Δs₀ = 0,005,
Δs ≤ Δs₀, 2,5h'ye kadar; Newton toleransı ‖R‖ ≤ 1e-12 ‖P̂‖.

| # | Kontrol | Eşik |
|:---|:---|:---|
| A1 | Her yakınsamış nokta analitik eğride: \|λ − P(w)/P̂\| ve \|u\| | ≤ 1e-9; \|u\| ≤ 1e-12 |
| A2a | Yol iki limit noktasını geçer: ayrık max λ ∈ [1 − 5e-3, 1 + 1e-9], min λ ∈ [−1 − 1e-9, −1 + 5e-3]; w 2,5h'ye ulaşır | ifade edildiği gibi |
| A2b | Limit noktası parabolik inceltme (max λ çevresindeki üç noktadan w'de parabol) | \|λ_tepe − 1\| ≤ 1e-3; \|w_tepe − w₁\| ≤ 0,05h |
| A3 | Kararlılık: det K işareti, limit noktasından bir adımdan uzak bütün noktalarda analitik dP/dw işaretiyle aynı | %100 |
| A4 | Negatif kontrol, yük kontrolü λ = 0 → 1,2 (uyarlamalı, 20 adım, min_step 1e-4): ya `step_limit` hatası ya da iki ardışık kabul edilmiş nokta arasında \|Δw\| ≥ (w₂ − w₁) sıçrama; hiçbir kabul edilmiş nokta kararsız dalda (det K < 0) değil | ifade edildiği gibi |
| A5 | FE sarmalayıcı (yay boyu ↔ yük kontrolü, sınır noktası yok): kalın konsol 16 × 6, h = 1, ν = 0,3, SVK düzlem gerilme, uç yükü toplam 0,06; yay boyu yolunun her λᵢ noktasında yük kontrolü çözümü | ‖u_yay − u_yük‖∞/‖u_yük‖∞ ≤ 1e-7 |
| A6 | FE basamaklı sığ kemer (Q4 düzlem gerilme SVK ν = 0,3, ızgara 40 × 8, h = 1; sütun i'de merkez yüksekliği y_c = round(3(1 − ((i + ½ − 20)/20)²)), kalınlık 2 eleman, aktif elemanlar y ∈ [y_c, y_c + 2); iki uçta alt dış düğüm mafsal (u = v = 0); tepe üst düğümü x = 20'de düşey yük): yay boyu yolu ↔ bağımsız yer değiştirme kontrolü (tepe düğümü w hedefleri yay boyu noktalarının w'leri) | \|λ_dc(wᵢ) − λ_yay,ᵢ\| ≤ 1e-7 max\|λ\|; w = 2·3'e kadar λ maksimumu yoksa A6 "SINIR NOKTASI YOK" (geçmez) |
| A6b | A6 kemerinde yük kontrolü negatif kontrolü (A4 ölçütü, λ hedefi 1,2 × yay boyu max λ) | A4 gibi |

### E: enerji interpolasyonu riski (§2)

| # | Kontrol | Eşik |
|:---|:---|:---|
| E1 | Sabit kesme (`cutoff_step = 0`), c ∈ {0,1, 0,3}: konsol 12 × 4 neo-Hooke düzlem gerilme, rastgele gri tasarım (E = 1e-6 + (1 − 1e-6)x³, x ∈ [0,05, 1]), uç yükü max\|u\| ≥ açıklığın %5'i; 8 bileşende merkezi FD (adım 1e-6) ↔ adjoint, son uyum | max\|g − g_FD\|/max\|g_FD\| ≤ 1e-6 |
| E2 | Ölçüm (eşiksiz): 24 × 8 konsol, neo-Hooke, varsayılan uyarlamalı kesme, P = 0,01; x(t) = x_a + t(x_b − x_a), x_a rastgele gri (tohum 1), x_b rastgele ikili; t ∈ [0,9, 1], 21 nokta: kesme(t), J(t), her değişim noktasında aynı tasarımda iki kesmenin J farkı (model sıçraması) ve komşu noktalar arası J farkının adjoint eğim tahminine oranı | ölçüm |
| E3 | Ölçüm (eşiksiz): seçenek karşılaştırması aynı küçük problemde (MMA, 30 iterasyon): (a) her değerlendirmede c₀'dan uyarlamalı (mevcut), (b) "monoton" kesme — kesme bir optimizasyon boyunca azalmaz, artışta optimizatör yeniden başlatılır (MMA geçmişi atılır), (c) "ret" — dönemin kesmesinden yüksek kesme gerektiren tasarım adımı reddedilir, hareket yarıya indirilir (en çok 5 kez), sonra (b) gibi kabul. Rapor: son J (ortak, en yüksek kesmede yeniden değerlendirme), değişim sayısı, yeniden başlatma sayısı, monotonluk ihlalleri, süre | ölçüm |
| E4 | Ölçüm (eşiksiz): ρ̃ = (E/E0)^(1/p) yoğunluk geri kazanımının gerçek ρ'dan farkı (SIMP E_min ∈ {1e-9, 1e-6}, p ∈ {1, 3} ve yorum üssü 3 iken SIMP p ∈ {1, 2, 3} sürdürme durumu) | ölçüm |

### D: V4 neo-Hooke strict xfail'ının nedeni (yeniden inceleme, ön kayıtlı hipotez)

Hipotez H: V4 kirişinde (4 katman, L/H = 50, α = 2, 10 adım) her adımın son kalıntısı yuvarlama tabanındadır
ve üç noktalı mertebe bu nedenle tabanlı bir değer içerir. φ = ε_mach (‖K_T‖₁ ‖u‖ + ‖λF‖)/‖λF‖ yakınsamış durumda.

| # | Kontrol | Eşik |
|:---|:---|:---|
| D1 | neo-Hooke ve SVK, λ ≥ 0,6 bütün adımlar: son kalıntı r_son ≤ 1e3 φ ve ondan önceki r ≥ 1e4 φ | bütün adımlarda |
| D2 | Ölçüm: 1e3 φ'nin üstündeki kalıntılarla kullanılabilir üçlü sayısı adım başına; SVK'nın V4'ü geçiren üçlüsünde payda \|log(r_k/r_{k−1})\| | ölçüm |

D1 tutarsa V4'ün başarısızlığı (ve SVK'nın geçişi) ölçütün tabanla etkileşimidir; tutmazsa neden başka yerdedir
ve kayıtta açık kalır. V4'ün kendisi strict xfail olarak kalır (ön kayıtlı başarısız kontrol).

### T: ikinci dereceden Taylor kalanı (yeni yanıt)

Yeni yanıt `EndStrainEnergy` W(u*(E), E) (yakınsamış dengede depolanan enerji; enerji interpolasyonu dahil).
r(h) = \|f(x + hd) − f(x) − h g·d\|, h_k = 0,05 · 2^(−k), k = 0..6, x ∈ [0,3, 0,7], max\|d\| = 1 (mevcut
`test_taylor_remainder.py` kuralı), filtre + SIMP + doğrusal olmayan çözüm, sabit kesme.

| # | Kontrol | Eşik |
|:---|:---|:---|
| T1 | `EndStrainEnergy`, SVK ve neo-Hooke 2-B: ardışık iki p_k | [1,8, 2,2] |
| T2 | Negatif kontrol: yön türevi × 1,01 ile son iki p_k | ≤ 1,3 |
| T3 | Ek: `EndStrainEnergy` FD (E1 kurgusu) | ≤ 1e-6 |

## 3. Başarısızlık kriterleri

Bir kontrol eşiği tutmazsa strict xfail + ölçülen değer; uygulama hatası bulunursa düzeltilir, eşik değişmez ve
düzeltme kayıtta "ilk koşuda bulunan hata" diye yazılır. P0 tutmazsa refactor geri alınır ya da fark bilinçli
olarak gerekçelendirilir (bu durumda P0 FAIL olarak kalır).

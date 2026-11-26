# Nonlineer çözücüler, konstitutif protokol ve J2 plastisite: sonuç

Ön kayıt: `nonlineer_cozucu_onkayit.md` (26 Kasım 2026 02:51 +0300, commit `cc08f30`, ilk koşudan önce). Taban
`10daa03`; parmak izi çapası `e9379c0` (tabanda yazıldı). Ham veri: `nonlineer_cozucu.json` (betik
`nonlineer_cozucu.py --run`, E3b için `--e3b 0.12`), figür `nonlineer_cozucu_sekil.{png,pdf}` (datawiz denetimi
`nonlineer_cozucu_sekil.denetim.txt`: 0 HATA · 0 UYARI). Testler: `tests/toporlogy/test_constitutive.py`,
`tests/toporlogy/test_nonlinear_solvers.py` (değerler `-s` çıktısından, aşağıda). Ortam: Python 3.12.1, Windows 11,
CPU, SuperLU/PARDISO yolu değişmedi (sürümler JSON'da `environment`).

## 0. Özet

| Grup | PASS | FAIL (strict xfail, ölçülen değerle) | Not |
|:---|:---|:---|:---|
| P protokol | P0 (13/13 bit düzeyinde), P0b, P1, P2, P3, P4, P5 | — | |
| J J2 | J1, J2, J3, J4, J4b, J5, J6, J7 (ε), J8 (a, c, d, e), J9 | J7 α maddesi (Q4 1,1e-12; H8 1,0), J8 (b) (27/34 = 0,79 > 1/1,5) | post hoc ikisi de geçiyor |
| N Newton | N1 (9 kod), N2, N2b, N3 | — | N3'te bulgu: kalıntı merit'i takılıyor |
| A yol izleme | A1, A3, A4, A5, A6 | A2a/A2b (ön kayıt tanım hatası), A6b (çatallanma) | A2 post hoc geçiyor |
| E enerji interp. | E1 | — | E2, E3, E4 ölçüm; E3 bilgi taşımıyor, E3b post hoc |
| D V4 nedeni | — | D1 (ikinci madde) | D2 ölçüm |
| T Taylor | T1, T2, T3 | — | |

## 1. Kontroller ve ölçülen değerler

### P

- **P0** `nonlineer_cozucu_parmakizi.py`: 13/13 yapılandırma bit düzeyinde eşit (U, her iterasyonun kalıntı ve η
  geçmişi, kesme / bölme / yeniden başlatma, eleman kuvveti / tanjantı, enerji, adjoint gradyanı, başarısız
  çözümün hata metni). Son kod üzerinde yeniden koşuldu: 13/13.
- **P0b** `test_nonlinear.py`, `test_nonlinear_orientation.py`, `test_fem_mms_extended.py`, `test_taylor_remainder.py`:
  değişmeden geçiyor (V4 neo-Hooke strict xfail olarak kaldı). `pytest -m baseline`: §6.
- **P1** AD kahini (en kötü, 20 durum, ν ∈ {−0,3, 0, 0,3, 0,45}): SVK 3-B/düzlem gerinim ψ ≤ 5,9e-16, S ≤ 2,3e-16,
  ℂ ≤ 9,6e-17; neo-Hooke 3-B ψ ≤ 1,1e-14, S ≤ 1,3e-14, ℂ ≤ 1,2e-15; düzlem gerinim ψ ≤ 2,2e-13, S ≤ 3,7e-15;
  düzlem gerilme (3-B kahinin σ_zz, σ_yz, σ_xz yoğuşturması) ψ ≤ 1,3e-13, S ≤ 5,9e-15, ℂ ≤ 2,9e-15. Eşik ψ, S ≤
  1e-11; ℂ ≤ 1e-10. SVK düzlem gerilme P1 dışında (2-B yasa, yoğuşturulmuş 3-B enerji değil; test atlıyor, gerekçeli).
- **P2** nesnellik: SVK S 2,8e-15, ψ 2,8e-15; neo-Hooke S 2,9e-15, ψ 4,1e-15 (eşik 1e-12).
- **P3** salt okunur committed ile J2 değerlendirmesi hata vermiyor, SHA-256 önce/sonra eşit, rollback sonrası eşit.
- **P4** küçük şekil değiştirme doğrusal modeli ↔ `Elasticity`: Q4 1,6e-14, H8 5,4e-15 (eşik 1e-12).
- **P5** σ_y = 1e30 J2 ↔ doğrusal: Q4 düzlem gerinim 5,4e-16, H8 1,9e-15.

### J (ν = 0,3, σ_y = 1e-3, H = 0,05; Voce σ_∞ = 1,5e-3, δ = 500)

- **J1** tek eksenli gerilme, çekme ve basma: gerilme 1,1e-15 σ_y, α 3,5e-15 ε_y, yanal gerilme 6,5e-19 (eşik
  1e-10).
- **J2** çevrimsel 0 → 5 → −5 → 8 ε_y: gerilme 1,1e-15 σ_y, α 1,0e-14 ε_y; 18 elastik boşaltma adımında α bit
  düzeyinde aynı ve Δσ = EΔε (≤ 1e-12 σ_y); son α = 20,5 ε_y.
- **J3** orantılı karışık yol: q 6,5e-16 σ_y, p 8,7e-16 σ_y (eşik 1e-11).
- **J4** tutarlı tanjant ↔ merkezi FD: doğrusal 1,4e-10, Voce 1,3e-10 (20 durum, 12 plastik; eşik 1e-6).
- **J4b** süreklilik tanjantı ↔ FD: 0,37 (Δα = 6,8 ε_y; eşik ≥ 1e-3): FD testi tutarlı ile tutarsız tanjantı ayırıyor.
- **J5/J6** plastik adımlarda |f| doğrusal 8,7e-16 σ_y, Voce 8,0e-14 σ_y (eşik 1e-12); elastik adımlarda max f = 0;
  D = σ_y Δα ve σ:Δε_p en küçük 0; ayrık enerji özdeşliği 5,5e-16 (doğrusal), 3,0e-16 (Voce) (eşik 1e-12).
- **J7** homojen çubuk ↔ malzeme noktası: ε₁₁ Q4 düzlem gerinim 4,0e-12, H8 5,7e-15; kalıcı ε 1,4e-12 / 5,7e-15
  (eşik 1e-9). **α maddesi FAIL**: adım başına bağıl yayılım Q4 1,1e-12 (ilk plastik adım, küçük α'nın
  yuvarlaması), H8 1,0 (λ = 0,5 tam akma gerilmesine düşüyor: f_trial = ±1e-19, noktaların bir kısmı α = 3,6e-19
  ile dönüyor, kalanı elastik). Post hoc (aynı eşik, yolun en büyük α'sına göre): Q4 3,2e-13, H8 1,5e-14 → geçiyor.
- **J8** plastik konsol (20 × 5, Q4 düzlem gerinim, ilk akmanın 1,6 katı, 10 + 10 adım): (a) bölme yok; (c)
  max|u_res| = 1,74e-3 ≥ 1e-3 · 8,07e-2; (d) adım dissipasyonu ≥ 0, toplam 5,27e-7; (e) global enerji özdeşliği
  9,6e-12 (dış iş 3,204e-6 = elastik 2,47e-8 + algoritmik 2,649e-6 + plastik iş 5,30e-7; eşik 1e-8). **(b) FAIL**:
  toplam iterasyon 27 (algoritmik) / 34 (süreklilik) = 0,79, eşik < 1/1,5. Toplamı elastik adımlar belirliyor;
  post hoc adım başına: plastik adımlar λ = 0,9 ve 1,0'da 3 ve 5 iterasyon (kalıntı 1e-1 → 2e-4 → 1e-9; 2e-2 → 5e-6
  → 7e-13: ikinci mertebe) ↔ 6 ve 9 iterasyon (1e-7 → 2e-9 → 4e-11: doğrusal, oran ≈ 0,02).
- **J9** (a) zorlanmış başarısız artıştan sonra committed baytları eşit, trial atıldı; (b) checkpoint + yeniden
  başlatma kesintisiz yolla bit düzeyinde aynı (U ve durum); (c) iki yük durumlu çözümde ikinci durum tek başına
  çözülmüşle bit düzeyinde aynı.

### N

- **N1** dokuz kod kurgulanmış problemlerde tam eşleşti: `singular_tangent`, `divergence`, `stagnation`,
  `max_iterations`, `non_finite`, `invalid_deformation` (neo-Hooke tek adım büyük basma), `return_mapping` (Voce,
  yerel iterasyon sınırı 1), `step_limit` (`NonlinearConvergenceError.code`, `cause = max_iterations`),
  `complex_roots` (yanlış tanjantlı yay boyu; `ContinuationError.code = step_limit`, `cause = complex_roots`).
- **N2** atan, x₀ = 2: tam Newton 6 iterasyonda `stagnation`; geri izlemeli line search 4 iterasyonda |x| = 7,6e-16
  (adım uzunlukları 0,422, 1, 1, 1).
- **N2b** belirsiz tanjantta enerji merit'i reddedildi ve günlüğe yazıldı; korunumsuz problemde "energy merit not
  available" notu.
- **N3** günlük alanları dolu (neo-Hooke, enerji merit'i ile; min det F 0,838). **Bulgu (ilk koşu, post hoc test):**
  kalıntı merit'li (½‖R‖²) geri izleme 8 × 3 neo-Hooke konsolda ilk artışta takılıyor (her iterasyonda adım ≈ 0,1,
  `stagnation`), tam Newton 4 iterasyonda, enerji merit'i 4 iterasyonda (tam adımlar) yakınsıyor. Eğilmede tam
  Newton adımı ‖R‖'yi geçici olarak büyütüyor (V4'teki 1e2–5e3 × ‖λF‖ sıçraması); kalıntı merit'i bu adımı
  reddediyor. Hiperelastik problemde varsayılan `merit="energy"` daha uygun; varsayılan `line_search="none"` kaldı.

### A

- **Seçim**: Crisfield silindirik yay boyu (gerekçe ön kayıt §1).
- **A1** kafes: 51 nokta, ret yok; |λ − P(w)/P̂| ≤ 1,1e-13, |u| = 0; w = 2,5h'ye ulaştı.
- **A3** det K işareti dP/dw işaretiyle 47/47 noktada aynı (limit noktasına bir yay boyundan yakın 4 nokta hariç).
- **A2a/A2b FAIL — ön kayıt tanım hatası**: "yolun max λ'sı ≤ 1 + 1e-9" diye yazıldı; 2,5h'ye kadarki yol w = 2h'den
  (P = 0) sonra çekme dalına giriyor ve uçta λ = P(2,5h)/P_max = 4,87 (analitik eğrinin kendisi, A1 geçiyor). Post
  hoc (w ∈ [0, 2h], aynı eşikler): ayrık max 0,997661, min −0,997661; parabolik inceltme 1,0001173 (w = 0,42292h,
  tam 0,42265h) ve −1,0001173 (w = 1,57708h, tam 1,57735h) → geçiyor.
- **A4** yük kontrolü negatif kontrolü: λ = 1,0000000000000002'ye kadar (limit noktası, 20 adımın sonuncusu tam
  limit yüküne düşüyor) kabul, sonra `divergence` ile 9 bölme ve `step_limit`; kararsız dalda kabul edilmiş nokta yok.
- **A5** FE sarmalayıcı (kalın konsol, max|u|/L = 0,31): 14 yay boyu noktasında yük kontrolüyle en kötü 1,6e-11
  (eşik 1e-7).
- **A6** basamaklı sığ kemer: limit noktası var (max λ = 4,399e-3, w = 2,47), 185 nokta, ret yok; bağımsız yer
  değiştirme kontrolüyle |Δλ|/max λ = 7,3e-15 (eşik 1e-7).
- **A6b FAIL**: yük kontrolü maksimumun %99,98'inde `step_limit` ile duruyor (son artış `stagnation`; sıçrama/başarısızlık maddesi tutuyor),
  ama 3 kabul edilmiş noktada det K < 0. Neden ölçüldü: simetrik limit noktasından önce λ = 0,9972 λ_max'ta
  (w = 2,30) **antisimetrik çatallanma**; oradaki en küçük özdeğerin vektörü %100 antisimetrik (simetrik kısım
  1,3e-12). Yük kontrolü ve yay boyu simetrik birincil yolda kalıyor; çatallanmadan sonraki noktalar kararsız
  denge. Dal değiştirme yok ve garanti edilmiyor (modül notları); en küçük özdeğer izleme ya da çatallanma tespiti
  sonraki iş.

### E

- **E1** sabit kesme (c = 0,1 ve 0,3), 8 bileşen merkezi FD: son uyum 1,2e-9 / 2,0e-9; `EndStrainEnergy` 8,6e-9 /
  2,5e-8 (eşik 1e-6); max|u|/açıklık 0,76 / 0,66.
- **E2** (ölçüm; 24 × 8 neo-Hooke, P = 0,01, t ∈ [0,9, 1], 21 nokta): uyarlamalı kesme t boyunca **monoton değil**:
  0,1 (t ≤ 0,935) → 0,92 → 0,96 → 0,96 → 0,92 → 0,1 (t = 0,96) → 0,96 → 0,98 → 0,98 → 1,0. 21 noktada 7 model
  değişimi (`model_epoch` 7). Aynı tasarımda iki kesmenin J farkı (model sıçraması): t = 0,935'te %17,8
  (c = 0,1 → 0,92), 0,94'te %6,6, 0,96'da %7,9, 0,965'te %0,29, 0,975'te %2,2. İki değişimde (t = 0,95 ve 0,955)
  yüksek kesmeli çözüm aynı tasarımda YAKINSAMIYOR: "daha doğrusal" model her zaman daha kolay değil, kesme kuralı
  tasarıma göre sıçrıyor. Komşu noktalar arası J farkı / adjoint eğim tahmini: düzgün aralıklarda 0,68–1,05,
  değişim aralıklarında 0,08–1,22; bu çözünürlükte (Δt = 0,005) J'nin dikliği sıçramayı FD oranında gizliyor, model
  sıçraması ancak aynı tasarımda iki model karşılaştırılınca görünüyor.
- **E3** (ölçüm; 24 × 8, P = 0,02, MMA 30 iterasyon): üç seçenekte de kesme hiç değişmedi (0,1), üç koşu aynı
  (J = 0,14442, hacim 0,39997, 0 yeniden başlatma, 0 ret). **Karşılaştırma bilgi taşımıyor** (yük, ön kayıtta
  seçilirken değişim üretip üretmeyeceği denenmedi).
- **E3b (post hoc)**: P = 0,12 (kısa bir keşifte 8 iterasyonda kesme 0,1–0,26 arasında gidip geliyordu):
  30 MMA iterasyonu, aynı başlangıç (x = 0,4).

  | Seçenek | Kesme geçmişi | Model değişimi | Monotonluk ihlali | Son J (kendi modeli) | J, ortak c = 0,36 | Süre |
  |:---|:---|---:|---:|---:|---:|---:|
  | uyarlamalı (mevcut) | 0,1 → 0,26 → 0,16 → 0,24 → 0,14 → 0,1 (12 değişim; 13. iterasyondan sonra 0,1) | 12 | 0 | 2,157 | 2,153 | 200 s |
  | monoton | ilk değerlendirmede 0,36, sonra sabit | 0 | 4 | 2,272 | 2,272 | 70 s |
  | ret | monotonla aynı (taban hiç yükselmedi: ret dalı hiç çalışmadı) | 0 | 4 | 2,272 | 2,272 | 61 s |

  Okuma: (i) uyarlamalı koşunun amaç geçmişi düzgün görünüyor (ihlal 0) ama 12 farklı modelin değerlerinden oluşuyor;
  değişim noktalarında klasik türev yok ve MMA geçmişi farklı problemlerde kullanıldı. (ii) Monoton seçenek modeli ilk
  değerlendirmenin (gri tasarım) gerektirdiği 0,36'da sabitledi: tek model, 2,8 kat daha hızlı (yeniden başlatmalı
  çözüm yok), 4 küçük amaç artışı, ortak modelde %5,5 daha kötü son tasarım. (iii) Ret seçeneği bu kurulumda
  sınanmadı (NOT VERIFIED). Hiçbir seçenek bu tek örnekte baskın değil; karar için daha çok problem ve monoton
  tabanın (ilk gri tasarımın gerektirdiği kesme) aşırı doğrusallaştırma etkisinin ölçülmesi gerekir.
- **E4** (ölçüm): ρ̃ = E^(1/3) geri kazanımı, SIMP üssü yorum üssüne eşitken (p = 3) yalnız E_min tabanı kadar hatalı
  (ρ = 0'da ρ̃ = E_min^(1/3) = 1e-3 / 1e-2; kesme 0,1'de 3,3e-8 / 3,3e-5). Sürdürmede SIMP üssü 1 ya da 2 iken
  (yorum üssü 3 kalırsa) en büyük hata 0,385 (ρ = 0,19) / 0,148 (ρ = 0,30); kesme 0,1'de 0,364 / 0,115: γ yanlış
  yoğunlukta değerlendiriliyor. Fizik yoğunluğu değil modülü görüyor; gerçek ρ'nun rezidüye geçmesi `Study`
  arayüzünü gerektiriyor (perf / material-design dosyaları; sonuç notunda "gereken değişiklik").

### D (V4 neo-Hooke strict xfail'ının nedeni)

- **D1 FAIL (ikinci madde)**: son kalıntı her adımda tabanda (≤ 3,1e2 φ, 20 adımın hepsi; birinci madde tutuyor),
  ama bir öncekisi bazı adımlarda 1e4 φ'nin altında: SVK λ = 0,6 / 0,7'de 1,4e3 / 6,9e2 φ, neo-Hooke λ = 0,9 / 1,0'da
  4,6e3 / 1,7e3 φ.
- **D2** (ölçüm): V4'ün en büyük üç noktalı mertebesi ve paydası |log(r_k/r_{k−1})|: neo-Hooke λ = 0,6–1,0'da
  0,94 → 0,77, payda 6,3 → 9,2 (büyük düşüşten sonra tabana çarpan üçlü); SVK λ = 1,0'da 18,1, payda 0,36
  (r_k/r_{k−1} ≈ 0,70: paydası sıfıra yakın üçlü), λ = 0,9'da 10,3 (payda 0,56).
- **Sonuç**: V4'ün başarısızlığı Newton'un ya da tanjantın kusuru değil; ölçüt bu problemde kullanılamaz. Newton
  ~1e-1'den iki ikinci mertebe adımda yuvarlama tabanına iniyor; tabanın üstünde asimptotik bölgede üç kalıntı yok.
  neo-Hooke'un "başarısızlığı" tabanlı son kalıntıdan, SVK'nın "başarısı" paydası küçük üçlülerden geliyor.
  Yakınsama mertebesinin kanıtı hata daralması testi (V4 post hoc, geçiyor) ve J8 post hoc adım geçmişi. V4 strict
  xfail olarak kalıyor (ön kayıtlı başarısız kontrol); kütükteki gerekçe bu kayda bağlandı.

### T

- **T1** `EndStrainEnergy` Taylor mertebeleri: SVK 1,944, 1,967, 1,982, 1,991, 1,995, 1,998; neo-Hooke 1,923 …
  1,998 (iki ardışık [1,8, 2,2] içinde).
- **T2** ×1,01: son iki mertebe SVK 0,54, 0,81; neo-Hooke 0,67, 0,86 (≤ 1,3).
- **T3** = E1'in `EndStrainEnergy` satırları.

## 2. İlk koşuda bulunan hatalar (eşik değişmedi)

1. `solve_history` sıfır yüke boşaltmada ‖λF‖ = 0'a bölüyordu (`ZeroDivisionError`, J7–J9). Düzeltme: λ = 0'da
   bağıl kalıntı yolun yüküne göre (`_newton(..., ref=f)`); yük artışı yolunda bir şey değişmedi (P0 13/13).
2. N1 tekil tanjant kurgusunda başlangıç noktası kökün kendisiydi (x₀ = 1, R = x − 1; test girdisi). R = x − 2.
3. Malzeme noktası sürücüsü (`drive_material_point`) plastik adımdan sonraki boşaltmada ilk iterasyonu akma
   yüzeyindeki noktanın plastik tanjantıyla atıyordu; düzlem gerinimde ters akmaya sıçrayıp yakınsamıyordu (J7
   Q4). Düzeltme: her adımın ilk iterasyonu elastik tanjantla (`Ce6`). J1–J6 yeniden koşuldu, değerler aynı mertebede.
4. A6b sözleşme düzeltmesi değil, gerçek sonuç (yukarıda); N3 kurulumunda kalıntı merit'i yerine enerji merit'i
   (alan kontrolü aynı; bulgu ayrı post hoc testte).

## 3. Ön kayıttan sapmalar ve post hoc etiketler

- A2a/A2b ön kayıt tanım hatası → strict xfail + `test_a2_posthoc_local_limit_points`.
- J7 α maddesi (bağıl ölçü α ≈ 0'da tanımsız) → strict xfail + `test_j7_alpha_uniform_relative_to_the_path_maximum`.
- J8 (b) → strict xfail + `test_j8b_posthoc_per_step_iterations` (adım başına; süreklilik ≥ algoritmik).
- E3 bilgi taşımadı → E3b post hoc (P = 0,12).
- `test_residual_merit_stalls_where_full_newton_converges`, `test_stateful_model_refuses_autograd_interpolation_and_
  warm_start`, `test_protocol_refuses_wrong_kinematics_and_plane_stress_j2`, `test_return_mapping_failure_is_reported`:
  post hoc sözleşme testleri.

## 4. Değişen API ve uyumluluk

- Yeni `toporlogy.core.constitutive`: `Kinematics`, `MaterialRequest`, `MaterialResponse`, `StateStore`,
  `ConstitutiveFailure` / `InadmissibleState` / `ReturnMappingFailure`, `SaintVenantKirchhoff`, `NeoHooke`,
  `LinearElastic`, `J2Plasticity`, `neo_hooke_response` (fem.nonlinear'dan taşındı), `energy_oracle`,
  `svk_energy_torch`, `neo_hooke_energy_torch`, `drive_material_point`.
- Yeni `toporlogy.solvers.newton` (`newton`, `NewtonSettings`, `Linearization`, `StepRecord` — fem.nonlinear'dan
  taşındı ve genişledi: `code`, `updates`, `step_lengths`, `min_det_f`, `energy`, `dissipation`, `rolled_back`,
  `notes`), `toporlogy.solvers.continuation` (`load_control`, `arc_length`, `displacement_control`,
  `IndefiniteDirectSolver`, `ContinuationError`).
- `fem.nonlinear.NonlinearElasticity`: `material=` bir model nesnesi de alır (küçük şekil değiştirme, J2);
  `line_search`, `merit`, `log_energy`, `tangent_kind`; `solve_history`, `continuation_problem`, `min_det_f`,
  `element_energies`, `model_epoch`, `model_log`, `reset_model`, `last_states`; `EnergyInterpolation(policy=
  "adaptive" | "monotone")`. `NonlinearConvergenceError.code` "nonlinear_convergence" → "step_limit" (+ `cause`,
  `load_factor`); mesaj metni aynı. Kırıcı: `fem.nonlinear.StepRecord` → `solvers.newton.StepRecord`,
  `fem.nonlinear.neo_hooke_response` → `core.constitutive` (takma ad yok; kullanımlar güncellendi).
- `optimization.nonlinear.EndStrainEnergy` (yeni yanıt; YAML kaydı yok).
- Uyumluluk: J2 / küçük şekil değiştirme modelleri yalnız API; `solve` (autograd), enerji interpolasyonu ve warm start
  durumlu modelde reddediliyor; J2 düzlem gerilme reddediliyor. Yol bağımlı duyarlılık yok.

## 4b. API örneği (çalıştırılabilir; `PYTHONPATH=src` ile)

```python
import numpy as np
from toporlogy.core.constitutive import J2Plasticity
from toporlogy.core.domain import Domain
from toporlogy.fem.nonlinear import NonlinearElasticity
from toporlogy.solvers.continuation import ArcLengthSettings, arc_length

d = Domain((20, 5), h=1.0)
X = d.node_coords()
fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
F = np.zeros(d.ndof)
tip = np.flatnonzero(np.abs(X[:, 0] - 20) < 1e-9)
F[2 * tip + 1] = -3.3e-4 / len(tip)

# J2 plastisite, düzlem gerinim: yükle, tam boşalt; durum her yakınsamış artışta commit edilir
nl = NonlinearElasticity(d, fixed, material=J2Plasticity(0.3, 1e-3, 0.05), plane="strain")
U, report, checkpoint = nl.solve_history(np.ones(nl.n_elements), F, [1.0, 0.0])
print(np.abs(U[:, 1]).max(), [s.dissipation for s in report.steps][:3])

# hiperelastik problemde yay boyu (limit noktası ötesi): continuation_problem + arc_length
nh = NonlinearElasticity(d, fixed, material="neo_hooke")
prob = nh.continuation_problem(np.ones(nh.n_elements), 100 * F)
path = arc_length(prob, np.zeros(len(nh.free)), 0.0, ArcLengthSettings(ds=0.5), lambda x, lam: lam >= 1.0)
print(path.lam[-1], path.rejected)
```

## 5. Ertelenen

Yol bağımlı adjoint (J2 tasarım gradyanı), sonlu şekil değiştirme düzlem gerilme genel anizotropi, neredeyse
sıkıştırılamaz neo-Hooke (03 karma eleman), kinematik pekleşme, sonlu şekil değiştirme plastisitesi, izleyen yük,
çatallanma tespiti / dal değiştirme, gerçek yoğunluğun rezidüye geçmesi (`Study` arayüzü), YAML: `cutoff_policy`
ve `end_strain_energy` kaydı (mevcut components dosyaları bu oturumun değil).

## 6. Paketler

- `pytest tests/toporlogy -m "not slow" -q -p no:randomly`: 1208 passed, 8 skipped, 27 xfailed, 2 failed (48 dk; aynı
  anda E3b koşuyordu). İki başarısızlık kütük testleriydi (yeni bir `pytest.skip` kütükte yoktu; risk matrisi yeniden
  üretilmemişti): skip parametrizasyondan çıkarıldı (4 atlama gitti), matris üretildi; düzeltme sonrası
  `test_catalog_ledger`, `test_catalog`, `test_api_surface`, `test_architecture`, `test_constitutive`: 68 passed.
  Taban 1149 passed / 22 xfailed; +5 hızlı strict xfail (A2, J7 α × 2, J8b, D1), +1 yavaş (A6b).
- `pytest -m baseline`: 8 passed, bütün örneklerde en büyük bağıl fark 0.
- `ruff check src tests`: temiz; `python -m mypy`: temiz (116 dosya).
- Yavaş: A6 geçti, A6b strict xfail (ayrıca koşuldu).

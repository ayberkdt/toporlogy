# Geometrik doğrusal olmama (büyük yer değiştirme) — doğrulama kaydı

Dal: `claude/geometric-nonlinear-9712d5` (koordinasyon notundaki adı `feat/geometric-nonlinear`). Kod:
`src/toporlogy/fem/nonlinear.py`, `src/toporlogy/optimization/nonlinear.py`. Testler:
`tests/toporlogy/test_nonlinear.py`, bağımsız elastika referansı `tests/toporlogy/_elastica.py`.

Etiketler `literatur_2026-11-21.md` ile aynı: **[Ö]** ölçüldü, **[O]** kaynaktan / koddan okundu, **[H]** hafızadan,
doğrulanmadı.

## 0. Ön kayıt — 2026-11-25T17:58:30+03:00 (yeni kodun hiçbir koşusundan önce)

Bu bölüm, `fem/nonlinear.py` yazılmadan ve hiç koşturulmadan önce yazıldı. Sonradan eklenen her ölçüt ayrı
bölümde "post hoc" etiketiyle yer alır. Tutmayan bir ön kayıt test dosyasında `xfail(strict=True)` olur ve
gerekçesi docstring'de ve bu kayıtta yazılır; tolerans gevşetilmez.

### 0.1 Toleransları belirlemek için önceden yapılan ölçümler (yeni kod değil)

**Doğrusal Q4 kilitlenmesi [Ö].** Mevcut, doğrulanmış `Elasticity` (Q4, ν = 0, kare eleman) ile L/H = 50 konsol,
uçta düzgün yayılı (trapez ağırlıklı) düşey yük, kökte bütün serbestlikler tutulu. Uç sehiminin Timoshenko kirişine
oranı (`docs/olcumler/geometrik_dogrusal_olmama.py`, `linear_locking()`; ön kayıttan önce aynı kod bir karalama betiğiyle koşturuldu):

| kalınlıkta eleman n | eleman | v / v_Timoshenko |
|---:|---:|---:|
| 2 | 200 | 0,8889 |
| 4 | 800 | 0,9697 |
| 8 | 3 200 | 0,9922 |
| 16 | 12 800 | 0,9980 |

Hata yaklaşık 0,5 / n² (n = 16'da %0,20). Doğrusal olmayan konsolda bükülme aynı elemanla taşındığı için aynı
mertebede kilitlenme hatası beklenir.

**Elastika referansı [Ö].** `tests/toporlogy/_elastica.py`: aynı denklemin (θ'' = −α cos θ, α = P L²/EI) iki
bağımsız çözümü — birinci integralden kapalı biçimli x_L ve Gauss–Jacobi karesel toplamı; DOP853 ile atış. Fark
(u/L, v/L, θ_L üzerinden en büyük mutlak fark): α = 0,1: 1,1e-16; 1: 5,0e-16; 2: 1,6e-15; 5: 2,7e-15; 10: 1,3e-14.

| α = PL²/EI | u/L (kısalma) | v/L (sehim) | θ_L (rad) |
|---:|---:|---:|---:|
| 1 | 0,0564332363 | 0,3017207738 | 0,4613519497 |
| 2 | 0,1606417208 | 0,4934574804 | 0,7817498316 |

Mattiasson (1981) tablosu okunmadı [H]; referans olarak yalnız bu iki bağımsız çözüm kullanılır.

**Düşük yoğunluk çaresinin kaynağı [O].** Wang, Lazarov, Sigmund & Jensen (2014, CMAME 276:453-472) makalesinin
aslı okunamadı (erişim yok) [H]. Şemanın biçimi ikincil kaynaktan okundu: Feng, Zhang & Khandelwal, *Finite Strain
Robust Topology Optimization Considering Multiple Uncertainties*, arXiv:2501.15337, Denk. (11)–(14), s. 10–11:

- iç kuvvet F_int^e = ∫ γ B:P(F(γ)) dV + ∫ (1 − γ²) B_L:ℂ:ε dV, F(γ) = I + γ ∇_X u^e;
- γ(ρ) = exp(β0 ρ) / (exp(c β0) + exp(β0 ρ));
- tanjant K_T^e = ∫ γ² B:𝔸:B dV + ∫ (1 − γ²) B_L:ℂ:B_L dV;
- β0 = 120; c, c0 = 0,1'den başlayıp Newton en küçük adımda yakınsamazsa Δc = 0,02 artırılır (Zhang, Alberdi &
  Khandelwal 2018'e atıfla; o makale de okunmadı [H]); her tasarım değerlendirmesinde c = c0'a döner.

Bu iç kuvvet, eleman enerjisi Φ(u) = ψ_NL(γu) − ψ_L(γu) + ψ_L(u)'nun u'ya göre türevidir (ψ_L(γu) = γ² ψ_L(u)
olduğundan) — bunu kendim türettim; Wang ve ark. 2014'ün enerjiyi bu biçimde yazdığı [H].
Neo-Hooke biçimi de aynı kaynaktan okundu (Denk. 9–10): ψ = ½κ(J − 1)² + ½μ(Ī₁ − 3), Ī₁ = tr(J^(−2/3) C),
κ = E/(3(1 − 2ν)), μ = E/(2(1 + ν)).

**Sapma (bilerek):** fizik nesnesi yoğunluğu değil yalnız eleman modülünü E_e görür (`Study` arayüzü). γ bu yüzden
ρ̃ = (E_e / E0)^(1/p) üzerinden hesaplanır (p varsayılan 3; SIMP'te E_min = 0 ve frac = 1 iken ρ̃ = ρ tam).

### 0.2 Ön kayıtlı ölçütler

Newton yakınsama ölçütü (kodda da aynı): her yük adımında ‖R_serbest‖₂ ≤ rtol · ‖λF_serbest‖₂ ya da geri hata
η = ‖R‖ / (‖Σ_e |f_e|‖ + ‖λF‖) ≤ 1e-13. Aşağıdaki doğrulamalar rtol = 1e-12 ile koşar (aksi yazılmadıkça).

| # | Ölçüt | Eşik |
|:---|:---|:---|
| V1 | Küçük yük (en büyük |∇u| ≤ 1e-8 olacak ölçekte) — ‖u_NL − u_L‖ / ‖u_L‖, `Elasticity` ile; SVK ve neo-Hooke; Q4 düzlem gerilme, Q4 düzlem gerinim, H8; enerji interpolasyonu açık ve gri tasarımda | ≤ 1e-6 |
| V1b | Aynı küçük yükte uyum gradyanı dC/dE_e, `Elasticity` ile (aynı tasarım) | bağıl ≤ 1e-6 |
| V1c | u = 0'da tanjant = geom · `element_matrix_from_D` (SVK ve neo-Hooke, Q4 ve H8) | bağıl ≤ 1e-12 |
| V2 | Homojen tek eksenli çekme (ölü yük P = 0,3 E, ν = 0,3), boyuna ve enine germe oranı; kapalı biçim: SVK S₁ = E·E₁ (düzlem gerinimde E/(1−ν²)), enine E₂ = −ν E₁ (düzlem gerinimde −ν/(1−ν) E₁); neo-Hooke S₂ = 0'ın skaler kökü; Q4 düzlem gerilme / gerinim, H8; iki malzeme | mutlak ≤ 1e-9 |
| V3 | Tutarlı tanjant: ‖K δu − (R(u+εδu) − R(u−εδu))/(2ε)‖ / ‖K δu‖, büyük şekil değiştirmiş rastgele durumda; altı yapılandırma + enerji interpolasyonu açık gri tasarım | ≤ 1e-6 |
| V3b | Tasarım türevi ∂R/∂E_e (γ'nın E'ye bağlılığı dahil), merkezi sonlu fark | bağıl ≤ 1e-6 |
| V4 | Newton'da ikinci mertebe: her yük adımında, kalıntı 1e-11'in üstündeyken ardışık üçlülerden p = log(r_{k+1}/r_k)/log(r_k/r_{k−1}); en az bir üçlüsü olan adımlarda en büyük p (adım başına) | ≥ 1,8 |
| V5a | Büyük sehimli konsol (L/H = 50, ν = 0, SVK ve neo-Hooke düzlem gerilme, α = 1 ve 2), n = 16: v hatası / u hatası (elastikaya bağıl) | ≤ %0,5 / ≤ %1,0 |
| V5b | n = 8 ve 16'dan Richardson (p = 2) ekstrapolasyonu: v / u | ≤ %0,1 / ≤ %0,2 |
| V5c | Hata n = 4 → 8 → 16 boyunca azalır (v ve u) | monoton |
| V6 | Son uyum gradyanı, sonlu fark (filtre + SIMP + doğrusal olmayan çözüm zinciri; rastgele gri tasarım; en büyük yer değiştirme ≥ açıklığın %5'i; merkezi fark, adım 1e-5): max|g_AD − g_FD| / max|g_FD| — SVK ve neo-Hooke 2-B, SVK 3-B; ayrıca `Displacement` yanıtı (genel adjoint) | ≤ 1e-5 |
| V7a | Kıyas problem (iki ucu ankastre kiriş, ortada nokta yük; yük: doğrusal tasarımın doğrusal orta sehimi açıklığın %10'u olacak biçimde): C_NL(x_NL) ≤ 0,99 · C_NL(x_L) | sağlanmalı |
| V7b | Aynı problemde C_L(x_L) ≤ 1,005 · C_L(x_NL) | sağlanmalı |
| V7c | Buhl, Pedersen & Sigmund (2000) sayılarıyla karşılaştırma | kaynak okunamadı: NOT VERIFIED |

Gerekçeler: V1 — doğrusal olmayan katkı |∇u| mertebesinde (1e-8), eşik iki basamak payla. V2 — Newton rtol 1e-12
ve homojen alan Q4/H8 ile tam temsil edilir; kalan hata yalnız yuvarlama. V3 — merkezi farkın kesme hatası
O(ε²) ≈ 1e-12, yuvarlama ≈ 1e-16/ε; ε = 1e-6 ile her ikisi 1e-9 altında. V5a — n = 16'da doğrusal kilitlenme %0,20;
SVK ile neo-Hooke'un elastikadan farkı en büyük birim şekil değiştirme ε = αH/(2L) (α = 2'de %2) için O(ε²);
kesme şekil değiştirmesi (E/4kG)(H/L)² = %0,024; u ≈ v²/… olduğundan hatası iki katı alındı. V5b — 1/n² davranışı
doğrusal ölçümde görüldü. V6 — Newton rtol 1e-12 ile çözüm gürültüsü 1e-12, merkezi fark kesme hatası 1e-10.
V7 — tasarımlar yerel optimum; yine de doğrusal olmayan tasarımın kendi modelinde en az %1 iyi olması beklenir.

Kapsam dışı (bu işte yapılmadı): post-buckling ve yay uzunluğu (arc-length) — yükleme yalnız yük kontrollü, limit
noktası geçilmez; plastisite; temas; GMG çözücü; sıfırdan farklı zorlanmış yer değiştirme; takipçi (follower) yük;
H8IM / Q4IM'in doğrusal olmayan sürümü; ortotrop neo-Hooke (SVK herhangi bir D ile çalışır).

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

## 1. Sonuç özeti [Ö]

Kaynak: `python -m pytest tests/toporlogy/test_nonlinear.py -m "not slow" -s -v -p no:randomly` (25 Kasım 2026,
düzeltmelerden sonraki son koşu) ve `PYTHONPATH=src python docs/olcumler/geometrik_dogrusal_olmama.py`
(`geometrik_dogrusal_olmama.json`). Her satırda altı yapılandırmanın (SVK / neo-Hooke × Q4 düzlem gerilme, Q4
düzlem gerinim, H8) en kötüsü verilir.

| # | Ölçüt | Eşik | Ölçülen (en kötü) | Karar |
|:---|:---|---:|---:|:---|
| V1 | küçük yük, ‖u_NL − u_L‖/‖u_L‖ | 1e-6 | 1,4e-9 | PASS |
| V1b | küçük yük, uyum gradyanı | 1e-6 | 2,6e-9 | PASS |
| V1c | u = 0'da tanjant = geom · `element_matrix_from_D` | 1e-12 | 3,5e-16 | PASS |
| V2 | tek eksenli çekme, germe oranları (mutlak) | 1e-9 | 2,2e-14 | PASS |
| V3 | tutarlı tanjant (merkezi fark) | 1e-6 | 1,9e-10 | PASS |
| V3b | ∂f_e/∂E_e (merkezi fark) | 1e-6 | 8,5e-10 | PASS |
| V4 | Newton, adım başına en büyük üç noktalı mertebe | ≥ 1,8 | SVK 2,3; neo-Hooke 0,77 | SVK PASS; neo-Hooke FAIL (strict xfail) |
| V5a | konsol n = 16, v / u hatası | %0,5 / %1,0 | %0,13 / %0,31 | PASS |
| V5b | Richardson (8, 16), v / u | %0,1 / %0,2 | %0,040 / %0,067 | PASS |
| V5c | hata n = 4 → 8 → 16 azalır | monoton | dört durumda da monoton | PASS |
| V6 | son uyum ve yer değiştirme gradyanı (sonlu fark) | 1e-5 | 1,2e-9 | PASS |
| V7a | C_NL(x_NL) / C_NL(x_L) | ≤ 0,99 | 0,259 | PASS |
| V7b | C_L(x_L) / C_L(x_NL) | ≤ 1,005 | 0,916 | PASS |
| V7c | Buhl, Pedersen & Sigmund (2000) sayıları | — | kaynak okunamadı | NOT VERIFIED |

Ayrıntılar (ölçülen değerler, yapılandırma başına; sıra SVK ps, SVK pe, SVK 3-B, NH ps, NH pe, NH 3-B):

- V1: 9,6e-10; 9,5e-10; 1,4e-9; 7,2e-10; 7,3e-10; 1,1e-9. V1b: 6,5e-10; 6,0e-10; 1,1e-9; 1,7e-9; 1,3e-9; 2,6e-9.
- V2 (P = 0,3 E, ν = 0,3): SVK λ₁ = 1,221196686181 (düzlem gerilme ve 3-B), 1,205390135714 (düzlem gerinim);
  neo-Hooke λ₁ = 1,422861641171 (düzlem gerilme ve 3-B aynı, beklendiği gibi), 1,386363266269 (düzlem gerinim).
  Hatalar ≤ 2,2e-14; serbest yüz düğümlerinin yer değiştirme farkı ≤ 2,2e-16 (homojen).
- V3: 1,8e-10; 1,8e-10; 9,7e-11; 1,6e-10; 1,9e-10; 1,5e-10. V3b: 7,8e-10; 7,0e-10; 1,6e-10; 7,1e-10; 8,5e-10;
  2,5e-10 (0,6 rad dönme, %20'ye varan germe ve gürültü; modüller enerji interpolasyonunun geçiş bölgesini kapsayacak
  biçimde 2e-4 ile 1 arasında).
- V6 (en büyük yer değiştirme açıklığın %34–43'ü): SVK 2-B 1,2e-9 / 1,2e-9 (son uyum / uç yer değiştirmesi),
  neo-Hooke 2-B 1,1e-9 / 8,8e-10, SVK 3-B 1,2e-10 / 1,0e-10.

Konsol (V5; L/H = 50, ν = 0, düzlem gerilme, uçta ölü düşey yük; elastikaya bağıl hata):

| Malzeme | α | n | v hatası (%) | u hatası (%) | Newton it. |
|:---|---:|---:|---:|---:|---:|
| SVK | 1 | 4 | −2,49 | −5,08 | 49 |
| SVK | 1 | 8 | −0,61 | −1,26 | 46 |
| SVK | 1 | 16 | −0,13 | −0,28 | 40 |
| SVK | 2 | 4 | −1,75 | −3,83 | 55 |
| SVK | 2 | 8 | −0,41 | −0,93 | 53 |
| SVK | 2 | 16 | −0,08 | −0,18 | 50 |
| neo-Hooke | 1 | 4 | −2,49 | −5,12 | 40 |
| neo-Hooke | 1 | 8 | −0,61 | −1,30 | 40 |
| neo-Hooke | 1 | 16 | −0,13 | −0,31 | 40 |
| neo-Hooke | 2 | 4 | −1,74 | −3,87 | 50 |
| neo-Hooke | 2 | 8 | −0,41 | −0,96 | 50 |
| neo-Hooke | 2 | 16 | −0,07 | −0,21 | 48 |

Hata, n her ikilendiğinde yaklaşık dörtte birine iniyor (1/n²; doğrusal Q4 kilitlenmesiyle aynı davranış: n = 4'te
doğrusal %−3,0, doğrusal olmayan v %−2,5). Richardson ekstrapolasyonu (p = 2): SVK α = 1 v %+0,028, u %+0,054;
α = 2 v %+0,036, u %+0,067; neo-Hooke α = 1 v %+0,030, u %+0,018; α = 2 v %+0,040, u %+0,034. Her durumda 10 yük
adımı, adım küçültme yok. Süre: n = 16 koşuları 30–94 s, bölüm toplam 355 s.

## 2. İlk koşunun ve düzeltmelerin kaydı

Ön kayıttan sonraki ilk koşu (`pytest ... -m "not slow"`, rastgele sıra) 900 s'lik pytest zaman aşımına takıldı;
özet yazılmadı. Sıralı koşuda nedenler tek tek bulundu:

1. **Kod hatası: Green şekil değiştirmesi ½(FᵀF − I) ile hesaplanıyordu.** |∇u| = 1e-8'de iç kuvvette 1,4e-6 bağıl
   yuvarlama hatası (3-B). Newton hiçbir yakınsama ölçütünü geçemedi; uyarlamalı kesme her yeniden başlatmada en
   küçük adıma kadar yarıladı, test asılı göründü. Düzeltme: E = ½(H + Hᵀ + HᵀH), J − 1 H'nin değişmezlerinden,
   neo-Hooke'ta S = κJ(J − 1)C⁻¹ + 2μJ^(−2/3) C⁻¹ dev E (1'in çıkarıldığı terim yok). Aynı durumda Newton artık 2
   iterasyonda 1,4e-14'e iniyor. V1'in 2-B satırları ilk koşuda da geçmişti (≤ 1,8e-9).
2. **Ön kayıttan sapma (yakınsama ölçütü): η'nın tanımı değişti.** Ön kayıtta η = ‖R‖ / (‖Σ_e |f_e|‖ + ‖λF‖)
   yazıyordu. İnce konsolda (L/H = 50) bağıl kalıntı 2,7e-9'da yuvarlama tabanına oturdu (‖K‖‖u‖ ≫ ‖F‖; eleman
   kuvvetlerinin büyüklüğü bu tabanı göstermiyor), V4 ve V5 yakınsamadı. η, doğrusal çözücünün `checked_residual`
   tanımına getirildi: η = ‖R‖ / (‖K_T‖₁ ‖u‖ + ‖λF‖); eşik 1e-13 değişmedi. Bunun bedeli: ölçüt ileri hatayı ancak
   κ(K_T) · 1e-13 ile sınırlar (doğrudan doğrusal çözümle aynı tür sınır).
3. **Test girdisi hatası (V3, neo-Hooke):** rastgele şekil değiştirmiş durum, 0,6 rad dönmeden sonra ankastre
   serbestlikleri sıfırladığı için ankastre yanındaki elemanları ters çeviriyordu (J en az −0,81; neo-Hooke'un tanım
   kümesi dışında, `det F <= 0`). Taban durumunun mesnetleri sağlaması gerekmez (yalnız δu serbest serbestliklerde
   kalır); sıfırlama kaldırıldı, J en az 0,72. Eşik değişmedi.
4. **V4, neo-Hooke: ön kayıt tutmadı → strict xfail.** Ölçüt (adım başına, 1e-11 üstündeki ardışık üç kalıntıdan en
   büyük mertebe ≥ 1,8) SVK'da her adımda sağlanıyor (2,3–18); neo-Hooke'ta λ = 0,1'de 1,79, λ = 0,6–1,0'da
   0,77–0,94. Ölçütün kendisi bu problem için uygun değil: ince kirişte her adımın ilk kalıntısı ‖λF‖'nin 1e2–5e3
   katına sıçrıyor ve üç kalıntı asimptotik bölgeden önceki evreyi kapsıyor. Kanıt olarak **post hoc** test:
   yakınsamış 0,9 F durumundan tam yükte Newton, iyi koşullu konsolda (16 × 6, cond(K_T) ≈ 1,2e4), hata
   e_k = |u_k − u*|∞/|u*|∞: SVK 1,0e-1 → 6,3e-4 → 1,2e-6 → 2,7e-16; neo-Hooke 1,0e-1 → 5,5e-4 → 1,1e-6 → 4,0e-16;
   C_k = e_{k+1}/e_k² ≤ 3,7 (post hoc eşik ≤ 10, en az iki iterasyon, üç basamak). Doğrusal yakınsayan bir yöntemde
   C_k = r/e_k sınırsız büyür. V3 (tanjant ≤ 1,9e-10) ile birlikte ikinci mertebe yakınsamanın kanıtı budur.
5. **Sağlamlık ve hız (ölçütleri değiştirmeyen):** Newton'da durgunluk (5 iterasyonda kalıntı yarıya inmiyor)
   başarısız adım sayılıyor; uyarlamalı kesme, hiçbir elemanın γ'sını 1e-3'ten fazla değiştirmeyen değerleri
   atlıyor (aynı başarısız çözümü tekrarlamamak için); eleman çekirdeğinde `einsum` yerine toplu `matmul`, neo-Hooke
   tanjantı doğrudan Voigt biçiminde. V6'da sonlu fark bileşen sayısı 8'den 6'ya indi (süre).

## 3. Uygulama notları [O: koddan]

- Total Lagrangian, Q4 (2-B) ve H8 (3-B), tam 2 × 2(× 2) Gauss; `Elasticity` ile aynı ağ, düğüm ve serbestlik sırası,
  aynı `E_active`. Küçük yer değiştirmede tanjant `Elasticity`'nin eleman matrisine 3,5e-16 ile eşit (V1c).
- SVK herhangi bir birim modüllü D ile (ortotrop ve düzlem gerinim dahil); neo-Hooke izotrop, düzlem gerilmede C₃₃
  her Gauss noktasında S₃₃ = 0'dan çözülür ve tanjant yoğunlaştırılır.
- Newton–Raphson: eşit yük adımları (varsayılan 10), başarısız adım yarılanır (en küçük `min_step` = 1e-4), iki
  başarılı adımdan sonra 1,5 kat büyür. Yakınsamazsa `NonlinearConvergenceError` (ulaşılan yük katsayısı ve kalıntı
  geçmişiyle); sessiz kısmi sonuç yok. Her yük durumu ayrı çözülür (süperpozisyon yok); `last_reports` her durumun
  adım, iterasyon, kalıntı, η, kesme ve yeniden başlatma kaydını tutar.
- Duyarlılık: geri yayılımda yakınsamış durumda tanjant kurulur ve yük durumu başına TEK adjoint çözülür,
  K_T λ = ∂J/∂U; ∂J/∂E_e = −λ_eᵀ ∂f_e/∂E_e (γ'nın E'ye bağlılığı dahil). `Study` değişmedi: fizik `solve(E, F)`
  arayüzünü sağlıyor; U'ya bağlı her yanıt (`compliance`, `displacement`, `end_compliance`) çalışıyor.
- YAML: `physics: {type: nonlinear_elasticity, material: svk | neo_hooke, energy_interpolation: true, ...}`,
  `objective: end_compliance` (doğrusal fizikle `ConfigError`).

## 4. Dokunulan paylaşılan dosyalar (koordinasyon notu)

Yalnız eklemeli: `configuration/components/physics.py` (yeni `nonlinear_elasticity` kaydı),
`configuration/components/responses.py` (yeni `end_compliance` kaydı), `configuration/schema.py` (docstring'de iki
satır), `references/_structures.py` (üç kaynak). `fem/*`, `design/*`, `optimization/study.py`,
`optimization/responses.py` değişmedi; `EndCompliance` yalnız `Compliance`'ın alt sınıfı. Paket `__init__.py`
dosyalarına ekleme yapılmadı (içe aktarma `toporlogy.fem.nonlinear` ve `toporlogy.optimization.nonlinear` üzerinden).
`tests/toporlogy/baseline/*` yerelde yeniden üretildi ve testten sonra geri alındı; birleştirmede `list.txt`'ye
`nonlinear_elasticity` ve `end_compliance` satırları, `api_surface.json`'a iki yeni modül girecek.

## 5. Kıyas: doğrusal ve doğrusal olmayan tasarım (V7) [Ö]

Kaynak: `PYTHONPATH=src python docs/olcumler/geometrik_dogrusal_olmama.py benchmark` (25 Kasım 2026, 489 s;
`geometrik_dogrusal_olmama.json` → `benchmark`). Problem (`BENCH`): iki ucu ankastre kiriş 96 × 16 eleman (h = 1),
açıklık ortasında üst kenarda üç düğüme yayılmış düşey yük, hacim ≤ 0,4, yoğunluk filtresi r = 3 eleman, SIMP p = 3,
E_min = 1e-6 E0, MMA (move 0,2), 150 iterasyon, ν = 0,3, neo-Hooke düzlem gerilme, enerji interpolasyonu açık. Yük:
önce doğrusal tasarım P = 1 ile bulunur (orta sehim 64,34); P = 0,1 · 96 / 64,34 = 0,1492 ile doğrusal orta sehim
açıklığın %10'u olur. Doğrusal olmayan tasarım aynı P ile `end_compliance` en küçüklenerek bulunur.

| Tasarım | C_NL | C_L | orta sehim, doğrusal olmayan | orta sehim, doğrusal | gri ölçüsü |
|:---|---:|---:|---:|---:|---:|
| x_L (doğrusal) | 4,12 | 1,43 | 27,6 | 9,6 | 0,33 |
| x_NL (doğrusal olmayan) | 1,07 | 1,56 | 7,2 | 10,5 | 0,32 |

Doğrusal olmayan tasarımın son uyumu, doğrusal tasarımınkinin dörtte biri (V7a: 0,259 ≤ 0,99, PASS); doğrusal
modelde ise doğrusal tasarım %8 daha iyi (V7b: 0,916 ≤ 1,005, PASS). Doğrusal tasarım büyük yükte doğrusal tahminin
2,9 katı sehim yapıyor (27,6'ya karşı 9,6; açıklığın %29'u): doğrusal modelin görmediği bir yumuşama. Her iki
koşuda hacim kısıtı sağlanıyor (0,39999…).

Sınırlamalar: iki koşu da 150 iterasyonda durdu (`max_iter`; son tasarım değişimi 0,015 > tol 0,01; doğrusal
olmayan amaç son 10 iterasyonda 1,0668'de sabit). Tasarımlar gri (gri ölçüsü ortalama 4x(1 − x) = 0,32–0,33; izdüşüm
yok), karşılaştırma bu gri tasarımlar arasında. İlk iterasyonda (düzgün x = 0,4) Newton enerji interpolasyonunun
kesmesini 0,10'dan 0,40'a çıkararak yakınsadı (3 yeniden başlatma, 30 adım küçültme); son tasarımların
değerlendirilmesinde kesme 0,10'da kaldı. Malzeme neo-Hooke seçildi: SVK ile düzgün gri başlangıç tasarımı (karalama koşusu, 48 × 8 ağ, P = 0,05)
λ ≈ 0,51'de durdu; en olası neden SVK'nın güçlü basmada kararlılığını yitirmesi (doğrulanmadı). Ön kayıt malzemeyi
belirtmiyordu.

Buhl, Pedersen & Sigmund (2000) ile sayısal karşılaştırma yapılmadı: makale okunamadı; problem onların problemi
değil, yalnız aynı türden (iki ucu ankastre kiriş, ortada yük). **NOT VERIFIED.** Onların doğrusal ve doğrusal
olmayan tasarımların farklılaştığını gösterdiği bilgisi hafızadan [H].

Test yolu: `pytest tests/toporlogy/test_nonlinear.py -m slow` (V5 dört durum + V7) 5 passed, 716 s (V7 tek başına
584 s; aynı `benchmark()` işlevi, aynı sayılar).

Paket düzeyinde (25 Kasım 2026): `pytest -m "not slow" tests/toporlogy` 688 passed, 4 skipped, 4 xfailed (üçü bu
daldan önce vardı); `pytest -m baseline` 8 passed, bit düzeyi fark 0; `ruff check src tests` ve `mypy` temiz.

## 6. Kapsam dışı ve açık işler

- Yay uzunluğu (arc-length) ve limit noktası ötesi (snap-through, post-buckling): yük kontrolü limit noktasında
  `NonlinearConvergenceError` verir.
- Plastisite, temas, takipçi yük, sıfırdan farklı zorlanmış yer değiştirme.
- GMG / yinelemeli çözücü: her Newton iterasyonu doğrudan çözücüyle faktörleşir (2-B 1 536 eleman: tasarım
  iterasyonu başına 2–9 s).
- H8IM / Q4IM'in doğrusal olmayan sürümü (Q4 eğilmede kilitlenir: L/H = 50 konsolda kalınlıkta 4 eleman ile %2,5).
- Ortotrop neo-Hooke; doğrusal olmayan gerilme ve burkulma yanıtları (yalnız U'ya bağlı yanıtlar çalışır).
- Wang ve ark. (2014) ve Zhang ve ark. (2018) asıl makaleleri okunmadı; şema ikincil kaynaktan [O: Feng ve ark.].

## 7. README için önerilen metin

"Not provided" satırından `nonlinear analysis` çıkar; "Buckling" satırındaki "(no geometric non-linearity, …)"
ifadesi doğrusal burkulma için doğru kalır. Önerilen yeni madde (Limitations listesine):

> - **Geometric non-linearity:** large-displacement static analysis (`physics: nonlinear_elasticity`; Total
>   Lagrangian, Q4 plane stress/strain and H8, St. Venant–Kirchhoff with any material D or compressible
>   neo-Hooke) and end-compliance design (`objective: end_compliance`). Load control only: no arc-length, so a
>   limit point (snap-through) ends the solve with `NonlinearConvergenceError`; dead loads, homogeneous supports
>   along x / y / z, one support set; direct solver only. Low-density elements use the energy interpolation of
>   Wang et al. (2014) as written by Feng, Zhang & Khandelwal (2025) (original paper not read); with its adaptive
>   cutoff the objective can jump between designs. Verified against closed-form uniaxial tension (2e-14), the
>   linear solver at small load (1.4e-9), the elastica of a cantilever (n = 16 through the thickness: deflection
>   −0.13 %, Richardson 0.04 %) and finite differences (tangent 1.9e-10, design gradient 1.2e-9); one
>   pre-registered check (three-point Newton order, neo-Hooke) failed and is recorded
>   ([`docs/olcumler/geometrik_dogrusal_olmama.md`](docs/olcumler/geometrik_dogrusal_olmama.md)). No plasticity,
>   contact, follower loads, incompatible-mode elements or non-linear stress/buckling responses.

## 8. Birleştirme sonrası: eleman başına D ile SVK (koordinatör)

A dalı (elemana göre malzeme yönü) ile birleşince `nonlinear_elasticity` eleman başına bir D alanını
"elasticity must be 3 x 3" hatasıyla reddediyordu (yanlış sonuç yoktu). SVK yasası her elemanın kendi D'siyle
genişletildi; küçük yer değiştirme matrisi K0, `Elasticity` gibi farklı D başına bir kez kurulur.

Ön kayıt (2026-11-25 19:53 +0300, ilk koşudan önce; eşikler C'nin V1, V3, V6 eşikleri):

| # | Kontrol | Eşik |
|:---|:---|:---|
| N1 | Her yerde aynı D'li alan = tek D (iç kuvvet, tanjant, son uyum) | ≤ 1e-13 göreli |
| N2 | Küçük yük: değişken açı alanında ‖u_NL − u_L‖/‖u_L‖, u_L aynı alanla `Elasticity` | ≤ 1e-6 |
| N3 | Tutarlı tanjant, değişken açı alanı (merkezi fark) | ≤ 1e-6 |
| N4 | Son uyum gradyanı, değişken açı alanı (sonlu fark) | ≤ 1e-5 |
| N5 | YAML: `angle_field` + `nonlinear_elasticity` kuruluyor, değerlendirme sonlu | geçer |

İlk koşu (`python -m pytest tests/toporlogy/test_nonlinear_orientation.py -s`, 2026-11-25) [Ö]: N1 kuvvet 3,4e-16,
tanjant 1,2e-16, enerji 0; N2 7,8e-10; N3 3,3e-11; N4 1,1e-8 (en büyük yer değiştirme açıklığın %54'ü); N5 geçti.
Hepsi PASS. Tek D yolu değişmedi (o dalda aynı ifadeler korunuyor).

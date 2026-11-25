# Ön kayıt: laminat (CLT), belirsizlik ve gürbüz tasarım, Pareto, ters problem, POD ROM

Görev: araştırma komutu `07_HOMOGENIZATION_RESEARCH.md` §2–§4 (00_MASTER kurallarıyla). §1 (RVE homojenleştirme)
bu kayıtta yok. Dal `claude/laminate-uq-3d5554`, taban `integrate/research-round-1` = `10daa03`.

Yazıldı: 2026-11-26 02:38 +0300 (`date` çıktısı), aşağıdaki modüllerin ilk satırından ve ilk koşudan önce. Bu dosya
koşudan sonra değiştirilmez; sonuçlar `laminat_uq.md` dosyasına yazılır.

Kurallar: tutmayan ön kayıtlı kontrol gevşetilmez; test `xfail(strict=True)` olarak kalır, ölçülen değer docstring'e,
sonuç kaydına ve `docs/vv_kutugu.json`'a yazılır. Sonradan eklenen her kontrol "post hoc" etiketlenir.

## 0. Tasarım kararları

- Modüller (yalnız yeni dosya): `core/laminate.py` (CLT), `optimization/laminate_failure.py` (ilk kat hasarı),
  `optimization/uq.py`, `optimization/pareto.py`, `optimization/identification.py`, `optimization/rom.py`.
- Tensör cebiri kopyalanmaz: kat malzemesi `core.linear_elastic.ElasticTensor`, indirgeme `core.material.plane_stress`,
  dönüşüm `core.tensor.rotate4` / `rotate2`. Tsai–Wu için `optimization.strength.Criterion` kullanılır.
- Laminat konvansiyonu: z orta düzlemden ölçülür, katlar alttan (z = −h/2) üste sıralanır; açı derece, +z çevresinde
  global x'ten 1 eksenine; Voigt `[xx, yy, xy]`, mühendislik kayması. `[N; M] = [[A, B]; [B, D]] [ε⁰; κ] − [N_T; M_T]`.
- Dayanım değerleri her zaman kullanıcı girdisi. Testlerdeki ve betikteki bütün malzeme ve dayanım sayıları
  SENTETİKTİR; ölçülmüş bir malzemeye ait değildir.
- Hashin düzlem gerilme biçimi: Abaqus 2025 belgesi "Damage initiation for fiber-reinforced composites"
  (docs.software.vt.edu aynası, 26 Kasım 2026'da okundu; α = 1 → Hashin 1980). Hashin'in makalesi okunmadı.
- UQ: örnek atılmaz. Başarısız örnek varsa istatistik ya hata verir ya da açıkça seçilen "en kötü" (+∞) politikası
  uygulanır.
- ROM sonucu "yaklaşık" etiketi ve hata sınırı taşır.

## 1. Tolerans gerekçesi

Cebirsel özdeşlikler float64 ile 1e-12 göreli. Ters alma / çözüm içeren denge denetimleri 1e-10. Türevler: merkezi
fark h = 1e-6 · ölçek, göreli 1e-6; Taylor kalanı eğimi ≥ 1,9. İstatistiksel ölçütler sabit tohumla deterministiktir;
sınırlar kestiricinin standart hatasının yaklaşık 4 katı ya da yöntemin beklenen yakınsama mertebesidir.

## 2. Laminat (L)

| # | Kontrol | Referans / bağımsız taraf | Eşik |
|:--|:---|:---|---:|
| L1 | Kat indirgenmiş rijitliği Q | Q11 = E1/(1−ν12ν21), Q22, Q12 = ν12E2/(1−ν12ν21), Q66 = G12 | göreli ≤ 1e-12 |
| L2 | Q̄(θ), 25 açı | (a) klasik trigonometrik formüller (testte), (b) 3-B döndürme + düzlem gerilme | ≤ 1e-12 |
| L3 | Tek kat | A = Q̄h, D = Q̄h³/12, B = 0 | ≤ 1e-12 |
| L4 | ABD, 20 rastgele dizilim (1–12 kat) | kalınlık boyunca 3 noktalı Gauss, 3-B döndürme yolu | ≤ 1e-12 |
| L5a | 20 rastgele simetrik dizilim | B = 0: max\|B\| / (max\|A\| h) | ≤ 1e-12 |
| L5b | Dengeli dizilimler | \|A16\|, \|A26\| / max\|A\| | ≤ 1e-12 |
| L5c | [0/90], kat kalınlığı t | B11 = (Q22−Q11)t²/2 = −B22, B12 = B66 = 0 | ≤ 1e-12 |
| L6 | Yarı izotrop [0/45/−45/90]s | A11 = A22, A66 = (A11−A12)/2, A16 = A26 = 0 | ≤ 1e-12 |
| L7a | İzotrop tek kat, ΔT | ε = αΔT, κ = 0, kat gerilmesi / (E α ΔT) | ≤ 1e-12 |
| L7b | [0/90]s, tekdüze ΔT | ∫σ dz = 0, ∫σz dz = 0 (max\|σ\| h ile ölçekli) | ≤ 1e-12 |
| L8 | 20 rastgele dizilim, N, M, ΔT0 + zΔT1 | geri kazanılan kat gerilmelerinin Gauss integrali = N, M | ≤ 1e-10 |
| L9a | ∂ABD/∂θ_k, ∂ABD/∂t_k (autograd) | merkezi fark | ≤ 1e-6 |
| L9b | ½ rᵀ ABD⁻¹ r (θ, t) | Taylor kalanı eğimi | ≥ 1,9 |
| L10a | Tsai–Wu: [0], Nx = Xt h | indeks = 1 | ≤ 1e-9 |
| L10b | Termal artık gerilmeli ilk kat yük çarpanı λ* | λ*'da kuadratik hasar fonksiyonu = 1 | ≤ 1e-9 |
| L10c | Hashin dört mod, tek eksenli ve birleşik durumlar | indeks 1; σ/I'da F = 1 | ≤ 1e-12 |
| L10d | Eksen dışı [θ] çekme, Tsai–Wu dayanımı | testte bağımsız kuadratik kök | ≤ 1e-10 |
| L11 | Başarısızlık testleri | t ≤ 0, boş dizilim, Hashin'de S23 yok, uzunluk uyumsuzluğu | hata |

## 3. Belirsizlik ve gürbüz tasarım (U)

| # | Kontrol | Referans | Eşik |
|:--|:---|:---|---:|
| U1a | MC n = 2¹⁴, Y = a + bX, X ~ N(0,1) | a, b², a + bΦ⁻¹(α), a + bφ(Φ⁻¹(α))/(1−α), α = 0,95 | ortalama ve varyans ≤ 4 SH; kantil ≤ 0,07 b; CVaR ≤ 0,08 b |
| U1b | Sobol ve LHS n = 2¹⁴, aynı model | aynı | ortalama ≤ 1e-3 b; varyans göreli ≤ 5e-3 (LHS 2e-2); kantil ve CVaR ≤ 1e-2 b |
| U2 | Y ~ U(0,1), Sobol n = 2¹² | q_α = α, CVaR = (1+α)/2 | ≤ 1e-3 |
| U3 | %95 t güven aralığı kapsaması, Y = X², n = 200, 1000 tohum | 1/3 | kapsama ∈ [0,925; 0,975] |
| U4 | Ishigami ortalaması, 40 tohum, n = 2⁶…2¹³ | 3,5 | MC RMSE eğimi ∈ [−0,65; −0,35]; Sobol ≤ −0,8; LHS RMSE ≤ MC RMSE (her n) |
| U5 | Sobol indisleri, Ishigami, taban n = 2¹³ | analitik S1, ST | mutlak ≤ 0,02 |
| U6 | Kesik dağılımlar | örnekler sınır içinde; kesik normal ortalaması (scipy.truncnorm) göreli ≤ 1e-3; fiziksel sınır dışı destek → hata | — |
| U7 | Gauss kopula ρ = 0,7 | normal skorların örnek korelasyonu | ≤ 0,02 |
| U8 | Uzayın bir bölümünde hata veren model | istatistik hata verir; "worst" politikasında ortalama +∞, başarısızlık oranı < 1−α iken kantil sonlu | davranış |
| U9 | Nadir olay | 10'dan az aşım → `reliable = False`; Clopper–Pearson aralığı kesin olasılığı içerir | davranış |
| U10a | Rastgele yük açısı θ ~ U(−30°, 30°), uyum | senaryo uyumu = birim yük kuadratik formu (≤ 1e-9); Sobol n = 64 ortalaması analitik E[c] (≤ 1e-3) | göreli |
| U10b | ∂(ortalama + k·std)/∂x ve ∂CVaR/∂x | yönlü merkezi fark | göreli ≤ 1e-5 |
| U10c | Konsol 60 × 30, hacim 0,4: nominal optimum ile gürbüz (ortalama + 3·std, 16 senaryo, ortak rastgele sayılar) | ayrı tohumlu 512 senaryo | gürbüz tasarımın ortalama+3std ve CVaR0,95 değerleri nominalden KÜÇÜK; hacim ihlali ≤ 1e-3 |

## 4. Pareto (P)

| # | Kontrol | Eşik |
|:--|:---|---:|
| P1 | Baskınlık filtresi, elle kurulmuş küme | tam eşleşme |
| P2a | Konveks olmayan cephe (f1 = x1, f2 = g(1−(f1/g)²), g = 1 + 9x2): 21 ağırlık | her ağırlıklı toplam çözümü f1 ∈ {0, 1} (≤ 1e-3): iç noktalar bulunmaz |
| P2b | Aynı problem, 11 ε | \|f2 − (1 − f1²)\| ≤ 1e-5; ardışık f1 aralığı ≤ 0,15 |
| P3 | Konveks cephe (x², (x−1)²) | x* = 1 − w (≤ 1e-6) |
| P4a | İki yük durumlu konsol 40 × 20, hacim 0,4, w1 ∈ {0,1; 0,3; 0,5; 0,7; 0,9} | beşi de uygun ve karşılıklı baskın değil |
| P4b | aynı | w1 arttıkça c1 artmaz, c2 azalmaz (%1 pay) |
| P4c | ε-kısıt, ε = w1 = 0,5 çözümünün c2'si | c2 ≤ ε(1 + 1e-3) ve c1 ≤ 1,05 · c1(w1 = 0,5) |
| P5 | Başarısızlık testleri: negatif / toplamı sıfır ağırlık; ulaşılamaz ε uygun sayılmaz | davranış |

## 5. Ters problem / parametre tanılama (I)

Veri sentetiktir (CLT ileri modeli + tohumlu Gauss gürültüsü). Bilinmeyenler E1, E2, G12, ν12.

| # | Kontrol | Eşik |
|:--|:---|---:|
| I1 | Analitik Jacobian (`compliance_derivative` + `plane_stress_derivative`) ile merkezi fark | göreli ≤ 1e-6 |
| I2 | Gürültüsüz geri kazanım ([0]8, [90]8, [±45]2s çekme) | göreli ≤ 1e-8 |
| I3 | %1 gürültü: hata / sonsal std; ayrılmış doğrulama kuponları ağırlıklı RMS'i | her parametre ≤ 4; RMS ∈ [0,5; 1,6] |
| I4 | Yalnız [0] çekme (Nx) | rank = 2; sıfır uzayı E2–G12 (izdüşüm normu ≥ 0,99); kovaryans yok |
| I5 | 16 LHS başlangıcı | en az 14'ü aynı minimuma (göreli 1e-6) |
| I6 | Model uyumsuzluğu (veri doğrusal olmayan kaymayla, model doğrusal) | ayrılmış doğrulama ağırlıklı RMS ≥ 3 |
| I7 | Rank eksik veri + Tikhonov önseli | tanılanamayan parametreler önselde kalır (≤ 1e-6) |

## 6. POD indirgenmiş model (R)

Konsol 40 × 20, dört bölge modülü θ_q ∈ [0,5; 2] (K = Σ θ_q K_q), sabit yük; 40 eğitim (LHS), 50 test (ayrı tohum).

| # | Kontrol | Eşik |
|:--|:---|---:|
| R1 | Σ‖s − VVᵀs‖² = Σ_{i>r} σ_i² | toplam enerjiye göre ≤ 1e-12 |
| R2 | Tam rank tabanla eğitim anlık görüntüleri | göreli enerji hatası ≤ 1e-6 |
| R3 | Hata sınırı Δ = √(rᵀK̄⁻¹r / α) (min-θ), her test noktası | 1 − 1e-6 ≤ Δ/e ≤ √(maks θ oranı) (1 + 1e-6) |
| R4 | r = 15, test kümesi göreli enerji hatası (maks) | ≤ 1e-2 |
| R5 | Galerkin uyumu | s_r ≤ s (1 + 1e-10); s − s_r = ‖e‖²_K (≤ 1e-6 göreli) |
| R6 | Başarısızlık: tek yükten kurulan taban başka yükte | göreli hata ≥ 0,5 ve Δ ≥ e; eğitim kutusu dışı sorgu işaretlenir; r > anlık görüntü rankı ve θ ≤ 0 → hata |
| R7 | Süre (tam çözüm / indirgenmiş) | yalnız ölçüm, eşik yok |

## 7. Kapsam dışı (bu kayıtta doğrulanmayacak)

İlerleyen hasar, laminat plaka/kabuk eleman bağlantısı, enine kayma ve katlar arası gerilmeler, ayrık dizilim
optimizasyonu, RVE homojenleştirme, güvenilirlik indisi yöntemleri (FORM/SORM), hiper indirgeme, doğrusal olmayan ROM,
ölçülmüş deney verisi.

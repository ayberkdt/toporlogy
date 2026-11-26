# Genel ağ FEM çekirdeği, eleman sözleşmesi ve kilitlenme: ön kayıt

Yazıldı: 26 Kasım 2026 02:41 +0300 (`date`), kodun ilk koşusundan önce. Taban: `integrate/research-round-1` =
`10daa03`. Bu dosya koşudan sonra değiştirilmez; sonuçlar ve sapmalar `fem_genel.md`'ye yazılır. Eşik gevşetilmez:
tutmayan kontrol strict xfail olur ve ölçülen değer kaydedilir; sonradan eklenen her ölçüm "post-hoc" diye
etiketlenir.

Kapsam: `03_FEM_ELEMENTS.md` (araştırma turu 2). Kod yeni paket `src/toporlogy/fem/mesh/`; mevcut `fem/*.py`
dosyaları değişmez (okunur, çağrılır). Testler `tests/toporlogy/test_mesh_*.py`. Ölçüm betiği
`docs/olcumler/fem_genel.py` → `fem_genel.json`.

## 0. Sözleşme (koşudan önce sabit)

* Referans hücreler: dörtgen/altıyüzlü `[-1, 1]^d`; üçgen/dörtyüzlü birim simpleks (köşeler 0, e₁, e₂(, e₃)).
* Elemanlar: `Q4`, `Q9` (tam ikinci dereceden Lagrange; Q8 seçilmedi: serendipite uzayı bilineer eşlemeyle
  bozulmuş dörtgende yaklaşım mertebesi kaybeder — Arnold, Boffi & Falk 2002, **okunmadı**, bilinen sonuç olarak
  not), `T3`, `T6`, `H8`, `T4`, `T10` (düğüm sırası gmsh / `post/verify.py` ile aynı), yüz elemanları `L2`, `L3`.
* Quadrature: dörtgen/altıyüzlüde Gauss–Legendre çarpımı (`numpy.polynomial.legendre.leggauss`); simplekste
  Duffy (1982) çökertilmiş Gauss–Legendre çarpımı (Jacobian çarpanı açık), ayrıca klasik merkez (derece 1), 3 noktalı
  üçgen (derece 2) ve 4 noktalı Keast (derece 2, `post/verify.py`'deki sabitler). Her kural beyan ettiği dereceye
  kadar bütün tek terimlileri tam integre etmeli.
* detJ denetimi: kullanılan kuralın her noktasında ve referans düğümlerinde. Ölçek s_e = (elemanın en uzun kenarı)^d.
  detJ < −1e-12 s_e → `InvertedElementError`; |detJ| ≤ 1e-12 s_e → `DegenerateElementError`. Mutlak değer hiçbir
  yerde alınmaz; hata eleman numarasını söyler.
* Malzeme: fiziksel D (Voigt, mühendislik kayması, `core.tensor` sırası); 2-B'de 3 × 3 (düzlem gerilme ya da düzlem
  şekil değiştirme, çağıran seçer), kalınlık t; eleman başına D (ne, k, k) kabul edilir.
* Dirichlet: (DOF, değer) çiftleri, sıfır olmayan değer ve yük durumu başına değer; aynı DOF'a iki farklı değer
  hata. Reaksiyon r = K u − f (kısıtlı DOF'larda). Çözüm: SPD sistem `solvers.direct.DirectSolver`; simetrik belirsiz
  (karışık u-p) sistem SciPy SuperLU, kısmi pivotlu; PCG yalnız SPD.
* Destek tanısı: bağlı her bileşende rijit modlar (2-B: 3, 3-B: 6) kısıtlı DOF satırlarına kısıtlanır; rank eksiği
  > 0 → `MechanismError` (eksik mod sayısı ve bileşen). Her çözümde `fem.elastic.checked_residual`.
* Karışık u-p (seçilen kararlı yöntem, §5): σ = 2μ ε(u) − p I, p = −λ div u (basınçta pozitif); sistem
  [[A, Bᵀ], [B, −M/λ]], A = ∫2μ ε:ε, B = −∫ψ div φ, M = ∫ψψ. ν = 0,5'te C = 0; sabit basınç sıfır uzayındaysa
  (kısıtsız serbest DOF'larda ‖Bᵀ1‖ ≤ 1e-12 ‖B‖_F) ∫p = 0 Lagrange çarpanıyla eklenir. Düzlem gerilmede reddedilir
  (λ* = 2λμ/(λ+2μ) sınırlı; hacimsel kilitlenme yok). Yalnız izotrop.

## 1. Eleman sözleşmesi (E)

Her eleman (`Q4 Q9 T3 T6 H8 T4 T10`, yüz elemanları `L2 L3` dahil yalnız E1–E5):

| # | Kontrol | Eşik |
|:---|:---|:---|
| E1 | Kronecker: N_i(ξ_j) = δ_ij | max \|·\| ≤ 1e-14 |
| E2 | Birim bölüşüm Σ N_i = 1 ve türev toplamı Σ ∂N_i/∂ξ = 0, 50 rastgele nokta | ≤ 1e-14 / ≤ 1e-13 |
| E3 | ∂N/∂ξ merkezi farka karşı (adım 1e-6) | ≤ 1e-8 mutlak |
| E4 | Tamlık: elemanın polinom uzayının her tek terimlisi düğüm değerlerinden tam yeniden üretilir (Q4: 1, ξ, η, ξη; Q9: ξ^a η^b, a, b ≤ 2; T3/T4: P1; T6/T10: P2; H8: üç doğrusal) | ≤ 1e-13 |
| E5 | Quadrature tamlığı: her kural, beyan ettiği dereceye kadar her tek terimliyi tam integre eder (küp: Π 2/(a+1) ya da 0; simpleks: a! b! c! / (a+b+c+d)!) | göreli ≤ 1e-14 |
| E5b | Gauss–Legendre sıkılığı: n noktalı kural derece 2n'de en az bir tek terimlide tam değildir | hata > 1e-10 |
| E6 | Ters (aynalanmış) eleman → `InvertedElementError`, çökmüş eleman (iki düğüm aynı yerde) → `DegenerateElementError`; mesaj eleman numarasını içerir | hata fırlatır |
| E7 | Eleman rijitliği: simetri; özdeğerlerde tam olarak n_rbm (2-B 3, 3-B 6) sıfır (\|λ\| ≤ 1e-10 λ_max), kalanlar > 1e-8 λ_max; bir afin ve bir afin olmayan bozulmuş eleman; ν = 0,3 | simetri ≤ 1e-14 göreli |
| E8 | Genel yol Q4 / H8 birim kare/küpte `fem.elements.ke_q4_from_D` / `ke_h8_from_D` ile aynı (aynı Gauss kuralı) | max göreli ≤ 1e-14 |
| E9 | Q4 tek noktalı indirgenmiş integrasyon: rank tanısı 2 sahte (hourglass) mod bulur | tam 2 |

## 2. Ağ, montaj ve sınır koşulları (G)

| # | Kontrol | Eşik |
|:---|:---|:---|
| G1 | `Mesh.from_domain` Q4 (düzlem gerilme, t ≠ 1, h ≠ 1, maskeli alan, eleman başına E) ve H8 (maskeli, eleman başına E): K ve u, `fem.elastic.Elasticity` ızgara yoluna eşit | K max göreli ≤ 1e-14; u max\|Δu\|/max\|u\| ≤ 1e-12 |
| G2 | Afin yama testi (bütün elemanlar, iç düğümler ±0,2 h bozulmuş, kenarlar düz; 2-B düzlem gerilme ve şekil değiştirme, 3-B): kenarda doğrusal alanın sıfır olmayan Dirichlet değerleri, iç serbest | u ≤ 1e-10 göreli; Gauss noktası gerilmesi ≤ 1e-9 göreli |
| G2b | Eğri kenarlı yama (orta düğümler ±0,1 h bozulmuş): Q9, T6 (varsayılan kural); T10 derece-3 kuralıyla | ≤ 1e-10 göreli |
| G2c | Eğri kenarlı T10, varsayılan 4 noktalı kural (Bᵀσ detJ derece 3, kural derece 2: geçmesi beklenmez) | eşiksiz, ölçüm |
| G3 | Denge: Σ reaksiyon + Σ dış yük = 0 (kuvvet ve moment), yama testinde ve hacim kuvvetli bir problemde | ≤ 1e-12 göreli |
| G4 | Tutarlı yükler: sabit çekiş Σ f = t·A; doğrusal çekişin momenti tam; bir yüzdeki basınç Σ f = −p A n (n dış normal); kapalı yüzeydeki sabit basınç Σ f = 0; hacim kuvveti Σ f = b V (2-B'de t ile). L2, L3 (eğri dahil), T3/T6 ve Q4 yüzleri | ≤ 1e-12 göreli |
| G5 | Destek tanısı: yetersiz destek (2-B tek düğüm iki yönde: 1 eksik mod; 3-B bir yüz yalnız x'te: 5 eksik) ve desteksiz ikinci parça → `MechanismError`, eksik mod sayısı doğru; yeterli destek geçer | tam sayı |
| G6 | Desteklenmeyen kombinasyonlar açık hata: aynı DOF'a çelişen iki Dirichlet değeri; iç yüzde çekiş/basınç; yük durumu sayısı uyuşmayan Dirichlet değerleri; karışık u-p düzlem gerilmede; ν = 0,5 ile yalnız yer değiştirmeli eleman; uyumsuz modlu eleman genel ağda; bilinmeyen eleman; yanlış D boyutu | hata fırlatır |
| G7 | Çok sağ taraf: 3 yük durumu tek çözümde = ayrı ayrı; tek çarpanlara ayırma | ≤ 1e-13 göreli; sayaç 1 |
| G8 | Seyreklik deseni yeniden kullanımı: farklı E alanıyla montaj yeni desen kurmaz ve taze montajla aynı | ≤ 1e-15 göreli |
| G9 | Eleman başına C: seri iki malzemeli çubuk (ν = 0, E₂ = 4 E₁), Q4, Q9, T3, T6, H8, T4, T10: parçalı doğrusal tam çözüm | ≤ 1e-11 göreli |
| G10 | `TetMesh` → genel ağ T10 rijitliği = `post.verify.stiffness_t10` | ≤ 1e-13 göreli (Frobenius) |
| M1 | Tutarlı kütle: bir bileşenin bütün girdilerinin toplamı = ρV; HRZ toplanmış kütle pozitif köşegen ve aynı toplam (T6, T10 dahil) | ≤ 1e-13 göreli |
| M2 | Isı iletimi: doğrusal sıcaklık yama testi (sıfır olmayan Dirichlet) ve Robin yüzü olan bir boyutlu levha (h(T − T∞)), kapalı çözüm doğrusal | ≤ 1e-10 göreli |

## 3. Bağımsız ve analitik doğrulama (V)

| # | Kontrol | Eşik |
|:---|:---|:---|
| V1 | İkinci dereceden tam çözüm: T6, Q9 (afin/paralelkenar), T10 (afin) uzayında olan ikinci dereceden u, sabit hacim kuvveti, tüm sınır Dirichlet: FE çözümü tam | ≤ 1e-10 göreli |
| V2 | scikit-fem 12 (bağımsız uygulama): aynı ağda global K (DOF'lar koordinatla eşlenir) T3, T6, T4, T10 (afin bozulmuş), Q4, Q9 (bilineer bozulmuş, aynı Gauss mertebesi), H8 (bozulmuş, 2×2×2); hacim kuvveti yük vektörü; Dirichlet + hacim kuvvetli çözüm | K, f Frobenius göreli ≤ 1e-12; u ≤ 1e-10 |
| V3 | MMS (düzgün polinom alan, §4'teki ψ, φ; c = 1; ν = 0,3; düzlem şekil değiştirme ve 3-B), köşe düğümleri ±0,2 h bozulmuş (her düzeyde yeni tohum), orta düğümler düz kenarda; 2-B n = 4, 8, 16, 32; 3-B H8/T4 n = 4, 8, 16, T10 n = 2, 4, 8. Mertebe en ince iki ağdan | L2: p ≥ k + 1 − 0,2; enerji: p ≥ k − 0,2 (k = eleman derecesi) |
| V3b | Eğri kenarlı O(h) orta düğüm bozulması (Q9, T6) | eşiksiz, ölçüm (mertebe kaybı beklenebilir) |

## 4. Poisson oranı ve kilitlenme raporu (L)

MMS (birim kare / birim küp, tüm sınırda u = 0): g(t) = t²(1 − t)², 2-B ψ = φ = g(x) g(y), 3-B ψ = g(x) g(y) g(z);
u = curl ψ + c ∇φ (3-B: curl(ψ, ψ, ψ)). div u = c Δφ, p = −λ c Δφ (ortalaması sıfır). Hacim kuvveti
f = −μ Δu − (λ + μ) ∇(div u), polinomlardan tam türetilir.
"(A)": c = 0 (sıfır diverjanslı, p = 0, en zor kilitlenme durumu); "(B)": c = μ/λ (λ → ∞'da sonlu p = −μ Δφ).

ν taraması: −0,9, −0,5, 0, 0,3, 0,45, 0,49, 0,499, 0,4999; E = 1 sabit ve G = 1 sabit ayrı; 2-B düzlem şekil
değiştirme n = 16 düzenli ağ (Q4, Q9, T3, T6; ızgara yolunda Q4IM), 3-B n = 6 (H8, T4, T10; ızgara yolunda H8IM);
karışık u-p (Taylor–Hood T6-T3, Q9-Q4, T10-T4). Ölçü: göreli L2 ve göreli enerji hatası (A); ayrıca düzlem gerilme
Q4 ve T3 (yalnız ölçüm).

| # | İddia | Eşik |
|:---|:---|:---|
| L1a | Ölçek değişmezliği: E sabit ile G sabit göreli hatalar aynı (C'nin sabitle ölçeklenmesi u'yu değiştirmez) | fark ≤ 1e-10 |
| L1b | Kilitlenme var: Q4 ve T3 (düzlem şekil değiştirme, A) e_L2(0,4999) ≥ 5 e_L2(0,3); H8 ve T4 (3-B) aynı | oran ≥ 5 |
| L1c | Kararlı yöntem kilitlenmez: TH (T6-T3, Q9-Q4) e_L2(0,4999) ≤ 2 e_L2(0,3); ν = 0,5 tam (λ = ∞, ayar ile) çözülür ve e(0,5)/e(0,4999) ∈ [0,9, 1,1] | oran |
| L1d | Aynı yer değiştirme uzayı: TH T6-T3, ν = 0,4999'da T6 (yalnız yer değiştirme) hatasından küçük | e_TH < e_T6 |
| L1e | Ağ yakınsaması ν = 0,4999, (B), bozulmuş ağ n = 8, 16, 32: TH yer değiştirme L2 mertebesi ≥ 2,8, basınç L2 mertebesi ≥ 1,8; ν = 0,5 tam aynı eşikler | mertebe |
| L2 | inf-sup (tüm sınır Dirichlet, n = 4, 8, 16): genelleştirilmiş özdeğer B A⁻¹ Bᵀ q = β² M q; TH: tam 1 sıfır özdeğer (hidrostatik) ve β_h(16) ≥ 0,5 β_h(4); Q4-P0 (kararsız karşılaştırma, B-bar/ortalama dilatasyona denk): en az 2 sıfır özdeğer (hidrostatik + dama tahtası) | sayı, oran |
| L3 | Eğilme (Timoshenko & Goodier düzlem gerilme kapalı çözümü, depoda H1 ile denetli; ν = 0,3 2-B; ν = 0'da 3-B tam çözüm): L/h = 5, 10, 50, 100. Aile F1: kare eleman, n_y = 1, 2, 4 (3-B 1, 2), n_x = (L/h) n_y, ızgara Q4, Q4IM, H8, H8IM ve genel ağ elemanları; aile F2: n_x = 20 n_y sabit (eleman en-boy oranı L/h / 20), yalnız genel ağ. Ölçü R = U_h / U_tam (iki uçta tam çekiş, rijit hareket üç/altı DOF ile tutulur; uyumlu elemanda R ≤ 1). | aşağıda |
| L3a | F2, n_y = 2, L/h = 100: Q4 ve H8 R ≤ 0,5 (parazit kayma) | ≤ 0,5 |
| L3b | F2, n_y = 2, her L/h: Q9 R ≥ 0,95 | ≥ 0,95 |
| L3c | F1, n_y = 2, L/h ≥ 10: Q4IM ve H8IM R ≥ 0,95 | ≥ 0,95 |
| L3d | Kayma kilitlenmesi fiziksel kayma değildir: tam çözümde kayma enerjisi oranı L/h = 100'de < 1e-3 ve L3a'daki Q4 açığı 1 − R bu oranın 100 katından büyük | iki eşitsizlik |
| L4 | Koşul sayısı ≠ kilitlenme (Q4 ve TH, ν = 0,3, 0,49, 0,499, 0,4999, n = 16): κ₁ (onenormest) ham ve Jacobi ölçekli, Jacobi-PCG yineleme sayısı (1e-10), bir adım yinelemeli iyileştirmenin u değişimi. İddialar: Q4'te κ(0,4999)/κ(0,3) ≥ 100; Jacobi ölçeklemesi çözümü değiştirmez (≤ 1e-8 göreli) ama e(0,4999)/e(0,3) ≥ 5 kalır; iyileştirme adımı ≤ 1e-8 göreli (hata yuvarlamadan gelmiyor) | eşikler |

Performans (eşiksiz, ölçüm): her ağda DOF, nnz(K), SuperLU nnz(L + U), montaj, çarpanlara ayırma ve çözüm süresi,
süreç tepe belleği (psutil peak working set). PARDISO bu makinede yüklenmiyor (`pypardiso` yok): SuperLU ölçülür.
GPU: yok, ölçülmez.

## 5. Kararlı yöntem seçimi (koşudan önce)

Seçim: **Taylor–Hood u-p** (P2–P1 üçgen/dörtyüzlü, Q2–Q1 dörtgen). Gerekçe: (i) inf-sup kararlı çift olduğu bilinen
sonuç (Brezzi & Fortin 1991 — **okunmadı**, bilinen sonuç); ν = 0,5 tam kısıtı ve basıncı doğrudan verir; (ii) kendi
T6/T10/Q9 elemanlarımızı kullanır, yani genel çerçevenin ikinci dereceden + simpleks genişlemesini sınar; (iii)
B-bar / ortalama dilatasyon (Hughes 1980; Hughes 2000 §4 — **bu oturumda okunmadı**) Q4/H8 için ucuz ama Q1–P0'a
denk ve inf-sup kararlı değil (dama tahtası basınç modu; L2'de ölçülecek); seçici indirgenmiş integrasyon hourglass
kontrolü ister. F-bar doğrusal olmayan çözücüyle (04) birlikte doğrulanmadan açılmaz. Önerilen kaynak
(doi:10.1007/s00466-019-01760-w) **okunmadı**. Sınır: TH yalnız izotrop doğrusal elastisite, düzlem şekil
değiştirme ve 3-B; simetrik belirsiz sistem SuperLU ile (PARDISO yok).

## 6. Kapsam dışı (bu turda yok; NOT VERIFIED olarak raporlanır)

Q8, H20/H27, kabuk/kiriş, MPC/periyodik/eğik mesnet (genel ağda açık hata), matris içermeyen ve yinelemeli üretim
çözücüsü (PCG yalnız L4 ölçümünde), geometrik rijitlik, F-bar, Cook zarı (yayınlanmış değerler bu oturumda kaynağından
okunmadı), gerçek parçalarda T10 kaynak ihtiyacı (gmsh kurulu değil).

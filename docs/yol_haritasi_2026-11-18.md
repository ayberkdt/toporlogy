# toporlogy — geliştirme planı (18 Kasım 2026)

**Girdi:** "TOPORLOGY — Sayısal Güvenilirlik, Yapısal Doğrulama ve Araştırma Yol Haritası" (dış belge, 18 Kasım 2026,
referans `main` = `de35073`) ve aynı gün İdeathon braketi üzerinde yapılan ölçümler
(`studies/final_2026/ansys/toporlogy_2026-11-18/`, depoda değil).
**Taban:** `main` = `de35073`, çalışma ağacı temiz.
**Etiketler:** [Ö] bu belgede komutla ölçüldü, [O] koddan okundu (dosya:satır), [D] doğrulanmadı.

Bu belge `gelistirme_plani.md`'nin (envanter, G-aşamaları, B-bulguları) devamıdır; sıradaki işlerin sırasını ve kabul
kapılarını tutar.

## 1. Dış belgenin iddiaları kaynağa karşı

| İddia | Durum | Kaynak |
|---|---|---|
| `checked_residual` bütün yük sütunlarını tek Frobenius normunda birleştiriyor; ‖b‖ = 0 ise x'e bakmadan 0 döndürüyor | doğru [O] | `fem/elastic.py:157-162` |
| Mekanizma denetimi yoğun `(AᵀA).toarray()` + `eigh` kullanıyor; maliyet ayrık küme sayısının küpüyle büyüyor | doğru [O] | `fem/mechanisms.py:138-139` |
| `factor()` E değerlerini (NaN, sıfır, negatif) denetlemiyor; kinematik denetim pozitif modül varsayıyor | doğru [O] | `fem/elastic.py:109-115` |
| T10: r1'de iki ağ arasında %12,4 değişim; başka örnekte ince ağ yalnız %1,8 eleman eklemiş; 12 girdiden 7'si bütün kapıları geçiyor | doğru [O] | `docs/olcumler/robustness.md:11, 276-279` |
| `de35073`'te CI'ın test işleri "devam ediyordu" | güncel değil: dört iş de (mypy, Ubuntu 3.11, 3.12, Windows 3.12) başarılı [Ö] | GitHub Actions koşusu 44303045199 |
| Top88/Top3D ve NAFEMS karşılaştırması yok; 3-D gerilme, burkulma, doğrusal olmayanlık, homojenleştirme yok | doğru [O] | depoda bu adlarla modül ya da test yok |

Belgenin önceliklendirmesi (P0 → P1 → araştırma; P0 bitmeden FEM çekirdeğine yeni fizik yok) bu plana aynen alındı.

## 2. Braket çalışmasından gelen üç yeni bulgu

**2.1 Son yüzeydeki basamaklar zarftan geliyor [Ö].** `finalize`, parçayı tasarım alanının dış sınırıyla (`Bound`
rolü, ad `envelope`) kırpıyor. STL alanında bu sınır CAD yüzeyi değil, 7,5 mm vokselin kısmi hacimlerinden kurulan bir
alan (`post/sdf.py::envelope_field`, `post/pipeline.py:132`). Sıfır düzeyi CAD yüzeyinden ortalama 0,46 mm, p95
1,29 mm, en çok 3,37 mm sapıyor; R80 kulak yayında yarıçap 78,99–80,99 mm arasında, yaklaşık 5,7 mm periyotla
dalgalanıyor. Son parçanın düğümlerinin %49'u bu zarfın üstünde. Taubin yumuşatması dalgayı azaltıyor, ardından
gevşetme adımı (`project_to_implicit=True`) düğümleri dalgalı zarfa geri izdüşürüyor. `upsample`'ı artırmak bunu
çözmez: zarfın hatası ince ızgaradan değil, optimizasyon vokselinden geliyor.

**2.2 Panel (b) yanıltıcıydı [Ö].** Sunum figüründeki yoğunluk yüzeyi görselleştirme betiğinde ek yumuşatmayla
(windowed sinc, 30 iterasyon) ve zarfla kırpılmadan çizildi; düğümlerinin %24,6'sı parçanın 0,5 mm'den fazla
dışında (en çok 8,2 mm). (b) ile (c) aynı koşullarda karşılaştırılmadı.

**2.3 Asimetriyi bağlantı bağları ekiyor [Ö].** e_sym = ‖ρ − R(ρ)‖₂ / ‖ρ‖₂:

| Düzlem | Izgara eşleşmesi | e_sym(frac) | e_sym(x̄) |
|---|---|---|---|
| x = 500 (pim ekseni, cıvata deseni merkezi) | tam (kayma 0 mm) | 0,012 | 0,478 |
| y = 200,56 (parça ortası) | kayma ≤ 3,62 mm | 0,117 | 0,474 |

x = 500'e göre tasarım alanı ve yük kümesi simetrik (LC2'nin momenti aynada işaret değiştirir, uyum yükte karesel
olduğu için değişmez); sonuç yine güçlü biçimde asimetrik. Korunan hücrelerin (`keep`) aynaya göre farkı 228 hücre:
bağlantı bağlarının 114 hücresi tek tarafta, ayna karşılıkları korunmuyor. Bağlar optimizasyondan önce donduruluyor;
asimetriyi ilk eken onlar. Problemin dışbükey olmaması bunu büyütmüş olabilir [D]: ayrıştırmak için bağsız koşu
gerekiyor (PR-11). y'ye göre ise tasarım alanının kendisi simetrik değil (kulak kalınlıkları 70 / 71,12 mm, ızgara
hizası); orada simetri beklemek yanlış olur.

## 3. PR sırası

Her PR tek problem; önce eski davranışı yakalayan kırmızı test, sonra en küçük düzeltme. Sayıyı değiştiren her PR
baseline farkını gerekçesiyle yazar; baseline sessizce yenilenmez.

| PR | Kapsam | Kırmızı test | Kabul | Bağımlılık |
|---|---|---|---|---|
| **PR-01** | Yük sütunu başına residual; sıfır yük sütununda ‖Kx‖ denetimi; NaN/Inf'te hata; `SolveResidualError` sütun indeksini yazar; `last_residual` en kötü sütun, sütun vektörü ayrıca saklanır; GMG ve `ConstrainedElasticity` aynı metrikte | ‖b₁‖ = 1e8, ‖b₂‖ = 1, x₂ yanlış → eski kod geçer; b = 0, x ≠ 0 → eski kod geçer | iki kırmızı test yeşil; çok yüklü örneklerde yanlış alarm yok; ek süre ölçülüp yazıldı | — |
| **PR-02** | Kinematik denetimin sözleşmesi: E şekli, NaN/Inf, sıfır ve negatif değer politikası; önbelleğin ömrü (geometri ya da mesnet değişince yeniden); mekanizma / kötü koşullanma / yakınsamama ayrı kodlarla | E içinde 0, < 0, NaN; mesnet değiştikten sonra eski önbellek | hatalı model sessiz geçmiyor; `Modal` istisnası korunuyor | PR-01 |
| **PR-03** | Mekanizma denetiminin ölçeklenmesi: önce ölçüm (k = 1, 10, 100, 300, 1000 küme; süre, bellek), sonra temas grafiğinde bileşen bileşen rank; sınır aşılırsa `MechanismCheckIndeterminate`, sessiz başarı yok | binlerce ayrık ada | 160 rastgele test dahil eski testler geçiyor; kaynak tüketimi kayıtlı | PR-02 |
| **PR-04** | CI: sürüm ve ortam kaydı her koşuda; baseline için kilitli referans ortamı; PARDISO ve gmsh/T10 için ayrı iş akışı, yoksa SKIP / NOT VERIFIED | — | zorunlu işler yeşil, opsiyonel işlerin durumu raporda ayrı | — |
| **PR-05** | Tam CAD zarfı (2.1): STL alanında `Bound`, CAD üçgen ağının işaretli uzaklığından ince ızgarada kurulur; YAML'da `finalize: {envelope: cad}` | zarf sıfır düzeyinin CAD'den p95 sapması (bugün 1,29 mm) | p95 ≤ 0,25 h_f, R80 yayında dalga ≤ 0,2 mm; braketin son yüzeyinde basamak yok; baseline farkı gerekçeli | — |
| **PR-06** | SDF / CSG sözleşmesi: negatif ofsetle yok olan küre, silindir, kutu; sıfır eksen; `element_fraction` örnekleme yakınsaması | bozuk geometrinin sessiz kabulü | belirlenmiş davranış, analitik karşılaştırmalar | — |
| **PR-07** | T10 ağ yakınsaması: aynı son geometri, en az üç gerçekten farklı ağ (eleman sayısı, kenar uzunluğu, kalite kayıtlı), yük eşdeğer dağıtılmış; asimptotik bölge yoksa NOT CONVERGED | mevcut %12,4'lük farkın "yakınsadı" sayılmaması | yakınsayan ve yakınsamayan örnekler ayrılıyor; 12'li geçmiş değiştirilmeden kalıyor | PR-04 (gmsh), PR-05 |
| **PR-08** | Sonuç sözleşmesi: PASS / FAIL / UNVERIFIED; eski `result.json`'dan okunan kısıtlar UNVERIFIED; `summary`, `result`, `read_run` aynı tasarımı söyler | eski dosyadan "ok" çıkması | doğrulanmamış sonuç "ok" altında kaybolmuyor | — |
| **PR-09** | Bağımsız V&V: orijinal Top88 / Top3D ile aynı problem (önce amaç ve gradyan, sonra geçmiş), analitik ya da NAFEMS elastisite problemi, çoklu modal, toleranslar deneyden önce yazılı | önceden seçilmiş referanslarla fark | karşılaştırma tekrarlanabilir, farkların nedeni yazılı | PR-01 |
| **PR-10** | Simetri tanısı (2.3): `summary.json`'a seçilen düzlemlere göre e_sym(frac), e_sym(keep), e_sym(x̄) ve son STL için geometrik simetri hatası; simetri bağlama kullanıcı seçimi olarak kalır, varsayılan değil | — | braket için yukarıdaki sayılar tekrar üretiliyor | — |
| **PR-11** | Bağ yerleşimi (2.3): bağlar simetri grubunun bütün görüntüleriyle ya da simetrik problem algılanınca simetrik yerleştirilir; deney: aynı braket (a) bağsız, (b) simetrik bağlı, (c) `SymmetryTying` ile; uyum, KKT, uygunluk, tek gövde, en ince kesit karşılaştırılır | x = 500'de e_sym(keep) = 0,224 | simetrik problemde e_sym(keep) = 0; asimetrinin bağdan mı dışbükey olmayan problemden mi geldiği ölçümle ayrıldı | PR-10 |
| **PR-12** | Araştırma prototipi: R-01 (mekaniği koruyan geometri sonlandırma); önceden kayıtlı problem kümesi, temel yöntemler (eşik + marching cubes, bugünkü `finalize`, yalnız hacim düzeltmeli), üç ağ seviyesiyle bağımsız son-geometri FEM, başarısızlıklar raporda | temel yönteme üstünlüğün sınanması | hipotez, ablasyon, başarısız örnekler yazılı | PR-05, PR-07 |

**Durum:** PR-01 yapıldı (dal `fix/per-rhs-residual`): yük sütunu başına bağıl ve mutlak residual ile geri hata
(‖K‖₁ üst sınırıyla), sıfır yük sütununda sıfır çözüm şartı, NaN/Inf'te hata, hata iletisinde yük durumu numarası;
`Elasticity`, `ConstrainedElasticity` (indirgenmiş yerel sistemde) ve `ElasticityGMG` (PCG'nin yeniden hesaplanan
residual'ı, aynı adlarla) aynı raporu veriyor. Eski kodda kırmızı olan iki test: 1e8 / 1 büyüklüğündeki iki yükte
%1 yanlış ikinci sütun ve sıfır yükte sıfır olmayan çözüm. Ek maliyet (48 × 24 × 24 H8, 90 000 serbest DOF, 3 yük,
PARDISO): denetim 74 ms + çarpanlara ayırma başına ‖K‖₁ yaklaşık 46 ms, bir çözümün yaklaşık %2,5'i; uçtan uca
ölçülen fark +%5,2 (PARDISO süresinin yayılımı dahil). GMG CUDA yolu bu makinede denenmedi.

PR-02 yapıldı (dal `fix/kinematic-contract`): modüller her çarpanlara ayırmadan önce denetleniyor (şekil, sonluluk,
pozitiflik; yoğunluk için sıfıra izin var, çünkü iki aralıklı kütle interpolasyonu boşluk elemanlarını bilerek kütlesiz
yapıyor; ilk denemede yoğunluğu da pozitif istemek modal testleri ve `eigenfrequency2d` baseline'ını kırdı); doğrudan,
çok ızgaralı ve modal yolların hepsinde. Eski kodda negatif ya da sıfır modül sessizce çözülüyordu (negatif modülle K
belirsiz, residual yine 1e-10'un altında). `fixed` ve `free` salt okunur; mesnet denetimi nesne başına bir kez.
Hata türlerinin ayrı kodları var: `mechanism`, `invalid_moduli`, `solve_residual`.
**PR-01'in yanlış alarmı düzeltildi:** yüksek rijitlik kontrastında (SIMP boşlukları) doğru çözülmüş sistemin bağıl
residual'ı 1e-6'yı aşıyor (ölçülen: 120 × 60, kontrast 1e-9, 4e-4), geri hatası yuvarlama düzeyinde kalıyor (PARDISO
ve SuperLU, 2-D ve 3-D, kontrast 1–1e-9 arası en çok 1,1e-16). Artık bir yük durumu yalnız bağıl residual > 1e-6 VE
geri hata > 1e-14 ise reddediliyor; kontrast 1e-9'da %1 yanlış çözüm (η 1,5e-13) hâlâ reddediliyor, eşik 1e-12'ye
gevşetilince test düşüyor.

PR-03 yapıldı (dal `fix/mechanism-scaling`): mesnet denetimi küme temas grafiğinin bileşenlerine ayrıldı (denklemler
yalnız ortak düğümde bağlandığından blok köşegen; kesin, yaklaşık değil); her bileşen kendi merkezine ve boyutuna
ölçeklenip tekil değerlerle (normal matris yok) çözülüyor. Ölçüm (aynı makine): 1000 ayrı mesnetli ada 28,9 s / 574 MB →
0,51 s / 10 MB; tek bileşende 300 küme 0,97 s → 0,79 s. Tek bileşen 3000 bilinmeyeni (500 küme, 3,0 s) aşınca
`MechanismCheckIndeterminate` (kod `mechanism_indeterminate`); sınır `check_kinematics(max_unknowns=...)` ile bilerek
büyütülebilir. 160 rastgele ve 40 eğik mesnetli karşılaştırma yeni yöntemle de birebir.

PR-04 yapıldı (dal `ci/environment-and-backends`): her test koşusu başlıkta ve sonda ortamı (git SHA, Python, NumPy,
SciPy, torch, BLAS, işletim sistemi, opsiyonel paketler) ve "verification status" bölümünü yazıyor; `TOPORLOGY_ENV_REPORT`
ile JSON. PARDISO, gmsh/T10 ve CUDA testleri işaretli; atlanan arka uç NOT VERIFIED (nedeniyle), asla geçti sayılmıyor.
Baseline yalnız dosyadaki referans ortamda (sürümler, OS, mimari) bit düzeyinde; başka ortamda skalerler 1e-6 bağıl
toleransla, dizi ve dosya özetleri NOT VERIFIED. Yeni `backends.yml`: PARDISO, gmsh/T10 ve Linux'ta sayısal baseline
(haftalık, elle ve ilgili dosyalar değişince). 1e-6 toleransın Linux'ta tutup tutmadığı bu iş akışının ilk koşusunda
görülecek (henüz ölçülmedi). CUDA için GPU koşucusu yok: NOT VERIFIED.

İlk Linux koşusunda baseline işi kaldı. Neden tolerans değil, kütüphanedeki bir tekrarlanabilirlik hatasıydı:
`tie_regions` ve `bridge` eşit maliyetli yollar arasında SciPy Dijkstra'nın öncül dizisine uyuyordu ve bu seçim SciPy
sürümüne bağlı. `lbracket_bore3d`'de 252 dondurulmuş bağ hücresinin 14'ü SciPy 1.14.1 ile 1.18.1 arasında farklıydı;
amaç ilk iterasyonda 1,6e-3 bağıl farklıydı, 10 iterasyon sonunda en büyük fark 2,2e-2'ydi (bağlantı kısıtı). Yol artık uzaklıklardan geri izleniyor,
eşitlikte en küçük hücre indeksi seçiliyor (dal `fix/baseline-linux-tolerance`). Baseline'da yalnız `lbracket_bore3d`
değişti. Windows'ta NumPy 2.5.5 / SciPy 1.18.1 / torch 2.15.0 ile en büyük skaler fark 2,7e-10; 1e-6 tolerans
değişmedi. Linux'ta (PR #9'un `backends.yml` koşusu) en büyük fark 1,5e-10 (`lbracket_bore3d`).

PR-05 yapıldı (dal `feat/cad-envelope`): `finalize` için `envelope: cad` (Python: `PostParams(envelope="cad")`,
`finalize(..., part=...)`). Zarf, parça geometrisinin işaretli uzaklığından ince ızgarada kuruluyor: CSG alanında
şekillerin `sdf`'i, STL alanında yeni `geometry.mesh_sdf_grid` (yüzeyden 3 h_f içinde nokta–üçgen uzaklığı tam,
işaret x, y, z ışın paritesinin çoğunluğu). Alan kurucuları geometriyi `BuildContext.part`'a yazıyor; box alanında
`envelope: cad` yapılandırma hatası. Varsayılan `voxel` kaldı, baseline değişmedi. Braket (h = 7,5 mm, h_f = 2,5 mm,
VTK ile ölçüldü): zarf sıfır düzeyinin CAD'den p95 sapması 1,29 → 0,010 mm, en çok 3,37 → 0,51 mm (keskin kenarlarda);
R80 yayında zarfın yarıçapı 79,05–80,68 → 79,93–80,00 mm. Son parçada (mevcut tasarım, 150 iterasyon) yayın 109
açı × kalınlık diliminin 97'si 79,95–80,00 mm'de; `voxel` ile hiçbiri bu aralıkta değil, 73'ü 80,0–80,5 mm'de (CAD'in
dışında). Kulak ön yüzündeki basamak çerçevesi gözle kayboldu. Bedeli: braket zarfı 42 s (voksel 4–8 s), `finalize`
107 → 142 s. Varsayılanın `cad` olması ayrı karar: baseline'da `lbracket_bore3d`'nin geometri dosyalarını değiştirir.

Görselleştirme betiği (panel b) PR değil, çalışma klasöründe düzeltilir: ham iso-yüzey, CAD'le kırpılmış, ek
yumuşatma yok, figür notu buna göre.

## 4. Bu planın dışında kalanlar

- **R-02** (hacim bütçesine duyarlı bağlantı): PR-12'den sonra ikinci araştırma hattı adayı. Bugünkü 14 senaryoda
  `one_body()` ve donmuş bağlar ikisi de 14/14 geçiyor; ortanca uyum oranı 1,30 / 1,25. Birleşik yöntemin üstün
  olduğu gösterilmiş değil (`docs/olcumler/robustness.md`).
- **R-03** (kafes, homojenleştirme): PR-01–PR-03 ve PR-09 geçmeden başlamaz; ayrı hat.
- **R-04** (3-D gerilme, burkulma, doğrusal olmayanlık): tek bir fizik seçilmeden ve doğrulaması planlanmadan
  başlamaz.

## 5. Açık kararlar

| No | Karar | Seçenekler | Öneri |
|---|---|---|---|
| Y1 | İlk araştırma hattı | R-01 / R-02 | R-01: PR-05 ve PR-07'nin altyapısını doğrudan kullanır |
| Y2 | Simetrik problem algılanınca bağlar | her zaman simetrik / kullanıcı seçimi | PR-11'in ölçümünden sonra karar |
| Y3 | gmsh / T10 CI'da | her PR / planlı iş akışı | planlı iş akışı (koşu süresi) |

## 6. Ek (21 Kasım 2026): sıra dışı güçlendirme çalışması

Depo sahibinin isteğiyle, §3'teki sıranın dışında bir çalışma yapıldı: literatür taraması, güncel yöntemlerin
eklenmesi ve yöntem türetme altyapısı. Kayıt: [`literatur_2026-11-21.md`](literatur_2026-11-21.md); ölçümler:
[`olcumler/yontemler.md`](olcumler/yontemler.md). Bu planla ilişkisi:

| Bu plandaki madde | 21 Kasım çalışmasının etkisi |
|---|---|
| "P0 bitmeden FEM çekirdeğine yeni fizik yok" (§1) | **Çiğnendi:** gerilme ve doğrusal burkulma eklendi. Çekirdeğin mevcut yolları değişmedi (OC/MMA, `Elasticity`, zincir); eklenenler ayrı modüller. PR-05–PR-12 açık kalıyor. |
| R-04 (3-D gerilme, burkulma: "tek bir fizik seçilmeden ve doğrulaması planlanmadan başlamaz") | İkisi de başladı. Doğrulama: kapalı çözümler ve sonlu farklar (yapıldı); bağımsız kodlarla karşılaştırma (yapılmadı; PR-09 ile aynı eksik). |
| PR-04'ün açık sorusu: "1e-6 toleransın Linux'ta tutup tutmadığı" | **Ölçüldü [Ö]:** Linux, Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, torch 2.15.0'da, değiştirilmemiş `5f97b86` üzerinde, yedi örnekten altısı 1e-6'yı tutuyor; `lbracket_bore3d` tutmuyor (amaç 285,434 ↔ 283,693, %0,6; kısıtlar da farklı). Neden araştırılmadı; aday: yüksek β'da robust + bağlantılılık kısıtlı problemin çözücü yuvarlamasına duyarlılığı [D]. `backends.yml`'in sayısal baseline işi bu örnekte kırmızı olacaktır. |
| PR-09 (bağımsız V&V) | Yeni modüller de aynı durumda: kendi kapalı çözümlerine karşı doğrulandı, topBuck250 / PolyStress / MFEM SiMPL örneğine karşı değil. |
| PR-12 / R-01 (mekaniği koruyan geometri sonlandırma) | Destekleyen yeni ölçüm: gri → 0/1 geçişte gerilme aşımı %2,5–71 (tek problem). `optimization/benchmark.py` bu hattın ölçüm aracının ilk parçası. |
| Y1 (ilk araştırma hattı) | Öneri değişmedi: R-01. Literatür taramasında bu boşluğu sistematik ölçen çalışma bulunamadı (yokluğun kanıtı değil). |

Yeni açık işler (öncelik sırasıyla): (1) tam metinlerin okunup kodlardaki "text not read" notlarının kapatılması;
(2) robust uzunluk ölçeği tablolarının Wang–Lazarov–Sigmund 2022 düzeltmesine karşı denetimi; (3) `lbracket_bore3d`
platform farkının nedeni; (4) burkulmada küme kesilmesi uyarısı; (5) GMG float32 yolunda rijit gövde modu enerjisi
ölçümü; (6) yeni yöntemlerin GMG fiziğiyle (`ElasticityGMG`) çalıştırılması — denenmedi.

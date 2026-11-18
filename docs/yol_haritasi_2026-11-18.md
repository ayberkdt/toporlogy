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

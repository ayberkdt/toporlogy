# toporlogy — literatür durumu, boşluklar ve güçlendirme çalışması (21 Kasım 2026)

**Amaç:** (1) yoğunluk tabanlı topoloji optimizasyonunda güncel gelişmeleri ve açık problemleri saptamak,
(2) deponun bu gelişmeleri kapsayan yöntemleri içermesini sağlamak, (3) depoyu yeni yöntem türetmeye elverişli hale
getirmek. Bu belge üçünün kaydıdır; ölçümler [`olcumler/yontemler.md`](olcumler/yontemler.md), türetme kılavuzu
[`deriving_methods.md`](deriving_methods.md).

**Taban:** `main` = `5f97b86`. **Etiketler** (`mimari_plan.md` ile aynı): **[Ö]** bu çalışmada ölçüldü, **[O]** koddan
okundu, **[K]** kayıt doğrulandı, **[H]** hafızadan / ikincil kaynaktan, doğrulanmadı, **[Y]** benim yorumum.

## 0. Önce okunması gereken sınırlar

1. **Hiçbir makalenin tam metni okunamadı.** Oturumun ağ politikası arxiv.org, link.springer.com, doi.org,
   orbit.dtu.dk ve Crossref'e erişimi engelledi. [K] etiketi yalnız şunu söyler: başlık, yazarlar, dergi ve DOI /
   arXiv numarası arama sonuçlarındaki yayıncı, arXiv ya da kurum kaydıyla eşleşti. Bir makalenin *ne söylediğine*
   dair cümleler özet ve arama parçacıklarından gelir; aksi belirtilmedikçe [H] sayılmalıdır.
2. **Formüllerin kaynağı.** Kodda kullanılan formüller ya ilk ilkelerden türetildi ve testle doğrulandı (gerilme,
   burkulma, toplama fonksiyonları, PDE filtresi), ya da yazarların açık kodundan okundu (SiMPL: MFEM örnek 37 ve
   `mmto-simpl`; GCMMA sabitleri: Svanberg kodunun Python portu, GPL olduğu için yalnız sabitler okundu, kod
   alınmadı). Her modülün docstring'i hangisi olduğunu yazar. `mimari_plan.md` §6'daki kural ("[L] öğeler kaynağı
   okunmadan uygulanmaz ya da docstring'de 'okunmadı' yazılır") ikinci yoldan karşılandı.
3. **Sıra kuralından sapma.** `yol_haritasi_2026-11-18.md` "P0 bitmeden FEM çekirdeğine yeni fizik yok" diyor ve
   PR-05–PR-12'yi sıraya koyuyor. Bu çalışma, depo sahibinin 21 Kasım'daki açık isteğiyle (güncel yöntemlerin
   eklenmesi) o sıranın dışına çıktı: gerilme ve burkulma eklendi. Çekirdekteki mevcut davranış değiştirilmedi
   (§5); PR-05–PR-12 hâlâ açık ve bu çalışma onların yerine geçmez.
4. **Kapsam.** Aşağıdaki ölçümlerin tamamı küçük 2-B problemlerde, tek başlangıç tasarımıyla yapıldı. "Yöntem X
   daha iyi" türünden genel bir sonuç bu belgeden çıkarılamaz.

## 1. Depo incelemesinin özeti [O]

18 072 satır kaynak, 433 test (yavaş olmayanlar; bu ortamda 428 geçti, 5 atlandı [Ö]); bu çalışmayla 88 test eklendi. Güçlü yanlar: türevlerin
tek bir elle yazılmış adjoint + autograd ile gelmesi (yeni yanıt = ileri hesap + sonlu fark testi), kanıt kültürü
(her iddia bir teste bağlı, önceden yazılmış eşikler), katman kuralları, anlık görüntü testleri. Bu çalışmanın
başında eksik olanlar (README "Known limitations" ve `mimari_plan.md` §4 ile uyumlu):

| Alan | Durum (21 Kasım sabahı) |
|---|---|
| Gerilme kısıtı | yok (eski 2-B sürücü yalnız regresyon kâhini olarak duruyor, testlerde kullanılmıyor) |
| Burkulma | yok |
| İyileştirici | OC (tek kısıt), düz MMA (genel yakınsama garantisi yok, 2-döngü belgelenmiş) |
| Filtre | koni filtresi (seyrek / konvolüsyon); PDE filtresi yok |
| Yanıt bileşimi | her yanıt ayrı sınıf; toplama fonksiyonları üç modülde ayrı ayrı yazılmış (robust, lengthscale, dynamics) |
| Yerel kısıtlar | yalnız toplanmış biçim (local_volume); artırılmış Lagrange yok |
| Adil karşılaştırma | gri tasarımın amacı raporlanıyor; 0/1 tasarımın yeniden değerlendirilmesi yok |
| Mekanizma / yer değiştirme amacı | yok |

## 2. Güncel durum, konu konu

Her başlıkta: literatürün durumu, depoya ne eklendiği, neyin eklenmediği.

### 2.1 Gerilme kısıtları

- **Durum.** İki ana yol var: (a) toplama — P-norm / KS ve Le vd. (2010) [K] uyarlamalı ölçeklemesi; Kennedy &
  Hicken (2015) [K] klasik toplamaların ağ inceltmede yakınsamadığını, "induced" biçimlerin yakınsadığını bildirir
  (özet); (b) yerel kısıtların artırılmış Lagrange ile amaca taşınması — Senhora vd. (2020) [K], PolyStress
  (Giraldo-Londoño & Paulino 2021) [K]; N ne olursa olsun tek adjoint. da Silva vd. (2021, IJNME 122:6003) [K]
  karşılaştırması: yerel stratejiler sürekliliğe daha az duyarlı, P-ortalama ancak P > 100'de rekabetçi (özet [H]).
  Norato vd. (2022) [K] "maximum rectifier" yeniden ölçekleme gerektirmeyen bir toplama önerir.
- **Eklendi.** `optimization/stress.py` (`VonMises`: katı gerilmesi, eleman merkezi, 2-B ve 3-B; qp gevşetmesi ve
  polinom "vanishing" biçimi), `optimization/aggregates.py` (altı toplama + uyarlamalı ölçek, sınırları özellik
  testiyle doğrulanmış), `optimization/local.py` (`Aggregated`, `AugmentedLagrangian`, `MultiplierUpdate`).
  Doğrulama: tek eksenli çekme 2-B/3-B'de 1e-9 [Ö]; eğilmede Q4 merkez gerilmesi %1,8 sapıyor [Ö] (eleman eğilmeyi
  tam temsil etmez); gradyanlar ≤ 1e-6, altı toplama × 2 boyut ve artırılmış Lagrange'ın iki biçimi [Ö].
- **Ölçülen sonuç** (`olcumler/yontemler.md` §3): L-brakette iki strateji de *gevşetilmiş* kısıtı *gri* tasarımda
  sağlıyor; katı elemanlardaki gevşetilmemiş gerilme %10–13, 0/1 tasarımda toplanmış yöntemde %71, artırılmış
  Lagrange'da %2,5 sınırın üstünde [Ö]. Artırılmış Lagrange ilk ayarlarla çalışmadı; iyileştirici sıfırlaması
  kaldırılınca çalıştı [Ö]. Ayarlar tek problemde bulundu.
- **Eklenmedi.** Bölgesel (kümelenmiş) P-norm, maximum-rectifier, robust (aşınmış/genişlemiş) gerilme, PolyStress'in
  kendi parametreleri ve normalizasyonu (okunamadı), cisme oturan ağda gerilme doğrulaması.

### 2.2 Doğrusal burkulma

- **Durum.** Ferrari & Sigmund (2019) [K]: adjoint terimi atlanınca duyarlılık yanlış (özet: "wrong sign");
  topBuck250 (Ferrari, Sigmund & Guest 2021) [K]: gerilme rijitliği için ayrı interpolasyon, μ = 1/λ üzerinde KS.
  Büyük ölçek: Ferrari & Sigmund (2020) [K] modları kaba ızgarada yaklaşıklar. Açık: sahte modların tam giderilmesi,
  kaç özdeğerin toplanacağı (Prabha & Kumar, arXiv:2609.19603 [K]), doğrusal olmayan / kusura duyarlı burkulmayla
  fark (Zhang vd. 2023 [K]), mod şekli denetimi (Li & Kennedy 2024 [K]).
- **Eklendi.** `fem/buckling.py` (geometrik rijitlik, özdeğer problemi (−K_G) φ = μ K φ, statik çözümün
  çarpanlarına ayırması yeniden kullanılıyor), `optimization/buckling.py` (`BucklingKS`, `BucklingLoadFactor`).
  **Türev elle yazılmadı:** modlar sabit tutulup Rayleigh oranı torch ifadesi olarak yazılıyor; yer değiştirmeye
  bağlı terim doğrusal çözümün mevcut adjoint'inden geliyor. Doğrulama: Euler kolonu 2-B'de 1,125 → 1,031 → 1,007
  → 1,0015 (ikinci mertebe) [Ö], 3-B H8IM 0,9975 [Ö]; gradyan 8,5e-9 / 2,6e-9 [Ö]; u'ya bağlı terim kesilince
  gradyan %96 sapıyor [Ö].
- **Eklenmedi.** Ferrari & Sigmund (2023) [K] gerilme alanı düzenlemesi (katıdaki yerel sahte modlar), kaba ızgara
  mod yaklaşımı ve GMG ile birleştirme, kümelenmiş özdeğerler için uyarlamalı mod sayısı. Yalnız düz `Elasticity`
  (doğrudan çözücü, eksen hizalı mesnet) destekleniyor.

### 2.3 İyileştiriciler

- **Durum.** GCMMA (Svanberg 2002 [H], 2007 notu [K: URL]) muhafazakâr yaklaşımlarla genel yakınsama verir.
  SiMPL (Kim, Lazarov, Surowiec & Keith, arXiv:2411.19421 [K]; analiz arXiv:2409.19341 [K]): gizli değişkende ayna
  inişi, sınırlar yapı gereği sağlanır; çok malzemeli genişleme arXiv:2605.11994 [K]. PGD tabanlı alternatifler
  (arXiv:2511.13905, 2412.07634 [K]). Rojas-Labanda & Stolpe (2015) [K: başlık/özet] genel amaçlı çözücülerin
  (IPOPT, SNOPT) klasik TO çözücüleri kadar verimli olduğunu bildirir.
- **Eklendi.** `optimizers/gcmma.py`, `optimizers/simpl.py`; `Study` iyileştiriciye gradyansız değer fonksiyonu
  veriyor (`needs_value_fn`). GCMMA: kabul edilen her adım muhafazakâr; düz MMA'nın belgelenmiş 2-döngüsü (genlik
  4e-3) kayboluyor, kalan salınım kabul toleransı düzeyinde (f'de 1e-9) [Ö]. SiMPL: bilinen çözümlü problemlerde
  1e-10 [Ö]; MBB'de OC/MMA ile aynı sonuç [Ö].
- **Ölçülen sonuç** (§1): MBB'de MMA, GCMMA ve SiMPL'in üç sürümü ikili uyumda %0,1 içinde; GCMMA ve çizgi aramalı
  SiMPL yaklaşık iki kat çözüm harcıyor ve bu problemde ölçülebilir kazanç sağlamıyor [Ö].
- **Bulunan hata (kendi uygulamamda).** SiMPL'de gizli değişkeni kaydırmadan *önce* kırpmak izdüşümü bozuyor ve
  büyük adımlarda inişi kırıyordu; kırpma kaydırmadan sonraya alındı, gerekçesi kodda [Ö].
- **Eklenmedi.** SiMPL'in genel doğrusal olmayan kısıtlara genişletilmesi (literatürde de bulunamadı — açık),
  makaledeki KKT tabanlı durdurma, iç nokta / SQP çözücüleri, Anderson ivmelendirmesi.

### 2.4 Filtre, izdüşüm, uzunluk ölçeği

- **Durum.** Helmholtz filtresi (Lazarov & Sigmund 2011 [K]); sınır etkisi ve çareleri (Clausen & Andreassen 2017
  [K]; Wallin vd. 2020 [K]). Alt-piksel yumuşatılmış izdüşüm SSP (Hammond vd., arXiv:2503.20189 [K]) β = ∞'da
  türevlenebilirlik sağlar; topoloji değişiminde türevlenemez, SSP2 (arXiv:2601.10737 [K]) bunu ele alır.
  Geometrik uzunluk ölçeği kısıtları (Zhou vd. 2015 [K]); hiperparametresiz sürüm (Arrieta vd., arXiv:2507.16108
  [K]). Otomatik β artışı (Dunning & Wein 2025 [K]). Wang, Lazarov & Sigmund (2011) için 2022 düzeltmesi var [K].
- **Eklendi.** `design/pde_filter.py` (`HelmholtzFilter`): hacim korur, [0,1]'i korur, sabiti duvarda ve delikte
  de aynen verir [Ö]. **Ölçülen:** `r = rmin/(2√3)` bağıntısı koni filtresinin ikinci momentini yalnız 1-B'de
  eşliyor; 2-B'de PDE filtresi %11, 3-B'de %25 daha geniş [Ö, sürekli ortam türetmesi + ayrık ölçüm]. Koni filtresi
  için kalibre edilmiş robust en küçük boyut ayarları PDE filtresine taşınamaz.
- **Eklenmedi.** SSP / SSP2, hacim koruyan izdüşüm (Xu vd. 2010; Li & Khandelwal 2015 [K]), Zhou tipi geometrik
  kısıtlar, Robin / dolgu sınır koşulları, otomatik β. **Depodaki bir risk:** robust uzunluk ölçeği tabloları
  2011 makalesine dayanıyorsa 2022 düzeltmesine karşı denetlenmeli (bu çalışmada denetlenmedi) [Y].

### 2.5 Makine öğrenmesi

- **Durum.** Sinirsel yeniden parametrelendirme (Hoyer vd. 2019; TOuNN 2021; Fourier özellikli sürüm 2022 [K]),
  yayılım modelleri (TopoDiff 2023 [K]), "temel model" denemesi OAT (arXiv:2510.23667 [K]). Eleştirel
  değerlendirmeler: Woldseth vd. (2022) [K] — doğrudan tahmin modelleri ikna edici değil, umut verenler ivmelendirme,
  son işlem, yeniden parametrelendirme, alt problemler; Sanu vd. (2025) [K] — ağ, dışbükey problemlere bile
  dışbükey olmama getirir; Yang & Wang (arXiv:2610.16051 [K], ön baskı) — verimli klasik kodla karşılaştırınca
  bildirilen hızlanmalar 20 kata kadar küçülüyor.
- **Eklenmedi (bilinçli).** Sinir ağı yok. Gerekçe: eleştirel literatürün vardığı yer ve deponun kanıt kuralı.
  Eklenen `optimization/benchmark.py`, bu tür iddiaları sınamak için gereken ölçütü (0/1 tasarım, toplam çözüm
  sayısı) sağlıyor.

### 2.6 Büyük ölçek, çok ölçek, son işlem, belirsizlik

- **Büyük ölçek.** Träff vd. (2023) [K] tek GPU'da matris tutmayan GMG; TOF3D (arXiv:2611.12027 [K], ön baskı)
  float32'de eleman rijitlik matrisinin yuvarlanmasının rijit gövde modlarına sahte enerji verdiğini ve uyumu
  %5,4'e kadar kaydırdığını bildirir. **Depo için risk:** `solvers/gmg*.py` karma hassasiyet kullanıyorsa bu
  sınanmalı (bu çalışmada bakılmadı) [Y].
- **Çok ölçek / de-homojenizasyon.** Groen & Sigmund (2018), Groen vd. (2020), Woldseth vd. (2024, fazör gürültüsü),
  deHomTop808 (2024) [K]. 3-B'de tekillikler eğri boyunca; yöntemler pahalı ya da kırılgan (özet [H]). Depoda yok.
- **Son işlem.** Optimum sonrası geometri çıkarmada performans kaybı "uzun süredir çözülmemiş" diye anılıyor
  (Swierstra & Gupta 2020 [K]); da Silva vd. (2021, arXiv:2006.12927 [K]) voksel ile cisme oturan model arasında
  %4,91 gerilme farkı bildirir. Bir kıyas kümesinde uyum/gerilme farkını sistematik ölçen 2022–2026 çalışması
  **bulunamadı** (yokluğun kanıtı değil).
- **Belirsizlik.** Robust üç alanlı formülasyon (depoda var), çok yüklü durumda rastgele örnekleme (Zhang vd.
  2017 [K]) — depoda yok, autograd ile kısa.

## 3. Literatürdeki boşluklar ve deponun konumu

"Boşluk" burada iki şeyden biri: derlemelerin açık diye andığı problem [K/H], ya da aramada karşılığı bulunamayan
bileşim [Y]. İkincisi yokluğun kanıtı değildir; başlamadan önce yeniden aranmalıdır.

| No | Boşluk | Dayanak | Deponun bugünkü imkânı | İlk ölçülebilir adım |
|---|---|---|---|---|
| G-1 | Yoğunluk alanından geometriye geçişte uyum ve gerilme kaybının sistematik ölçümü; kaybı azaltan sonlandırma | §2.6; `olcumler/yontemler.md` §3 (gri → 0/1: %2,5–71 gerilme aşımı [Ö]) | `post/` hattı + T10 yeniden analiz + `discrete_report`; yol haritasındaki R-01 / PR-12 ile aynı konu | önceden kayıtlı problem kümesinde gri / 0/1 / yumuşatılmış / T10 dörtlüsünün uyum ve en büyük gerilmesi |
| G-2 | Sabit ızgarada sınıra doğru gerilme (cisme oturan doğrulama olmadan) | da Silva vd. 2021 [H]; ölçülen fark [Ö] | `VonMises` + SDF tabanlı kısmi hacimler (`frac`) | merkez gerilmesi yerine SDF ağırlıklı / yeniden yapılandırılmış gerilme; Kirsch problemiyle ağ yakınsaması |
| G-3 | Burkulmada kaç mod, kümelenme ve sahte modlar | Prabha & Kumar 2026 [K]; ölçülen kümelenme (λ₂, λ₃ %2 arayla [Ö]) | `find_clusters` (modal) + `BucklingKS` | kümeyi kesmeyen uyarlamalı mod sayısı; küme kesildiğinde uyarı (modal'daki gibi) |
| G-4 | SiMPL'in doğrusal olmayan kısıtlara genişletilmesi | aramada bulunamadı [Y] | `SiMPL` + `AugmentedLagrangian` aynı arayüzde | gerilme AL amacı + hacim kaydırması; `stress_al` örneğinde MMA ile karşılaştırma |
| G-5 | PDE filtresiyle robust uzunluk ölçeği kalibrasyonu | çekirdek farkı ölçüldü [Ö] | `robust.blueprint_size` koni için sayısal; aynı yordam PDE çekirdeğine uygulanabilir | PDE filtresi için η–boyut tablosu ve ölçümle doğrulama |
| G-6 | Yeniden parametrelendirme = filtre = ön koşullayıcı birliği | biçimsel eşdeğerlik bildiren makale bulunamadı [Y]; Sanu vd. 2025 [K] manzara etkisini gösterir | zincirin ilk halkası indirgenmiş değişken alabiliyor (`n_in`, `reduce`) | doğrusal konvolüsyon halkası ile yoğunluk filtresinin aynı yörüngeyi verdiğini gösteren test; sonra doğrusal olmayan halka |
| G-7 | İyileştirici kıyaslaması (SiMPL, PGD, GCMMA; 3-B; performans profilleri) | 2015 sonrası kıyas bulunamadı [Y] | dört iyileştirici aynı arayüzde, `compare_runs` | 2-B/3-B problem kümesi, eşit çözüm bütçesi, 0/1 amaç |
| G-8 | float32 GMG'de rijit gövde modu hatası | TOF3D 2026 [K, ön baskı] | `solvers/gmg*` | rijit öteleme/dönme alanının eleman enerjisi float32'de ölçülür |

Öneri [Y]: G-1 ile G-2 aynı hat (yol haritasının R-01'i) ve deponun en özgün altyapısına (son işlem + T10) dayanıyor;
ilk araştırma hattı için Y1 kararındaki öneriyi (R-01) bu ölçümler destekliyor. G-4 ve G-6 en kısa yoldan yeni yöntem
üretebilecek olanlar; ikisinin de literatürde gerçekten boş olduğu, tam metinler okunmadan iddia edilemez.

## 4. Bu çalışmada eklenenler [O]

| Dosya | İçerik | Test |
|---|---|---|
| `optimization/aggregates.py` | KS, alt sınır KS, P-norm, P-ortalama, induced üstel / kuvvet; uyarlamalı ölçek | `test_aggregates.py` (12) |
| `optimization/local.py` | yerel alan → toplanmış yanıt ya da artırılmış Lagrange; çarpan güncelleme takvimi | `test_stress.py` |
| `optimization/stress.py` | von Mises alanı (2-B, 3-B), üç gevşetme biçimi | `test_stress.py` (23) |
| `fem/buckling.py`, `optimization/buckling.py` | geometrik rijitlik, burkulma modları, KS yanıtı | `test_buckling.py` (10) |
| `optimizers/gcmma.py`, `optimizers/simpl.py` | GCMMA, SiMPL | `test_gcmma.py` (6), `test_simpl.py` (7) |
| `design/pde_filter.py` | Helmholtz filtresi | `test_pde_filter.py` (6) |
| `optimization/compose.py` | `Functional`, `WeightedSum`, `Ratio`, `Shifted`, `Displacement` | `test_compose.py` (4) |
| `optimization/benchmark.py` | 0/1 değerlendirme, karşılaştırma tablosu; `discrete_check` son işlem adımı | `test_benchmark.py` (3) |
| `references/_topopt_methods.py` | 16 kaynak, her birinin doğrulama durumu notunda | içe aktarmada anahtar denetimi |
| `configuration/components/*` | 10 yeni YAML bileşeni (95 bileşen, 13 tür) | `test_method_components.py` (6), `test_research_examples.py` (7) |
| `examples/research/*.yaml` | beş örnek | `test_research_examples.py` |
| `docs/deriving_methods.md` | dört türetme örneği (yerel ölçüt, genel yanıt, zincir halkası, iyileştirici) | `test_deriving_methods.py` (4) |

README'deki "87 bileşen" sayısı bu çalışmadan önce de `list` çıktısıyla uyuşmuyordu (85 [Ö]); 95 olarak düzeltildi.

## 5. Mevcut davranışa dokunan değişiklikler [O]

- `optimizers/base.py`: `Optimizer.needs_value_fn`, `Optimizer.value_fn` (varsayılan kapalı).
- `optimization/study.py`: `run()` gerekirse `value_fn` verir ve sonunda kaldırır; `settings()` iç değerlendirme
  sayısını yazar.
- `optimization/study.py`: `run()` başında amaç ve kısıt yanıtlarının (ve sardıkları yanıtların) `reset_scale()`
  yöntemi çağrılır. Gerekçe: README'deki "iki koşu aynı sonucu verir" sözü, ilk değerlendirmede ölçek donduran
  yeni yanıtlarla (burkulma KS, uyarlamalı gerilme ölçeği, `WeightedSum("first")`) bozuluyordu; kırmızı test
  `test_method_components.py::test_two_runs_of_one_study_are_identical_with_scaled_responses` [Ö]. Mevcut
  `MinEigenvalueKS` ve `Robust` de bu çağrıyı alır; denenen özfrekans örneğinde düzeltme olmadan da iki koşu
  aynıydı [Ö], yani mevcut yanıtlarda ölçülmüş bir davranış değişikliği yok. Taze süreçteki ilk koşu etkilenmez
  (ölçek zaten boş), dolayısıyla bit düzeyi baseline da etkilenmez.
- `tests/toporlogy/baseline/list.txt` ve `api_surface.json`: yeni bileşenler ve adlar eklendi; hiçbir ad silinmedi
  ya da yeniden adlandırılmadı. `examples.json` (bit düzeyi baseline) değişmedi; yeni örnekler bu yüzden
  `examples/research/` altında.

## 6. Kaynaklar

Kodda atıf yapılanlar `references/_topopt_methods.py` içinde, doğrulama durumlarıyla. Yalnız bu belgede anılanlar
(hepsi [K] kayıt düzeyinde, aksi yazılmadıkça; tam metin okunmadı):

- Arrieta, Romano, Johnson. Hyperparameter-free minimum-lengthscale constraints for topology optimization. arXiv:2507.16108.
- Clausen, Andreassen (2017). On filter boundary conditions in topology optimization. SMO 56:1147–1155. doi:10.1007/s00158-017-1709-1.
- da Silva, Aage, Beck, Sigmund (2021). Three-dimensional manufacturing tolerant topology optimization with hundreds of millions of local stress constraints. IJNME 122(2):548–578. doi:10.1002/nme.6548.
- Dunning, Wein (2025). Automatic projection parameter increase for three-field density-based topology optimization. SMO 68:33. doi:10.1007/s00158-025-03968-2.
- Ferrari, Sigmund (2020). Towards solving large-scale topology optimization problems with buckling constraints at the cost of linear analyses. CMAME. doi:10.1016/j.cma.2020.112911.
- Ferrari, Sigmund (2023). A strategy for avoiding spurious localized buckling modes in topology optimization. IJNME. doi:10.1002/nme.7309.
- Gangl, Keith, Kim, Lazarov, Surowiec (2026). The SiMPL method for density-based multi-material topology optimization. arXiv:2605.11994.
- Groen, Sigmund (2018). Homogenization-based topology optimization for high-resolution manufacturable microstructures. IJNME 113(8):1148–1163. doi:10.1002/nme.5575. (başlık [H])
- Hammond, Oskooi, Hammond, Chen, Ralph, Johnson (2025). Unifying and accelerating level-set and density-based topology optimization by subpixel-smoothed projection. arXiv:2503.20189.
- Hoyer, Sohl-Dickstein, Greydanus (2019). Neural reparameterization improves structural optimization. arXiv:1909.04240.
- Li, Kennedy (2024). Buckling mode constraints for topology optimization using eigenvector aggregates. CMAME. doi:10.1016/j.cma.2024.117145.
- Mazé, Ahmed (2023). Diffusion models beat GANs on topology optimization. AAAI 37(8):9108–9116. arXiv:2208.09591.
- Nobari, Ahmed (2025). PGD-TO: a scalable alternative to MMA using projected gradient descent for multi-constraint topology optimization. arXiv:2511.13905.
- Norato, Smith, Deaton, Kolonay (2022). A maximum-rectifier-function approach to stress-constrained topology optimization. SMO 65:286. doi:10.1007/s00158-022-03357-z. (başlık [H])
- Pedrini, Andrade, Ferro (2026). An open source Python code for three-dimensional topology optimization in single precision on a consumer GPU. arXiv:2611.12027.
- Prabha, Kumar (2026). Topology optimization with buckling constraints: adaptive eigenvalue aggregation and modality identification. arXiv:2609.19603.
- Rojas-Labanda, Stolpe (2015). Benchmarking optimization solvers for structural topology optimization. SMO. (DOI doğrulanmadı)
- Romano, Arrieta, Johnson (2026). Differentiating through binarized topology changes: second-order subpixel-smoothed projection. arXiv:2601.10737.
- Sanu, Aragón, Bessa (2025). Neural topology optimization: the good, the bad, and the ugly. SMO 68:213. doi:10.1007/s00158-025-04135-3.
- Swierstra, Gupta (2020). Automated and accurate geometry extraction and shape optimization of 3D topology optimization results. arXiv:2004.05448.
- Träff, Rydahl, Karlsson, Sigmund, Aage (2023). Simple and efficient GPU accelerated topology optimisation: codes and applications. CMAME 410:116043. doi:10.1016/j.cma.2023.116043.
- Wallin, Ivarsson, Amir, Tortorelli (2020). Consistent boundary conditions for PDE filter regularization in topology optimization. SMO 62:1299–1311. doi:10.1007/s00158-020-02556-w.
- Wang, Lazarov, Sigmund (2011). On projection methods, convergence and robust formulations in topology optimization. SMO 43:767–784. doi:10.1007/s00158-010-0602-y; düzeltme (2022) doi:10.1007/s00158-022-03326-6. (başlık [H])
- Woldseth, Aage, Bærentzen, Sigmund (2022). On the use of artificial neural networks in topology optimisation. SMO 65:294. doi:10.1007/s00158-022-03347-1.
- Woldseth, Bærentzen, Sigmund (2024). Phasor noise for dehomogenisation in 2D multiscale topology optimisation. CMAME 418:116551. doi:10.1016/j.cma.2023.116551.
- Yang, Wang (2026). How the classical baseline shapes machine-learning speed claims in topology optimization. arXiv:2610.16051. (yazarlar ikincil kaynaktan)
- Zhang, de Sturler, Paulino (2017). Stochastic sampling for deterministic structural topology optimization with many load cases. CMAME 325:463–487. doi:10.1016/j.cma.2017.06.035. (başlık [H])
- Zhou, Lazarov, Wang, Sigmund (2015). Minimum length scale in topology optimization by geometric constraints. CMAME 293:266–282. doi:10.1016/j.cma.2015.05.003.

Tam metin doğrulaması için oturumun ağ ayarlarında arxiv.org, link.springer.com, doi.org ve orbit.dtu.dk'ya izin
verilmesi gerekir; ondan sonra §2'deki [H] cümleleri ve kodlardaki "text not read" notları tek tek kapatılmalıdır.

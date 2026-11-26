# Genel ağda yoğunluk optimizasyonu: ilk dilim — ön kayıt

Ön kayıt 2026-11-26 15:59 +0300, kod yazılmadan önce. `Study` ve tasarım zinciri düzenli ızgaraya (`Domain`) bağlı; genel ağın
`Study`/YAML'a girmesi mimari karar ister. Bu dilim onu değiştirmez: `optimization/mesh_topology.py` ayrı bir modül.

* `MeshDensityFilter(mesh, rmin)`: hücre ağırlık merkezleri arasında koni ağırlık max(0, rmin − mesafe) (rmin
  uzunluk biriminde), satır toplamıyla normalize, seyrek matris (KD-ağacı komşuluğu).
* `MeshCompliance(problem, F, fixed)`: `ElasticProblem`'in birim modüllü eleman matrisleriyle K(E) = Σ E_e K_e,
  c = Fᵀu, dc/dE_e = −u_eᵀ K_e u_e (öz-eşlenik).
* `optimize_mesh(...)`: SIMP E = E_min + x̄^p (1 − E_min), hacim hücre hacimleriyle ağırlıklı ortalama, MMA.

| # | Kontrol | Eşik |
|:---|:---|:---|
| MS1 | `Mesh.from_domain` ağında filtre = ızgara `DensityFilter(d, rmin).matrix()` (rmin_uzunluk = rmin · h) | ≤ 1e-14 mutlak |
| MS2 | Aynı ağda uyum ve dc/dE = ızgara `Elasticity` ile hesaplanan (aynı E, aynı mesnet ve yük) | ≤ 1e-12 göreli |
| MS3 | Bozulmuş Q4 ağında ve T3 ağında tam zincirin (filtre + SIMP + uyum) gradyanı merkezi farkla | ≤ 1e-6 göreli |
| MS4 | Bozulmuş Q4 ağında konsol, hacim ≤ 0,4, 30 MMA iterasyonu | son uyum < ilk uyum; hacim ≤ 0,4 (1 + 1e-3) |
| MS5 | rmin ≤ 0, hacim sınırı (0, 1] dışında, yanlış satırlı yük | açık hata |
| MS6 | var olan kod | `pytest -m baseline` bit düzeyinde (yeni modül başka dosyaya dokunmaz) |

## Sonuç (ilk ölçüm, 2026-11-26)

`tests/toporlogy/test_mesh_topology.py` (6 test); hepsi ilk koşuda geçti.

| # | Durum | Ölçülen |
|:---|:---|:---|
| MS1 | PASS | `from_domain` ağında hücre sırası = eleman sırası; filtre matrisi farkı ≤ 1e-14 (rmin 1,5 ve 2,5 eleman) |
| MS2 | PASS | uyum 2,7e-14, dc/dE 3,1e-14 göreli |
| MS3 | PASS | bozulmuş Q4: 1,1e-8; bozulmuş T3: 2,0e-8 göreli |
| MS4 | PASS | bozulmuş Q4 konsol (16 × 8): uyum 653 → 176 (30 iterasyon), hacim 0,4000 |
| MS5 | PASS | rmin = 0, hacim 1,2, yanlış satırlı yük: açık hata |
| MS6 | PASS | baseline 8 passed bit düzeyinde |

Sonraki adım (mimari karar): genel ağın `Study` / YAML'a girmesi — ızgaraya bağlı olmayan bir tasarım zinciri (bu
modüldeki filtre gibi) ve ağ etiketleriyle mesnet / yük.

## Ek ön kayıt: genel ağ `Study` içinde (MD1–MD5)

Ön kayıt 2026-11-26 16:08 +0300, kod yazılmadan önce. Karar (`tasarim_genel_ag_study.md` önerisinin varsayılanları, kullanıcının
"ne yapılması gerekiyorsa onu yap" yönergesiyle): ayrı `MeshStudy` yok; `Study` değişmeden kalır ve ağ, hücreleri
1-B bir "ızgara" olarak sunan `MeshDomain` ile girer (`shape = (n_cells,)`, hepsi aktif, `frac = 1`). Yeni modül
`optimization/mesh_study.py`: `MeshDomain`, `MeshFilterLink` (ağ filtresi, seyrek autograd halkası),
`MeshElasticity` (fizik; geri adım öz-eşlenik adjoint), `MeshVolume` (`Volume` alt sınıfı, hücre hacmiyle ağırlıklı).
YAML bu adımda yok.

| # | Kontrol | Eşik |
|:---|:---|:---|
| MD1 | `Mesh.from_domain` ağında ağ `Study`'si = ızgara `Study`'si (aynı x, rmin, SIMP, mesnet, yük): değer ve gradyan | ≤ 1e-12 göreli |
| MD2 | aynı iki çalışma, 10 MMA iterasyonu: amaç geçmişleri | her iterasyonda ≤ 1e-8 göreli |
| MD3 | bozulmuş Q4 ve T3 ağlarında `Study` gradyanı (amaç uyum) merkezi farkla | ≤ 1e-6 göreli |
| MD4 | bozulmuş Q4 konsol, `MeshVolume` ≤ 0,4, 30 MMA iterasyonu | son uyum < ilk uyum; hacim ≤ 0,4 (1 + 1e-3) |
| MD5 | var olan kod | `pytest -m baseline` bit düzeyinde |

### MD1–MD5 sonucu (ilk ölçüm)

Hepsi ilk koşuda geçti (`test_mesh_topology.py::test_md*`).

| # | Durum | Ölçülen |
|:---|:---|:---|
| MD1 | PASS | değer 1,3e-14, gradyan 6,1e-14 göreli; hacim farkı 0 |
| MD2 | PASS (sınıra yakın) | 10 iterasyonda en büyük fark 8,1e-9 (eşik 1e-8) |
| MD3 | PASS | bozulmuş Q4: 1,4e-7; bozulmuş T3: 2,6e-7 |
| MD4 | PASS | uyum 653 → 176 (30 iterasyon), hacim 0,4000 (`optimize_mesh`'in MS4 sonucuyla aynı sayılar) |
| MD5 | PASS | baseline 8 passed bit düzeyinde; mimari (katman) testleri 9 passed |

Kalan: YAML (`domain: {type: mesh}`, ağ etiketleriyle mesnet / yük), ızgaraya özgü bileşenlerin ağ alanında açık hatası
(katalog kuralı), kabuk elemanı.

## Ek ön kayıt: ızgaraya özgü bileşenlerin ızgara dışı alanda açık reddi (RF1–RF4)

Ön kayıt 2026-11-26 17:41 +0300, kod yazılmadan önce. `MeshDomain` ve `ShellDomain` hücreleri `shape = (n,)` olan 1-B
bir "ızgara" olarak sunar, `dim` ise uzay boyutudur (2 ya da 3). Izgaraya özgü bir halka ya da yanıt bu alanlarda bugün
ya anlaşılmaz bir hatayla düşüyor ya da sessizce yanlış hesaplıyor. Ortak denetim `toporlogy.core.domain` içinde tek
yardımcı olacak: `is_grid(domain)` (`len(shape) == dim`; her `Domain` için doğru, `MeshDomain` / `ShellDomain` için
yanlış) ve `require_grid(domain, bileşen, alternatif)`, ızgara dışı alanda bileşenin adını ve varsa ağ karşılığını
(`MeshFilterLink`, `MeshVolume`, açık eşlemeli `Tying`) veren bir `ValueError` atar. Uyumluluk adı yok.

Bugünkü davranış bu oturumda bir yoklama betiğiyle ölçüldü (2-B Q4 `MeshDomain`, `Mesh.from_domain` ile 3-B H8
`MeshDomain`, 8 × 6 silindir parçası `ShellDomain`; üç alanda da aynı sonuç). Izgaraya özgü bulunanlar ve nedeni:

| Bileşen | Neden ızgaraya özgü | Bugün | Ağ karşılığı |
|:---|:---|:---|:---|
| `DensityFilter`, `standard_chain`, `manufacturing_chain` | `dim` boyutlu evrişim çekirdeği, mesafe eleman biriminde | `IndexError` / conv3d `RuntimeError` | `MeshFilterLink` |
| `HelmholtzFilter` | Laplasyen her eksende ızgara komşuları arasında | **sessizce çalışıyor**: (n,) dizisinde k ile k+1 numaralı hücreyi komşu sayıyor | `MeshFilterLink` |
| `PeriodicAngleFilter` | `DensityFilter` çekirdeği | `IndexError` / `RuntimeError` | yok |
| `AMFilter`, `Casting` | ızgara ekseni boyunca katmanlar, `h`, `origin` | `IndexError` | yok |
| `SymmetryTying`, `PatternTying` (`translational`, `cyclic`), `mirror_closure` | `element_centers`, en yakın ızgara hücresi | `AttributeError` | açık eşlemeli `Tying` |
| `ExtrusionTying` | ızgara ekseni boyunca hücre çizgileri | numpy `AxisError` | açık eşlemeli `Tying` |
| `overhang_map`, `overhang_violations`, `undercut_map`, `undercuts` (alan verilince) | ızgara ekseni | `ValueError: direction '+y' is not valid in 1-D` | yok |
| `MultiPhaseSIMP`, `TensorPhases` | alan rankı = `domain.dim` | kendiyle çelişen `ValueError` ("(2, 24) … need (2, *domain.shape)") | yok |
| `phase_report` | `domain.h` | yoklanmadı (`h` okur) | yok |
| `Volume` (düz) | x̄ rankını `dim` ile karşılaştırır; eleman sayısıyla ortalar | yanıltıcı `ValueError` ("several phases") | `MeshVolume` |
| `Mass`, `Cost` | eleman hacmi `h^dim` | çok fazlı enterpolasyon olmadan erişilmiyor; olursa `h` okur | yok |
| `LocalVolume` | ızgarada top evrişimi | conv `RuntimeError` | yok |
| `VonMises`, `FailureIndex` | `edof`, `h`, Q4/H8 merkez B matrisi | `VonMises` kurulur, ilk değerlendirmede `AttributeError: h`; `FailureIndex` kurulurken | yok |
| `StressRecovery` | fiziğin `element`'i, `edof` | `AttributeError: element` | yok |
| `ConductionConnectivity`, `CouplingCompliance`, `analyse_connectivity`, `tie_regions`, `bridge`, `one_body` | yüz komşuluğu, düğüm ızgarası, ızgara FE fiziği | `AttributeError: edof` / `KeyError: '1'` (son üçü yoklanmadı) | yok |
| `CentreOfMass`, `InertiaComponent` (`mass_properties`) | `h`, `origin`, ızgara merkezleri | `AttributeError: h` | yok |
| `RobustSet` gerçekleşmeleri | zincirdeki `DensityFilter` çıktısını okur | `KeyError: no DensityFilter link` | yok |

Bugün ızgara dışı alanda çalışan ve çalışmaya devam etmesi gerekenler (yoklamada üç alanda da çalıştı): `Embed`,
`Passive`, `HeavisideProjection`, `SIMP`, `RAMP`, `LinearMass`, `TwoRangeMass`, `TwoRangeSIMP`, `MeshFilterLink`,
`MeshVolume`, `Compliance`, `GreyLevel`, açık eşlemeli `Tying`. Kapsam dışı: ızgara fizikleri (`Elasticity`,
`HeatConduction`, `Modal` …; halka ya da yanıt değil), dizi alan ölçüm fonksiyonları (`measure_lengthscale` gibi alan
almayanlar), `OrientedEmbed` (filtresiz hâli bir `Embed`; filtresi `PeriodicAngleFilter` üzerinden reddedilir).

| # | Kontrol | Eşik |
|:---|:---|:---|
| RF1 | yukarıdaki tablodaki her bileşen, üç ızgara dışı alanın her birinde | `ValueError`; iletide bileşenin adı ve "grid"; ağ karşılığı olanlarda onun adı; başka tür istisna 0 |
| RF2 | "çalışmaya devam etmesi gerekenler" listesi, üç alanda | hatasız çalışır; MD1–MD5 ve KS1–KS5 geçer |
| RF3 | `is_grid` doğruluk tablosu: 1-B, 2-B, 3-B `Domain` (boşluk / koruma / kısmi `frac` dâhil) doğru; 2-B ve 3-B `MeshDomain`, `ShellDomain` yanlış | tablonun tamamı |
| RF4 | ızgara davranışı | `pytest -m baseline` 8 passed bit düzeyinde; dokunulan modüllerin test dosyaları ve mimari / API yüzeyi testleri geçer |

### RF1–RF4 sonucu (2026-11-26)

`tests/toporlogy/test_grid_refusals.py` (145 test: RF1 35 bileşen × 3 alan, RF2 13 × 3, RF3 1). Ortak denetim
`toporlogy.core.domain.is_grid` / `require_grid`; ileti biçimi "`<bileşen>` needs a structured element grid (Domain),
not a non-grid `<MeshDomain|ShellDomain>` (n cells as a 1-D list); use `<karşılık>` instead" ya da "… it has no mesh
or shell counterpart yet". Ret, alanı kurucuda alan bileşenlerde kurucuda (çözümden önce), alanı yalnız bağlamdan
okuyan yanıtlarda (`Volume`, `Mass`, `Cost`, `LocalVolume`, `CentreOfMass`, `InertiaComponent`, `RobustSet`) ilk
değerlendirmede atılır.

| # | Durum | Ölçülen |
|:---|:---|:---|
| RF1 | PASS | 105 / 105 durumda `ValueError`, iletide bileşen adı, "grid", alan türü; ağ karşılığı olanlarda `MeshFilterLink` (`DensityFilter`, `HelmholtzFilter`), "a Chain of Embed, MeshFilterLink, Passive" (`standard_chain`, `manufacturing_chain`), `MeshVolume` (`Volume`), "a Tying with explicit element maps" (simetri / desen / ekstrüzyon) |
| RF2 | PASS | 39 / 39 durumda hatasız, sonlu çıktı; MD1–MD5 ve KS1–KS5 geçti |
| RF3 | PASS | 2-B, 3-B ve boşluk / koruma / kısmi `frac`'lı 2-B `Domain` doğru; 2-B ve 3-B `MeshDomain`, `ShellDomain` yanlış |
| RF4 | PASS | `pytest -m baseline` 8 passed, en büyük bağıl fark 0,00e+00; dokunulan modüllerin test dosyaları + mimari / API yüzeyi / kütük: 525 passed, 6 xfailed (önceden var olan strict xfail'ler) |

Notlar. (1) RF3 satırı ön kayıtta "1-B `Domain`" de sayıyordu; `Domain` 1-B şekli kabul etmiyor (`len(shape) ∈ {2, 3}`),
o satır kurulamadığı için tabloda yok (kurulum düzeltmesi, eşik değil). (2) İlk iki koşuda dört RF1 durumu düştü
(ilk koşu `-x` ile ilk hatada durdu) ve düzeltildi: `StressRecovery`-kabuk testi `MeshElasticity`'yi bir `ShellMesh` ile kurmaya çalışıyordu (kurulum
hatası; kabukta `ShellPhysics` verilir, `domain` özniteliği olmadığından ret iletisi `ShellPhysics`'i adlandırır);
üç alanda `phase_report`: ret, çok fazlı enterpolasyon çağrısından sonra yazılmıştı (uygulama sırası; denetim fonksiyonun
başına alındı). Eşik değişmedi. (3) `Mass` / `Cost` ret denetimi fraksiyon çağrısından önce, yanıtın başında. (4)
`HelmholtzFilter` ön kayıttaki tek "sessizce yanlış" bileşendi; artık reddediyor. Ağ karşılığı olmayanlar için
(`PeriodicAngleFilter`, AM, döküm, çok fazlı, gerilme, bağlantılılık, kütle özellikleri, sağlam gerçekleşmeler)
ileti bunu açıkça söylüyor. Izgara fizikleri (`Elasticity`, `HeatConduction` …) kapsam dışı kaldı.

## Ek ön kayıt: ızgara FE fiziklerinin ızgara dışı alanda açık reddi (GF1–GF3)

Ön kayıt 2026-11-26 18:21 +0300, kod yazılmadan önce. RF1–RF4 halkaları ve yanıtları kapsadı, ızgara fiziklerini kapsam
dışı bıraktı. Bu adımda `fem` altındaki, bir `Domain` alıp ızgara özniteliklerini (`edof`, `h`, `node_coords`, `nn`,
`shape`) okuyan kurucular ve açık fonksiyonlar, ilk satırlarında `require_grid(domain, bileşen, alternatif)` çağırır.
Ağ karşılığı yalnız elastisitede var (`MeshElasticity`); `fem` katmanı `optimization`'ı içe aktarmaz, alternatif yalnız
iletide ad olarak geçer. Izgara alanında `require_grid` hiçbir şey yapmadan döner; ızgara davranışı bit düzeyinde aynı
kalmalı.

Reddedecekler: `Elasticity` ve `ElasticityGMG` (karşılık `MeshElasticity`); `TensorElasticity`, `Modal`,
`HeatConduction`, `ThermoElastic`, `TransientHeat`, `Newmark`, `NonlinearElasticity`, `MindlinPlate`, `GridSPD` ve
fonksiyonlar `element_nodes`, `assemble_conduction`, `nodal_source`, `hard_simply_supported`, `homogenize_elasticity`,
`homogenize_conduction`, `periodic_map` (karşılık yok). Ret, boyut ya da eleman denetimlerinden önce gelir (3-B ağ
`ElasticityGMG`'nin "3-D only" denetiminden, 2-B ağ `MindlinPlate`'in "2-D" denetiminden geçerdi). Kapsam dışı:
`ConductivityInterp` (yalnız `frac` okur, ağda çalışır), `LinearBuckling` (alan değil `Elasticity` modeli alır; başka
model türünü zaten `TypeError` ile reddediyor), `*Physics` sarmalayıcıları (`NewmarkPhysics`, `TransientHeatPhysics`,
`ThermoElasticPhysics`, `PlatePhysics`: reddeden modelleri alır), `mechanisms` iç denetimleri (yalnız reddeden
kuruculardan çağrılır), `fem.mesh` ve `fem.shell` (ağ / kabuk fizikleri).

| # | Kontrol | Eşik |
|:---|:---|:---|
| GF1 | yukarıdaki 18 bileşen, RF1'in üç ızgara dışı alanında (2-B Q4 `MeshDomain`, 3-B `MeshDomain`, `ShellDomain`) | `ValueError`; iletide bileşenin adı, "grid", alan türü; `Elasticity` / `ElasticityGMG` iletisinde `MeshElasticity`, diğerlerinde "no mesh or shell counterpart"; başka tür istisna 0 |
| GF2 | `ConductivityInterp` üç alanda; RF1–RF3, MD1–MD5, KS1–KS5 | hatasız, sonlu çıktı; hepsi geçer |
| GF3 | ızgara davranışı ve katmanlar | `pytest -m baseline` 8 passed bit düzeyinde; dokunulan `fem` modüllerinin test dosyaları, mimari ve API yüzeyi testleri geçer |

### GF1–GF3 sonucu (2026-11-26)

`tests/toporlogy/test_grid_refusals.py`'ye eklendi (`test_gf1_*` 18 bileşen × 3 alan, `test_gf2_*` 3 alan); dosya
202 passed. Her kurucu / fonksiyon ilk satırında `require_grid` çağırıyor (`TensorElasticity` ve `Modal` kendi
adlarıyla, iç `Elasticity`'den önce; `ElasticityGMG` ve `MindlinPlate` boyut denetiminden önce).

| # | Durum | Ölçülen |
|:---|:---|:---|
| GF1 | PASS | 54 / 54 durumda `ValueError`; iletide bileşen adı, "grid", alan türü; `Elasticity` ve `ElasticityGMG`'de "use MeshElasticity instead", diğer 16'da "it has no mesh or shell counterpart yet"; başka tür istisna yok |
| GF2 | PASS | `ConductivityInterp` üç alanda sonlu; RF1–RF3 (145), MD1–MD5, KS1–KS5 geçti |
| GF3 | PASS | `pytest -m baseline` 8 passed, en büyük bağıl fark 0,00e+00; dokunulan `fem` modüllerinin testleri + mimari / API yüzeyi / gradyan / ağ / kabuk: 436 passed, 2 skipped (pypardiso yok), 3 xfailed (önceden var olan strict xfail'ler); katalog / kütük / uyumluluk 131 passed; ruff ve mypy temiz |

Testler ilk koşuda geçti; kurulum düzeltmesi yok. API yüzeyi anlık görüntüsü değişmedi (imzalar aynı).

## Ek ön kayıt: genel ağ için YAML, ayrı kurulum yolu (AY1–AY6)

Ön kayıt 2026-11-26 18:28 +0300, kod yazılmadan önce. Karar (koordinatör): `domain.type: mesh` olan bir yapılandırma ayrı bir yoldan
kurulur, `configuration/mesh_build.py`; `resolve` ve `build` yalnız bir dağıtım satırıyla oraya yönlendirir, ızgara yolu
(`resolve._resolve`, `build.build`, `boundary/*`) değişmez. Yeni şema (en küçük kapsam):

* `domain: {type: mesh, generator: rectangle | box, n: [nx, ny(, nz)], size: [lx, ly(, lz)], element: Q4 | T3 | H8 |
  T4, perturb: 0.0, seed: 0}` → `perturb(rectangle | box(...), perturb, seed)` (perturb > 0 ise) → `MeshDomain`.
* `material: {E0, nu, plane: stress}`: birim modüllü D, 2-B'de düzlem gerilme, 3-B'de izotrop; E0 SIMP'e gider.
* `physics: elasticity` (ya da yok) → `MeshElasticity` (kalınlık 1).
* `supports: [{set: left, fix: [x, y]}]` (adlı düğüm kümesi: left right bottom top back front);
  `loads: [{set: right | point: [x, y], force: [fx, fy]}]`: kümede toplam kuvvet düğümlere eşit bölünür, `point`'te
  en yakın düğüme uygulanır; tek yük durumu.
* `chain: [{type: density_filter, rmin: <uzunluk>}]` → `Embed, MeshFilterLink, Passive` (embed / passive yoksa
  eklenir). **rmin burada uzunluk biriminde**; ızgara yolunda eleman biriminde (rmin_uzunluk = rmin_eleman · h).
  Zincir zorunlu (uzunluk biriminde makul bir varsayılan yok).
* `interpolation: simp` (penal, emin_ratio; varsayılanlar ızgara yolundaki gibi), `objective: compliance`,
  `constraints: [{response: volume, upper: V}]` → `MeshVolume`, `optimizer: mma` (ağ yolunda varsayılan mma; oc
  desteklenmiyor), `run: {max_iter, tol, seed, feasibility_tol}`, `output: {folder, snapshot_every}` (anlık
  görüntüler düz x̄ vektörü, ızgara varsaymaz).
* Geri kalan her şey (`load_cases`, `combinations`, `robust`, `connectivity`, `schedule`, boş olmayan `post`, `units`,
  başka zincir halkaları, başka yanıtlar / amaç / optimizer / fizik / enterpolasyon, `orthotropic`, `strength`,
  `phases`, `plane: strain`, `run.start`, `run.checkpoint_every`, `run.move_beta_ref`) anahtarı adlandıran ve ağ
  yolunun onu "henüz desteklemediğini" söyleyen bir `ConfigError`. Çıktı: `run_setup`'ın yazdığı her şey
  (`config.resolved.yaml`, `setup.json`, `history.jsonl`, `result.npz`, `result.json`, `summary.json`); VTU ve diğer
  son işlemler `post` reddiyle birlikte yok.

Örnek `examples/research/mesh_cantilever2d.yaml`: bozulmuş Q4 konsol (48 × 16, 3 × 1, h = 0,0625, perturb 0,01 (0,16 h), rmin 0,125 (2 h), sol kenar ankastre, [3, 0]'da [0, −1],
hacim ≤ 0,4, mma). Öngörü: `test_units.py` U1 anlık görüntüsü (`baseline/resolved_configs.json`) bütün `examples/**`
dosyalarını topladığı için yeni örnek U1'i `--write`'a kadar düşürür; `--write` sonrası dosyada **yalnız bir yeni
anahtar** (`examples/research/mesh_cantilever2d.yaml`) eklenir, var olan girdiler bayt bayt aynı kalır.
`test_research_examples.py` örnekleri kendiliğinden toplar ama `test_examples_are_found` adları sabit listeyle
karşılaştırır; listeye yeni ad eklenecek (bilinçli düzenleme). `test_baseline.py` yalnız `examples/*.yaml`'ı
topladığı için etkilenmez.

| # | Kontrol | Eşik |
|:---|:---|:---|
| AY1 | YAML ağ çalışması = aynı çalışma Python'da (`MeshDomain`, `MeshElasticity`, `MeshFilterLink`, `MeshVolume`, `SIMP`, `Compliance`, `MMA`), rastgele x'te değer, gradyan ve hacim; örnek (bozulmuş Q4), bozulmuş T3 dikdörtgen, H8 kutu | ≤ 1e-12 göreli |
| AY2 | bozulmamış Q4 dikdörtgende (24 × 8, 3 × 1, h = 0,125; rmin 0,25 uzunluk) YAML ağ çalışması = ızgara YAML konsolu (`box [24, 8]`, h 0,125, rmin 2 eleman, sol kenar ankastre, [max, min] noktasında [0, −1]): hücre ağırlık merkezleri = eleman merkezleri (sıra aynı); varsayılan başlangıçta ve rastgele x'te uyum ve gradyan; 10 MMA iterasyonunda amaç geçmişi | merkezler ≤ 1e-12 mutlak; değer, gradyan ≤ 1e-12 göreli; geçmiş her iterasyonda ≤ 1e-8 göreli (MD2 eşiği; MD2'de ölçülen 8,1e-9) |
| AY3 | örnek CLI'dan: `python -m toporlogy run examples/research/mesh_cantilever2d.yaml --out <klasör>` ve `validate` | `validate` çıkış 0; `run` çıkış kodu = 0 ise `status` ok, değilse 1; `summary.json`'da amaç ve kısıt değeri sonlu; son uyum < ilk uyum; hacim ≤ 0,4 (1 + 1e-3); `config.resolved.yaml` yeniden çözülünce kendisini verir ve aynı çalışmayı kurar (varsayılan başlangıçta amaç bit düzeyinde eşit) |
| AY4 | desteklenmeyen her anahtar (yukarıdaki liste; bilinmeyen küme adı, öğeye uymayan boyut, `n` / `size` uzunluğu yanlış, sıfır yük dâhil) | `ConfigError`; iletide anahtarın yolu ve "mesh"; desteklenmeyenlerde "not support"; başka tür istisna 0 |
| AY5 | ızgara davranışı | `pytest -m baseline` 8 passed bit düzeyinde; U1 öngörüsü (yukarıda) tutar; `test_research_examples.py`, `test_cli.py`, `test_units.py`, `test_architecture.py`, `test_api_surface.py`, katalog testleri geçer |
| AY6 | ızgara yolu dokunulmamış | `resolve.py` ve `build.py` farkı yalnız dağıtım satırları (ve içe aktarma); `boundary/*` farkı yok |

### AY1–AY6 sonucu (2026-11-26)

`src/toporlogy/configuration/mesh_build.py` (`is_mesh_config`, `resolve_mesh`, `build_mesh`); `resolve()` ve
`build()` birer dağıtım satırıyla oraya yönlendiriyor. Testler `tests/toporlogy/test_mesh_yaml.py` (AY1 3, AY2 1 +
1 sonradan, AY3 1, AY4 34). Ön kayıtta yazılmayan iki küçük ek: CLI `validate` ağ alanında ızgara satırları yerine
ağ satırlarını basıyor (yoksa `d.h` okurken düşüyordu) ve `run`'ın "problem" satırı eleman adını ağdan alıyor
(`execution/cli.py`); çözülmüş ağ yapılandırması CLI'ın okuduğu `schedule: null`'ı taşıyor.

| # | Durum | Ölçülen |
|:---|:---|:---|
| AY1 | PASS | örnek (bozulmuş Q4), bozulmuş T3 (24 × 8), H8 kutu (6 × 3 × 3): değer, gradyan, hacim ve hacim gradyanı farkı 0,00e+00 (üçünde de bit düzeyinde aynı); varsayılan başlangıçlar eşit |
| AY2 | **FAIL → strict xfail** | merkezler 0,00e+00; varsayılan başlangıçta değer 9,59e-13, gradyan **2,24e-12** (> 1e-12); rastgele x'te değer 1,97e-13, gradyan 5,79e-13; 10 MMA iterasyonu geçmişi **1,46e-8** (> 1e-8) |
| AY3 | PASS | `validate` çıkış 0; `run` çıkış 0, status ok, 101 / 200 iterasyonda yakınsadı; uyum 1908,3 → 308,84; hacim 0,400000; `result.npz` x̄ (768,); `config.resolved.yaml` yeniden çözülünce kendisini veriyor ve aynı çalışmayı kuruyor (varsayılan başlangıçta amaç bit düzeyinde eşit); `show` da çalışıyor |
| AY4 | PASS | 24 desteklenmeyen anahtar + 10 geçersiz giriş: 34 / 34 `ConfigError`, iletide yol ve "mesh"; desteklenmeyen 24'te "not supported by the mesh path (domain type mesh) yet"; başka tür istisna yok |
| AY5 | PASS | `pytest -m baseline` 8 passed, en büyük bağıl fark 0,00e+00; U1 öngörüsü tuttu: yeni örnek U1'i `--write`'a kadar düşürdü ("examples added or removed"), `--write` sonrası `resolved_configs.json`'da yalnız `examples/research/mesh_cantilever2d.yaml` eklendi (107 satır ekleme, 0 silme; diğer 15 girdi aynı); `test_mesh_yaml`, `test_grid_refusals`, `test_mesh_topology`, `test_shell_study`, `test_api_surface`, `test_catalog`, `test_catalog_ledger`, `test_compatibility`, `test_units`, `test_research_examples`, `test_cli`, `test_architecture`, `test_config`: 457 passed, 2 xfailed (AY2 ve önceden var olan U8); ruff ve mypy temiz |
| AY6 | PASS | `resolve.py` +2 / −1 satır (içe aktarma + dağıtım), `build.py` +3 satır (dağıtım); `boundary/*` farkı yok; `schema.py` yalnız belge satırı |

AY2 notu. Fark iki farklı doğrudan çözücünün (ızgara `Elasticity`: SuperLU `splu`; ağ `MeshElasticity`:
`fem.mesh` `Factorization`) yuvarlama düzeyinde: x = 0,4'te cond(K) = 3,0e4, eps · cond(K) = 6,8e-12. Sonradan
(post hoc) eklenen `test_ay2_posthoc_differences_are_within_the_solve_roundoff_bound`: aynı x'te değer farkı 4,86e-13,
gradyan farkı 1,49e-12, ikisi de eps · cond(K) altında (PASS). Ayrıca varsayılan başlangıçlar 1 ulp farklı (ızgara
`Domain.uniform_start` 0,4000000000000001, `MeshDomain` 0,4); bu fark tek başına değeri ~1e-16 oynatır, ölçülen
farkı açıklamaz. Eşikler değişmedi; kütük girdisi `docs/vv_kutugu.json` (risk low). Kurulum düzeltmesi: ilk
koşudaki AY2 testinde ön kayıtta olmayan bir "varsayılan başlangıçlar bit düzeyinde eşit" iddiası vardı ve bu 1 ulp
yüzünden ilk o düştü; çıkarıldı (ön kayıtlı ölçütlere dokunulmadı), ölçülen değerler yukarıda.

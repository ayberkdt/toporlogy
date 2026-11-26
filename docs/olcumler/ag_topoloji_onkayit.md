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

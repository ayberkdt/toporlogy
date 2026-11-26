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

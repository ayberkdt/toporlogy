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

# Periyodik yoğunluk filtresi — ön kayıt

Ön kayıt 2026-11-26 09:55 +0300, kod yazılmadan önce. Gerekçe: Study yanıtları `effective_stiffness` / `effective_poisson` hücreyi
periyodik sayar, ama `density_filter` hücre kenarında kesilir (`homojenlestirme_onkayit.md`, H15–H17 post hoc notu).

Değişiklik: `DensityFilter(domain, rmin, periodic=False)`; `periodic: true` ağırlıkları kutunun karşı yüzlerine sarar
(seyrek matrisde indeks mod şekil, evrişimde dairesel dolgu). Bütün elemanların aktif olmasını ister (yoksa açık
hata). YAML: `{type: density_filter, rmin: …, periodic: true}`.

| # | Kontrol | Eşik |
|:---|:---|:---|
| PF1 | 2-B ve 3-B tam kutuda, iki arka uçta (sparse, conv): sonuç = `scipy.ndimage.convolve(x, w/Σw, mode="wrap")` | ≤ 1e-14 mutlak |
| PF2 | Dairesel öteleme eşdeğerliği: F(roll(x)) = roll(F(x)) | ≤ 1e-14 mutlak |
| PF3 | Eşlenik: ⟨F x, y⟩ = ⟨x, Fᵀ y⟩ autograd geri adımıyla, iki arka uçta | ≤ 1e-12 göreli |
| PF4 | Aktif olmayan eleman (void ya da dış bölge) varken `periodic: true` | ValueError |
| PF5 | `periodic` verilmezse davranış değişmez | `pytest -m baseline` bit düzeyinde, çözülmüş yapılandırma anlık görüntüsü aynı |

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

## Sonuç (ilk ölçüm, 2026-11-26)

`tests/toporlogy/test_periodic_filter.py` (9 test). İlk koşuda PF3'ün dört durumu testin kendi hatasıyla düştü
(`torch.as_tensor` `requires_grad` almaz; ölçüm değil), `torch.tensor` ile düzeltildi, eşik aynı.

| # | Durum | Ölçülen (en kötü durum) |
|:---|:---|:---|
| PF1 | PASS | 4,4e-16 (3-B 6 × 5 × 4, rmin 1,8, sparse); 2-B 9 × 7, rmin 2,5: 2,2e-16 |
| PF2 | PASS | 2,2e-16 |
| PF3 | PASS | 8,6e-16 göreli |
| PF4 | PASS | void bölge ve dış bölge (frac 0) için ValueError |
| PF5 (anlık görüntü) | FAIL (ön kayıtlı ölçütün harfi) | çözülmüş yapılandırma anlık görüntüsü aynı kalmadı: 14 yapılandırmadaki 13 `density_filter` girdisine `"periodic": false` satırı eklendi; başka satır değişmedi (`git diff -U0` ile sayıldı). Varsayılan değerler çözülmüş yapılandırmaya yazıldığı için bu kaçınılmazdı; ön kayıtta öngörülmedi. Anlık görüntü yeniden yazıldı. |
| PF5 (baseline) | PASS | `pytest -m baseline`: 8 passed (örnekler bit düzeyinde aynı) |

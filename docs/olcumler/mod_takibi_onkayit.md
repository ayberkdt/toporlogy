# Mod takibi (MAC ve alt uzay): ön kayıt (MT1–MT6)

Ön kayıt 2026-11-26 04:24 +0300, kod yazılmadan ve hiçbir koşudan önce (05 §3: normalizasyon, işaret belirsizliği, tekrarlı özdeğer
ve değişmez alt uzay, mode tracking).

**Yöntem.** `optimization/mode_tracking.py`: M-ağırlıklı MAC_ij = (φ_iᵀ M ψ_j)² / ((φ_iᵀ M φ_i)(ψ_jᵀ M ψ_j));
eşleme MAC'i en büyükleyen atama (Macar yöntemi, `scipy.optimize.linear_sum_assignment`); özdeğerleri göreli
`cluster_tol` içinde olan modlar küme sayılır ve küme bir alt uzay olarak karşılaştırılır (M-ortonormal tabanlar
arasındaki en büyük ana açı). İşaret, eşleşen modun MAC işaretine göre hizalanır.

| # | Kontrol | Eşik |
|:---|:---|:---|
| MT1 | Rastgele M-ortonormal modlar, karıştırılmış sıra ve işaret: eşleme ve işaret geri bulunur; MAC köşegeni 1 | MAC ≥ 1 − 1e-12, atama tam |
| MT2 | Ölçek ve işaret değişmezliği (φ → −3φ) | MAC değişmez, ≤ 1e-14 |
| MT3 | Tekrarlı özdeğer: kümenin içinde rastgele dönme; tek tek MAC < 0,99 olabilir ama alt uzay açısı ≈ 0 | açı ≤ 1e-7 rad |
| MT4 | Gerçek modal örnek (kare kesitli konsol, `fem.modal`): iki yakın tasarım arasında ilk 6 mod takip edilir; ikili özdeğer kümesi tanınır | küme {1,2} ve {3,4} (ya da kayıtta ölçülen), alt uzay açısı ≤ 0,05 rad |
| MT5 | Negatif kontrol: ilgisiz (rastgele) modlarla MAC köşegeni küçük | max MAC < 0,5 |
| MT6 | Yanlış şekiller ve M pozitif tanımlı değil | açık `ValueError` |

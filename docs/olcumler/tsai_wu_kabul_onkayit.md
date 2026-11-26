# Tsai–Wu kabul edilebilirliği: ön kayıt (TW1–TW4)

Ön kayıt 2026-11-26 04:46 +0300, düzeltme yazılmadan önce (06 §1: kırılma ölçütlerinin kabul edilebilirliği ve türevlenebilirliği).

**Bulgu (düzeltmeden önce, `optimization/strength.py`).** Normal bileşenlerin kuadratik formu
diag(√F_ii) · [[1, f12, f12], [f12, 1, f12], [f12, f12, 1]] · diag(√F_ii); özdeğerleri 1 + 2 f12 ve 1 − f12 (çift).
3-B'de pozitif yarı tanımlı ancak −1/2 ≤ f12 < 1 iken; `OrthotropicStrength` ise −1 < f12 < 1 kabul ediyor. Ölçüm:
Xt = 1500, Xc = 1200, Yt = 50, Yc = 200, S12 = 70, σ_i = −100/√F_ii (i = 1, 2, 3) için indeks f12 = −0,4'te 20,15,
−0,5'te 0 (yüzey bu yönde açık: basınca duyarsız, von Mises gibi), **−0,8'de −103,2** (I ≥ 0 sözleşmesi bozuluyor;
tasarım sessizce "güvenli"). Düzlem gerilmede (σ3 = σ23 = σ13 = 0) yalnız 2 × 2 blok etkili ve |f12| < 1 yeterli.

**Düzeltme.** `Criterion.index` (tsai_wu): a = σᵀFσ < 0 olan bir gerilme açık `ValueError` (f12 ve 3-B
yarı tanımlılık sınırı adlandırılır). Düzlem gerilme durumları etkilenmez. Modül notuna f12 = −1/2'nin 3-B'de açık
yüzey olduğu yazılır.

| # | Kontrol | Eşik |
|:---|:---|:---|
| TW1 | f12 ∈ {−0,5; −0,2; 0; 0,5; 0,9}, 10⁴ rastgele 3-B gerilme: I ≥ 0 | hepsi |
| TW2 | f12 = −0,8: yukarıdaki 3-B gerilme | `ValueError`, mesajda "f12" |
| TW3 | f12 = −0,8, 10⁴ rastgele düzlem gerilme durumu (σ3 = σ23 = σ13 = 0) | hata yok, I ≥ 0 |
| TW4 | Mevcut `test_strength.py` ve gerilme testleri | değişmeden geçer |

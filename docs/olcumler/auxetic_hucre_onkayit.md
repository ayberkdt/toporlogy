# Negatif Poisson oranlı periyodik hücre (ters homojenleştirme, küçük örnek): ön kayıt (A1–A4)

Ön kayıt 2026-11-26 04:50 +0300, betik yazılmadan ve hiçbir koşudan önce (07 §1: "effective property target objective (ör. negative
Poisson ratio) ancak RVE doğrulandıktan sonra küçük örnek"; RVE H1–H9 `homojenlestirme_onkayit.md`).

**Problem.** 2-B düzlem gerilme, 30 × 30 Q4 periyodik hücre, ν0 = 0,3, SIMP p = 3, E_min = 1e-9 E0, periyodik
koni filtresi r = 1,5 eleman, amaç ν_eff = C12 / C11 en küçük, kısıtlar: hacim ≤ 0,40 ve C11 ≥ 0,02 E0 (bir taşıyıcı
yön kalsın). MMA (kütüphanenin), en çok 200 iterasyon, başlangıç: x = 0,4 + 0,2 cos(2π i/n) cos(2π j/n). Kod
`docs/olcumler/auxetic_hucre.py`; testler `tests/toporlogy/test_auxetic_cell.py` (yavaş: tam koşu).

| # | Kontrol | Eşik |
|:---|:---|:---|
| A1 | Amaç ve iki kısıtın x'e göre gradyanı (filtre + SIMP + homojenleştirme + oran) merkezi farkla, başlangıç noktasında | ≤ 1e-6 göreli |
| A2 | Koşu hatasız biter; döndürülen (gri) tasarımda kısıtlar sağlanır | hacim ≤ 0,40 + 1e-3, C11 ≥ 0,02 E0 (1e-3 göreli pay) |
| A3 | Döndürülen tasarımın 0,5 eşikli 0/1 hâli yeniden homojenleştirilince ν_eff | < 0 |
| A4 | 0/1 tasarım tek parça (yüz komşuluğu, periyodik: hücre kopyaları arasında bağlantı) ve C pozitif tanımlı | aynen |

## Sonuç (2026-11-26, ilk koşu)

`python -m pytest tests/toporlogy/test_auxetic_cell.py -q -p no:randomly -s -m "slow or not slow"` → 2 passed;
kayıt `auxetic_hucre.json` (`python docs/olcumler/auxetic_hucre.py`). A1–A4 PASS.

| | gri (döndürülen) | 0/1 (x̃ > 0,5) |
|:---|---:|---:|
| ν12 = C12/C11 | −0,951 | −1,010 |
| ν21 = C12/C22 | — | −0,831 |
| Hacim | 0,400 | 0,404 |
| C11 / E0 | 0,0200 (kısıt etkin) | 0,0279 |
| En küçük özdeğer / E0 | — | 1,13e-3 (C66) |

* 191 iterasyon; gri oranı M_nd = 0,354 (yüksek: β projeksiyonu yok). Hücre ortotrop (C11 ≠ C22), bu yüzden
  ν12 < −1 pozitif tanımlılıkla çelişmiyor (|C12| < √(C11 C22)).
* Bu bir gösterim örneğidir: 30 × 30 ağ, ağ/filtre duyarlılığı, gürbüzlük ve üretilebilirlik incelenmedi; gri
  tasarım ile 0/1 tasarımın ν değerleri farklı. Gerçek bir metamalzeme iddiası değildir.

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

# Doğrudan çözücüde eleman sırası ve sayısal kararlılık — ölçüm kaydı (25 Kasım 2026)

Kod: `src/toporlogy/solvers/ordering.py` (yeni), `solvers/direct.py`, `fem/elastic.py`, `fem/heat.py`.
Test: `tests/toporlogy/test_ordering.py`. Ölçüm betikleri oturumun geçici klasöründeydi; komutlar ve çıktıları
aşağıda olduğu gibi.

Etiketler `literatur_2026-11-21.md` ile aynı: **[Ö]** ölçüldü, **[O]** koddan okundu, **[H]** hafızadan.

## 0. Ön ölçüm: bugünkü doğrudan çözüm ne kadar doğru, zaman nereye gidiyor? [Ö]

Bu makinede pypardiso kurulu değil; doğrudan çözüm SuperLU (`splu`, `MMD_AT_PLUS_A`). Bit düzeyi örnek kayıtları da
bu yolla alınıyor (`test_baseline.py`). Problem: 3-B konsol, H8, x = 0'da ankastre, uçta düzgün düşey yük; tasarım
düzgünleştirilmiş rastgele alanın sigmoidi (çoğu 0/1'e yakın, bir kısmı gri), E = Emin + x³(1 − Emin).

| Izgara | DOF | Emin | Toplama | Çarpanlara ayırma | Dolgu | Göreli artık | Geri hata η | κ₁(K) | κ₁(Jacobi ölçekli K) | Bir iyileştirme adımının uyuma etkisi |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24×12×12 | 12 168 | 1e-6 | 0,014 s | 1,35 s | 12,5 | 2,0e-11 | 4,9e-17 | 1,6e8 | 1,4e6 | 9,6e-13 |
| 24×12×12 | 12 168 | 1e-9 | 0,018 s | 1,31 s | 12,5 | 1,9e-11 | 4,5e-17 | 8,6e10 | 1,5e6 | 3,3e-14 |
| 40×20×20 | 52 920 | 1e-6 | 0,065 s | 29,2 s | 24,3 | 4,0e-11 | 2,9e-18 | 1,3e8 | 3,2e6 | 2,8e-14 |
| 40×20×20 | 52 920 | 1e-9 | 0,062 s | 26,1 s | 24,3 | 4,0e-11 | 6,4e-21 | 6,7e10 | 3,8e6 | 0 |

Okuma:

- Doğrudan çözüm geriye doğru kararlı (η ≤ 5e-17). Bir adım yinelemeli iyileştirme uyumu en çok 1e-12 değiştiriyor;
  eklenmedi.
- κ(K)'nin büyüklüğü (1e8–1e11) neredeyse tamamen ölçekten: Jacobi ölçeklemesiyle 1e6'ya iniyor (van der Sluis).
  Cholesky benzeri simetrik eliminasyon köşegen ölçeklemeye karşı değişmez [H], dolayısıyla ölçekleme de eklenmedi.
- Zamanın neredeyse tamamı çarpanlara ayırmada (toplamanın 400 katı). Hızlanma ancak dolguyu azaltmakla gelir.
- SuperLU seçenekleri: `SymmetricMode` + `diag_pivot_thresh = 0` %10'dan az fark etti; `COLAMD` iki kat yavaş.

## 1. Ön kayıt (2026-11-25 19:43 +0300, testlerin ilk koşusundan önce)

Değişiklik: SPD sistemlerde (elastisite, ısı iletimi) SuperLU'ya geometrik iç içe bölme (nested dissection) sırası
verilir; SuperLU bu sırayı köşegen pivotlarla (simetrik eliminasyon) uygular. PARDISO kendi sıralamasını kullanmaya
devam eder. Modal analizdeki K − σM ve doğrusal olmayan tanjant indefinit olabildiği için bu yola girmez.

| # | Kontrol | Eşik |
|:---|:---|:---|
| O1 | `nested_dissection_nodes` her düğümü bir kez veriyor (2-B, 3-B, bir eleman genişliğinde ızgaralar, yaprak 1 / 64 / 10⁶) | tam |
| O2 | Ayırıcı özelliği: her bölmede iki yarı arasında ortak eleman yok (iki yarının düğümlerine birlikte dokunan eleman yok) | tam |
| O3 | Yeni bir ızgarada (3-B 32×16×16 ve 2-B 400×120) dolgu, MMD'den küçük | kesin küçük |
| O4 | Aynı problemde çözüm: iç içe bölme ile MMD | uyum ≤ 1e-10 göreli, ‖ΔU‖/‖U‖ ≤ 1e-7 |
| O5 | Kısmi alanlı (frac), keep/void ve simetri düzlemli bir problemde de aynı çözüm | O4'ün eşikleri |
| O6 | Uyum gradyanı (autograd) iki sırada aynı | ≤ 1e-9 göreli |
| O7 | Hızlı test paketi değişmeden geçiyor; örnek kayıtları (`test_baseline.py`) sayısal kipte REL_TOL içinde | geçer |

Hız bir test eşiği değil, kayıt: iç içe bölmenin bu makinedeki süresi raporlanır.

## 2. Sıralama karşılaştırması (ön kayıttan önce, keşif) [Ö]

İç içe bölme ile MMD, aynı matris; süreler paralel bir test koşusuyla aynı anda alındı (mutlak değerler gürültülü,
oran yol gösterici).

| Izgara | DOF | MMD | İç içe bölme (yaprak 64) | Hızlanma | Dolgu MMD → iç içe bölme | Uyum farkı |
|:---|---:|---:|---:|---:|:---|---:|
| 24×12×12 | 12 168 | 1,41 s | 1,18 s | 1,2× | 12,5 → 10,5 | 1,6e-12 |
| 40×20×20 | 52 920 | 33,2 s | 15,5 s | 2,1× | 24,3 → 18,0 | 2,0e-14 |
| 48×24×24 | 90 000 | 154 s | 54 s | 2,8× | 31,1 → 22,2 | 7,5e-15 |
| 300×100 (2-B) | 60 600 | 0,70 s | 0,64 s | 1,1× | 9,9 → 8,6 | — |
| 600×200 (2-B) | 241 200 | 8,13 s | 3,20 s | 2,5× | 15,2 → 10,2 | — |

Yaprak boyutu (en çok kaç düğümlük kutu bölünmez): 300×100'de 16 / 64 / 256 → 0,69 / 0,64 / 0,91 s;
40×20×20'de 16 / 64 / 256 → 16,0 / 15,5 / 19,8 s. 64 seçildi.

## 3. Sonuçlar (ilk koşu, `python -m pytest tests/toporlogy/test_ordering.py -s`) [Ö]

| # | Ölçülen | Eşik | Sonuç |
|:---|:---|:---|:---|
| O1 | 21 ızgara × yaprak birleşimi, her düğüm bir kez | tam | PASS |
| O2 | 21×11 ve 17×9×9: yarılar önce, ayırıcı son; Laplace çarpanında yarılar arası giriş 0 | tam | PASS |
| O3 | dolgu 2-B 400×120: MMD 12,84 → 9,03; 3-B 32×16×16: `slow` testte | kesin küçük | PASS (2-B) |
| O4 | 16×8×8: uyum 7,8e-12, U 8,1e-12; 120×40: uyum 5,2e-11, U 9,7e-11 | 1e-10 / 1e-7 | PASS |
| O5 | kısmi hacim + keep/void + eğik mesnet (`ConstrainedElasticity`): uyum 2,4e-16, U 7,7e-16 | 1e-10 / 1e-7 | PASS |
| O6 | uyum gradyanı 7,7e-14 | 1e-9 | PASS |
| O7 | yedi örnek, sayısal kip: en büyük fark 1,7e-10 (`multiload3d`); modal örnek bit düzeyinde aynı | REL_TOL 1e-6 | PASS |

O7'den sonra bit düzeyi kayıt (`baseline/examples.json`) yeni sırayla yeniden yazıldı: değişiklik kasıtlı, sayısal
fark yukarıda.

Bu turda yapılmayanlar: PDE filtresi (`design/pde_filter.py`, eleman ızgarasında SPD), burkulmanın ön yüklü
çarpanlara ayırması (`fem/buckling.py`, SuperLU varsayılan COLAMD) ve GMG'nin kaba ızgara doğrudan çözümü aynı
sıralamadan yararlanabilir; ölçülmedi.

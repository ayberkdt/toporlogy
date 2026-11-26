# Burkulma güvenilirliği ve geometri aktarımı: ön kayıt (06 §2, §3, §4'ün ilgili kısmı)

Ön kayıt 2026-11-26 05:03 +0300 (`date` çıktısı), kod yazılmadan ve aşağıdaki ölçümlerin hiçbiri koşulmadan önce.
Taban `382ebaa` (`integrate/research-round-1`'in bir önceki ucu; sonraki tek commit `853b1e4` başka bir konunun ön
kaydı). Sonuç: `burkulma_geometri.md`, ham veri `burkulma_geometri.json`, betik `burkulma_geometri.py`.

Bu dosya yazılırken bilinen sayılar (ölçüm değil, önceki kayıtlar): Euler kolonu Q4 hataları 0,123 / 0,030 / 0,006
(2 / 4 / 8 eleman genişlik, `test_buckling.py`); tekil duvar λ1 2,85 / 2,11 / 1,60 / 1,20 (40² … 320²,
`malzeme_modelleri.md`); yerelleşme 0,88 / 0,92 / 0,95. Eşikler bunlara bakılarak seçildi; yeni koşu bu sayıları
yeniden ölçer.

## Neyi değiştiriyorum (kod)

* `optimization/buckling.py`: her değerlendirmede **güvenilirlik kaydı**. `reliable = False` şu durumlardan biri
  varsa: küme kesimi (`cluster_cut`: toplanan son mod ile sonraki bir küme; KS türevlenemez), sahte mod
  (`spurious_low_density`: 1. modun K-enerjisinin ≥ 0,5'i x̄ < 0,5 elemanlarda), burkulmayan mod (μ1 ≤ 0).
  Tek ağda doğrulanamayan durum ayrı bayrak: `localized` (1. modun enerjisinin ≥ 0,5'i elemanların %1'inde: ağ
  inceltmesiyle bakılmalı) — `reliable`'ı düşürmez ama "doğrulandı" anlamına gelen bir alan da yoktur:
  `reliable = True` yalnız "bayrak yok" demektir. Mod takibi: ardışık değerlendirmeler arasında
  `optimization/mode_tracking.track_modes` (K-ağırlıklı MAC, küme alt uzayı); kopya yok.
* `fem/buckling.py`: gerilme rijitliği enerjisinin düşük yoğunluk payı (`stress_energy_share`; K_G'nin itici,
  basınç kısmının x̄ < eşik elemanlardaki payı), K-enerjisi payının (`low_density_share`) yanına.
* `post/pipeline.py`: `finalize(..., keep_fields=True)` ara alanları saklar (eşik, SDF, morfoloji, son alan);
  varsayılan davranış ve rapor değişmez.
* Yeni `post/chain.py`: yoğunluk → eşik → SDF → yumuşatma → yüzey → (hacim ağı) → yeniden analiz zincirinin her
  adımı için ayrı kayıt (hacim, bağlantı, su geçirmezlik, en küçük kalınlık, CAD toleransı) ve durum
  `PASS` / `FAIL` / `UNVERIFIED` / `NOT VERIFIED`. Geometri kararı ile yapısal karar ayrı; birleşik PASS yok.
* Yeni `post/reanalysis.py`: son örtük geometrinin ince voksel (merdiven) ağında yeniden analizi; aynı ağda
  scikit-fem (`ElementHex1`) ile bağımsız montaj. Merdiven sınırı gövdeye uyumlu değildir; bu T10'un yerini
  tutmaz (T10 için gmsh gerekir, bu makinede yok → NOT VERIFIED).

## A. Ağ inceltme serileri (burkulma)

Yakınsama kararı `post.verify.assess_convergence` (varsayılanlar: en az 3 ağ, inceltme oranı ≥ 1,3, monoton, mertebe
0,5–6, ince ağ GCI ≤ %5, 4+ ağda mertebe farkı ≤ %30).

| # | Problem | Kontrol | Eşik |
|:--|:--|:--|:--|
| A1 | Euler kolonu (ankastre-serbest), b = 4, L = 160, E = 1000, ν = 0, Q4; 2, 4, 8, 16 eleman genişlik | karar; görünür mertebe p; Richardson değeri / Timoshenko (P_T = P_e/(1 + P_e/(κGA)), κ = 5/6) | CONVERGED; 1,5 ≤ p ≤ 2,5; \|λ_ext/P_T − 1\| ≤ 1e-3 |
| A2 | Aynı kolon, Q4IM, 16 eleman genişlik | \|λ/P_T − 1\| | ≤ 1e-3 |
| A3 | Basınç elemanı: b = 4, L = 80 kolon, 5b genişlikte tasarım alanının ortasında; çevresi boş (x = 0), E_min = 1e-6, E_G = E0 x³ (alt sınırsız); alt kenarın tamamı ankastre; 2, 4, 8, 16 eleman kolon genişliği | her ağda λ_gömülü / λ_tek (aynı ağda yalnız kolon); seri kararı | \|oran − 1\| ≤ 1e-3; CONVERGED |
| A4 | Tekil duvar (Ferrari, Sigmund & Guest 2021 §5.2 kurulumu, `test_buckling_physics._wall`); n = 40, 80, 160, 320 | karar; yerelleşme | NOT CONVERGED (beklenen: tekil); yerelleşme kesin artan |
| A5 | Duvar, sıkıştırma ucu çevresinde r = 0,05 dışlama (`exclude`); n = 40, 80, 160 | karar | NOT CONVERGED (tahmin: mod açıklık köşesine taşınır, o da tekil) |
| A6 | Duvar 80², dışlama yarıçapı r = 0,025 / 0,05 / 0,1 | λ1(0,1)/λ1(0,025) − 1 | ≥ 0,10 (sonuç bölge boyutuna bağlı: dışlama çözüm değil) |

## B. Sahte (düşük yoğunluk) mod teşhisi ve azaltma yöntemleri

Gerilme modülü yöntemleri: `stiffness` (E_G = E, E_min dahil), `pg=1`, `pg=2`, `pg=3`, `pg=4` (E_G = E0 x̄^pg, alt
sınırsız; kütüphane varsayılanı pg = 3 = p). Rijitlik her yerde SIMP p = 3, E_min ∈ {1e-3, 1e-6, 1e-9}.

Tasarımlar: D1 = A3'ün nb = 4 gömülü kolonu (0/1, referans: tek başına kolon). D2 = `test_buckling.py`
`test_constraint_raises_the_load_factor` kurulumunun yalnız uyum + hacim (0,35) çözümü (16 × 48, MMA 60 iterasyon,
E_min 1e-6, filtre 2,0): gri sınırlı gerçek bir tasarım.

| # | Kontrol | Eşik |
|:--|:--|:--|
| SP1 | D1, E_min ≤ 1e-6, her yöntem: \|λ1/λ_tek − 1\| | ≤ 1e-3 |
| SP2 | D2, pg = 1, E_min = 1e-6 (pozitif kontrol: E_G/E = x̄^-2 düşük yoğunlukta büyür): 1. modun K-enerjisi düşük yoğunluk payı ve λ1(pg=1)/λ1(pg=3) | pay ≥ 0,5 ve oran ≤ 0,5 |
| SP3 | D2, pg ∈ {3, 4}, her E_min: 1. modun düşük yoğunluk payı | < 0,5 |
| SP4 | D2, bütün yöntem × E_min: λ1, iki pay, yerelleşme | betimsel (eşik yok) |

## C. Güvenilirlik işareti (küme kesimi) ve mod takibi

Kare kesitli 3-B kolon (3 × 3 × 24, h = 0,5, x = 1, `standard_chain(d, 1.2)`, SIMP E0 = 50): λ1 = λ2.

| # | Kontrol | Eşik |
|:--|:--|:--|
| RL1 | n_modes = 1 → `reliable` False ve `cluster_cut` bayrağı; n_modes = 2 → `reliable` True | tam |
| RL2 | Tek taraflı türevler D± (Richardson, ±h, ±2h, h = 1e-4), 4 rastgele yön (tohum 0). n_modes = 1 (μ1, kesik küme): max_d \|D+ − D−\| / max(\|D+\|, \|D−\|) | ≥ 1e-2 (türevlenemez: kesik sonuç güvenilir değil) |
| RL2b | n_modes = 2 KS (ρ = 20, s = 1): max_d \|D+ − D−\|/\|D+\| ve AD ile merkezî Richardson FD farkı | ≤ 1e-5 ve ≤ 1e-6 |
| RL3 | Aynı tasarımın yeniden değerlendirilmesi: mod takibi birim sıra, MAC ≥ 1 − 1e-8 | tam |
| RL4 | D2 kurulumunda burkulma kısıtlı 60 iterasyonluk koşu (test_constraint'in ikinci koşusu): iterasyonlar arası mod değişimi sayısı, takip edilen 1. modun en küçük MAC'i | betimsel |

Doğrusal olmayan burkulma (kusurlu geometri, burkulma sonrası yol): 04'ün `solvers/continuation` modülü bu tabanda
yok (`claude/nonlinear-solvers-2d3dd3` dalında) → NOT VERIFIED, bu turda yapılmaz.

## D. Geometri zinciri (3-B)

GC: analitik basamaklı çubuk. Alan 32 × 12 × 12, h = 1; katı: x ∈ [0, 32], y, z ∈ [2, 10] (8 × 8), yalnız
x ∈ [12, 20]'de y ∈ [4, 8] (4 × 8 boyun). Tam hacim 1792, tek gövde, boyun kalınlığı 4, zarf = alan kutusu.
`finalize` varsayılanları (h_f = 0,25).

| # | Kontrol | Eşik |
|:--|:--|:--|
| GC1 | Hacim: yoğunluk adımı / eşik adımı / yüzey | 1792 (≤ 1e-9 göreli) / ≤ 5e-3 / ≤ 3e-2 |
| GC2 | Her ızgara adımında ve yüzeyde gövde = 1, kapalı boşluk = 0 | tam |
| GC3 | Yüzey: açık, manifold olmayan, ters yönlü kenar = 0 | tam |
| GC4 | Boyun orta hattında (x ∈ [14, 18], y = z = 6) yerel kalınlık, eşik ve son alan adımında | 4 ± 0,5 |
| GC5 | CAD toleransı: son yüzey düğümlerinin zarf dışına en büyük taşması | ≤ 0,05 h_f |
| GC6 | Değerlendirilemeyen kısıtlar (kendi kendini kesme, simetri, delik koruma) | `UNVERIFIED`; geometri kararı bunlar gerekli sayılırsa PASS olamaz |
| GC7 | Aynı voksel ağında (h = 0,5) kütüphane H8 ile scikit-fem `ElementHex1` uyumu | göreli fark ≤ 1e-8 |
| GC8 | Son geometrinin merdiven ağlarında (h = 1/2, 1/3, 1/4) eksenel çekme uyumu (ν = 0, P = 1, E = 1, x = 0 ankastre, x = 32'de düzgün çekme); nominal kesitlerin seri çubuk değeri C_a = 24/64 + 8/32 = 0,625 | en ince ağ \|C/C_a − 1\| ≤ 0,05; seri kararı betimsel (merdiven sınırı geometri hatası taşır) |
| GC9 | T10 yeniden analiz | gmsh yok → `NOT VERIFIED` (PASS değil) |

Gerçek tasarım: `test_post_cantilever.py`'nin 30 × 10 × 10 konsolu (OC, 40 iterasyon) → `finalize` → zincir
kaydı → merdiven yeniden analizi (h = 1/2, 1/3, 1/4): betimsel; `volume_match="threshold"` ile `"final"`
karşılaştırması (her ikisinin hacmi hedefin %1'i içinde olmalı: VC3).

## E. Hacim düzeltmesi gerilme ve burkulmayı garanti etmez

D2'den eşit hacimli değişkenler: V0 gri (optimum), V1 hacim koruyan eşik (0/1), V2 ve V3 Gauss bulanıklığı
(σ = 1 ve 2 eleman) + hacim koruyan eşik. Her biri için uyum, λ1 (pg = 3), katı elemanlarda (x ≥ 0,5) merkez
von Mises'in en büyüğü (katı malzeme gerilmesi), gövde sayısı.

| # | Kontrol | Eşik |
|:--|:--|:--|
| VC1 | Hacimleri birbirinin %0,5'i içinde olan iki değişken arasında λ1 ya da en büyük von Mises farkı | en az bir çiftte ≥ %5 |
| VC2 | Uyum farkları | betimsel |
| VC3 | 3-B konsolda iki hacim eşleme seçeneğinin son hacmi / hedef | ikisi de \|· − 1\| ≤ 0,01 |

Not: en büyük von Mises tek ağda ölçülür; içbükey köşelerde ağa bağlıdır (06 §1, `gerilme.md`). VC1 yalnız aynı ağda
değişkenler arası farkı söyler, yakınsamış gerilme iddiası değildir.

## Başarısızlık kuralı

Bir eşik tutmazsa ilk ölçülen değer kaydedilir, test `strict xfail` olur, eşik gevşetilmez. Sonradan eklenen her
analiz "post-hoc" etiketiyle ayrı yazılır.

# PR-11 — Simetrik bağ yerleşimi: braket ölçümü (24 Kasım 2026)

Kod: `design.manufacturing.mirror_closure`, `optimization.connectivity.one_body(..., symmetry=)`, YAML'da
`connectivity.symmetry` (verilmezse zincirdeki `symmetry` bağlantısının düzlemleri). Testler:
`tests/toporlogy/test_symmetric_ties.py`. Betik: `docs/olcumler/simetrik_baglar.py`, ham kayıt:
`docs/olcumler/simetrik_baglar.json`. Bu dosyadaki her sayı o kayıttan.

## 0. Özet

Aynalı bağlar keep maskesini simetrik yapıyor (x = 500'de e_sym(keep) 0,224 → 0). Tasarımı simetrik yapmıyor:
e_sym(x̄) tek taraflı bağda 0,419, aynalı bağda 0,454, bağsız 0,275. Asimetrinin ana kaynağı bağ değil, dışbükey
olmayan problem. Tam simetri yalnız `SymmetryTying` ile elde ediliyor (e_sym(x̄) 2e-7). Bedeli uyumda %1,5;
aynalı bağların bedeli %2,2. Yan bulgu: aynalı bağlarla bütün arayüzler dondurulmuş hücrelerle birbirine
bağlanıyor ve kapının en dar boyun ölçüsü tanımsız kalıyor (36 çiftin 36'sı kesilemez).

## 1. Deney

Braket (`studies/final_2026/ansys/toporlogy_2026-11-18/official_bracket.yaml`; 91 × 48 × 41 H8, h = 7,5 mm,
3 yük durumu, MMA, β sürekliliği, 150 iterasyon, hacim ≤ 0,33, bağlantı kısıtı ≤ 20). Dört kol, YAML'ın geri
kalanı aynı:

| Kol | Değişiklik |
|---|---|
| r: tek taraflı bağ | yok (bugünkü kod, `ties: 1`) |
| (a) bağsız | `connectivity.ties: null` |
| (b) aynalı bağ | `connectivity.symmetry: {x: 500}` |
| (c) `SymmetryTying` | zincirin başına `{type: symmetry, planes: {x: 500}}` |

Komutlar:

```
python docs/olcumler/simetrik_baglar.py --configs <klasör>
python -m toporlogy run <klasör>/<kol>.yaml --out <klasör>/<kol>
python docs/olcumler/simetrik_baglar.py <klasör>
```

18 Kasım'da saklanan koşu (`ref_stored`) karşılaştırmaya alınmadı. O koşu, bağ yollarını değiştiren 29e2ac5'ten
önce optimize edildi. Bugünkü kodla kurulan keep hücrelerinden 23'ü onun tasarımında boş. Bu yüzden r kolu
bugünkü kodla yeniden koşuldu. Saklanan koşunun satırı kayıtta duruyor.

## 2. Sonuçlar

| Kol | Uyum | Karar | KKT | Bağlantı kısıtı | Aşınmış x̄'de gövde | En dar boyun (hücre) | Kesilemez çift | keep hücresi | e_sym(keep) | e_sym(x̄) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r: tek taraflı bağ | 60 000 | PASS | 0,153 | 19,10 | 21 | 65 | 32 / 36 | 4 542 | 0,224 | 0,419 |
| (a) bağsız | 60 011 | PASS | 0,159 | 19,14 | 5 | 28 | 10 / 36 | 4 428 | 0 | 0,275 |
| (b) aynalı bağ | 61 329 | PASS | 0,159 | 14,74 | 1 | — | 36 / 36 | 4 656 | 0 | 0,454 |
| (c) `SymmetryTying` | 60 882 | PASS | 0,058 | 16,70 | 1 | — | 36 / 36 | 4 656 | 0 | 2e-7 |

Dört kolun hepsi 150 iterasyon sınırında durdu ("not converged"). Karşılaştırma eşit iterasyon bütçesinde
yapıldı. Hepsi hacim kısıtında (0,33), en büyük kısıt ihlali 0 ve 0,5 eşiğinde tek gövde. 0,3 ve 0,7 eşiklerinde
de tek gövde. Süreler karşılaştırılamaz: (b) ve (c) aynı anda koşuldu (4 702 ve 4 564 s), r ile (a) tek başına
koşuldu (1 428 ve 1 833 s).

- **Keep simetrisi.** Tek taraflı bağ keep'e 114 hücre ekliyor (4 542 − 4 428), aynalı bağ 228. e_sym(keep)
  0,224'ten 0'a iniyor. §2.3'teki 0,224 bugünkü kodla da üretiliyor.
- **Tasarım simetrisi.** Simetrik keep tasarımı simetrik yapmıyor: (a) 0,275, (b) 0,454. Tek taraflı bağın
  0,419'u (b)'den küçük. Bağı aynalamak x̄'nin asimetrisini azaltmadı.
- **Uyum.** r ile (a) aynı düzeyde (60 000 / 60 011). (b) %2,2, (c) %1,5 daha yüksek. (c)'nin KKT artığı en
  küçük olanı (0,058, ötekiler 0,15–0,16).
- **Aşınmaya dayanıklılık.** Aşındırılmış x̄'de r 21, (a) 5 gövdeye ayrılıyor; (b) ve (c) tek gövde kalıyor. r ve
  (a)'da da bütün arayüzler aynı gövdede kalıyor, ayrılanlar arayüze değmeyen parçalar. Kapı FRAGILE kararını
  arayüzlerin ayrılmasına göre veriyor, gövde sayısına göre değil (`connectivity.py`, `fragile`); dördüne de PASS.
- **Boyun ölçüsü.** `min_cut_cells` yalnız tasarlanabilir hücreleri kesebiliyor. Bir arayüz çifti dondurulmuş
  hücrelerle (keep, bağ, arayüz pedleri) birleşiyorsa çiftin boyunu yok sayılıyor. (b) ve (c)'de dondurulmuş katı
  hücreler tek bileşen ve 9 arayüzün 36 çiftinin hiçbiri kesilemiyor, bu yüzden "en dar boyun" yazılmıyor. r'de de
  36 çiftin 32'si kesilemiyor; raporlanan 65 hücre kalan 4 çiftten geliyor. Bu kollarda boyun ölçüsü bağlantının
  sağlamlığı hakkında bir şey söylemiyor.

## 3. Karar için (yol haritası Y2)

Ölçüm Y2'deki "simetrik problem algılanınca bağlar her zaman simetrik" seçeneğini desteklemiyor. Aynalı bağ
uyumu %2,2 artırıyor ve tasarımı simetrik yapmıyor. Kazancı, keep'in simetrik olması ve aşındırılmış tasarımın
tek gövde kalması. Tasarımın simetrik olması isteniyorsa yol `SymmetryTying` (bu problemde %1,5). Bu yüzden
`connectivity.symmetry` isteğe bağlı kalıyor, varsayılan değişmedi. Karar kullanıcının.

## 4. Sınırlar

- Tek problem, tek düzlem (x = 500), tek tohum. Yerel en iyiler arasındaki %2'lik farklar başka başlangıçta
  değişebilir. Bu sınanmadı.
- 150 iterasyonda hiçbir kol yakınsamadı (KKT 0,06–0,16).
- y düzlemi ölçülmedi: tasarım alanı y'de simetrik değil (§2.3).

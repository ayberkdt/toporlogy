# R-01 ön kaydı: mekaniği koruyan geometri sonlandırma (PR-12)

Yazıldı: 24 Kasım 2026, deneyden önce. Bu dosya koşudan sonra değiştirilmez; sapmalar sonuç dosyasında
(`r01.md`) "Ön kayıttan sapmalar" başlığı altında yazılır. Kod: `docs/olcumler/r01.py`, ham kayıt: `r01.json`.

## 1. Soru ve hipotez

Yoğunluk alanından son geometriye geçişte (eşik, yumuşatma, kesin arayüzler) parçanın rijitliği değişiyor.
`docs/olcumler/yontemler.md` §3'te gri → 0/1 geçişte gerilme aşımı %2,5–71 ölçüldü (tek problem). Bugünkü
`finalize` eşik seviyesini hacmi tutturacak biçimde seçiyor; mekanik büyüklüğü hiç tutturmuyor.

**H1.** Eşik seviyesini hacim yerine uyumu (compliance) tutturacak biçimde seçen sonlandırma (M1), son
geometrinin bağımsız T10 uyumunu gri tasarımın uyumuna temel yöntemlerden daha yakın getirir.

## 2. Problem kümesi (sabit, üç problem)

| Kod | Örnek | Neden |
|---|---|---|
| P1 | `examples/cantilever3d.yaml` (kutu 24×8×4, 150 iterasyon) | düz sınırlar, tek yük |
| P2 | `examples/symmetric_am3d.yaml` (kutu 24×8×12, 60 iterasyon) | daha kalın, simetrik |
| P3 | `examples/lbracket_bore3d.yaml` (CSG, delik, robust, 40 iterasyon) | eğri sınır, kesin delik |

Her problem kendi YAML'ıyla, değiştirilmeden optimize edilir; son işlem adımları kapatılır ve aşağıdaki
yöntemler aynı x̄ üzerinde uygulanır. Ince ızgara: `upsample = 3` (hepsinde).

## 3. Yöntemler

Hepsi `toporlogy.post.finalize` ile, aynı x̄ ve aynı kesin arayüzlerle (P3'te delik):

* **B1 eşik + marching cubes:** seviye 0,5, hacim eşleme yok, açma/kapama yok, yumuşatma yok.
* **B2 bugünkü finalize:** varsayılan `PostParams` (hacim eşlemeli seviye, açma, kapama, Taubin).
* **B3 yalnız hacim düzeltmeli:** hacim eşlemeli seviye, açma/kapama yok, yumuşatma yok.
* **M1 uyum eşlemeli seviye (prototip):** seviye t, ince ızgaraya üç doğrusal büyütülmüş x̄'nin {x̄ > t} bölgesinin
  eleman başına hacim oranı ile kurulan H8 modelinin (E = E0 · oran) uyumu gri uyuma eşit olacak biçimde ikiye
  bölmeyle seçilir; sonra `finalize` bu seviyeyle (hacim eşleme yok), varsayılan açma/kapama ve yumuşatmayla.
* **M1-ham (ablasyon):** M1'in seviyesi, açma/kapama ve yumuşatma olmadan. Seviye seçimini yumuşatmadan ayırır.

## 4. Ölçüm

* **Gri uyum C_g:** x̄ üzerinde SIMP (p = 3, E_min = 1e-9 E0) H8 voksel modeli, son işlemin T10 modeliyle aynı
  sınır koşulu eşlemesiyle (`configuration.boundary_mapping.bc_mapping`, `post.verify.voxel_reference`).
* **Son geometri uyumu C_T:** `post.verify.mesh_convergence` (PR-07) ile T10, üç ağ: tet boyu
  1,0 h / 0,65 h / 0,42 h, katı E0. CONVERGED ise Richardson değeri, değilse en ince ağın değeri kullanılır ve
  "yakınsamadı" diye işaretlenir.
* **Birincil ölçüt:** d = |ln(C_T / C_g)| (problem ve yöntem başına).
* **İkincil:** son hacmin optimize edilmiş hacme oranı; T10 yakınsama kararı; süre.

## 5. Karar kuralı (önceden yazıldı)

H1 **desteklenir**: M1'in d değeri üç problemin en az ikisinde B1, B2 ve B3'ün en iyisinden küçükse VE M1'in
ortanca d'si en iyi temel yöntemin ortanca d'sinden en az %25 küçükse.
H1 **reddedilir**: M1 üç problemin en az ikisinde en iyi temel yöntemden kötüyse.
Arada kalan durum **kararsız** olarak yazılır. Yakınsamayan T10 değerleri sonuçta ayrıca işaretlenir; bir
problemde M1'in ya da en iyi temel yöntemin T10'u yakınsamadıysa o problem karar sayımında "belirsiz" sayılır.

## 6. Bilinen sınırlar (koşudan önce)

* M1'in yerine koyduğu model (eleman başına hacim oranı, doğrusal E) T10'un değil, kaba bir voksel modelinin
  uyumunu tutturuyor; aradaki fark (kısmi hacim modeli ile gerçek geometri) yöntemin kendi hatasıdır.
* M1 hacmi serbest bırakır: uyumu tutturmak hacim kısıtını aşabilir. Hacim oranı ikincil ölçüt olarak raporlanır;
  hacmi %5'ten fazla aşan sonuç "kısıtı çiğnedi" diye işaretlenir.
* Üç problem bir genelleme değildir; sonuç yalnız bu kümeyi söyler.
* Gerilme bu ön kayıtta yok (yalnız uyum).

# Öneri: genel ağın `Study` / YAML'a girmesi ve kabuk elemanı

Durum: öneri, karar bekliyor. Kod değişikliği yok. Dayanak: bu turda okunan kod (aşağıda dosya adlarıyla);
"denetlenmedi" diye işaretli maddeler okunmadı.

## Bugün ızgaraya bağlı olan yerler (okundu)

* `optimization/study.py`: `Study` bir `Domain` alır; `domain.active_elements`, `domain.n_design`,
  `domain.uniform_start(...)` (başlangıç), yük satırı denetimi (`state_dofs` yoksa `domain.ndof`).
* `design/chain.py`: `Embed` (`domain.design_elements`, `domain.keep`, ızgara şekli), `DensityFilter` (ızgara
  evrişimi / ızgara ofsetleriyle seyrek matris), `Passive`.
* `configuration/build.py`: mesnet ve yükler `Domain` düğümlerinde seçicilerle kurulur (`bcs.F("all")`, 2-3 DOF/düğüm),
  rigid-body denetimi `_check_bc`.
* Genel ağ tarafı hazır: `fem.mesh` (`Mesh`, `ElasticProblem`, etiketli yüzey/düğüm kümeleri, `Mesh.from_domain`),
  ilk optimizasyon dilimi `optimization/mesh_topology.py` (`MeshDensityFilter`, `MeshCompliance`;
  `ag_topoloji_onkayit.md`, MS1–MS6).

## Önerilen mimari

1. **Tasarım uzayı arayüzü.** `Study`'nin `Domain`'den kullandığı küçük küme bir protokol olur: `n_design`,
   `active_elements` (ya da tüm hücreler), hücre hacimleri, ağırlık merkezleri, başlangıç tasarımı. `Domain` bunu
   bugünkü davranışıyla sağlar (bit düzeyinde aynı), yeni `MeshDomain(mesh)` aynı protokolü ağ hücreleriyle sağlar.
2. **Ağ zinciri halkaları.** `MeshEmbed` (keep / void hücre etiketleriyle) ve `MeshDensityFilter`'ın torch halkası
   (`mesh_topology.py`'deki seyrek matris + `_SparseApply` benzeri autograd). Izgara halkaları değişmez;
   `validate_chain` halka ile alan türünü eşleştirir.
3. **Ağ fiziği.** `MeshElasticity`: `solve(E, F)` → U, autograd ile (K(E) = Σ E_e K_e; `MeshCompliance` ile aynı
   çözüm), `state_dofs` = ağın DOF sayısı. Böylece `Compliance`, `Volume` (hücre hacmiyle ağırlıklı) ve
   gradyan doğrulaması aynı kalır.
4. **YAML.** `domain: {type: mesh, generator: rectangle | box, …}` ya da `{type: mesh, file: …}` (gmsh okuyucu bu
   makinede yok: NOT VERIFIED); mesnet ve yük seçicileri ağ etiketleriyle (`where: {type: tag, name: left}`).
   Izgaraya özgü bileşenler (AM filtresi, simetri bağlama, bağlantılılık, gerilme kurtarma) ağ alanında açık hata
   verir; katalog bunu kural olarak taşır.
5. **Doğrulama zinciri (ön kayıt önerisi).** `Mesh.from_domain` ağında ağ yolu = ızgara yolu (değer, gradyan,
   birkaç MMA iterasyonu bit düzeyine yakın); bozulmuş Q4 / T3 / T4 ağlarında gradyan merkezi farkla; YAML ağ
   örneği uçtan uca; ızgara örnekleri bit düzeyinde (baseline).

Tahmini kapsam: `Study` ve `build` arayüzünde orta büyüklükte değişiklik; halka ve fizik yeni dosyalarda. Hangi
yanıtların gizlice ızgara varsaydığı tek tek denetlenmedi (gerilme, uzunluk ölçeği, bağlantılılık muhtemelen
varsayıyor: denetlenmedi).

## Kabuk elemanı

Düz ızgarada kabuk = plaka + zar, ikisi birbirinden ayrılır (`fem.plate` + düzlem gerilme); anlamlı bir kabuk eğri
geometri ister. Öneri: 3-B yüzey ağında (T3) düz parçalı kabuk — zar için CST, eğilme için MITC3 ya da DKT, yerel
eksenlerden global eksenlere dönüşüm, düğüm başına 6 DOF (dönme serbestliğinin sayısal kararlılığı için küçük
"drilling" rijitliği). Doğrulama: Scordelis–Lo çatısı ve kıstırılmış silindir gibi kabuk benchmark'ları (kaynakları bu
oturumda okunmadı: NOT VERIFIED; ön kayıtta kaynaktan alınmalı). Ön koşul: yukarıdaki `MeshDomain`.

## Karar gereken sorular

1. `Study`'ye protokol mü (öneri), yoksa ayrı bir `MeshStudy` mı?
2. Ağ girişi: yalnız üreteçler mi, gmsh dosyası da mı (gmsh kurulumu gerekir)?
3. Kabuk için öncelik: önce `MeshDomain` mı, yoksa ayrı bir kabuk FEM modülü (optimizasyonsuz) mu?

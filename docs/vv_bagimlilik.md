# Çekirdek bağımlılık grafiği, skaler-E varsayımları ve desteklenmeyen kombinasyonlar (26 Kasım 2026)

00_MASTER "Başlangıç denetimi" maddesi. Etiketler: **[Ö]** betikle ölçüldü, **[O]** koddan okundu (bu oturumda,
dosya:satır verildi), **[K]** keşif koşusu (ön kayıtsız).

İçe aktarma grafiği betikle üretildi: `python docs/olcumler/baseline_bagimlilik.py docs/olcumler/baseline_bagimlilik.json`
(AST; modül düzeyi ve fonksiyon içi "tembel" içe aktarmalar ayrı; JSON'da bütün modüllerin grafiği ve SHA). [Ö]

## 1. Katmanlar ve kenarlar

```
configuration.build ──(tembel)──► optimization.study.Study ──► design.chain, optimization.responses (EvalContext),
      │                                   │                    optimizers.base/schedule, optimization.diagnostics
      │                                   └─(tembel)─► design.phases (MultiPhaseSIMP denetimi)
      ├──► configuration.registry  (hiçbir toporlogy modülüne bağlı değil; 15 modül onu içe aktarıyor)
      ├──(tembel)─► configuration.catalog.check_build   (bu dilimde eklendi)
      └──(tembel)─► boundary, design.manufacturing, optimization.robust / connectivity

registry ◄── configuration.components.* (core, physics, responses, design, ...) ── fizik kurucuları:
   physics.elasticity          → boundary.BoundaryConditions.physics → fem.elastic.Elasticity / ConstrainedElasticity / PerCaseSupports
   physics.elasticity_gmg      → fem.elastic_gmg.ElasticityGMG ──► fem.elastic, solvers.gmg, solvers.gmg_kernels
   physics.modal               → fem.modal.Modal ──► fem.elastic, solvers.direct
   physics.nonlinear_elasticity→ fem.nonlinear.NonlinearElasticity ──► fem.elastic, fem.elements, core.material

fem.elastic.Elasticity ──► core.domain, fem.elements, fem.mechanisms, solvers.direct, solvers.ordering
fem.elements ──► references (+ tembel core.material)      core.material: içe aktarması yok (yaprak)
fem.buckling ──► fem.elastic, fem.elements, solvers.direct   optimization.stress ──► fem.elements, optimization.local/strength
post.verify (T10) ──► references (+ tembel fem, geometry, post.mesh, post.pipeline, solvers)
configuration.post_steps ──(tembel)─► post, optimization.benchmark / connectivity / lengthscale
references: içe aktarması yok; 37 modül onu içe aktarıyor (@cite)
```

| Modül | Modül düzeyi içe aktarır | Onu modül düzeyinde içe aktaran | Tembel içe aktaran |
|:---|---:|---:|---:|
| `optimization.study` (Study) | 6 | 3 | 2 |
| `optimization.responses` (EvalContext) | 1 | 13 | 1 |
| `core.domain` (Domain) | 0 | 21 | 2 |
| `fem.elastic` (Elasticity) | 5 | 11 | 1 |
| `fem.elastic_gmg` (ElasticityGMG) | 7 | 0 | 2 |
| `fem.nonlinear` (NonlinearElasticity) | 7 | 0 | 3 |
| `design.phases` (MultiPhaseSIMP) | 2 | 0 | 4 |
| `configuration.registry` | 0 | 15 | 1 |
| `references` | 0 | 37 | 0 |
| `post.verify` (T10) | 1 | 1 | 0 |

Kaynak: `docs/olcumler/baseline_bagimlilik.json` (SHA `8d44453` üstünde üretildi). [Ö] Tablo sırası 00_MASTER'daki
listenin sırası.

Gözlem [O]: `ElasticityGMG`, `NonlinearElasticity` ve `MultiPhaseSIMP` hiçbir modülce modül düzeyinde içe
aktarılmıyor; yalnız registry'nin tembel kurucuları ve testler üzerinden erişiliyor. İsteğe bağlı fiziklerin
eski örneklere yük bindirmemesi (00_MASTER kural 2) bu sayede sağlanıyor.

## 2. Ortak sözleşme: skaler modül (scalar-E) varsayımı [O]

Bütün fizikler `solve(E_active, F)` arayüzünü kullanıyor (`optimization/study.py:142` Study belgesi): tasarım
fiziğe **eleman başına tek bir skaler** E üzerinden giriyor; malzeme yönü/anizotropi birim modüllü D'de sabit.

| Yer | Varsayım | Sonuç |
|:---|:---|:---|
| `fem/elastic.py:79` `Elasticity` | `E_active` (n_active,) bir birim D'yi (eleman başına farklı olabilir) ölçekler | yoğunluk anizotropiyi değiştiremez; D tasarım değişkeni değil |
| `fem/elastic_gmg.py:84` | eleman başına D reddedilir; tek birim D | GMG yalnız tek malzeme yönü |
| `design/phases.py:102` `MultiPhaseSIMP` | fazlar ν'yü ve tek birim D'yi paylaşır, eleman başına tek E | fazlara ayrı ν / anizotropi yok (`configuration/schema.py` belgesi) |
| `fem/nonlinear.py:376` | genel D yalnız `svk` ile; neo-Hooke izotrop E0, ν | anizotrop neo-Hooke yok |
| `configuration/components/core.py:120-132` SIMP/RAMP/Hashin–Shtrikman | `E0` skaler; HS yalnız izotrop | — |
| `configuration/components/responses.py:83` `coupling_compliance` | yardımcı çözüm `nu=material["nu"]` ile izotrop | ortotrop malzemede bağlantı yanıtı izotrop varsayımla hesaplanır (risk, test yok) |
| `configuration/components/responses.py:140` `stress` (von Mises yolu) | `nu`, `E0` izotrop; ortotropta `FailureIndex` + D kullanılır | — |
| `configuration/post_steps.py:202-222` `reverify` | ortotrop D geçiriliyor; izotropta E0, ν | A oturumunun açık notu kapanmış görünüyor (`test_material_config.py::test_reverify_step_passes_the_orthotropic_elasticity`) |

## 3. Yinelenen matematik ve NumPy/Torch dikişleri [O]

* İzotrop D iki yerde: `fem/elements.py:53` `D_iso(nu)` (birim modül, Voigt γ) ve `post/verify.py:74` `d_iso(E, nu)`
  (E ile ölçekli, Voigt sırası `[εx, εy, εz, γyz, γxz, γxy]`). Voigt sırası iki dosyada ayrı yazılmış; aynı olduğu
  bu oturumda denetlenmedi (NOT VERIFIED). Tensör çekirdeğine bağlanması tensor-core oturumunun işi (01).
* Voigt yardımcıları üç yerde: `fem/buckling.py:96` `voigt_pairs`, `fem/nonlinear.py:175` `_voigt`, `fem/elements.py`
  B matrisleri.
* Her fizikte iki yol: `solve_numpy` (autograd yok) ve `solve` (torch, özel adjoint) — 5 `solve_numpy` tanımı;
  kaynakta 15 `.detach().numpy()` noktası. Sayısal sonuç iki yolda aynı mı sorusu fizik başına testlerde
  (`test_gmg.py`, `test_nonlinear.py`) ayrı ayrı yanıtlanıyor; genel bir sözleşme testi yok.

## 4. Desteklenmeyen kombinasyonlar

Tam liste ve durumları üretilmiş katalogda: `docs/vv_katalog.md` §5–6 (kaynak
`src/toporlogy/configuration/catalog.py`). Bu dilimde keşif koşusuyla [K] (44 kombinasyon kuruldu, şüpheli 11'i
iki iterasyon koşturuldu) bulunan iki geç ve ilgisiz hata artık kurulumda reddediliyor:

| Kombinasyon | Önce (gözlenen, ilk değerlendirmede) | Şimdi |
|:---|:---|:---|
| `physics: modal` + durum isteyen yanıt (varsayılan `compliance` dahil) | `AttributeError: 'Modal' object has no attribute 'solve'` | kurulumda ConfigError (katalog) |
| `buckling` + eğik mesnet / yay (`ConstrainedElasticity`) | `ValueError: wrong M dimensions (161, 161), should be (168, 168)` | kurulumda ConfigError (katalog) |
| `buckling` + `elasticity_gmg` / `nonlinear_elasticity` | ilk değerlendirmede açık TypeError | kurulumda ConfigError (katalog) |

Python API yolu (Study doğrudan kurulursa) bu denetimden geçmiyor: `LinearBuckling` `isinstance(physics,
Elasticity)` denetliyor ve `ConstrainedElasticity` bir `Elasticity` alt sınıfı (`optimization/buckling.py:62`).
Kökten düzeltme ayrı iş.

Kurulumun kabul ettiği ama hiçbir testin sınamadığı kombinasyonlar NOT VERIFIED olarak katalogda: ör. `stress` +
`nonlinear_elasticity` (küçük şekil değiştirme gerilmesi büyük yer değiştirmede; doğrulanmış bir ölçü değil),
`robust_compliance` + `elasticity_gmg`, fazlar + `modal`.

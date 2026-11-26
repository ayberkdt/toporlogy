# Model dönemi değişiminde optimizatörü sıfırlama: ön kayıt (ME1–ME4)

Ön kayıt 2026-11-26 07:48 +0300, kod yazılmadan ve hiçbir koşudan önce (04 §2: "Optimizer'a model değişimini bildir; eski
tangent/multiplier/moving asymptote geçmişini yanlış problemde kullanma"; nonlinear-solvers notu: `model_epoch`
okunarak sıfırlama kancası).

**Değişiklik.** `Study(reset_on_model_change=True)`: her değerlendirmeden sonra `physics.model_epoch` (yalnız bu
özniteliği olan fizikler) önceki değerden farklıysa `optimizer.reset()` çağrılır (asimptotlar, önceki iterasyonlar,
ölçek); sayı `Result.settings`'e ve geçmişe (`history["model_epoch"]`) yazılır. `False` eski davranış.

| # | Kontrol | Eşik |
|:---|:---|:---|
| ME1 | Uyarlamalı kesmeli nonlineer çalışmada `optimizer.reset` çağrı sayısı = koşu sırasındaki epoch artışı sayısı; çağrılar epoch'un değiştiği iterasyonlarda | aynen |
| ME2 | `model_epoch` olmayan fizik (doğrusal `Elasticity`): `pytest -m baseline` bit düzeyinde ve geçmiş anahtarları aynı | aynen |
| ME3 | Sabit kesme (`cutoff_step = 0`, epoch değişmez): `reset_on_model_change` True ve False aynı sonucu verir | bit düzeyinde |
| ME4 | Ölçüm (eşiksiz): aynı uyarlamalı problemde True / False son amaç, iterasyon, epoch değişimi sayısı | rapor |

## Sonuç (2026-11-26, ilk koşu)

`python -m pytest tests/toporlogy/test_model_epoch.py -q -p no:randomly -s` → 3 passed, 2 xfailed; `pytest -m baseline`
→ 8 passed (ME2: doğrusal örnekler bit düzeyinde, geçmiş anahtarları aynı: `model_epoch` yalnız bu özniteliği olan
fizikte eklenir).

| # | Sonuç |
|:---|:---|
| ME1 | **FAIL as written** (strict xfail): `optimizer.reset` 6 kez çağrıldı, epoch değişimi 5. Fazlası koşu başındaki `restart()` (varsayılanı `reset()`); ön kayıt bunu saymıştı. Kancanın kendi sayacı = 5 = değişim sayısı (post hoc test) |
| ME2 | PASS |
| ME3 | **FAIL** (strict xfail): sabit kesme 0,1 ve uç yükü 0,05 ile Newton bir tasarımda 0,58 yük çarpanında durdu, iki ayarda da; karşılaştırma yapılamadı. Post hoc 0,01 yükte True/False bit düzeyinde aynı |
| ME4 | ölçüm: 30 iterasyon, 5 epoch değişimi; sıfırlamayla son amaç 0,322404, sıfırlamasız 0,32317 (tek problem; genelleme yok) |

Varsayılan `reset_on_model_change=True` (04 §2'nin istediği). Sabit kesmeli ya da epoch'u olmayan fizikte etkisi yok.

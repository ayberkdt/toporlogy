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

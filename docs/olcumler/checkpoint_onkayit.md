# Optimizasyon checkpoint/restart: ön kayıt (CK1–CK5)

Ön kayıt 2026-11-26 08:14 +0300, kod yazılmadan ve hiçbir koşudan önce (08 §4: "Checkpoint/restart: design, optimizer, schedules, RNG,
material state, load history, version compatibility, atomic write"; 07 §4).

**Tasarım.** `Study.run(checkpoint=path, checkpoint_every=k)` her k iterasyonda bir dosyaya (geçici dosya +
`os.replace`, atomik) şunları yazar: tasarım vektörü x, iterasyon, geçmiş, zincir (süreklemenin değiştirdiği β
dahil), takvim, optimizatör durumu (asimptotlar, önceki iterasyonlar, ölçek), yanıtların dondurulmuş ölçekleri,
torch ve NumPy RNG durumları, kütüphane sürümü ve problemin parmak izi (değişken sayısı, ayarlar).
`Study.run(resume=path)` aynı yapılandırmadan kurulmuş yeni bir `Study`'de kaldığı yerden sürer.

| # | Kontrol | Eşik |
|:---|:---|:---|
| CK1 | MMA + β süreklemesi (mbb2d küçültülmüş): 24 iterasyonluk kesintisiz koşu ile 10'da checkpoint + yeni Study'de resume | x, amaç, geçmiş bit düzeyinde aynı |
| CK2 | Aynı kontrol GCMMA ve OC ile | bit düzeyinde aynı |
| CK3 | Başka bir problemin (farklı değişken sayısı ya da ayar) checkpoint'inden resume | açık `ValueError` (parmak izi) |
| CK4 | Yazım atomik: checkpoint dosyası her an okunabilir (geçici dosya kalmaz) | aynen |
| CK5 | checkpoint verilmeden koşu eski davranış (`pytest -m baseline`) | bit düzeyinde |

## Sonuç (2026-11-26, ilk koşu)

`python -m pytest tests/toporlogy/test_checkpoint.py -q -p no:randomly` → 5 passed (ikinci koşu; aşağıya bakın).

| # | Sonuç |
|:---|:---|
| CK1 | PASS: MMA + β süreklemesi (30 × 10, β her 6 iterasyonda ikiye; kesme 10. iterasyonda, β adımlarının arasında): 24 iterasyonluk kesintisiz koşu ile sürdürülen koşu x, amaç, iterasyon sayısı ve geçmişte bit düzeyinde aynı |
| CK2 | PASS: GCMMA ve OC için aynı |
| CK3 | PASS: başka alan boyutu ("different domain, n_var") ve başka optimizatör ("optimizer") reddediliyor |
| CK4 | PASS: geçici dosya kalmıyor |
| CK5 | PASS: `pytest -m baseline` 8 passed (checkpoint verilmeden davranış bit düzeyinde aynı) |

**İlk koşudan sonra yapılan açıklama.** Ön kayıt "geçmiş bit düzeyinde aynı" diyordu. İlk koşuda tek fark geçmişin
`seconds` sütunundaydı (duvar saati süresi). Bu sütun kesintisiz iki koşu arasında da aynı olamaz, bu yüzden
karşılaştırmanın dışında bırakıldı; geri kalan bütün sütunlar (amaç, kısıtlar, değişim, β) bit düzeyinde aynı. Bu
bir eşik değişikliği değil, ölçülemez bir sütunun ayrılmasıdır; kayıtta açıkça belirtildi.

Kapsam dışı: fiziğin kendi durumu (ör. nonlineer enerji interpolasyonunun uyarlamalı kesmesi, malzeme geçmişi)
checkpoint'e yazılmaz; böyle bir fizikle sürdürülen koşu bit düzeyinde aynı olmayabilir (sınanmadı). YAML/CLI
bağlantısı (`run.checkpoint_every`) yapılmadı.

## Ek ön kayıt: YAML ve CLI (CK6–CK7)

Ön kayıt 2026-11-26 08:26 +0300, kod yazılmadan önce. `run.checkpoint_every: k` (varsayılan 0 = yok) koşu klasörüne
`checkpoint.ckpt` yazar; `python -m toporlogy run CONFIG --out DIR --resume` bitmemiş bir koşuyu oradan sürdürür.

| # | Kontrol | Eşik |
|:---|:---|:---|
| CK6 | CLI: checkpoint_every 1 ile kesilen (5. iterasyonda) koşu `--resume` ile bitirilir; `result.npz` x'i kesintisiz CLI koşusununkiyle | bit düzeyinde aynı |
| CK7 | `--resume` checkpoint olmayan klasörde | çıkış 2, açık mesaj |

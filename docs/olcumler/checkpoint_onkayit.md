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

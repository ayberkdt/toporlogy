# Sürdürülebilir tarama (manifesto, --resume, kaynak hesabı): ön kayıt (R1–R6)

Ön kayıt 2026-11-26 04:30 +0300, kod yazılmadan ve hiçbir koşudan önce (08 §4: bağımsız koşular için iş manifestosu, sürdürme, kaynak
hesabı, belirlenimci toplama; 07 §4: deney yürütücüsü, checkpoint/restart).

**Tasarım.** `run --sweep` ilk koşudan önce kök klasöre `sweep_manifest.json` yazar: istek (yapılandırma
dosyası, `--set`, tarama anahtarları ve değerleri), isteğin SHA-256 özeti, ortam (Python, NumPy, torch, git), her
koşu için klasör, kombinasyon, durum (pending / done / failed / config error), süre ve süreç tepe belleği (psutil
varsa; yoksa null). Her koşudan sonra geçici dosyaya yazılıp `os.replace` ile değiştirilir (atomik). `--resume`
aynı isteği sürdürür: `summary.json` olan koşular yeniden koşulmaz, kalanlar koşulur, `sweep.csv` bütün koşulardan
indeks sırasıyla yeniden üretilir. Optimizasyon durumunun yarı yolda checkpoint'i bu dilimde değil (`Study.run`
hızlandırma oturumunda).

| # | Kontrol | Eşik |
|:---|:---|:---|
| R1 | Manifesto ilk çözümden önce var; her koşudan sonra güncel; JSON her an okunabilir (atomik yazım) | aynen |
| R2 | 3 koşuluk tarama 2. koşudan sonra kesilir (test, bir istisnayla); `--resume` 1–2'yi yeniden koşmaz (`result.npz` baytları aynı), 3'ü koşar | aynen |
| R3 | Sürdürülen taramanın `sweep.csv`'si, kesilmemiş aynı taramanınkiyle süre sütunu dışında aynı; amaç değerleri bit düzeyinde | aynen |
| R4 | `--resume` farklı bir istekle (başka değer, başka `--set`) | çıkış 2, mesaj farkı adlandırır |
| R5 | `--resume` olmadan dolu klasöre tarama: eski davranış (çıkış 2) | aynen |
| R6 | Manifesto ortamı ve her koşunun süresini taşır; psutil varken tepe bellek > 0 | aynen |

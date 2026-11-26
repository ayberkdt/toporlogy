# YAML şema sürümü ve birim denetimi: ön kayıt (U1–U8)

Ön kayıt 2026-11-26 03:03 +0300, kod yazılmadan ve hiçbir koşudan önce (08 §4: şema sürümü, çözümlenmiş yapılandırma, birim
denetimi ve boyut tutarlılığı; kütle/ağırlık, basınç/kuvvet, sıcaklık/sıcaklık farkı karışıklıkları).

**Tasarım.**
- Üst düzey `schema_version` (yoksa "1"; desteklenen: "1"; başkası açık hata) çözümlenmiş yapılandırmaya yazılır.
- Üst düzey `units: {length, force, mass?, temperature?}` isteğe bağlı. Verilirse, birim türü bildirilmiş
  (`Param.unit`: length, force, force/length^2, mass/length^3, length/time^2, Hz …) bir parametre "210 GPa",
  "1 mm", "-1 kN" gibi birimli metin alabilir. Değer boyut analiziyle kullanıcının tutarlı birim sistemine çevrilir
  (zaman saniye; kütle = kuvvet · s² / uzunluk). Boyut uyuşmazlığı, adı geçen iki boyutla açık `ConfigError`.
- `units` yoksa davranış aynen: birimli metin "expected a number" hatası (mevcut fault-injection durumu
  `h_with_unit_text` aynı kalır), sayılar dönüştürülmez.
- Yeni modül `configuration/units.py`; çözümleme `Param.coerce` içinde (tek yer).

| # | Kontrol | Eşik |
|:---|:---|:---|
| U1 | 12 örneğin çözümlenmiş yapılandırması `schema_version` / `units` anahtarları dışında değişmez | eşit (dict) |
| U2 | `units: {length: mm, force: N}`: "1 mm" → 1, "0.001 m" → 1, "210 GPa" → 210000, "1 kN" → 1000, "7850 kg/m^3" → 7.85e-9 | ≤ 1e-15 göreli |
| U3 | Boyut uyuşmazlığı: kuvvete "5 kg", E0'a "100 N", uzunluğa "1 s" | `ConfigError`; mesaj beklenen ve verilen boyutu adlandırır (ör. "force", "mass") |
| U4 | Sıcaklık: mutlak "20 degC" bir sıcaklık-farkı türüne | açık hata; "5 K" ikisine de kabul (birimler modülü düzeyinde) |
| U5 | Tutarsız beyan: `units: {length: mm, force: N, mass: kg}` (tutarlı kütle t) | `ConfigError` "inconsistent" |
| U6 | `units` yokken "1 mm" | `ConfigError` "number" (mevcut davranış) |
| U7 | `schema_version`: yok → "1"; 1 / "1" kabul; "2" | "2" için `ConfigError` sürümü ve desteklenenleri adlandırır |
| U8 | `mbb2d` sabit tasarımda: mm/N ile E0 "210 GPa", kuvvet "-1 kN" uyumu = birimsiz uyum · (1000² / 210000) | ≤ 1e-12 göreli |

## Sonuç (2026-11-26, ilk koşu)

Komut: `python -m pytest tests/toporlogy/test_units.py -q -p no:randomly` → 22 passed, 1 xfailed.

| # | Sonuç |
|:---|:---|
| U1 | PASS: 12 örneğin çözümlenmiş yapılandırması, değişiklikten önceki kodun (`cb1a483`) yazdığı `tests/toporlogy/baseline/resolved_configs.json` ile aynı (yeni iki anahtar dışında) |
| U2 | PASS: sekiz dönüşüm ≤ 1e-15 göreli (mm, m, GPa, kN, kg/m³ → t/mm³, m/s² → mm/s², N/mm²) |
| U3 | PASS: "5 kg" kuvvete, "100 N" E0'a, "1 s" uzunluğa; mesaj iki boyutu da adlandırıyor |
| U4 | PASS |
| U5 | PASS: mm/N ile kg reddediliyor, t kabul ediliyor |
| U6 | PASS: `units` yokken "expected number … (a value with a unit needs the top-level units: section …)" |
| U7 | PASS |
| U8 | **FAIL** (strict xfail): uyum oranı 1000²/210000'den 3,75e-12 göreli sapıyor, eşik 1e-12. Girdiler tam ölçekli (post hoc: yük vektörü 1000 × birimsiz, E0 = 210000 bit düzeyinde); fark çözümün yuvarlamasında olmalı (Emin ile cond K ≈ 1e8–1e11; neden doğrulanmadı) ve eşik bunu öngörmemişti |

İlk denemede bir uygulama hatası çıktı (kontrol eşiği değil): liste parametrelerinde "min"/"max" gibi anahtar
sözcükler de birim olarak okunmaya çalışıldı; yalnız "sayı + birim" biçimindeki öğeler dönüştürülecek şekilde
düzeltildi.

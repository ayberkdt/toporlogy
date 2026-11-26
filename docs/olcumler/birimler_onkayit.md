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

# Malzeme veri kataloğu: şema, doğrulama ve YAML başvurusu — ön kayıt (C1–C7)

Ön kayıt 2026-11-26 03:28 +0300, kod yazılmadan ve hiçbir koşudan önce (08 §3 malzeme veri kataloğu, §5 "geçerli anizotropik Poisson
oranlarını yanlış reddetmeyen malzeme doğrulaması").

**Kapsam.** `configuration/catalog.py` `MaterialRecord` 08 §3 alanlarıyla genişler: kimlik, tür (synthetic /
nominal / typical / test_measured / design_allowable), aile (isotropic / orthotropic), sınıf, durum, sıcaklık,
değerler ve birim sistemi, belirsizlik, kaynak, sayfa/tablo, lisans, uygulanabilirlik. **Gerçek malzeme değeri
girilmez**: bu oturumda kaynağından ve lisansından okunmuş açık veri yok; kayıtlar sentetiktir ve öyle etiketlenir.
YAML'da `material: {catalog: <id>}` kaydın değerlerini `material` bölümüne yazar (değerleri SI olan kayıt
`units` bölümü ister ve onun sistemine çevrilir).

| # | Kontrol | Eşik |
|:---|:---|:---|
| C1 | Her kayıt şemayı geçer: benzersiz kimlik, bilinen tür/aile, sentetik olmayanda kaynak + sayfa/tablo + lisans dolu | aynen |
| C2 | Her kaydın değerleri kabul edilebilir: izotropta E > 0, −1 < ν < 1/2, ρ > 0; ortotropta `Orthotropic` (pozitif tanımlı uyum) | aynen |
| C3 | Negatif kontroller: izotrop ν = 0,5; pozitif tanımlı olmayan ortotrop; lisanssız `typical` kayıt | her biri reddedilir, mesaj alanı adlandırır |
| C4 | Pozitif tanımlı ama ν23 > 0,5 olan ortotrop kayıt ve ν < 0 (auxetic) izotrop kayıt | kabul edilir |
| C5 | `material: {catalog: <sentetik id>}` çözümlenmiş `material` bölümü, değerleri elle yazmakla aynı | eşit (dict) |
| C6 | SI kayıt + `units: {length: mm, force: N}`: E0 `units.convert` ile aynı sayı | bit düzeyinde aynı |
| C7 | Bilinmeyen kimlik (öneriyle), `catalog` ile aynı anahtarın elle de verilmesi, `units` olmadan SI kayıt | her biri `ConfigError` |

## Sonuç (2026-11-26, ilk koşu)

Komut: `python -m pytest tests/toporlogy/test_material_catalog.py tests/toporlogy/test_catalog.py -q -p no:randomly`
→ 29 passed. C1–C7 PASS.

* Beş sentetik kayıt: birim izotrop (varsayılan), auxetic izotrop (ν = −0,5), README'deki ortotrop kat, ν23 = 0,65
  olan pozitif tanımlı ortotrop, SI birimli izotrop. Gerçek malzeme kaydı yok (NOT VERIFIED: bu oturumda kaynağı
  ve lisansı okunmuş açık veri yok).
* Katalog düzeni sürümü 1.0.0 → 1.1.0 (yeni alanlar: sınıf, durum, sıcaklık, belirsizlik, sayfa/tablo, lisans,
  uygulanabilirlik; eklemeli). `docs/vv_katalog.{md,json}` yeniden üretildi.
* YAML: `material: {catalog: <id>}`; SI kayıt `units` bölümüne çevrilir (mm/N ile 2e11 Pa → 200000).

# PR-09 devamı — yayımlanmış sayılara karşı gerilme ve burkulma (25 Kasım 2026)

Testler: `tests/toporlogy/test_published_values.py`. Kaynak kodların aktarımı: `tests/toporlogy/published_codes.py`
(`TopBuck250`, `DENG_B`, `deng_pnorm`). Ham sayılar: [`yayimlanmis_degerler.json`](yayimlanmis_degerler.json).
Komut: `python -m pytest tests/toporlogy/test_published_values.py -s` (duvar testleri için `-m slow`). Bu dosyadaki
her sayı bu komutun çıktısından ya da §4'te adı geçen deneme betiğinden; etiketler `literatur_2026-11-21.md` ile
aynı (**[Ö]** ölçüldü, **[O]** kaynaktan okundu).

## 0. Özet

PR-09 Top88 ve Top3D'yi kodlarına karşı doğruladı; gerilme ve burkulma açık kalmıştı. Bu çalışma ikisini de,
tam metni okunan iki makalenin basılı sayısına ve basılı koduna karşı sınadı.

| Vaka | Aktarım / yayımlanan | log10 bağıl fark, toporlogy – aktarım | Karar |
|:---|---:|---:|:---|
| B-duvar: λ1, katı tasarım (yayımlanan 3,15) | 0,41 | −10,0 | FAIL |
| S-konsol: p-norm, x = 0,3 (yayımlanan 33,17) | 2,06 | −5,2 | FAIL |
| B-kolon: λ1, katı tasarım (kod–kod) | — | −12,4 | PASS |
| B-kolon: λ1, rastgele tasarım (kod–kod) | — | −12,8 | PASS |

Satırlar önce yayımlanmış sayılı vakalar, sonra kod–kod vakaları olarak sıralı. "Aktarım", makalenin kendi basılı
kodunun NumPy aktarımıdır.

- **toporlogy basılı kodlarla aynı sonucu veriyor [Ö].** Burkulmada aynı formülasyonda (Gauss noktası gerilmesi)
  λ1 farkı 8,9e-11 (duvar) ve ≤ 4,4e-13 (kolon); uyum, gerilme ve K_G ≤ 2e-13. Gerilmede p-norm farkı 6,6e-6; bu
  fark basılı B matrisinin 5 anlamlı basamağından geliyor.
- **Basılı iki sayı, makalelerin kendi basılı koduyla da üretilemedi [Ö].** Duvarda aktarım 1,303 veriyor, makale
  3,15 diyor. Konsolda aktarım 68,45 veriyor, makale 33,17 diyor. toporlogy aktarıma uyduğu için fark toporlogy'de
  değil. Üç olası yer var ve aralarında ayrım yapılamadı: makalenin basmadığı kurulum (sürücü, çağrı), basılı
  sayının kendisi, ya da benim aktarımımda kalan bir hata (kod MATLAB'de koşulamadı, §5). Bu yüzden iki sayı da
  toporlogy'yi **doğrulamıyor**. Karar önceden kayıtlı ölçüte göre FAIL.
- **toporlogy'nin burkulma formülasyonu topBuck250'ninkinden farklı [Ö, O].** topBuck250 K_G'yi eleman merkezindeki
  gerilmeyle kuruyor (satır 124), toporlogy her Gauss noktasındaki gerilmeyle (`fem/buckling.py`). Fark kolonda
  %0,01–0,5, duvarda %7,8 (merkez gerilmesi λ1'i büyütüyor). Duvardaki farkın neden büyük olduğu araştırılmadı.
- **Deng vd. kodu gerilmeyi merkezde değil, tek bir Gauss noktasında hesaplıyor [Ö].** Basılı B, birim
  altıyüzlünün (+g, +g, +g) noktasındaki şekil fonksiyonu türevleri (3,3e-6 içinde; diğer yedi nokta ≥ 0,455 uzak).
  toporlogy'nin `VonMises` yanıtı merkezde değerlendirdiği için aynı p-norm 52,71 çıkıyor. Bu bir hata değil;
  aynı nicelik değil.

## 1. Ön kayıt (2026-11-25T14:37+03:00, ilk koşudan önce; toleranslar sonradan değişmedi)

Kaynaklar (arXiv PDF'inin tam metni, `pdftotext` ile okundu):

- **[FSG21]** Ferrari, Sigmund, Guest, "Topology Optimization with linearized buckling criteria in 250 lines of
  Matlab", arXiv:2101.02973v1 (SMO 63:3045–3066, 2021). Kod: Ek C. Sayı: §5.2, s. 12, "The fundamental BLF is
  λ1 = 3.15 for the initial, solid structure".
- **[DVT21]** Deng, Vulimiri, To, "An efficient 146-line 3D sensitivity analysis code of stress-based topology
  optimization written in MATLAB", arXiv:2104.01210v1. Kod: Ek A. Sayı: §6.2, "The p-norm stress value decreases
  from initial 33.17 to 3.31".

**Vaka B (burkulma), [FSG21] §5.2.** 320 × 320, Lx = 1, E0 = 1, Emin = 1e-6, ν = 0,3. 1. döngüde pK = pG = 3
(çağrı `topBuck250(320,320,3,3,2,'N',0.5,2,[0.1,0.7,1.2],500,1,3,12,160,{['V','C','B'],[2.5,1.05]})`). A ve P1
üzerinde x = 1, P0 üzerinde 0 (`prSel{1}(1) == 'V'` olduğu için volfrac = 1). 1:9, 120:128, 257:265, 313:321
sütunlarındaki alt düğümler tutulu. Sol kenarın her düğümünde 1e-2/320 x-yükü var; uç düğümlerde yarısı.

- B1: basıldığı haliyle aktarım (K_G merkez gerilmesiyle): |λ1 − 3,15| ≤ 0,0051.
- B2: toporlogy `LinearBuckling`: |λ1/3,15 − 1| ≤ 1e-2.
- B3: aktarımın Gauss noktası kipi ile toporlogy arasında bağıl fark ≤ 1e-8.
- B4 (hızlı, yayımlanmış sayı yok): topBuck250'nin kendi kolon kurulumu, 60 × 30, x = 1 ve tohumlu rastgele x.
  Sınırlar: uyum 1e-9, merkez gerilmesi 1e-9, K_G 1e-10, en düşük dört λ 1e-8. Merkez kipindeki λ'lar yalnız
  yazdırılır.

**Vaka S (gerilme), [DVT21] §6.2.** 200 × 60 × 1 H8, E0 = 1, Emin = 1e-9, ν = 0,3, pl = 3, q = 0,5, p = 10.
x = 0 yüzü tutulu; x = 200, y = 0 kenarının iki düğümünde y yönünde −1 (sonlu eleman kısmı top3d'ninkiyle aynı).
**Varsayım:** sürücü kod basılmadığı için "initial" x = volfrac = 0,3 alındı.

- S1: basılı B ile aktarım: |pn − 33,17| ≤ 0,01.
- S2: toporlogy ile aktarım arasında U farkı: ≤ 1e-9.
- S3: basılı B, toporlogy'nin 8 Gauss noktası matrisinden tam birine ≤ 1e-5 içinde eşit olmalı. O noktadaki
  toporlogy p-norm'u için iki sınır: |pn − 33,17| ≤ 0,006 ve aktarıma göre bağıl fark ≤ 2e-4.
- S4: toporlogy `VonMises` (merkez) p-norm'u sınırsız yazdırılır.

## 2. Sonuçlar [Ö]

İlk koşu 25 Kasım 2026, ~14:55 (hızlı) ve ~15:05 (duvar, 162 s).

| Ölçüt | Ölçülen | Sınır | Karar |
|---|---|---|---|
| B1 aktarım, merkez kipi | λ1 = 1,303072 | 3,15 ± 0,0051 | FAIL |
| B2 toporlogy | λ1 = 1,201863 (bağıl −0,618) | ±1e-2 | FAIL |
| B3 toporlogy – Gauss kipi | 8,92e-11 | 1e-8 | PASS |
| B4 kolon, katı | uyum 1,5e-13; σ 1,9e-13; K_G 1,3e-13; λ 4,4e-13 | 1e-9 / 1e-9 / 1e-10 / 1e-8 | PASS |
| B4 kolon, rastgele | uyum 1,4e-13; σ 1,7e-13; K_G 1,0e-13; λ 1,7e-13 | aynı | PASS |
| S1 aktarım | pn = 68,447505 | 33,17 ± 0,01 | FAIL |
| S2 U farkı | 9,68e-8 | 1e-9 | FAIL |
| S3 basılı B = Gauss noktası 7 | 3,33e-6 (öbürleri ≥ 0,455) | 1e-5, tek nokta | PASS |
| S3 toporlogy – yayımlanan | pn = 68,447056 | 33,17 ± 0,006 | FAIL |
| S3 toporlogy – aktarım | 6,56e-6 | 2e-4 | PASS |
| S4 toporlogy `VonMises` (merkez) | pn = 52,711275 | — | ölçüm |

Ölçütler B1 → S4 sırasında, ön kayıttaki sırayla. Kolon λ değerleri (toporlogy / basılı kodun merkez kipi):
katıda 42,980 / 42,985, 166,17 / 166,96, 167,09 / 167,98, 172,65 / 173,09; rastgelede 10,679 / 10,680,
47,156 / 47,173, 51,419 / 51,670, 60,043 / 60,056.

Testlerde FAIL olan ölçütler `xfail(strict=True)` ile işaretli ve kayıt docstring'de. Bir gün geçerlerse test
düşer; o zaman bu not da güncellenmelidir.

## 3. Ön kayıtta olmayan denemeler (yalnız yorum için; hiçbiri benimsenmedi)

- **S2'nin nedeni.** cond(K_ff) ≈ 5,4e9; 200 × 60 × 1 konsol tek eleman kalınlığında. İki çözümün de
  toporlogy'nin K'sindeki bağıl residual'ı küçük: toporlogy 5,4e-12, aktarım 1,1e-11. Yinelemeli iyileştirilmiş
  çözüme göre toporlogy 7,6e-9, aktarım 5,1e-8 uzakta. 1e-9 sınırı koşullanmayı hesaba katmamıştı (Top3D 24 × 8 × 4
  vakasından taşındı). Bunu sabitleyen test sonradan yazıldı ve öyle etiketli: `test_deng_displacements_are_
  backward_stable`, residual ≤ 1e-10.
- **S1 için:** tek düğüme yük 17,26; düzgün x = 0,4 → 33,34, 0,5 → 19,09, 1 → 3,374. 33,17'yi veren yok.
- **B1 için:** ters yük (−x) λ1 = 0,863. Basıldığı haliyle ilk dört λ: 1,303, 1,592, 1,697, 1,795.

Betikler oturumun geçici klasöründe (`deng_probe.py`, `wall_probe.py`) kaldı, depoya alınmadı. Sayıları
JSON'da `post_hoc_probe` alanında.

## 4. Denenmeyenler (gerekçe ilk koşudan önce yazıldı)

| Aday | Durum | Neden |
|---|---|---|
| top99neo Tablo 2 (MBB 300 × 100, 2500 it., c = 252,7) | UNVERIFIED | OC çarpan aralığı, simetrik filtre sınırı ve durma ölçütü toporlogy'den farklı; 2500 iterasyonda yakınsamamış: aynı problem değil |
| PolyStress (Giraldo-Londoño & Paulino 2021) | UNVERIFIED | arXiv'de yok, metin okunamadı; çokgen ağ |
| GCMMA, Svanberg 2007 notu | UNVERIFIED | people.kth.se/~krille/mmagcmma.pdf artık "page no longer in use" sayfası |
| [FSG21] §5.1 kolon sayıları (λ1 0,75 / 8,53; c 8,57e-6) | UNVERIFIED | optimizasyon geçmişinin tamamı gerekiyor |
| [DVT21] son değerler (3,31 / 5,51 / 4,72), 2-B L-braket 56,23 | UNVERIFIED | MMA koşusu gerekiyor; L-brakette pasif eleman ve filtre sırası basılmamış |

## 5. Sonuç ve açık iş

Yayımlanmış sayılara karşı bağımsız doğrulama hedefi **bu çalışmayla kapanmadı**. Seçilen iki sayı, kendi
makalelerinin basılı koduyla tekrar üretilemedi. Kazanılan şey kod düzeyinde bağımsız iki karşılaştırma:
toporlogy'nin burkulma ve gerilme hesabı topBuck250 ve Deng vd. kodlarıyla yuvarlama düzeyinde aynı. Ayrıca iki
formülasyon farkı ölçüldü: K_G'de merkez ya da Gauss noktası gerilmesi (%0,01–7,8) ve gerilmenin değerlendirildiği
nokta (p-norm 68,45 ↔ 52,71).

Açık iş: (1) yazarların yayımladığı `.m` dosyalarının (makale ekindeki metin değil) MATLAB/Octave'da koşulup
3,15 ve 33,17'nin tekrar üretilmesi. Bu oturumda denenemedi: bu makinedeki MATLAB R2022b
(`D:\MatlabDosyam`) `matlab -batch` ile "Licensing error: -10,32" verdi, Octave kurulu değil. Böyle bir koşu NumPy
aktarımındaki olası bir indeks çevirisi hatasını da dışlar; bu hata olasılığını bu notta yalnız kolon ve duvar
kurulum denetimleri sınırlıyor (§2, `test_topbuck250_element_matrix_and_wall_setup`). (2) Bağımsız bir gerilme
ya da burkulma sayısı için tam kurulumu basılı başka bir kaynak.

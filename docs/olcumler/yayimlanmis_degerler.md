# PR-09 devamı — yayımlanmış sayılara karşı gerilme ve burkulma (25 Kasım 2026)

Testler: `tests/toporlogy/test_published_values.py`. Kaynak kodların aktarımı: `tests/toporlogy/published_codes.py`
(`TopBuck250`, `DENG_B`, `deng_pnorm`). Ham sayılar: [`yayimlanmis_degerler.json`](yayimlanmis_degerler.json).
Komut: `python -m pytest tests/toporlogy/test_published_values.py -s` (duvar testleri için `-m slow`). Bu dosyadaki
her sayı bu komutun çıktısından ya da §4'te adı geçen deneme betiğinden; etiketler `literatur_2026-11-21.md` ile
aynı (**[Ö]** ölçüldü, **[O]** kaynaktan okundu).

## 0. Özet

PR-09 Top88 ve Top3D'yi kodlarına karşı doğruladı; gerilme ve burkulma açık kalmıştı. Bu çalışma ikisini de,
tam metni okunan iki makalenin basılı sayısına ve basılı koduna karşı sınadı. İlk koşudan sonra "neden üretilemiyor,
kodumuzda hata mı var" sorusu ayrıca araştırıldı (§3b).

| Vaka | x0'da aktarım / yayımlanan | Bir MMA adımı sonrası / yayımlanan | log10 fark, toporlogy – aktarım | Karar |
|:---|---:|---:|---:|:---|
| B-duvar: λ1, katı tasarım (3,15) | 0,41 | — | −10,0 | FAIL, açıklanamadı |
| S-konsol: p-norm (33,17) | 2,06 | 0,999 | −5,2 | FAIL, açıklandı (post hoc) |
| S-L-braket: p-norm (56,23), post hoc | 2,10 | 1,008 | — | ön kayıtta yok |
| B-kolon: λ1, katı (kod–kod) | — | — | −12,4 | PASS |
| B-kolon: λ1, rastgele (kod–kod) | — | — | −12,8 | PASS |

Satırlar önce yayımlanmış sayılı vakalar, sonra kod–kod vakaları olarak sıralı. "Aktarım", makalenin kendi basılı
kodunun NumPy aktarımıdır. "Bir MMA adımı sonrası" sütunu ön kayıtta yoktu (§3b).

- **toporlogy'de hata bulunmadı [Ö].** toporlogy iki basılı kodun aktarımıyla yuvarlama düzeyinde aynı sonucu
  veriyor. Burkulmada aynı formülasyonda (Gauss noktası gerilmesi) λ1 farkı 8,9e-11 (duvar) ve ≤ 4,4e-13 (kolon);
  uyum, gerilme ve K_G ≤ 2e-13. Gerilmede p-norm farkı 6,6e-6; bu fark basılı B matrisinin 5 anlamlı basamağından
  geliyor. Aktarımın kendisini bağımsız bir okuma (ayrı bir ajan, makale metnine karşı satır satır) denetledi ve
  sayıları değiştiren bir çeviri hatası bulmadı.
- **Gerilme (Deng vd.): fark açıklandı, kod hatası değil, okuma hatası [Ö, post hoc].** Makalenin "initial"
  dediği değer x0'daki değil, ilk MMA güncellemesinden sonraki değer. x0 = 0,3'ten makalede yazan move = 0,1 ile
  tek bir MMA adımı atılınca konsolda 33,130 (yayımlanan 33,17, %−0,12), L-brakette 56,70 (yayımlanan 56,23,
  %+0,84) çıkıyor. Ön kayıttaki varsayım ("initial" = x0 = 0,3) yanlıştı; ön kayıtlı karar FAIL olarak kalıyor.
  Kalan %0,1–0,8'lik farkın olası kaynağı yazarların MMA'sının basılmamış ayrıntıları (alt problem çözücüsü, c, d);
  bu sınanmadı.
- **Burkulma (topBuck250 duvarı): fark açıklanamadı [Ö].** Basılı kod ve çağrıyla λ1 = 1,303 çıkıyor, makale 3,15
  diyor. Yazarların DTU sayfasındaki dosyası (`topbuck250_finalsamo.m`, okundu, çalıştırılmadı) analiz satırlarında
  basılı listeyle aynı; demek ki aktarılan kod yazarların kodu. Duvar kurulumu yalnız makale metninde var. Denenen
  varyantların hiçbiri 3,15'i vermedi (§3b). Fark toporlogy'de değil; basılmamış bir kurulum farkında ya da basılı
  sayıda. Bu sayı toporlogy'yi doğrulamıyor.
- **toporlogy'nin burkulma formülasyonu topBuck250'ninkinden farklı [Ö, O].** topBuck250 K_G'yi eleman merkezindeki
  gerilmeyle kuruyor (satır 124), toporlogy her Gauss noktasındaki gerilmeyle (`fem/buckling.py`). Fark kolonda
  %0,01–0,5, duvarda %7,8 (merkez gerilmesi λ1'i büyütüyor). Bu bir hata değil, bir seçim; duvarda neden büyük
  olduğu araştırılmadı.
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

## 3b. Neden üretilemedi: araştırma (25 Kasım 2026, ön kayıt dışı)

**Aktarımın denetimi.** Ayrı bir ajan iki aktarımı makale metnine karşı satır satır okudu: numaralama, cMat, Ke0,
dN/B0, merkez gerilmesi, G'nin 10 katsayılı sıkıştırması ve G + G' − diag, özdeğer problemi, duvarın a1…a6
aralıkları, mesnet ve yük, başlangıç tasarımı; Deng'de yük, mesnet, edofMat, lk_H8, B, D, von Mises ve p-norm.
Sayıları değiştiren bir çeviri hatası bulmadı. Makale metninin kendi içinde iki tutarsızlık buldu; ikisini DTU
dosyasında ben doğruladım: imzada `maxit` `ocPar`'dan önce geliyor (DTU dosyasında tersi), `ocUpdate` 10 argümanla
tanımlı ama 11 argümanla çağrılıyor (DTU dosyasında 11). İkisi de 1. döngüdeki λ'yı değiştirmiyor; ama basılı
listenin, koşulan kodun birebir kopyası olmadığını gösteriyorlar.

**Gerilme, Deng vd.** Önce L-braket (§6.3, yayımlanan 56,23) aynı okumayla hesaplandı: x0 = 0,3'te oran 2,10,
konsoldakine (2,06) yakın, yani sistematik bir fark. İkisi de düzgün x ≈ 0,40'a denk geliyordu; bu da 0,3 + move
0,1 eder. Sınama: toporlogy'nin MMA'sı (Svanberg 2007 katsayıları, `normalise=None`, move 0,1) ile, stress_minimize'ın
kurduğu problemde (filtre, p-norm amaç, ortalama x̄ − 0,3 ≤ 0) tek adım atıldı ve yeni tasarım aktarımla
değerlendirildi:

| Vaka | x0 = 0,3'te | Bir MMA adımı sonrası | Yayımlanan | Fark |
|:---|---:|---:|---:|---:|
| Konsol 200 × 60 × 1 | 68,45 | 33,13 | 33,17 | %−0,12 |
| L-braket 200 × 200 × 1 | 117,91 | 56,70 | 56,23 | %+0,84 |

Satırlar makaledeki sırada. Move 0,2 ile konsolda 39,15 çıkıyor (%+18); sonuç hareket sınırına duyarlı, makalede
yazan 0,1 ile tutuyor. L-brakette pasif bölgenin nerede uygulandığı basılmamış; filtreden sonra 1e-4 ve x0 = 0,3
alındı. Başka bir okuma (x0'da da pasif 1e-4) ilk adımdan sonra 101,47 verdi. Testler: `test_deng_cantilever_
published_value_is_after_the_first_mma_step` (hızlı, sınır %0,5) ve `test_deng_lbracket_..._mma_step` (yavaş, sınır
%2). İki sınır da değerler görüldükten sonra kondu.

**Burkulma, topBuck250 duvarı.** Denenen varyantların ilk λ'ları (λ1 ∝ 1/|q|, sabit toplam yükte ∝ Lx; 3,15 için
|q| ≈ 4,1e-3 ya da Lx ≈ 2,42 gerekirdi, makale ikisini de basmıyor):

| Varyant | λ1 |
|:---|---:|
| Basıldığı gibi | 1,303 |
| Yük ters (−x) | 0,863 |
| Alt kenarın bütün düğümleri tutulu | 2,329 |

Basıldığı gibi kurulumda ilk 40 λ 1,303 ile 3,196 arasında; 39.'su 3,156. Onu seçmek için bir gerekçe yok.
Betikler (`wall_variants.py`, `deng_mma_step.py`, `deng_lbracket.py`) oturumun geçici klasöründe.

**Sonuç.** Kodumuzda hata bulunmadı. Gerilmede fark okuma farkıydı ve kapandı. Burkulmadaki fark kaynak tarafında
kalıyor; DTU dosyasını MATLAB'de duvar çağrısıyla koşmak bunu çözer, ama bu makinede MATLAB lisans hatası veriyor.

## 4. Denenmeyenler (gerekçe ilk koşudan önce yazıldı)

| Aday | Durum | Neden |
|---|---|---|
| top99neo Tablo 2 (MBB 300 × 100, 2500 it., c = 252,7) | UNVERIFIED | OC çarpan aralığı, simetrik filtre sınırı ve durma ölçütü toporlogy'den farklı; 2500 iterasyonda yakınsamamış: aynı problem değil |
| PolyStress (Giraldo-Londoño & Paulino 2021) | UNVERIFIED | arXiv'de yok, metin okunamadı; çokgen ağ |
| GCMMA, Svanberg 2007 notu | UNVERIFIED | people.kth.se/~krille/mmagcmma.pdf artık "page no longer in use" sayfası |
| [FSG21] §5.1 kolon sayıları (λ1 0,75 / 8,53; c 8,57e-6) | UNVERIFIED | optimizasyon geçmişinin tamamı gerekiyor |
| [DVT21] son değerler (3,31 / 5,51 / 4,72), 2-B L-braket 56,23 | UNVERIFIED | MMA koşusu gerekiyor; L-brakette pasif eleman ve filtre sırası basılmamış |

## 5. Sonuç ve açık iş

Yayımlanmış sayılara karşı bağımsız doğrulama hedefi **yarı yarıya kapandı**. Gerilmede yayımlanan iki
sayı (33,17 ve 56,23), "initial" ilk MMA adımı sonrası diye okununca %0,1–0,8 içinde üretiliyor (post hoc; ön
kayıtlı karar FAIL). Burkulmadaki sayı (3,15) üretilemedi ve nedeni bulunamadı. Kazanılan şey kod düzeyinde bağımsız iki karşılaştırma:
toporlogy'nin burkulma ve gerilme hesabı topBuck250 ve Deng vd. kodlarıyla yuvarlama düzeyinde aynı. Ayrıca iki
formülasyon farkı ölçüldü: K_G'de merkez ya da Gauss noktası gerilmesi (%0,01–7,8) ve gerilmenin değerlendirildiği
nokta (p-norm 68,45 ↔ 52,71).

Açık iş: (1) DTU'daki `topbuck250_finalsamo.m` dosyasının duvar çağrısıyla (§5.2) MATLAB ya da Octave'da
koşulup 1. döngüdeki λ1'in okunması. Bu oturumda denenemedi: bu makinedeki MATLAB R2022b (`D:\MatlabDosyam`)
`matlab -batch` ile "Licensing error: -10,32" verdi, Octave kurulu değil. 1,303 çıkarsa fark makaledeki sayıda ya
da basılmamış kurulumdadır; 3,15 çıkarsa aktarımda bulunamamış bir fark vardır. (2) Bağımsız bir burkulma sayısı
için tam kurulumu basılı başka bir kaynak (ör. Ferrari & Sigmund 2019, arXiv:1809.05300; okunmadı).

## 6. İkinci tur (25 Kasım 2026, dal `feat/material-models`)

Soru: yeni fizik (`malzeme_modelleri.md`) referanslarla olan farkı kapattı mı; kapatmadıysa neden? Ek olarak yeni
referanslar. Ön kayıt: oturum notlarındaki `pr09b_published_values_preregistration.md`, "Round 2" bölümleri; R1
16:58'de, R2 17:11'de, ilgili koşulardan önce yazıldı.

**Duvar (topBuck250, 3,15): yeni fizik farkı kapatmadı; kapatamaz da [Ö].** Yeni dalda yeniden koşuldu: toporlogy
1,201863, aktarım 1,303072; değişiklik yok. Neden ölçüldü: katı duvarın ağa göre yakınsayan bir λ1'i yok.

| Ağ | λ1 (Q4) | λ1 (Q4IM) | Mod 1 enerjisinin tutulu bölge kenarına 0,02 içindeki payı |
|---:|---:|---:|---:|
| 40² | 2,852 | — | 0,49 |
| 80² | 2,106 | 1,946 | 0,66 |
| 160² | 1,602 | 1,492 | 0,79 |
| 320² | 1,202 | 1,117 | 0,93 |

Satırlar ağ incelme sırasıyla. Birinci mod, sol kapı çerçevesinin tutulu bölgesinin bittiği noktada (x ≈ 0,37,
alt kenar) toplanıyor. Orada doğrusal elastik gerilme tekil, ağ inceldikçe λ1 → 0. O bölgede gerilme rijitliği
kaldırılınca mod önce kapı boşluğunun içe dönük köşesine, sonra kaldırılan bölgenin kenarına kayıyor. Sonuç:
basılı tek bir λ1, kullanılan ağın bir özelliği; bu referans bir burkulma kodunu doğrulayamaz. Bu, yazarların
kendi kodu için de geçerli.

**Gerilme (Deng vd.): okuma örneklem dışında doğrulandı [Ö].** "initial = ilk MMA adımından sonraki değer"
okuması iki 2-B vakadan türetilmişti. 3-B L-braket (100 × 100 × 30, yayımlanan 46,12) için ön kayda alındı
(sınır %2) ve sonra koşuldu: x0'da 94,115, tek MMA adımından sonra 45,806, fark −%0,68 → **PASS**. Çözücü:
`ElasticityGMG`, CUDA, 948 693 DOF, 17 s. Test: `test_deng_3d_lbracket_out_of_sample` (yavaş; CUDA yoksa atlanır).

**Yeni referans: Ferrari & Sigmund (2019), arXiv:1809.05300v2 (tam metin okundu).** Ucundan basılan konsol kolon
(Lx = 10, Ly = 1, t = 0,005, E = 2e11, ν = 0, Pc = 2,05617e6, f0 = 2,5e5). Şekil 1, yedi ağda Q4 ve Q6 hatalarını
veriyor. toporlogy'nin ızgarası kare elemanlı; bu ağlardan yalnız 20 × 2 kare.

| Ölçüt (ön kayıtlı) | toporlogy | Yayımlanan | Sınır | Karar |
|:---|---:|---:|---:|:---|
| R1a Q4, 20 × 2 | 0,118684 | 0,119 | ±5e-4 | PASS |
| R1b Q4IM (Q6), 20 × 2 | −0,005107 | −0,0066 | ±5e-5 | FAIL |
| R1c Q4IM, 20 × 2 … 160 × 16, \|e\| | ≤ 6,9e-3 | — | ≤ 1e-2 | PASS |

Ölçütler ön kayıttaki sırayla. R1b'nin ön kayıt dışı incelemesi dikdörtgen elemanlı ayrı bir küçük kodla yapıldı
(bütün ağlar hesaplanabilsin diye). Bu kod beş kaba Q4 satırını basılı basamaklarda veriyor (0,4939; 0,4933;
0,1187; 0,1181; 0,0243). İki ince satırı ve Q6 sütununu ise hiçbir varyantla vermiyor. Benim Q4 ve Q6 dizilerim
aynı limite (≈ −0,0070) ikinci mertebeden yakınsıyor; makalede en ince ağda Q4 ile Q6 hâlâ 3e-3 farklı. Bu
kaynaktan kullanılabilir kısım kaba Q4 satırları.

Bu inceleme bir fizik iyileştirmesi verdi: uyumsuz mod elemanlarında modlar K_G'ye de giriyor
(`LinearBuckling(modes_in_kg=True)`, varsayılan). 20 × 2'de Q4IM −0,005107 → −0,00561, yakınsak değere göre hata
1,9e-3 → 1,4e-3. Q4IM 2-B'ye yeni eklendi; Q4 ise saf eğilmede %11 kilitleniyor.

**Denenmeyen adaylar (kurulum metinde tam değil):** FS2019 çerçeve örneği (J0 = 4,714e-3, λ1,0 = 0,662) ve FS2020,
arXiv:1909.13585 (J(0) = 2,501e-3, λ1(0) = 0,597). Mafsal ve yük düğümleri, düzgün tasarımın yoğunluğu ve
geometri yalnız şekillerde; ikincisinde ayrıca kaba seviyeden yaklaşık modlar kullanılıyor. Birden çok yorumu
deneyip tutanı seçmek sonuç avcılığı olacağından koşulmadı (UNVERIFIED).

**Durum özeti.** Gerilmede iki 2-B ve bir 3-B yayımlanmış sayı açıklandı; sonuncusu örneklem dışında, ön kayıtla.
Burkulmada bir yayımlanmış FE sayısı (FS2019, Q4, 20 × 2) ön kayıtla tuttu. Duvar sayısının ise doğrulama
referansı olamayacağı ölçüldü.

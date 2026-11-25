# Malzeme modelleri ve burkulma fiziği — doğrulama kaydı (25 Kasım 2026)

Dal: `feat/material-models`. Kod: `src/toporlogy/core/material.py`, `optimization/strength.py`,
`optimization/stress.py` (`FailureIndex`), `fem/elements.py` (`element_matrix_from_D`), `fem/buckling.py`,
`optimization/buckling.py`, `design/interp.py`. Testler: `tests/toporlogy/test_material.py`, `test_strength.py`,
`test_buckling_physics.py`, `test_material_config.py`. Bu dosyadaki sayıların kaynağı şu komutun çıktısı:

```
python -m pytest tests/toporlogy/test_material.py tests/toporlogy/test_strength.py tests/toporlogy/test_buckling_physics.py tests/toporlogy/test_material_config.py -s
```

Etiketler `literatur_2026-11-21.md` ile aynı: **[Ö]** ölçüldü, **[O]** koddan okundu, **[H]** hafızadan, doğrulanmadı.

## 0. Neden

Burkulma ve malzeme modeli koddan okunarak incelendi ve eksikler çıkarıldı [O]:

- λ bütün yükü çarpıyordu; ön yük yoktu.
- Burkulma ile akma birlikte denetlenmiyordu.
- 2-B modelde düzlem dışı burkulma görülemiyor; bu, hiçbir yerde yazmıyordu.
- Burkulmada küme kesilme uyarısı ve sahte mod tanısı yoktu.
- Malzeme yalnız izotroptu; 2-B'de düzlem gerinim yoktu.
- Dayanım malzemede değil, yanıtın içindeydi (`sigma_lim`). Tek kriter von Mises'ti.
- Gri malzemeye fiziksel bir anlam veren interpolasyon yoktu.

Hepsi eklendi. Yalnız düzlem dışı burkulma model olarak eklenmedi; sınırlama olarak yazıldı.

## 1. Eklenenler ve doğrulamaları [Ö]

| Ekleme | Bağımsız karşılaştırma | Ölçülen |
|:---|:---|---:|
| Ortotrop C, döndürme (tensör), düzlem gerilme ve gerinim indirgemesi | izotrop özel durum = `D_iso`, `D_plane_stress`, `D_plane_strain` | ≤ 1e-14 |
| | izotrop C döndürmeye karşı değişmez; 1/E(n) = n n n n S | ≤ 1e-13 |
| Genel D'den Q4 / H8 / H8IM | rijit hareket enerjisiz, u = ε·x enerjisi εᵀDε (ortotrop, döndürülmüş D) | ≤ 1e-12 |
| | izotrop D ile `ke_q4` ve `ke_h8` | ≤ 1e-14 |
| Düzlem gerinim (izotrop ve 25° döndürülmüş ortotrop) | tek eleman kalınlığında, w = 0 tutulu 3-B levha | ≤ 1e-10 |
| Ortotrop şerit, çekme | θ = 0'da E1, θ = 90°'de E2 | ≤ 1e-10 |
| Dayanım kriterleri | indeks 1: tek eksenli çekme ve basma, saf kayma (von Mises τ = σ/√3, Tresca τ = σ/2), Tsai–Wu'nun 8 dayanım durumu | ≤ 1e-12 |
| | 1. dereceden homojen; izotroplar döndürmeye karşı değişmez; σc = σt'de DP ve Raghava = von Mises | ≤ 1e-10 |
| `FailureIndex` | von Mises ölçütüyle eski `VonMises` alanı (2-B, 3-B) | ≤ 1e-12 |
| | düzlem gerinimde σzz = ν(σxx + σyy) ile elle hesaplanan von Mises | ≤ 1e-12 |
| | p-norm gradyanı, sonlu fark (Richardson): DP, Raghava, Tsai–Wu | 1,4e-8; 1,2e-8; 2,7e-8 |
| Ön yük | aynı desende c·P ön yükü λ'yı tam c kadar düşürür (c = 1,127 ve 3,006; λ0 = 3,757) | ≤ 1e-8 |
| | ön yük tek başına burkuluyorsa hata (1,05·λ0) | hata veriyor |
| | gradyan, sonlu fark | 1,45e-8 |
| Küme kesilme uyarısı | kare kesitli 3-B kolon: 1 mod uyarı veriyor, 2 mod vermiyor | doğru |
| Düşük yoğunluk payı | tanım: her yer 0,9 → 0; her yer 0,1 → 1; yarısı gri → (0, 1) | doğru |
| İlk hasar yük katsayısı | tek eksenli çekme: izin 20, σ1 = 2 → 10; σ0 = 5 ön yükle → 7,5 | 10; 7,5 |
| Genel D ile burkulma | izotrop D, eski yolla aynı λ | ≤ 1e-8 |
| | ince ortotrop kolon (eksende E/4) ile izotrop E/4 kolon, 4 / 8 / 16 eleman genişlik | 2,1e-2; 5,9e-3; 2,1e-3 |
| Hashin–Shtrikman (HS) bulk sınırı | kaplı küre / silindir: oyuklu küre ve disk için Lamé çözümü, ν = 0; 0,25; 0,3; 0,45 ve x = 0,1–0,95 | ≤ 1e-12 |
| HS kayma sınırı | yalnız uç değerler (x = 0 → 0, x = 1 → G) ve monotonluk | doğru |
| `simp_admissible_penal` | p* ile x^p K ≤ K_HS ve x^p G ≤ G_HS (x ∈ (0, 1]); 0,95·p* ile ihlal | doğru |

Satırlar eklemelerin sırasıyla: önce elastisite, sonra dayanım, burkulma, ara yoğunluk.

## 2. İlk koşuda kalanlar ve nedenleri [Ö]

- **Ortotrop kolon.** Testin ilk sürümü tek bir ağda (4 eleman genişlik) %1 sınırı koymuştu; %2,1 ile kaldı.
  Neden ölçüldü: Q4 elemanın eğilme hatası. Parazit kayma G/E_eksen ile büyüyor ve bu malzemede iki kat büyük.
  Oran ağ inceldikçe ikinci mertebeden 1'e gidiyor (2,1e-2, 5,9e-3, 2,1e-3). Test artık bu yakınsamayı arıyor;
  kayıt testin docstring'inde.
- **Kendi hatam, testle bulundu.** Küme kesilme denetimi `find_clusters`'a azalan μ dizisi veriyordu; fonksiyon
  artan dizi bekliyor. Bu yüzden her ardışık çift küme sayılıyordu (3,69 ile 3,35 bile). Uyarıyı hata sayan bir
  koşu bunu gösterdi; dizi `−μ` ile veriliyor.
- **Test kurulumu.** HS karşılaştırması ilk koşuda 1e-8 sapıyordu. Neden: testteki `torch.tensor(x)` float32'ydi;
  float64'te fark ≤ 1e-12. HS'nin x = 0'daki Young modülü 0/0 veriyordu; K/x ve G/x ile yazıldı.

## 3. Kaynak durumu

- Ortotrop esneklik, döndürme ve indirgemeler: ilk ilkelerden yazıldı ve yukarıdaki özdeşliklerle doğrulandı.
- Kriterler tanımlarından yazıldı. Raghava paraboloidinin yazar atfı (Raghava, Caddell & Yeh 1973) [H]. Tsai–Wu'da
  depoda kayıt var (`tsai_wu_1971`), metin okunmadı. f12 = −0,5 yaygın bir seçim [H], ölçülmüş değer değil.
- HS sınırları genel iki fazlı ifadeden yazıldı [H]. Bulk kısmı kesin çözümle doğrulandı. Kayma kısmı için yalnız
  dolaylı bir kanıt var: iki sınırın x = 1'deki eğiminden türetilen SIMP koşulları (2-B max(2/(1−ν), 4/(1+ν)),
  3-B max(3(1−ν)/(2(1−2ν)), 15(1−ν)/(7−5ν))), Bendsøe & Sigmund (1999)'a atfedilen ve hafızadan bilinen koşullarla
  aynı çıkıyor [H; metin okunmadı].
- Kaynak kayıtları (`hashin_shtrikman_1963`, `bendsoe_sigmund_1999`) hafızadan girildi ve notlarında böyle yazıyor.

## 4. Davranış değişikliği

Varsayılan yol değişmedi: `material.orthotropic`, `material.strength` ve `plane: strain` verilmeyen her YAML aynı
nesneleri kuruyor (`test_defaults_build_the_original_objects`). Bit düzeyindeki örnek kayıtlarına
(`baseline/examples.json`) dokunulmadı.

Bir yan etki var: burkulma yanıtı küme denetimi için artık bir mod fazla hesaplıyor (n + 1). Bu, özdeğer
çözücüsünün yineleme ayrıntısını değiştirir; mevcut burkulma testleri kendi toleranslarıyla geçiyor.

API kaydı yeniden yazıldı (`tests/toporlogy/test_api_surface.py --write`). Hiçbir ad silinmedi. Eklenenler:
`core.material`, `optimization.strength`, `LinearBuckling.low_density_share`; `inverse_load_factors`'a yeni
parametreler. Bileşen listesine `hashin_shtrikman` eklendi; `stress` ve `stress_augmented_lagrangian`'a
`criterion`, `buckling`'e `preload_case` eklendi; `sigma_lim` artık isteğe bağlı.

## 5. Yapılmayanlar

- 2-B düzlem dışı (plaka) burkulma: kabuk ya da plaka elemanı gerektiriyor.
- Geometrik doğrusal olmayanlık, kusur duyarlılığı.
- Elemana göre değişen fiber yönü.
- Ortotrop malzemede bağlantılılık fiziği ve T10 doğrulaması (ikisi de izotrop kaldı).
- RAMP'in kaynağının okunması.
- Gibson–Ashby kafes yasaları: SIMP p = 2 / p = 1 okumasıyla aynı, ayrı kod eklenmedi.

# Elemana göre malzeme yönü ve ortotrop T10 — doğrulama kaydı

Dal: `claude/per-element-orientation-78c052` (koordinasyon notunda `feat/per-element-orientation` adıyla geçiyor),
taban `d566290`. Etiketler `malzeme_modelleri.md` ile aynı: **[Ö]** ölçüldü, **[O]** koddan okundu, **[H]**
hafızadan, doğrulanmadı.

## 0. Ön kayıt (2026-11-25 17:54 +0300, ilk test koşusundan önce yazıldı)

Kodun ilk satırı yazılmadan ve hiçbir yeni test koşmadan önce aşağıdaki karşılaştırmalar ve toleranslar sabitlendi.
Tutmayan bir ön kayıt gevşetilmez: test `xfail(strict=True)` olarak kalır ve nedeni docstring'e yazılır. Ön kayıttan
sonra eklenen her test "post hoc" diye etiketlenir.

Tasarım kararı (toleransları etkilediği için burada): eleman başına D verildiğinde eleman matrisleri **yalnız farklı
D'ler için** hesaplanır (tam eşitlikle `np.unique`) ve her biri bugünkü tek-D fonksiyonuyla
(`element_matrix_from_D`, Q4IM/H8IM yoğuşturması dahil) kurulur. Bu yüzden her elemanda aynı R verildiğinde eleman
matrisi bit düzeyinde tek-D yolununkiyle aynı olmalı; P1'deki 1e-14 bu beklentinin üstünde bir paydır.

| # | Karşılaştırma | Bağımsız taraf | Tolerans (göreli) |
|:--|:---|:---|---:|
| P1 | Her elemanda aynı R ile eleman başına D = tek D yolu: D, KE, toplanmış K, U, uyum, `FailureIndex` değeri, burkulma λ, modal λ; 2-B (Q4, Q4IM) ve 3-B (H8, H8IM) | tek D yolu (bugünkü kod) | ≤ 1e-14 |
| P2a | Paralel dizilim (Voigt): yük x yönünde, iki katman y'de üst üste (θ = 0° ve 90°), 2-B düzlem gerilme ve 3-B (ν13 = ν23 olan malzeme) | E_eff = Σ E_i H_i / H; sağ uçta u_x düzgün = ε L | ≤ 1e-10 |
| P2b | Seri dizilim (Reuss): iki katman x'te arka arkaya, düzgün çekme; 2-B ve 3-B (ν13/E1 = ν23/E2 olan malzeme, ara yüzde enine şekil değiştirme uyumlu) | 1/E_eff = Σ (L_i/L)/E_i | ≤ 1e-10 |
| P2c | Seri şeritte Tsai–Wu indeksi: 0° katmanda σ/Xt, 90° katmanda σ/Yt | strength ratio'nun kapalı çözümü | ≤ 1e-10 |
| P3a | Uyum gradyanı, eleman başına D (sürekli açı alanı), 2-B Q4 ve 3-B H8IM | merkezi sonlu fark, Richardson | ≤ 1e-6 |
| P3b | `FailureIndex` p-norm gradyanı, eleman başına D ve R, Tsai–Wu, 2-B | aynı | ≤ 1e-6 |
| P3c | Burkulma KS gradyanı, eleman başına D, 2-B Q4IM (modes_in_kg) | aynı | ≤ 1e-6 |
| P4a | T10 ortotrop çubuk çekmesi, düzgün θ ∈ {0, 30, 60, 90}° (z etrafında), yalnız rijit hareketi alan mesnetler | 1/E(θ) = c⁴/E1 + (1/G12 − 2ν12/E1)c²s² + s⁴/E2 | ≤ 1e-9 |
| P4b | Aynı çubukta H8 (`voxel_reference`, kesit 1 × 1 eleman: eşit dağıtılmış yük tutarlı yük olur) | aynı kapalı form | ≤ 1e-9 |
| P4c | T10 iki yönlü seri çubuk, tet yönü ağırlık merkezinin düştüğü vokselden; aynı çubuk H8 ile | Reuss kapalı formu (P2b) | ≤ 1e-9 |
| P5 | İzotrop T10 ve `voxel_reference` varsayılan yolu: d566290'da bu makinede kaydedilen sonuçlar (K'nin SHA-256'sı, uyum, en büyük yer değiştirme, en büyük von Mises) | eski kod | tam eşitlik |
| P6 | YAML: her yerde aynı açılı `angle_field` = `angle` ile kurulan çalışma (uyum) | tek D yolu | ≤ 1e-14 |
| P7 | `ElasticityGMG` eleman başına D'yi reddeder | açık hata | hata veriyor |

P5'in kayıtlı değerleri (d566290, `t10_fingerprint.py` betiği, bu makine): K SHA-256
`c37c6bb4a85341e2bb619a86df7cd6bf8ddf5071c2829a42b485e52ac6a7df6f`; T10 uyum `3.482339726026348`, en büyük yer
değiştirme `3.70297627743529`, en büyük von Mises `13.115959479529819`; voksel uyum `57.1186691763344`, en büyük yer
değiştirme `59.143123678603374`. Bu sayılar BLAS'a bağlı olabilir; başka makinede farklı çıkarsa ölçüt göreli
1e-14'e iner ve bu kayda yazılır.

Bit düzeyindeki örnek kayıtları (`tests/toporlogy/test_baseline.py`) değişmeden geçmeli.

## 1. Ne eklendi [O]

- `core/material.py`: `rotations_about_z` (2-B açı alanı), `rotations_from_axes` (3-B eksen alanı),
  `unit_elasticity_field` (n, k, k), `out_of_plane_rows`, `unique_matrices`; `elasticity_from_config` artık
  `angle_field` / `axis_field` (+ `axis2`) okuyor (`domain` parametresi eklendi).
- `fem/elements.py`: `element_matrix_field` (farklı D başına bir kez `element_matrix_from_D`, Q4IM/H8IM yoğuşturması
  eleman başına), `mode_recovery_from_D`, `indexed_matmul`.
- `fem/elastic.py`: `elasticity` (k, k) ya da (n_active, k, k). Eleman başına durumda `KE` farklı matrisler
  (m, nloc, nloc), `ke_index` (n_active,), `D_distinct`; `per_element`, `element_stiffness()`, `element_forces()`.
  Toplama ve eleman enerjisi farklı matris sayısı ≤ `GROUPED_MAX` = 32 ise grup grup tek-matris aritmetiğiyle,
  üstünde parçalı toplu çarpımla. Tek D yolu dokunulmadan duruyor.
- `fem/modal.py`, `boundary/constrained.py` (`reaction_forces`, `PerCaseSupports`): eleman başına matrisleri
  kullanıyor. `ConstrainedElasticity` kalıtımla çalışıyor. `fem/elastic_gmg.py`: eleman başına D'yi açık hatayla
  reddediyor (bu üç dosya koordinasyon notunda A'ya yazılı değildi; görev metni istediği için küçük değişiklik
  yapıldı, B ve C'nin dosyaları değil).
- `fem/buckling.py`: σ = D_e B u_e, Q4IM/H8IM'de eleman başına R_e ve T_e = [I; R_e] (modes_in_kg).
- `optimization/strength.py`: `rotate_stress_to_material` satır başına R; `Criterion.index(s, R=None)`,
  `Criterion.per_element`. `optimization/stress.py`: `FailureIndex` eleman başına D, `szz_row` ve R.
- `post/verify.py`: `stiffness_t10`, `solve_t10`, `voxel_reference`, `reverify` için `elasticity` (birim modüllü D,
  E ile ölçeklenir; (6, 6) ya da eleman başına); `tet_voxel_rows` (tet ağırlık merkezinin düştüğü voksel; aktif
  voksel dışındaysa en yakın aktif voksel merkezi, sayısı raporlanıyor).
- `configuration/helpers.py` (`material_model`), `configuration/schema.py` (belge): YAML alanları.
- `references/_composites.py`: `voigt_1889`, `reuss_1929` (hafızadan girildi, metinler okunmadı [H]). Kapalı
  formlar testte tam katmanlı çözümden türetildi; E(θ) formülü Jones'a atfediliyor [H], kütüphanenin tensör
  döndürmesinden bağımsız yazıldı.

YAML örneği (2-B):

```yaml
material:
  E0: 40.0
  orthotropic: {E1: 40.0, E2: 9.0, E3: 9.0, G12: 4.0, G13: 4.0, G23: 3.2, nu12: 0.28, nu13: 0.28, nu23: 0.4,
                angle_field: fiber_aci.npy}      # eleman ızgarası biçiminde, derece (z etrafında)
```

3-B: `axis_field: eksen.npy` ((*ızgara, 3), 1. eksen) ile `axis2: [0, 0, 1]`, ya da (*ızgara, 2, 3) iki eksen birden.

## 2. Sonuçlar [Ö]

Kaynak: `python -m pytest tests/toporlogy/test_element_orientation.py -s` (üçüncü koşu, 2026-11-25; P2 satırları
yalnız `print` satırları değiştikten sonraki yeniden koşudan). Bütün ön kayıtlı maddeler tuttu; strict xfail yok.

Bütün ön kayıtlı karşılaştırmalar toleransının en az dört mertebe altında kaldı:

| # | Karşılaştırma | Ön kayıtlı tolerans | Ölçülen göreli hata |
|:---|:---|:---|:---|
| P1 | Aynı R ile eleman başına D = tek D: K, U, uyum (Q4, Q4IM, H8, H8IM) | 1e-14 | 0 (tam eşit) |
| P1 | Aynı karşılaştırma, FailureIndex (Tsai–Wu) değeri | 1e-14 | 2,0e-16 – 5,6e-16 |
| P1 | Aynı karşılaştırma, burkulma λ ve modal λ | 1e-14 | 0 (tam eşit) |
| P2a | Paralel katmanlar (Voigt), 2-B Q4 / 3-B H8 | 1e-10 | 6,2e-15 / 2,2e-16 |
| P2b | Seri katmanlar (Reuss), 2-B Q4IM / 3-B H8IM | 1e-10 | 8,9e-16 / 8,9e-16 |
| P2c | Seri şeritte Tsai–Wu indeksi (σ/Xt, σ/Yt) | 1e-10 | 1,8e-15 |
| P3a | Uyum gradyanı, 2-B Q4 (40 farklı D) / 3-B H8IM (18 farklı D) | 1e-6 | 3,0e-8 / 5,4e-10 |
| P3b | Tsai–Wu p-norm gradyanı, eleman başına D ve R | 1e-6 | 6,0e-9 |
| P3c | Burkulma KS gradyanı, Q4IM, eleman başına D | 1e-6 | 1,9e-7 |
| P4a | T10 çubuk, E(θ), θ = 0, 30, 60, 90° | 1e-9 | ≤ 3,6e-15 |
| P4b | Aynı çubuk, H8 (voxel_reference) | 1e-9 | ≤ 1,6e-15 |
| P4c | T10 ve H8, iki yönlü seri çubuk, yön vokselden | 1e-9 | 2,4e-15 / 2,2e-16 |
| P5 | İzotrop T10 ve voksel yolu, d566290 kaydı | tam eşitlik | tam eşit |
| P6 | YAML angle_field = angle, uyum | 1e-14 | 0 (tam eşit) |
| P7 | ElasticityGMG eleman başına D'yi reddeder | hata | hata veriyor |

P3a'da 2-B alan 40 farklı matrisle `GROUPED_MAX`'ın üstündeki (toplu çarpım) yolu, 3-B alan 18 matrisle grup yolunu
sınıyor; ikisi de kapsandı. E(θ): 40; 15,32403959; 9,23254472; 9 (θ = 0, 30, 60, 90°).

Maliyet (post hoc ölçüm, tolerans yok; `test_cost_of_a_continuous_field`): 3-B H8, 24 × 12 × 12, her elemanda
ayrı D (3456 farklı): D alanı 1,53 s, `Elasticity` kurulumu 2,46 s (tek D: 0,18 s), toplama 0,02 s. Kurulum
farklı D sayısıyla doğrusal büyüyor (eleman başına ≈ 0,7 ms); 10⁵ farklı D'de dakika mertebesi beklenir [çıkarım,
ölçülmedi].

## 3. İlk koşuda ne oldu [Ö]

- **Ön kayıtta olmayan büyüklükler.** P1 testine eleman enerjisini (gradyanın girdisi), P6 testine gradyanları da
  koymuştum; ön kayıtta yoklardı ama aynı 1e-14 ile sınanıyorlardı. İlk koşuda eleman enerjisi 5,3e-15 (Q4),
  1,2e-14 (Q4IM), 3,9e-14 (H8), 1,7e-14 (H8IM), P6'nın birleşik farkı 8,7e-14 çıktı. Neden: eleman başına yol
  enerjiyi `indexed_matmul` + einsum ile, tek D yolu tek einsum ile topluyordu (toplama sırası farklı). Düzeltme:
  farklı matris sayısı ≤ 32 iken enerji ve kuvvetler grup grup tek-matris aritmetiğiyle hesaplanıyor; ikinci koşuda
  enerji farkı 0. Kalan tek fark P6'da gerilme kısıtının gradyanı: 1,3e-14 (eleman başına gerilme toplu einsum,
  tek D matris çarpımı; geri yayılım farklı yuvarlıyor). P6 testi ikiye ayrıldı: ön kayıtlı uyum ≤ 1e-14 (ölçülen 0),
  post hoc uyum gradyanı ve kısıt değeri ≤ 1e-14 (0), post hoc kısıt gradyanı ≤ 1e-13 (1,3e-14). Ön kayıtlı hiçbir
  tolerans gevşetilmedi.
- **Test düzeneği.** P4'ün dört durumu ilk koşuda `print` içindeki θ ve ° yüzünden düştü (cp1254 konsolu kodlayamıyor;
  `UnicodeEncodeError`). Ölçülen değerler hatadan önce yazılmıştı (θ = 0: T10 −2,3e-15, H8 −7,8e-16). `print`
  satırları ASCII yapıldı; iddialar değişmedi.

## 4. Davranış değişikliği

Varsayılan yol değişmedi: `elasticity` verilmeyen ya da tek D verilen her model bugünkü `KE` ve toplama kodundan
geçiyor (dal ayrımı `ke_index is None`). `test_baseline.py` bit düzeyinde geçti (yedi örnek, en büyük fark 0,00e+00;
tam paket çıktısı §6). İzotrop T10 sonuçları d566290 ile tam eşit (P5).

API: yalnız eklemeler, bir imza genişledi (`elasticity_from_config(material, dim, domain)`). Yerelde
`test_api_surface.py --write` ile yazılan fark koordinasyon klasöründe `A_api_surface.diff`; `tests/toporlogy/
baseline/*` commit edilmedi.

## 5. Yapılmayanlar

- Fiber açısının tasarım değişkeni olması (CFAO, eş zamanlı yön ve topoloji): yapılmadı.
- Laminat (katman dizilimi) teorisi, ABD matrisleri: yapılmadı.
- Hasar ilerlemesi (ilk hasardan sonrası): yapılmadı.
- `reverify` YAML adımı (`configuration/post_steps.py`) malzemeyi hâlâ izotrop E0, ν ile geçiriyor; dosya A'nın değil,
  dokunulmadı. Koordinatör için öneri: `reverify_step` içinde `material_model(ctx)` ortotropsa
  `elasticity=mm.elasticity` (3-B'de (6, 6) ya da (n_active, 6, 6)) ve `E=E1` geçirmek. Bugün ortotrop bir
  çalışmanın `reverify` sonucu izotrop malzemeyle hesaplanıyor [O].
- `reverify`'nin gmsh ile uçtan uca koşusu: bu ortamda gmsh yok; eleman başına yolun T10 kısmı yapılı tet ağıyla
  (P4c) ve `tet_voxel_rows` ile sınandı, gmsh ağıyla sınanmadı.
- Sürekli alanlarda kurulumun vektörleştirilmesi (§2 maliyet): yapılmadı.
- `ElasticityGMG` ile eleman başına D: desteklenmiyor, açık hata veriyor.

## 6. Test, ruff, mypy [Ö]

- `python -m pytest -m "not slow" tests/toporlogy` (API kaydı yerelde `--write` ile yazılmış hâlde):
  681 passed, 4 skipped, 46 deselected, 3 xfailed, 0 failed (742 s). Üç xfail önceden vardı (strict, yayımlanmış
  değerler ve FS2019 Q6); atlananlar pypardiso, gmsh, aerokit yokluğu. `test_baseline.py` bit düzeyinde: yedi örnekte
  en büyük fark 0,00e+00.
- `_composites.py` kayıtları eklendikten sonra API kaydı yeniden yazılıp `test_api_surface.py`, `test_architecture.py`,
  `test_core.py`, `test_baseline.py`, `test_element_orientation.py` yeniden koşuldu: 62 passed, 1 skipped. Sonra
  `git checkout -- tests/toporlogy/baseline`.
- `test_element_orientation.py` (yavaş maliyet testi dahil): 29 passed.
- `ruff check src tests`: temiz. Depo genelinde 59 bulgu var, hepsi önceden var olan `docs/olcumler/yontemler.py`
  (57) ve `simetri.py` (2) dosyalarında; bu dala dokunulmadı.
- `mypy src/toporlogy`: 103 dosya, hata yok.
- Ortamda yok: gmsh (`test_gmsh_tet_mesh_of_sphere`, `slow`, ilk dar koşuda `-m "not slow"` verilmediği için koştu ve
  bu yüzden düştü; kodla ilgisi yok).

## 7. README için önerilen metin

> **Elemana göre malzeme yönü.** Ortotrop malzemenin yönü eleman başına verilebilir: YAML'da
> `material.orthotropic.angle_field` (2-B, eleman ızgarası biçiminde `.npy`, derece) ya da `axis_field` (3-B,
> 1. eksen alanı ve `axis2`). Python'da `core.material.unit_elasticity_field` (n_active, k, k) D üretir ve
> `Elasticity`, `Modal`, `LinearBuckling`, `FailureIndex` (Tsai–Wu dahil, gerilme eleman eksenine döner) ve T10
> yeniden analizi (`post.verify`, tet yönü vokselden) bunu kabul eder. Doğrulama: düzgün alan tek-D yoluyla bit
> düzeyinde aynı; paralel ve seri iki yönlü şeritte Voigt ve Reuss kapalı formları ≤ 1e-14; T10 ve H8 off-axis çubuk
> E(θ) ≤ 4e-15; gradyanlar sonlu farkla ≤ 2e-7 (`docs/olcumler/eleman_yonu.md`). Fiber açısı tasarım değişkeni
> değildir; laminat teorisi ve hasar ilerlemesi yoktur; GMG çözücüsü eleman başına malzemeyi desteklemez.

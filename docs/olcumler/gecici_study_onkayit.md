# Geçici ısı iletimi ve Newmark dinamiği `Study` fiziği olarak — ön kayıt (HS1–HS7, DS1–DS7, G1)

Ön kayıt 2026-11-26 12:36 +0300, kod yazılmadan ve hiçbir koşudan önce. `fem.transient_heat.TransientHeat` (TH1–TH6,
`gecici_isi_onkayit.md`) ve `fem.dynamics_newmark.Newmark` (ND1–ND6, `newmark_onkayit.md`) şimdiye kadar yalnız
Python'dan kullanılıyordu; `Study` ve YAML bağlantıları yoktu.

## Tasarım

İki fizik de modal fiziğin kalıbını izler: statik durum u yoktur, `physics.solve` açık hatayla düşer; yanıtlar
`needs = {"history"}` bildirir ve zaman geçmişini `ctx.physics` ile kendileri koşar (değerlendirme başına bir kez,
`ctx.extra` önbelleğinde). Türev: her adım `GridSPD` autograd çözümü, torch zincir kuralıyla ayrık adjoint
(geri yayılım adımları tersten izler; faktörizasyon ileri koşudan yeniden kullanılır). Autograd grafiği bütün adımları
bellekte tutar (kontrol noktası yok; bu turda kapsam dışı).

**Isı (`physics: {type: transient_heat, …}`).** C(x̄) Ṫ + K(E) T = q, θ-yöntemi.
* İletkenlik k_e = k₀ · E_e / E0 (E çalışmanın interpolasyonundan, E0 malzemeden; `conductivity` = k₀).
* Isı kapasitesi c_e = c₀ (c_min + (1 − c_min) x̄_e) · frac_e (yoğunluğa doğrusal; `capacity` = c₀,
  `capacity_min_ratio` = c_min, varsayılan 1e-3; 1 verilirse düzgün kapasite). Gerekçe: kapasite kütleyle orantılıdır,
  iletkenlik SIMP ile cezalanır; c_min > 0 C'yi boşlukta da pozitif tanımlı tutar.
* Isı yutucuları: YAML `supports` düğümleri (her bileşeni sabit düğüm) T = `sink_temperature` (varsayılan 0).
  Isı problemi mekanik yük almaz: `load_cases` verilirse açık hata (modal fizik gibi `load_cases` isteğe bağlı olur;
  `configuration/resolve.py` ve `build.py`'deki "modal ise yük yok" denetimi bu fiziği de kapsayacak biçimde
  genişler, davranış başka fiziklerde aynı).
* Isı girişi q (tasarımdan bağımsız): `source_density` (birim hacimde, bütün etkin elemanlara tutarlı düğüm vektörü
  olarak) ve/veya `power` (toplam) `source` seçicisinin düğümlerine eşit paylaştırılmış. q = 0 açık hata.
* `dt`, `n_steps` zorunlu; `theta` varsayılan 1 (geri Euler, L-kararlı); `initial_temperature` düzgün T₀ (varsayılan 0).

**Yanıt `transient_thermal_compliance`.** J = (1/t_N) Σ_{n=0}^{N} w_n qᵀ T_n, yamuk ağırlıkları w = Δt·(½, 1, …, 1, ½),
t_N = N Δt: zamana göre ortalama ısıl uyum. Gerekçe: statik ısıl uyumun (qᵀ K⁻¹ q, ısı yutucusu tasarımının olağan
amacı) zaman ortalaması; t_N → ∞ iken statik değere gider (HS4 bunu ölçer), birimi statik uyumla aynıdır, q
tasarımdan bağımsız olduğundan J'nin türevi yalnız T_n üzerinden gelir.

**Dinamik (`physics: {type: dynamics_newmark, …}`).** M(x̄) ü + C u̇ + K(E) u = s(t) F, Newmark ortalama ivme.
* K = Σ E_e K_e⁰ (Q4 düzlem gerilme / H8; malzemenin tek D'si kabul, eleman başına D alanı açık hata), tutarlı kütle
  ρ_e = ρ₀ (ρ_min + (1 − ρ_min) x̄_e) · frac_e (ρ₀ = `material.rho`, varsayılan 1; `rhomin_ratio` varsayılan 1e-6:
  x̄ = 0 olan bölgelerde M tekil olmasın; başlangıç ivmesi M a₀ = f₀ ile çözülür). Yalnız doğrusal kütle bu turda.
* Rayleigh C = a M + b K (`rayleigh_alpha`, `rayleigh_beta`, varsayılan 0).
* F: tek yük durumunun yük vektörü (birden çok yük durumu / kombinasyon açık hata). Zaman profili s(t):
  `step` (t ≥ 0'da 1), `half_sine` (t ≤ τ'da sin(π t / τ), sonra 0; τ = `pulse_duration` zorunlu),
  `sine` (sin(2π t / P); P = `period` zorunlu). u₀ = v₀ = 0. `dt`, `n_steps` zorunlu.
* Tek mesnet takımı, yalnız x / y / z sıfır yer değiştirmesi (eğik mesnet, yay: açık hata).

**Yanıt `dynamic_compliance`.** J = (1/t_N) Σ_n w_n f(t_n)ᵀ u_n (aynı yamuk ağırlıkları): zamana göre ortalama dinamik
uyum ∫ fᵀu dt / t_N. Gerekçe: dış yükün yaptığı işin zaman ortalaması; sönümsüz basamak yükünde (u₀ = v₀ = 0) her mod
statik değerinin etrafında salındığından J → statik uyum (DS4), yani statik uyum tasarımının doğrudan dinamik
genellemesi; yük geçmişi sınırlıyken tanımlı ve türevlenebilir.

Katalog: iki fizik `PHYS` ve `PHYSICS` kayıtları; yeni yanıtlar kendi fizikleriyle SUPPORTED (kanıt: HS1/HS3,
DS1/DS3), başka fiziklerle UNSUPPORTED (bileşen düzeyinde). Statik durum okuyan yanıtlar (compliance, displacement,
stress, buckling, …) bu iki fizikle UNSUPPORTED (katalog düzeyinde: `check_build` u isteyen yanıtı reddeder); öz değer
yanıtları UNSUPPORTED (bileşen: "needs physics modal"). `tests/toporlogy/test_compatibility.py` bu hücreler için
sonda alır.

## Kontroller

Isı (`tests/toporlogy/test_transient_study.py`):

| # | Kontrol | Eşik |
|:---|:---|:---|
| HS1 | Study değeri = aynı k, c, q, T₀ ile doğrudan `TransientHeat.run` + yamuk toplamı (2-B, θ = ½) | ≤ 1e-12 göreli |
| HS2 | Study değeri = bağımsız yoğun numpy θ-yöntemi (eleman matrisleri testte kendi 2 × 2 Gauss döngüsüyle, yoğun birleştirme, `numpy.linalg.solve`), 2-B | ≤ 1e-10 göreli |
| HS3 | Study gradyanı merkezi farkla (yönlü türev, rastgele yön): 2-B θ = ½ ve 3-B θ = 1, kapasite yoğunluğa bağlı (c_min = 0,1) | ≤ 1e-6 göreli |
| HS4 | Uzun süre limiti: geri Euler, t_N = 200/λ₁ (λ₁: K φ = λ C φ'nin en küçük öz değeri), Δt = 0,1/λ₁; J / (qᵀ K⁻¹ q) − 1 (kuramsal sapma ≤ 1/(λ₁ t_N) = 5e-3) | ≤ 1e-2 mutlak |
| HS5 | YAML örneği kurulur, 5 MMA iterasyonu koşar | sonlu değerler; son iterasyonun amacı ilkinden küçük |
| HS6 | `load_cases` verilmiş, rulo (kısmi DOF) mesnet, eğik mesnet, Q4IM, θ ∉ [0, 1], Δt ≤ 0, n_steps < 1, ısı girişi yok, `compliance` + transient_heat, `transient_thermal_compliance` + elasticity | açık hata, beklenen iletiyle (`match=`) |
| HS7 | Bir gradyanlı değerlendirmede faktörizasyon sayısı | 1 |

Dinamik (aynı dosya):

| # | Kontrol | Eşik |
|:---|:---|:---|
| DS1 | Study değeri = aynı E, ρ ile doğrudan `Newmark.run` + yamuk toplamı (2-B, basamak yükü) | ≤ 1e-12 göreli |
| DS2 | Study değeri = bağımsız yoğun numpy Newmark (ivme biçimi: (M + γΔtC + βΔt²K) a_{n+1} = f − C(…) − K(…); kodun yer değiştirme biçiminden farklı yazım), Rayleigh sönümlü, 2-B. Eleman matrisleri kütüphaneden (`element_matrix_from_D`, `mass_matrix_element`): bağımsız değil | ≤ 1e-9 göreli |
| DS3 | Study gradyanı merkezi farkla: 2-B half_sine + Rayleigh, 3-B step | ≤ 1e-6 göreli |
| DS4 | Quasi-statik limit: sönümsüz basamak yükü, t_N = 50 T₁ (T₁ ilk periyot), Δt = T₁/40; J / (Fᵀ K⁻¹ F) − 1 (kuramsal sapma ≤ 1/(ω₁ t_N) = 3,2e-3) | ≤ 1e-2 mutlak |
| DS5 | YAML örneği kurulur, 5 MMA iterasyonu koşar | sonlu değerler; son iterasyonun amacı ilkinden küçük |
| DS6 | iki yük durumu, yük durumuna özel mesnet, eğik mesnet, Q4IM, eleman başına D (açı alanı), Δt ≤ 0, n_steps < 1, bilinmeyen profil, `pulse_duration`'sız half_sine, `compliance` + dynamics_newmark, `dynamic_compliance` + elasticity | açık hata, beklenen iletiyle |
| DS7 | Bir gradyanlı değerlendirmede faktörizasyon sayısı | K_eff 1, kütle 1 |

Ortak:

| # | Kontrol | Eşik |
|:---|:---|:---|
| G1 | `pytest -m baseline` (8 test) | bit düzeyinde geçer |
| G1b | Öngörü: `tests/toporlogy/baseline/resolved_configs.json` değişmez (yeni örnek YAML eklenmiyor); `baseline/list.txt`'de yalnız physics (6 → 8) ve response sayıları ile iki fizik ve iki yanıt satırı değişir | öngörüyle aynı |

Bir kontrol ilk koşuda tutmazsa eşik gevşetilmez: strict xfail olur ve ölçülen değer burada yazılır. İlk koşudan
sonra eklenen kontrol "post hoc" diye işaretlenir.

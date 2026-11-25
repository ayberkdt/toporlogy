# Başlangıç denetimi: ortam, CI, test tabanı ve örnek koşuları (26 Kasım 2026)

00_MASTER "Başlangıç denetimi". Etiketler: **[Ö]** bu oturumda ölçüldü (komut verildi), **[O]** dosyadan okundu,
**NOT VERIFIED** erişilemedi ya da koşulmadı.

## 1. Depo ve CI

* Taban: `main` = `1d1f027` (PR #19 birleşmesi) [O: `git log`]. Bu dal `claude/vv-infra-6ff2f9`, çalışma ağacı
  başlangıçta temizdi. Örnek koşuları `4629a99` üstünde, temiz ağaçta (`git_dirty: false`) [Ö].
* Açık PR'lar ve CI durumu: **NOT VERIFIED**. `gh` bu makinede yok (`where gh` bir şey bulmadı); GitHub API'sine
  istek gönderilmedi.
* `.github/workflows/` [O]:
  * `tests.yml` (her PR ve `main`'e push): ubuntu 3.11, ubuntu 3.12, windows 3.12; CPU torch + `.[post,dev]`;
    ruff, `pytest -m "not slow"`, ortam raporu artefaktı; ayrı iş `mypy src/toporlogy`.
  * `backends.yml` (elle, Pazartesi 03:00 UTC, çözücü/fem/post yollarına dokunan PR): PARDISO (`-m pardiso`),
    gmsh/T10 (`-m gmsh`, yavaşlar dahil), baseline sayısal kipte (Linux). CUDA işi yok: CI'da CUDA NOT VERIFIED.
  * macOS işi yok: macOS NOT VERIFIED.

## 2. Ortam [Ö]

Kaynak: `docs/olcumler/baseline_ornekler.json` → `environment` (betik `baseline_ornekler.py`).

| Öğe | Değer |
|:---|:---|
| İşletim sistemi | Windows-11-10.0.26200-SP0, AMD64, Intel64 Family 6 Model 158, 12 mantıksal çekirdek, 15,9 GiB RAM |
| GPU | NVIDIA GeForce GTX 1660 Ti (torch CUDA kullanılabilir) |
| Python, NumPy, SciPy, torch | 3.12.1; 2.2.6; 1.14.1; 2.5.1+cu121 |
| BLAS | NumPy: scipy-openblas 0.3.29 (DYNAMIC_ARCH, MAX_THREADS=24); torch: MKL 2024.2.2, OpenMP |
| İş parçacığı | torch 6 / interop 6; OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS ayarsız |
| İsteğe bağlı | pyvista 0.46.5, vtk 9.5.2, numba 0.63.1, scikit-fem 12.0.2; pypardiso, MKL paketi, gmsh **yok** |
| Araçlar | pytest 9.0.2 (pytest-randomly, pytest-timeout kurulu), hypothesis 6.155.2, mypy 2.1.0, ruff 0.17.2 |
| Doğrudan çözücü | SuperLU (torch toporlogy'den önce içe aktarılıyor; PARDISO yüklenmiyor) |
| Hassasiyet | float64 |

Baseline dosyasının (`tests/toporlogy/baseline/examples.json`) referans ortamı bu ortamla aynı (Python 3.12.1,
NumPy 2.2.6, SciPy 1.14.1, torch 2.5.1+cu121, Windows AMD64) [O]: bit düzeyinde karşılaştırma burada geçerli.

## 3. Test tabanı (değişikliklerden önce) [Ö]

Komut: `python -m pytest tests/toporlogy -m "not slow" -q -p no:randomly` (2026-11-26 00:21–00:34, `1d1f027` kodu).
Sonuç: **896 passed, 4 skipped, 55 deselected, 11 xfailed, 0 failed** (780 s). Doğrulama durumu: pardiso NOT
VERIFIED (pypardiso yok), gmsh NOT VERIFIED (gmsh yok), cuda VERIFIED (3 passed). Atlananlar: PARDISO, aerokit
(iki; depo dışı), gmsh. Yavaş paket (`-m slow`, 55 test) koşulmadı: NOT VERIFIED. Strict xfail'lerin ve
atlamaların kütüğü: `docs/vv_kutugu.md`.

## 4. Örnekler: tam `run.max_iter` ile bir koşu [Ö]

Komut: `python docs/olcumler/baseline_ornekler.py docs/olcumler/baseline_ornekler.json` (2026-11-26 00:49–00:55
+0300). On iki örnek (yedi `examples/*.yaml` + beş `examples/research/*.yaml`), her biri ayrı süreçte. Süreler tek
koşu, ısınma yok ve makine yükü denetlenmedi (aynı anda başka oturumlar koşuyordu): kayıttır, performans
ölçümü değildir. "Durum denklemi kalıntısı" son doğrusal çözümün en kötü yük durumu ‖Ku − f‖/‖f‖
(`physics.last_residual`; `Modal` kaydetmiyor: —). Tasarımın özeti (xphys SHA-256, ortalama) JSON'da.

Bulgu: on örnekten ikisi `ok`, sekizi `not converged` (çoğu `max_iter`'de durdu); iki gerilme örneği koşuyu
bitiremiyor.

| Örnek | Fizik | Optimizer | Durum | İterasyon | Amaç | Kısıtlar (değer/üst) | KKT kalıntısı | Durum denklemi kalıntısı | Süre (s) | Tepe bellek (MiB) |
|:---|:---|:---|:---|---:|:---|:---|:---|:---|---:|---:|
| cantilever3d | Elasticity | OC | not converged | 150 | 252.8 | volume 0.3/0.3 | 7.0e-01 | 1.9e-12 | 31.3 | 446 |
| eigenfrequency2d | Modal | MMA | not converged | 60 | -0.0001017 | volume 0.5/0.5 | 2.3e-03 | — | 12.4 | 424 |
| lbracket_bore3d | Elasticity | MMA | not converged | 40 | 666.9 | volume_dilated 0.5091/0.613; local_volume 0.3466/0.347; connectivity 9.758/20 | 3.9e-01 | 1.7e-11 | 114.3 | 807 |
| mbb2d | Elasticity | OC | ok | 160 | 219.7 | volume 0.5/0.5 | 1.5e-02 | 1.4e-12 | 15.3 | 426 |
| multiload3d | Elasticity | MMA | ok | 80 | 166 | volume 0.25/0.25 | 3.8e-03 | 1.8e-13 | 20.7 | 454 |
| split_connectivity2d | Elasticity | MMA | not converged | 120 | 4.181 | volume 0.15/0.15; connectivity 18.52/20 | 6.3e-03 | 3.5e-14 | 11.7 | 431 |
| symmetric_am3d | Elasticity | OC | not converged | 60 | 1.16 | volume 0.1978/0.2 | 5.5e-01 | 2.9e-14 | 47.8 | 509 |
| buckling_column2d | Elasticity | MMA | not converged | 80 | 12.87 | volume 0.35/0.35; buckling 45.81/50 | 3.9e-02 | 1.3e-14 | 28.6 | 433 |
| inverter2d | ConstrainedElasticity | MMA | not converged | 100 | -1.203 | volume 0.3/0.3 | 3.8e-02 | 7.5e-14 | 15.5 | 432 |
| mbb_simpl2d | Elasticity | SiMPL | not converged | 150 | 225.7 | volume 0.5/0.5 | — | 1.5e-12 | 30.4 | 425 |
| stress_al_lbracket2d | — | — | HATA exit 1 | — | — | — | — | — | — | — |
| stress_lbracket2d | — | — | HATA exit 1 | — | — | — | — | — | — | — |

Kısıtlar 4 anlamlı basamakla; amaç da öyle. Durum `not converged`: optimizer `max_iter`'e ulaştı ya da (symmetric_am3d)
takvim adımı son iterasyonda; bu bir başarısızlık değil, yakınsama iddiası da değil.

### 4.1 Gerilme örnekleri koşmuyor (FAIL, önceden var olan) [Ö]

`stress_lbracket2d` (10. iterasyondan sonra) ve `stress_al_lbracket2d` (11. iterasyonda `change: NaN`) şu hatayla
duruyor: `InvalidModuliError: modulus: 1024 (240) value(s) not finite (first at active element 0)`. Aynı iki
örnek `main` kodunun (`git archive 1d1f027 src`) ayrı bir kopyasıyla da aynı hatayı verdi: bu dalın değişikliği
değil. `tests/toporlogy/test_research_examples.py` örnekleri yalnız 4 iterasyon koşturduğu için hızlı paket bunu
görmüyor. Neden belirlenmedi; ayrı iş (sonuç notunda öneri).

## 5. Yeniden üretim

```
python docs/olcumler/baseline_ornekler.py <çıktı>.json [--only <örnek> ...]
python docs/olcumler/baseline_bagimlilik.py <çıktı>.json
```

GPU/CUDA ölçümü bu kayıtta yok (örnekler CPU yolunda); PARDISO ve gmsh kurulu olmadığı için onların yolları
ölçülmedi.

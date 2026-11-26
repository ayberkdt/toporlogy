# Optimizasyon döngüsünün hızlandırılması — ön kayıt

Zaman damgası: **2026-11-26 03:00:20 +0300** (`date` çıktısı; değişikliklerin "sonra" ölçümünden ve bu dosyadaki
testlerin ilk koşusundan önce). Taban: `10daa03` (`integrate/research-round-1`), dal `claude/perf-7c523d`.

Etiketler: **[Ö]** bu oturumda ölçüldü (kaynak verildi), **[O]** koddan okundu, **[K]** keşif ölçümü (ön kayıttan önce
alındı, hedefleri seçmek için kullanıldı; sonuç sayılmaz).

## 0. Keşif: zaman nereye gidiyor [K]

Kaynak: `docs/olcumler/hizlandirma_oncesi.json` (betik `docs/olcumler/hizlandirma_profil.py`, taban kodu, 02:41–03:00).
Her satır: 2 ısınma iterasyonu, sonra 3 (büyüklerde 2) × 5 (büyüklerde 3) iterasyon, iterasyon başına ortanca.
Makine boşta DEĞİL: dört başka oturum çalışıyordu, CPU ortancası satırların çoğunda %89–100 (JSON `load_cpu_percent`).
Torch 6 iş parçacığı, OpenBLAS ayarsız, SuperLU (PARDISO kurulu değil).

| Satır | Serbest DOF | İterasyon (s) | Çarpanlara ayırma | OC bisection (kısıt değerlendirmesi) | Geri kalan |
|---|---:|---:|---:|---:|---:|
| mbb2d | 2 540 | 0,076 | %29 | %54 | %17 |
| cantilever3d | 3 240 | 0,174 | %66 | %24 | %10 |
| multiload3d (MMA) | 3 780 | 0,259 | %78 | — | %22 |
| mbb2d 300×100 | 60 700 | 7,13 | %15 | **%78** | %7 |
| mbb2d 600×200 | 241 400 | 38,2 | %15 | **%67** | %18 |
| cantilever3d 40×20×10 | 27 720 | 19,2 | %33 | **%61** | %6 |
| multiload3d 40×16×12 (MMA) | 26 520 | 7,06 | %80 | — | %20 |

Okuma [K]:

1. **OC bisection'ı yük altında iş parçacığı çekişmesine boğuluyor.** 300×100'de on `Study.constraint_values`
   çağrısı torch 1 iş parçacığıyla 0,051 s, 6 iş parçacığıyla 2,82 s (55×; aynı dakika, CPU %100). cProfile:
   süre `Embed`'in `index_put`'u ve `Volume`'un boolean maske indeksinde; ikisi de 32 768 öğenin üstünde torch'un
   OpenMP paralel bölgesine giriyor ve yüklü makinede her bölge bütün iş parçacıklarının zaman dilimi almasını bekliyor.
   Küçük örneklerde (< 32 768 öğe) torch paralelleşmiyor, oran küçük.
2. Çarpanlara ayırma (SuperLU, iç içe bölme sırası, tek iş parçacığı) 3-B'de iterasyonun en büyük ikinci kalemi.
   Girdisinin hazırlanması (simetrikleştirme + iki permütasyon + CSC) her iterasyonda baştan yapılıyor [O].
3. Montaj, çözüm, kalıntı denetimi, filtre, duyarlılık ve MMA alt problemi (bu boyutlarda) tek tek %5'in altında;
   istisna `multiload3d`'de duyarlılık (iki yük durumu: 21 ms, tek durumlu `cantilever3d`'de 2 ms).
4. Keşif prototipi (bu kaydın H3'ü): tek iş parçacıklı süpernodal Cholesky `cantilever3d`'nin matrisinde (3 240 DOF)
   0,057–0,059 s, SuperLU 0,147–0,197 s, uyum farkı 1e-12 mertebesi; 4–6 iş parçacığında yük altında 2–9 s
   (çekişme). Daha büyük boyutta ölçülmedi.
5. H2'nin dizi eşitliği keşifte bir kez denendi (12 örnek × düzgün/rastgele alan, CSC dizileri bit düzeyinde aynı).
   Ön kayıtlı kontrol aynı denetimin teste dönüşmüş hâli (H2a).

## 1. Değişiklikler, hedefler ve eşikler

Doğruluk kapısı her değişiklik için ortak: **K0** `pytest tests/toporlogy/test_baseline.py -m baseline` bit düzeyinde
(8 passed, fark 0; taban bu oturumda 02:36–02:39 arası geçti, çıktı `docs/olcumler/hizlandirma_baseline_oncesi.txt`). **K1** gradyan testleri
(`test_gradients.py`, `test_taylor_remainder.py`) değişmeden geçer. **K2** hızlı paket değişmeden geçer (taban: 1149
passed, 22 xfailed, 0 failed, koordinasyon notu). Bit düzeyi tutmazsa: önce nedeni bul, bit düzeyini koruyan yolu
dene; olmuyorsa farkı ölç, yalnız yuvarlama düzeyindeyse (amaç ve kısıtlarda bağıl ≤ 1e-12) gerekçeli `--write`;
değilse değişiklik geri alınır.

Hız ölçümü: aynı betik, taban kodu (`git archive 10daa03 src`) ile yeni kod **aynı oturumda art arda**, aynı
satırlar; her satırda CPU yükü kaydı. Hedef, iterasyon başına ortanca süre oranı (taban / yeni). Yük farklıysa oran
yük bilgisiyle birlikte raporlanır; boşta makine ölçümü yapılamazsa NOT VERIFIED.

### H1 — Döngüde torch iş parçacığı politikası

`Study.run` döngüyü `torch.set_num_threads(torch_threads)` altında koşar (yeni `Study(..., torch_threads=1)`;
`None` = torch'un ayarına dokunma), çıkışta (hata da olsa) eski değer geri yüklenir. Gerekçe §0.1; ayrıca iş
parçacığı sayısına bağlı paralel indirgeme (toplam) sırası kalkar, sonuç çekirdek sayısından bağımsızlaşır.

| # | Kontrol | Eşik |
|---|---|---|
| H1a | OC bisection süresi (iterasyon başına, `opt_evals`), mbb2d 300×100 ve cantilever3d 40×20×10, yük altında | ≥ 10× azalır |
| H1b | İterasyon süresi, aynı iki satır | ≥ 2× azalır |
| H1c | `Study.run` sonrası `torch.get_num_threads()` çağrı öncesiyle aynı (normal bitiş ve hata) | tam |
| H1d | K0 (yedi örnek bit düzeyinde) | geçer |
| H1e | `torch_threads=None` ile sonuç, iş parçacığı ayarı değiştirilmeden alınan sonuçla bit düzeyinde aynı | tam |

Boşta makinede tek iş parçacığının bedeli (büyük eleman sayısında öğe öğe işlemler): NOT VERIFIED olarak kalırsa öyle
yazılır.

### H2 — SuperLU girdisinin önceden hesaplanmış indeks haritası

`DirectSolver.factor(A, order)`: simetrik tam matris + iki permütasyon + CSC dönüşümü yerine örüntü başına bir kez
kurulan harita (değerler kopyalanır, toplanmaz; tam sıfırlar eski yoldaki gibi atılır).

| # | Kontrol | Eşik |
|---|---|---|
| H2a | SuperLU'ya giden CSC dizileri (değer, satır indeksi, sütun göstergesi) eski yolla aynı: 12 örneğin matrisi × düzgün / rastgele alan, tam sıfırlı örüntü dahil | bit düzeyinde |
| H2b | `factor` süresi (girdi hazırlama dahil), mbb2d 300×100 ve cantilever3d 40×20×10 | ≥ %5 azalır |
| H2c | K0 | geçer |

### H3 — Süpernodal (multifrontal) Cholesky, isteğe bağlı

Yeni `toporlogy/solvers/supernodal.py`, `DirectSolver(method="cholesky")` (YAML: `physics.solver: {type: direct,
method: cholesky}`). İç içe bölmenin parçaları (yaprak kutular, ayırıcı düzlemler) yoğun bloklar; yoğun işlemler
torch (MKL) ile, iş parçacığı torch'un ayarı (H1 altında 1). Varsayılan çözücü DEĞİŞMEZ (SuperLU), K0 etkilenmez.

| # | Kontrol | Eşik |
|---|---|---|
| H3a | Çarpanlara ayırma süresi SuperLU-iç içe bölmeye göre, tek iş parçacığı, 3-B 40×20×10 ve 2-B 300×100 (aynı matris, aynı sıra) | ≥ 2× hızlı |
| H3b | Çözüm farkı SuperLU'ya göre: 16×8×8 ve 120×40, rastgele alan, kontrast 1e-9 | uyum ≤ 1e-10 bağıl, ‖ΔU‖/‖U‖ ≤ 1e-7 (cozucu_siralama O4 eşikleri) |
| H3c | Kısmi hacim + keep/void + eğik mesnet (`ConstrainedElasticity`), Q4 ve H8IM | H3b eşikleri |
| H3d | Uyum gradyanı (autograd) SuperLU'ya göre | ≤ 1e-9 bağıl |
| H3e | Aynı süreçte iki çarpanlara ayırma ve iki ayrı süreç: çözüm | bit düzeyinde aynı |
| H3f | SPD olmayan matris (negatif köşegen) ve tekil matris | `NotPositiveDefiniteError`, sessiz sonuç yok |
| H3g | Saklanan çarpan girdisi SuperLU'nun L + U girdisinden az (aynı iki matris) | kesin az |
| H3h | `Study` ile 10 iterasyon (mbb2d, cantilever3d): amaç geçmişi SuperLU'ya göre | her iterasyonda ≤ 1e-8 bağıl |

### Bu kayda girmeyen adaylar

Montaj (bincount), kalıntı denetimi, filtre, duyarlılık (çok yük durumlu einsum), `Volume`/`Embed` tensör önbelleği,
MMA alt problemi, modal ve burkulma fiziği: H1–H3'ten sonra yeniden profillenir; uygulanacak olanlar için bu dosyaya
**yeni zaman damgalı bölüm** eklenir (ilk ölçümlerinden önce). Uygulanmayanlar sonuç dosyasında ölçümüyle yazılır.

## 2. Ek kayıt: H4 (2026-11-26 03:47:27 +0300, değişiklik yazılmadan ve ölçülmeden önce)

H1–H3 sonrası keşif [K] (yeni kod, tek iş parçacığı, `cantilever3d` / `mbb2d`, 300 çağrı ortalaması, aynı süreç):
`Study.constraint_values` çağrısı 964 / 847 µs; bunun 493 / 471 µs'si zincir (Embed → yoğunluk filtresi → izdüşüm →
Passive), 166 / 153 µs'si `Volume` (her çağrıda `torch.as_tensor(domain.active)` ve `domain.n_active` = `active.sum()`),
81 / 75 µs'si interpolasyon (`design/interp.py`, bu oturumun dosyası değil). OC bisection'ı iterasyon başına ~35 çağrı:
bu boyutlarda iterasyonun %15–20'si (`hizlandirma_sonrasi.json`, küçük satırlar).

Ayrıca [K]: `torch.set_num_threads` yük altında pahalı (6'ya dönüş 0,16–0,27 s, ardından ilk paralel işlem 0,3–0,7 s;
1'e geçiş ilk seferden sonra ≤ 2 ms). H1 her `Study.run` başına bu bedeli bir kez öder; sonuç dosyasında ayrıca
raporlanır (hızlı paket süresi dahil).

### H4 — `Volume` ve `Passive`'in çağrı başı maliyeti

`Volume`: etkin eleman maskesi tensörü ve `n_active` alan nesnesi başına önbellekte (aynı işlem, aynı toplam sırası).
`Passive`: `torch.where(keep, 1, torch.where(off, 0, x))` yerine `x.masked_fill(off, 0).masked_fill(keep, 1)`
(yalnız seçme; değer ve gradyan aynı).

| # | Kontrol | Eşik |
|---|---|---|
| H4a | `Study.constraint_values` çağrı süresi, eski ve yeni uygulama aynı süreçte sırayla (cantilever3d, mbb2d) | ≥ %25 azalır |
| H4b | `Volume` ve `Passive` çıktısı ve gradyanı eski uygulamayla (rastgele alan, keep/void/dış elemanlı alan, iki alan) | bit düzeyinde aynı |
| H4c | K0 ve zincir gradyan testleri (`test_gradients.py`) | geçer |

## Ek ön kayıt: CLI süreç başında torch iş parçacığı (TT1–TT2)

Ön kayıt 2026-11-26 09:15 +0300, kod yazılmadan önce. `python -m toporlogy run` (tek koşu ve tarama) süreç başında
`torch.set_num_threads(n)` çağırır; n = `TOPORLOGY_TORCH_THREADS` ortam değişkeni, yoksa 1 (`Study.run`'ın
varsayılanı). Böylece her `Study.run` içindeki geçiş (ölçülen 0,2–0,7 s/koşu, yük altında) no-op olur.

| # | Kontrol | Eşik |
|:---|:---|:---|
| TT1 | `pytest -m baseline` (örnekler CLI'dan koşuyor) | bit düzeyinde aynı |
| TT2 | CLI koşusundan sonra süreçte `torch.get_num_threads()` = 1; ortam değişkeni 3 iken 3 | aynen |

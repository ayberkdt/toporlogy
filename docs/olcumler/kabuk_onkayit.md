# Düz parçalı kabuk elemanı (dikdörtgen parçalar, 6 DOF/düğüm) — ön kayıt

Ön kayıt 2026-11-26 16:49 +0300, kod yazılmadan önce. Yeni modül `fem/shell.py` (bağımsız; `Study` / YAML bağlantısı
bu turda yok). 3-B uzayda dikdörtgen parçalardan (a × b, kare şart değil) oluşan yüzey; düğüm başına global eksenlerde
(u, v, w, θx, θy, θz), θ sağ el kuralıyla dönme vektörü.

## Eleman

* **Parça çerçevesi.** Düğümler X1..X4 (yerel köşe sırası (−1,−1), (1,−1), (1,1), (−1,1)). e_x = (X2 − X1)/a,
  e_y = (X4 − X1)/b, e_z = e_x × e_y. Parça dikdörtgen değilse (‖X3 − X1 − (X2 − X1) − (X4 − X1)‖ > 1e-9 L ya da
  |(X2 − X1)·(X4 − X1)| > 1e-9 a b; düzlem dışı da buraya düşer) ya da kenarı sıfırsa açık hata. Global → yerel
  dönüşüm her düğümde blok R = [e_x; e_y; e_z] (satırlar), hem öteleme hem dönme üçlüsüne; K_g = Tᵀ K_l T.
* **Zar.** Bilineer düzlem gerilme, D = t/(1 − ν²) [[1, ν, 0], [ν, 1, 0], [0, 0, (1 − ν)/2]] (birim modül), 2 × 2 Gauss,
  det J = a b / 4.
* **Eğilme.** Mindlin–Reissner, `fem/plate.py` ile aynı formülasyon dikdörtgene genelleştirilmiş (∂/∂x = (2/a) ∂/∂ξ,
  ∂/∂y = (2/b) ∂/∂η): plaka DOF'ları (w, βx, βy), γ = ∇w − β, eğilme 2 × 2, kesme k_s = 5/6 ile tek nokta. Kod
  `plate.py`'den bağımsız yazılır (karşılaştırma kendi kendine olmasın diye); yalnız k_s sabiti ortak.
* **β ↔ θ bağıntısı (türetme; testle doğrulanacak).** Rijit dönme θ altında düğüm yer değiştirmesi θ × r. Yerel x ekseni
  üzerindeki nokta r = (x, 0, 0) için θ = (0, φ, 0) → w = −φ x; rijit harekette γ = 0 olmalı, yani βx = ∂w/∂x = −θy.
  r = (0, y, 0), θ = (φ, 0, 0) → w = φ y ⇒ βy = θx. Dolayısıyla **βx = −θy, βy = θx** (görev metnindeki örnek
  βx = θy, βy = −θx bu işaretlerle rijit dönmede kesme enerjisi üretir; KB2 negatif kontrolü bunu ölçer).
* **Dönme serbestliği (drilling).** Hughes–Brezzi türü ceza: Π_d = ½ γ_d ∫ (θz − ω)² dA, ω = ½ (∂v/∂x − ∂u/∂y)
  (yerel düzlem içi rijit dönme), θz bilineer bağımsız alan, **2 × 2 Gauss** (tek nokta θz'de hücre başına üç sahte
  sıfır mod bırakırdı). Ölçek γ_d = α G t, G = 1/(2(1 + ν)) (birim modül), varsayılan **α = 1e-3**. Rijit düzlem içi
  dönmede (u = −φy, v = φx, θz = φ) ω = θz, ceza enerjisizdir; düz levhada θz'yi tekil olmaktan kurtarır.
* **Atama.** K = Σ_e E_e K_e⁰ (K_e⁰ birim modüllü, global eksenlerde), seyrek; sabit DOF'lar sıfır. Çözüm torch autograd
  fonksiyonu: dc/dE_e = −λ_eᵀ K_e⁰ u_e (eşlenik), sağ taraf türevi λ.
* **Yük.** Düzgün basınç p parça normali e_z yönünde: her parça düğümüne p a b / 4 · e_z (bilineer N için ∫N dA).
* **Üreteçler.** Düz dikdörtgen (xy düzlemi, normal +z) ve silindir (eksen z, yarıçap R): açık dilim (açı ∈ (0, 2π))
  ya da kapalı boru; silindirde e_x ≈ e_φ, e_y = e_z ⇒ e_z = e_r (dışa doğru), iç basınç p > 0.

## Kapalı formlar (burada türetildi, kaynaktan okunmadı)

**(A) Uzun ince silindir, iç basınç, serbest uçlar (zar çözümü).** Yarıçap R, kalınlık t. Birim boylu yarım halkanın
dengesi: 2 N = p · 2R ⇒ halka kuvveti N = p R. Uçlar serbest ve eksenel yük yok ⇒ σ_eksenel = 0; düzlem gerilmede
ε_halka = σ_halka / E = N/(E t). Çevre 2πR → 2π(R + w), ε_halka = w / R ⇒ **w = p R² / (E t)**. Serbest uçta zar
çözümü N_x = 0, M = 0, Q = 0'ı zaten sağlar (sınır tabakası yok).

**(B) Aynı problemin ayrık çözümü (N_θ kenarlı çokgen boru).** Düğümler R yarıçapında, Δθ = 2π/N_θ, kiriş
a = 2R sin(Δθ/2). Simetriden aday alan: her düğüm radyal δ, eksenel u_z = ε_z (z − L/2), dönmeler 0. Parça çerçevesinde
kirişin iki ucu ±δ sin(Δθ/2) (düzlem içi, doğrusal), w_yerel = δ cos(Δθ/2) sabit, β = 0: eğilme ve drilling enerjisiz,
zar sabit şekil değiştirmeli (Q4 için tam). ε_x = 2δ sin(Δθ/2)/a = δ/R; σ_y = 0 ⇒ σ_x = E ε_x. Düğüm dengesi (iç
düğüm, dört parça): iki kirişin çekmesi içe 2 N b sin(Δθ/2); dış yük dört parçadan p a b/4 · e_z, radyal bileşen
toplamı p a b cos(Δθ/2) (teğet bileşenler sadeleşir) ⇒ N = p R cos(Δθ/2). Uç düğümlerde iki taraf da yarıdır. Eksenel
denge σ_y = 0 ile sağlanır. Sonuç: **δ = p R² cos(π/N_θ) / (E t)**, yani (A)'ya göre bağıl hata tam olarak
1 − cos(π/N_θ) (≈ π²/(2 N_θ²), mertebe 2). Rijit hareket, bu çözümün zaten sağladığı altı kısıtla kaldırılır: orta
halkada φ = 0 düğümünde u_y, u_z; φ = π/2'de u_x, u_z; φ = π'de u_y, u_z (ω × r açılımıyla altı rijit modun hepsini
yok ettiği kontrol edildi: u_z üçlüsü T_z, ω_x, ω_y'yi; u_y ikilisi T_y, ω_z'yi; u_x T_x'i).

**(C) Sıkıştırılmış halka (ν = 0, serbest uçlu kısa boru, çizgi yükü).** Birim boy başına P, φ = 0 ve φ = π
doğrularında içe doğru. ν = 0'da zar ve eğilme eksenel yönle bağlaşmaz, çözüm z'den bağımsız ⇒ birim genişlikli halka,
I = t³/12. Çift simetri: φ = π/2 kesitinde kesme kuvveti 0, basınç kuvveti P/2, bilinmeyen M0. Kesitten φ'ye
(0 ≤ φ ≤ π/2, φ = 0 yük noktası) moment, (0, R)'deki (P/2, 0) kuvvetinin kolu ile M(φ) = M0 − (P R/2)(1 − sin φ).
Simetri kesitinde dönme sıfır (Castigliano, ∂U/∂M0 = 0): ∫₀^{π/2} M dφ = 0 ⇒ M0 = (P R/2)(1 − 2/π),
M(φ) = (P R/2)(sin φ − 2/π). U = 4 ∫₀^{π/2} M² R dφ / (2 E I) = (P² R³ / (2 E I)) (π/4 − 2/π), çünkü
∫₀^{π/2} (sin φ − 2/π)² dφ = π/4 − 4/π + 2/π. Yükler arası yaklaşma **δ = ∂U/∂P = (π/4 − 2/π) P R³ / (E I)**.
İhmal edilen: zar ve enine kesme enerjisi (bağıl O((t/R)²); t/R = 0,01'de ~1e-4).

**(D) Navier plakası.** `plaka_onkayit.md`'de türetildi (PL3); burada `navier_deflection` ile.

## Kontroller

| # | Kontrol | Eşik |
|:---|:---|:---|
| KB1 | xy düzleminde kare parça (drilling açık): global matrisin {u, v, θz} ile {w, θx, θy} arası bloğu; zar parçası (`membrane` yerel matrisi) − t · `ke_q4(ν)`; (w, βx, βy)'ye eşlenmiş eğilme bloğu − `plate_element_matrix(h, t, ν)` | blok tam 0; ikisi ≤ 1e-12 max\|K\| |
| KB2 | Tek parça, rastgele yönelim: parça düzlemindeki iki eksen ve normal etrafında rijit dönme (u = θ × r) enerjisiz; negatif kontrol: βx = +θy, βy = −θx eşlemesiyle aynı dönme `plate_element_matrix` ile kesme enerjisi üretir | ‖K r‖ ≤ 1e-12 ‖K‖_F ‖r‖; negatif kontrolde enerji ≥ 1e-3 × (doğru eşlemeyle zar+eğilme ölçeği uᵀ diag(K) u) |
| KB3 | Yama testi, eşit olmayan dikdörtgenler (3 × 2 parça, farklı a, b), rastgele dönmüş düzlem: doğrusal zar alanı (θz = ω) enerjisi = ½ t A εᵀDε; sabit eğrilik w = x²/2, βx = x enerjisi = D11 A / 2 | ≤ 1e-12 göreli |
| KB4 | Serbest eğri silindir dilimi (R = 1, 90°, L = 1, 8 × 6 parça, t = 0,05, merkez dışı orijin): altı rijit mod; özdeğerlerden sahte mekanizma | ‖K r‖₂ ≤ 1e-10 ‖K‖_F ‖r‖₂ her mod için; 1e-10 λ_max altında tam 6 özdeğer |
| KB5 | Dönme değişmezliği: ankastre (bir kenar, 6 DOF) silindir dilimi, basınç + nokta kuvveti + nokta momenti; rastgele dönme + öteleme ile döndürülmüş yapı, döndürülmüş yükler | ½ Fᵀu farkı ≤ 1e-10 göreli |
| KB6 | Kapalı boru, iç basınç, R = 1, t = 0,01, L = 4, N_z = 8, N_θ = 16, 32, 64; (B)'deki ayrık çözüm: tüm düğümlerde u·e_r | ≤ 1e-8 göreli (her N_θ) |
| KB7 | Aynı, (A) zar çözümüne göre orta halka u·e_r hatası | N_θ = 64'te ≤ %1; hata 16 → 32 → 64 kesin azalan; 32 → 64 gözlenen mertebe ∈ [1,9; 2,1] |
| KB8 | Sıkıştırılmış halka (C): ν = 0, R = 1, t = 0,01, L = 0,4, N_z = 2, N_θ = 16, 32, 64; orta halkada yaklaşma | N_θ = 64'te ≤ %1; \|hata(64)\| < \|hata(16)\|; mertebe kayıt |
| KB9 | Kabuk parçalarından kare plaka, sert basit mesnet (w = 0; x = sabit kenarda θx = 0, y = sabit kenarda θy = 0; düzlem içi yalnız 3 rijit mod tutulur, θz serbest), (D) Mindlin Navier | a/t = 10 ve 100, n = 32: ≤ %1; 16 → 32 mertebe ≥ 1,5; 2 × 1 dikdörtgen plaka 32 × 32 parça (a ≠ b), a/t = 10 (kısa kenar): ≤ %1; n = 16, a/t = 10 merkez çökmesi = `MindlinPlate`: ≤ 1e-9 göreli |
| KB10 | Tasarım gradyanı: ankastre dilim, basınç, c = Fᵀu, E_e = 1e-6 + x_e³ (x ∈ [0,2; 0,9] rastgele), autograd; dördüncü mertebe merkezi fark [f(−2s) − 8f(−s) + 8f(s) − f(2s)]/(12s), s = 1e-3 | ≤ 1e-6 göreli |
| KB11 | t ≤ 0, ν ∉ (−1, 0,5), sıfır kenar, dikdörtgen olmayan (çarpık / düzlem dışı) parça, aralık dışı düğüm, α < 0, yanlış boyutlu E, açı ∉ (0, 2π) | açık hata |

Gerekçeler. KB10'da ikinci mertebe merkezi fark yerine dördüncü mertebe: PL6 aynı ölçütte 8,8e-7 ile sınıra yakındı
(adım 1e-6 yuvarlama ağırlıklı); s = 1e-3'te kesme hatası O(s⁴), yuvarlama O(ε/s). KB6'nın kapalı formu (B) ayrık
çözümün kendisi olduğu için ağ yakınsamasını KB7 ve KB8 ölçer. KB8 hatası geometri (çokgen) ve eleman hatalarının
toplamı, işaret değiştirebilir; bu yüzden yalnız uç ağlar karşılaştırılır.

Kayıt ölçümü (eşiksiz): drilling ölçeğinin etkisi, ankastre dilimin uç nokta yükü altındaki uyumu α = 1e-4, 1e-3, 1e-2
için.

Kurallar: ilk koşuda tutmayan kontrol, ölçülen değeriyle `xfail(strict=True)` olur ve `docs/vv_kutugu.json`'a girer;
eşik gevşetilmez; test kurulum hatası (ölçüm değil) düzeltilebilir ama burada yazılır; ilk koşudan sonra eklenen kontrol
"post hoc" diye işaretlenir.

## Sonuç (ilk ölçüm, 2026-11-26)

`tests/toporlogy/test_shell.py`. On bir kontrolden onu ilk koşuda geçti; KB3'ün eğilme yaması tutmadı (strict xfail,
`docs/vv_kutugu.json`). Değerler testlerin `-s` çıktısından.

| # | Durum | Ölçülen |
|:---|:---|:---|
| KB1 | PASS | {u, v, θz}–{w, θx, θy} bloğu tam 0; eğilme − `plate_element_matrix` 0,0; zar − t·`ke_q4` 2,8e-16 (drilling 0 iken global zar bloğu da 2,8e-16) |
| KB2 | PASS | üç eksen etrafında rijit dönme ≤ 5,7e-17; negatif kontrol (βx = +θy, βy = −θx): enerji / ölçek 0,90 |
| KB3 | **zar PASS, eğilme xfail** | zar yaması 1,6e-15; eğilme yaması **1,12e-12 > 1e-12** |
| KB4 | PASS | rijit artık ≤ 8,9e-18; 1e-10 λ_max altında tam 6 özdeğer; 7. özdeğer / λ_max 2,0e-7 |
| KB5 | PASS | enerji farkı 6,1e-13 göreli |
| KB6 | PASS | ayrık kapalı form (B): 4,2e-11 / 9,9e-11 / 6,2e-10 (N_θ = 16 / 32 / 64) |
| KB7 | PASS | zar çözümüne göre hata 1,92e-2 / 4,82e-3 / 1,20e-3 (= 1 − cos(π/N_θ) dört basamakta); mertebe 32 → 64: 1,9991 |
| KB8 | PASS | işaretli hata −9,70e-2 / −2,47e-2 / −6,09e-3 (N_θ = 16 / 32 / 64; eleman daha rijit, işaret değiştirmedi); mertebe 32 → 64: 2,02 (kayıt) |
| KB9 | PASS | kare, a/t = 10: 8,9e-4 / 2,2e-4 (n = 16 / 32), mertebe 2,00; a/t = 100: 1,26e-3 / 3,14e-4, mertebe 2,01; 2 × 1 dikdörtgen (a ≠ b): 3,4e-4; `MindlinPlate` ile n = 16 merkez çökmesi farkı 6,3e-14 |
| KB10 | PASS | 1,8e-11 göreli |
| KB11 | PASS | t = 0, ν = 0,5 ve −1, sıfır kenar, çarpık ve düzlem dışı parça, aralık dışı düğüm, α < 0, yanlış boyutlu E, açı 0 ve 2π: beklenen iletiyle açık hata |

**KB3 eğilme yaması (xfail).** Ölçülen 1,12e-12, eşik 1e-12; eşik değişmedi. Test bölündü: zar yaması
(`test_kb3_membrane_patch_on_unequal_rectangles`) zorunlu PASS olarak kaldı, eğilme yaması
(`test_kb3_bending_patch_on_unequal_rectangles`) ölçülen değerle strict xfail. Yorum (kanıtı aşağıda, nedeni
kesinleştirilmedi): t = 0,04 ve parça boyu 0,15–0,45 iken kesme rijitliği eğilmeden ~(a/t)² kat büyük; enerji
uᵀKu karesel formuyla, büyük terimlerin sadeleşmesiyle hesaplanıyor, yuvarlama tabanı 1e-12 civarında. Tanı ölçümleri
(eşiksiz, scratch betiği, ilk koşudan sonra): aynı yama dönmeden xy düzleminde global karesel formla 5,1e-13, parça parça
toplamla 7,4e-14; dönmüş düzlemde parça parça 2,7e-13.

**Post hoc (ilk koşudan sonra eklendi).** KB3b: aynı eğilme yaması t = 0,1 (a/t ≤ 12, PL2 gibi): 7,5e-14 (eşik 1e-12,
PASS). Bu, ön kayıtlı KB3'ün yerine geçmez.

**Kayıt ölçümü (eşiksiz): drilling ölçeği.** Ankastre 8 × 6 dilim, uç düğümde birim düşey kuvvet, uyum:
α = 1e-4: 184,90; 1e-3: 177,27; 1e-2: 175,33; 1e-4 ile 1e-2 arası fark uyumun %5,4'ü. Bu kaba eğri ağda drilling
cezası sonucu ihmal edilemeyecek ölçüde etkiliyor (α küçüldükçe yapı yumuşuyor). KB1–KB11'den hiçbiri bu etkiyi ölçmüyor:
basınçlı boru, halka ve düz plakada drilling enerjisi sıfır ya da ayrık. Varsayılan α = 1e-3 bir seçim, doğrulanmış bir
değer değil; eğri kabukta sonuç α'ya karşı ayrıca denetlenmeli.

**Kurulum notları.** Test kurulumunda ölçüm öncesi düzeltme yok. Testler ilk koşudan önce derlenip temizlendi: kullanılmayan
`K` parametresi ve ölü kod satırları kaldırıldı, davranış değişmedi. İlk koşudan sonra yalnız mypy tip notları eklendi
(`r: np.ndarray`, `u: np.ndarray`); bunlar sayıları değiştirmez.

**Görev metnindeki eşleme örneği.** Görevde örnek olarak verilen βx = θy, βy = −θx bu işaret kuralıyla yanlış:
KB2'nin negatif kontrolünde rijit dönme zar + eğilme ölçeğinin 0,90'ı kadar kesme enerjisi üretti. Modül
βx = −θy, βy = θx kullanır; KB2, KB3 ve KB4 bunu doğrular.

## Ek ön kayıt: kabuk `Study` içinde (KS1–KS5)

Ön kayıt 2026-11-26 17:05 +0300, kod yazılmadan önce. Örnek `optimization/mesh_study.py` (MD1–MD5): `Study`
değişmeden kalır; kabuğun parçaları 1-B bir "ızgara" olarak girer. Yeni modül `optimization/shell_study.py`
(`fem` katmanı `optimization` / `design`'ı içe aktarmaz, `test_architecture.py`):

* `ShellDomain(mesh)`: `ShellMesh` parçaları, `shape = (ne,)`, hepsi aktif ve dolu (`frac = 1`), `ndof = 6 nn`;
  `cell_volume` = parça alanı a b, `centroids` = dört köşenin ortalaması.
* Filtre: `MeshDensityFilter` merkezleri doğrudan da kabul edecek biçimde genelleştirilir (bir `Mesh` verilince
  davranışı aynı). Filtre halkası ve hacim yanıtı için yeni ad açılmaz: `MeshFilterLink` ve `MeshVolume` alanın
  `centroids` / `cell_volume` özniteliklerini okuyacak hâle getirilir, böylece `ShellDomain` ile de çalışır.
  Mesafe 3-B kiriş (Öklid) mesafesidir, yüzey üzerindeki jeodezik değil; hafif eğri dilimde fark küçük, ölçülmez.
  Kalınlık sabit olduğundan alanla ağırlıklı oran hacim oranıdır.
* `ShellPhysics(shell)`: `FlatShell` sarmalayıcısı, `solve(E, F)` → U (6 nn, n_yük), E'ye göre türevlenebilir
  (her yük sütunu `FlatShell.solve`; aynı E için çarpanlara ayırma önbellekte), `state_dofs = 6 nn`.

Ortak kurulum: R = 1, 90° açık dilim, L = 1, t = 0,05, ν = 0,3, ilk halka (z = 0) altı DOF ile ankastre;
zincir `Chain([Embed, MeshFilterLink, Passive])`, `SIMP(p = 3, E_min = 1e-9)`, `Compliance()`,
`Constraint(MeshVolume, 0,4)`, `MMA()`.

| # | Kontrol | Eşik |
|:---|:---|:---|
| KS1 | 8 × 6 dilim, iki yük durumu (basınç; serbest uç orta düğümünde nokta kuvveti), x ∈ [0,2; 0,9] rastgele: `Study.evaluate` değeri = doğrudan zincir (filtre matrisi `MeshDensityFilter(merkezler)` ile x̃ = H x, E = E_min + x̃³ (1 − E_min), u_c = `FlatShell.solve(E, F_c)`, c = Σ F_cᵀ u_c); hacim değeri = Σ A_e x̃_e / Σ A_e; Σ A_e = 8 · 2R sin(π/32) · L (çokgen dilimin alanı) | değer ≤ 1e-12 göreli; hacim ≤ 1e-12 göreli; alan ≤ 1e-12 göreli |
| KS2 | Aynı kurulum, `Study` gradyanı (filtre rmin = 0,3 + SIMP + kabuk, iki yük) rastgele yönde; dördüncü mertebe merkezi fark [f(−2s) − 8f(−s) + 8f(s) − f(2s)]/(12s), s = 1e-3 | ≤ 1e-6 göreli |
| KS3 | 16 × 12 dilim, serbest uç orta düğümünde içe radyal birim kuvvet, rmin = 0,15, hacim ≤ 0,4, 30 MMA iterasyonu | son uyum < ilk uyum; alanla ağırlıklı hacim ≤ 0,4 (1 + 1e-3) |
| KS4 | Yanlış satırlı yük (3 nn satır, plaka gibi): `Study` reddeder, ileti 6 nn durum DOF'unu adlandırır; `ShellPhysics.solve` da reddeder, ileti "6 per node" içerir | açık hata |
| KS5 | var olan kod: `MeshDensityFilter(mesh)` = `MeshDensityFilter(mesh.cell_centroids())` (bozulmuş Q4 ağı); `test_mesh_topology.py` değişmeden geçer; `pytest -m baseline` | filtre matrisleri bit düzeyinde eşit; MS/MD testleri geçer; baseline 8 passed bit düzeyinde |

Gerekçeler. KS2'de dördüncü mertebe fark: KB10 ve PL6'da ikinci mertebe merkezi farkla 1e-6 sınıra yakındı
(PL6: 8,8e-7); KB10 aynı kabukta dördüncü mertebeyle 1,8e-11 verdi. KS1 aynı sayıların iki yoldan hesabı olduğu
için eşik yuvarlama düzeyinde. KS3 yalnızca zincirin çalıştığını ve kısıtın tuttuğunu ölçer; tasarımın kalitesini
ölçmez.

Kurallar: ilk koşuda tutmayan kontrol, ölçülen değeriyle `xfail(strict=True)` olur ve `docs/vv_kutugu.json`'a girer;
eşik gevşetilmez; test kurulum hatası (ölçüm değil) düzeltilebilir ama burada yazılır; ilk koşudan sonra eklenen kontrol
"post hoc" diye işaretlenir.

### KS1–KS5 sonucu (ilk ölçüm, 2026-11-26)

`tests/toporlogy/test_shell_study.py` (5 test); beşi de ilk koşuda geçti. Değerler testlerin `-s` çıktısından.

| # | Durum | Ölçülen |
|:---|:---|:---|
| KS1 | PASS | değer 2,3e-14 göreli (c = 9564,5, iki yük durumu); hacim 1,9e-16; toplam alan 1,4e-16 göreli |
| KS2 | PASS | 8,3e-11 göreli (dördüncü mertebe fark, s = 1e-3) |
| KS3 | PASS | uyum 3,166e4 → 6,946e3; hacim 0,40000 (0,3999997); `max_iter = 30`, döngü 27 kayıttan sonra tasarım değişimi ≤ 0,01 ölçütüyle durdu (`termination = "design_change"`, `feasible = True`) |
| KS4 | PASS | `Study`: "… the physics 378 state DOFs" (6 · 63); `ShellPhysics.solve`: "… 378 state DOFs (6 per node)" |
| KS5 | PASS | bozulmuş Q4 ağında rmin = 0,2 ve 0,35 için filtre matrisleri (indptr, indices, data) bit düzeyinde eşit; `test_mesh_topology.py` değişmeden geçti; baseline 8 passed bit düzeyinde |

Notlar. KS3'te ön kayıttaki "30 MMA iterasyonu", MD4'teki gibi `max_iter = 30` olarak uygulandı; varsayılan
`tol = 0,01` ile döngü 27'de durdu (yeniden koşulmadı; eşikler son tasarım için). Test kurulumunda ölçüm öncesi ya da
sonrası düzeltme yok; ilk koşudan sonra yalnız mypy için `sd: Any` tip notu eklendi (sayıları değiştirmez).
`MeshFilterLink` ve `MeshVolume` artık alanın `centroids` / `cell_volume` özniteliklerini okuyor; `MeshDomain`
`centroids = mesh.cell_centroids()` taşıyor, böylece ağ yolu aynı sayıları üretiyor (KS5, MD1–MD4 geçti).

Kalan: YAML (`domain: {type: shell}`), ızgaraya özgü bileşenlerin kabuk alanında açık hatası, jeodezik filtre
mesafesi, drilling ölçeğinin (α) eğri kabuk optimizasyonuna etkisi (kayıt ölçümünde uyumun %5,4'ü kadar).

## Ek ön kayıt: drilling ölçeğine duyarlılık (DR0–DR3)

Ön kayıt 2026-11-26 17:51 +0300, ölçüm kodu yazılmadan ve koşulmadan önce. Soru: varsayılan α = 1e-3 (γ_d = α G t)
eğri dilimin statik uyumunu ve kısa bir kabuk optimizasyonunu ne kadar etkiliyor? α ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}.
Bu turda varsayılan değişmez (aşağıdaki karar kuralı); kural tutmazsa başarısız diye yazılır (strict xfail).

Ortak geometri: R = 1, 90° açık dilim, L = 1, ν = 0,3, ilk halka (z = 0) altı DOF ile ankastre, E = 1 (statik).
Yükler: (köşe) son düğümde (z = L, θ = 90°) birim z kuvveti, önceki kayıt ölçümünün yükü; (radyal) serbest halkanın
orta düğümünde içe radyal birim kuvvet, KS3'ün yükü. Kalınlıklar t/R = 0,01 ve 0,05.

| # | Kontrol | Eşik |
|:---|:---|:---|
| DR0 | kurulum: 8 × 6, t = 0,05, köşe yükü, α = 1e-4 / 1e-3 / 1e-2 uyumu = önceki kayıt 184,90 / 177,27 / 175,33 | her biri ±0,005 mutlak (kaydın iki ondalığı) |
| DR1 | statik, 16 × 12 ağ: her kalınlık (0,01; 0,05) ve her yük (köşe, radyal) için \|c(1e-4) − c(1e-2)\| / c(1e-3) | ≤ %1 (dört durumun her biri) |
| DR2 | optimizasyon, KS3 kurulumu (16 × 12, radyal yük, `Chain([Embed, MeshFilterLink(0,15), Passive])`, `SIMP(p = 3, E_min = 1e-9)`, `MeshVolume` ≤ 0,4, `MMA`, `max_iter = 30`, varsayılan `tol`), t = 0,05 ve t = 0,01: α = 1e-4 ile 1e-2 arasında (a) \|c_son(1e-4) − c_son(1e-2)\| / c_son(1e-3), (b) alanla ağırlıklı L1 Σ A_e \|x̄_a − x̄_b\| / Σ A_e | (a) ≤ %2, (b) ≤ 0,05 (iki kalınlığın her biri) |
| DR3 | kayıt (eşiksiz): beş α için statik uyum 8 × 6, 16 × 12, 32 × 24 ağlarında (iki kalınlık, iki yük), α = 1e-3'e göre bağıl fark; optimizasyonda beş α için son uyum, iterasyon sayısı, hacim, α = 1e-3 tasarımına göre max \|x̄_a − x̄_b\| ve L1; her son tasarımın α = 1e-3 fiziğiyle yeniden değerlendirilen uyumu / α = 1e-3 tasarımınınki | — |

Karar kuralı. DR1 ve DR2 tutarsa α = 1e-3 bu kurulumlarda kabul edilebilir: α 1e-4 ile 1e-2 arasında değişince statik
uyum %1'den, optimize uyum %2'den az değişir ve tasarım (L1) 0,05'ten az kayar. Tutmazsa varsayılan bu turda
**değişmez**: hangi α'nın "doğru" olduğunu söyleyecek bir referans (drilling'den bağımsız çözüm) bu ölçümde yok; kural
başarısız diye kaydedilir, test strict xfail olur, sonraki adım referansla ayrı bir ön kayıttır. DR3'teki ağ
incelmesiyle farkın küçülüp küçülmediği yalnız kayıttır.

Kurallar: ilk koşuda tutmayan kontrol, ölçülen değeriyle `xfail(strict=True)` olur ve `docs/vv_kutugu.json`'a girer;
eşik gevşetilmez; test kurulum hatası (ölçüm değil) düzeltilebilir ama burada yazılır; ilk koşudan sonra eklenen kontrol
"post hoc" diye işaretlenir.

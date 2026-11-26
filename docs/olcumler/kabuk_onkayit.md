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

### DR0–DR3 sonucu (ilk ölçüm, 2026-11-26)

`tests/toporlogy/test_shell_drilling.py` (8 test). Değerler ilk koşunun `-s` çıktısından; testlerde ilk koşudan önce ya
da sonra kurulum düzeltmesi yok. Ön kayıtlı karar kuralı **tutmadı**: DR1 dört durumdan üçünde, DR2 iki kalınlıktan
birinde başarısız. Kural gereği varsayılan α = 1e-3 **değişmedi**; başarısız durumlar ölçülen değerle strict xfail ve
`docs/vv_kutugu.json`'da.

| # | Durum | Ölçülen |
|:---|:---|:---|
| DR0 | PASS | 184,9039 / 177,2657 / 175,3294 (α = 1e-4 / 1e-3 / 1e-2; kayıt 184,90 / 177,27 / 175,33) |
| DR1 t = 0,01, köşe | PASS | yayılım 7,2e-3 (eşik 1e-2) |
| DR1 t = 0,01, radyal | **xfail** | yayılım 2,50e-2 > 1e-2 |
| DR1 t = 0,05, köşe | **xfail** | yayılım 5,57e-2 > 1e-2 |
| DR1 t = 0,05, radyal | **xfail** | yayılım 2,13e-1 > 1e-2 |
| DR2 t = 0,05 (KS3) | **xfail** | (a) son uyum yayılımı 5,80e-2 > 2e-2; (b) L1 1,35e-2 ≤ 0,05 tuttu |
| DR2 t = 0,01 | PASS | (a) 1,25e-2 ≤ 2e-2; (b) L1 2,36e-3 ≤ 0,05 |

Yayılım = \|c(1e-4) − c(1e-2)\| / c(1e-3).

**DR3 (kayıt): statik uyum α'ya göre ağ incelirken küçülmüyor.** c(1e-3) birim modülle; diğer sütunlar
c(α) / c(1e-3) − 1, yüzde; son sütun yayılım, yüzde.

| Ağ | t/R | Yük | c(1e-3) | α = 1e-5 | α = 1e-4 | α = 1e-2 | α = 1e-1 | Yayılım 1e-4…1e-2 |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|
| 8 × 6 | 0,01 | köşe | 1 169 | 2,9 | 0,5 | −0,1 | −0,2 | 0,6 |
| 16 × 12 | 0,01 | köşe | 1 492 | 4,2 | 0,6 | −0,1 | −0,2 | 0,7 |
| 32 × 24 | 0,01 | köşe | 1 700 | 4,2 | 0,6 | −0,1 | −0,1 | 0,6 |
| 8 × 6 | 0,01 | radyal | 43 418 | 8,1 | 2,1 | −0,3 | −0,6 | 2,4 |
| 16 × 12 | 0,01 | radyal | 55 977 | 11,4 | 2,2 | −0,3 | −0,4 | 2,5 |
| 32 × 24 | 0,01 | radyal | 60 901 | 11,6 | 1,8 | −0,2 | −0,3 | 2,0 |
| 8 × 6 | 0,05 | köşe | 177 | 7,7 | 4,3 | −1,1 | −1,3 | 5,4 |
| 16 × 12 | 0,05 | köşe | 213 | 9,0 | 4,6 | −1,0 | −1,2 | 5,6 |
| 32 × 24 | 0,05 | köşe | 245 | 8,6 | 4,2 | −0,9 | −1,0 | 5,1 |
| 8 × 6 | 0,05 | radyal | 1 853 | 27,3 | 15,8 | −5,1 | −6,0 | 20,9 |
| 16 × 12 | 0,05 | radyal | 2 026 | 32,8 | 17,1 | −4,2 | −4,8 | 21,3 |
| 32 × 24 | 0,05 | radyal | 2 089 | 34,6 | 17,1 | −3,6 | −4,1 | 20,7 |

Farkın kaynağı ağ kabalığı değil: 8 × 6'dan 32 × 24'e yayılım t = 0,05 radyal yükte %20,9 → %20,7, köşe yükünde
%5,4 → %5,1 kalıyor. α küçüldükçe yapı her durumda yumuşuyor; α = 1e-2 ile 1e-1 arası fark %1'in altında (en çok 0,9
puan, 8 × 6, t = 0,05 radyal), yani büyük α'da uyum doyuyor, küçük α'da doymuyor (α = 1e-5'te 32 × 24, t = 0,05
radyal +%34,6). Bu ölçüm hangi
α'nın doğru olduğunu söylemez; drilling'den bağımsız bir referans gerekir.

**DR3 (kayıt): optimizasyon.** KS3 kurulumu, `max_iter = 30`. Δx̄ ve L1 α = 1e-3 tasarımına göre; "Çapraz" her son
tasarımın α = 1e-3 fiziğiyle değerlendirilen uyumunun α = 1e-3 tasarımınınkine bağıl farkı (yüzde).

| t/R | α | c_son | İter. | Hacim | En büyük Δx̄ | L1 | Çapraz (%) |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0,05 | 1e-5 | 7 518 | 30 | 0,4000 | 0,189 | 0,0160 | 0,74 |
| 0,05 | 1e-4 | 7 248 | 30 | 0,4000 | 0,141 | 0,0108 | 0,30 |
| 0,05 | 1e-3 | 6 946 | 27 | 0,4000 | 0,000 | 0,0000 | 0,00 |
| 0,05 | 1e-2 | 6 845 | 30 | 0,4000 | 0,053 | 0,0029 | 0,03 |
| 0,05 | 1e-1 | 6 824 | 28 | 0,4000 | 0,048 | 0,0028 | 0,03 |
| 0,01 | 1e-5 | 139 355 | 27 | 0,4000 | 0,084 | 0,0079 | 0,22 |
| 0,01 | 1e-4 | 134 584 | 30 | 0,4000 | 0,025 | 0,0022 | −0,01 |
| 0,01 | 1e-3 | 133 174 | 30 | 0,4000 | 0,000 | 0,0000 | 0,00 |
| 0,01 | 1e-2 | 132 914 | 30 | 0,4000 | 0,005 | 0,0006 | 0,00 |
| 0,01 | 1e-1 | 132 766 | 30 | 0,4000 | 0,005 | 0,0006 | 0,00 |

Son uyum α ile fiziğin kendisi kadar değişiyor (t = 0,05'te %5,8), tasarım ise az: başka α ile bulunan tasarımlar
α = 1e-3 fiziğiyle değerlendirildiğinde α = 1e-3 tasarımından en çok %0,74 (α = 1e-5), α = 1e-4 ile 1e-1 arasında
en çok %0,30 kötü. Yani bu kurulumda α tasarımı değil, raporlanan uyum değerini taşıyor. Birkaç koşu
`max_iter = 30`'da durdu (`tol` ölçütüne varmadan); tasarım farklarının bir kısmı yakınsamamış iterasyondan olabilir,
ayrıca ölçülmedi.

Sonraki adım (ayrı ön kayıt): drilling'den bağımsız bir referans (örneğin aynı dilimin katı / düzgün kabuk çözümü ya
da drilling serbestliğini kullanmayan bir eleman) ile hangi α aralığının uyumu doğru verdiğini ölçmek; ancak ondan
sonra varsayılan değişebilir.

## Ek ön kayıt: drilling'den bağımsız referans, katı model (DB0–DB5)

Ön kayıt 2026-11-26 19:43 +0300, referans kodu yazılmadan ve koşulmadan önce. Ön kayıttan önce yalnız bir zamanlama
denemesi yapıldı (T10 ağında K_ff çarpanlaması; uyum hesaplanmadı, yazdırılmadı): 64 × 48 × 1 T10 111 456 DOF,
SuperLU 12,5 s, tepe bellek 1,8 GB; makinede o an 2,8 GB boş bellek vardı. Bu yüzden kalınlık yönünde bir T10
katmanı temel alındı (64 × 48 × 2 yaklaşık 3,5 GB isterdi), iki katman DB2'de daha kaba ağda denetlenir.

Soru: DR'deki dilimin drilling ölçeğinden (α) bağımsız bir 3-B katı çözümü hangi α'nın uyumu doğru verdiğini söylüyor
mu; varsayılan α = 1e-3 kalmalı mı?

**Eleman seçimi.** `fem.mesh` 3-B elemanları (`reference.py`, `ELEMENTS`): H8, T4, T10. H20 / H27 yok; uyumsuz modlu
H8 (`H8IM`) yalnız düzgün ızgarada (`fem.elastic`), eğri dilime eşlenemez. Eğilmede H8 kesme kilitlenmesi yapar, T10
(ikinci derece dörtyüzlü) yapmaz ölçüsünde değil ama çok daha az yapar; referans **T10**. H8 yalnız kayıt (DB5).

**Geometri.** R = 1 (orta yüzey), θ ∈ [0, π/2], z ∈ [0, L], L = 1, ν = 0,3, E = 1, t ∈ {0,01; 0,05}. Katı:
`box(nθ, nz, nr, π/2, 1, t, element="T10", origin=(0, 0, R − t/2))`, her düğüm (orta kenar düğümleri dahil, böylece
kenarlar izoparametrik eğri) (x, y, z) → (z cos x, z sin x, y) ile eşlenir; Jacobian det = r > 0 (sütunlar ∂/∂θ =
(−r sin θ, r cos θ, 0), ∂/∂z = (0, 0, 1), ∂/∂r = (cos θ, sin θ, 0)). Malzeme `isotropic_stiffness(1, 0,3)`. Ankastre:
`node_sets["bottom"]` (eksenel z = 0 yüzü), üç bileşen. Kabuk: `cylinder_surface(1, 1, nθ, nz, angle=π/2)`, ilk halka
altı DOF ile ankastre (DR ile aynı), `FlatShell(…, drill=α)`.

**Yük (yama).** Nokta yükü iki modelde de ağ incelirken ıraksar (türetme aşağıda), bu yüzden kabuğun orta yüzey yük
noktasında aynı yama iki modele de uygulanır:
* köşe: θ ∈ [7π/16, π/2], z ∈ [5/6, 1], yön d = +e_z (DR köşe yükünün noktası yamanın köşesi);
* radyal: θ ∈ [π/4 − π/32, π/4 + π/32], z ∈ [5/6, 1], yön d = −(cos π/4, sin π/4, 0) sabit vektör (KS3 / DR radyal yük
  noktası yamanın serbest kenar ortası).
Yama ağla hizalı: nθ 16'nın, nz 6'nın katı. Kabukta merkezi yamada olan parçalar, her birine A_e / A_P · d, köşe başına
dörtte biri (öteleme DOF'ları); toplam tam d. Katıda merkezi yamada olan hücrelere düzgün hacim kuvveti b = d / V_P,h
(V_P,h ayrık yama hacmi), tutarlı yük; birim ayrışımı ile Σ F = b V_P,h = d. Kalınlık boyunca düzgün olduğundan bileşke
orta yüzeyde (r̄ = ∫r² dr / ∫r dr = R + t²/(12R)). Ölçülen: **c = Fᵀu** (birim toplam kuvvet; yamada kuvvetle ağırlıklı
ortalama yer değiştirmenin d bileşeni).

*Nokta yükünün ıraksaması (türetme; eşik yok).* Mindlin plakasında w = w_b + w_s, kesme kısmı k_s G t ∇² w_s = −q.
Nokta kuvvet F için ∇² nin temel çözümü (1/2π) ln ρ ⇒ w_s ≈ −F ln ρ / (2π k_s G t): yük noktasında sonsuz, ağ h ile
c ≈ sabit + F² ln(1/h) / (2π k_s G t). Serbest düz kenarda yansıma (görüntü kaynak) çarpanı yaklaşık 2, 90° serbest
köşede yaklaşık 4. Köşe, t = 0,05: 4 ln 2 / (2π · (5/6) · (1/2,6) · 0,05) = 2,7726 / 0,10069 = 27,5 her ağ ikilemesinde;
DR3'te 8 × 6 → 16 × 12 → 32 × 24 artışlar +36, +32 (aynı mertebe; bu bir kanıt değil, yamanın gerekçesi). Katıda
nokta kuvvet Kelvin / Boussinesq 1/ρ tekilliği verir, uyum sonsuzdur.

**Kapalı formlar.** Katı hacim ∫₀^{π/2} ∫₀^L ∫_{R−t/2}^{R+t/2} r dr dz dθ = (π/2) L R t. Yama hacmi aynı yoldan
V_P = Δθ Δz R t = (π/16)(1/6) R t. Richardson (oran 2, c_h = c_∞ + C h^p): c₁, c₂, c₃ = 16 × 12, 32 × 24, 64 × 48;
Δ₁ = c₂ − c₁ = −C h₃^p 2^p (2^p − 1), Δ₂ = c₃ − c₂ = −C h₃^p (2^p − 1) ⇒ Δ₁/Δ₂ = 2^p, p = log₂(Δ₁/Δ₂),
c_∞ = c₃ − C h₃^p = c₃ + Δ₂/(2^p − 1) = **c₃ + Δ₂² / (Δ₁ − Δ₂)**; hata kestirimi ε_ref = |c_∞ − c₃| / |c_∞|.

| # | Kontrol | Eşik |
|:---|:---|:---|
| DB0 | kurulum (hızlı, varsayılan takım): T10 16 × 12 × 1, iki kalınlık: (a) Jacobian denetimi hatasız, Σ hücre hacmi / ((π/2) R t L) − 1; (b) yama hacmi V_P,h / ((π/16)(1/6) R t) − 1; (c) iki yükün toplam kuvveti = d, katı ve kabukta (16 × 12) | (a), (b) ≤ 1e-4 göreli; (c) ≤ 1e-12 |
| DB1 | referans yakınsaması: dört durum (iki t, iki yük), T10 nr = 1, 16 × 12, 32 × 24, 64 × 48: Δ₁ Δ₂ > 0 ve \|Δ₂\| < \|Δ₁\| (tek yönlü, büzülen), ε_ref | ε_ref ≤ %1 (her durum) |
| DB2 | kalınlık yönü: \|c(32 × 24 × 2) − c(32 × 24 × 1)\| / c(32 × 24 × 1) | ≤ %0,5 (her durum) |
| DB3 | kabuk ağ yakınsaması (yama yükü): \|c(128 × 96) − c(64 × 48)\| / c(128 × 96), beş α × dört durum | ≤ %1 (her biri) |
| DB4 | karar kuralının sonucu α* (aşağıda) = `toporlogy.fem.shell.DRILL` | eşit |
| DB5 | kayıt (eşiksiz): c₁, c₂, c₃, p, c_∞, ε_ref; kabuk 16 × 12 … 128 × 96 beş α (ve kabuğun kendi Richardson değeri); d(α) tablosu; H8 nr = 2, 64 × 48 ve 128 × 96, c / c_∞ (kilitlenme) | — |

**Karar kuralı.** d(α, k) = c_kabuk(α, 128 × 96, k) / c_∞(k) − 1 (k: dört durum), E(α) = max_k \|d(α, k)\|.
DB1, DB2, DB3'ün hepsi tutarsa: E(1e-3) ≤ %3 ve E(1e-3) − min_α E(α) ≤ 1 yüzde puanı ise α = 1e-3 kalır; değilse
varsayılan α* = argmin_α E(α) olur (α ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}; eşitlikte log ölçekte 1e-3'e yakın olan) ve
burada yazılır. DB1–DB3'ten biri tutmazsa referans güvenilir sayılmaz: kural uygulanmaz, α* = 1e-3 (değişmez),
tutmayan kontrol strict xfail. Varsayılan değişirse `test_shell.py`, `test_shell_study.py`, `test_shell_drilling.py`
yeniden koşulur; ön kayıtlı eşiği yeni varsayılanla tutmayan kontrol ölçülen değerle strict xfail olur (eşik
gevşetilmez); değişen ölçülen değerler burada yazılır, eski kayıtlar eski sayılarıyla ve bir notla kalır.

Kurallar: ilk koşuda tutmayan kontrol, ölçülen değeriyle `xfail(strict=True)` olur ve `docs/vv_kutugu.json`'a girer;
eşik gevşetilmez; test kurulum hatası (ölçüm değil) düzeltilebilir ama burada yazılır; ilk koşudan sonra eklenen kontrol
"post hoc" diye işaretlenir. Uzun hesaplar (DB1–DB5) `@pytest.mark.slow`; DB0 varsayılan takımda.

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

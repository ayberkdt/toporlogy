# Yol bağımlı malzemede tasarım duyarlılığı (J2 plastisite): ön kayıt (PS0–PS9)

Ön kayıt 2026-11-26 12:36 +0300, kod yazılmadan ve hiçbir koşudan önce. Taban `cc5d5b9` (`research/round-2`).
Taban çapası: `python docs/olcumler/nonlineer_cozucu_parmakizi.py` bu tabanda 13/13 bit düzeyinde eşit (12:36'da
koşuldu; çıktı "13 of 13 configurations equal").

## Sorun

`NonlinearElasticity.solve` durumlu (yol bağımlı) modelde tasarım duyarlılığını reddediyor (`fem/nonlinear.py` modül
notları; `nonlineer_cozucu.md` §5 "Ertelenen: yol bağımlı adjoint"). Son tanjantla tek adjoint çözümü tarihçeyi
görmez: k. artıştaki denge f_int(u_k, s_{k−1}; E) = λ_k F, iç değişkenlerin güncellemesi s_k = S(u_k, s_{k−1}).

## Seçilen yöntem (tam ayrık yol adjoint'i, kısıtlı sürüm değil)

İleri çözüm yakınsamış artışların (λ_k, u_k, s_{k−1}) dizisini kaydeder. J = j(u_N) + Σ_k D_k(u_k, s_{k−1}; E)
için geri yönde, k = N … 1:

    ū_k = ∂J/∂u_k (açık) + (∂S_k/∂u_k)ᵀ ν_k ,        K_T,k λ_k = ū_k   (K_T,k: (u_k, s_{k−1})'deki algoritmik tanjant)
    ν_{k−1} = ∂J/∂s_{k−1} (açık) + (∂S_k/∂s_{k−1})ᵀ ν_k − (∂R_k/∂s_{k−1})ᵀ λ_k ,     ν_N = 0
    dJ/dE_e = ∂J/∂E_e (açık) − Σ_k λ_{k,e}ᵀ ∂f_int,e/∂E_e(u_k, s_{k−1})

Yerel türevler Gauss noktası başına, J2 geri dönüş eşlemesinin (radyal dönüş) kapalı biçimli vektör–Jacobian
çarpımıyla (`J2Plasticity.evaluate_vjp`; σ, ε_p, α, Δα kotanjantlarından ε, ε_p^n, α_n kotanjantlarına; Voce'de
Δγ'nın kapalı türevi g(Δγ) = 0'dan örtük). Birim modül: iç değişkenler E_e'ye bağlı değil, E_e yalnız kuvveti ve
dissipasyonu ölçekler. Elastik noktalarda ∂S/∂u = 0; sağ tarafı tam sıfır olan artışta çözüm atlanır (elastik yolda
tek adjoint çözümüne iner). Akma kümesi (her artışta hangi noktanın plastik olduğu) sabit tutulur: türev tek taraflı
klasik türevdir; f* = 0'a tam düşen nokta ölçü sıfır, ele alınmaz. Yük adımı dizisi (bölmeler dahil) tasarıma
bağlı sayılmaz (ileri çözümün kabul ettiği dizi).

Yeni API: `solve` durumlu modelde türevlenebilir (yük 0 → 1 orantılı); `NonlinearElasticity.last_paths`;
`optimization.nonlinear.PlasticDissipation` (Σ_c w_c Σ_k Σ_e E_e Σ_g w σ_y Δα; YAML kaydı yok). `evaluate_vjp`
olmayan durumlu model açık hatayla reddedilir. `EndStrainEnergy` durumlu modeli reddetmeye devam eder.
`test_nonlinear_solvers.py::test_stateful_model_refuses_autograd_interpolation_and_warm_start`'ın son maddesi
(`solve` reddi) bu işin kaldırdığı sözleşmedir: `evaluate_vjp`'siz durumlu modelin reddiyle değiştirilir (planlı,
ön kayıtlı değişiklik); diğer iki maddesi (enerji interpolasyonu, warm start reddi) aynen kalır.

## Ortak kurulum

Q4 düzlem gerinim konsol 12 × 4 (h = 1), sol kenar ankastre, uçta düşey yük; ν = 0,3, σ_y = 1e-3, H = 0,05
(doğrusal pekleşme; Voce: σ_∞ = 1,5e-3, δ = 500); P = 1,6 × (homojen kirişin kök ilk akması σ_y·4²/(6·12));
E_e = 0,5 + 0,5·U(0, 1) (tohum 3); 10 yük adımı, rtol = 1e-12, algoritmik tanjant. H8: 6 × 2 × 2 konsol, aynı malzeme,
P = 1,6 × σ_y·2³/(6·6) (kare kesit 2 × 2). Merkezi fark adımı h = 1e-5 (mutlak, E_e ∈ [0,5; 1]); 8 bileşen (tohum 4,
en az 4'ü plastik noktalı eleman). Ölçü: ‖g − g_FD‖∞ / ‖g_FD‖∞.

## Kontroller

| # | Kontrol | Eşik |
|:---|:---|:---|
| PS0 | FD problemlerinin geçerliliği (ön koşul): en az 3 yakınsamış artışta plastik nokta var; hiçbir çözümde bölme yok; her denetlenen bileşende E, E ± h e_i çözümlerinin artış başına plastik nokta kümesi aynı | hepsi tutmalı; tutmazsa ilgili FD kontrolü geçersiz (FAIL olarak kaydedilir) |
| PS1 | Elastik limit: σ_y = 1e30 J2, uç uyumu gradyanı (yol adjoint'i, 10 adım) ↔ (a) `LinearElastic` modeliyle mevcut tek adjoint; (b) doğrusal `Elasticity` uyumunun −u_eᵀK0u_e gradyanı. Q4 düzlem gerinim ve H8 | ‖Δg‖∞/‖g‖∞ ≤ 1e-10 |
| PS2a | Uç uyumu gradyanı ↔ merkezi FD, Q4, doğrusal pekleşme, iki yük durumu (uç düşey; uç düşey + yatay ½), ağırlık (1; 0,5) | ≤ 1e-5 |
| PS2b | PS2a, Voce pekleşme (tek yük durumu) | ≤ 1e-5 |
| PS2c | PS2a, H8 3-B (tek yük durumu) | ≤ 1e-5 |
| PS3a | Yerel VJP ↔ `update`'in merkezi FD'si (ε, ε_p^n, α_n yönünde; çıktılar σ, ε_p, α, Δα), 20 rastgele nokta (≥ 10 plastik), doğrusal ve Voce | göreli ≤ 1e-6 |
| PS3b | Yalnız σ̄ kotanjantlı VJP'nin ε̄'si ↔ algoritmik tanjant C_algᵀσ̄ (aynı noktalar) | göreli ≤ 1e-12 |
| PS4 | `PlasticDissipation` gradyanı ↔ merkezi FD (Q4, doğrusal pekleşme, PS0 ön koşuluyla); D > 0 | ≤ 1e-5 |
| PS5 | Newton iterasyon sayısından bağımsızlık: ileri çözüm süreklilik tanjantıyla (toplam iterasyon algoritmikten fazla olmalı — ön koşul) ↔ algoritmik tanjantla; uç uyumu ve dissipasyon gradyanları | ≤ 1e-8 |
| PS6 | Negatif kontrol: tarihçeyi yok sayan "son tanjant" gradyanı (−λ_Nᵀ f_unit(u_N, s_{N−1}), K_T,N λ_N = ∂J/∂u_N) FD'den ayrılmalı (PS2a'nın ayırt ediciliği) | ‖g_son − g_FD‖∞/‖g_FD‖∞ ≥ 1e-2 |
| PS7 | Hiperelastik yol değişmedi: `nonlineer_cozucu_parmakizi.py` 13/13 bit düzeyinde; `pytest -m baseline` 8 passed bit düzeyinde; mevcut nonlineer testler (`test_nonlinear.py`, `test_model_epoch.py`, `test_nonlinear_solvers.py`, `test_nonlinear_orientation.py`, `test_taylor_remainder.py`, `test_frame_invariance.py`, `test_fem_mms_extended.py`) yeni başarısızlık yok | eşitlik / 0 yeni FAIL |
| PS8 | `Study` içinden (yoğunluk süzgeci + SIMP p = 3, x ∈ [0,4; 1]) J2 fiziğiyle `EndCompliance` gradyanı ↔ merkezi FD (x'te h = 1e-5, 6 bileşen) | ≤ 1e-5 |
| PS9 | Kapalı-hata sözleşmeleri: `evaluate_vjp`'siz durumlu model → `NotImplementedError`; tarihçesi verilen E'ye ait olmayan yol adjoint'i → `ValueError`; `PlasticDissipation` hiperelastik fizikte → `ValueError`; `EndStrainEnergy` J2'de → `ValueError` (mevcut) | açık hata |

Ölçüm (eşiksiz): geri geçişte adjoint çözüm sayısı (sağ tarafı sıfır olmayan artış sayısı) ve süre / ileri süre; 10
ile 20 yük adımı arasında J ve gradyanın göreli farkı (ayrıklaştırma etkisi; adjoint her ayrıklaştırma için tam).

Kurallar: başarısız kontrol ölçülen değerle strict xfail olur; eşik gevşetilmez; ilk koşudan sonra eklenen kontrol
"post hoc" etiketlenir. Ölçülen değerler bu dosyanın sonuna yazılır.

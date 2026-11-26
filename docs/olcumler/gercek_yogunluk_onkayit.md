# Enerji interpolasyonunda gerçek yoğunluk (E4 düzeltmesi) — ön kayıt

Ön kayıt 2026-11-26 10:53 +0300, kod yazılmadan önce. Sorun (`nonlineer_cozucu.md` E4): nonlineer fizik yalnız modülü görür ve
ρ̃ = (E/E0)^(1/3) ile yoğunluğu geri kazanır. SIMP üssü 3 değilken (sürdürmede p = 1, 2) γ yanlış yoğunlukta
değerlendirilir; ölçülen en büyük hata 0,385.

Değişiklik:
* `EnergyInterpolation.gamma(E, cutoff, density=None)`: `density = (ρ, dρ/dE)` verilirse γ onunla hesaplanır.
* `NonlinearElasticity.density_map`: E (aktif elemanlar) → (ρ, dρ/dE); None ise eski kural (ρ̃ = (E/E0)^(1/p)).
* `design.interp.inverse_modulus(interp, E, frac)`: SIMP ve RAMP için x̄ = E⁻¹(E) ve dx̄/dE, interpolasyonun **o anki**
  üssüyle (sürdürme ayarlarını çağrı anında okur). Diğer interpolasyonlar: açık hata.
* `Study`: fiziğin enerji interpolasyonu varsa, `density_map` boşsa ve interpolasyon SIMP / RAMP ise ters
  modülü kurar. Başka interpolasyonda eski kural kalır (belgelenir).
* x̄ = 0'da (p > 1) dE/dx̄ = 0, dγ/dE sonsuz olur. Orada dγ/dE = 0 alınır: γ'nin x̄ = 0'daki katkısı
  (β γ(1−γ), c = 0,1'de ≈ 7e-4) gradyandan düşer. Bu bilinen bir yaklaşıklıktır, ölçütlerin dışında tutulmaz (E4e).

| # | Kontrol | Eşik |
|:---|:---|:---|
| E4a | SIMP p ∈ {1, 2, 3}, RAMP q = 8, frac ∈ {1; 0,5}: x̄ ∈ [1e-3, 1] için ters modül = x̄ | ≤ 1e-12 mutlak |
| E4b | Study içinden fizik: γ = γ(x̄) (E4'ün düzeltmesi), p ∈ {1, 2, 3} | ≤ 1e-12 mutlak (önce 0,385) |
| E4c | Study gradyanı (uç uyumu, SIMP p = 2, enerji interpolasyonu, svk 2-B) merkezi farkla, x̄ ∈ [0,3; 0,9] | V6 ölçütü: ≤ 1e-5 göreli |
| E4d | Pasif boşluk (x̄ = 0) içeren tasarımda gradyan sonlu | NaN / sonsuz yok |
| E4e | Fizik tek başına (Study'siz) | eski kural, değerler bit düzeyinde aynı |
| E4f | Doğrusal örnekler | `pytest -m baseline` bit düzeyinde |

Var olan nonlineer Study testleri (V6, model dönemi, çözücü seçenekleri) artık γ'yı gerçek yoğunlukta
değerlendirir; sayıları değişir, ölçütleri değişmez. Bir ölçüt düşerse strict xfail olarak ölçülen değerle kaydedilir.

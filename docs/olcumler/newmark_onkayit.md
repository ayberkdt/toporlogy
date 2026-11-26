# Yapısal dinamik: Newmark ortalama ivme (β = 1/4, γ = 1/2): ön kayıt (ND1–ND6)

Ön kayıt 2026-11-26 05:11 +0300, kod yazılmadan ve hiçbir koşudan önce (05 §3: structural dynamics, time-step convergence, discrete
adjoint).

**Model.** M ü + C u̇ + K u = f(t), tutarlı kütle M = Σ_e ρ_e (∫ N Nᵀ dV ⊗ I), K = Σ_e E_e K_e⁰, Rayleigh sönümü
C = a M + b K; mesnetli DOF'lar sıfır. Newmark yer değiştirme biçimi; her adım `GridSPD` çözümü (K_eff iki eleman
matrisinin birleşimi), türev torch zincir kuralıyla (ayrık adjoint). Başlangıç ivmesi M a₀ = f₀ − C v₀ − K u₀.
Yeni modül `fem/dynamics_newmark.py`.

| # | Kontrol | Eşik |
|:---|:---|:---|
| ND1 | 2-B konsol, u₀ = φ₁ (ilk mod, M-normalize), v₀ = 0, sönümsüz, yük yok: u(t) ≈ φ₁ cos(ω₁ t); t = 2 periyotta hata Δt = T₁/20 … T₁/160 için gözlenen mertebe | ∈ [1,9; 2,1] |
| ND2 | Sönümsüz, yüksüz rastgele başlangıç: toplam enerji ½vᵀMv + ½uᵀKu sabit (ortalama ivme yöntemi) | ≤ 1e-12 göreli, 200 adım |
| ND3 | Modal sönüm: u₀ = φ₁, Rayleigh a, b ile ζ₁ = a/(2ω₁) + b ω₁/2 = 0,02; 10 periyotta genlik oranı exp(−ζ₁ ω₁ t) ile (Δt = T₁/200) | ≤ 1e-3 göreli |
| ND4 | J = Σ_n u_tip(t_n)² (darbe yükü), E ve ρ tasarıma bağlı: yönlü türev merkezi farkla | ≤ 1e-6 göreli |
| ND5 | Faktörizasyon: sabit Δt ve katsayılarla N adımda K_eff tek faktörize | sayaç = 1 (+ kütle çözümü 1) |
| ND6 | Δt ≤ 0, yanlış şekilli u₀/v₀ | açık `ValueError` |

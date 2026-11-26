# Geçici ısı iletimi (θ-yöntemi): ön kayıt (TH1–TH6)

Ön kayıt 2026-11-26 05:08 +0300, kod yazılmadan ve hiçbir koşudan önce (05 §3: "önce bir transient physics doğrulanmalı"; zaman adımı
yakınsaması, ayrık adjoint).

**Model.** C(x) Ṫ + K(x) T = q, C = Σ_e c_e C_e⁰ (tutarlı ısı kapasitesi ∫ N Nᵀ dV, 2^dim Gauss), K iletkenlik;
θ-yöntemi (C/Δt + θK) T_{n+1} = (C/Δt − (1−θ)K) T_n + q; θ = 1 geri Euler, θ = 1/2 Crank–Nicolson. Dirichlet
düğümlerinde T verilir. Türev: her adım `GridSPD` tipi autograd çözümü (katsayılara ve sağ tarafa göre) — torch'un
zincir kuralıyla ayrık adjoint. Yeni modül `fem/transient_heat.py`; `fem/grid_spd.py` birden çok eleman
matrisini kabul edecek biçimde genişler (davranışı korunur).

| # | Kontrol | Eşik |
|:---|:---|:---|
| TH1 | 2-B şerit, uçlarda T = 0, T₀ = sin(πx/L): zaman hatası (aynı ağda Δt/64 ile koşulmuş referansa göre, t = 0,1 L²ρc/k) Δt = 1/10 … 1/80 adımlarında gözlenen mertebe | geri Euler ∈ [0,9; 1,1]; Crank–Nicolson ∈ [1,9; 2,1] |
| TH2 | Uzun süre sonra (sabit kaynak, Dirichlet): kararlı hal çözümü | ≤ 1e-10 göreli |
| TH3 | Yalıtılmış (Dirichlet yok), kaynak yok: toplam ısı 1ᵀ C T sabit (her iki θ) | ≤ 1e-12 göreli |
| TH4 | J = Σ_n T_n(p) (bir düğüm), k(x) ve c(x) tasarıma bağlı: x'e göre yönlü türev merkezi farkla | ≤ 1e-6 göreli |
| TH5 | Analitik çözüme uzay+zaman yakınsaması: Crank–Nicolson, h ve Δt birlikte yarılanınca t'deki maks hata | gözlenen mertebe ∈ [1,8; 2,2] |
| TH6 | θ ∉ [0, 1], Δt ≤ 0, yanlış şekilli T₀ | açık `ValueError` |

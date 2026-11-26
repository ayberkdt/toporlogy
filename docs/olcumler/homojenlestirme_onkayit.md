# Periyodik RVE homojenleştirme (düzenli ızgara): ön kayıt (H1–H9)

Ön kayıt 2026-11-26 03:42 +0300, kod yazılmadan ve hiçbir koşudan önce (07 §1).

**Yöntem.** Tam kutu bir `Domain` periyodik hücredir. Makro şekil değiştirme ε̄ (Voigt, mühendislik kayması;
2-B 3, 3-B 6 birim durum) için u = ε̄ x + ũ, ũ periyodik (karşı yüzlerdeki düğüm çiftleri aynı DOF; usta-köle
eşlemesi P), bir düğümde öteleme sabitlenir (ayar). K_r = Pᵀ K P, sağ taraf −Pᵀ K u_aff. Etkin rijitlik enerjiden:
C_ij = (1/|Y|) u_iᵀ K u_j (u_i toplam yer değiştirme). İletkenlik aynı yolla (∇T̄). Duyarlılık: enerji biçimi
öz-eşlenik olduğundan dC_ij/dE_e = (1/|Y|) u_i^eᵀ K_e⁰ u_j^e. Yeni modül `fem/homogenization.py`; mevcut dosyalar
değişmez (`Elasticity.matrix` ve `assemble_conduction` okunur).

| # | Kontrol | Eşik |
|:---|:---|:---|
| H1 | Homojen malzeme: C_eff = birim-modül D (2-B düzlem gerilme, 3-B; Q4, H8) ve k_eff = k I | ≤ 1e-12 göreli (max) |
| H2 | 2-B iki fazlı katmanlar (x2 normalli): Backus kapalı formu C22 = ⟨1/C22⟩⁻¹, C12 = ⟨C12/C22⟩ C22, C11 = ⟨C11 − C12²/C22⟩ + ⟨C12/C22⟩² C22, C66 = ⟨1/C66⟩⁻¹; iletkenlik seri ⟨1/k⟩⁻¹ ve paralel ⟨k⟩ | ≤ 1e-10 göreli |
| H3 | Rastgele tasarım: C_eff simetrik ve pozitif tanımlı; Hill–Mandel: (1/|Y|) Σ σ:ε hacim ortalaması = ε̄ᵀ C ε̄ | simetri ≤ 1e-12, en küçük özdeğer > 0, Hill–Mandel ≤ 1e-12 göreli |
| H4 | Değişmezlik: tasarımı periyodik kaydırmak (np.roll) C'yi değiştirmez; 2-B 90° döndürme (transpoz) C11↔C22 yer değiştirir, C66 aynı | ≤ 1e-12 göreli |
| H5 | Gözenekli hücre (kare delik, E_min = 1e-9): Voigt ≥ C_eff ≥ Reuss (Loewner: farkların özdeğerleri ≥ −1e-12 · ‖C‖) | aynen |
| H6 | Sınır koşulu hiyerarşisi: KUBC (u = ε̄ x sınırda) ≥ PBC (Loewner) aynı rastgele tasarımda | farkın özdeğerleri ≥ −1e-12 · ‖C‖ |
| H7 | Duyarlılık: dC_ij/dx yönlü türevi merkezi farkla; Taylor kalanı mertebesi | ≤ 1e-6 göreli; mertebe ∈ [1,8; 2,2] |
| H8 | 3-B iki fazlı katman (x3 normalli) Backus 3-B formu (normal bileşenler harmonik ortalama, düzlem içi Voigt düzeltmeli) | ≤ 1e-10 göreli |
| H9 | Başarısızlık: tam kutu olmayan alan (void/kısmi hacim) ve tek elemanlı yön | açık `ValueError` |

# Mindlin–Reissner plaka elemanı (düzenli ızgara, Q4) — ön kayıt

Ön kayıt 2026-11-26 13:49 +0300, kod yazılmadan önce. Yeni modül `fem/plate.py`: 2-B ızgarada eğilme plakası, düğüm başına
(w, βx, βy); γ = ∇w − β, κ = [∂βx/∂x, ∂βy/∂y, ∂βx/∂y + ∂βy/∂x]. Eğilme D_b = E t³ / (12 (1 − ν²)) [[1, ν, 0], [ν, 1, 0],
[0, 0, (1 − ν)/2]] ile 2 × 2 Gauss; kesme k_s G t I (k_s = 5/6) ile **tek nokta** (seçici indirgenmiş integrasyon,
kesme kilitlenmesine karşı). Eleman matrisi birim modüllü; eleman modülü E_e ile ölçeklenir (`GridSPD`, autograd).
Düzgün basınç q: eleman başına her düğüme q h² / 4 (bilineer N için ∫N dA). Bu turda Study / YAML bağlantısı yok.

**Karşılaştırma çözümü (burada türetildi, bu oturumda kaynaktan okunmadı).** Kenarları sert basit mesnetli (w = 0 ve
teğet dönme 0) a × a kare plaka, düzgün q. Kirchhoff çözümü Navier serisi: q'nun çift sinüs açılımı tek m, n için
q_mn = 16 q / (π² m n); D∇⁴w_K = q ⇒ w_K,mn = q_mn / (D λ²), λ = π² (m² + n²) / a². Mindlin denklemlerinde
(Q = k_s G t (∇w − β), ∇·Q + q = 0, ∇·M = Q) sert basit mesnetli plaka için w = w_K − (D / (k_s G t)) ∇² w_K bağıntısı
sağlanır (Navier alanları w ~ sin sin, βx ~ cos sin, βy ~ sin cos sınır koşullarını tam karşılar); sinüs modunda
−∇² = λ olduğundan w_mn = q_mn [1 / (D λ²) + 1 / (k_s G t λ)]. Seri m, n ≤ 399 (tek) ile kesilir; kesme hatası
testte bir sonraki terim bloğuyla ölçülüp kaydedilir.

| # | Kontrol | Eşik |
|:---|:---|:---|
| PL1 | Rijit modlar (w = 1; w = x, βx = 1; w = y, βy = 1) enerjisiz: ‖KE r‖ | ≤ 1e-12 ‖KE‖ |
| PL2 | Sabit eğrilik yaması: düğümlerde w = x²/2, βx = x, βy = 0 ⇒ eleman enerjisi = D11 h² / 2 (tek nokta kesmede γ = 0) | ≤ 1e-12 göreli |
| PL3 | Navier (Mindlin), a/t = 10 ve 100, n = 8, 16, 32 eleman/kenar: merkez çökmesi hatası | n = 32'de ≤ %1; 16 → 32 gözlenen mertebe ≥ 1,5 |
| PL4 | Negatif kontrol: kesme 2 × 2 tam integrasyon, a/t = 1000, n = 16 | merkez çökmesi ≤ 0,5 × Navier (kilitlenme görünür) |
| PL5 | İnce sınır: a/t = 1000, n = 32, seçici integrasyon, Kirchhoff Navier'e göre | ≤ %2 |
| PL6 | Tasarım gradyanı: c = Fᵀw, E_e = 1e-6 + x_e³ (rastgele x), merkezi fark | ≤ 1e-6 göreli |
| PL7 | 3-B alan, t ≤ 0, ν ∉ (−1, 0,5) | açık hata |

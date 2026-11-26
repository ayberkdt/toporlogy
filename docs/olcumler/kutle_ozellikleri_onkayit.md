# Kütle özellikleri (kütle, ağırlık merkezi, atalet tensörü): ön kayıt (MP1–MP6)

Ön kayıt 2026-11-26 04:09 +0300, kod yazılmadan ve hiçbir koşudan önce (05 §3 son madde).

**Tanım.** Eleman e'nin kütlesi m_e = ρ_e · frac_e · V_e (V_e = h^dim, 2-B'de × kalınlık), ρ_e = ρ0 · ρ(x̄_e)
(yoğunluk interpolasyonu kullanıcının; varsayılan doğrusal). Eleman içinde düzgün yoğunluk: atalet tensörü elemanın
kendi (küp/dikdörtgen prizma) katkısı + paralel eksen terimi, kesin. Atalet tensörü ağırlık merkezine ya da verilen
noktaya göre: I = Σ_e [I_e^c + m_e (|r|² 1 − r rᵀ)], r = c_e − o. 2-B: düzlem dışı eksen dahil 3 × 3 (z kalınlık
boyunca simetrik). Yeni modül `core/mass_properties.py` (NumPy ve Torch; Torch'ta x̄'ye göre türevli).

| # | Kontrol | Eşik |
|:---|:---|:---|
| MP1 | Dolu dikdörtgen prizma (3-B) ve dolu levha (2-B): kütle, merkez, atalet tensörü kapalı formla | ≤ 1e-12 göreli |
| MP2 | Paralel eksen teoremi: I_o = I_c + m(|d|²1 − d dᵀ) rastgele tasarımda ve rastgele o'da | ≤ 1e-12 göreli |
| MP3 | Kütle ve merkez ötelemeyle değişmez (Domain origin kaydırma); atalet tensörü merkeze göre değişmez | ≤ 1e-12 göreli |
| MP4 | Delikli parça: dolu prizma − delik prizması (süperpozisyon) | ≤ 1e-12 göreli |
| MP5 | Kısmi hacim (frac) kütleyi ölçekler; void eleman katkı vermez | aynen |
| MP6 | Türev: kütle, merkez ve I_zz'nin x̄'ye göre yönlü türevi merkezi farkla | ≤ 1e-8 göreli |

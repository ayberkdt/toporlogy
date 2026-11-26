# Tek yönlü termoelastisite (T → u): ön kayıt (TE1–TE9)

Ön kayıt 2026-11-26 04:02 +0300, kod yazılmadan ve hiçbir koşudan önce (05 §2: ilk bağlı fizik; one-way coupling, tasarıma bağlı
termal yük, negatif kontrol).

**Model.** Küçük şekil değiştirme, ε_el = ε − α ΔT, σ = E(x) C₀ ε_el; ΔT düğüm alanı (verilen ya da ısı
iletiminden: −div(k(x) ∇T) = q, Dirichlet/Neumann). Termal yük f = ∫ Bᵀ β ΔT dV, β = C₀ α (Voigt, mühendislik
kayması: α_v = [α11, α22, α33, 2α23, 2α13, 2α12]); 2-B düzlem gerilmede β = D_ps α_p (σzz = σyz = σxz = 0
yoğuşturması termal şekil değiştirmeyi düzlem içi bileşenlerinde bırakır), düzlem şekil değiştirmede β = (C₀³ᴰ α³ᴰ)
düzlem içi bileşenleri (εzz = 0). 2×2(×2) Gauss, ΔT bilineer interpolasyonla. Yalnız Q4 ve H8 (uyumsuz modlu
elemanlar termal yükte mod terimi ister: reddedilir). Yeni modül `fem/thermoelastic.py`; çözümler kendi SPD
çözücüsüyle, E'ye ve yük vektörüne göre türevli (torch autograd); mevcut dosyalar değişmez.

| # | Kontrol | Eşik |
|:---|:---|:---|
| TE1 | Serbest genleşme (statik belirli mesnet), düzgün ΔT, izotrop α: u = α ΔT (x − x₀); eleman gerilmesi 0 | u ≤ 1e-12 göreli; σ ≤ 1e-12 · E α ΔT |
| TE2 | 2-B düzlem gerilme şerit, x yönü iki uçta tutulu, y serbest: σxx = −E α ΔT, σyy = 0; reaksiyon = E α ΔT · H · t | ≤ 1e-12 göreli |
| TE3 | 2-B düzlem şekil değiştirme, düzlem içi tamamen tutulu: σxx = σyy = −E α ΔT / (1 − 2ν) | ≤ 1e-12 göreli |
| TE4 | Döndürülmüş ortotrop α (2-B ve 3-B) serbest genleşme: ε = R diag(α) Rᵀ ΔT | ≤ 1e-12 göreli |
| TE5 | Düzgün ΔT'nin termal yükü = sınırda t = β ΔT · n çekişinin tutarlı düğüm yükü (diverjans teoremi; iç düğümler 0) | ≤ 1e-12 göreli |
| TE6 | İki katmanlı 1-B iletim (Dirichlet iki yüz): T parçalı doğrusal, seri kapalı form | ≤ 1e-12 göreli |
| TE7 | Uyum J = Fᵀu (F = F_mek + F_th(x), ΔT = T(k(x))) tasarım gradyanı, merkezi fark yönlü türev | ≤ 1e-6 göreli |
| TE8 | Negatif kontrol: termal yükün E'ye bağımlılığı koparılınca ya da T zinciri koparılınca TE7 karşılaştırması | göreli fark > 1e-2 (iki ayrı kontrol) |
| TE9 | Q4IM / H8IM ve yanlış şekilli ΔT | açık `ValueError` |

## Sonuç (2026-11-26, ilk koşu)

Komut: `python -m pytest tests/toporlogy/test_thermoelastic.py -q -p no:randomly -s` → 12 passed. TE1–TE9 PASS
(ilk koşudaki üç hata testin kendi kodundaydı: içe aktarılmış adları silen gereksiz `del` satırları; doğrulamalar o
satırlardan önce geçmişti, eşik değişmedi).

| # | Ölçülen |
|:---|:---|
| TE1 | 2-B ve 3-B serbest genleşme u = α ΔT x, gerilme ≈ 0 (≤ 1e-12) |
| TE2 | σxx = −E α ΔT, reaksiyon = E α ΔT H t |
| TE3 | düzlem şekil değiştirme σ = −E α ΔT / (1 − 2ν) |
| TE4 | 35° döndürülmüş ortotrop α ve C, 2-B ve 3-B: simetrik gradyan = R α Rᵀ ΔT |
| TE5 | düzgün ΔT yükü = β ΔT · n çekişinin tutarlı düğüm yükü |
| TE6 | iki katmanlı iletim seri kapalı form |
| TE7 | yönlü türev −3.7127340407e-02 (AD) ve −3.7127340405e-02 (merkezi fark) |
| TE8 | termal yükün E bağımlılığı koparılınca 2.16e-02, T zinciri koparılınca −2.93e-02 (doğrusu −3.71e-02) |
| TE9 | Q4IM reddi, yanlış uzunlukta ΔT |

Yapılmayanlar: Study/YAML yanıtı olarak bağlama (EvalContext ve `optimization/responses.py` bu sırada hızlandırma
oturumunda), iki yönlü bağlama, geçici ısı, sıcaklığa bağlı malzeme, bağımsız çözücü karşılaştırması.

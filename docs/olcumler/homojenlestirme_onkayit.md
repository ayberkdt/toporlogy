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

## Sonuç (2026-11-26, ilk koşu)

Komut: `python -m pytest tests/toporlogy/test_homogenization.py -q -p no:randomly -s` → 13 passed, 1 xfailed.

| # | Sonuç |
|:---|:---|
| H1 | PASS: Q4, Q4IM, H8, H8IM ve iletkenlik; homojen hücrede C_eff = E D (ör. 2-B 2,7e-15, 3-B 2,5e-16 mutlak fark, birim E) |
| H2 | PASS: 2-B iki fazlı katman, Backus blok formu ve seri/paralel iletkenlik ≤ 1e-10 |
| H3 | PASS: simetrik, pozitif tanımlı; ⟨ε⟩ = ε̄ ve ⟨σ⟩ = C ε̄ (merkez şekil değiştirmesinden, enerji C'siyle) ≤ 1e-12 |
| H4 | PASS: periyodik kaydırma ve 90° döndürme |
| H5 | PASS: kare delikli hücre Voigt ile Reuss arasında (Loewner) |
| H6 | PASS: KUBC ≥ PBC |
| H7 | Merkezi fark PASS (≤ 1e-6, elastisite ve iletkenlik). Taylor mertebesi **FAIL** (strict xfail): adımlar 0,1 … 0,0125 için 2,25, 2,10, 2,05; ilk çift bandın dışında (asimptotik değil). Post hoc 0,025 … 0,003125: 2,05, 2,02, 2,01 |
| H8 | PASS: 3-B iki fazlı katman, Backus ≤ 1e-10 |
| H9 | PASS, ikinci denemede: ilk sürüm void bölgeyi yakalamadı (void elemanlar `Domain`'de etkin sayılıyor); hücre denetimine `domain.void` eklendi, eşik değişmedi |

Kapsam dışı (sonraki dilim): SUBC (çekiş sınır koşulu), RVE boyut/ağ duyarlılığı çalışması, eleman başına D,
etkin özellik hedefli tasarım örneği (ör. negatif Poisson), lattice/dehomojenleştirme.

## Ek ön kayıt: RVE boyutu ve ağ duyarlılığı (H10–H11)

Ön kayıt 2026-11-26 04:55 +0300, bu iki kontrolün ilk koşusundan önce.

| # | Kontrol | Eşik |
|:---|:---|:---|
| H10 | Periyodik hücreyi k × k kopyalamak (k = 2, 3; rastgele tasarım, 2-B) C_eff'i değiştirmez | ≤ 1e-11 göreli |
| H11 | Kare delikli hücre (delik kenarı hücrenin yarısı, ızgaraya hizalı), n = 8, 16, 32, 64: C11 dizisi monoton, son iki farkın oranından gözlenen mertebe; Richardson değeri raporlanır | gözlenen mertebe ∈ [0,5; 2,5] (köşeli delik: tekillik nedeniyle 2'den düşük beklenir) ve son göreli değişim ≤ %2 |

### H10–H11 sonucu (ilk koşu)

H10 PASS (2 × 2 ve 3 × 3 kopya). H11 PASS: C11 / E0 = 0,57910, 0,56858, 0,56384, 0,56168 (n = 8 … 64, monoton
azalan); gözlenen mertebe 1,13 (köşeli delik), Richardson kestirimi 0,55987, son göreli değişim %0,38.

## Ek ön kayıt: post adımı `homogenize` (H12)

Ön kayıt 2026-11-26 06:59 +0300, adım yazılmadan önce. `post: [{type: homogenize}]` döndürülen tasarımın (gri x̄, çalışmanın
interpolasyonuyla E) periyodik etkin rijitliğini `summary.json`'a yazar; tam kutu olmayan alanda açık hata.

| # | Kontrol | Eşik |
|:---|:---|:---|
| H12 | Düzgün tasarımı döndüren bir koşu (hacim kısıtı 1,0, x̄ ≡ 1): C_eff = E0 · D(ν) ve doğrudan `homogenize_elasticity` ile aynı | ≤ 1e-12 göreli; aynı sayılar |

### H12 sonucu

PASS: post adımı x̄ ≡ 1'de C = E0 · D_ps(ν) (≤ 1e-12) ve doğrudan `homogenize_elasticity` ile bit düzeyinde aynı.

## Ek ön kayıt: SUBC (düzgün çekiş sınır koşulu, H13–H14)

Ön kayıt 2026-11-26 09:12 +0300, kod yazılmadan önce. `bc="subc"`: birim makro gerilme σ̄ için sınırda t = σ̄ n; sabit σ̄'nın tutarlı
düğüm yükü f = Σ_e B_eᵀ(merkez) σ̄ V_e (diverjans teoremi; doğrusal B için merkez integrali tamdır). Rijit hareket
bir düğümün ötelemeleri ve dönmeyi tutan en az DOF ile kaldırılır; ortalama şekil değiştirme ⟨ε⟩ = (1/|Y|) Σ_e B_e u_e V_e
S'nin sütunudur, C_SUBC = S⁻¹.

| # | Kontrol | Eşik |
|:---|:---|:---|
| H13 | Homojen hücre (2-B, 3-B): C_SUBC = E D | ≤ 1e-10 göreli |
| H14 | Rastgele ve gözenekli hücrede hiyerarşi C_SUBC ≤ C_PBC ≤ C_KUBC (Loewner) | farkların özdeğerleri ≥ −1e-10 · ‖C‖ |

### H13–H14 sonucu (ilk koşu)

PASS: homojen hücrede (2-B, 3-B) C_SUBC = E D ≤ 1e-10; rastgele ve gözenekli hücrede C_SUBC ≤ C_PBC ≤ C_KUBC.
Gözenekli 8 × 7 hücre (3 × 4 delik, E = 1e-3): C11 / E0 = 0,60621 (SUBC), 0,66892 (PBC), 0,67564 (KUBC).

## Ek ön kayıt: Study yanıtları (H15–H17)

Ön kayıt 2026-11-26 09:38 +0300, kod yazılmadan önce. `optimization/homogenization_responses.py`: `EffectiveStiffness(i, j)`
(C_ij) ve `EffectivePoisson` (2-B: C12/C11), x̄'nin çalışmanın interpolasyonuyla E'si üzerinden; torch autograd
fonksiyonu, geri adım dC/dE ile. Durum çözümü yok. YAML: `effective_stiffness`, `effective_poisson`.

| # | Kontrol | Eşik |
|:---|:---|:---|
| H15 | Study değeri = `homogenize_elasticity` doğrudan | ≤ 1e-14 göreli |
| H16 | Study gradyanı (filtre + SIMP + homojenleştirme + oran) merkezi farkla | ≤ 1e-6 göreli |
| H17 | YAML: amaç `effective_poisson`, kısıt hacim ve `effective_stiffness` (C11 alt sınırı), MMA birkaç iterasyon | sonlu değerler, kısıt değerlendirilir |

# Termoelastisite `Study` fiziği olarak — ön kayıt

Ön kayıt 2026-11-26 11:29 +0300, kod yazılmadan önce. `fem.thermoelastic.ThermoElastic` (TE1–TE9, `termoelastisite_onkayit.md`)
şimdiye kadar yalnız Python'dan kullanılıyordu; `Study` ve YAML bağlantısı yoktu.

Kapsam (bu tur): **verilen düzgün sıcaklık artışı** ΔT (ısı çözümü yok, sıcaklık tasarıma bağlı değil). Isıl yük
tasarıma bağlıdır: F_th(E) = Σ_e E_e G ΔT. Tasarıma bağlı sıcaklık (iletkenliği yoğunluktan) bu turda yok.

Değişiklik:
* `fem.thermoelastic.ThermoElasticPhysics(te, T)`: `solve(E, F)` → U, her yük durumu için u_c = K(E)⁻¹ (F_c + F_th(E));
  `total_load(E, F)` → F + F_th(E) (autograd E üzerinden hem K'dan hem yükten).
* `Compliance`: fizik `total_load` sunuyorsa c = Σ_c w_c F_tot,cᵀ U_c (termoelastik uyumun olağan tanımı);
  sunmuyorsa bugünkü F ᵀ U, değişmeden.
* YAML: `physics: {type: thermoelastic, alpha: …, delta_T: …, element: null}`; ν, düzlem ve D malzemeden.
  Tek mesnet takımı, yalnız x / y / z sıfır yer değiştirmesi; Q4 / H8; eleman başına D alanı: açık hata.

| # | Kontrol | Eşik |
|:---|:---|:---|
| TS1 | ΔT = 0: Study uyumu ve gradyanı = `elasticity` fiziğiyle aynı problem | ≤ 1e-12 göreli |
| TS2 | ΔT ≠ 0: Study değeri = doğrudan `ThermoElastic.displacement` ile F_totᵀu | ≤ 1e-12 göreli |
| TS3 | ΔT ≠ 0 ve mekanik yük: Study gradyanı merkezi farkla (2-B ve 3-B) | ≤ 1e-6 göreli |
| TS4 | YAML örneği kurulur ve 5 MMA iterasyonu koşar | sonlu değerler |
| TS5 | birden çok mesnet takımı, eğik mesnet, Q4IM, eleman başına D | açık hata |
| TS6 | elastik örnekler | `pytest -m baseline` bit düzeyinde; çözülmüş yapılandırma anlık görüntüsü değişmez (yeni anahtar yalnız yeni fizikte) |

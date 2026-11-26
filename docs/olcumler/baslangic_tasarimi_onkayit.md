# YAML başlangıç tasarımı (`run.start`) — ön kayıt

Ön kayıt 2026-11-26 11:22 +0300, kod yazılmadan önce. Gerekçe: düzgün başlangıç bazı amaçlarda durağan noktadır; ν12 = C12/C11 bütün
modüllerin ölçeklenmesiyle değişmez, düzgün hücrede gradyanı 5,9e-16 (`homojenlestirme_onkayit.md`, H17 post hoc).

Değişiklik: `run.start`: `null` (bugünkü düzgün başlangıç) | `{type: cosine, amplitude: a, periods: k}` |
`{type: random, amplitude: a}` (tohum `run.seed`). Pertürbasyon düzgün başlangıca eklenir, [0, 1]'e kırpılır,
sonra varsa simetri indirgemesinden geçer. cosine: p_e = a · Π_d cos(2π k i_d / n_d), i_d elemanın ızgara indisi.
random: p_e ~ U(−a, a). 0 < a ≤ 0,5; k ≥ 1 tamsayı. Birden çok faz ve açı bloğu: açık hata (bu turda yok).
`Study.perturbed_start(spec, seed)` Python'dan da çağrılır.

| # | Kontrol | Eşik |
|:---|:---|:---|
| ST1 | `run.start` verilmezse başlangıç aynı | `pytest -m baseline` bit düzeyinde; çözülmüş yapılandırma anlık görüntüsünde **yalnız** her yapılandırmaya bir `"start": null` satırı eklenir (öngörülen fark) |
| ST2 | cosine: x0 − x_düzgün = a Π cos(2π k i_d / n_d) (kırpılmayan elemanlarda) ve x0 ∈ [0, 1] | ≤ 1e-15 mutlak |
| ST3 | random: aynı tohum → aynı x0; farklı tohum → farklı; \|x0 − x_düzgün\| ≤ a | bit düzeyinde eşit / farklı |
| ST4 | YAML ters homojenleştirme: 10 × 10 hücre, `density_filter {rmin: 1.5, periodic: true}`, amaç `effective_poisson`, hacim ≤ 0,6, C11 ≥ 0,02, `start: {type: cosine, amplitude: 0.2}`, 40 MMA iterasyonu | son ν12 ≤ başlangıç ν12 − 0,1 |
| ST5 | bilinmeyen tür, a ≤ 0 ya da a > 0,5, k < 1, birden çok faz | açık hata |

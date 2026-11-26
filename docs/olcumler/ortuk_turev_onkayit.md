# Genel örtük türev (R(y, x) = 0): ön kayıt (I1–I6)

Ön kayıt 2026-11-26 04:17 +0300, kod yazılmadan ve hiçbir koşudan önce (05 §1: genel implicit differentiation, simetrik olmayan
sentetik rezidü için bağımsız analitik gradyan, transpoz hatasını yakalayan test, çoklu yanıt, fail-closed).

**Yöntem.** `optimization/implicit.py`: `ImplicitState(residual, jacobian=None)`. İleri: Newton (tam adım +
geri izleme, rezidü ve adım ölçütü ayrı), Jacobian R_y torch autograd'dan (yoğun; küçük problemler) ya da
kullanıcının seyrek Jacobian'ından. Geri: R_yᵀ λ = ∂J/∂y (transpoz çözüm, faktörizasyon yeniden kullanılır),
dJ/dx = −λᵀ ∂R/∂x (R'nin x'e göre VJP'si). Yakınsamama, tekil Jacobian ve sonlu olmayan rezidü ayrı hata türü.

| # | Kontrol | Eşik |
|:---|:---|:---|
| I1 | Simetrik olmayan doğrusal A(x) y = b(x): dJ/dx, elle türetilmiş yoğun analitik formül −λᵀ(A' y − b'), Aᵀλ = c | ≤ 1e-12 göreli |
| I2 | Negatif kontrol: geri adımda Aᵀ yerine A ile çözen sürüm (`transpose=False` test bayrağı) I1'den sapar | göreli fark > 1e-3 |
| I3 | Doğrusal olmayan bağlı rezidü (y_i + a y_i³ + Σ_j B_ij y_j − x_i = 0, B simetrik değil): merkezi fark yönlü türev; Taylor kalanı mertebesi | ≤ 1e-7 göreli; mertebe ∈ [1,8; 2,2] |
| I4 | Aynı durum için iki yanıt: tek ileri çözüm, iki geri (transpoz) çözüm, tek faktörizasyon | sayaçlar: 1 faktörizasyon |
| I5 | Newton yakınsamazsa (çözümü olmayan rezidü) ve sonlu olmayan rezidüde | ayrı açık hata türleri (`ImplicitSolveError` alt türleri), mesajda rezidü geçmişi |
| I6 | Seyrek kullanıcı Jacobian'ı ile autograd Jacobian'ı aynı gradyanı verir | ≤ 1e-12 göreli |

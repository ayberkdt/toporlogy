"""Tensor algebra of continuum mechanics: second / fourth order tensors, Voigt and Mandel forms, rotations,
invariants, kinematics and stress measures. One implementation for NumPy arrays and Torch tensors.

Conventions
-----------
* Components are Cartesian, indices 0, 1, 2 = x, y, z. Every function is batched over leading axes: a second order
  tensor is (..., d, d), a fourth order one (..., d, d, d, d), d = 3 (or 2 where stated).
* Backend: the array namespace follows the input (NumPy or Torch); no function converts between them. Torch is only
  imported when a Torch tensor is passed, and every Torch path is built from differentiable operations.
* Voigt order (the library's, :mod:`toporlogy.fem.elements`): 3-D ``[xx, yy, zz, yz, xz, xy]`` (:data:`VOIGT_3D`),
  2-D ``[xx, yy, xy]`` (:data:`VOIGT_2D`). Stresses carry the physical shear component σ_ij; strains carry the
  engineering shear γ_ij = 2 ε_ij. The two maps are separate functions (:func:`stress_to_voigt`,
  :func:`strain_to_voigt`) so that σ_vᵀ ε_v = σ : ε. Stiffness in Voigt form is C_ab = C_ijkl (no factors);
  compliance is S_ab = w_a w_b S_ijkl with w = (1, 1, 1, 2, 2, 2), so that S_voigt = C_voigt⁻¹.
* Mandel (Kelvin) form: the orthonormal basis of symmetric tensors, shear components scaled by √2
  (m = (1, 1, 1, √2, √2, √2)): M_C = diag(m) C_voigt diag(m), M_S = diag(1/m) S_voigt diag(1/m) = M_C⁻¹. The
  eigenvalues and condition numbers of an elasticity tensor are those of its Mandel matrix; the eigenvalues of the
  engineering Voigt matrix depend on the factor-of-two convention and have no physical meaning.
* Symmetry policy: a map that drops half of a tensor (:func:`stress_to_voigt`, :func:`strain_to_voigt`,
  :func:`tensor_to_mandel`) checks that the input is symmetric to ``max|A − Aᵀ| ≤ rtol · max|A|`` (default
  :data:`SYMMETRY_RTOL` = 1e-12) and raises otherwise; it never symmetrises silently. ``rtol=None`` skips the check
  for tensors that are symmetric by construction (inner loops). A first Piola–Kirchhoff stress is not symmetric and
  has no Voigt form.
* Rotations: ``R`` has the material (or new) axes as columns in global coordinates; ``rotate2(A, R) = R A Rᵀ``
  maps components from the R frame to the global frame, ``inverse=True`` the other way. An even-order tensor is the
  same under R and −R, so :func:`rotate4` accepts any orthogonal matrix; :func:`check_rotation` enforces det R = +1
  where handedness matters (frames, polar decomposition).

Kinematics (reference configuration X, current x = φ(X)): F = ∂x/∂X = I + H with H = ∇_X u; J = det F; C = FᵀF,
B = FFᵀ; Green–Lagrange E = ½(C − I) = ½(H + Hᵀ + HᵀH) (formed from H: forming FᵀF − I loses about log10(1/|H|)
digits); Euler–Almansi e = ½(I − B⁻¹); Hencky ln U = ½ ln C (material) and ln V = ½ ln B (spatial); infinitesimal
ε = sym H. J ≤ 0 is an inverted (J < 0) or degenerate (J = 0) map: :class:`InvertedDeformationError`,
:class:`SingularDeformationError`; :func:`deformation_state` also flags near-singular F (condition number above a
bound) so that callers can tell rounding trouble from inversion.

Stress measures (work conjugates; per unit reference volume unless stated): second Piola–Kirchhoff S ↔ Ė,
first Piola–Kirchhoff P = F S ↔ Ḟ (not symmetric), Kirchhoff τ = F S Fᵀ ↔ d = sym(Ḟ F⁻¹), Cauchy σ = τ / J ↔ d per
unit current volume: P:Ḟ = S:Ė = τ:d = J σ:d. A hyperelastic energy ψ(F) per reference volume gives P = ∂ψ/∂F and,
written as ψ(E), S = ∂ψ/∂E.

Invariants: I1 = tr A, I2 = ½((tr A)² − tr A²), I3 = det A; J2 = ½ s:s, J3 = det s (s = dev A); von Mises
σ_vM = √(3 J2); Lode angle θ ∈ [0, π/3] with cos 3θ = (3√3/2) J3 / J2^{3/2} (θ = 0 uniaxial tension, π/3 uniaxial
compression, π/6 pure shear); undefined (NaN) for J2 = 0. Non-smooth points: σ_vM at J2 = 0 (cone), Tresca and the
principal values where two eigenvalues coincide; :func:`principal_value_gradients` returns NaN and a flag there
instead of the gradient of an arbitrary branch.

Verification: ``tests/toporlogy/test_tensor.py`` (records T1-T9, K1-K2, S1 of ``docs/olcumler/tensor_elastisite.md``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["VOIGT_3D", "VOIGT_2D", "SYMMETRY_RTOL", "voigt_pairs", "sym", "skew", "trace", "deviator",
           "tensor_norm", "ddot", "outer", "ddot42", "ddot44", "identity4_sym", "volumetric_projector",
           "deviatoric_projector", "symmetry_errors4", "check_symmetric", "stress_to_voigt", "voigt_to_stress",
           "strain_to_voigt", "voigt_to_strain", "stiffness_to_tensor", "tensor_to_stiffness", "compliance_to_tensor",
           "tensor_to_compliance", "tensor_to_mandel", "mandel_to_tensor", "stiffness_to_mandel",
           "mandel_to_stiffness", "compliance_to_mandel", "mandel_to_compliance", "mandel_eigenvalues",
           "mandel_condition_number", "check_rotation", "rotate2", "rotate4", "rotation_from_rotvec",
           "rotvec_from_rotation", "random_rotations", "cofactor", "det_derivative", "principal_invariants", "j2",
           "j3", "lode_angle",
           "von_mises_squared", "principal_values", "principal_directions", "principal_value_gradients", "tresca",
           "DeformationError", "InvertedDeformationError", "SingularDeformationError", "deformation_gradient",
           "jacobian", "jacobian_minus_one", "deformation_state", "right_cauchy_green", "left_cauchy_green",
           "green_lagrange", "infinitesimal_strain", "euler_almansi", "polar_decomposition", "hencky_strain",
           "rate_of_deformation", "pk1_from_pk2", "pk2_from_pk1", "kirchhoff_from_pk2", "cauchy_from_pk2",
           "pk2_from_cauchy"]

#: tensor index pairs of the 3-D Voigt components (toporlogy order)
VOIGT_3D = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
#: tensor index pairs of the 2-D Voigt components
VOIGT_2D = ((0, 0), (1, 1), (0, 1))
#: relative asymmetry above which a tensor is refused (module notes)
SYMMETRY_RTOL = 1e-12

_SQ2 = math.sqrt(2.0)


def voigt_pairs(dim: int) -> tuple[tuple[int, int], ...]:
    """(i, j) of every Voigt component in ``dim`` = 2 or 3."""
    if dim == 3:
        return VOIGT_3D
    if dim == 2:
        return VOIGT_2D
    raise ValueError("dim must be 2 or 3")


# ------------------------------------------------------------------------------------------------- backend
def _is_torch(a: Any) -> bool:
    return type(a).__module__.split(".")[0] == "torch"


def _arr(a: Any) -> Any:
    """``a`` as an array of its backend (lists and scalars become float NumPy arrays)."""
    return a if (_is_torch(a) or isinstance(a, np.ndarray)) else np.asarray(a, float)


def _const(values: Any, like: Any) -> Any:
    """A constant array on the backend, dtype and device of ``like``."""
    if _is_torch(like):
        import torch
        return torch.as_tensor(np.asarray(values), dtype=like.dtype, device=like.device)
    return np.asarray(values, dtype=like.dtype if np.issubdtype(like.dtype, np.floating) else float)


def _index(values: Any, like: Any) -> Any:
    if _is_torch(like):
        import torch
        return torch.as_tensor(np.asarray(values), dtype=torch.long, device=like.device)
    return np.asarray(values, np.int64)


def _linalg(a: Any) -> Any:
    if _is_torch(a):
        import torch
        return torch.linalg
    return np.linalg


def _swap(a: Any) -> Any:
    return a.transpose(-1, -2) if _is_torch(a) else np.swapaxes(a, -1, -2)


def _einsum(spec: str, *ops: Any) -> Any:
    if _is_torch(ops[0]):
        import torch
        return torch.einsum(spec, *ops)
    return np.einsum(spec, *ops)


def _stack(xs: list[Any], axis: int) -> Any:
    if _is_torch(xs[0]):
        import torch
        return torch.stack(xs, dim=axis)
    return np.stack(xs, axis=axis)


def _eye(d: int, like: Any) -> Any:
    return _const(np.eye(d), like)


def _amax_abs(a: Any) -> float:
    return float(a.abs().max()) if _is_torch(a) else float(np.max(np.abs(a))) if np.size(a) else 0.0


def _dim2(A: Any) -> int:
    d = A.shape[-1]
    if A.ndim < 2 or A.shape[-2] != d or d not in (2, 3):
        raise ValueError(f"expected (..., d, d) with d = 2 or 3, got {tuple(A.shape)}")
    return int(d)


def _dimv(v: Any) -> int:
    n = v.shape[-1]
    if n == 6:
        return 3
    if n == 3:
        return 2
    raise ValueError(f"expected a Voigt vector (..., 6) or (..., 3), got {tuple(v.shape)}")


# index tables
def _tables(dim: int) -> tuple[list[int], list[int], np.ndarray, np.ndarray, np.ndarray]:
    """(I, J, index map (d, d) -> Voigt position, engineering weights w, Mandel scale m)."""
    pairs = voigt_pairs(dim)
    I = [p[0] for p in pairs]
    J = [p[1] for p in pairs]
    idx = np.zeros((dim, dim), np.int64)
    for a, (i, j) in enumerate(pairs):
        idx[i, j] = idx[j, i] = a
    w = np.array([1.0 if i == j else 2.0 for i, j in pairs])
    m = np.array([1.0 if i == j else _SQ2 for i, j in pairs])
    return I, J, idx, w, m


_TAB = {2: _tables(2), 3: _tables(3)}


# ------------------------------------------------------------------------------------------------- 2nd order
def sym(A: Any) -> Any:
    """½(A + Aᵀ)."""
    A = _arr(A)
    return 0.5 * (A + _swap(A))


def skew(A: Any) -> Any:
    """½(A − Aᵀ)."""
    A = _arr(A)
    return 0.5 * (A - _swap(A))


def trace(A: Any) -> Any:
    """tr A (...)."""
    return _einsum("...ii->...", _arr(A))


def deviator(A: Any) -> Any:
    """A − (tr A / d) I."""
    A = _arr(A)
    d = _dim2(A)
    return A - (trace(A) / d)[..., None, None] * _eye(d, A)


def ddot(A: Any, B: Any) -> Any:
    """A : B = A_ij B_ij (...)."""
    return _einsum("...ij,...ij->...", _arr(A), _arr(B))


def tensor_norm(A: Any) -> Any:
    """Frobenius norm √(A : A) (...)."""
    A = _arr(A)
    q = ddot(A, A)
    return q.sqrt() if _is_torch(q) else np.sqrt(q)


def outer(A: Any, B: Any) -> Any:
    """(A ⊗ B)_ijkl = A_ij B_kl."""
    return _einsum("...ij,...kl->...ijkl", _arr(A), _arr(B))


def ddot42(C: Any, A: Any) -> Any:
    """(C : A)_ij = C_ijkl A_kl."""
    return _einsum("...ijkl,...kl->...ij", _arr(C), _arr(A))


def ddot44(C: Any, D: Any) -> Any:
    """(C : D)_ijmn = C_ijkl D_klmn."""
    return _einsum("...ijkl,...klmn->...ijmn", _arr(C), _arr(D))


def identity4_sym(dim: int = 3) -> np.ndarray:
    """Symmetric fourth-order identity 𝕀ˢ_ijkl = ½(δ_ik δ_jl + δ_il δ_jk) (𝕀ˢ : A = sym A)."""
    voigt_pairs(dim)
    d = np.eye(dim)
    return 0.5 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))


def volumetric_projector(dim: int = 3) -> np.ndarray:
    """ℙ_vol = (1/d) I ⊗ I (ℙ_vol : A = (tr A / d) I)."""
    voigt_pairs(dim)
    d = np.eye(dim)
    return np.einsum("ij,kl->ijkl", d, d) / dim


def deviatoric_projector(dim: int = 3) -> np.ndarray:
    """ℙ_dev = 𝕀ˢ − ℙ_vol (ℙ_dev : A = dev sym A)."""
    return identity4_sym(dim) - volumetric_projector(dim)


def symmetry_errors4(C: Any) -> tuple[float, float]:
    """(minor, major) relative asymmetry of a fourth-order tensor: max|C_ijkl − C_jikl|, max|C_ijkl − C_ijlk| and
    max|C_ijkl − C_klij|, each over max|C|."""
    C = np.asarray(C, float)
    s = max(float(np.abs(C).max()), 1e-300)
    minor = max(float(np.abs(C - np.swapaxes(C, -4, -3)).max()), float(np.abs(C - np.swapaxes(C, -2, -1)).max()))
    major = float(np.abs(C - np.moveaxis(C, (-4, -3), (-2, -1))).max())
    return minor / s, major / s


def check_symmetric(A: Any, rtol: float = SYMMETRY_RTOL, what: str = "matrix") -> Any:
    """``A`` unchanged if ``max|A − Aᵀ| ≤ rtol · max|A|`` over its last two axes, else ValueError (no
    symmetrisation)."""
    A = _arr(A)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"{what} must be square, got {tuple(A.shape)}")
    if not np.isfinite(_amax_abs(A)):
        raise ValueError(f"{what} must be finite")
    scale = _amax_abs(A)
    err = _amax_abs(A - _swap(A))
    if err > rtol * scale:
        raise ValueError(f"{what} is not symmetric (max |A - A^T| = {err:.3e}, {err / max(scale, 1e-300):.3e} "
                         f"relative > {rtol:g}); it is not symmetrised silently")
    return A


# ------------------------------------------------------------------------------------------------- Voigt
def stress_to_voigt(sigma: Any, rtol: float | None = SYMMETRY_RTOL) -> Any:
    """Symmetric stress (..., d, d) → Voigt (..., 6 | 3) with the physical shear components (module notes)."""
    sigma = _arr(sigma)
    d = _dim2(sigma)
    if rtol is not None:
        check_symmetric(sigma, rtol, "stress tensor")
    I, J, _, _, _ = _TAB[d]
    return sigma[..., I, J]


def voigt_to_stress(v: Any) -> Any:
    """Voigt stress (..., 6 | 3) → symmetric tensor (..., d, d)."""
    v = _arr(v)
    return v[..., _index(_TAB[_dimv(v)][2], v)]


def strain_to_voigt(eps: Any, rtol: float | None = SYMMETRY_RTOL) -> Any:
    """Symmetric strain (..., d, d) → Voigt (..., 6 | 3) with engineering shear γ_ij = 2 ε_ij."""
    eps = _arr(eps)
    d = _dim2(eps)
    if rtol is not None:
        check_symmetric(eps, rtol, "strain tensor")
    I, J, _, w, _ = _TAB[d]
    return eps[..., I, J] * _const(w, eps)


def voigt_to_strain(v: Any) -> Any:
    """Voigt strain (..., 6 | 3), engineering shear → symmetric tensor (..., d, d)."""
    v = _arr(v)
    d = _dimv(v)
    idx = _TAB[d][2]
    half = np.where(np.eye(d, dtype=bool), 1.0, 0.5)
    return v[..., _index(idx, v)] * _const(half, v)


def stiffness_to_tensor(C6: Any) -> Any:
    """Voigt stiffness (..., 6, 6 | 3, 3) → C_ijkl (..., d, d, d, d) (minor and major symmetric by construction)."""
    C6 = _arr(C6)
    d = _dimv(C6[..., 0, :])
    idx = _TAB[d][2]
    return C6[..., _index(idx[:, :, None, None], C6), _index(idx[None, None, :, :], C6)]


def tensor_to_stiffness(C4: Any) -> Any:
    """C_ijkl (..., d, d, d, d) → Voigt stiffness (..., 6, 6 | 3, 3) (reads the (i ≤ j, k ≤ l) components)."""
    C4 = _arr(C4)
    d = int(C4.shape[-1])
    I, J, _, _, _ = _TAB[d]
    Ia, Ja = np.array(I), np.array(J)
    return C4[..., _index(Ia[:, None], C4), _index(Ja[:, None], C4), _index(Ia[None, :], C4),
              _index(Ja[None, :], C4)]


def compliance_to_tensor(S6: Any) -> Any:
    """Voigt compliance (engineering shear) → S_ijkl = S_ab / (w_a w_b)."""
    S6 = _arr(S6)
    d = _dimv(S6[..., 0, :])
    _, _, idx, w, _ = _TAB[d]
    wi = w[idx]
    return stiffness_to_tensor(S6) * _const(1.0 / np.einsum("ij,kl->ijkl", wi, wi), S6)


def tensor_to_compliance(S4: Any) -> Any:
    """S_ijkl → Voigt compliance S_ab = w_a w_b S_ijkl (engineering shear on both sides)."""
    S4 = _arr(S4)
    w = _TAB[int(S4.shape[-1])][3]
    return tensor_to_stiffness(S4) * _const(np.outer(w, w), S4)


# ------------------------------------------------------------------------------------------------- Mandel
def tensor_to_mandel(A: Any, rtol: float | None = SYMMETRY_RTOL) -> Any:
    """Symmetric (..., d, d) → Mandel vector (..., 6 | 3), shear components × √2 (orthonormal basis)."""
    A = _arr(A)
    d = _dim2(A)
    if rtol is not None:
        check_symmetric(A, rtol, "tensor")
    I, J, _, _, m = _TAB[d]
    return A[..., I, J] * _const(m, A)


def mandel_to_tensor(v: Any) -> Any:
    """Mandel vector (..., 6 | 3) → symmetric tensor (..., d, d)."""
    v = _arr(v)
    d = _dimv(v)
    _, _, idx, _, m = _TAB[d]
    return v[..., _index(idx, v)] * _const(1.0 / m[idx], v)


def stiffness_to_mandel(C6: Any) -> Any:
    """M_C = diag(m) C_voigt diag(m)."""
    C6 = _arr(C6)
    m = _TAB[_dimv(C6[..., 0, :])][4]
    return C6 * _const(np.outer(m, m), C6)


def mandel_to_stiffness(M: Any) -> Any:
    """C_voigt = diag(1/m) M_C diag(1/m)."""
    M = _arr(M)
    m = _TAB[_dimv(M[..., 0, :])][4]
    return M * _const(1.0 / np.outer(m, m), M)


def compliance_to_mandel(S6: Any) -> Any:
    """M_S = diag(1/m) S_voigt diag(1/m) (= M_C⁻¹)."""
    S6 = _arr(S6)
    m = _TAB[_dimv(S6[..., 0, :])][4]
    return S6 * _const(1.0 / np.outer(m, m), S6)


def mandel_to_compliance(M: Any) -> Any:
    """S_voigt = diag(m) M_S diag(m)."""
    M = _arr(M)
    m = _TAB[_dimv(M[..., 0, :])][4]
    return M * _const(np.outer(m, m), M)


def mandel_eigenvalues(A6: Any, kind: str = "stiffness") -> Any:
    """Ascending eigenvalues of the Mandel matrix of a Voigt stiffness (``kind="stiffness"``) or compliance."""
    if kind not in ("stiffness", "compliance"):
        raise ValueError("kind: 'stiffness' or 'compliance'")
    A6 = check_symmetric(_arr(A6), SYMMETRY_RTOL, f"Voigt {kind}")
    M = stiffness_to_mandel(A6) if kind == "stiffness" else compliance_to_mandel(A6)
    return _linalg(M).eigvalsh(M)


def mandel_condition_number(A6: Any, kind: str = "stiffness") -> Any:
    """λ_max / λ_min of the Mandel matrix (inf or negative if it is not positive definite)."""
    w = mandel_eigenvalues(A6, kind)
    return w[..., -1] / w[..., 0]


# ------------------------------------------------------------------------------------------------- rotations
def check_rotation(R: Any, proper: bool = True, atol: float = 1e-12) -> np.ndarray:
    """``R`` (..., 3, 3) as float array if RᵀR = I to ``atol`` (and det R = +1 for ``proper``), else ValueError."""
    Rm = np.asarray(R, float)
    if Rm.shape[-2:] != (3, 3):
        raise ValueError(f"R must be (..., 3, 3), got {Rm.shape}")
    if not np.isfinite(Rm).all():
        raise ValueError("R must be finite")
    err = float(np.abs(np.swapaxes(Rm, -1, -2) @ Rm - np.eye(3)).max()) if Rm.size else 0.0
    if err > atol:
        raise ValueError(f"R must be orthonormal (max |R^T R - I| = {err:.3e} > {atol:g})")
    if proper and Rm.size and float(np.linalg.det(Rm).min()) < 0:
        raise ValueError("R must be a proper rotation (det R = +1); got a reflection")
    return Rm


def rotate2(A: Any, R: Any, inverse: bool = False) -> Any:
    """R A Rᵀ (components from the R frame to global), or Rᵀ A R with ``inverse``. ``R`` (3, 3) for all, or
    batched like ``A``."""
    A = _arr(A)
    R = _const(R, A) if not (_is_torch(R) or isinstance(R, np.ndarray)) or _is_torch(A) != _is_torch(R) else R
    if R.ndim == 2:
        spec = "pi,...pq,qj->...ij" if inverse else "ip,...pq,jq->...ij"
    else:
        spec = "...pi,...pq,...qj->...ij" if inverse else "...ip,...pq,...jq->...ij"
    return _einsum(spec, R, A, R)


def rotate4(C: Any, R: Any, inverse: bool = False) -> Any:
    """C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs (``inverse``: R_pi R_qj R_rk R_sl C_pqrs); ``R`` (3, 3) or batched."""
    C = _arr(C)
    R = _const(R, C) if not (_is_torch(R) or isinstance(R, np.ndarray)) or _is_torch(C) != _is_torch(R) else R
    if R.ndim == 2:
        spec = "pi,qj,rk,sl,...pqrs->...ijkl" if inverse else "ip,jq,kr,ls,...pqrs->...ijkl"
        if not _is_torch(C) and C.ndim == 4:
            spec = spec.replace("...", "")                       # the 4-index loop of the original rotate_stiffness
    else:
        spec = ("...pi,...qj,...rk,...sl,...pqrs->...ijkl" if inverse
                else "...ip,...jq,...kr,...ls,...pqrs->...ijkl")
    return _einsum(spec, R, R, R, R, C)


def _hat(w: np.ndarray) -> np.ndarray:
    W = np.zeros(w.shape[:-1] + (3, 3))
    W[..., 0, 1], W[..., 0, 2], W[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    return W - np.swapaxes(W, -1, -2)


def rotation_from_rotvec(w: Any) -> np.ndarray:
    """Exponential map: R = exp(ŵ), rotation by |w| about w / |w| (Rodrigues). A chart of SO(3) without
    singularity for |w| < π; |w| = π and |w| + 2π k give the same R (the chart is not one-to-one there)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    W = _hat(w)
    small = th < 1e-4
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(small, 1.0 - th ** 2 / 6 + th ** 4 / 120, np.sin(th) / th)
        b = np.where(small, 0.5 - th ** 2 / 24 + th ** 4 / 720, (1.0 - np.cos(th)) / th ** 2)
    return np.eye(3) + a * W + b * (W @ W)


def rotvec_from_rotation(R: Any) -> np.ndarray:
    """Logarithm of a proper rotation: w with exp(ŵ) = R, |w| ∈ [0, π]. Ill-conditioned near |w| = π (there the axis
    is taken from the symmetric part; the sign of w is arbitrary at exactly π)."""
    R = check_rotation(R, proper=True, atol=1e-10)
    c = 0.5 * (np.einsum("...ii->...", R) - 1.0)
    v = 0.5 * np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.linalg.norm(v, axis=-1)                           # sin θ
    th = np.arctan2(s, c)                                    # well conditioned for small θ (arccos is not)
    out = np.empty(R.shape[:-1])
    near_pi = th > math.pi - 1e-3
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(th < 1e-8, 1.0 + th ** 2 / 6, th / s)
    out[...] = f[..., None] * v
    if np.any(near_pi):
        # sym R = cos θ I + (1 − cos θ) a aᵀ exactly, so a aᵀ = (sym R − cos θ I) / (1 − cos θ); the axis is the
        # column of largest diagonal (an earlier (R + I)/2 assumed cos θ = −1 and kept the skew part: O(π − θ) error)
        Rn, cn = R[near_pi], c[near_pi]
        B = (0.5 * (Rn + np.swapaxes(Rn, -1, -2)) - cn[..., None, None] * np.eye(3)) / (1.0 - cn)[..., None, None]
        k = np.argmax(np.einsum("...ii->...i", B), axis=-1)
        a = np.take_along_axis(B, k[..., None, None].repeat(3, axis=-1), axis=-2)[..., 0, :]
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        sgn = np.sign(np.einsum("...i,...i->...", a, v[near_pi]))
        sgn[sgn == 0] = 1.0
        out[near_pi] = (sgn * th[near_pi])[..., None] * a
    return out


def random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 3, 3) proper rotations, uniform on SO(3) (QR of a Gaussian matrix with the sign of diag(R) fixed)."""
    Q, Rr = np.linalg.qr(rng.standard_normal((n, 3, 3)))
    Q = Q * np.sign(np.einsum("nii->ni", Rr))[:, None, :]
    Q[np.linalg.det(Q) < 0, :, 0] *= -1.0
    return Q


# ------------------------------------------------------------------------------------------------- invariants
def principal_invariants(A: Any) -> tuple[Any, Any, Any]:
    """(I1, I2, I3) of (..., 3, 3)."""
    A = _arr(A)
    I1 = trace(A)
    I2 = 0.5 * (I1 * I1 - _einsum("...ij,...ji->...", A, A))
    return I1, I2, _linalg(A).det(A)


def j2(A: Any) -> Any:
    """J2 = ½ s : s of the deviator s of sym A."""
    s = deviator(sym(A))
    return 0.5 * ddot(s, s)


def j3(A: Any) -> Any:
    """J3 = det s of the deviator s of sym A."""
    s = deviator(sym(A))
    return _linalg(s).det(s)


def lode_angle(A: Any, rtol: float = 1e-12) -> np.ndarray:
    """Lode angle θ ∈ [0, π/3] (module notes); NaN where J2 ≤ (rtol · |A|)² (hydrostatic: undefined)."""
    A = np.asarray(A, float)
    J2, J3 = j2(A), j3(A)
    scale = tensor_norm(A)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.clip(1.5 * math.sqrt(3.0) * J3 / J2 ** 1.5, -1.0, 1.0)
        th = np.arccos(c) / 3.0
    return np.where((rtol * scale) ** 2 < J2, th, np.nan)


def von_mises_squared(s: Any) -> Any:
    """σ_vM² = 3 J2 of Voigt stresses (..., 6), or (..., 3) plane stress [xx, yy, xy] (σzz = 0)."""
    s = _arr(s)
    if s.shape[-1] == 3:
        sx, sy, txy = (s[..., k] for k in range(3))
        return 0.5 * ((sx - sy) ** 2 + sy ** 2 + sx ** 2) + 3.0 * txy ** 2
    if s.shape[-1] != 6:
        raise ValueError(f"expected Voigt stresses (..., 6) or (..., 3), got {tuple(s.shape)}")
    sx, sy, sz, tyz, txz, txy = (s[..., k] for k in range(6))
    return 0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2)


def principal_values(A: Any) -> Any:
    """Ascending eigenvalues of symmetric (..., d, d)."""
    A = _arr(A)
    return _linalg(A).eigvalsh(A)


def principal_directions(A: Any) -> tuple[Any, Any]:
    """(ascending eigenvalues, eigenvectors as columns) of symmetric (..., d, d)."""
    A = _arr(A)
    return _linalg(A).eigh(A)


def principal_value_gradients(A: Any, gap_rtol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(λ (..., d), ∂λ_i/∂A = n_i ⊗ n_i (..., d, d, d), repeated (..., d) bool). The derivative of a principal value
    exists only where it is simple; where |λ_i − λ_j| ≤ gap_rtol · max|λ| for some j ≠ i its gradient is NaN and
    ``repeated`` is True (the eigenvector, and the derivative of a single branch, are not defined there)."""
    A = np.asarray(A, float)
    w, V = np.linalg.eigh(A)
    G = np.einsum("...ai,...bi->...iab", V, V)
    gap = np.where(np.eye(w.shape[-1], dtype=bool), np.inf, np.abs(w[..., :, None] - w[..., None, :]))
    scale = np.max(np.abs(w), axis=-1, keepdims=True)
    rep = (gap.min(axis=-1) <= gap_rtol * np.maximum(scale, 1e-300))
    G = np.where(rep[..., None, None], np.nan, G)
    return w, G, rep


def tresca(A: Any) -> Any:
    """λ_max − λ_min of symmetric (..., 3, 3) (non-smooth where principal values coincide)."""
    w = principal_values(A)
    return w[..., -1] - w[..., 0]


# ------------------------------------------------------------------------------------------------- kinematics
class DeformationError(ValueError):
    """A deformation gradient outside the admissible set J > 0."""


class InvertedDeformationError(DeformationError):
    """det F < 0: the map inverts material (orientation reversed)."""


class SingularDeformationError(DeformationError):
    """det F = 0 (to rounding): the map collapses volume to zero."""


def deformation_gradient(H: Any) -> Any:
    """F = I + H from the displacement gradient H = ∇_X u."""
    H = _arr(H)
    return _eye(_dim2(H), H) + H


def cofactor(F: Any) -> Any:
    """Cofactor matrix cof F = (adj F)ᵀ of square 2×2 or 3×3 matrices (batched over leading axes; NumPy or torch),
    from the explicit 2×2 minors — no inverse, so it is exact for singular F (rank 2: one nonzero rank-1 cofactor;
    rank ≤ 1 in 3-D: zero). For invertible F, cof F = det(F) F⁻ᵀ
    (``docs/foundations/prereg/TK_cofactor_rotation.md``)."""
    F = _arr(F)
    if F.ndim < 2 or F.shape[-1] != F.shape[-2] or F.shape[-1] not in (2, 3):
        raise ValueError(f"cofactor needs (..., 2, 2) or (..., 3, 3) matrices, got shape {tuple(F.shape)}")
    if F.shape[-1] == 2:
        rows = [[F[..., 1, 1], -F[..., 1, 0]], [-F[..., 0, 1], F[..., 0, 0]]]
    else:
        def c(i: int, j: int) -> Any:
            r = [k for k in range(3) if k != i]
            q = [k for k in range(3) if k != j]
            minor = F[..., r[0], q[0]] * F[..., r[1], q[1]] - F[..., r[0], q[1]] * F[..., r[1], q[0]]
            return minor if (i + j) % 2 == 0 else -minor
        rows = [[c(i, j) for j in range(3)] for i in range(3)]
    return _stack([_stack(r, -1) for r in rows], -2)


def det_derivative(F: Any, H: Any) -> Any:
    """Directional derivative D(det F)[H] = cof(F) : H (exact also for singular F)."""
    return ddot(cofactor(F), _arr(H))


def _singular_tol(F: Any) -> Any:
    n = tensor_norm(F)
    return (16 * np.finfo(float).eps) * n ** int(F.shape[-1])


def jacobian(F: Any, check: bool = True) -> Any:
    """J = det F; with ``check`` raises :class:`InvertedDeformationError` (J < 0) or
    :class:`SingularDeformationError` (|J| ≤ 16 eps |F|^d)."""
    F = _arr(F)
    _dim2(F)
    J = _linalg(F).det(F)
    if check:
        Jn = np.asarray(J.detach().cpu() if _is_torch(J) else J, float)
        tol = np.asarray(_singular_tol(F.detach().cpu().numpy() if _is_torch(F) else F), float)
        if np.any(np.abs(Jn) <= tol):
            raise SingularDeformationError(f"det F = 0 to rounding (min |J| = {np.abs(Jn).min():.3e})")
        if np.any(Jn < 0):
            raise InvertedDeformationError(f"det F < 0 (min J = {Jn.min():.6g}): inverted element / material")
    return J


def jacobian_minus_one(H: Any) -> Any:
    """det(I + H) − 1 = I1(H) + I2(H) [+ I3(H)] without cancellation (..., ); H = ∇_X u (..., d, d)."""
    H = _arr(H)
    tr = trace(H)
    if _dim2(H) == 2:
        return tr + (H[..., 0, 0] * H[..., 1, 1] - H[..., 0, 1] * H[..., 1, 0])
    if H.ndim == 3 and not _is_torch(H):
        i2 = 0.5 * (tr ** 2 - np.einsum("nij,nji->n", H, H))
    else:
        i2 = 0.5 * (tr ** 2 - _einsum("...ij,...ji->...", H, H))
    return tr + i2 + _linalg(H).det(H)


def deformation_state(F: Any, cond_max: float = 1e8) -> np.ndarray:
    """Classification of every F (..., d, d): ``"inverted"`` (J < 0), ``"singular"`` (|J| ≤ 16 eps |F|^d),
    ``"near_singular"`` (J > 0 but σ_max/σ_min > ``cond_max``), ``"regular"``."""
    F = np.asarray(F, float)
    J = np.linalg.det(F)
    sv = np.linalg.svd(F, compute_uv=False)
    with np.errstate(divide="ignore"):
        cond = sv[..., 0] / sv[..., -1]
    out = np.full(J.shape, "regular", dtype=object)
    out[cond > cond_max] = "near_singular"
    out[J < 0] = "inverted"
    out[np.abs(J) <= _singular_tol(F)] = "singular"
    return out


def right_cauchy_green(F: Any) -> Any:
    """C = Fᵀ F."""
    F = _arr(F)
    return _swap(F) @ F


def left_cauchy_green(F: Any) -> Any:
    """B = F Fᵀ."""
    F = _arr(F)
    return F @ _swap(F)


def green_lagrange(H: Any) -> Any:
    """E = ½(H + Hᵀ + HᵀH) from the displacement gradient H = F − I (module notes: no FᵀF − I cancellation)."""
    H = _arr(H)
    if H.ndim == 3 and not _is_torch(H):
        return 0.5 * (H + np.swapaxes(H, 1, 2) + np.einsum("nki,nkj->nij", H, H))
    return 0.5 * (H + _swap(H) + _einsum("...ki,...kj->...ij", H, H))


def infinitesimal_strain(H: Any) -> Any:
    """ε = sym H."""
    return sym(H)


def euler_almansi(F: Any) -> Any:
    """e = ½(I − B⁻¹) = F⁻ᵀ E F⁻¹ (requires J > 0)."""
    F = _arr(F)
    jacobian(F)
    Fi = _linalg(F).inv(F)
    return 0.5 * (_eye(_dim2(F), F) - _swap(Fi) @ Fi)


def polar_decomposition(F: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, U, V) with F = R U = V R, R proper orthogonal, U, V SPD (requires J > 0; SVD based)."""
    F = np.asarray(F, float)
    jacobian(F)
    W, s, Vt = np.linalg.svd(F)
    R = W @ Vt
    U = np.swapaxes(Vt, -1, -2) @ (s[..., :, None] * Vt)
    V = W @ (s[..., :, None] * np.swapaxes(W, -1, -2))
    return R, 0.5 * (U + np.swapaxes(U, -1, -2)), 0.5 * (V + np.swapaxes(V, -1, -2))


def hencky_strain(F: Any, frame: str = "material") -> np.ndarray:
    """Logarithmic strain: ln U = ½ ln C (``frame="material"``) or ln V = ½ ln B (``"spatial"``); requires J > 0
    (C, B are then SPD and the logarithm is real and unique)."""
    F = np.asarray(F, float)
    jacobian(F)
    if frame not in ("material", "spatial"):
        raise ValueError("frame: 'material' or 'spatial'")
    A = right_cauchy_green(F) if frame == "material" else left_cauchy_green(F)
    w, V = np.linalg.eigh(A)
    return np.einsum("...ik,...k,...jk->...ij", V, 0.5 * np.log(w), V)


def rate_of_deformation(F: Any, Fdot: Any) -> Any:
    """d = sym(Ḟ F⁻¹) (spatial rate of deformation)."""
    F = _arr(F)
    return sym(_arr(Fdot) @ _linalg(F).inv(F))


# ------------------------------------------------------------------------------------------------- stress measures
def pk1_from_pk2(F: Any, S: Any) -> Any:
    """P = F S (first Piola–Kirchhoff; not symmetric, no Voigt form)."""
    return _arr(F) @ _arr(S)


def pk2_from_pk1(F: Any, P: Any) -> Any:
    """S = F⁻¹ P."""
    F = _arr(F)
    return _linalg(F).inv(F) @ _arr(P)


def kirchhoff_from_pk2(F: Any, S: Any) -> Any:
    """τ = F S Fᵀ."""
    F = _arr(F)
    return F @ _arr(S) @ _swap(F)


def cauchy_from_pk2(F: Any, S: Any) -> Any:
    """σ = F S Fᵀ / J (requires J > 0)."""
    F = _arr(F)
    J = jacobian(F)
    return kirchhoff_from_pk2(F, S) / J[..., None, None]


def pk2_from_cauchy(F: Any, sigma: Any) -> Any:
    """S = J F⁻¹ σ F⁻ᵀ (requires J > 0)."""
    F = _arr(F)
    J = jacobian(F)
    Fi = _linalg(F).inv(F)
    return J[..., None, None] * (Fi @ _arr(sigma) @ _swap(Fi))

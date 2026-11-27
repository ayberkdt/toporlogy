"""Anisotropic linear viscoelasticity at the material point: generalized Maxwell with PSD branch tensors (WP12 V05).

Physical units throughout (SI: Pa, s). Every tensor is a material input, never a design variable scaled by an element
modulus. The model sets ``physical = True`` and needs a :class:`TimeStep` (``docs/foundations/MATERIAL_CONTRACT.md``).
Derivations, tolerances and what is verified: ``docs/foundations/ANISOTROPIC_VISCOELASTICITY.md``; preregistration:
``docs/foundations/prereg/VA_anisotropic_ve.md``.

Model: σ = C∞ ε + Σ_i s_i,  ṡ_i + s_i/τ_i = C_i ε̇  (small strain, constant properties), C∞ symmetric positive
definite (positive semi-definite for a flagged fluid), every C_i symmetric positive semi-definite, τ_i > 0. The
tensors are given as 6×6 matrices in the Voigt convention of this repository (order [xx, yy, zz, yz, xz, xy], physical
shear stress, engineering shear strain) and are **checked and used in Mandel form** M = diag(m) C diag(m),
m = (1, 1, 1, √2, √2, √2), in which the Euclidean inner product of 6-vectors is the tensor contraction A:B. Positive
semi-definiteness, ranges and pseudo-inverses are statements about M, never about the Voigt matrix (which is not
orthogonally similar to M, so its eigenvalues and pseudo-inverse differ whenever a matrix is singular).

Update over a step Δt for strain linear in the step (exact for that strain): a = Δt/τ_i, r = e^{-a},
φ = (1 − e^{-a})/a:  s_i ← r s_i + φ C_i Δε. Algorithmic tangent C∞ + Σ φ_i C_i. Stored energy
ψ = ½ εᵀC∞ε + Σ ½ s_iᵀ C_i⁺ s_i with C_i⁺ the Moore–Penrose pseudo-inverse of the Mandel matrix on its range;
dissipation rate Σ s_iᵀ C_i⁺ s_i / τ_i, integrated exactly over each step (closed form, see the document).

Singular C_i: eigen-decomposition of the Mandel matrix, eigenvalues λ ≤ ``RANGE_RTOL`` λ_max are the null space (the
pseudo-inverse is zero there), the rest is inverted. A branch tensor with an eigenvalue below −``PSD_RTOL`` λ_max is
refused (never clipped). The state s_i stays in range(C_i) (it is built from C_i Δε), so ½ s_iᵀ C_i⁺ s_i is the exact
free energy of the branch; its range is the part of strain space the branch can store energy in.

Orientation: ``rotation`` R (3×3, proper) maps material axes to global axes; every tensor is rotated once,
C' = R ⊛ C (``tensor.rotate4``), so the model works in global components. A per-point field of orientations is not
implemented (one orientation per model instance).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse, \
    require_step, small_strain_history_evaluate
from toporlogy.core.linear_elastic import check_positive_definite
from toporlogy.core.viscoelastic import _g3, _phi

__all__ = ["PSD_RTOL", "RANGE_RTOL", "AnisotropicMaxwell", "isotropic_stiffness"]

PSD_RTOL = 1e-12            # λ_min ≥ −PSD_RTOL λ_max accepted as positive semi-definite (rounding of the input)
RANGE_RTOL = 1e-12          # λ ≤ RANGE_RTOL λ_max is the null space of a branch tensor (pseudo-inverse = 0 there)
_REQ = MaterialRequest()
_M = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])


def isotropic_stiffness(K: float, G: float) -> np.ndarray:
    """6×6 Voigt stiffness 3K P_vol + 2G P_dev of an isotropic material (K, G ≥ 0, not both 0; K = 0 or G = 0 gives a
    positive semi-definite branch tensor). Units Pa."""
    K, G = float(K), float(G)
    if not (np.isfinite(K) and np.isfinite(G) and K >= 0.0 and G >= 0.0):
        raise ValueError(f"isotropic_stiffness: K and G must be finite and >= 0, got K={K!r}, G={G!r}")
    one = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    ident = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    return K * np.outer(one, one) + 2.0 * G * (ident - np.outer(one, one) / 3.0)


def _mandel(C6: np.ndarray) -> np.ndarray:
    return C6 * np.outer(_M, _M)


def _spectrum(M: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of a Mandel matrix; ValueError if an eigenvalue is below −PSD_RTOL λ_max."""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    wmax = float(w[-1])
    if wmax < 0.0 or float(w[0]) < -PSD_RTOL * wmax:
        raise ValueError(f"AnisotropicMaxwell: {what} is not positive semi-definite (smallest Mandel eigenvalue "
                         f"{w[0]:.6g}, largest {wmax:.6g}); a branch with a negative eigenvalue produces negative "
                         "dissipation and is not clipped")
    return w, V


def _check_tensor(C: Any, what: str) -> np.ndarray:
    A = np.asarray(C, float)
    if A.shape != (6, 6):
        raise ValueError(f"AnisotropicMaxwell: {what} must be a (6, 6) Voigt matrix, got {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"AnisotropicMaxwell: {what} must be finite")
    A = T.check_symmetric(A, T.SYMMETRY_RTOL, what)
    return 0.5 * (A + A.T)


def _rotate(C6: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = T.tensor_to_stiffness(T.rotate4(T.stiffness_to_tensor(C6), R))
    return 0.5 * (out + out.T)


class AnisotropicMaxwell:
    """Anisotropic generalized Maxwell model ``AnisotropicMaxwell(C_inf, branches, rotation=None, allow_fluid=False)``.

    ``C_inf`` (6, 6) Voigt [Pa]: long-time stiffness, symmetric positive definite; if its Mandel matrix is singular
    (λ_min ≤ ``RANGE_RTOL`` λ_max) it is a **fluid** in the null directions and is refused unless ``allow_fluid=True``
    (then :attr:`is_fluid` is True and the quasistatic stiffness is singular; the instantaneous stiffness
    C∞ + ΣC_i must still be positive definite). ``branches``: sequence of (C_i, τ_i) with C_i (6, 6) Voigt
    symmetric positive semi-definite [Pa] and τ_i > 0 [s]; empty = elastic. ``rotation``: (3, 3) proper rotation or
    None. Invalid input raises ValueError; nothing is clipped.

    State: ``strain_prev`` (n, 3, 3), ``s`` (n, nb, 6) the branch stresses in Mandel components (shear × √2). Needs
    the time step. Isothermal (no shift). :meth:`update` returns the stress (n, 3, 3), the algorithmic tangent
    C∞ + Σ φ_i C_i (n, 6, 6, Voigt) (``tangent_kind="continuum"``: the instantaneous C∞ + Σ C_i), the trial state,
    the stored energy ψ [J/m³] and the dissipation increment of the step [J/m³] (exact integral, ≥ 0)."""

    name = "anisotropic_maxwell"
    kinematics = "small"
    conservative = False
    physical = True
    state_fields = ("strain_prev", "s")

    def __init__(self, C_inf: Any, branches: Sequence[tuple[Any, float]] = (), rotation: Any = None,
                 allow_fluid: bool = False) -> None:
        Ci = _check_tensor(C_inf, "C_inf")
        mats: list[np.ndarray] = []
        taus: list[float] = []
        for k, br in enumerate(branches):
            if len(br) != 2:
                raise ValueError(f"AnisotropicMaxwell: branch {k} must be a pair (C_i, tau_i)")
            tau = float(br[1])
            if not (np.isfinite(tau) and tau > 0.0):
                raise ValueError(f"AnisotropicMaxwell: branch {k}: tau must be finite and > 0, got {br[1]!r}")
            mats.append(_check_tensor(br[0], f"branch {k} tensor"))
            taus.append(tau)
        R = None if rotation is None else T.check_rotation(rotation, proper=True)

        w, _ = _spectrum(_mandel(Ci), "C_inf")
        self.is_fluid = bool(w[0] <= RANGE_RTOL * w[-1])
        if self.is_fluid and not allow_fluid:
            raise ValueError("AnisotropicMaxwell: C_inf is singular (a fluid in some strain directions); pass "
                             "allow_fluid=True to accept it (the quasistatic stiffness is then singular)")
        if not self.is_fluid:
            check_positive_definite(Ci, "stiffness", "C_inf")
        pinv: list[np.ndarray] = []
        for k, C in enumerate(mats):
            wk, Vk = _spectrum(_mandel(C), f"branch {k} tensor")
            keep = wk > RANGE_RTOL * max(float(wk[-1]), 0.0) if wk[-1] > 0.0 else np.zeros(6, bool)
            pinv.append((Vk[:, keep] / wk[keep]) @ Vk[:, keep].T)
        C0 = Ci + sum(mats, np.zeros((6, 6)))
        if self.is_fluid:
            check_positive_definite(C0, "stiffness", "C_inf + sum C_i (instantaneous stiffness)")

        self.rotation = R
        if R is not None:
            Ci = _rotate(Ci, R)
            mats = [_rotate(C, R) for C in mats]
            pinv = [_rotate_mandel_pinv(P, R) for P in pinv]
        self.C_inf = Ci
        self.branch_tensors = np.array(mats).reshape(len(mats), 6, 6)
        self.taus = np.array(taus, float)
        self._M_inf = _mandel(Ci)
        self._Mb = self.branch_tensors * np.outer(_M, _M)[None]
        self._Mp = np.array(pinv).reshape(len(mats), 6, 6)

    # ---------------------------------------------------------------- properties
    @property
    def n_branches(self) -> int:
        return int(self.taus.shape[0])

    def instantaneous_tangent(self) -> np.ndarray:
        """6×6 Voigt instantaneous (dt → 0) tangent C∞ + Σ C_i."""
        return self.C_inf + self.branch_tensors.sum(0)

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"strain_prev": np.zeros((n, 3, 3)), "s": np.zeros((n, self.n_branches, 6))}

    # ---------------------------------------------------------------- closed-form relaxation and DMA
    def relaxation(self, t: Any) -> np.ndarray:
        """Relaxation tensor C(t) = C∞ + Σ C_i e^{-t/τ_i} [Pa], Voigt, shape (*t.shape, 6, 6); t ≥ 0."""
        tt = np.asarray(t, float)
        if not (np.isfinite(tt).all() and (tt >= 0.0).all()):
            raise ValueError("relaxation: t must be finite and >= 0")
        w = np.exp(-tt[..., None] / self.taus)
        return self.C_inf + np.einsum("...b,bij->...ij", w, self.branch_tensors)

    def complex_modulus(self, omega: Any, form: str = "voigt") -> np.ndarray:
        """Complex modulus tensor C*(ω) = C∞ + Σ C_i iωτ_i/(1 + iωτ_i) (convention e^{iωt}, ω in rad/s), shape
        (*ω.shape, 6, 6): ``form="voigt"`` or ``"mandel"``. Its real part (storage) and imaginary part (loss) are
        symmetric positive semi-definite for this model (passivity)."""
        if form not in ("voigt", "mandel"):
            raise ValueError("form: voigt | mandel")
        om = np.asarray(omega, float)
        if not (np.isfinite(om).all() and (om >= 0.0).all()):
            raise ValueError("complex_modulus: omega must be finite and >= 0")
        x = om[..., None] * self.taus
        wgt = (x * x + 1j * x) / (1.0 + x * x)
        C = self.C_inf + np.einsum("...b,bij->...ij", wgt, self.branch_tensors)
        return C * np.outer(_M, _M) if form == "mandel" else C

    def storage_loss(self, omega: Any) -> tuple[np.ndarray, np.ndarray]:
        """(C'(ω), C''(ω)): storage and loss tensors [Pa], Voigt (real, imaginary part of :meth:`complex_modulus`)."""
        Cs = self.complex_modulus(omega)
        return Cs.real, Cs.imag

    # ---------------------------------------------------------------- update
    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray], request: MaterialRequest = _REQ
               ) -> HistoryUpdate:
        """Stress (n, 3, 3), algorithmic tangent (n, 6, 6), trial state, stored energy and dissipation increment at
        the strains ``eps`` (n, 3, 3) (symmetric) for the step ``request.step``; ``committed`` is not modified."""
        if request.tangent_kind not in ("algorithmic", "continuum"):
            raise ValueError("tangent_kind: algorithmic | continuum")
        step = require_step(self, request)
        eps = np.asarray(eps, float)
        if eps.ndim != 3 or eps.shape[1:] != (3, 3):
            raise ValueError(f"eps must be (n, 3, 3), got {eps.shape}")
        n = eps.shape[0]
        e = T.tensor_to_mandel(eps)                                      # (n, 6); raises for an asymmetric strain
        ep = T.tensor_to_mandel(np.asarray(committed["strain_prev"], float))
        de = e - ep
        s0 = np.asarray(committed["s"], float)
        a = step.dt / self.taus                                          # (nb,)
        r = np.exp(-a)
        phi = _phi(a)
        s1 = r[None, :, None] * s0 + phi[None, :, None] * np.einsum("bij,nj->nbi", self._Mb, de)
        sig_m = e @ self._M_inf.T + s1.sum(1)
        sig = T.mandel_to_tensor(sig_m)

        C = None
        if request.tangent:
            Ct = self.instantaneous_tangent() if request.tangent_kind == "continuum" else \
                self.C_inf + np.einsum("b,bij->ij", phi, self.branch_tensors)
            C = np.broadcast_to(Ct, (n, 6, 6)).copy()

        psi = None
        if request.energy:
            psi = 0.5 * np.einsum("ni,ij,nj->n", e, self._M_inf, e) \
                + 0.5 * np.einsum("nbi,bij,nbj->n", s1, self._Mp, s1)

        # exact dissipation increment, per branch: ½ q(2−q) s0ᵀC⁺s0 + a φ² s0·Δε + g3(a) ΔεᵀCΔε  (q = 1 − e^{-a})
        q = -np.expm1(-a)
        ss = np.einsum("nbi,bij,nbj->nb", s0, self._Mp, s0)
        sd = np.einsum("nbi,ni->nb", s0, de)
        dd = np.einsum("ni,bij,nj->nb", de, self._Mb, de)
        D = (0.5 * q * (2.0 - q) * ss + a * phi ** 2 * sd + _g3(a) * dd).sum(1)
        trial = {"strain_prev": eps.copy(), "s": s1}
        return HistoryUpdate(sig, C, trial, {}, psi, D)

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _REQ) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)


def _rotate_mandel_pinv(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate a Mandel-form symmetric 4th-order operator (a pseudo-inverse compliance-like matrix) by R: the Mandel
    matrix of R ⊛ A is Q M Qᵀ with the orthogonal Mandel rotation Q of R; Q is built from the images of the Mandel
    basis."""
    Q = np.empty((6, 6))
    for k in range(6):
        E = np.zeros(6)
        E[k] = 1.0
        Q[:, k] = T.tensor_to_mandel(T.rotate2(T.mandel_to_tensor(E), R))
    return Q @ P @ Q.T

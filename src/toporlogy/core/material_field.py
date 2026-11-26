"""Material fields: per-element elasticity (and, on request, mass density and conductivity) of a design
(02_MATERIAL_DESIGN §1; record docs/olcumler/malzeme_tasarimi.md).

What lives here
---------------
* :class:`PhaseMaterial`: one solid phase with its own :class:`~toporlogy.core.linear_elastic.ElasticTensor`
  (any symmetry, any frame), mass density, cost per unit volume and optional isotropic conductivity.
* :func:`in_plane_stiffness`: the element-level elasticity matrix of a phase, (6, 6) in 3-D, (3, 3) in 2-D from
  :func:`toporlogy.core.material.plane_stress` / ``plane_strain`` (Voigt [xx, yy, xy], engineering shear).
* :func:`rotate_in_plane`: the (3, 3) matrix of a material turned by θ about z, written as a fourth-order rotation
  C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs with R = [[cos θ, −sin θ], [sin θ, cos θ]] (material axis 1 at θ from x, as
  :func:`toporlogy.core.material.rotation_about_z`). Batch and differentiable (torch). A rotation about z moves
  σzz, σyz, σxz only among themselves, so turning the reduced (3, 3) matrix equals reducing the turned (6, 6) one
  (check A1 of the record). Periodicity: a turn by π about z is diag(−1, −1, 1), which flips the components with
  an odd number of z indices (Voigt C14, C15, C24, C25, C34, C35, C46, C56). The reduced (3, 3) matrix is built
  from components with four in-plane indices only, so D(θ + π) = D(θ) for EVERY material; the full 3-D tensor is
  π-periodic about z only when z is a two-fold axis of the material (monoclinic about z, e.g. orthotropic with
  axis 3 = z). Check A2 (the pre-registration claimed the 3-D statement for every material: wrong, measured).
* :class:`MaterialField`: the result of an interpolation for the ACTIVE elements only: ``stiffness`` (n_active, k, k)
  in physical units with the element's solid fraction ``frac`` applied once; ``mass_density`` and ``conductivity``
  only when requested (None otherwise). Nothing of the size of the box grid times k × k is allocated.
* :func:`symmetric_basis` / :func:`stiffness_coefficients`: C = Σ_a c_a S_a with S_ii = e_i e_iᵀ (c = C_ii) and
  S_ij = e_i e_jᵀ + e_j e_iᵀ (i < j, c = C_ij), the parametrisation through which a physics differentiates with
  respect to C (:class:`toporlogy.fem.tensor_elastic.TensorElasticity`).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from toporlogy.core.linear_elastic import ElasticTensor
from toporlogy.core.material import plane_strain, plane_stress, rotation_about_z

__all__ = ["PhaseMaterial", "MaterialField", "as_phase_materials", "in_plane_stiffness", "rotate_in_plane",
           "rotate_in_plane_derivative", "symmetric_basis", "basis_pairs", "stiffness_coefficients",
           "FIELD_REQUESTS", "PLANES"]

PLANES = ("stress", "strain")
FIELD_REQUESTS = ("stiffness", "mass_density", "conductivity")
_ELASTIC_KINDS = ("isotropic", "orthotropic", "cubic", "transversely_isotropic", "stiffness")


@dataclass(frozen=True)
class PhaseMaterial:
    """One solid phase: ``tensor`` (:class:`ElasticTensor`, force/length²), mass ``density`` (mass/length³),
    ``cost`` per unit volume of fully dense phase, isotropic ``conductivity`` (optional), and ``rotate``: whether a
    designed fibre angle turns this phase (an isotropic phase is unchanged by it either way)."""

    name: str
    tensor: ElasticTensor
    density: float = 1.0
    cost: float = 1.0
    conductivity: float | None = None
    rotate: bool = True

    def __post_init__(self) -> None:
        if self.density < 0 or self.cost < 0:
            raise ValueError(f"phase {self.name}: density and cost must be >= 0")
        if self.conductivity is not None and not self.conductivity > 0:
            raise ValueError(f"phase {self.name}: conductivity > 0 is required")

    def stiffness(self, dim: int, plane: str = "stress") -> np.ndarray:
        """Element-level elasticity of this phase: (6, 6) in 3-D, (3, 3) in 2-D (:func:`in_plane_stiffness`)."""
        return in_plane_stiffness(self.tensor.stiffness(), dim, plane)


def _elastic(spec: Mapping[str, Any], where: str) -> ElasticTensor:
    kinds = [k for k in _ELASTIC_KINDS if k in spec]
    if len(kinds) != 1:
        raise ValueError(f"{where}: give exactly one of {', '.join(_ELASTIC_KINDS)}")
    kind, v = kinds[0], spec[kinds[0]]
    angle = spec.get("angle")
    R = None if angle is None else rotation_about_z(float(angle))
    if kind == "stiffness":
        return ElasticTensor.from_stiffness(np.asarray(v, float), R)
    if not isinstance(v, Mapping):
        raise ValueError(f"{where}.{kind}: expected a mapping of constants")
    ctor = getattr(ElasticTensor, kind)
    try:
        t = ctor(**{k: float(x) for k, x in v.items()})
    except TypeError as e:
        raise ValueError(f"{where}.{kind}: {e}") from None
    return t if R is None else t.rotated(R)


def as_phase_materials(spec: Sequence[Any]) -> tuple[PhaseMaterial, ...]:
    """Phases from :class:`PhaseMaterial` objects or mappings ``{name, <kind>: {...}, angle, density, cost,
    conductivity, rotate}`` with ``<kind>`` one of isotropic {E, nu}, orthotropic {E1, E2, E3, G12, G13, G23, nu12,
    nu13, nu23}, cubic {C11, C12, C44}, transversely_isotropic {E_p, E_t, nu_p, nu_pt, G_t}, stiffness (6 × 6
    Voigt); ``angle`` (degrees about z) turns the phase's frame."""
    out = []
    allowed = {"name", "angle", "density", "cost", "conductivity", "rotate", *_ELASTIC_KINDS}
    for i, p in enumerate(spec):
        if isinstance(p, PhaseMaterial):
            out.append(p)
            continue
        if not isinstance(p, Mapping):
            raise ValueError(f"phase {i}: expected a mapping {{name, isotropic | orthotropic | ..., density, cost}}")
        unknown = set(p) - allowed
        if unknown:
            raise ValueError(f"phase {i}: unknown key(s) {sorted(unknown)}; allowed {sorted(allowed)}")
        name = str(p.get("name", f"phase{i + 1}"))
        k = p.get("conductivity")
        out.append(PhaseMaterial(name, _elastic(p, f"phase {name}"), float(p.get("density", 1.0)),
                                 float(p.get("cost", 1.0)), None if k is None else float(k),
                                 bool(p.get("rotate", True))))
    names = [p.name for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"phase names must be unique: {names}")
    if not out:
        raise ValueError("at least one phase is required")
    return tuple(out)


def in_plane_stiffness(C6: np.ndarray, dim: int, plane: str = "stress") -> np.ndarray:
    """(6, 6) unchanged in 3-D; (3, 3) of the ``plane`` convention in 2-D."""
    if dim == 3:
        return np.array(C6, float)
    if dim != 2:
        raise ValueError("dim must be 2 or 3")
    if plane not in PLANES:
        raise ValueError(f"plane: one of {PLANES}, got {plane!r}")
    return plane_stress(C6) if plane == "stress" else plane_strain(C6)


# ---------------------------------------------------------------------------------------------- 2-D rotation
_PAIRS_2D = ((0, 0), (1, 1), (0, 1))


def _voigt_index_2d() -> np.ndarray:
    v = np.empty((2, 2), np.int64)
    for a, (i, j) in enumerate(_PAIRS_2D):
        v[i, j] = v[j, i] = a
    return v


_V2 = _voigt_index_2d()


def _to_tensor4(D: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) Voigt (engineering shear) → (..., 2, 2, 2, 2) with C_ijkl = D[v(ij), v(kl)]."""
    idx = torch.as_tensor(_V2.reshape(-1), device=D.device)
    C = D.index_select(-2, idx).index_select(-1, idx)
    return C.reshape(D.shape[:-2] + (2, 2, 2, 2))


def _from_tensor4(C: torch.Tensor) -> torch.Tensor:
    i = [p[0] for p in _PAIRS_2D]
    j = [p[1] for p in _PAIRS_2D]
    flat = C.reshape(C.shape[:-4] + (4, 4))
    rows = torch.as_tensor([a * 2 + b for a, b in zip(i, j, strict=True)], device=C.device)
    return flat.index_select(-2, rows).index_select(-1, rows)


def _rot2(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rotate_in_plane(D: Any, theta: Any) -> torch.Tensor:
    """In-plane (3, 3) matrix ``D`` (Voigt [xx, yy, xy], engineering shear) of a material turned by ``theta``
    (radians, about z): (..., 3, 3) for ``theta`` of shape (...) and ``D`` (3, 3) or (..., 3, 3). Differentiable."""
    th = torch.as_tensor(theta, dtype=torch.float64)
    Dt = torch.as_tensor(D, dtype=torch.float64)
    R = _rot2(th)
    C = _to_tensor4(Dt)
    out = torch.einsum("...ip,...jq,...kr,...ls,...pqrs->...ijkl", R, R, R, R, C)
    Dr = _from_tensor4(out)
    return 0.5 * (Dr + Dr.transpose(-1, -2))


def rotate_in_plane_derivative(D: Any, theta: Any) -> torch.Tensor:
    """d/dθ of :func:`rotate_in_plane` written out (product rule, dR/dθ = [[−sin θ, −cos θ], [cos θ, −sin θ]]);
    the reference of check A3 against autograd."""
    th = torch.as_tensor(theta, dtype=torch.float64)
    Dt = torch.as_tensor(D, dtype=torch.float64)
    R = _rot2(th)
    c, s = torch.cos(th), torch.sin(th)
    dR = torch.stack([torch.stack([-s, -c], -1), torch.stack([c, -s], -1)], -2)
    C = _to_tensor4(Dt)
    e = "...ip,...jq,...kr,...ls,...pqrs->...ijkl"
    out = (torch.einsum(e, dR, R, R, R, C) + torch.einsum(e, R, dR, R, R, C) + torch.einsum(e, R, R, dR, R, C)
           + torch.einsum(e, R, R, R, dR, C))
    Dr = _from_tensor4(out)
    return 0.5 * (Dr + Dr.transpose(-1, -2))


# ---------------------------------------------------------------------------------------------- symmetric basis
def basis_pairs(k: int) -> tuple[tuple[int, int], ...]:
    """(i, j) of every basis matrix: the diagonal first, then the upper triangle row by row."""
    return tuple((i, i) for i in range(k)) + tuple((i, j) for i in range(k) for j in range(i + 1, k))


def symmetric_basis(k: int) -> np.ndarray:
    """(k(k + 1)/2, k, k) basis S_a of the symmetric k × k matrices (module notes)."""
    pairs = basis_pairs(k)
    S = np.zeros((len(pairs), k, k))
    for a, (i, j) in enumerate(pairs):
        S[a, i, j] = 1.0
        S[a, j, i] = 1.0
    return S


def stiffness_coefficients(C: torch.Tensor) -> torch.Tensor:
    """(..., k(k + 1)/2) coefficients c with C = Σ_a c_a S_a (C symmetric (..., k, k))."""
    k = C.shape[-1]
    pairs = basis_pairs(k)
    i = torch.as_tensor([p[0] for p in pairs], device=C.device)
    j = torch.as_tensor([p[1] for p in pairs], device=C.device)
    return C[..., i, j]


# ---------------------------------------------------------------------------------------------- the field
@dataclass
class MaterialField:
    """Material of the active elements of one evaluation (module notes).

    ``stiffness``: (n_active, k, k) physical elasticity incl. ``frac`` (k = 3 in 2-D, 6 in 3-D), in the order of
    ``domain.active_elements``. ``mass_density`` (n_active,): Σ_i η_i ρ_i with the unpenalised volume fractions
    (mass/length³; ``frac`` not applied: it is the density of the solid part, the element volume carries frac).
    ``conductivity`` (n_active,): interpolated with the stiffness weights. ``scale`` (n_active,): trace(C_e) /
    trace(C_ref), an informational scalar (the grid the interpolation hands to ``Study``). ``requested``: which of
    :data:`FIELD_REQUESTS` were computed; the others are None.
    """

    stiffness: torch.Tensor
    scale: torch.Tensor
    dim: int
    plane: str = "stress"
    mass_density: torch.Tensor | None = None
    conductivity: torch.Tensor | None = None
    requested: frozenset[str] = field(default_factory=lambda: frozenset({"stiffness"}))

    @property
    def n_elements(self) -> int:
        return int(self.stiffness.shape[0])

    def coefficients(self) -> torch.Tensor:
        """(n_active, k(k + 1)/2): C_e = Σ_a c_ea S_a (:func:`symmetric_basis`)."""
        return stiffness_coefficients(self.stiffness)

    def min_eigenvalue(self) -> float:
        """Smallest eigenvalue of the stiffness matrices over the elements (Voigt form as stored; positive iff
        every element is positive definite)."""
        with torch.no_grad():
            return float(torch.linalg.eigvalsh(self.stiffness).min())

    def nbytes(self) -> int:
        """Bytes held by the tensors of this field."""
        ts: Iterable[torch.Tensor | None] = (self.stiffness, self.scale, self.mass_density, self.conductivity)
        return int(sum(t.element_size() * t.numel() for t in ts if t is not None))

"""Material interpolation x̄ → element modulus (docs/mimari_plan.md §2).

What a grey element means
-------------------------
SIMP makes intermediate densities uneconomical; it does not say what material they are. If x̄ is read as the volume
fraction of a porous solid (solid + void), its moduli are bounded above by the Hashin-Shtrikman (HS) bounds, which
for void inclusions (phase 0 with K = G = 0) are, with K, G the solid's bulk and shear moduli,

    3-D:  K_HS = 4 G K x / (3 K (1 − x) + 4 G),       G_HS = G x (9K + 8G) / (15K + 20G − 6x(K + 2G))
    2-D (plane, area fraction; κ = E / (2(1 − ν)) the plane bulk modulus):
          κ_HS = κ G x / (κ (1 − x) + G),             G_HS = G κ x / (2(κ + G) − x (κ + 2G))

(written here from the general two-phase HS expressions; the bulk bounds are the exact effective moduli of the
coated-sphere / coated-cylinder assemblage and are checked in ``tests/toporlogy/test_material.py`` against the Lamé
solution of a hollow sphere and a hollow cylinder; the shear bounds are checked for their limits only).

:func:`simp_admissible_penal` is the smallest SIMP exponent for which x^p K ≤ K_HS and x^p G ≤ G_HS for all x
(slope condition at x = 1, the binding one): 2-D max(2/(1 − ν), 4/(1 + ν)), 3-D max(3(1 − ν)/(2(1 − 2ν)),
15(1 − ν)/(7 − 5ν)). Derived from the bounds above; they agree with the conditions attributed to Bendsøe & Sigmund
(1999) as remembered (text not read). For ν = 0.3: 2-D 3.08, 3-D 2.63; in 2-D, p = 3 is admissible only at ν = 1/3.

:class:`HashinShtrikman` interpolates E with the HS upper bound (best porous material; it does NOT penalise grey,
use it to judge a grey design, not to drive an optimisation). The finite element keeps the solid's ν, while the HS
porous material has ν_HS(x) ≠ ν: the E-only form is an approximation whose ν error the test reports.
"""
from __future__ import annotations

import torch

from toporlogy.references import cite
from toporlogy.core.domain import Domain

__all__ = ["SIMP", "RAMP", "HashinShtrikman", "hashin_shtrikman_moduli", "simp_admissible_penal"]


def _bulk_shear(E: float, nu: float, dim: int) -> tuple[float, float]:
    """(bulk, shear): 3-D K = E / (3(1 − 2ν)); 2-D plane-stress κ = E / (2(1 − ν)); G = E / (2(1 + ν))."""
    if not -1 < nu < 0.5:
        raise ValueError("-1 < nu < 0.5 is required")
    G = E / (2 * (1 + nu))
    return (E / (3 * (1 - 2 * nu)) if dim == 3 else E / (2 * (1 - nu))), G


def _hs_slopes(x: torch.Tensor, E: float, nu: float, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K_HS / x and G_HS / x (finite at x = 0)."""
    K, G = _bulk_shear(E, nu, dim)
    if dim == 3:
        return 4 * G * K / (3 * K * (1 - x) + 4 * G), G * (9 * K + 8 * G) / (15 * K + 20 * G - 6 * x * (K + 2 * G))
    if dim == 2:
        return K * G / (K * (1 - x) + G), G * K / (2 * (K + G) - x * (K + 2 * G))
    raise ValueError("dim must be 2 or 3")


def hashin_shtrikman_moduli(x: torch.Tensor, E: float, nu: float, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """HS upper bounds (bulk, shear) of a porous solid with volume fraction ``x`` (module notes)."""
    k, g = _hs_slopes(x, E, nu, dim)
    return x * k, x * g


@cite("bendsoe_sigmund_1999", note="admissibility derived here from the HS bounds; agrees with the remembered form")
def simp_admissible_penal(nu: float, dim: int) -> float:
    """Smallest SIMP exponent with x^p E below the HS upper bounds of a porous solid (module notes)."""
    if not -1 < nu < 0.5:
        raise ValueError("-1 < nu < 0.5 is required")
    if dim == 2:
        return max(2 / (1 - nu), 4 / (1 + nu))
    if dim == 3:
        return max(3 * (1 - nu) / (2 * (1 - 2 * nu)), 15 * (1 - nu) / (7 - 5 * nu))
    raise ValueError("dim must be 2 or 3")


@cite("sigmund_2007", note="modified SIMP E = Emin + x^p (E0 − Emin)")
class SIMP(torch.nn.Module):
    """``E_e = (Emin + x̄_e^p (E0 − Emin)) · frac_e`` with ``Emin = emin_ratio · E0``; ``penal`` may be scheduled."""

    frac: torch.Tensor

    def __init__(self, domain: Domain, E0: float = 1.0, penal: float = 3.0, emin_ratio: float = 1e-6) -> None:
        super().__init__()
        self.E0, self.penal, self.emin_ratio = float(E0), float(penal), float(emin_ratio)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        Emin = self.emin_ratio * self.E0
        return (Emin + xphys ** self.penal * (self.E0 - Emin)) * self.frac.to(xphys.dtype)


class RAMP(torch.nn.Module):
    """RAMP: ``E_e = (Emin + x̄ / (1 + q (1 − x̄)) (E0 − Emin)) · frac``.

    Stolpe & Svanberg (2001) form; source NOT read in this session (docs/mimari_plan.md §6 rule) — experimental,
    not used by any default.
    """

    frac: torch.Tensor

    def __init__(self, domain: Domain, E0: float = 1.0, q: float = 8.0, emin_ratio: float = 1e-6) -> None:
        super().__init__()
        self.E0, self.q, self.emin_ratio = float(E0), float(q), float(emin_ratio)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        Emin = self.emin_ratio * self.E0
        return (Emin + xphys / (1.0 + self.q * (1.0 - xphys)) * (self.E0 - Emin)) * self.frac.to(xphys.dtype)


@cite("hashin_shtrikman_1963", note="HS upper bounds for void inclusions, written from the general two-phase form; "
                                    "bulk bound checked against the coated-sphere / cylinder Lamé solution")
class HashinShtrikman(torch.nn.Module):
    """``E_e = (Emin + E_HS(x̄) / E0 · (E0 − Emin)) · frac``: Young's modulus of the HS upper bounds of a porous
    solid (module notes; E = 9KG/(3K + G) in 3-D, 4κG/(κ + G) in 2-D plane stress). Not penalising."""

    frac: torch.Tensor

    def __init__(self, domain: Domain, E0: float = 1.0, nu: float = 0.3, emin_ratio: float = 1e-6) -> None:
        super().__init__()
        self.E0, self.nu, self.emin_ratio, self.dim = float(E0), float(nu), float(emin_ratio), domain.dim
        _bulk_shear(1.0, self.nu, self.dim)                                   # validates nu
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def young(self, x: torch.Tensor) -> torch.Tensor:
        """E_HS(x) / E0 (1 at x = 1, 0 at x = 0); written with K/x, G/x so that it is finite at x = 0."""
        k, g = _hs_slopes(x, 1.0, self.nu, self.dim)
        return x * (9 * k * g / (3 * k + g) if self.dim == 3 else 4 * k * g / (k + g))

    def poisson(self, x: torch.Tensor) -> torch.Tensor:
        """ν of the HS bounds at x (differs from the solid's ν; the element matrix keeps the solid's)."""
        k, g = _hs_slopes(x, 1.0, self.nu, self.dim)
        return (3 * k - 2 * g) / (2 * (3 * k + g)) if self.dim == 3 else (k - g) / (k + g)

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        Emin = self.emin_ratio * self.E0
        x = xphys.clamp(0.0, 1.0)
        return (Emin + self.young(x) * (self.E0 - Emin)) * self.frac.to(xphys.dtype)


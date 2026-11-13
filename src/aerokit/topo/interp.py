"""Material interpolation x̄ → element modulus (docs/mimari_plan.md §3)."""
from __future__ import annotations

import torch

from aerokit.references import cite
from aerokit.topo.domain import Domain

__all__ = ["SIMP", "RAMP"]


@cite("sigmund_2007", note="modified SIMP E = Emin + x^p (E0 − Emin)")
class SIMP(torch.nn.Module):
    """``E_e = (Emin + x̄_e^p (E0 − Emin)) · frac_e`` with ``Emin = emin_ratio · E0``; ``penal`` may be scheduled."""

    def __init__(self, domain: Domain, E0: float = 1.0, penal: float = 3.0, emin_ratio: float = 1e-6) -> None:
        super().__init__()
        self.E0, self.penal, self.emin_ratio = float(E0), float(penal), float(emin_ratio)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        Emin = self.emin_ratio * self.E0
        return (Emin + xphys ** self.penal * (self.E0 - Emin)) * self.frac.to(xphys.dtype)


class RAMP(torch.nn.Module):
    """RAMP: ``E_e = (Emin + x̄ / (1 + q (1 − x̄)) (E0 − Emin)) · frac``.

    Stolpe & Svanberg (2001) form; source NOT read in this session (docs/mimari_plan.md §7 rule) — experimental,
    not used by any default.
    """

    def __init__(self, domain: Domain, E0: float = 1.0, q: float = 8.0, emin_ratio: float = 1e-6) -> None:
        super().__init__()
        self.E0, self.q, self.emin_ratio = float(E0), float(q), float(emin_ratio)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        Emin = self.emin_ratio * self.E0
        return (Emin + xphys / (1.0 + self.q * (1.0 - xphys)) * (self.E0 - Emin)) * self.frac.to(xphys.dtype)

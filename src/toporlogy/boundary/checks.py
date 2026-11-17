"""Check report and the rigid-body / table helpers of the boundary-condition summary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Issue:
    level: str                  # "error" | "warning" | "info"
    code: str
    message: str
    case: str | None = None

    def __str__(self) -> str:
        where = f" [{self.case}]" if self.case else ""
        return f"{self.level.upper():7s} {self.code}{where}: {self.message}"


@dataclass
class CheckReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self, level: str | None = None) -> list[str]:
        return [i.code for i in self.issues if level is None or i.level == level]

    def raise_on_error(self) -> None:
        if self.errors:
            raise ValueError("boundary-condition check failed:\n" + "\n".join(str(i) for i in self.errors))

    def __str__(self) -> str:
        head = f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([head] + [str(i) for i in self.issues])


def _rb_basis(X: np.ndarray, c: np.ndarray, L: float) -> np.ndarray:
    """(n, dim, n_rb) rigid-body displacement of nodes X for θ = (translation, rotation·L) about c."""
    r = (X - c) / L
    n, dim = X.shape
    if dim == 3:
        Phi = np.zeros((n, 3, 6))
        Phi[:, :, :3] = np.eye(3)
        Phi[:, 0, 4], Phi[:, 0, 5] = r[:, 2], -r[:, 1]
        Phi[:, 1, 3], Phi[:, 1, 5] = -r[:, 2], r[:, 0]
        Phi[:, 2, 3], Phi[:, 2, 4] = r[:, 1], -r[:, 0]
    else:
        Phi = np.zeros((n, 2, 3))
        Phi[:, :, :2] = np.eye(2)
        Phi[:, 0, 2], Phi[:, 1, 2] = -r[:, 1], r[:, 0]
    return Phi


def _describe_mode(theta: np.ndarray, c: np.ndarray, L: float, dim: int) -> str:
    lam = theta[:dim]
    mu = theta[dim:] / L
    fmt = lambda v: "(" + ", ".join(f"{x:.3g}" for x in np.where(np.abs(v) < 1e-9, 0.0, v)) + ")"  # noqa: E731

    def canon(v: np.ndarray) -> np.ndarray:                  # a mode has no sign: largest component positive
        v = v / np.linalg.norm(v)
        return -v if v[int(np.argmax(np.abs(v)))] < 0 else v
    if np.linalg.norm(mu) * L < 1e-6 * np.linalg.norm(theta):
        return f"translation along {fmt(canon(lam))}"
    if dim == 2:
        p = c + np.array([-lam[1], lam[0]]) / mu[0]
        return f"rotation about the point {fmt(p)}"
    m2 = float(mu @ mu)
    p = c + np.cross(mu, lam) / m2
    pitch = float(lam @ mu) / m2
    s = f"rotation about the axis {fmt(canon(mu))} through {fmt(p)}"
    return s + (f" with axial translation (pitch {pitch:.3g})" if abs(pitch) > 1e-6 * L else "")


def _fmt_vec(v: np.ndarray, d: int) -> str:
    return "(" + ", ".join(f"{x:.{d}f}" for x in np.where(np.abs(v) < 0.5 * 10.0 ** (-d), 0.0, v)) + ")"


def _decimals(vals: list[np.ndarray]) -> int:
    m = max([float(np.abs(v).max()) for v in vals if np.size(v)] + [0.0])
    return 0 if m == 0 else int(max(0, 3 - math.floor(math.log10(m))))


def _format_loads(rows: list[list[Any]], L: float) -> str:
    dF = _decimals([r[3] for r in rows] + [r[4] for r in rows])
    fmax = max(float(np.abs(np.r_[r[3], r[4]]).max()) for r in rows)
    dM = _decimals([r[5] for r in rows] + [np.array([fmax * L])])    # moments shown on the scale F*L
    body = [[r[0], r[1], r[2], _fmt_vec(r[3], dF), _fmt_vec(r[4], dF), _fmt_vec(r[5], dM), f"{r[6]:.1e}"]
            for r in rows]
    return _table(["Load", "Kind", "Nodes", "F intended [F]", "F applied [F]", "M about ref. [F*L]", "Rel. error"],
                  body, "llrrrrr")


def _table(headers: list[str], rows: list[list[Any]], align: str) -> str:
    """Plain-text table: two-space gutters, no vertical rules, numbers right-aligned."""
    cells = [[str(x) for x in r] for r in rows]
    for r in cells:
        r[0] = r[0] if len(r[0]) <= 38 else r[0][:35] + "..."
    w = [max(len(headers[k]), *(len(r[k]) for r in cells)) if cells else len(headers[k]) for k in range(len(headers))]
    just = lambda s, k: s.ljust(w[k]) if align[k] == "l" else s.rjust(w[k])  # noqa: E731
    out = ["  " + "  ".join(just(h, k) for k, h in enumerate(headers)),
           "  " + "  ".join("-" * w[k] for k in range(len(headers)))]
    out += ["  " + "  ".join(just(s, k) for k, s in enumerate(r)) for r in cells]
    return "\n".join(out)

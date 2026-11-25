"""Large-deflection elastica of a cantilever with a dead vertical tip load: independent reference values.

Inextensible Euler-Bernoulli beam of length L, clamped at s = 0, vertical tip load P (fixed direction), moment
EI dθ/ds. With θ(s) the rotation (positive in the load direction) and α = P L² / EI, equilibrium of the part beyond s
gives θ'' = −α cos θ (L = 1), θ(0) = 0, θ'(1) = 0 (no tip moment). Two independent computations:

* :func:`elastica_quadrature`: the first integral θ'² / 2 = α (sin θ_L − sin θ) gives
  1 = ∫₀^θ_L dθ / √(2α (sin θ_L − sin θ)) for the tip rotation θ_L, the horizontal tip position
  x_L = √(2 sin θ_L / α) in closed form and the vertical one y_L = ∫₀^θ_L sin θ dθ / √(2α (sin θ_L − sin θ))
  by quadrature (Gauss-Jacobi weight for the inverse square-root endpoint);
* :func:`elastica_shooting`: the initial-value problem integrated with a Runge-Kutta method and θ'(0) found by root
  finding on θ'(1) = 0.

They share no code beyond NumPy/SciPy. Returned: (u/L, v/L, θ_L) with u the shortening L − x_L and v the deflection.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

__all__ = ["elastica_quadrature", "elastica_shooting"]


def _length_and_y(theta_l: float, alpha: float) -> tuple[float, float]:
    def g(phi: float) -> float:                    # (φ / (sin θ_L − sin(θ_L − φ)))^{1/2}, smooth on [0, θ_L]
        if phi == 0.0:
            return 1.0 / math.sqrt(math.cos(theta_l))
        # sin a − sin(a − φ) = 2 cos(a − φ/2) sin(φ/2): no cancellation for small φ
        return math.sqrt(phi / (2.0 * math.cos(theta_l - 0.5 * phi) * math.sin(0.5 * phi)))

    c = 1.0 / math.sqrt(2.0 * alpha)
    opts = dict(weight="alg", wvar=(-0.5, 0.0), epsabs=1e-13, epsrel=1e-12, limit=200)
    length = c * quad(g, 0.0, theta_l, **opts)[0]
    y = c * quad(lambda p: g(p) * math.sin(theta_l - p), 0.0, theta_l, **opts)[0]
    return length, y


def elastica_quadrature(alpha: float) -> tuple[float, float, float]:
    """(u/L, v/L, θ_L) of the tip for α = P L² / EI from the first integral (closed form x_L, quadrature)."""
    if not alpha > 0:
        raise ValueError("alpha > 0 is required")
    lo = 0.1 * min(alpha, 1.0)                    # θ_L ≈ α/2 for small α, so the root lies above lo
    theta_l = brentq(lambda t: _length_and_y(t, alpha)[0] - 1.0, lo, math.pi / 2 - 1e-4, xtol=1e-15, rtol=1e-15)
    x_l = math.sqrt(2.0 * math.sin(theta_l) / alpha)
    return 1.0 - x_l, _length_and_y(theta_l, alpha)[1], theta_l


def elastica_shooting(alpha: float) -> tuple[float, float, float]:
    """(u/L, v/L, θ_L) of the tip for α = P L² / EI by shooting on θ'' = −α cos θ."""
    if not alpha > 0:
        raise ValueError("alpha > 0 is required")

    def rhs(_s: float, y: np.ndarray) -> list[float]:
        th, k, _x, _v = y
        return [k, -alpha * math.cos(th), math.cos(th), math.sin(th)]

    def run(k0: float) -> np.ndarray:
        sol = solve_ivp(rhs, (0.0, 1.0), [0.0, k0, 0.0, 0.0], method="DOP853", rtol=1e-13, atol=1e-14)
        return np.asarray(sol.y[:, -1])

    k0 = brentq(lambda k: run(k)[1], 1e-9, alpha * 1.0 + 1e-9, xtol=1e-15, rtol=1e-15)
    th, _k, x, v = run(k0)
    return 1.0 - float(x), float(v), float(th)

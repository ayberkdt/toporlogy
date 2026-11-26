"""Poisson convention of the homogenised cell (docs/foundations/prereg/E04_poisson.md, PO1-PO6)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core.linear_elastic import poisson_uniaxial


def _ortho_plane_stress() -> tuple[np.ndarray, np.ndarray]:
    E1, E2, G12, nu12 = 40.0, 10.0, 4.0, 0.3
    S = np.array([[1 / E1, -nu12 / E1, 0.0], [-nu12 / E1, 1 / E2, 0.0], [0.0, 0.0, 1 / G12]])
    return np.linalg.inv(S), S


def _rotate_plane(C: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 2-D Voigt stiffness [xx, yy, xy] (engineering shear) by ``deg`` about z (written out here)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    # stress transformation (physical shear) sigma' = Ts sigma; strain (engineering) eps' = Te eps
    Ts = np.array([[c * c, s * s, 2 * c * s], [s * s, c * c, -2 * c * s], [-c * s, c * s, c * c - s * s]])
    Te = np.array([[c * c, s * s, c * s], [s * s, c * c, -c * s], [-2 * c * s, 2 * c * s, c * c - s * s]])
    # sigma = C eps in the material frame -> in the rotated frame C' = Ts^-1 C Te
    return np.linalg.solve(Ts, C @ Te)


def test_po1_orthotropic_values() -> None:
    C, _S = _ortho_plane_stress()
    assert abs(poisson_uniaxial(C, 0, 1) - 0.3) <= 1e-12 * 0.3
    assert abs(poisson_uniaxial(C, 1, 0) - 0.075) <= 1e-12 * 0.075
    assert abs(C[0, 1] / C[0, 0] - 0.075) <= 1e-12 * 0.075                 # the legacy ratio is nu21 here
    Ct = torch.as_tensor(C)
    assert abs(float(poisson_uniaxial(Ct, 0, 1)) - 0.3) <= 1e-12 * 0.3


def test_po2_rotated_coupled_cell_and_negative_control() -> None:
    C0, _ = _ortho_plane_stress()
    C = _rotate_plane(C0, 27.0)
    assert abs(C[0, 2]) > 1e-3 * C[0, 0]                                     # normal-shear coupling present
    nus = {}
    for i, j in [(0, 1), (1, 0)]:
        e = np.zeros(3)
        e[i] = 1.0
        eps = np.linalg.solve(C, e)
        ref = -eps[j] / eps[i]
        got = float(poisson_uniaxial(torch.as_tensor(C), i, j))
        assert abs(got - ref) <= 1e-12 * abs(ref)
        nus[(i, j)] = ref
    S = np.linalg.solve(C, np.eye(3))
    E1, E2 = 1 / S[0, 0], 1 / S[1, 1]
    assert abs(nus[(0, 1)] / E1 - nus[(1, 0)] / E2) <= 1e-12 * abs(nus[(0, 1)] / E1)
    legacy = C[0, 1] / C[0, 0]                                                # PO2n
    assert abs(legacy - nus[(0, 1)]) > 1e-3 * abs(nus[(0, 1)])
    assert abs(legacy - nus[(1, 0)]) > 1e-3 * abs(nus[(1, 0)])


def _cell_study(objective: dict, shape=(10, 10)):
    from toporlogy.configuration.build import build_setup
    cfg = {"domain": {"type": "box", "shape": list(shape), "h": 1.0}, "material": {"E0": 1.0, "nu": 0.3},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max"] * len(shape)},
                                      "force": [0, -1] + [0] * (len(shape) - 2)}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}], "objective": objective,
           "constraints": [{"response": "volume", "upper": 0.6}], "optimizer": "mma"}
    return build_setup(cfg).study


def test_po3_uniaxial_gradient_matches_fd() -> None:
    st = _cell_study({"type": "effective_poisson", "convention": "uniaxial"})
    rng = np.random.default_rng(5)
    x = 0.2 + 0.6 * rng.random(st.n_var)
    v = rng.normal(size=st.n_var)
    ev = st.evaluate(x)
    g = float(ev.df0 @ v)
    errs = []
    for t in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        fd = (st.evaluate(x + t * v, False).f0 - st.evaluate(x - t * v, False).f0) / (2 * t)
        errs.append(abs(g - fd) / abs(fd))
    assert min(errs) <= 1e-6, errs


def test_po4_yaml_convention_required_and_legacy_bitwise() -> None:
    from toporlogy.configuration.registry import ConfigError                 # the YAML layer's error type
    with pytest.raises(ConfigError, match="uniaxial.*stiffness_ratio"):
        _cell_study({"type": "effective_poisson"})
    with pytest.raises(ConfigError, match="convention"):
        _cell_study({"type": "effective_poisson", "convention": "nu12"})
    from toporlogy.fem.homogenization import homogenize_elasticity
    st = _cell_study({"type": "effective_poisson", "convention": "stiffness_ratio"})
    x = 0.2 + 0.6 * np.random.default_rng(1).random(st.n_var)
    ev = st.evaluate(x)
    d = st.domain
    E = st.interp(torch.as_tensor(ev.xphys)).numpy().ravel()[d.active_elements]
    C = homogenize_elasticity(d, E, 0.3).C
    assert ev.f0 == C[0, 1] / C[0, 0]


@pytest.mark.parametrize("shape", [(6, 6), (3, 3, 3)])
def test_po5_homogeneous_cell_gives_material_nu(shape) -> None:
    st = _cell_study({"type": "effective_poisson", "convention": "uniaxial"}, shape)
    x = np.ones(st.n_var)
    ev = st.evaluate(x, False)
    assert abs(ev.f0 - 0.3) <= 1e-10


def test_po6_invalid_axes() -> None:
    C, _ = _ortho_plane_stress()
    for i, j in [(0, 0), (0, 2), (2, 0), (-1, 1)]:
        with pytest.raises(ValueError):
            poisson_uniaxial(C, i, j)
    C6 = np.eye(6)
    for i, j in [(1, 1), (0, 3), (3, 0)]:
        with pytest.raises(ValueError):
            poisson_uniaxial(C6, i, j)
    from toporlogy.optimization.homogenization_responses import EffectivePoisson
    for kw in [dict(load=0, transverse=0), dict(load=0, transverse=3)]:
        with pytest.raises(ValueError):
            EffectivePoisson("uniaxial", **kw)

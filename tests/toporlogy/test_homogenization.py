"""Periodic RVE homogenisation (docs/olcumler/homojenlestirme_onkayit.md, H1-H9, pre-registered before the code).

The laminate reference is the exact layered-medium average (Backus 1962 form, written here from the continuity
argument, not from the paper): across the layers the normal-type stresses N and along them the in-plane strains T
are continuous, so with the blocks of C,

    C_NN* = <C_NN^-1>^-1,  C_TN* = <C_TN C_NN^-1> C_NN*,
    C_TT* = <C_TT - C_TN C_NN^-1 C_NT> + <C_TN C_NN^-1> C_NN* <C_NN^-1 C_NT>.
"""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import D_iso, D_plane_stress, b_centroid_q4
from toporlogy.fem.homogenization import homogenize_conduction, homogenize_elasticity

NU = 0.3


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def _backus(Cs: list[np.ndarray], fractions: list[float], N: list[int]) -> np.ndarray:
    k = Cs[0].shape[0]
    T = [i for i in range(k) if i not in N]
    avg = lambda f: sum(w * f(C) for C, w in zip(Cs, fractions, strict=True))   # noqa: E731
    inv = np.linalg.inv
    CNN = inv(avg(lambda C: inv(C[np.ix_(N, N)])))
    A = avg(lambda C: C[np.ix_(T, N)] @ inv(C[np.ix_(N, N)]))
    CTN = A @ CNN
    CTT = avg(lambda C: C[np.ix_(T, T)] - C[np.ix_(T, N)] @ inv(C[np.ix_(N, N)]) @ C[np.ix_(N, T)]) + A @ CNN @ A.T
    out = np.zeros((k, k))
    out[np.ix_(N, N)], out[np.ix_(T, N)], out[np.ix_(N, T)], out[np.ix_(T, T)] = CNN, CTN, CTN.T, CTT
    return out


@pytest.mark.parametrize(("shape", "element"), [((6, 5), "Q4"), ((6, 5), "Q4IM"), ((3, 4, 3), "H8"),
                                                 ((3, 4, 3), "H8IM")])
def test_h1_homogeneous_material(shape: tuple[int, ...], element: str) -> None:
    d = Domain(shape, h=0.7, thickness=0.4)
    D = D_plane_stress(NU) if d.dim == 2 else D_iso(NU)
    h = homogenize_elasticity(d, np.full(d.n_active, 2.5), NU, element=element)
    assert _rel(h.C, 2.5 * D) <= 1e-12
    k = homogenize_conduction(d, np.full(d.n_active, 1.7))
    assert _rel(k.C, 1.7 * np.eye(d.dim)) <= 1e-12


def _layers(d: Domain, axis: int, values: tuple[float, float], n_first: int) -> np.ndarray:
    idx = np.indices(d.shape)[axis]
    return np.where(idx < n_first, values[0], values[1]).ravel()[d.active_elements]


def test_h2_two_phase_layers_2d() -> None:
    d = Domain((4, 8))
    E = _layers(d, 1, (1.0, 7.0), 3)                       # layers normal to x2, fractions 3/8, 5/8
    D = D_plane_stress(NU)
    ref = _backus([1.0 * D, 7.0 * D], [3 / 8, 5 / 8], N=[1, 2])
    assert _rel(homogenize_elasticity(d, E, NU).C, ref) <= 1e-10
    k = homogenize_conduction(d, E).C
    assert abs(k[0, 0] - (3 / 8 * 1 + 5 / 8 * 7)) <= 1e-10 * k[0, 0]                     # parallel
    assert abs(k[1, 1] - 1 / (3 / 8 / 1 + 5 / 8 / 7)) <= 1e-10 * k[1, 1]                 # series


def _random_E(d: Domain, seed: int = 0) -> np.ndarray:
    return 0.05 + np.random.default_rng(seed).random(d.n_active)


def test_h3_symmetry_positive_definiteness_hill_mandel() -> None:
    d = Domain((7, 6), h=0.5, thickness=0.3)
    E = _random_E(d)
    h = homogenize_elasticity(d, E, NU)
    C = h.C
    assert np.linalg.eigvalsh(C).min() > 0
    # volume averages of strain and stress from the centroid strain (exact element mean for Q4)
    Bc = b_centroid_q4() / d.h
    edof = d.edof[d.active_elements]
    vol_e = d.h ** 2 * d.thickness
    D = D_plane_stress(NU)
    for i in range(3):
        eps = (Bc @ h.fields[edof, i].T).T                    # (n_el, 3)
        sig = E[:, None] * (eps @ D.T)
        mean_eps = eps.sum(0) * vol_e / h.volume
        mean_sig = sig.sum(0) * vol_e / h.volume
        unit = np.eye(3)[i]
        assert np.abs(mean_eps - unit).max() <= 1e-12
        assert _rel(mean_sig, C[:, i]) <= 1e-12               # <sigma> = C eps_bar: Hill-Mandel with the energy C


def test_h4_shift_and_rotation_invariance() -> None:
    d = Domain((6, 6))
    grid = (0.05 + np.random.default_rng(1).random(d.shape))
    C = homogenize_elasticity(d, grid.ravel(), NU).C
    shifted = homogenize_elasticity(d, np.roll(grid, (2, 3), axis=(0, 1)).ravel(), NU).C
    assert _rel(shifted, C) <= 1e-12
    rot = homogenize_elasticity(d, grid.T.ravel(), NU).C
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert _rel(rot, P @ C @ P.T) <= 1e-12


def test_h5_voigt_and_reuss_bounds_of_a_porous_cell() -> None:
    d = Domain((8, 8))
    grid = np.ones(d.shape)
    grid[2:6, 2:6] = 1e-9
    E = grid.ravel()
    C = homogenize_elasticity(d, E, NU).C
    D = D_plane_stress(NU)
    CV, CR = E.mean() * D, D / np.mean(1 / E)
    tol = 1e-12 * np.abs(C).max()
    assert np.linalg.eigvalsh(CV - C).min() >= -tol
    assert np.linalg.eigvalsh(C - CR).min() >= -tol


def test_h6_kinematic_bc_is_stiffer_than_periodic() -> None:
    d = Domain((7, 6))
    E = _random_E(d, 2)
    Cp = homogenize_elasticity(d, E, NU).C
    Ck = homogenize_elasticity(d, E, NU, bc="kubc").C
    assert np.linalg.eigvalsh(Ck - Cp).min() >= -1e-12 * np.abs(Cp).max()


def _h7_setup():
    d = Domain((5, 4))
    E = _random_E(d, 3)
    dE = np.random.default_rng(4).normal(size=d.n_active)
    h = homogenize_elasticity(d, E, NU, sensitivities=True)
    assert h.dC is not None
    return d, E, dE, h, np.einsum("eij,e->ij", h.dC, dE)


def test_h7_sensitivity_by_central_differences() -> None:
    d, E, dE, h, g = _h7_setup()
    t = 1e-5
    fd = (homogenize_elasticity(d, E + t * dE, NU).C - homogenize_elasticity(d, E - t * dE, NU).C) / (2 * t)
    assert _rel(g, fd) <= 1e-6
    k = homogenize_conduction(d, E, sensitivities=True)
    assert k.dC is not None
    gk = np.einsum("eij,e->ij", k.dC, dE)
    fdk = (homogenize_conduction(d, E + t * dE).C - homogenize_conduction(d, E - t * dE).C) / (2 * t)
    assert _rel(gk, fdk) <= 1e-6


def _orders(d, E, dE, h, g, steps):
    rem = [np.abs(homogenize_elasticity(d, E + s * dE, NU).C - h.C - s * g).max() for s in steps]
    return [float(np.log2(rem[i] / rem[i + 1])) for i in range(len(rem) - 1)]


@pytest.mark.xfail(strict=True, reason="pre-registered H7 Taylor order in [1.8, 2.2] FAILED on the first run: orders "
                   "2.25, 2.10, 2.05 for steps 0.1 ... 0.0125; the first pair is not yet asymptotic (the gradient "
                   "itself passes the central-difference check)")
def test_h7_taylor_remainder_order() -> None:
    d, E, dE, h, g = _h7_setup()
    orders = _orders(d, E, dE, h, g, (1e-1, 5e-2, 2.5e-2, 1.25e-2))
    print(f"\n  H7 orders {orders}")
    assert all(1.8 <= o <= 2.2 for o in orders)


def test_h7_posthoc_taylor_order_in_the_asymptotic_range() -> None:
    """Post hoc (after H7 failed): the same remainder from step 0.025 down is second order."""
    d, E, dE, h, g = _h7_setup()
    orders = _orders(d, E, dE, h, g, (2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3))
    print(f"\n  H7 post hoc orders {orders}")
    assert all(1.8 <= o <= 2.2 for o in orders)


def test_h8_two_phase_layers_3d() -> None:
    d = Domain((3, 3, 6))
    E = _layers(d, 2, (2.0, 0.5), 2)                       # layers normal to x3, fractions 1/3, 2/3
    D = D_iso(NU)
    ref = _backus([2.0 * D, 0.5 * D], [1 / 3, 2 / 3], N=[2, 3, 4])
    assert _rel(homogenize_elasticity(d, E, NU).C, ref) <= 1e-10


def test_h9_refuses_a_cell_that_is_not_the_full_box() -> None:
    void = np.zeros((4, 4), bool)
    void[1, 1] = True
    with pytest.raises(ValueError, match="full box"):
        d = Domain((4, 4), void=void)
        homogenize_elasticity(d, np.ones(d.n_active), NU)
    frac = np.ones((4, 4))
    frac[0, 0] = 0.5
    with pytest.raises(ValueError, match="full box"):
        d = Domain((4, 4), frac=frac)
        homogenize_elasticity(d, np.ones(d.n_active), NU)
    with pytest.raises(ValueError, match="at least 2"):
        d = Domain((4, 1))
        homogenize_elasticity(d, np.ones(d.n_active), NU)


def test_h10_tiling_the_periodic_cell_changes_nothing() -> None:
    grid = 0.05 + np.random.default_rng(7).random((5, 4))
    C1 = homogenize_elasticity(Domain((5, 4)), grid.ravel(), NU).C
    for k in (2, 3):
        Ck = homogenize_elasticity(Domain((5 * k, 4 * k)), np.tile(grid, (k, k)).ravel(), NU).C
        assert _rel(Ck, C1) <= 1e-11


def _hole_cell(n: int) -> np.ndarray:
    g = np.ones((n, n))
    g[n // 4:3 * n // 4, n // 4:3 * n // 4] = 1e-9
    return g.ravel()


def test_h11_mesh_sensitivity_of_a_holed_cell() -> None:
    ns = (8, 16, 32, 64)
    c = [homogenize_elasticity(Domain((n, n), h=1.0 / n), _hole_cell(n), NU).C[0, 0] for n in ns]
    d1, d2 = c[1] - c[2], c[2] - c[3]
    order = float(np.log2(d1 / d2))
    rich = c[3] - d2 / (2 ** order - 1)
    print(f"\n  H11 C11 {c}, observed order {order:.3f}, Richardson {rich:.6f}")
    assert all(np.diff(c) < 0) or all(np.diff(c) > 0)
    assert 0.5 <= order <= 2.5
    assert abs(d2) / abs(c[3]) <= 0.02


def test_h12_homogenize_post_step() -> None:
    import types

    from toporlogy.configuration.build import build_setup
    cfg = {"domain": {"type": "box", "shape": [6, 5], "h": 1.0}, "material": {"E0": 2.0, "nu": NU},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                      "force": [0, -1]}]}],
           "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma",
           "post": [{"type": "homogenize"}]}
    setup = build_setup(cfg)
    step = dict(setup.post)["homogenize"]
    d = setup.study.domain
    out = step(types.SimpleNamespace(xphys=np.ones(d.shape)), setup.study, ".")
    E = setup.study.interp(__import__("torch").ones(d.shape, dtype=__import__("torch").float64)).numpy().ravel()
    direct = homogenize_elasticity(d, E[d.active_elements], NU).C
    assert _rel(np.array(out["C"]), 2.0 * D_plane_stress(NU)) <= 1e-12
    assert np.array_equal(np.array(out["C"]), direct)


@pytest.mark.parametrize("shape", [(6, 5), (3, 4, 3)])
def test_h13_subc_homogeneous_cell(shape: tuple[int, ...]) -> None:
    d = Domain(shape, h=0.7, thickness=0.4)
    D = D_plane_stress(NU) if d.dim == 2 else D_iso(NU)
    assert _rel(homogenize_elasticity(d, np.full(d.n_active, 2.5), NU, bc="subc").C, 2.5 * D) <= 1e-10


@pytest.mark.parametrize("kind", ["random", "porous"])
def test_h14_bound_hierarchy(kind: str) -> None:
    d = Domain((8, 7))
    if kind == "random":
        E = _random_E(d, 9)
    else:
        g = np.ones(d.shape)
        g[2:6, 2:5] = 1e-3
        E = g.ravel()
    Cs, Cp, Ck = (homogenize_elasticity(d, E, NU, bc=b).C for b in ("subc", "periodic", "kubc"))
    tol = 1e-10 * np.abs(Cp).max()
    assert np.linalg.eigvalsh(Cp - Cs).min() >= -tol
    assert np.linalg.eigvalsh(Ck - Cp).min() >= -tol

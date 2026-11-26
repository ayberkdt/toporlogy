"""Independent and analytical verification of the general mesh (docs/olcumler/fem_genel_onkayit.md §3, V1-V3,
pre-registered 2026-11-26 02:41 +0300 before the first run): quadratic exact solutions, scikit-fem cross-check,
manufactured-solution convergence on distorted meshes."""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from toporlogy.core.linear_elastic import isotropic_moduli, isotropic_plane_strain, isotropic_stiffness
from toporlogy.fem.mesh import Dirichlet, ElasticProblem, Mesh, box, get, perturb, rectangle
from toporlogy.fem.mesh.manufactured import Manufactured

NU = 0.3


def _lame(E=1.0, nu=NU):
    m = isotropic_moduli(E=E, nu=nu)
    return m.G, m.lam


def _D(dim, E=1.0, nu=NU):
    return isotropic_plane_strain(E, nu) if dim == 2 else isotropic_stiffness(E, nu)


def _fmt(rows, key):
    return [format(x[key], '.2e') for x in rows]


# ---------------------------------------------------------------------------------------------- V1
@pytest.mark.parametrize("name", ["T6", "Q9", "T10"])
def test_v1_quadratic_solution_is_reproduced_exactly(name):
    dim = get(name).dim
    m = rectangle(4, 3, element=name) if dim == 2 else box(3, 2, 2, element=name)
    S = np.array([[1.0, 0.35, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]])[:dim, :dim]
    m.points = m.points @ S.T                              # global affine map: parallelograms / affine tets
    rng = np.random.default_rng(7)
    a = rng.normal(size=dim)
    B = rng.normal(size=(dim, dim))
    H = rng.normal(size=(dim, dim, dim))
    H = 0.5 * (H + np.swapaxes(H, 1, 2))                   # H[i] symmetric: u_i = a_i + B_i·x + ½ xᵀ H_i x
    mu, lam = _lame()

    def u(X):
        return a + X @ B.T + 0.5 * np.einsum("nj,ijk,nk->ni", X, H, X)
    lap = np.trace(H, axis1=1, axis2=2)
    graddiv = np.einsum("kki->i", H)
    b = -(mu * lap + (lam + mu) * graddiv)                 # −div σ, constant
    P = ElasticProblem(m, _D(dim))
    bn = m.boundary_nodes()
    val = u(m.points[bn])
    bc = Dirichlet([(m.dofs(bn, [c]), val[:, c]) for c in range(dim)])
    s = P.solve(P.body_force(b), bc)
    ue = u(m.points)
    err = np.abs(s.u[:, 0].reshape(-1, dim) - ue).max() / np.abs(ue).max()
    print(f"\n  V1 {name}: {err:.1e}")
    assert err <= 1e-10


# ---------------------------------------------------------------------------------------------- V2
skfem = pytest.importorskip("skfem")
from skfem import (Basis, BilinearForm, ElementHex1, ElementQuad1, ElementQuad2, ElementTetP1,  # noqa: E402
                   ElementTetP2, ElementTriP1, ElementTriP2, ElementVector, LinearForm, MeshHex, MeshQuad,
                   MeshTet, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace  # noqa: E402

SKFEM = {"T3": (MeshTri, ElementTriP1, 2), "T6": (MeshTri, ElementTriP2, 2), "T4": (MeshTet, ElementTetP1, 2),
         "T10": (MeshTet, ElementTetP2, 2), "Q4": (MeshQuad, ElementQuad1, 2), "Q9": (MeshQuad, ElementQuad2, 4),
         "H8": (MeshHex, ElementHex1, 2)}


def _sk_mesh(name, m, n):
    """scikit-fem mesh with the same (distorted) vertices; quads and hexes through init_tensor (its own corner
    order), vertices moved to the library mesh's positions matched on the undistorted grid."""
    Msk, _, _ = SKFEM[name]
    ref = get(name)
    if ref.cell in ("tri", "tet"):
        v, t = np.unique(m.cells[:, :ref.n_vertices], return_inverse=True)     # vertices only
        return Msk(m.points[v].T, t.reshape(-1, ref.n_vertices).T)
    dim = ref.dim
    lengths = (1.2, 0.9) if dim == 2 else (1.0, 0.8, 0.7)
    sk = Msk.init_tensor(*[np.linspace(0.0, L, k + 1) for L, k in zip(lengths, n, strict=False)])
    und = rectangle(*n[:2], lx=1.2, ly=0.9, element=name) if dim == 2 else \
        box(*n, lx=1.0, ly=0.8, lz=0.7, element=name)
    d, idx = cKDTree(und.points).query(sk.p.T)
    assert d.max() < 1e-10
    return Msk(m.points[idx].T, sk.t)


def _setup_v2(name):
    dim = get(name).dim
    n = (4, 3) if dim == 2 else (3, 2, 2)
    base = rectangle(*n, lx=1.2, ly=0.9, element=name) if dim == 2 else box(*n, lx=1.0, ly=0.8, lz=0.7, element=name)
    h = min(1.2 / 4, 0.9 / 3) if dim == 2 else 0.7 / 2
    m = perturb(base, 0.2 * h, seed=11)
    Msk, Esk, order = SKFEM[name]
    mesh = _sk_mesh(name, m, n)
    basis = Basis(mesh, ElementVector(Esk()), intorder=order)
    # library DOF of every scikit-fem DOF, matched by DOF location and component
    tree = cKDTree(m.points)
    perm = np.empty(basis.N, np.int64)
    for c, idx in enumerate(basis.split_indices()):
        dd, node = tree.query(basis.doflocs[:, idx].T)
        assert dd.max() < 1e-10
        perm[idx] = dim * node + c
    assert len(np.unique(perm)) == basis.N
    return m, basis, perm


@BilinearForm
def _elast(u, v, w):
    mu, lam = _lame()
    return 2.0 * mu * ddot(sym_grad(u), sym_grad(v)) + lam * trace(sym_grad(u)) * trace(sym_grad(v))


@pytest.mark.parametrize("name", list(SKFEM))
def test_v2_scikit_fem_stiffness_load_and_solution(name):
    m, basis, perm = _setup_v2(name)
    dim = m.dim
    bvec = np.arange(1, dim + 1) * 0.7

    @LinearForm
    def load(v, w):
        return sum(bvec[i] * v.value[i] for i in range(dim))
    Ksk = asm(_elast, basis)
    fsk = asm(load, basis)
    P = ElasticProblem(m, _D(dim))
    K = P.K[perm][:, perm]
    f = P.body_force(bvec)[perm]
    dK = spla.norm(K - Ksk) / spla.norm(Ksk)
    df = np.linalg.norm(f - fsk) / np.linalg.norm(fsk)
    left = np.flatnonzero(m.points[:, 0] < 1e-9)
    D_sk = basis.get_dofs(lambda x: x[0] < 1e-9).all()
    usk = solve(*condense(Ksk, fsk, D=D_sk))
    s = P.solve(P.body_force(bvec), Dirichlet.fixed(m.dofs(left)))
    du = np.abs(s.u[perm, 0] - usk).max() / np.abs(usk).max()
    print(f"\n  V2 {name}: K {dK:.1e}, f {df:.1e}, u {du:.1e}")
    assert dK <= 1e-12 and df <= 1e-12
    assert du <= 1e-10


# ---------------------------------------------------------------------------------------------- V3
def _mms_errors(name, n, seed, curved=0.0, amp=0.2):
    dim = get(name).dim
    base = rectangle(n, n, element=name) if dim == 2 else box(n, n, n, element=name)
    m = perturb(base, amp / n, seed=seed, curved=curved / n)
    mu, lam = _lame()
    ms = Manufactured(dim, c=1.0, lam_c=lam)
    P = ElasticProblem(m, _D(dim))
    k = get(name).degree
    f = P.body_force(lambda X: ms.body_force(mu, X), degree=2 * k + 4)
    bn = m.boundary_nodes()
    s = P.solve(f, Dirichlet.fixed(m.dofs(bn)))
    return P.errors(s.u[:, 0], ms.u, ms.grad)


def _rates(name, ns, curved=0.0, amp=0.2):
    e = [_mms_errors(name, n, seed=100 + n, curved=curved, amp=amp) for n in ns]
    pl2 = math.log(e[-2]["l2"] / e[-1]["l2"]) / math.log(ns[-1] / ns[-2])
    pen = math.log(e[-2]["energy"] / e[-1]["energy"]) / math.log(ns[-1] / ns[-2])
    return e, pl2, pen


MMS_LEVELS = {"Q4": (4, 8, 16, 32), "T3": (4, 8, 16, 32), "Q9": (4, 8, 16, 32), "T6": (4, 8, 16, 32),
              "H8": (4, 8, 16), "T4": (4, 8, 16), "T10": (2, 4, 8)}


_V3_XFAIL = {
    "T4": "pre-registered V3 T4 (n = 4, 8, 16): L2 rate 1.796 < 1.8 (energy 0.92 passes); pre-asymptotic, "
          "n = 16 -> 24 gives 1.92 (post hoc, test_v3_posthoc_one_more_level) (fem_genel.md §3)",
    "T10": "pre-registered V3 T10 (n = 2, 4, 8): energy rate 1.788 < 1.8 (L2 2.92 passes); pre-asymptotic, "
           "n = 8 -> 12 gives 1.89 (post hoc) (fem_genel.md §3)"}


def _v3_param(name):
    marks = [pytest.mark.slow] if get(name).dim == 3 else []
    if name in _V3_XFAIL:
        marks.append(pytest.mark.xfail(strict=True, reason=_V3_XFAIL[name]))
    return pytest.param(name, marks=marks)


@pytest.mark.parametrize("name", [_v3_param(n) for n in MMS_LEVELS])
def test_v3_mms_convergence_on_distorted_mesh(name):
    k = get(name).degree
    e, pl2, pen = _rates(name, MMS_LEVELS[name])
    print(f"\n  V3 {name}: L2 {_fmt(e, 'l2')} p = {pl2:.3f}; energy "
          f"{_fmt(e, 'energy')} p = {pen:.3f}")
    assert pl2 >= k + 1 - 0.2
    assert pen >= k - 0.2


@pytest.mark.slow
@pytest.mark.parametrize("name,ns", [("T4", (16, 24)), ("T10", (8, 12))])
def test_v3_posthoc_one_more_level(name, ns):
    """Post hoc (not pre-registered): the rate between the finest pre-registered level and one more."""
    k = get(name).degree
    e, pl2, pen = _rates(name, ns)
    print(f"\n  V3 post hoc {name} {ns}: L2 p = {pl2:.3f}, energy p = {pen:.3f}")
    assert pl2 >= k + 1 - 0.2
    assert pen >= k - 0.2


def test_v3_3d_smoke():
    """Fast-suite stand-in for the slow 3-D V3 cases: the errors fall on two coarse levels."""
    for name in ("H8", "T4", "T10"):
        a, b = _mms_errors(name, 2, 102), _mms_errors(name, 4, 104)
        assert b["l2"] < 0.6 * a["l2"] and b["energy"] < a["energy"]


@pytest.mark.parametrize("name", ["Q9", "T6"])
def test_v3b_curved_and_distorted_cells_are_refused(name):
    """Measured: vertices ±0.2 h plus curved O(h) mid nodes ±0.1 h invert some cells; the detJ check refuses the mesh
    (pre-registered V3b had no threshold; it could not be measured on this family)."""
    from toporlogy.fem.mesh import InvertedElementError
    with pytest.raises(InvertedElementError):
        _rates(name, (4, 8, 16, 32), curved=0.1)


@pytest.mark.parametrize("name", ["Q9"])
def test_v3b_curved_mesh_measured_posthoc(name):
    """Post hoc measurement (no threshold): straight vertex grid, curved O(h) mid nodes ±0.1 h (Q9 only: the T6
    family with the same amplitude has inverted cells at n = 32, refused by the detJ check)."""
    e, pl2, pen = _rates(name, (4, 8, 16, 32), curved=0.1, amp=0.0)
    print(f"\n  V3b {name} curved O(h) mid nodes (post hoc measurement): L2 p = {pl2:.2f}, energy p = {pen:.2f}")


def test_v3_mesh_type_is_general():
    """The distorted meshes of V3 really are distorted: Jacobians vary within and between cells."""
    m = perturb(rectangle(8, 8, element="Q4"), 0.2 / 8, seed=108)
    from toporlogy.fem.mesh import kernels as kn
    from toporlogy.fem.mesh.quadrature import rule
    g = kn.geometry(get("Q4"), m.cell_points(), rule("quad", 2))
    r = g.detJ.max(axis=1) / g.detJ.min(axis=1)
    assert r.max() > 1.2 and g.detJ.max() / g.detJ.min() > 1.5
    assert isinstance(m, Mesh)

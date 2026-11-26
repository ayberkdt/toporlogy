"""General mesh, assembly and boundary conditions (docs/olcumler/fem_genel_onkayit.md §2, G1-G10, M1-M2,
pre-registered 2026-11-26 02:41 +0300 before the first run)."""
from __future__ import annotations

import re

import numpy as np
import pytest
import scipy.sparse as sp

from toporlogy.core.domain import Domain
from toporlogy.core.linear_elastic import isotropic_plane_strain, isotropic_plane_stress, isotropic_stiffness
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import D_iso, D_plane_stress
from toporlogy.fem.mechanisms import MechanismError
from toporlogy.fem.mesh import (Dirichlet, ElasticProblem, HeatProblem, Mesh, MixedElasticity,
                                UnsupportedBoundaryCondition, box, get, perturb, rectangle)

SOLID = ("Q4", "Q9", "T3", "T6", "H8", "T4", "T10")


def _mesh(name, n=(4, 3, 2), amp=0.2, seed=3, curved=0.0):
    if get(name).dim == 2:
        m = rectangle(n[0], n[1], lx=1.2, ly=0.9, element=name)
        h = min(1.2 / n[0], 0.9 / n[1])
    else:
        m = box(n[0], n[1], n[2], lx=1.0, ly=0.8, lz=0.7, element=name)
        h = min(1.0 / n[0], 0.8 / n[1], 0.7 / n[2])
    return perturb(m, amp * h, seed=seed, curved=curved * h) if amp or curved else m


def _linear_field(dim, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) * 1e-2
    a = rng.normal(size=dim) * 1e-2
    return A, a


def _patch(m, D, thickness=1.0, degree=None):
    A, a = _linear_field(m.dim)
    P = ElasticProblem(m, D, thickness=thickness, degree=degree)
    bn = m.boundary_nodes()
    val = m.points[bn] @ A.T + a
    bc = Dirichlet([(m.dofs(bn, [c]), val[:, c]) for c in range(m.dim)])
    s = P.solve(np.zeros(P.ndof), bc)
    used = m.used_nodes()
    ue = m.points[used] @ A.T + a
    uh = s.u[:, 0].reshape(-1, m.dim)[used]
    err_u = np.abs(uh - ue).max() / np.abs(ue).max()
    eps = np.array([A[0, 0], A[1, 1], A[0, 1] + A[1, 0]]) if m.dim == 2 else \
        np.array([A[0, 0], A[1, 1], A[2, 2], A[1, 2] + A[2, 1], A[0, 2] + A[2, 0], A[0, 1] + A[1, 0]])
    sig = P.stress(s.u[:, 0])
    sx = D @ eps
    err_s = np.abs(sig - sx).max() / np.abs(sx).max()
    return err_u, err_s, s, P


def _D(dim, nu=0.3, plane="strain", E=1.0):
    if dim == 3:
        return isotropic_stiffness(E, nu)
    return isotropic_plane_strain(E, nu) if plane == "strain" else isotropic_plane_stress(E, nu)


# ---------------------------------------------------------------------------------------------- G1
def test_g1_from_domain_q4_equals_grid_path():
    frac = np.ones((12, 6))
    frac[4:7, 2:4] = 0.0                                          # a hole
    d = Domain((12, 6), h=0.7, frac=frac, thickness=0.6)
    E = np.random.default_rng(1).uniform(0.2, 2.0, d.n_active)
    left = d.nodes_where(lambda X: X[:, 0] < 1e-9)
    fixed = d.dofs(left)
    F = d.load_vector(2)
    tip = d.nodes_where(lambda X: X[:, 0] > 12 * 0.7 - 1e-9)
    F[d.dofs(tip, 1), 0] = -1.0
    F[d.dofs(tip, 0), 1] = 0.5
    grid = Elasticity(d, fixed, nu=0.3, element="Q4")
    Ug = grid.solve_numpy(E, F)
    m = Mesh.from_domain(d)
    P = ElasticProblem(m, E[:, None, None] * D_plane_stress(0.3), thickness=0.6)
    s = P.solve(F, Dirichlet.fixed(fixed, 2))
    Kg = grid.matrix(E)
    Km = sp.triu(P.K[grid.free][:, grid.free], format="csr")
    dK = abs(Kg - Km).max() / abs(Kg).max()
    du = np.abs(s.u - Ug).max() / np.abs(Ug).max()
    print(f"\n  G1 Q4: K {dK:.1e}, u {du:.1e}")
    assert dK <= 1e-14
    assert du <= 1e-12


def test_g1_from_domain_h8_equals_grid_path():
    frac = np.ones((5, 4, 3))
    frac[2, 1:3, 1] = 0.0
    d = Domain((5, 4, 3), h=0.5, frac=frac)
    E = np.random.default_rng(2).uniform(0.2, 2.0, d.n_active)
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))
    F = d.load_vector(1)
    F[d.dofs(d.nodes_where(lambda X: X[:, 0] > 2.5 - 1e-9), 2), 0] = -1.0
    grid = Elasticity(d, fixed, nu=0.3, element="H8")
    Ug = grid.solve_numpy(E, F)
    m = Mesh.from_domain(d)
    P = ElasticProblem(m, E[:, None, None] * D_iso(0.3))          # physical cube of edge h: K = E · h · KE(unit)
    s = P.solve(F, Dirichlet.fixed(fixed))
    Kg = grid.matrix(E)
    Km = sp.triu(P.K[grid.free][:, grid.free], format="csr")
    dK = abs(Kg - Km).max() / abs(Kg).max()
    du = np.abs(s.u - Ug).max() / np.abs(Ug).max()
    print(f"\n  G1 H8: K {dK:.1e}, u {du:.1e}")
    assert dK <= 1e-14
    assert du <= 1e-12


# ---------------------------------------------------------------------------------------------- G2, G3
@pytest.mark.parametrize("plane", ["stress", "strain"])
@pytest.mark.parametrize("name", SOLID)
def test_g2_affine_patch_test(name, plane):
    if get(name).dim == 3 and plane == "stress":
        pytest.skip("3-D has no plane assumption")
    m = _mesh(name)
    D = _D(m.dim, plane=plane)
    eu, es, s, P = _patch(m, D, thickness=0.7 if m.dim == 2 else 1.0)
    # G3: no external load, so the reactions alone are in equilibrium (force and moment)
    R = s.reactions[:, 0].reshape(-1, m.dim)
    X = m.points
    scale = np.abs(R).sum()
    f_err = np.abs(R.sum(axis=0)).max() / scale
    mom = (X[:, 0] * R[:, 1] - X[:, 1] * R[:, 0]).sum() if m.dim == 2 else np.cross(X, R).sum(axis=0)
    m_err = np.abs(mom).max() / (scale * np.abs(X).max())
    print(f"\n  G2 {name} {plane}: u {eu:.1e}, stress {es:.1e}; G3 force {f_err:.1e}, moment {m_err:.1e}")
    assert eu <= 1e-10
    assert es <= 1e-9
    assert f_err <= 1e-12 and m_err <= 1e-12


@pytest.mark.parametrize("name,degree", [("Q9", None), ("T6", None), ("T10", 3)])
def test_g2b_curved_patch_test(name, degree):
    m = _mesh(name, curved=0.1)
    eu, es, _, _ = _patch(m, _D(m.dim), degree=degree)
    print(f"\n  G2b {name} curved (degree {degree}): u {eu:.1e}, stress {es:.1e}")
    assert eu <= 1e-10


def test_g2c_curved_t10_default_rule_measured():
    m = _mesh("T10", curved=0.1)
    eu, es, _, _ = _patch(m, _D(3))
    print(f"\n  G2c T10 curved, 4-point rule (measurement, no threshold): u {eu:.3e}, stress {es:.3e}")


def test_g3_equilibrium_with_body_force_and_traction():
    m = _mesh("T6")
    P = ElasticProblem(m, _D(2), thickness=0.5)
    f = P.body_force([0.3, -1.0]) + P.traction(m.facet_sets["right"], [0.2, 0.4])
    s = P.solve(f, Dirichlet.fixed(m.dofs(m.node_sets["left"])))
    R = s.reactions[:, 0].reshape(-1, 2)
    Fv = f.reshape(-1, 2)
    X = m.points
    tot = R + Fv
    scale = np.abs(Fv).sum()
    assert np.abs(tot.sum(axis=0)).max() <= 1e-12 * scale
    mom = (X[:, 0] * tot[:, 1] - X[:, 1] * tot[:, 0]).sum()
    assert abs(mom) <= 1e-12 * scale * np.abs(X).max()


# ---------------------------------------------------------------------------------------------- G4
@pytest.mark.parametrize("name", SOLID)
def test_g4_consistent_loads(name):
    m = _mesh(name)
    t = 0.7 if m.dim == 2 else 1.0
    P = ElasticProblem(m, _D(m.dim), thickness=t)
    dim = m.dim
    L = (1.2, 0.9) if dim == 2 else (1.0, 0.8, 0.7)
    area = (0.9 * t) if dim == 2 else 0.8 * 0.7
    vol = np.prod(L) * t if dim == 2 else np.prod(L)
    right = m.facet_sets["right"]
    tv = np.arange(1, dim + 1) * 0.3
    ft = P.traction(right, tv).reshape(-1, dim).sum(axis=0)
    assert np.abs(ft - tv * area).max() <= 1e-12 * np.abs(tv * area).max()
    # linear traction in y on the face x = L: force and moment exact
    a, b = 0.4, 1.3
    fl = P.traction(right, lambda X: np.column_stack([np.zeros(len(X)), a + b * X[:, 1]] + [np.zeros(len(X))] * (dim - 2)))
    fl = fl.reshape(-1, dim)
    ly = L[1]
    w = t if dim == 2 else L[2]
    Fy = (a * ly + b * ly ** 2 / 2) * w
    Mz = L[0] * Fy                                     # Σ x f_y with x = L on the face (f_x = 0)
    assert abs(fl[:, 1].sum() - Fy) <= 1e-12 * Fy
    mom = (m.points[:, 0] * fl[:, 1] - m.points[:, 1] * fl[:, 0]).sum()
    assert abs(mom - Mz) <= 1e-12 * Mz
    # pressure on one face: Σ f = −p A n, n = +x
    fp = P.pressure(right, 2.5).reshape(-1, dim).sum(axis=0)
    exp = np.zeros(dim)
    exp[0] = -2.5 * area
    assert np.abs(fp - exp).max() <= 1e-12 * 2.5 * area
    # closed surface: zero net force
    fc = P.pressure(m.boundary_facets(), 1.7).reshape(-1, dim).sum(axis=0)
    assert np.abs(fc).max() <= 1e-12 * 1.7 * area
    # body force
    bv = np.arange(1, dim + 1) * -0.6
    fb = P.body_force(bv).reshape(-1, dim).sum(axis=0)
    assert np.abs(fb - bv * vol).max() <= 1e-12 * np.abs(bv * vol).max()


def _curved_facet_case(name, degree):
    m = rectangle(1, 1, element=name)
    f = m.facet_sets["right"]
    fn = m.facet_nodes(f)[0]
    m.points[fn[2], 0] += 0.2                          # bulge the facet outward (mid node of the L3 facet)
    P = ElasticProblem(m, _D(2))
    tv = np.array([0.3, -0.8])
    tot = P.traction(f, tv, degree=degree).reshape(-1, 2).sum(axis=0)
    from toporlogy.fem.mesh import kernels as kn
    fg = kn.facet_geometry(get("L3"), m.points[fn][None], m.cell_centroids()[f[:, 0]], 40)
    length = fg.dS.sum()                               # arc length, 40th-degree rule
    assert length > 1.0 + 1e-3
    return np.abs(tot - tv * length).max() / np.abs(tv * length).max()


@pytest.mark.xfail(strict=True, reason="pre-registered G4 (curved L3, default facet rule): measured 1.5e-5 > 1e-12; "
                                       "the arc-length integrand sqrt(poly) is not integrated exactly by the "
                                       "default degree-6 facet rule (fem_genel.md §2)")
@pytest.mark.parametrize("name", ["Q9", "T6"])
def test_g4_curved_facet_traction(name):
    err = _curved_facet_case(name, None)
    print(f"\n  G4 curved {name}, default rule: {err:.2e}")
    assert err <= 1e-12


@pytest.mark.parametrize("name", ["Q9", "T6"])
def test_g4_curved_facet_traction_posthoc_high_order_rule(name):
    """Post hoc (not pre-registered): with a degree-40 facet rule the total equals t times the arc length."""
    err = _curved_facet_case(name, 40)
    print(f"\n  G4 curved {name}, degree 40 (post hoc): {err:.2e}")
    assert err <= 1e-12


# ---------------------------------------------------------------------------------------------- G5
def test_g5_support_rank_diagnosis():
    m = rectangle(4, 2, element="Q4")
    P = ElasticProblem(m, _D(2))
    f = np.zeros(P.ndof)
    with pytest.raises(MechanismError, match=r"1 free rigid motion"):
        P.solve(f, Dirichlet.fixed(m.dofs([0])))
    # two pieces, only one supported
    a = rectangle(2, 2, element="T3")
    b = rectangle(2, 2, element="T3", origin=(3.0, 0.0))
    two = Mesh(np.vstack([a.points, b.points]), np.vstack([a.cells, b.cells + a.n_nodes]), "T3")
    P2 = ElasticProblem(two, _D(2))
    with pytest.raises(MechanismError) as ei:
        P2.solve(np.zeros(P2.ndof), Dirichlet.fixed(two.dofs(a.node_sets["left"])))
    assert re.search(r"piece 1 \(9 nodes\): 3 free rigid motion", str(ei.value))
    P.solve(f, Dirichlet.fixed(m.dofs(m.node_sets["left"])))          # adequate supports pass


def _face_x_only_message():
    m3 = box(2, 2, 2, element="H8")
    P3 = ElasticProblem(m3, _D(3))
    with pytest.raises(MechanismError) as ei:
        P3.solve(np.zeros(P3.ndof), Dirichlet.fixed(m3.dofs(m3.node_sets["left"], [0])))
    return str(ei.value)


@pytest.mark.xfail(strict=True, reason="pre-registered G5 expected 5 free rigid motions for a face held in x only; "
                                       "measured 3 (u_x = 0 on the plane x = 0 also fixes the rotations about y and "
                                       "z): the expectation was wrong, the diagnosis is right (fem_genel.md §2)")
def test_g5_3d_face_held_in_x_only_preregistered():
    assert re.search(r"5 free rigid motion", _face_x_only_message())


def test_g5_3d_face_held_in_x_only_posthoc():
    """Post hoc: translations y, z and the rotation about x stay free."""
    assert re.search(r"piece 0 \(27 nodes\): 3 free rigid motion", _face_x_only_message())


# ---------------------------------------------------------------------------------------------- G6
def test_g6_unsupported_combinations_raise():
    m = rectangle(3, 2, element="T6")
    P = ElasticProblem(m, _D(2))
    with pytest.raises(ValueError, match="constrained twice"):
        Dirichlet([(np.array([0, 1]), 0.0), (np.array([1]), 0.5)])
    inner = np.array([[0, 1]])                         # facet 1 of cell 0 (the diagonal) is interior
    assert not m.is_boundary(inner)[0]
    with pytest.raises(UnsupportedBoundaryCondition, match="not on the boundary"):
        P.traction(inner, [1.0, 0.0])
    with pytest.raises(UnsupportedBoundaryCondition):
        P.pressure(inner, 1.0)
    with pytest.raises(ValueError, match="load case"):
        P.solve(np.zeros((P.ndof, 3)), Dirichlet([(m.dofs(m.node_sets["left"]), 0.0)], n_cases=2))
    with pytest.raises(ValueError, match="plane stress"):
        MixedElasticity(m, 1.0, 10.0, plane="stress")
    with pytest.raises(ValueError):                    # ν = 0.5 has no displacement-only D
        isotropic_plane_strain(1.0, 0.5)
    lam = 1e300
    with pytest.raises(ValueError):
        ElasticProblem(m, np.array([[lam, lam, 0], [lam, lam, 0], [0, 0, 1.0]]))   # not positive definite
    with pytest.raises(ValueError, match="grid only"):
        get("Q4IM")
    with pytest.raises(ValueError, match="grid"):
        Mesh.from_domain(Domain((3, 2)), "Q4IM")
    with pytest.raises(ValueError, match="unknown element"):
        get("Q8")
    with pytest.raises(ValueError, match="D must be"):
        ElasticProblem(m, np.eye(6))
    with pytest.raises(UnsupportedBoundaryCondition):
        P.constrain("periodic")
    with pytest.raises(ValueError, match="mixed cell types"):
        Mesh(m.points, {"T6": m.cells}, "T6")


# ---------------------------------------------------------------------------------------------- G7, G8
def test_g7_multiple_right_hand_sides_one_factorization():
    m = _mesh("T10")
    P = ElasticProblem(m, _D(3))
    rng = np.random.default_rng(4)
    F = rng.normal(size=(P.ndof, 3))
    bc = Dirichlet.fixed(m.dofs(m.node_sets["left"]), 3)
    U = P.solve(F, bc).u
    sep = np.column_stack([P.solve(F[:, j], Dirichlet.fixed(m.dofs(m.node_sets["left"]))).u[:, 0]
                           for j in range(3)])
    assert np.abs(U - sep).max() <= 1e-13 * np.abs(U).max()
    assert P.n_factorizations == 1


def test_g8_pattern_reuse():
    m = _mesh("Q9")
    P = ElasticProblem(m, _D(2))
    K0 = P.K
    pat = P.pattern
    n0 = pat.n_assemblies
    E = np.random.default_rng(5).uniform(0.5, 2, m.n_cells)
    D2 = E[:, None, None] * _D(2)
    P.set_D(D2)
    K1 = P.K
    assert P.pattern is pat and pat.n_assemblies == n0 + 1
    Kf = ElasticProblem(m, D2).K
    assert abs(K1 - Kf).max() <= 1e-15 * abs(Kf).max()
    assert abs(K1 - K0).max() > 0


# ---------------------------------------------------------------------------------------------- G9, G10
@pytest.mark.parametrize("name", SOLID)
def test_g9_two_material_bar(name):
    E1, E2, s0 = 1.0, 4.0, 0.3
    if get(name).dim == 2:
        m = rectangle(8, 2, lx=2.0, ly=0.5, element=name)
        Dz = np.diag([1.0, 1.0, 0.5])
    else:
        m = box(8, 2, 2, lx=2.0, ly=0.5, lz=0.5, element=name)
        Dz = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    E = np.where(m.cell_centroids()[:, 0] < 1.0, E1, E2)
    P = ElasticProblem(m, E[:, None, None] * Dz)
    tv = np.zeros(m.dim)
    tv[0] = s0
    s = P.solve(P.traction(m.facet_sets["right"], tv), Dirichlet.fixed(m.dofs(m.node_sets["left"])))
    x = m.points[:, 0]
    ux = np.where(x <= 1.0, s0 * x / E1, s0 / E1 + s0 * (x - 1.0) / E2)
    U = s.u[:, 0].reshape(-1, m.dim)
    err = max(np.abs(U[:, 0] - ux).max(), np.abs(U[:, 1:]).max()) / np.abs(ux).max()
    print(f"\n  G9 {name}: {err:.1e}")
    assert err <= 1e-11


def test_g10_t10_equals_post_verify_stiffness():
    from toporlogy.post.verify import TetMesh, stiffness_t10
    m = _mesh("T10")
    faces = m.facet_nodes(m.boundary_facets())
    tm = TetMesh(m.points, m.cells, faces)
    Kv = stiffness_t10(tm, 2.0, 0.3)
    K = ElasticProblem(Mesh(tm.X, tm.T, "T10"), isotropic_stiffness(2.0, 0.3)).K
    d = __import__("scipy.sparse.linalg", fromlist=["norm"]).norm(K - Kv) / \
        __import__("scipy.sparse.linalg", fromlist=["norm"]).norm(Kv)
    print(f"\n  G10: {d:.1e}")
    assert d <= 1e-13


# ---------------------------------------------------------------------------------------------- M1, M2
@pytest.mark.parametrize("name", SOLID)
def test_m1_mass(name):
    m = _mesh(name)
    t = 0.7 if m.dim == 2 else 1.0
    P = ElasticProblem(m, _D(m.dim), thickness=t)
    vol = (1.2 * 0.9 * t) if m.dim == 2 else 1.0 * 0.8 * 0.7
    rho = 2.3
    M = P.mass(rho)
    assert abs(M.sum() / m.dim - rho * vol) <= 1e-13 * rho * vol
    Ml = P.mass(rho, lumped=True)
    d = Ml.diagonal()[P.active]
    assert (d > 0).all()
    assert abs(Ml.sum() / m.dim - rho * vol) <= 1e-13 * rho * vol


@pytest.mark.parametrize("name", SOLID)
def test_m2_heat_patch_and_robin(name):
    m = _mesh(name)
    H = HeatProblem(m, kappa=1.7, thickness=0.7 if m.dim == 2 else 1.0)
    b = np.arange(1, m.dim + 1) * 0.4
    bn = m.boundary_nodes()
    s = H.solve(np.zeros(H.ndof), Dirichlet([(bn, 2.0 + m.points[bn] @ b)]))
    used = m.used_nodes()
    Tx = 2.0 + m.points[used] @ b
    e1 = np.abs(s.u[used, 0] - Tx).max() / np.abs(Tx).max()
    # slab: T0 on x = 0, convection h (T − T∞) on x = Lx, other faces insulated
    Lx, kap, h, T0, Tinf = (1.2 if m.dim == 2 else 1.0), 1.7, 3.0, 1.0, 5.0
    H2 = HeatProblem(m, kappa=kap, thickness=0.7 if m.dim == 2 else 1.0)
    fr = H2.robin(m.facet_sets["right"], h, Tinf)
    s2 = H2.solve(fr, Dirichlet([(m.node_sets["left"], T0)]))
    x = m.points[used, 0]
    Te = T0 + (Tinf - T0) * h * x / (kap + h * Lx)
    e2 = np.abs(s2.u[used, 0] - Te).max() / np.abs(Te).max()
    print(f"\n  M2 {name}: patch {e1:.1e}, Robin slab {e2:.1e}")
    assert e1 <= 1e-10 and e2 <= 1e-10
    with pytest.raises(MechanismError):
        HeatProblem(m).solve(np.zeros(H.ndof))

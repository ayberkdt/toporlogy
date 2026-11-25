"""A second, separately written finite-element implementation for the independent checks of
docs/olcumler/dogrulama_turu.md (B, S): bilinear / trilinear elements on the regular grid, written from the
definitions with nothing of the library except the grid conventions that the tests assert first (element e ↔ grid
index in C order, node n ↔ grid index of the (shape + 1) node grid in C order, DOF = node · dim + component).

Conventions written out here:

* reference cell ξ ∈ [0, 1]^dim, X = (cell index + ξ) h, N_a(ξ) = Π_d (ξ_d if corner_d = 1 else 1 − ξ_d), corners
  in C order of {0, 1}^dim (the order does not matter: assembly goes through the node numbers);
* 2-point Gauss rule on [0, 1] per axis, ξ = (1 ± 1/√3)/2, weights 1/2;
* Voigt order [xx, yy, xy] (2-D) and [xx, yy, zz, yz, xz, xy] (3-D), engineering shear strains;
* orthotropic stiffness from the engineering constants by inverting the compliance, rotated as a fourth-order
  tensor C'_ijkl = R_ia R_jb R_kc R_ld C_abcd (R columns = material axes in global coordinates); plane stress by
  inverting the in-plane block of the rotated 3-D compliance.
"""
from __future__ import annotations

import itertools

import numpy as np

VOIGT = {2: [(0, 0), (1, 1), (0, 1)], 3: [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]}
_GP = np.array([(1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2])


def corners(dim: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=dim)))


def gauss(dim: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array(list(itertools.product(_GP, repeat=dim)))
    return pts, np.full(len(pts), 0.5 ** dim)


def shape_gradients(xi: np.ndarray, h: float) -> np.ndarray:
    """dN_a/dX_d (n_corners, dim) at ξ for an element of edge h."""
    dim = len(xi)
    out = np.empty((2 ** dim, dim))
    for a, c in enumerate(corners(dim)):
        for d in range(dim):
            g = 1.0
            for q in range(dim):
                f = xi[q] if c[q] else 1.0 - xi[q]
                df = 1.0 if c[q] else -1.0
                g *= df if q == d else f
            out[a, d] = g / h
    return out


def strain_matrix(dN: np.ndarray) -> np.ndarray:
    """B (n_voigt, dim · n_corners), node-major DOFs, engineering shear."""
    n, dim = dN.shape
    B = np.zeros((len(VOIGT[dim]), dim * n))
    for v, (i, j) in enumerate(VOIGT[dim]):
        for a in range(n):
            if i == j:
                B[v, a * dim + i] = dN[a, i]
            else:
                B[v, a * dim + i] = dN[a, j]
                B[v, a * dim + j] = dN[a, i]
    return B


def element_dofs(shape: tuple[int, ...]) -> np.ndarray:
    """(n_elements, dim · 2^dim) global DOFs of every element, in this module's corner order."""
    dim = len(shape)
    node_shape = np.array(shape) + 1
    cells = np.indices(shape).reshape(dim, -1).T
    out = []
    for cell in cells:
        nodes = [int(np.ravel_multi_index(tuple(cell + c), node_shape)) for c in corners(dim)]
        out.append([n * dim + k for n in nodes for k in range(dim)])
    return np.array(out)


# ----------------------------------------------------------------------------------------------- materials
def isotropic_3d(E: float, nu: float) -> np.ndarray:
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[3:, 3:] = mu * np.eye(3)
    return C


def plane_stress_isotropic(E: float, nu: float) -> np.ndarray:
    return E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def orthotropic_3d(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23) -> np.ndarray:
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / G23, 1 / G13, 1 / G12
    return np.linalg.inv(S)


def _tensor(Cv: np.ndarray) -> np.ndarray:
    C = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT[3]):
        for J, (k, m) in enumerate(VOIGT[3]):
            for a, b in {(i, j), (j, i)}:
                for c, d in {(k, m), (m, k)}:
                    C[a, b, c, d] = Cv[I, J]
    return C


def rotate(Cv: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Voigt stiffness in global axes for material axes R (columns)."""
    C = np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, _tensor(Cv))
    out = np.zeros((6, 6))
    for I, (i, j) in enumerate(VOIGT[3]):
        for J, (k, m) in enumerate(VOIGT[3]):
            out[I, J] = C[i, j, k, m]
    return out


def plane_stress(C3: np.ndarray) -> np.ndarray:
    S = np.linalg.inv(C3)
    ip = [0, 1, 5]
    return np.linalg.inv(S[np.ix_(ip, ip)])


def rot_z(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------------------------- assembly
def stiffness(shape, h, t, Ds, E) -> np.ndarray:
    """Dense K (ndof, ndof): Ds (n_el, nv, nv) physical-unit D per element (times E[e])."""
    dim = len(shape)
    edofs = element_dofs(shape)
    ndof = int(np.prod(np.array(shape) + 1)) * dim
    K = np.zeros((ndof, ndof))
    pts, w = gauss(dim)
    vol = h ** dim * (t if dim == 2 else 1.0)
    for e, ed in enumerate(edofs):
        ke = np.zeros((len(ed), len(ed)))
        for xi, wg in zip(pts, w, strict=True):
            B = strain_matrix(shape_gradients(xi, h))
            ke += wg * vol * B.T @ (E[e] * Ds[e]) @ B
        K[np.ix_(ed, ed)] += ke
    return K


def gauss_stresses(shape, h, Ds, E, u) -> np.ndarray:
    """(n_el, n_gauss, nv) stresses σ = E_e D_e B u_e."""
    dim = len(shape)
    edofs = element_dofs(shape)
    pts, _ = gauss(dim)
    out = np.empty((len(edofs), len(pts), len(VOIGT[dim])))
    for e, ed in enumerate(edofs):
        for g, xi in enumerate(pts):
            out[e, g] = E[e] * Ds[e] @ strain_matrix(shape_gradients(xi, h)) @ u[ed]
    return out


def centroid_stresses(shape, h, Ds, E, u) -> np.ndarray:
    """(n_el, nv) stresses at the element centre."""
    dim = len(shape)
    B = strain_matrix(shape_gradients(np.full(dim, 0.5), h))
    return np.array([E[e] * Ds[e] @ B @ u[ed] for e, ed in enumerate(element_dofs(shape))])


def geometric_stiffness(shape, h, t, Ds, E, u) -> np.ndarray:
    """Dense K_G = Σ_e ∫ ∇Nᵀ σ ∇N dV ⊗ I for the reference displacement u."""
    dim = len(shape)
    edofs = element_dofs(shape)
    ndof = int(np.prod(np.array(shape) + 1)) * dim
    KG = np.zeros((ndof, ndof))
    pts, w = gauss(dim)
    vol = h ** dim * (t if dim == 2 else 1.0)
    sig = gauss_stresses(shape, h, Ds, E, u)
    for e, ed in enumerate(edofs):
        kn = np.zeros((2 ** dim, 2 ** dim))
        for g, (xi, wg) in enumerate(zip(pts, w, strict=True)):
            S = np.zeros((dim, dim))
            for v, (i, j) in enumerate(VOIGT[dim]):
                S[i, j] = S[j, i] = sig[e, g, v]
            dN = shape_gradients(xi, h)
            kn += wg * vol * dN @ S @ dN.T
        ke = np.kron(kn, np.eye(dim))                      # node-major DOFs: (a, i), (b, j) -> kn[a, b] δ_ij
        KG[np.ix_(ed, ed)] += ke
    return KG


# ----------------------------------------------------------------------------------- incompatible modes, mass, heat
def _mode_strain(xi: np.ndarray, h: float) -> np.ndarray:
    """B (n_voigt, dim · dim) of Wilson's modes 1 − s_d² (s = 2ξ − 1 ∈ [−1, 1]), one per direction d and
    displacement component c; columns ordered (mode d, component c)."""
    dim = len(xi)
    s = 2 * xi - 1
    dM = np.zeros((dim, dim))                               # dM[d, j] = ∂(1 − s_d²)/∂X_j = −2 s_d · 2/h δ_dj
    for d in range(dim):
        dM[d, d] = -4.0 * s[d] / h
    return strain_matrix(dM)                               # same layout as nodes: "node" = mode, DOF = component


def element_stiffness(dim: int, h: float, t: float, D: np.ndarray, incompatible: bool = False) -> np.ndarray:
    """Element stiffness (dim · 2^dim square), modes condensed statically when ``incompatible``."""
    pts, w = gauss(dim)
    vol = h ** dim * (t if dim == 2 else 1.0)
    nu_ = dim * 2 ** dim
    na = dim * dim if incompatible else 0
    K = np.zeros((nu_ + na, nu_ + na))
    for xi, wg in zip(pts, w, strict=True):
        B = strain_matrix(shape_gradients(xi, h))
        if incompatible:
            B = np.hstack([B, _mode_strain(xi, h)])
        K += wg * vol * B.T @ D @ B
    if not incompatible:
        return K
    Kuu, Kua, Kaa = K[:nu_, :nu_], K[:nu_, nu_:], K[nu_:, nu_:]
    return Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)


def assemble(shape, element_matrices) -> np.ndarray:
    """Dense global matrix from per-element matrices in this module's DOF order (any DOFs per node)."""
    dim = len(shape)
    nper = element_matrices.shape[1] // 2 ** dim
    node_shape = np.array(shape) + 1
    n = int(np.prod(node_shape)) * nper
    A = np.zeros((n, n))
    for cell, ke in zip(np.indices(shape).reshape(dim, -1).T, element_matrices, strict=True):
        nodes = [int(np.ravel_multi_index(tuple(cell + c), node_shape)) for c in corners(dim)]
        ed = [m * nper + k for m in nodes for k in range(nper)]
        A[np.ix_(ed, ed)] += ke
    return A


def element_mass(dim: int, h: float, t: float, rho: float) -> np.ndarray:
    """Consistent mass ρ ∫ NᵀN dV ⊗ I (2-point Gauss is exact: degree 2 per axis)."""
    pts, w = gauss(dim)
    vol = h ** dim * (t if dim == 2 else 1.0)
    m = np.zeros((2 ** dim, 2 ** dim))
    for xi, wg in zip(pts, w, strict=True):
        N = np.array([np.prod([xi[q] if c[q] else 1 - xi[q] for q in range(dim)]) for c in corners(dim)])
        m += wg * vol * rho * np.outer(N, N)
    return np.kron(m, np.eye(dim))


def element_conduction(dim: int, h: float, t: float, k: float) -> np.ndarray:
    pts, w = gauss(dim)
    vol = h ** dim * (t if dim == 2 else 1.0)
    c = np.zeros((2 ** dim, 2 ** dim))
    for xi, wg in zip(pts, w, strict=True):
        dN = shape_gradients(xi, h)
        c += wg * vol * k * dN @ dN.T
    return c

"""Mathematical invariants of the core pieces, checked on random inputs (hypothesis, derandomised).

Each property is something the implementation must satisfy whatever its internals, and each expected value is
computed here independently (own flood fill, explicit sums, the second filter backend), so a test cannot pass by
reproducing the code under test. The docstrings name the defect each one catches.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import toporlogy as T
from toporlogy.design.chain import DensityFilter, HeavisideProjection
from toporlogy.optimization.connectivity import label_bodies
from toporlogy.optimization.responses import EvalContext, Volume

PROFILE = settings(derandomize=True, max_examples=25, deadline=None,
                   suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


@st.composite
def domains(draw, dim=None, min_active=0.5):
    dim = draw(st.sampled_from([2, 3])) if dim is None else dim
    shape = tuple(draw(st.integers(3, 7 if dim == 2 else 4)) for _ in range(dim))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    frac = np.where(rng.random(shape) < min_active + (1 - min_active) * rng.random(), 1.0, 0.0)
    frac.flat[rng.integers(frac.size)] = 1.0                       # never empty
    partial = rng.random(shape) < 0.2
    frac[partial & (frac > 0)] = rng.uniform(0.2, 1.0, int((partial & (frac > 0)).sum()))
    h = draw(st.sampled_from([0.5, 1.0, 2.5]))                       # h != 1: 3-D stiffness scales with h
    return T.Domain(shape, h, frac=frac), rng


def _grid(d, rng):
    return torch.as_tensor(np.where(d.active, rng.random(d.shape), 0.0))


# ---------------------------------------------------------------------------------------------- density filter
@PROFILE
@given(domains(), st.floats(1.0, 3.0))
def test_filter_partition_of_unity_and_bounds(dr, rmin):
    """A constant design is filtered to the same constant on the part (weights normalised over active elements
    only), and a filtered field stays within the range of the input: catches unnormalised weights and weights that
    reach outside the part."""
    d, rng = dr
    f = DensityFilter(d, rmin)
    c = float(rng.uniform(0.1, 0.9))
    out = f(torch.as_tensor(np.where(d.active, c, 0.0))).numpy()
    assert np.allclose(out[d.active], c, rtol=0, atol=1e-13)
    assert np.all(out[~d.active] == 0.0)
    x = _grid(d, rng)
    y = f(x).numpy()[d.active]
    xa = x.numpy()[d.active]
    assert y.min() >= xa.min() - 1e-13 and y.max() <= xa.max() + 1e-13


@PROFILE
@given(domains(), st.floats(1.0, 3.0))
def test_filter_is_linear_and_its_backward_is_the_transpose(dr, rmin):
    """F(a x + b y) = a F(x) + b F(y), and the custom sparse backward gives F^T: <F x, w> = <x, F^T w> with F^T w
    taken from autograd. Catches a backward that uses H instead of H^T or forgets the 1/Hs normalisation."""
    d, rng = dr
    f = DensityFilter(d, rmin, backend="sparse")
    x, y, w = _grid(d, rng), _grid(d, rng), _grid(d, rng)
    a, b = rng.uniform(-2, 2, 2)
    assert torch.allclose(f(a * x + b * y), a * f(x) + b * f(y), rtol=0, atol=1e-12)
    xr = x.clone().requires_grad_(True)
    (f(xr) * w).sum().backward()
    ftw = xr.grad
    # explicit F^T w through the matrix the forward uses
    H = f.matrix()
    ftw_ref = torch.as_tensor((H.T @ w.numpy().ravel()).reshape(d.shape))
    assert torch.allclose(ftw, ftw_ref, rtol=0, atol=1e-12)
    assert float((f(x) * w).sum()) == pytest.approx(float((x * ftw).sum()), rel=1e-12, abs=1e-12)


@PROFILE
@given(domains(), st.floats(1.0, 3.0))
def test_sparse_and_convolution_filters_agree(dr, rmin):
    """The two backends are independent implementations (sparse matrix on CPU, convolution elsewhere); the
    convolution path had no test on CPU. They must give the same field and the same gradient."""
    d, rng = dr
    x, w = _grid(d, rng), _grid(d, rng)
    out = {}
    for backend in ("sparse", "conv"):
        xr = x.clone().requires_grad_(True)
        y = DensityFilter(d, rmin, backend=backend)(xr)
        (y * w).sum().backward()
        out[backend] = (y.detach(), xr.grad)
    assert torch.allclose(out["sparse"][0], out["conv"][0], rtol=0, atol=1e-12)
    assert torch.allclose(out["sparse"][1], out["conv"][1], rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------- projection
@PROFILE
@given(st.floats(0.5, 64.0), st.floats(0.05, 0.95))
def test_projection_fixes_0_and_1_and_is_monotone(beta, eta):
    p = HeavisideProjection(beta, eta)
    x = torch.linspace(0, 1, 401, dtype=torch.float64)
    y = p(x)
    assert float(y[0]) == pytest.approx(0.0, abs=1e-14) and float(y[-1]) == pytest.approx(1.0, abs=1e-14)
    assert bool(torch.all(torch.diff(y) >= -1e-15))
    xe = torch.tensor([eta], dtype=torch.float64)
    a, dd = np.tanh(beta * eta), np.tanh(beta * (1 - eta))
    assert float(p(xe)) == pytest.approx(a / (a + dd), rel=1e-14)    # value at the threshold, closed form


@PROFILE
@given(st.floats(0.5, 64.0))
def test_projection_with_eta_one_half_is_antisymmetric(beta):
    p = HeavisideProjection(beta, 0.5)
    x = torch.linspace(0, 1, 201, dtype=torch.float64)
    assert torch.allclose(p(1 - x), 1 - p(x), rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------------------------- stiffness
def _assembled_here(phys, E):
    """Full stiffness on phys.free, assembled here element by element: sum_e E_e geom P_e^T KE P_e."""
    K = np.zeros((phys.domain.ndof, phys.domain.ndof))
    for e, dofs in enumerate(phys.edof):
        K[np.ix_(dofs, dofs)] += E[e] * phys.geom * phys.KE
    return K[np.ix_(phys.free, phys.free)]


@PROFILE
@given(domains(min_active=0.7), st.integers(0, 2**31 - 1))
def test_unconstrained_stiffness_is_symmetric_psd_with_rigid_body_null_space(dr, seed):
    """K = K^T >= 0 and every rigid-body motion (translations; the linearised rotations) costs no energy. Catches a
    wrong element matrix, a sign error in assembly and wrong node coordinates."""
    d, _ = dr
    rng = np.random.default_rng(seed)
    phys = T.Elasticity(d, np.zeros(0, np.int64))
    E = rng.uniform(0.1, 2.0, d.n_active)
    assert np.abs(phys.KE - phys.KE.T).max() <= 1e-15 * np.abs(phys.KE).max()       # element matrix symmetric
    K = _assembled_here(phys, E)
    A = phys.matrix(E).toarray()                                     # library: upper triangle on phys.free
    assert np.allclose(A, np.triu(K), rtol=0, atol=1e-12 * np.abs(K).max())
    assert np.abs(np.tril(A, -1)).max() == 0.0
    ev = np.linalg.eigvalsh(K)
    assert ev.min() >= -1e-10 * ev.max()
    X = d.node_coords()
    used = np.unique(phys.edof // d.dim)                              # nodes of active elements
    modes = [np.tile(np.eye(d.dim)[c], (len(X), 1)) for c in range(d.dim)]
    pairs = [(0, 1)] if d.dim == 2 else [(0, 1), (1, 2), (0, 2)]
    for i, j in pairs:
        r = np.zeros((len(X), d.dim))
        r[:, i], r[:, j] = -X[:, j], X[:, i]
        modes.append(r)
    assert K.shape == (len(phys.free), len(phys.free))               # the DOFs of nodes of active elements
    for m in modes:
        u = np.zeros(d.ndof)
        u[(used[:, None] * d.dim + np.arange(d.dim)).ravel()] = m[used].ravel()
        uf = u[phys.free]
        assert np.abs(K @ uf).max() <= 1e-10 * np.abs(K).max() * max(1.0, np.abs(uf).max())


@PROFILE
@given(domains(min_active=1.0), st.integers(0, 2**31 - 1))
def test_compliance_equals_summed_element_energy_and_scales(dr, seed):
    """F^T U equals the sum of element energies E_e geom u_e^T KE u_e (assembly vs element level), doubling all
    moduli halves U, and U is linear in F. Catches assembly that drops or doubles element contributions.
    Every element active (partial volumes allowed): a random void pattern can leave a floating piece, and the
    solve of that singular system does not raise (recorded as B-22)."""
    d, _ = dr
    rng = np.random.default_rng(seed)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    phys = T.Elasticity(d, fixed)
    used = np.unique(phys.edof // d.dim)
    assert len(np.setdiff1d(used, fixed // d.dim)) > 0                # every element active: free nodes exist
    E = rng.uniform(0.1, 2.0, d.n_active)
    F1, F2 = np.zeros(d.ndof), np.zeros(d.ndof)
    free_used = np.setdiff1d((used[:, None] * d.dim + np.arange(d.dim)).ravel(), fixed)
    F1[free_used] = rng.normal(size=free_used.size)
    F2[free_used] = rng.normal(size=free_used.size)
    U1 = phys.solve_numpy(E, F1)[:, 0]
    energy = float((E * phys.geom * phys.element_energy(U1)[:, 0]).sum())
    assert float(F1 @ U1) == pytest.approx(energy, rel=1e-9)
    assert np.allclose(phys.solve_numpy(2 * E, F1)[:, 0], U1 / 2, rtol=0, atol=1e-10 * np.abs(U1).max())
    U12 = phys.solve_numpy(E, F1 + 3 * F2)[:, 0]
    U2 = phys.solve_numpy(E, F2)[:, 0]
    assert np.allclose(U12, U1 + 3 * U2, rtol=0, atol=1e-9 * np.abs(U12).max())
    assert np.all(U1[fixed] == 0.0)


# ---------------------------------------------------------------------------------------------- volume
@PROFILE
@given(domains())
def test_volume_matches_its_definition(dr):
    """Volume = mean of x over the active elements; weighted = sum(frac x) / sum(frac). Computed independently."""
    d, rng = dr
    x = _grid(d, rng)
    ctx = EvalContext(d, x, torch.zeros_like(x))
    assert float(Volume()(ctx)) == pytest.approx(float(x.numpy()[d.active].mean()), rel=1e-13)
    assert float(Volume(weighted=True)(ctx)) == pytest.approx(
        float((d.frac * x.numpy()).sum() / d.frac.sum()), rel=1e-13)


# ---------------------------------------------------------------------------------------------- connectivity
def _flood_labels(solid, diagonal=False):
    """Independent component count by BFS (face neighbours, or all 3^dim - 1 with diagonal)."""
    seen = np.zeros(solid.shape, bool)
    offs = [o for o in np.ndindex(*(3,) * solid.ndim) if any(v != 1 for v in o)]
    offs = [tuple(v - 1 for v in o) for o in offs]
    if not diagonal:
        offs = [o for o in offs if sum(abs(v) for v in o) == 1]
    sizes = []
    for start in zip(*np.nonzero(solid)):
        if seen[start]:
            continue
        q, n = deque([start]), 0
        seen[start] = True
        while q:
            p = q.popleft()
            n += 1
            for o in offs:
                nb = tuple(a + b for a, b in zip(p, o))
                if all(0 <= nb[i] < solid.shape[i] for i in range(solid.ndim)) and solid[nb] and not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        sizes.append(n)
    return sorted(sizes, reverse=True)


@PROFILE
@given(st.sampled_from([2, 3]), st.integers(0, 2**31 - 1), st.floats(0.2, 0.7))
def test_label_bodies_matches_an_independent_flood_fill(dim, seed, density):
    rng = np.random.default_rng(seed)
    shape = (9, 7) if dim == 2 else (5, 4, 4)
    solid = rng.random(shape) < density
    lab, sizes = label_bodies(solid, "face")
    assert sizes.tolist() == _flood_labels(solid)                      # same bodies, sorted by decreasing size
    assert np.all((lab > 0) == solid)
    for k, n in enumerate(sizes, start=1):
        assert int((lab == k).sum()) == n                              # label k is the k-th largest body
    _, sizes_v = label_bodies(solid, "vertex")
    assert sizes_v.tolist() == _flood_labels(solid, diagonal=True)


# ---------------------------------------------------------------------------------------------- STL round trip
@PROFILE
@given(seed=st.integers(0, 2**31 - 1), n_tri=st.integers(1, 40))
def test_stl_round_trip_keeps_every_triangle(tmp_path_factory, seed, n_tri):
    """write_stl / read_stl of geometry and post.export: same triangles (as float32) for random non-degenerate
    meshes, shared vertices merged once."""
    from toporlogy import geometry as G
    from toporlogy.post import export as E
    rng = np.random.default_rng(seed)
    V = rng.uniform(-5, 5, (3 * n_tri, 3))
    F = np.arange(3 * n_tri).reshape(-1, 3)
    F = np.vstack([F, F[:1, ::-1]]) if n_tri > 1 else F                 # one reversed duplicate: shared vertices
    want = np.sort(np.round(V[F].astype(np.float32).astype(float), 6).reshape(len(F), -1), axis=0)
    tmp = tmp_path_factory.mktemp("stl")
    for name, write, read in (("g", G.write_stl, G.read_stl), ("e", E.write_stl, E.read_stl)):
        p = tmp / f"{name}.stl"
        write(p, V, F)
        V2, F2 = read(p)
        got = np.sort(np.round(V2[F2], 6).reshape(len(F2), -1), axis=0)
        assert got.shape == want.shape and np.allclose(got, want, atol=1e-5), name
        assert len(V2) == 3 * n_tri                                     # duplicate triangle reuses its vertices

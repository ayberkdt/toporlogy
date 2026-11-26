"""Speed-ups of the optimisation loop: the checks pre-registered in docs/olcumler/hizlandirma_onkayit.md
(2026-11-26 03:00:20 +0300, before their first run).

H1 torch intra-op threads inside ``Study.run`` (restored afterwards); H2 SuperLU input built from a cached index map
(the same arrays as before, bit for bit); H3 the opt-in supernodal Cholesky (``DirectSolver("cholesky")``): same
solution as SuperLU to rounding, deterministic, fail-closed on a matrix that is not SPD. The speed targets (H1a, H1b,
H2b, H3a) are measurements, recorded in docs/olcumler/hizlandirma.md, not tests. A failing pre-registered check
stays here as a strict xfail with the measured value.
"""
from __future__ import annotations

import glob
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

import toporlogy as T
from toporlogy import boundary
from toporlogy.solvers.direct import DirectSolver, _PermutedPattern
from toporlogy.solvers.supernodal import NotPositiveDefiniteError, SupernodalCholesky

REPO = Path(__file__).resolve().parents[2]


def _cantilever(shape, frac=None, keep=None, void=None):
    d = T.Domain(shape, frac=frac, keep=keep, void=void)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    F = d.load_vector()
    tip = np.flatnonzero(np.isclose(X[:, 0], shape[0]) & np.isin(np.arange(d.nn), d.active_dofs // d.dim))
    F[d.dim * tip + d.dim - 1, 0] = -1.0 / len(tip)
    return d, fixed, F


def _design(d, seed, emin=1e-9):
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    f = ndimage.gaussian_filter(rng.standard_normal(d.shape), 1.5).ravel()[d.active_elements]
    x = 1 / (1 + np.exp(-4 * (f - f.mean()) / f.std()))
    return emin + x ** 3 * (1 - emin)


def _mbb_study(shape=(24, 8), **kw):
    d = T.Domain(shape)
    X = d.node_coords()
    fixed = np.concatenate([d.dofs(np.flatnonzero(X[:, 0] == 0), [0]),
                            d.dofs(np.flatnonzero((X[:, 0] == shape[0]) & (X[:, 1] == 0)), [1])])
    F = d.load_vector()
    F[d.dofs(np.flatnonzero((X[:, 0] == 0) & (X[:, 1] == shape[1])), [1])] = -1.0
    from toporlogy.design.chain import standard_chain
    from toporlogy.design.interp import SIMP
    from toporlogy.optimization.responses import Compliance, Volume
    from toporlogy.optimization.study import Constraint, Study
    from toporlogy.optimizers.oc import OC
    phys = T.Elasticity(d, fixed, 0.3, solver=kw.pop("solver", None))
    return Study(d, phys, F, standard_chain(d, 1.5, projection=False), SIMP(d, penal=3.0, emin_ratio=1e-9), Compliance(),
                 [Constraint(Volume(), 0.5)], OC(dv_scale=d.n_active), **kw)


# ------------------------------------------------------------------------------------------------------ H1
def test_h1c_threads_restored_after_a_run_and_after_an_error():
    n0 = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        st = _mbb_study()
        seen = []
        st.run(max_iter=2, tol=0.0, callback=lambda it, ev, ch: seen.append(torch.get_num_threads()))
        assert seen == [1, 1] and torch.get_num_threads() == 3

        def boom(it, ev, ch):
            raise RuntimeError("stop")
        with pytest.raises(RuntimeError, match="stop"):
            st.run(max_iter=2, tol=0.0, callback=boom)
        assert torch.get_num_threads() == 3
        st2 = _mbb_study(torch_threads=None)
        st2.run(max_iter=1, tol=0.0, callback=lambda it, ev, ch: seen.append(torch.get_num_threads()))
        assert seen[-1] == 3 and torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(n0)


def test_h1e_threads_none_is_the_unchanged_loop():
    """``torch_threads=None`` does not touch the setting, so it is the loop as before; at the current thread count
    it gives the same bits as an explicit ``torch_threads`` equal to it."""
    n0 = torch.get_num_threads()
    ra = _mbb_study(torch_threads=None).run(max_iter=4, tol=0.0)
    rb = _mbb_study(torch_threads=n0).run(max_iter=4, tol=0.0)
    assert ra.x.tobytes() == rb.x.tobytes() and ra.history["f0"] == rb.history["f0"]
    assert _mbb_study().settings()["torch_threads"] == 1


def test_h1_bad_thread_count_is_refused():
    with pytest.raises(ValueError, match="torch_threads"):
        _mbb_study(torch_threads=0)


# ------------------------------------------------------------------------------------------------------ H2
def _examples():
    return sorted(glob.glob(str(REPO / "examples" / "*.yaml"))) + sorted(glob.glob(str(REPO / "examples" /
                                                                                       "research" / "*.yaml")))


@pytest.mark.parametrize("path", _examples(), ids=lambda p: Path(p).stem)
def test_h2a_superlu_input_identical(path):
    """CSC arrays handed to SuperLU equal those of the earlier construction, bit for bit: uniform moduli (exact
    zeros from cancelling neighbours, which the sparse sum drops) and a random field."""
    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.resolve import read_config
    st = build_setup(read_config(path)).study
    ph = getattr(st.physics, "elastic", st.physics)
    order = ph.elimination_order()
    rng = np.random.default_rng(1)
    for E in (np.full(ph.n_elements, 0.5), 1e-9 + rng.random(ph.n_elements) ** 3):
        A = ph.matrix(E)
        old = (A + sp.triu(A, k=1).T).tocsr()[order][:, order].tocsc()
        new = _PermutedPattern(A, order).matrix(A.data)
        assert np.array_equal(old.indptr, new.indptr) and np.array_equal(old.indices, new.indices)
        assert old.data.tobytes() == new.data.tobytes()


def test_h2_map_is_rebuilt_when_the_pattern_or_order_changes():
    d, fixed, F = _cantilever((6, 3, 3))
    m = T.Elasticity(d, fixed, 0.3, solver=DirectSolver("splu"))
    m.solve_numpy(_design(d, 1), F)
    pm = m.solver._perm_map
    m.solve_numpy(_design(d, 2), F)
    assert m.solver._perm_map is pm                        # same pattern and order: reused
    d2, fixed2, F2 = _cantilever((7, 3, 3))
    m2 = T.Elasticity(d2, fixed2, 0.3, solver=m.solver)
    m2.solve_numpy(_design(d2, 3), F2)
    assert m.solver._perm_map is not pm
    ref = T.Elasticity(d2, fixed2, 0.3, solver=DirectSolver("splu")).solve_numpy(_design(d2, 3), F2)
    assert np.array_equal(m2.solve_numpy(_design(d2, 3), F2), ref)


# ------------------------------------------------------------------------------------------------------ H3
def _pair(make_ref, make_ch, E, F):
    a, b = make_ref(), make_ch()
    Ua, Ub = a.solve_numpy(E, F), b.solve_numpy(E, F)
    ca, cb = float(np.sum(F * Ua)), float(np.sum(F * Ub))
    return abs(ca - cb) / abs(ca), np.linalg.norm(Ua - Ub) / np.linalg.norm(Ua)


@pytest.mark.parametrize("shape", [(16, 8, 8), (120, 40)])
def test_h3b_same_solution_as_superlu(shape):
    d, fixed, F = _cantilever(shape)
    E = _design(d, 5)
    dc, du = _pair(lambda: T.Elasticity(d, fixed, 0.3, solver=DirectSolver("splu")),
                   lambda: T.Elasticity(d, fixed, 0.3, solver=DirectSolver("cholesky")), E, F)
    print(f"\n  H3b {shape}: compliance {dc:.1e}, U {du:.1e}")
    assert dc <= 1e-10 and du <= 1e-7


@pytest.mark.parametrize("shape,element", [((8, 5, 5), "H8IM"), ((30, 14), "Q4")])
def test_h3c_partial_volumes_keep_void(shape, element):
    rng = np.random.default_rng(6)
    frac = np.clip(rng.uniform(0.2, 1.6, shape), 0, 1)
    sl = (slice(int(0.75 * shape[0]), None), slice(0, 2)) + ((slice(None),) if len(shape) == 3 else ())
    frac[sl] = 0.0                                          # a notch: elements outside the part
    keep = np.zeros(shape, bool)
    keep[(slice(None), shape[1] - 1)] = True
    void = np.zeros(shape, bool)
    void[tuple(s // 2 for s in shape)] = True
    d, fixed, F = _cantilever(shape, frac=frac, keep=keep, void=void)
    E = _design(d, 7)
    dc, du = _pair(lambda: T.Elasticity(d, fixed, 0.3, element=element, solver=DirectSolver("splu")),
                   lambda: T.Elasticity(d, fixed, 0.3, element=element, solver=DirectSolver("cholesky")), E, F)
    print(f"\n  H3c {shape} {element}: compliance {dc:.1e}, U {du:.1e}")
    assert dc <= 1e-10 and du <= 1e-7


@pytest.mark.xfail(strict=True, raises=NotImplementedError,
                   reason="H3c oblique support: ConstrainedElasticity (boundary/constrained.py, not this session's "
                          "file) changes the elimination order and gives no elimination_blocks(); the cholesky "
                          "solver refuses it (measured 26 Nov 2026: NotImplementedError)")
def test_h3c_oblique_support():
    shape = (8, 5, 5)
    d = T.Domain(shape)
    d1 = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    bottom = boundary.OnPlane([0, 0, 0], 2)
    side = boundary.Roller(boundary.OnPlane([0, 0, 0], 0), 0)
    load = [boundary.LoadCase("t", [boundary.SurfaceForce(boundary.OnPlane([0, 0, 5], 2), [1.0, 0.5, -1.0])])]
    B = boundary.BoundaryConditions(d, [boundary.Fixed(bottom, [d1, [0, 0, 1]]), side], load)
    ref, ch = B.physics(), B.physics()
    ch.solver = DirectSolver("cholesky")
    E = _design(d, 7)
    Ua, Ub = ref.solve_numpy(E, B.F()), ch.solve_numpy(E, B.F())
    assert np.linalg.norm(Ua - Ub) / np.linalg.norm(Ua) <= 1e-7


def test_h3d_same_compliance_gradient():
    d, fixed, F = _cantilever((14, 6, 6))
    E0 = torch.tensor(_design(d, 8), dtype=torch.float64)
    grads = []
    for method in ("splu", "cholesky"):
        m = T.Elasticity(d, fixed, 0.3, solver=DirectSolver(method))
        E = E0.clone().requires_grad_(True)
        U = m.solve(E, F)
        (torch.as_tensor(F) * U).sum().backward()
        assert E.grad is not None
        grads.append(E.grad.numpy().copy())
    rel = np.max(np.abs(grads[0] - grads[1])) / np.max(np.abs(grads[0]))
    print(f"\n  H3d: gradient {rel:.1e}")
    assert rel <= 1e-9


_DET = r"""
import sys, hashlib, numpy as np, torch
sys.path.insert(0, sys.argv[1] + "/src"); sys.path.insert(0, sys.argv[1] + "/tests/toporlogy")
from test_speedups import _cantilever, _design
import toporlogy as T
from toporlogy.solvers.direct import DirectSolver
d, fixed, F = _cantilever((12, 6, 6))
m = T.Elasticity(d, fixed, 0.3, solver=DirectSolver("cholesky"))
print(hashlib.sha256(m.solve_numpy(_design(d, 9), F).tobytes()).hexdigest())
"""


def test_h3e_deterministic_in_one_process_and_across_processes():
    d, fixed, F = _cantilever((12, 6, 6))
    m = T.Elasticity(d, fixed, 0.3, solver=DirectSolver("cholesky"))
    E = _design(d, 9)
    U1 = m.solve_numpy(E, F)
    m.solve_numpy(_design(d, 10), F)                       # another factorisation in between
    U2 = m.solve_numpy(E, F)
    assert U1.tobytes() == U2.tobytes()
    import hashlib
    here = hashlib.sha256(U1.tobytes()).hexdigest()
    outs = [subprocess.run([sys.executable, "-c", _DET, str(REPO)], capture_output=True, text=True, check=True)
            .stdout.strip() for _ in range(2)]
    # the child processes run with torch's default threads (no Study): the kernels must not depend on them here
    assert outs[0] == outs[1]
    print(f"\n  H3e: in-process {here[:12]}, processes {outs[0][:12]} {outs[1][:12]}")


def test_h3f_not_spd_and_singular_are_refused():
    n = 30
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    order, blocks = rng.permutation(n), np.array([7, 0, 13, 10])
    ch = SupernodalCholesky(order, blocks)
    ch.factor(sp.triu(sp.csr_matrix(A)).tocsr())
    x = ch.solve(np.ones(n))
    assert np.linalg.norm(A @ x - 1) <= 1e-12 * np.linalg.norm(A) * np.linalg.norm(x)
    B = A.copy()
    B[4, 4] = -1.0
    with pytest.raises(NotPositiveDefiniteError):
        ch.factor(sp.triu(sp.csr_matrix(B)).tocsr())
    S = np.zeros((n, n))
    S[:-1, :-1] = A[:-1, :-1]
    S[-1, -1] = 0.0
    S[-1, 0] = S[0, -1] = 0.0
    Su = sp.triu(sp.csr_matrix(A)).tocsr()                 # pattern of A, values of the singular S
    Su.data[:] = S[Su.nonzero()]
    with pytest.raises(NotPositiveDefiniteError):
        ch.factor(Su)
    with pytest.raises(ValueError, match="blocks"):
        DirectSolver("cholesky").factor(sp.triu(sp.csr_matrix(A)).tocsr())


@pytest.mark.parametrize("shape", [(16, 8, 8), (120, 40)])
def test_h3g_fewer_stored_factor_entries_than_superlu(shape):
    d, fixed, _ = _cantilever(shape)
    m = T.Elasticity(d, fixed, 0.3, solver=DirectSolver("cholesky"))
    A = m.matrix(_design(d, 4))
    order, blocks = m.elimination_order(), m.elimination_blocks()
    K = (A + sp.triu(A, k=1).T).tocsc()
    lu = spla.splu(K[order][:, order].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                   options={"SymmetricMode": True})
    ch = SupernodalCholesky(order, blocks)
    ch.factor(A)
    print(f"\n  H3g {shape}: Cholesky {ch.factor_nnz}, SuperLU L+U {lu.L.nnz + lu.U.nnz}")
    assert ch.factor_nnz < lu.L.nnz + lu.U.nnz


@pytest.mark.parametrize("which", ["mbb2d", "cantilever3d"])
def test_h3h_study_history_close_to_superlu(which):
    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.resolve import read_config
    hist = []
    for method in ("splu", "cholesky"):
        st = build_setup(read_config(str(REPO / "examples" / f"{which}.yaml")),
                         ["physics.type=elasticity", f"physics.solver={{type: direct, method: {method}}}"]).study
        assert st.physics.solver.method == method
        hist.append(np.array(st.run(max_iter=10, tol=0.0).history["f0"]))
    rel = float(np.max(np.abs(hist[0] - hist[1]) / np.abs(hist[0])))
    print(f"\n  H3h {which}: largest relative objective difference over 10 iterations {rel:.1e}")
    assert rel <= 1e-8


def test_cholesky_solver_refuses_a_physics_without_blocks():
    from toporlogy.fem.modal import Modal
    d = T.Domain((6, 3), h=1.0)
    X = d.node_coords()
    mo = Modal(d, d.dofs(np.flatnonzero(X[:, 0] < 1e-9)), nu=0.3, n_modes=2, solver=DirectSolver("cholesky"))
    with pytest.raises(ValueError, match="cholesky"):
        mo.solve_numpy(np.ones(d.n_active), np.ones(d.n_active), 2)

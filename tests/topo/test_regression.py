"""A1 acceptance (docs/mimari_plan.md §5, A1 (1) and (2)): the new core reproduces the old drivers.

Oracles (unchanged modules):

* :func:`topo_oracles.topo3d.optimize` — 3-D SIMP + density filter + tanh projection + OC on a
  :class:`~topo_oracles.voxfem.VoxelMesh`;
* :func:`topo_oracles.topology.top88` — 2-D half MBB beam, density filter (ft = 2), OC.

Pre-registered threshold: compliance history relative difference ≤ 1e-6 (same algorithm, same start). The final
physical density is compared too (max |Δx̄| ≤ 1e-6).

Mapping rules for the 3-D oracle: VoxelMesh numbers only the solid voxels in ``np.argwhere`` order (= C-order of the
active elements of :class:`Domain`) and compacts the nodes; ``VoxelMesh.grid_node[k]`` is the box node number of
compact node k (same formula as :meth:`Domain.node_number`). Loads and supports are written once in box numbering
and translated to the compact numbering for the oracle.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch  # (tests/topo/conftest.py has loaded aerokit.topo, i.e. PARDISO, first)

from topo_oracles import topo3d
from topo_oracles.voxfem import VoxelMesh
from topo_oracles import topology
from aerokit.topo import (OC, SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, Study, Volume,
                          standard_chain)

RTOL = 1e-6          # pre-registered (A1 acceptance)
XTOL = 1e-6


# ---------------------------------------------------------------------------------------------- helpers
def _cantilever_bc(d: Domain, two_cases: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Box-numbered loads (ndof, nc) and fixed DOFs: x = 0 face clamped; tip line load at x = L, z = 0."""
    L, eps, o = d.shape[0] * d.h, 1e-6 * d.h, d.origin
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] - o[0] < eps))
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - o[0] - L) < eps) & (X[:, 2] - o[2] < eps))
    F = d.load_vector(2 if two_cases else 1)
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    if two_cases:
        F[d.dofs(tip, 1), 1] = 0.5 / len(tip)
    return F, fixed


def _to_voxel(d: Domain, mesh: VoxelMesh, F: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate box DOF numbers to the compact VoxelMesh numbering (every loaded/fixed node must be in the mesh)."""
    gn = mesh.grid_node

    def compact(dofs: np.ndarray) -> np.ndarray:
        nodes, dirs = dofs // 3, dofs % 3
        k = np.searchsorted(gn, nodes)
        assert np.all(gn[np.minimum(k, len(gn) - 1)] == nodes), "load/support on a node outside the part"
        return 3 * k + dirs

    loaded = np.flatnonzero(np.any(F != 0, axis=1))
    Fv = np.zeros((mesh.ndof, F.shape[1]))
    Fv[compact(loaded)] = F[loaded]
    return Fv, compact(np.asarray(fixed, np.int64))


def _run_pair(d: Domain, F: np.ndarray, fixed: np.ndarray, *, volfrac: float = 0.3, rmin: float = 1.5,
              projection: bool = False, beta_every: int = 10, element: str = "H8", weights=None,
              max_iter: int = 25, tol: float = 0.01) -> tuple[dict, object, object]:
    """Run topo3d.optimize and the new Study with identical settings; return metrics and both results."""
    mesh = VoxelMesh(d.active, d.h, frac=d.frac[d.active])
    Fv, fixv = _to_voxel(d, mesh, F, fixed)
    keep_v, void_v = d.keep[d.active], d.void[d.active]
    old = topo3d.optimize(mesh, Fv, fixv, volfrac, rmin=rmin, projection=projection, beta_every=beta_every,
                          max_iter=max_iter, tol=tol, element=element, weights=weights,
                          keep=keep_v if keep_v.any() else None, void=void_v if void_v.any() else None)

    phys = Elasticity(d, fixed, nu=0.3, element=element)
    chain = standard_chain(d, rmin, projection=projection)
    sched = BetaContinuation(chain, 1.0, 16.0, beta_every, tol) if projection else None
    study = Study(d, phys, F, chain, SIMP(d, emin_ratio=1e-6), Compliance(weights),
                  [Constraint(Volume(), volfrac, scale=1.0)], OC(dv_scale=d.n_active), sched)
    new = study.run(max_iter=max_iter, tol=tol)

    c_old, c_new = np.asarray(old.compliance), np.asarray(new.history["f0"])
    assert len(c_old) == len(c_new), f"iteration counts differ: {len(c_old)} vs {len(c_new)}"
    m = {"iterations": len(c_new),
         "c_rel": float(np.max(np.abs(c_new - c_old) / np.abs(c_old))),
         "x_abs": float(np.max(np.abs(new.xphys[d.active] - old.xphys))),
         "c_final": float(c_new[-1])}
    if projection:
        m["beta_equal"] = list(old.beta) == list(new.history["beta"])
    print(f"\n  regression metrics: {m}")
    return m, old, new


def _check(m: dict) -> None:
    assert m["c_rel"] <= RTOL, m
    assert m["x_abs"] <= XTOL, m
    assert m.get("beta_equal", True), m


# ---------------------------------------------------------------------------------------------- 3-D vs topo3d
SHAPE = (24, 8, 4)


def test_topo3d_box_no_projection():
    d = Domain(SHAPE, 1.0)
    F, fixed = _cantilever_bc(d)
    m, _, _ = _run_pair(d, F, fixed, projection=False, max_iter=30)
    _check(m)


def test_topo3d_box_projection_beta_continuation():
    d = Domain(SHAPE, 1.0)
    F, fixed = _cantilever_bc(d)
    m, old, _ = _run_pair(d, F, fixed, projection=True, beta_every=10, max_iter=30)
    assert max(old.beta) > 1.0, "β continuation never triggered"
    _check(m)


def test_topo3d_h8im():
    d = Domain((16, 6, 3), 1.0)
    F, fixed = _cantilever_bc(d)
    m, _, _ = _run_pair(d, F, fixed, element="H8IM", max_iter=20)
    _check(m)


def test_topo3d_two_load_cases_weighted():
    d = Domain((16, 6, 4), 1.0)
    F, fixed = _cantilever_bc(d, two_cases=True)
    # weights 1/c0 of each case on the uniform start design (as a user would normalise)
    phys = Elasticity(d, fixed)
    E = SIMP(d, emin_ratio=1e-6)(torch.full(d.shape, 0.3, dtype=torch.float64))
    U = phys.solve_numpy(E.numpy().ravel()[d.active_elements], F)
    c0 = (F * U).sum(axis=0)
    w = 1.0 / c0
    m, _, _ = _run_pair(d, F, fixed, weights=w, max_iter=20)
    _check(m)


def test_topo3d_keep_and_void():
    d0 = Domain(SHAPE, 1.0)
    keep = np.zeros(SHAPE, bool)
    keep[-2:, :, :2] = True                 # solid pad under the tip load
    void = np.zeros(SHAPE, bool)
    void[10:14, 2:6, :] = True              # forced hole
    d = Domain(SHAPE, 1.0, keep=keep, void=void)
    F, fixed = _cantilever_bc(d0)
    m, old, new = _run_pair(d, F, fixed, projection=True, beta_every=8, max_iter=25)
    assert np.all(new.xphys[keep] == 1.0) and np.all(new.xphys[void] == 0.0)
    _check(m)


def _l_shape_with_hole() -> np.ndarray:
    mask = np.ones(SHAPE, bool)
    mask[14:, 5:, :] = False                # L shape (upper right corner removed)
    mask[6:10, 2:5, 1:3] = False            # interior cavity
    mask[17:20, 1:3, :] = False             # through hole
    return mask


def test_topo3d_non_box_part():
    mask = _l_shape_with_hole()
    d = Domain(SHAPE, 1.0, frac=mask.astype(float))
    assert d.n_active == mask.sum() and len(d.active_dofs) < d.ndof
    F, fixed = _cantilever_bc(d)
    m, _, _ = _run_pair(d, F, fixed, projection=True, beta_every=10, max_iter=25)
    _check(m)


def test_topo3d_non_box_partial_frac_and_h():
    mask = _l_shape_with_hole()
    rng = np.random.default_rng(3)
    frac = np.where(mask, 0.5 + 0.5 * rng.random(SHAPE), 0.0)
    d = Domain(SHAPE, 0.004, origin=(0.1, -0.2, 0.0), frac=frac)    # non-unit h exercises the K ∝ E·h scaling
    F, fixed = _cantilever_bc(d)
    m, _, _ = _run_pair(d, F, fixed, projection=False, max_iter=25)
    _check(m)


# ---------------------------------------------------------------------------------------------- 2-D vs top88
def test_top88_half_mbb():
    """top88(nelx, nely, 0.5, 3, rmin, ft=2): half MBB, density filter, OC (bisection tol 1e-3, move 0.2).

    top88's OC bisects λ ∈ [0, 1e9] with ``dv = ∂Σx̄/∂x`` and checks ``Σ x̄(x_new) > V*·nel``; the new core's
    constraint is mean(x̄) − V* ≤ 0, so ``dv_scale = nel`` gives the same λ sequence. Grid2D numbers rows from the
    top (y down); loads and supports are placed by coordinates, element fields are flipped for the comparison.
    """
    nelx, nely, vf, rmin, iters = 60, 20, 0.5, 2.0, 40
    old = topology.top88(nelx, nely, vf, 3.0, rmin, 2, max_iter=iters)

    d = Domain((nelx, nely))
    load = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == nely))
    F = d.load_vector()
    F[d.dofs(load, 1), 0] = -1.0
    fixed = np.concatenate([d.dofs(d.nodes_where(lambda X: X[:, 0] == 0), 0),
                            d.dofs(d.nodes_where(lambda X: (X[:, 0] == nelx) & (X[:, 1] == 0)), 1)])
    phys = Elasticity(d, fixed, nu=0.3, element="Q4")
    chain = standard_chain(d, rmin, projection=False)
    study = Study(d, phys, F, chain, SIMP(d, penal=3.0, emin_ratio=1e-9), Compliance(),
                  [Constraint(Volume(), vf, scale=1.0)],
                  OC(move=0.2, l1=0.0, l2=1e9, bisection_tol=1e-3, dv_scale=d.n_active))
    new = study.run(max_iter=iters, tol=0.01)

    c_old, c_new = np.asarray(old["compliance"]), np.asarray(new.history["f0"])
    assert len(c_old) == len(c_new)
    c_rel = float(np.max(np.abs(c_new - c_old) / c_old))
    xnew_old_layout = new.xphys.T[::-1]                     # (nely, nelx), row 0 at the top
    x_abs = float(np.max(np.abs(xnew_old_layout - old["xfiz"])))
    print(f"\n  top88 metrics: iterations={len(c_new)} c_rel={c_rel:.3e} x_abs={x_abs:.3e} c_final={c_new[-1]:.6f}")
    assert c_rel <= RTOL
    assert x_abs <= XTOL


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q", "-s"])

"""O11 (PDE filter) and O12 (GMG coarse grid): factorisation time, MMD vs nested dissection, same matrix."""
import json, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import toporlogy  # noqa: F401
from toporlogy.core.domain import Domain
from toporlogy.design.pde_filter import helmholtz_matrix
from toporlogy.fem.elastic import Elasticity
from toporlogy.solvers.ordering import nested_dissection_dofs

def best(f, n=3):
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); t.append(time.perf_counter() - t0)
    return min(t), r

out = {}
# O11: Helmholtz matrix on the element grid (one unknown per element)
shape = (60, 30, 30)
A, el = helmholtz_matrix(np.ones(shape, bool), 1.5 / (2 * np.sqrt(3)))
b = np.random.default_rng(0).random(A.shape[0])
t_mmd, lu = best(lambda: spla.splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}))
order = nested_dissection_dofs(shape, 1, np.arange(A.shape[0]))
Ap = A[order][:, order].tocsc()
t_nd, lu2 = best(lambda: spla.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True}))
x1 = lu.solve(b); x2 = np.empty_like(b); x2[order] = lu2.solve(b[order])
out["O11_pde_filter_60x30x30"] = {"n": int(A.shape[0]), "mmd_s": t_mmd, "nd_s": t_nd, "speedup": t_mmd / t_nd,
                                  "rel_diff": float(np.abs(x1 - x2).max() / np.abs(x1).max())}
# O12: coarse grid of 10 x 10 x 10 elements = 3993 DOF (max_coarse_dof = 4000), one face clamped
d = Domain((10, 10, 10))
X = d.node_coords()
m = Elasticity(d, d.dofs(np.flatnonzero(X[:, 0] < 1e-9)), ordering="solver")
Au = m.matrix(np.ones(d.n_active)); full = (Au + sp.triu(Au, 1).T).tocsc()
t_mmd, lu = best(lambda: spla.splu(full, permc_spec="MMD_AT_PLUS_A"), 5)
o = m.elimination_order(); Fp = full[o][:, o].tocsc()
t_nd, _ = best(lambda: spla.splu(Fp, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True}), 5)
out["O12_gmg_coarse_10x10x10"] = {"n": int(full.shape[0]), "mmd_s": t_mmd, "nd_s": t_nd, "speedup": t_mmd / t_nd}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))

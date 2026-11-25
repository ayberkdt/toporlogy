"""O8/O9 timing and values: run on the code before and after the change; prints JSON."""
import json, sys, time
import numpy as np
import toporlogy  # noqa: F401  (PARDISO before torch)
from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.buckling import LinearBuckling
from toporlogy.fem.modal import Modal

out = {}
def buck(shape):
    d = Domain(shape, h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 1] < 1e-9))
    top = np.flatnonzero(X[:, 1] > X[:, 1].max() - 1e-9)
    F = np.zeros(d.ndof); F[d.dim * top + 1] = -1.0 / len(top)
    F0 = np.zeros(d.ndof); F0[d.dim * top] = 0.1 / len(top)
    rng = np.random.default_rng(0); E = 0.1 + 0.9 * rng.random(d.n_active)
    m = Elasticity(d, fixed, nu=0.3); U = m.solve_numpy(E, np.column_stack([F, F0]))
    lb = LinearBuckling(m); KG0 = lb.geometric_matrix(E, U[:, 1])
    t = time.perf_counter(); mu, _ = lb.modes(E, E, U[:, 0], k=6, KG0=KG0); t = time.perf_counter() - t
    return {"dof": int(len(m.free)), "seconds": t, "mu": mu.tolist()}

def modal(shape, sigma):
    d = Domain(shape, h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    mo = Modal(d, fixed, nu=0.3, n_modes=6, sigma=sigma)
    rng = np.random.default_rng(1); E = 0.1 + 0.9 * rng.random(d.n_active)
    t = time.perf_counter(); lam, _ = mo.solve_numpy(E, np.ones(d.n_active), 6); t = time.perf_counter() - t
    return {"dof": int(mo.n_free), "seconds": t, "lam": np.asarray(lam).tolist()}

out["buckling_preload_2d_400x100"] = buck((400, 100))
out["buckling_preload_3d_40x10x10"] = buck((40, 10, 10))
out["modal_3d_40x20x20_sigma0"] = modal((40, 20, 20), 0.0)
out["modal_3d_24x12x12_sigma_neg"] = modal((24, 12, 12), -1e-4)
out["modal_3d_24x12x12_sigma_pos"] = modal((24, 12, 12), 1e-4)
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out.items(): print(k, v["dof"], f"{v['seconds']:.2f}s")

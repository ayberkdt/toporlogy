"""Small CPU benchmark set (WP01 A03, WP05 L07): median and range over repeats, after one warm-up, single process.

    python docs/foundations/scripts/benchmarks.py          -> docs/foundations/benchmarks_<sha>.json

Measurements only: no speed-up is claimed and no threshold is applied. Thread counts and versions are recorded; a
loaded machine (other sessions) inflates the numbers, which is why the range is kept.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests" / "toporlogy")]

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from toporlogy.core import spectral, tensor  # noqa: E402
from toporlogy.core.constitutive import J2Plasticity, MaterialRequest, TimeStep  # noqa: E402
from toporlogy.core.creep import NortonCreep  # noqa: E402
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries  # noqa: E402
from toporlogy.fem.mesh import ElasticProblem  # noqa: E402
from toporlogy.fem.mesh.linear import Factorization  # noqa: E402
from toporlogy.fem.mesh.mesh import rectangle  # noqa: E402


def timeit(fn, repeats=5):
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)), "min_s": float(min(ts)), "max_s": float(max(ts)), "repeats": repeats}


rng = np.random.default_rng(0)
out: dict = {"sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
             "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
             "torch": torch.__version__, "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
             "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "cases": {}}
C = out["cases"]
for nb in (1, 1000, 100_000):
    A = rng.normal(size=(nb, 3, 3))
    A = A @ np.swapaxes(A, 1, 2) + 0.5 * np.eye(3)
    H = rng.normal(size=(nb, 3, 3))
    H = 0.5 * (H + np.swapaxes(H, 1, 2))
    C[f"spectral_log_frechet_batch{nb}"] = timeit(lambda A=A, H=H: spectral.spectral_frechet(A, H, "log"))
    C[f"stress_to_voigt_batch{nb}"] = timeit(lambda A=A: tensor.stress_to_voigt(A))
for n in (2_000, 20_000):
    R = sp.random(n, n, density=5.0 / n, random_state=rng)
    K = sp.csr_matrix(R @ R.T + 10 * sp.eye(n))
    B = rng.normal(size=(n, 8))
    C[f"spd_factor_solve_n{n}_rhs8"] = timeit(lambda K=K, B=B: Factorization(K, "spd").solve(B), repeats=3)
for nx in (40, 120):
    m = rectangle(nx, nx // 2, 2.0, 1.0)
    from toporlogy.core.linear_elastic import isotropic_plane_strain
    prob = ElasticProblem(m, isotropic_plane_strain(1.0, 0.3))
    P = m.points
    left = np.flatnonzero(np.isclose(P[:, 0], 0.0))
    from toporlogy.fem.mesh.problem import Dirichlet
    bc = Dirichlet.fixed(np.concatenate([2 * left, 2 * left + 1]))
    f = np.zeros(prob.ndof)
    f[-1] = -1.0
    C[f"general_mesh_Q4_{nx}x{nx // 2}_assemble_solve"] = timeit(
        lambda prob=prob, f=f, bc=bc: (prob._invalidate(), prob.solve(f, bc)), repeats=3)
npts = 10_000
eps = 1e-3 * rng.normal(size=(npts, 3, 3))
eps = 0.5 * (eps + np.swapaxes(eps, 1, 2))
req = MaterialRequest(step=TimeStep(0.0, 1.0))
j2 = J2Plasticity(0.3, 1e-3, 0.05)
C["material_point_legacy_J2_10k"] = timeit(lambda: j2.update(eps, j2.initial_state(npts), req))
gm = GeneralizedMaxwell(PronySeries(1e6, [2e6, 3e6], [1.0, 100.0]), PronySeries(5e6, [4e6], [10.0]))
C["material_point_generalized_maxwell_10k"] = timeit(lambda: gm.update(eps, gm.initial_state(npts), req))
nc = NortonCreep(3.33e9, 0.77e9, 1e-6, 1e7, 3.0)
C["material_point_norton_10k"] = timeit(lambda: nc.update(eps * 10, nc.initial_state(npts), req))
import test_viscoelastic_topology as T  # noqa: E402
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure  # noqa: E402
mc, fixed, tip = T._cantilever()
st = ViscoelasticStructure(mc, T._general(), T.TIMES, T._schedule(mc.n_nodes * 2, tip, T.TIMES, 2.0), fixed)
xx = 0.3 + 0.6 * rng.random(mc.n_cells)


def vo_grad(k=None):
    xt = torch.as_tensor(xx).requires_grad_(True)
    st.response(T._weights(xt), checkpoint_every=k).compliance[-1].backward()


C["viscoelastic_adjoint_gradient_8x4_12steps"] = timeit(vo_grad)
C["viscoelastic_adjoint_gradient_8x4_12steps_checkpoint4"] = timeit(lambda: vo_grad(4))
dst = ROOT / "docs" / "foundations" / f"benchmarks_{out['sha']}.json"
dst.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
for k, v in C.items():
    print(f"{k:58s} median {v['median_s'] * 1e3:10.3f} ms  [{v['min_s'] * 1e3:.3f}, {v['max_s'] * 1e3:.3f}]")

"""Measurements for docs/olcumler/yontemler.md (run from the repository root)."""
import json, time, warnings, platform
import numpy as np, torch, scipy, yaml
warnings.filterwarnings("ignore")
import toporlogy as T
from toporlogy.configuration.build import build_setup
from toporlogy.optimization.benchmark import compare_runs, format_table, binarize
from toporlogy.optimizers import GCMMA, SiMPL
from toporlogy.design.pde_filter import HelmholtzFilter
from toporlogy.optimization.stress import VonMises
from toporlogy.optimization.buckling import BucklingKS
out = {"environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                       "torch": torch.__version__, "machine": platform.machine(), "system": platform.system()}}

def mbb(opt, nx=90, ny=30, rmin=2.4, pde=False):
    d = T.Domain((nx, ny))
    left = d.nodes_where(lambda X: X[:, 0] == 0); roller = d.nodes_where(lambda X: (X[:, 0] == nx) & (X[:, 1] == 0))
    tl = d.nodes_where(lambda X: (X[:, 0] == 0) & (X[:, 1] == ny))
    F = d.load_vector()[:, 0]; F[d.dofs(tl, 1)] = -1.0
    chain = T.Chain([T.Embed(d), HelmholtzFilter(d, rmin) if pde else T.DensityFilter(d, rmin), T.Passive(d)])
    return T.Study(d, T.Elasticity(d, np.concatenate([d.dofs(left, 0), d.dofs(roller, 1)])), F, chain,
                   T.SIMP(d, emin_ratio=1e-9), T.Compliance(), [T.Constraint(T.Volume(), 0.5)], opt)

runs, rows = {}, []
for name, mk, pde in (("OC", lambda: T.OC(), False), ("MMA", lambda: T.MMA(), False), ("GCMMA", lambda: GCMMA(), False),
                      ("GCMMA move 0.2", lambda: GCMMA(move=0.2), False),
                      ("SiMPL armijo", lambda: SiMPL(), False), ("SiMPL bregman", lambda: SiMPL("bregman"), False),
                      ("SiMPL no search", lambda: SiMPL("none"), False),
                      ("OC + Helmholtz", lambda: T.OC(), True)):
    st = mbb(mk(), pde=pde); t = time.perf_counter(); res = st.run(max_iter=200, tol=1e-3)
    runs[name] = (st, res); sec = time.perf_counter() - t
    rows.append({"name": name, "seconds": sec, "termination": res.termination, "kkt": res.kkt_residual})
table = compare_runs(runs)
for r, x in zip(table, rows): r.update(x)
print(format_table(table))
for r in table: print(f"  {r['name']:<18} it {r['iterations']:3d} {r['termination']:<14} {r['seconds']:.1f}s kkt {r['kkt']}")
out["mbb_90x30"] = [{k: v for k, v in r.items() if k not in ("constraints", "not_evaluated")} for r in table]

# buckling: Euler columns
from toporlogy.fem.buckling import LinearBuckling
eul = []
for nb in (2, 4, 8, 16):
    d = T.Domain((nb, 40 * nb), 4.0 / nb); X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 1] == 0)); top = np.flatnonzero(np.isclose(X[:, 1], 160.0))
    F = d.load_vector(); w = np.ones(len(top)); w[[0, -1]] = .5; F[2 * top + 1, 0] = -w / w.sum()
    m = T.Elasticity(d, fixed, 0.0); E = np.full(d.n_active, 1000.0); t = time.perf_counter()
    U = m.solve_numpy(E, F); mu, _ = LinearBuckling(m).modes(E, E, U[:, 0], 4)
    eul.append({"elements_across": nb, "dof": int(d.ndof), "ratio": float(1 / mu[0] / (np.pi ** 2 * 1000 * 64 / 12 / (4 * 160 ** 2))),
                "mode_ratios": [float(mu[0] / mu[1]), float(mu[0] / mu[2])], "seconds": time.perf_counter() - t})
    print(eul[-1])
out["euler_column_2d"] = eul

# stress: two strategies on the L-bracket
st_rows = []
for f in ("stress_lbracket2d", "stress_al_lbracket2d"):
    raw = yaml.safe_load(open(f"examples/research/{f}.yaml")); st = build_setup(raw).study; t = time.perf_counter()
    res = st.run(max_iter=raw["run"]["max_iter"], tol=raw["run"]["tol"]); d = st.domain
    ex = np.zeros(d.shape, bool); ex[34:40, 12:16] = True
    vm = VonMises(d, 2.5, 0.3, 1.0, q=0.0, exclude=ex)
    def solid(x):
        xt = torch.as_tensor(x); E = st.interp(xt); U = st.physics.solve(E.reshape(-1)[st._active], st.F)
        r = vm(T.EvalContext(d, xt, E, U, st.F, interp=st.interp)).numpy()[:, 0]; m = x.ravel()[vm.elements] > 0.5
        return float(r[m].max()), float(x[d.active].mean())
    g, b = solid(res.xphys), solid(binarize(res.xphys, d, None, 0.5)); s = np.array(res.history["stress"])
    st_rows.append({"example": f, "iterations": res.iterations, "evaluations": st.n_evaluations, "seconds": time.perf_counter() - t,
                    "volume_grey": g[1], "max_solid_ratio_grey": g[0], "relaxed_max_value": res.constraint_values["stress"],
                    "relaxed_max_last20": [float(s[-20:].min()), float(s[-20:].max())], "volume_binary": b[1],
                    "max_ratio_binary": b[0], "m_nd": float(np.mean(4 * res.xphys[d.design] * (1 - res.xphys[d.design])))})
    print(st_rows[-1])
out["stress_lbracket"] = st_rows

# buckling-constrained design
raw = yaml.safe_load(open("examples/research/buckling_column2d.yaml"))
bk = []
for with_b in (False, True):
    r = yaml.safe_load(yaml.safe_dump(raw))
    if not with_b: r["constraints"] = r["constraints"][:1]
    st = build_setup(r).study; res = st.run(max_iter=r["run"]["max_iter"], tol=r["run"]["tol"])
    mon = BucklingKS(6, 30.0); mon(st.context(torch.as_tensor(res.x), True))
    bk.append({"buckling_constraint": with_b, "compliance": res.objective, "lambda": [float(v) for v in mon.last_load_factors[:3]],
               "iterations": res.iterations, "feasible": res.feasible}); print(bk[-1])
out["buckling_column"] = bk
json.dump(out, open("docs/olcumler/yontemler.json", "w"), indent=1)

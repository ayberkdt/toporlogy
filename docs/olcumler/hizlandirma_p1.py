import sys, time, json, torch, numpy as np
torch.set_num_threads(1)
sys.path.insert(0, "src")
import toporlogy as T
from toporlogy.solvers.supernodal import SupernodalCholesky
from toporlogy.solvers.direct import DirectSolver
out = []
for shape in [tuple(int(v) for v in a.split("x")) for a in sys.argv[2:]]:
    d = T.Domain(shape); X = d.node_coords()
    m = T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 0] == 0)), 0.3)
    A = m.matrix(1e-9 + np.random.default_rng(0).random(m.n_elements) ** 3)
    o, b = m.elimination_order(), m.elimination_blocks()
    B = np.random.default_rng(1).random((len(o), 1))
    res = {"shape": shape, "n": len(o)}
    s = DirectSolver("splu"); s.factor(A, order=o)
    ts = []
    for _ in range(2):
        t = time.perf_counter(); s.factor(A, order=o); ts.append(time.perf_counter() - t)
    t = time.perf_counter(); X0 = s.solve(B); res["splu"] = (min(ts), time.perf_counter() - t)
    for am in (0, 256, 1024):
        ch = SupernodalCholesky(o, b, amalgamate=am); ch.factor(A)
        ts = []
        for _ in range(2):
            t = time.perf_counter(); ch.factor(A); ts.append(time.perf_counter() - t)
        t = time.perf_counter(); X1 = ch.solve(B); tsol = time.perf_counter() - t
        res[f"chol{am}"] = (min(ts), tsol, len(ch._blocks), ch.factor_nnz, float(np.linalg.norm(X1 - X0) / np.linalg.norm(X0)))
    print(res, flush=True); out.append(res)
json.dump(out, open(sys.argv[1], "w"), indent=1)

"""Geometric multigrid preconditioned CG for H8-type elasticity on a full box grid (phase A2).

Matrix-free fine level (shared 24×24 element matrix times a per-element modulus), element-level Galerkin coarse
levels with trilinear prolongation (exact ``P̃ᵀ K P̃`` with the masked prolongation, see
:mod:`toporlogy.solvers.gmg_kernels`), Chebyshev smoothing, and a coarse solve that is either a dense Cholesky
inverse on the device (``coarse="dense"``), a smoothed-aggregation AMG V-cycle on the CPU (``coarse="amg"``, pyamg,
rigid-body near-null space) or an exact sparse factorisation (``coarse="direct"``, diagnostics). The outer Krylov
method is flexible (Polak–Ribière) CG in float64; the V-cycle may run in float32 (``vcycle_dtype``). On CUDA the
outer operator and the V-cycle are captured as CUDA graphs and replayed; :meth:`GMGSolver.update` writes the new
hierarchy into the same buffers, so a graph is captured once per problem structure.

Degrees of freedom that no element stiffens (zero diagonal: orphan nodes of masked domains, padding) are pinned.
Optionally, nodes whose adjacent elements are all softer than ``pin_rel · max(E)`` are pinned too (an
approximation; off by default).

Layout: nodal fields ``(nx+1, ny+1, nz+1, 3)`` (C order = DOF order 3n + d of :class:`toporlogy.Domain`),
element fields ``(nx, ny, nz)``.

Design and measurements: ``docs/olcumler/A2.md``. Methods are textbook multigrid (general knowledge; no multigrid
source was read in this session and none is in ``toporlogy.references``).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from toporlogy.solvers import gmg_kernels as gk
from toporlogy.solvers.gmg_deflation import IslandDeflation
from toporlogy.solvers.gmg_transfer import BlackBoxTransfer

__all__ = ["GMGSolver", "GMGConvergenceError", "solve", "pcg"]


class GMGConvergenceError(RuntimeError):
    """CG did not reach the requested relative residual within the iteration cap."""


def pcg(A: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, M: Callable[[torch.Tensor], torch.Tensor],
        x0: torch.Tensor | None = None, tol: float = 1e-8, maxit: int = 500) -> tuple[torch.Tensor, int, float, list]:
    """Flexible (Polak–Ribière) PCG. Stops on the recursively updated relative residual ‖r‖/‖b‖ ≤ tol and
    returns ``(x, iterations, true relative residual, history)``."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    bn = float(b.norm())
    if bn == 0.0:
        return torch.zeros_like(b), 0, 0.0, [0.0]
    r = b - A(x) if x0 is not None else b.clone()
    hist = [float(r.norm()) / bn]
    it = 0
    if hist[-1] <= tol:
        return x, 0, hist[-1], hist
    z = M(r)
    p = z.clone()
    rz = float((r * z).sum())
    while it < maxit:
        Ap = A(p)
        pAp = float((p * Ap).sum())
        if not pAp > 0:                    # breakdown (indefinite preconditioned step) - stop, caller decides
            break
        alpha = rz / pAp
        x.add_(p, alpha=alpha)
        r_new = r - alpha * Ap
        it += 1
        hist.append(float(r_new.norm()) / bn)
        if hist[-1] <= tol or not np.isfinite(hist[-1]):
            break
        z_new = M(r_new)
        beta = float((r_new * (z_new - z)).sum()) / rz
        rz = float((r_new * z_new).sum())
        p = z_new + beta * p
        r, z = r_new, z_new
    true_res = float((b - A(x)).norm()) / bn
    return x, it, true_res, hist


def _graphed(fn: Callable[..., Any], *examples: torch.Tensor) -> tuple[Callable[..., Any], list, Any]:
    """Capture ``fn(*static_inputs)`` as a CUDA graph. Returns (call, static_inputs, static_output); ``call(*xs)``
    copies the inputs in, replays and returns the static output (overwritten by the next replay)."""
    static_in = [e.clone() for e in examples]
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            fn(*static_in)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = fn(*static_in)

    def call(*xs: torch.Tensor) -> Any:
        for s, x in zip(static_in, xs, strict=True):
            s.copy_(x)
        g.replay()
        return static_out
    call.graph = g  # type: ignore[attr-defined]
    return call, static_in, static_out


class _DenseCoarse:
    """Dense Cholesky factor of the coarsest Galerkin level on the device; the solve (two triangular solves,
    ``cholesky_solve``) is graph-capturable. The factorisation runs in float64 (``cholesky_inverse`` was measured
    16× slower than the factorisation at n = 3375 on the GTX 1660 Ti, so no explicit inverse is formed)."""

    kind = "dense"

    def __init__(self, L: gk.Level, dtype: torch.dtype) -> None:
        self.n = L.ndof
        self.Lc = torch.linalg.cholesky(self._dense(L)).to(dtype)

    @staticmethod
    def _dense(L: gk.Level) -> torch.Tensor:
        n = L.ndof
        ed = torch.as_tensor(gk.edof_grid(L.shape), device=L.free.device)
        Ke = L.element_matrices()
        A = torch.zeros(n * n, dtype=Ke.dtype, device=Ke.device)
        A.index_add_(0, (ed[:, :, None] * n + ed[:, None, :]).reshape(-1), Ke.reshape(-1))
        return A.view(n, n) + torch.diag(L.fixed.reshape(-1).to(Ke.dtype))

    def refresh(self, new: _DenseCoarse) -> bool:
        if new.Lc.shape != self.Lc.shape:
            return False
        self.Lc.copy_(new.Lc)
        return True

    def __call__(self, b: torch.Tensor) -> torch.Tensor:
        return torch.cholesky_solve(b.reshape(-1, 1), self.Lc).view_as(b)


class _HostCoarse:
    """Coarse solve on the CPU: pyamg smoothed aggregation (one or more V-cycles) or an exact factorisation."""

    on_host = True

    def __init__(self, L: gk.Level, kind: str, amg_options: dict | None, cycles: int) -> None:
        import scipy.sparse as sp
        self.kind, self.cycles = kind, int(cycles)
        A = gk.assemble_csr(L.element_matrices(), L.shape)
        self.free_idx = np.flatnonzero(L.free.detach().cpu().numpy().reshape(-1) > 0.5)
        Af = A[self.free_idx][:, self.free_idx].tocsr()
        self.n, self.nnz = Af.shape[0], Af.nnz
        nx, ny, nz = L.shape
        self.shape = (nx + 1, ny + 1, nz + 1, 3)
        t0 = time.perf_counter()
        if kind == "amg":
            import pyamg
            opts = {"symmetry": "hermitian", "max_coarse": 500}
            opts.update(amg_options or {})
            B = gk.rigid_body_modes((nx + 1, ny + 1, nz + 1))[self.free_idx]
            self.ml = pyamg.smoothed_aggregation_solver(Af, B=B, **opts)
            self._M = self.ml.aspreconditioner(cycle="V")
            self.info = {"levels": len(self.ml.levels), "operator_complexity": float(self.ml.operator_complexity())}
        elif kind == "direct":
            from toporlogy.solvers.direct import DirectSolver
            self._ds = DirectSolver()
            self._ds.factor(sp.triu(Af, format="csr"))
            self.info = {"method": self._ds.method}
        else:
            raise ValueError(kind)
        self.info.update(n=self.n, nnz=int(self.nnz), setup_s=time.perf_counter() - t0)

    def solve_host(self, b: np.ndarray) -> np.ndarray:
        bf = b.reshape(-1)[self.free_idx]
        out = np.zeros(int(np.prod(self.shape)))
        if self.kind == "amg":
            x = self._M.matvec(bf)
            for _ in range(self.cycles - 1):          # extra cycles as stationary iteration x += M(b - A x)
                x = x + self._M.matvec(bf - self.ml.levels[0].A @ x)
        else:
            x = self._ds.solve(bf)
        out[self.free_idx] = x
        return out.reshape(self.shape)

    def __call__(self, b: torch.Tensor) -> torch.Tensor:
        x = self.solve_host(b.detach().to("cpu", torch.float64).numpy())
        return torch.from_numpy(x).to(device=b.device, dtype=b.dtype)


class GMGSolver:
    """GMG-preconditioned CG on a structured grid.

    Parameters
    ----------
    shape : (nx, ny, nz)
        Elements per axis (padding with zero-modulus elements is the caller's job; geometric coarsening stops at
        the first odd dimension).
    KE : (24, 24) array
        Unit element stiffness; element e contributes ``E_e · scale · KE``.
    scale : float
        Geometric factor (``h`` for a unit-cube element matrix scaled to edge h).
    device : str or torch.device, optional
        Default ``"cuda"`` when available, else ``"cpu"``.
    vcycle_dtype : torch.dtype, optional
        float32 on CUDA (mixed precision), float64 on the CPU by default.
    graphs : bool, optional
        Capture CUDA graphs (default: on CUDA).
    coarse : {"amg", "dense", "direct"}
        Coarse-level solver (see module docstring).
    n_geometric : int, optional
        Number of geometric grids including the fine one (the coarse solver acts on the last). Default: for
        ``"dense"`` coarsen until ``ndof ≤ max_coarse_dof``; for ``"amg"``/``"direct"`` coarsen until
        ``ndof ≤ amg_max_dof``.
    smoother_degree, cheb_lo, cheb_hi, power_its :
        Chebyshev degree and interval [lo, hi]·λmax(D⁻¹A); λmax from ``power_its`` power steps.
    pin_rel : float
        Pin nodes whose adjacent elements are all below ``pin_rel · max(E)`` (0 = only zero-stiffness DOFs).
    tol, maxit :
        Default relative residual tolerance and iteration cap of :meth:`solve`.
    """

    def __init__(self, shape: tuple[int, int, int], KE: Any, scale: float = 1.0, device: Any = None,
                 vcycle_dtype: torch.dtype | None = None, graphs: bool | None = None, coarse: str = "dense",
                 n_geometric: int | None = None, max_coarse_dof: int = 4000, amg_max_dof: int = 60000,
                 smoother_degree: int = 3, cheb_lo: float = 0.1, cheb_hi: float = 1.1, power_its: int = 15,
                 pin_rel: float = 0.0, tol: float = 1e-8, maxit: int = 500, amg_options: dict | None = None,
                 amg_cycles: int = 1, transfer: Any = "trilinear", cycle: str = "K",
                 deflate_tau: float | None = 1e-3) -> None:
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("shape must be (nx, ny, nz)")
        if coarse not in ("amg", "dense", "direct"):
            raise ValueError("coarse: amg | dense | direct")
        tr = [transfer] if isinstance(transfer, str) else list(transfer)
        if any(t not in ("blackbox", "trilinear") for t in tr):
            raise ValueError("transfer: blackbox | trilinear, or a list per level")
        if cycle not in ("V", "W", "K"):
            raise ValueError("cycle: V | W | K")
        self.transfer, self.cycle = transfer, cycle
        self._tr = tr
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vdtype = vcycle_dtype or (torch.float32 if self.device.type == "cuda" else torch.float64)
        self.graphs = (self.device.type == "cuda") if graphs is None else bool(graphs and self.device.type == "cuda")
        self.coarse_kind, self.n_geometric = coarse, n_geometric
        self.max_coarse_dof, self.amg_max_dof = int(max_coarse_dof), int(amg_max_dof)
        self.degree, self.cheb_lo, self.cheb_hi, self.power_its = int(smoother_degree), cheb_lo, cheb_hi, power_its
        self.pin_rel, self.tol, self.maxit = float(pin_rel), float(tol), int(maxit)
        self.amg_options, self.amg_cycles = amg_options, int(amg_cycles)
        self.deflate_tau = deflate_tau
        self.deflation: IslandDeflation | None = None
        self.scale = float(scale)
        self.KE = torch.as_tensor(np.asarray(KE, float), dtype=torch.float64, device=self.device)
        if self.KE.shape != (24, 24):
            raise ValueError("KE must be 24×24")
        self._Pl = torch.as_tensor(gk.local_P(), dtype=torch.float64, device=self.device)
        nx, ny, nz = self.shape
        self.node_shape = (nx + 1, ny + 1, nz + 1, 3)
        self.ndof = int(np.prod(self.node_shape))
        self.levels: list[gk.Level] = []       # V-cycle levels (vdtype); levels[-1] is the coarse-solver level
        self.transfers: list[Any] = []          # per smoothing level: BlackBoxTransfer (vdtype) or None (trilinear)
        self.A64: gk.Level | None = None        # outer operator (float64)
        self.coarse: Any = None
        self._coef: list[torch.Tensor] = []     # Chebyshev coefficients per smoothing level (device tensors)
        self._A_call: Any = None
        self._M_call: Any = None
        self._structure: tuple | None = None
        self.fixed_user: torch.Tensor | None = None
        self.n_pinned = 0
        self.n_updates = 0
        self.n_graph_captures = 0
        self.last: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup
    def _to_dev(self, a: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(a, dtype=dtype).to(self.device) if not isinstance(a, torch.Tensor) \
            else a.detach().to(device=self.device, dtype=dtype)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update(self, E_elem: Any, fixed_mask: Any = None) -> None:
        """Rebuild the hierarchy for element moduli ``E_elem`` (shape or (ne,)) and, optionally, a new fixed mask
        (bool, ``(nx+1, ny+1, nz+1, 3)`` or (ndof,)). Buffers are reused when the structure is unchanged."""
        t0 = time.perf_counter()
        f64 = torch.float64
        E = self._to_dev(E_elem, f64).reshape(-1)
        if E.numel() != int(np.prod(self.shape)):
            raise ValueError(f"E has {E.numel()} entries, the grid {int(np.prod(self.shape))} elements")
        if fixed_mask is not None:
            self.fixed_user = self._to_dev(fixed_mask, torch.bool).reshape(self.node_shape)
        if self.fixed_user is None:
            self.fixed_user = torch.zeros(self.node_shape, dtype=torch.bool, device=self.device)
        s = E * self.scale
        # fine-level pins: user-fixed, zero diagonal, and (optionally) near-void nodes
        L0 = gk.Level(self.shape, torch.ones(self.node_shape, dtype=f64, device=self.device), s=s, KE=self.KE)
        d0 = L0.raw_diag()
        pin = self.fixed_user | (d0 <= 0)
        if self.pin_rel > 0:
            nm = gk.node_max(E, self.shape)
            pin |= (nm <= self.pin_rel * float(E.max()))[..., None].expand_as(pin)
        self.n_pinned = int((pin & ~self.fixed_user).sum())
        free = (~pin).to(f64)
        L0 = gk.Level(self.shape, free, s=s, KE=self.KE)
        levels = [L0]
        transfers: list[Any] = []
        target = self.n_geometric
        while True:
            L = levels[-1]
            nx, ny, nz = L.shape
            if target is not None and len(levels) >= target:
                break
            limit = self.max_coarse_dof if self.coarse_kind == "dense" else self.amg_max_dof
            if target is None and L.ndof <= limit:
                break
            if nx % 2 or ny % 2 or nz % 2:
                break
            cshape = (nx // 2, ny // 2, nz // 2)
            if self._tr[min(len(transfers), len(self._tr) - 1)] == "blackbox":
                T = BlackBoxTransfer(L)
                Kc = T.galerkin(L)
                transfers.append(T)
            elif L.Ke is None:
                Kc = gk.galerkin_from_scale(L.s, L.KE, L.shape, L.free, self._Pl)
                transfers.append(None)
            else:
                Kc = gk.galerkin_from_elements(L.Ke, L.shape, self._Pl)
                transfers.append(None)
            cd = gk.scatter(torch.diagonal(Kc, dim1=1, dim2=2).contiguous(), cshape,
                            torch.zeros((cshape[0] + 1, cshape[1] + 1, cshape[2] + 1, 3), dtype=f64,
                                        device=self.device))
            cfree = (cd > 0).to(f64)
            gk.mask_elements(Kc, cfree, cshape)
            levels.append(gk.Level(cshape, cfree, Ke=Kc))
        if len(levels) < 2 and self.coarse_kind == "dense" and levels[0].ndof > 4 * self.max_coarse_dof:
            raise ValueError(f"grid {self.shape} cannot be coarsened (odd dimension); pad it to even sizes")
        for L in levels[:-1]:
            L.dinv = 1.0 / L.diag()
            L.lmax = gk.power_lmax(L, self.power_its)
        t_h = time.perf_counter()
        if self.coarse_kind == "dense":
            coarse: Any = _DenseCoarse(levels[-1], self.vdtype)
        else:
            coarse = _HostCoarse(levels[-1], self.coarse_kind, self.amg_options, self.amg_cycles)
        t_c = time.perf_counter()
        defl = None
        if self.deflate_tau:
            defl = IslandDeflation(L0, self.fixed_user, self.deflate_tau)
            if defl.n == 0:
                defl = None
        t_d = time.perf_counter()
        coef = [self._cheb_coef(L.lmax) for L in levels[:-1]]
        vlev = [L.to(self.vdtype) for L in levels]
        vtr = [None if T is None else T.to(self.vdtype) for T in transfers]
        structure = (tuple(L.shape for L in levels), self.coarse_kind, tuple(self._tr), self.cycle,
                     None if defl is None else (defl.n, int(defl.sel.numel()), defl.rank))
        old_defl, self.deflation = self.deflation, defl
        reuse = self._structure == structure and self.graphs and self._A_call is not None
        if reuse:
            if defl is not None and old_defl is not None:
                for k in ("sel", "zid", "R", "free_sel", "Ginv"):
                    getattr(old_defl, k).copy_(getattr(defl, k))
                old_defl.n_components, old_defl.island_nodes = defl.n_components, defl.island_nodes
                self.deflation = old_defl
            for old, new in zip(self.levels, vlev, strict=True):
                for k in ("free", "fixed", "s", "Ke", "dinv"):
                    o, n = getattr(old, k), getattr(new, k)
                    if o is not None:
                        o.copy_(n)
                old.lmax = new.lmax
            for old, new in zip(self._coef, coef, strict=True):
                old.copy_(new)
            for old, new in zip(self.transfers, vtr, strict=True):
                if old is not None:
                    for p in old.W:
                        old.W[p].copy_(new.W[p])
            assert self.A64 is not None
            self.A64.free.copy_(L0.free)
            self.A64.fixed.copy_(L0.fixed)
            self.A64.s.copy_(L0.s)
            if isinstance(coarse, _DenseCoarse):
                reuse = self.coarse.refresh(coarse)
            else:
                self.coarse = coarse
        if not reuse:
            self.levels, self._coef, self.coarse, self.transfers = vlev, coef, coarse, vtr
            self.A64 = L0
            self._A_call = self._M_call = None
            self._structure = structure
        self.n_updates += 1
        self._sync()
        self.last_setup = {"setup_s": time.perf_counter() - t0, "hierarchy_s": t_h - t0, "coarse_setup_s": t_c - t_h,
                           "levels": [list(L.shape) for L in levels], "coarse": self.coarse_kind,
                           "coarse_ndof": int(levels[-1].ndof), "lmax": [L.lmax for L in levels[:-1]],
                           "deflation_s": t_d - t_c,
                           "deflation": None if defl is None else defl.info(),
                           "n_pinned": self.n_pinned, "graph_reused": bool(reuse), "transfer": self.transfer,
                           "cycle": self.cycle,
                           "trilinear_fallback_nodes": [0 if T is None else T.n_trilinear_fallback
                                                        for T in transfers]}
        if isinstance(coarse, _HostCoarse):
            self.last_setup["coarse_info"] = coarse.info

    def _cheb_coef(self, lmax: float) -> torch.Tensor:
        """[1/θ, a_1, b_1, …, a_{k−1}, b_{k−1}] of the Chebyshev recurrence as a device tensor (graph-safe)."""
        hi, lo = self.cheb_hi * lmax, self.cheb_lo * lmax
        theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
        sigma = theta / delta
        rho = 1.0 / sigma
        c = [1.0 / theta]
        for _ in range(self.degree - 1):
            rn = 1.0 / (2.0 * sigma - rho)
            c += [rn * rho, 2.0 * rn / delta]
            rho = rn
        return torch.tensor(c, dtype=self.vdtype, device=self.device)

    # ------------------------------------------------------------------ V-cycle
    def _smooth(self, lev: int, x: torch.Tensor | None, b: torch.Tensor) -> torch.Tensor:
        L, c = self.levels[lev], self._coef[lev]
        dinv = L.dinv
        r = b if x is None else b - L.apply(x)
        d = dinv * r * c[0]
        x = d if x is None else x + d
        for k in range(self.degree - 1):
            r = r - L.apply(d)
            d = c[1 + 2 * k] * d + c[2 + 2 * k] * dinv * r
            x = x + d
        return x

    def _down(self, b: torch.Tensor) -> tuple[list, list, torch.Tensor]:
        xs, bs = [], []
        for lev in range(len(self.levels) - 1):
            L = self.levels[lev]
            x = self._smooth(lev, None, b)
            r = b - L.apply(x)
            xs.append(x)
            bs.append(b)
            T = self.transfers[lev]
            b = self.levels[lev + 1].free * (gk.restrict(r) if T is None else T.restrict(r))
        return xs, bs, b

    def _up(self, xs: list, bs: list, e: torch.Tensor) -> torch.Tensor:
        for lev in reversed(range(len(self.levels) - 1)):
            L = self.levels[lev]
            T = self.transfers[lev]
            x = xs[lev] + L.free * (gk.prolong(e) if T is None else T.prolong(e))
            e = self._smooth(lev, x, bs[lev])
        return e

    def _restrict(self, lev: int, r: torch.Tensor) -> torch.Tensor:
        T = self.transfers[lev]
        return self.levels[lev + 1].free * (gk.restrict(r) if T is None else T.restrict(r))

    def _prolong(self, lev: int, e: torch.Tensor) -> torch.Tensor:
        T = self.transfers[lev]
        return self.levels[lev].free * (gk.prolong(e) if T is None else T.prolong(e))

    def _cycle(self, lev: int, b: torch.Tensor) -> torch.Tensor:
        """Multigrid cycle at level ``lev`` (V, W or K) with zero initial guess."""
        last = len(self.levels) - 1
        if lev == last:
            return self.coarse(b)
        L = self.levels[lev]
        x = self._smooth(lev, None, b)
        rc = self._restrict(lev, b - L.apply(x))
        if lev + 1 == last or self.cycle == "V":
            ec = self._cycle(lev + 1, rc)
        elif self.cycle == "W":
            ec = self._cycle(lev + 1, rc)
            ec = ec + self._cycle(lev + 1, rc - self.levels[lev + 1].apply(ec))
        else:
            ec = self._kcycle(lev + 1, rc)
        return self._smooth(lev, x + self._prolong(lev, ec), b)

    def _kcycle(self, lev: int, r: torch.Tensor) -> torch.Tensor:
        """Two flexible-CG steps on level ``lev`` preconditioned by its cycle (K-cycle; scalars stay on the device,
        so the whole cycle is graph-capturable)."""
        A = self.levels[lev].apply
        v1 = self._cycle(lev, r)
        w1 = A(v1)
        rho1 = (v1 * w1).sum()
        a1 = (v1 * r).sum()
        r2 = r - (a1 / rho1) * w1
        v2 = self._cycle(lev, r2)
        w2 = A(v2)
        gam = (v2 * w1).sum()
        bet = (v2 * w2).sum()
        a2 = (v2 * r2).sum()
        rho2 = bet - gam * gam / rho1
        ok = rho2 > 1e-30 * bet.abs()
        c2 = torch.where(ok, a2 / torch.where(ok, rho2, torch.ones_like(rho2)), torch.zeros_like(rho2))
        return (a1 / rho1 - c2 * gam / rho1) * v1 + c2 * v2

    def vcycle(self, r: torch.Tensor) -> torch.Tensor:
        """One multigrid cycle (``self.cycle``) applied to a float64 residual (no graph)."""
        b = r.to(self.vdtype)
        if self.deflation is None:
            return self._cycle(0, b).to(r.dtype)
        return self.deflation.wrap(self.levels[0].apply, lambda q: self._cycle(0, q), b).to(r.dtype)

    def _build_calls(self) -> None:
        assert self.A64 is not None
        A64 = self.A64
        if not self.graphs:
            self._A_call, self._M_call = A64.apply, self.vcycle
            return
        ex = torch.zeros(self.node_shape, dtype=torch.float64, device=self.device)
        A_call, _, _ = _graphed(A64.apply, ex)
        if not getattr(self.coarse, "on_host", False):
            M_call, _, _ = _graphed(self.vcycle, ex)
        elif self.cycle != "V" and len(self.levels) > 2:
            self._A_call, self._M_call = A64.apply, self.vcycle      # host coarse inside W/K cycles: no graph
            return
        else:
            def down(r: torch.Tensor) -> tuple:
                return self._down(r.to(self.vdtype))
            down_call, _, (xs, bs, bc) = _graphed(down, ex)

            def up(e: torch.Tensor) -> torch.Tensor:
                return self._up(xs, bs, e).to(torch.float64)
            up_call, _, _ = _graphed(up, torch.zeros_like(bc))

            def M_call(r: torch.Tensor) -> torch.Tensor:
                _, _, bc_ = down_call(r)
                return up_call(self.coarse(bc_))
        self.n_graph_captures += 1
        self._A_call, self._M_call = (lambda u: A_call(u).clone()), (lambda r: M_call(r).clone())

    # ------------------------------------------------------------------ solve
    def operator(self, u: Any) -> torch.Tensor:
        """``A u`` (float64, fixed DOFs as identity) for a nodal field."""
        assert self.A64 is not None, "update() first"
        return self.A64.apply(self._to_dev(u, torch.float64).reshape(self.node_shape))

    @property
    def free_mask(self) -> torch.Tensor:
        assert self.A64 is not None, "update() first"
        return self.A64.free > 0.5

    def solve(self, E_elem: Any, F: Any, fixed_mask: Any = None, x0: Any = None, tol: float | None = None,
              maxit: int | None = None) -> Any:
        """Solve ``K(E) u = F`` with zero displacement on fixed / pinned DOFs.

        ``E_elem`` None reuses the current hierarchy. ``F`` (and ``x0``) as nodal field ``(nx+1, ny+1, nz+1, 3)``,
        (ndof,) or (ndof, k); NumPy in → NumPy out, tensor in → tensor out (float64, on this solver's device).
        Statistics of the last call: :attr:`last` — ``requested_tol``, ``achieved_true_rel_residual``
        (‖F − K u‖/‖F‖ recomputed after CG, worst load case) and ``converged = achieved ≤ requested`` (no hidden
        factor). When the recursive CG residual meets ``tol`` but the recomputed one does not, CG is restarted from
        the current iterate (at most 5 times, within ``maxit`` in total). An unconverged result is returned with
        ``converged = False``; raising is the caller's decision (:class:`ElasticityGMG` fails closed).
        """
        if E_elem is not None:
            self.update(E_elem, fixed_mask)
        elif fixed_mask is not None:
            raise ValueError("a new fixed_mask needs E_elem")
        if self.A64 is None:
            raise RuntimeError("update(E) first")
        if self._A_call is None:
            self._build_calls()
        tol = self.tol if tol is None else float(tol)
        maxit = self.maxit if maxit is None else int(maxit)
        is_np = not isinstance(F, torch.Tensor)
        Ft = self._to_dev(F, torch.float64)
        squeeze = Ft.dim() == 1 or Ft.shape == self.node_shape
        Fm = Ft.reshape(self.ndof, -1)
        X0 = None if x0 is None else self._to_dev(x0, torch.float64).reshape(self.ndof, -1)
        free = self.A64.free
        out = torch.zeros_like(Fm)
        rows = []
        t0 = time.perf_counter()
        for c in range(Fm.shape[1]):
            b = free * Fm[:, c].reshape(self.node_shape)
            xi = None if X0 is None else free * X0[:, c].reshape(self.node_shape)
            u, it, tres, _ = pcg(self._A_call, b, self._M_call, xi, tol, maxit)
            restarts = 0
            while tres > tol and it < maxit and restarts < 5:      # recursive residual met tol, true one did not
                u, it2, tres, _ = pcg(self._A_call, b, self._M_call, u, tol, maxit - it)
                it += it2
                restarts += 1
                if it2 == 0:
                    break
            out[:, c] = u.reshape(-1)
            rows.append({"iters": it, "restarts": restarts, "requested_tol": tol, "achieved_true_rel_residual": tres,
                         "converged": bool(tres <= tol),
                         "dropped_load_norm": float(((1 - free) * Fm[:, c].reshape(self.node_shape)).norm())})
        self._sync()
        worst = max(r["achieved_true_rel_residual"] for r in rows)
        self.last = {"solve_s": time.perf_counter() - t0, "cases": rows,
                     "iters": max(r["iters"] for r in rows), "requested_tol": tol,
                     "achieved_true_rel_residual": worst, "true_rel_residual": worst,
                     "converged": all(r["converged"] for r in rows)}
        res = out[:, 0] if squeeze else out
        if Ft.shape == self.node_shape:
            res = res.reshape(self.node_shape)
        return res.cpu().numpy() if is_np else res


def solve(E_elem: Any, F: Any, fixed_mask: Any, KE: Any, scale: float = 1.0, x0: Any = None, tol: float = 1e-8,
          **kw: Any) -> Any:
    """One-off convenience wrapper: ``GMGSolver(E.shape, KE, scale, **kw).solve(E, F, fixed_mask, x0, tol)``."""
    shape = tuple(np.shape(E_elem)) if np.ndim(E_elem) == 3 else kw.pop("shape")
    s = GMGSolver(shape, KE, scale, **kw)
    return s.solve(E_elem, F, fixed_mask, x0=x0, tol=tol)

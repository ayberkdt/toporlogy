"""Study: one optimisation problem and its loop (docs/mimari_plan.md §3).

``Study(domain, physics, loads, chain, interp, objective, constraints, optimizer, schedule)``; ``run()`` follows
the loop of the legacy voxel driver (kept as a regression oracle in ``tests/topo``):

1. x̄ = chain(x), E = interp(x̄), U = physics.solve(E) (if a response needs it);
2. objective and constraints, gradients by autograd;
3. optimizer step (OC gets a gradient-free constraint function for its bisection);
4. change = max|Δx|; schedule update — if it changed the problem (β step) the convergence test is skipped and the
   optimizer is reset; else stop when change ≤ tol and the schedule is finished.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from aerokit.topo.chain import Chain
from aerokit.topo.domain import Domain
from aerokit.topo.optim.base import Optimizer
from aerokit.topo.optim.schedule import Schedule
from aerokit.topo.responses import EvalContext, Response, Volume

__all__ = ["Constraint", "Study", "Result", "Evaluation"]


@dataclass
class Constraint:
    """``response ≤ upper``, passed to the optimizer as ``g = (response − upper) · scale ≤ 0``.

    ``scale`` default: 1/|upper| (so g is a relative violation), 1 when upper = 0.
    """

    response: Response
    upper: float
    scale: float | None = None
    name: str | None = None            # history / report key; default the response's name (must be unique)

    @property
    def key(self) -> str:
        return self.name or self.response.name

    def g(self, value: torch.Tensor) -> torch.Tensor:
        s = self.scale if self.scale is not None else (1.0 / abs(self.upper) if self.upper != 0 else 1.0)
        return (value - self.upper) * s


@dataclass
class Evaluation:
    f0: float
    df0: np.ndarray
    g: np.ndarray
    dg: np.ndarray
    xphys: np.ndarray
    U: np.ndarray | None
    values: dict[str, float]


@dataclass
class Result:
    x: np.ndarray                      # final design vector
    xphys: np.ndarray                  # final physical density (domain.shape), at the final β
    history: dict[str, list[float]]
    iterations: int
    converged: bool
    seconds: float
    U: np.ndarray | None               # displacements of the FINAL design (re-evaluated), None if not needed
    settings: dict[str, Any] = field(default_factory=dict)
    final: Evaluation | None = None    # objective / constraints / U re-evaluated at ``x`` (the returned design)

    @property
    def objective(self) -> float:
        """Objective of the returned design (``final``); the history holds the designs analysed in the loop."""
        return self.final.f0 if self.final is not None else self.history["f0"][-1]

    @property
    def constraint_values(self) -> dict[str, float]:
        """Raw constraint response values of the returned design (keys = constraint names)."""
        if self.final is None:
            return {}
        return {k: v for k, v in self.final.values.items() if k in self.settings.get("constraint_keys", ())}


class Study:
    """One topology optimisation problem.

    Parameters
    ----------
    domain : Domain
    physics : object with ``solve(E_active_tensor, F) -> U tensor`` and ``elements`` (active element numbers)
    loads : array (ndof,) or (ndof, n_cases)
    chain : Chain
    interp : torch module x̄ → E (element grid)
    objective : Response
    constraints : sequence of Constraint
    optimizer : Optimizer
    schedule : Schedule, optional
    """

    def __init__(self, domain: Domain, physics: Any, loads: np.ndarray, chain: Chain, interp: torch.nn.Module,
                 objective: Response, constraints: Sequence[Constraint], optimizer: Optimizer,
                 schedule: Schedule | None = None, recorder: Any = None,
                 move_beta_ref: float | None = None) -> None:
        self.domain, self.physics, self.chain, self.interp = domain, physics, chain, interp
        # move limit shrinks as move0 · min(1, move_beta_ref / β) after every β step of the schedule (sharp
        # projections make large steps unstable; 2.0 is the common choice). None: move limit unchanged.
        self.move_beta_ref = None if move_beta_ref is None else float(move_beta_ref)
        F = np.asarray(loads, float)
        self.F = F[:, None] if F.ndim == 1 else F
        if self.F.shape[0] != domain.ndof:
            raise ValueError(f"loads have {self.F.shape[0]} rows, the domain {domain.ndof} DOFs")
        self.objective, self.constraints = objective, list(constraints)
        self.optimizer, self.schedule, self.recorder = optimizer, schedule or Schedule(), recorder
        if optimizer.max_constraints is not None and len(self.constraints) > optimizer.max_constraints:
            raise ValueError(f"{optimizer.name} handles at most {optimizer.max_constraints} constraint(s)")
        keys = [c.key for c in self.constraints]
        if len(set(keys)) != len(keys):
            raise ValueError(f"constraint names must be unique: {keys}; set Constraint(name=...)")
        self._active = torch.as_tensor(domain.active_elements, dtype=torch.long)
        self.n_evaluations = 0
        self.config: dict[str, Any] | None = None          # resolved declarative config, when built from one
        # A first link that maps a reduced variable vector z → design vector (symmetry, extrusion, pattern) declares
        # ``n_in`` (length of z) and ``reduce(x_design) -> z`` (used for the start design).
        first = chain.links[0] if len(chain.links) else None
        self.n_var = int(getattr(first, "n_in", domain.n_design))
        self._reduce = getattr(first, "reduce", None)

    # ------------------------------------------------------------------ evaluation
    def _needs(self, responses: Sequence[Response]) -> set[str]:
        out: set[str] = set()
        for r in responses:
            out |= set(r.needs)
        return out

    def context(self, x: torch.Tensor, need_u: bool) -> EvalContext:
        fields, v = [], x
        for link in self.chain.links:
            v = link(v)
            fields.append(v)
        xphys = v
        E = self.interp(xphys)
        U = self.physics.solve(E.reshape(-1)[self._active], self.F) if need_u else None
        return EvalContext(self.domain, xphys, E, U, self.F, x=x, fields=fields, physics=self.physics,
                           interp=self.interp, chain=self.chain)

    def evaluate(self, x: np.ndarray, gradients: bool = True) -> Evaluation:
        """Objective, constraints and (optionally) their gradients at the design vector ``x``."""
        self.n_evaluations += 1
        xt = torch.as_tensor(np.asarray(x, float), dtype=torch.float64).clone().requires_grad_(gradients)
        rs = [self.objective] + [c.response for c in self.constraints]
        with torch.set_grad_enabled(gradients):
            ctx = self.context(xt, "u" in self._needs(rs))
            f0 = self.objective(ctx)
            raw = [c.response(ctx) for c in self.constraints]
            gs = [c.g(v) for c, v in zip(self.constraints, raw, strict=True)]
        values = {self.objective.name: float(f0)}
        for c, v in zip(self.constraints, raw, strict=True):
            values[c.key] = float(v)
        n = len(xt)
        df0, dg = np.zeros(n), np.zeros((len(gs), n))
        if gradients:
            outs = [f0] + gs
            for i, o in enumerate(outs):
                if not o.requires_grad:
                    continue
                (gr,) = torch.autograd.grad(o, xt, retain_graph=i < len(outs) - 1, allow_unused=True)
                if gr is None:
                    continue
                if i == 0:
                    df0 = gr.detach().numpy().copy()
                else:
                    dg[i - 1] = gr.detach().numpy()
        return Evaluation(float(f0), df0, np.array([float(g) for g in gs]), dg,
                          ctx.xphys.detach().numpy().copy(),
                          None if ctx.U is None else ctx.U.detach().numpy().copy(), values)

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        """Constraints g(x) without gradients (only for constraints that need no state solve)."""
        if "u" in self._needs([c.response for c in self.constraints]):
            raise ValueError("constraint_values: a constraint needs U; use evaluate()")
        with torch.no_grad():
            ctx = self.context(torch.as_tensor(np.asarray(x, float), dtype=torch.float64), False)
            return np.array([float(c.g(c.response(ctx))) for c in self.constraints])

    def physical(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.chain(torch.as_tensor(np.asarray(x, float), dtype=torch.float64)).numpy().copy()

    def default_start(self) -> np.ndarray:
        """Uniform start from the first Volume constraint's bound (topo3d rule), else 0.5."""
        x = None
        for c in self.constraints:
            if isinstance(c.response, Volume):
                x = self.domain.uniform_start(c.upper)
                break
        if x is None:
            x = np.full(self.domain.n_design, 0.5)
        if self._reduce is not None:
            x = np.asarray(self._reduce(x), float)
        return x

    # ------------------------------------------------------------------ loop
    def run(self, x0: np.ndarray | None = None, max_iter: int = 100, tol: float = 0.01,
            callback: Callable[[int, Evaluation, float], None] | None = None) -> Result:
        t0 = time.perf_counter()
        x = self.default_start() if x0 is None else np.asarray(x0, float).copy()
        if x.shape != (self.n_var,):
            raise ValueError(f"x0 must have {self.n_var} entries")
        cfn = None
        if "u" not in self._needs([c.response for c in self.constraints]):
            cfn = self.constraint_values
        hist: dict[str, list[float]] = {"f0": [], "change": [], "seconds": []}
        for c in self.constraints:
            hist[c.key] = []
        for k in self.schedule.state():
            hist[k] = []
        if self.recorder is not None:
            self.recorder.start(self)
        it, change, ev = 0, 1.0, None
        move0 = getattr(self.optimizer, "move", None)
        while it < max_iter:
            it += 1
            for k, v in self.schedule.state().items():
                hist[k].append(v)
            ev = self.evaluate(x)
            hist["f0"].append(ev.f0)
            for c in self.constraints:
                hist[c.key].append(ev.values[c.key])
            xnew = self.optimizer.step(x, ev.f0, ev.df0, ev.g, ev.dg, cfn)
            change = float(np.max(np.abs(xnew - x))) if x.size else 0.0
            x = xnew
            hist["change"].append(change)
            hist["seconds"].append(time.perf_counter() - t0)
            if callback is not None:
                callback(it, ev, change)
            if self.recorder is not None:
                self.recorder.log(it, ev, change, self.schedule.state())
            if self.schedule.update(it, change):
                self.optimizer.reset()
                beta = self.schedule.state().get("beta")
                if self.move_beta_ref is not None and move0 is not None and beta:
                    self.optimizer.move = float(move0) * min(1.0, self.move_beta_ref / float(beta))
                continue
            if change <= tol and self.schedule.finished:
                break
        # The loop analysed x_k and then stepped to x_{k+1}: re-evaluate the RETURNED design so that its objective,
        # constraints, U and x̄ belong to the same design (history[-1] belongs to the previous one).
        final = self.evaluate(x, gradients=False)
        res = Result(x, final.xphys, hist, it, change <= tol and self.schedule.finished, time.perf_counter() - t0,
                     final.U, self.settings(), final)
        if move0 is not None:
            self.optimizer.move = move0                 # leave the optimizer as it was given
        if self.recorder is not None:
            self.recorder.finish(res)
        return res

    def settings(self) -> dict[str, Any]:
        s: dict[str, Any] = {"domain": repr(self.domain), "optimizer": self.optimizer.name,
                             "objective": self.objective.name,
                             "constraints": [(c.key, c.upper) for c in self.constraints],
                             "constraint_keys": [c.key for c in self.constraints],
                             "links": [type(m).__name__ for m in self.chain.links],
                             "interp": type(self.interp).__name__}
        for k in ("element", "nu"):
            if hasattr(self.physics, k):
                s[k] = getattr(self.physics, k)
        if hasattr(self.physics, "solver"):
            s["solver"] = getattr(self.physics.solver, "name", type(self.physics.solver).__name__)
        return s

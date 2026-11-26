"""Study: one optimisation problem and its loop (docs/mimari_plan.md §2).

``Study(domain, physics, loads, chain, interp, objective, constraints, optimizer, schedule)``; ``run()`` follows
the loop of the legacy voxel driver (kept as a regression oracle in ``tests/toporlogy``):

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
from typing import Any, Protocol, cast

import numpy as np
import torch

from toporlogy.design.chain import Chain
from toporlogy.core.domain import Domain
from toporlogy.optimizers.base import Optimizer
from toporlogy.optimizers.schedule import Schedule
from toporlogy.optimization.responses import EvalContext, Response, Volume
from toporlogy.optimization.diagnostics import kkt_estimate

__all__ = ["Constraint", "Study", "Result", "Evaluation", "DensityEvaluation"]


class _HasMove(Protocol):
    """An optimizer with a move limit (MMA, OC); ``Study`` scales it with β when ``move_beta_ref`` is set."""

    move: float


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
class DensityEvaluation:
    """Responses of a physical density x̄ that the chain did not produce (e.g. a repaired design).

    ``values``: objective and constraint values (keys as in :attr:`Evaluation.values`) of the responses that need
    only x̄ (and U). ``skipped``: key → reason for the responses that need the design chain or the design vector
    (robust realisations, eroded-field connectivity, bound formulation); they get no value rather than a wrong one.
    """

    values: dict[str, float]
    skipped: dict[str, str]
    U: np.ndarray | None


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
    # Why the loop stopped (docs/gelistirme_plani.md G7): "design_change" (change <= tol with the schedule finished:
    # ``converged``), "max_iter", or "schedule_step_at_max_iter" (the schedule changed the problem, e.g. a beta step,
    # in the last iteration: the returned design was never optimised at the final schedule state).
    termination: str = ""
    # Constraint feasibility of the returned design, same rule as the run summary: value <= upper + tol * |upper|.
    feasible: bool | None = None
    max_violation: float | None = None  # max over constraints of max(0, (value - upper) / |upper|)
    feasibility_tol: float | None = None  # the tol of that rule (Study.run's feasibility_tol)
    # Estimate of the KKT residual at the returned design (optimizers with ``kkt_multipliers``: MMA, OC): max of the
    # relative projected stationarity |s grad f0 + mu^T grad g| (box 0..1) and the complementarity |mu_j g_j|, with
    # the multipliers of the last step (an estimate: they belong to the previous iterate). None without multipliers.
    kkt_residual: float | None = None
    # Optimizer-independent first-order optimality of the returned design (any optimizer, with constraints):
    # ``diagnostics.kkt_estimate`` residual, multipliers fitted to the returned design itself (NNLS over the active
    # constraints). None without constraints.
    stationarity: float | None = None

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


def _reset_scales(response: Any, _seen: set[int] | None = None) -> None:
    """Call ``reset_scale()`` on a response and on the responses it wraps (attributes ``response``, ``responses``,
    ``objective``, ``numerator``, ``denominator``): a scale taken at the first evaluation (KS scale, adaptive
    constraint scale, reference of a weighted sum) belongs to one run."""
    seen = set() if _seen is None else _seen
    if response is None or id(response) in seen:
        return
    seen.add(id(response))
    fn = getattr(response, "reset_scale", None)
    if callable(fn):
        fn()
    for name in ("response", "objective", "numerator", "denominator"):
        _reset_scales(getattr(response, name, None), seen)
    for child in getattr(response, "responses", None) or ():
        _reset_scales(child, seen)


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
    reset_on_model_change : bool
        A physics with ``model_epoch`` (the energy interpolation of the non-linear physics changes its model when its
        adaptive cutoff moves) resets the optimizer when the epoch changes, so asymptotes and multipliers of the old
        model are not reused (docs/olcumler/model_donemi_onkayit.md). No effect on other physics.
    torch_threads : int or None
        Torch intra-op threads while :meth:`run` loops (``torch.set_num_threads``; the previous value is restored when
        the run ends, also on an error). Default 1: the loop's tensors are element grids and vectors, whose parallel
        regions gain little and, on a loaded machine, wait for every thread to get a time slice (measured
        26 Nov 2026 at 30 000 elements under full load: the OC bisection's chain evaluations 55x slower with 6 threads
        than with 1, docs/olcumler/hizlandirma.md); one thread also makes the reductions independent of the core
        count. ``None`` leaves torch's setting alone (e.g. the torch CPU path of :mod:`toporlogy.solvers.gmg`).
    """

    def __init__(self, domain: Domain, physics: Any, loads: np.ndarray, chain: Chain, interp: torch.nn.Module,
                 objective: Response, constraints: Sequence[Constraint], optimizer: Optimizer,
                 schedule: Schedule | None = None, recorder: Any = None,
                 move_beta_ref: float | None = None, torch_threads: int | None = 1,
                 reset_on_model_change: bool = True) -> None:
        if torch_threads is not None and int(torch_threads) < 1:
            raise ValueError("torch_threads: None or an integer >= 1")
        self.torch_threads = None if torch_threads is None else int(torch_threads)
        self.reset_on_model_change = bool(reset_on_model_change)
        self.n_model_resets = 0
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
        if getattr(optimizer, "needs_constraint_fn", False) and "u" in self._needs([c.response for c in constraints]):
            names = [c.key for c in self.constraints if "u" in c.response.needs]
            raise ValueError(f"{optimizer.name} evaluates the constraint inside its step and cannot use constraint(s) "
                             f"that need a state solve: {names}; use MMA")
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
        self.n_fields = int(getattr(interp, "n_fields", 1))
        if self.n_fields > 1 or int(getattr(first, "n_fields", 1)) > 1:
            self._check_phases(first)

    # Multi-phase designs (toporlogy.design.phases): what has been verified with several design fields.
    _PHASE_LINKS = ("Embed", "DensityFilter", "HeavisideProjection", "Passive")
    _PHASE_RESPONSES = ("Compliance", "PhaseVolume", "Mass", "Cost")

    def _check_phases(self, first: Any) -> None:
        nf = int(getattr(first, "n_fields", 1))
        if nf != self.n_fields:
            raise ValueError(f"the chain's embed link has {nf} design field(s), the interpolation "
                             f"{type(self.interp).__name__} {self.n_fields}; build Embed(domain, n_fields=m) for m "
                             "phases")
        bad = [type(lk).__name__ for lk in self.chain.links if type(lk).__name__ not in self._PHASE_LINKS]
        if bad or type(first).__name__ != "Embed":
            raise ValueError(f"with several phases the chain may hold only {', '.join(self._PHASE_LINKS)} (embed "
                             f"first); got {[type(lk).__name__ for lk in self.chain.links]}")
        if getattr(self.optimizer, "needs_constraint_fn", False):
            raise ValueError(f"{self.optimizer.name} does not handle several phases: its update assumes one "
                             "constraint whose gradient is positive in every variable, while the mass and cost of a "
                             "phase field may fall as it grows (OC), or a latent simplex variable (multi-material "
                             "SiMPL, arXiv:2605.11994) that is not implemented; use MMA or GCMMA")
        rs = [self.objective] + [c.response for c in self.constraints]
        bad = [r.name for r in rs if type(r).__name__ not in self._PHASE_RESPONSES]
        if bad:
            raise ValueError(f"with several phases only {', '.join(self._PHASE_RESPONSES)} are verified as objective "
                             f"or constraint; got {bad}")

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

    def evaluate_density(self, xphys: np.ndarray) -> DensityEvaluation:
        """Objective and constraint responses of the physical density ``xphys`` (domain.shape) without the chain.

        For a density the optimiser did not return (connectivity.bridge adds solid cells to x̄; there is no design
        vector x that maps to it). No gradients. A response that needs the chain or x raises inside (no chain or x
        in the context) and is reported in ``skipped`` instead of a value; a failing state solve raises.
        """
        xp = np.asarray(xphys, float)
        if xp.shape != tuple(self.domain.shape):
            raise ValueError(f"xphys has shape {xp.shape}, the domain {tuple(self.domain.shape)}")
        rs = [(self.objective.name, self.objective)] + [(c.key, c.response) for c in self.constraints]
        with torch.no_grad():
            xt = torch.as_tensor(xp, dtype=torch.float64)
            E = self.interp(xt)
            U = self.physics.solve(E.reshape(-1)[self._active], self.F) if "u" in self._needs(
                [r for _, r in rs]) else None
            ctx = EvalContext(self.domain, xt, E, U, self.F, physics=self.physics, interp=self.interp)
            values, skipped = {}, {}
            for key, r in rs:
                try:
                    values[key] = float(r(ctx))
                except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
                    skipped[key] = f"{type(e).__name__}: {e}"
        return DensityEvaluation(values, skipped, None if U is None else U.detach().numpy().copy())

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
        if self.n_fields > 1:
            return self._phase_start()
        x = None
        for c in self.constraints:
            if isinstance(c.response, Volume):
                x = self.domain.uniform_start(c.upper, weighted=bool(getattr(c.response, "weighted", False)))
                break
        if x is None:
            x = np.full(self.domain.n_design, 0.5)
        if self._reduce is not None:
            x = np.asarray(self._reduce(x), float)
        return x

    def _phase_start(self) -> np.ndarray:
        """Uniform start whose phase volume fractions are the bounds of the single-phase PhaseVolume constraints
        (other fields 0.5); with only a solid-volume bound V every field starts at τ = 1 − (1 − V)^(1/m), which
        gives solid fraction V (a field at 0 would stay there: d t^p / dt = 0 at t = 0)."""
        from toporlogy.design.phases import phase_start
        names = [p.name for p in getattr(self.interp, "phases", ())]
        fr: list[float | None] = [None] * self.n_fields
        solid = None
        for c in self.constraints:
            ph = getattr(c.response, "phases", "?")
            if type(c.response).__name__ != "PhaseVolume" or getattr(c.response, "weighted", False):
                continue
            if ph is None:
                solid = c.upper
            elif isinstance(ph, list) and len(ph) == 1 and ph[0] in names:
                fr[names.index(ph[0])] = c.upper
        if solid is not None and all(f is None for f in fr):
            tau = 1.0 - (1.0 - min(max(float(solid), 0.0), 1.0)) ** (1.0 / self.n_fields)
            return np.full(self.n_var, tau)
        return phase_start(self.n_fields, self.domain.n_design, fr)

    # ------------------------------------------------------------------ loop
    def run(self, x0: np.ndarray | None = None, max_iter: int = 100, tol: float = 0.01, feasibility_tol: float = 1e-3,
            callback: Callable[[int, Evaluation, float], None] | None = None, checkpoint: str | None = None,
            checkpoint_every: int = 0, resume: str | None = None) -> Result:
        """Optimise. ``checkpoint`` / ``checkpoint_every``: write the run state every k iterations (atomic);
        ``resume``: continue from such a file in a study built from the same configuration
        (:mod:`toporlogy.optimization.checkpoint`; the physics' own state, e.g. an adaptive cutoff, is not saved)."""
        threads = torch.get_num_threads()
        if self.torch_threads is not None:
            torch.set_num_threads(self.torch_threads)
        try:
            return self._run(x0, max_iter, tol, feasibility_tol, callback, checkpoint, int(checkpoint_every), resume)
        finally:
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)

    def _run(self, x0: np.ndarray | None, max_iter: int, tol: float, feasibility_tol: float,
             callback: Callable[[int, Evaluation, float], None] | None, checkpoint: str | None = None,
             checkpoint_every: int = 0, resume: str | None = None) -> Result:
        t0 = time.perf_counter()
        # a run starts from the initial continuation state and a fresh optimizer, whatever a previous run left
        # (β, its counter, an updated volume bound, MMA asymptotes and iterates); before default_start, which
        # reads the volume bound
        self.schedule.reset()
        self.optimizer.restart()
        for r in [self.objective, *[c.response for c in self.constraints]]:
            _reset_scales(r)                                   # scales frozen at a first evaluation are taken again
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
        epoch = getattr(self.physics, "model_epoch", None)
        if epoch is not None:
            hist["model_epoch"] = []
        self.n_model_resets = 0
        if self.recorder is not None:
            self.recorder.start(self)
        it, change, ev = 0, 1.0, None
        termination, stepped_last = "max_iter", False
        if resume is not None:
            from toporlogy.optimization.checkpoint import load_checkpoint
            state = load_checkpoint(resume, self)
            x, it, change, stepped_last = np.asarray(state["x"], float).copy(), int(state["it"]), \
                float(state["change"]), bool(state["stepped_last"])
            hist = state["hist"]
            epoch, self.n_model_resets = state["epoch"], int(state["n_model_resets"])

        def save() -> None:
            if checkpoint is not None and checkpoint_every > 0 and it % checkpoint_every == 0:
                from toporlogy.optimization.checkpoint import save_checkpoint
                vf = getattr(self.optimizer, "value_fn", None)
                save_checkpoint(checkpoint, self, {"x": x, "it": it, "change": change, "stepped_last": stepped_last,
                                                   "hist": hist, "epoch": epoch,
                                                   "n_model_resets": self.n_model_resets, "move0": move0})
                if vf is not None:
                    self.optimizer.value_fn = vf
        move0 = getattr(self.optimizer, "move", None)
        if resume is not None and "move0" in state:
            move0 = state["move0"]
        if getattr(self.optimizer, "needs_value_fn", False):
            def value_fn(xc: np.ndarray) -> tuple[float, np.ndarray]:
                e = self.evaluate(xc, gradients=False)
                return e.f0, e.g
            self.optimizer.value_fn = value_fn
        try:
            while it < max_iter:
                it += 1
                for k, v in self.schedule.state().items():
                    hist[k].append(v)
                ev = self.evaluate(x)
                if epoch is not None:
                    now = self.physics.model_epoch
                    hist["model_epoch"].append(float(now))
                    if self.reset_on_model_change and now != epoch:
                        self.optimizer.reset()
                        self.n_model_resets += 1
                    epoch = now
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
                stepped_last = self.schedule.update(it, change)
                if stepped_last:
                    self.optimizer.reset()
                    beta = self.schedule.state().get("beta")
                    if self.move_beta_ref is not None and move0 is not None and beta:
                        cast(_HasMove, self.optimizer).move = float(move0) * min(1.0, self.move_beta_ref / float(beta))
                    save()
                    continue
                save()
                if change <= tol and self.schedule.finished:
                    termination = "design_change"
                    break
            # The loop analysed x_k and then stepped to x_{k+1}: re-evaluate the RETURNED design so that its objective,
            # constraints, U and x̄ belong to the same design (history[-1] belongs to the previous one).
            if termination != "design_change" and stepped_last and it == max_iter:
                termination = "schedule_step_at_max_iter"
            mult = self.optimizer.kkt_multipliers() if hasattr(self.optimizer, "kkt_multipliers") else None
            lam = np.zeros(0) if mult is None else np.asarray(mult, float).reshape(-1)
            with_kkt = len(self.constraints) > 0 and lam.size == len(self.constraints)
            final = self.evaluate(x, gradients=len(self.constraints) > 0)
            feasible, worst = True, 0.0
            for c in self.constraints:
                v, scale = final.values[c.key], max(abs(c.upper), 1e-12)
                worst = max(worst, (v - c.upper) / scale)
                feasible = feasible and bool(v <= c.upper + feasibility_tol * scale)
            kkt = self._kkt_residual(x, final, lam) if with_kkt else None
            stat = (kkt_estimate(x, final.df0, final.g, final.dg)["residual"] if len(self.constraints) > 0
                    else None)
            res = Result(x, final.xphys, hist, it, termination == "design_change", time.perf_counter() - t0,
                         final.U, self.settings(), final, termination=termination, feasible=feasible,
                         max_violation=max(worst, 0.0), feasibility_tol=float(feasibility_tol), kkt_residual=kkt,
                         stationarity=stat)
        finally:
            if getattr(self.optimizer, "needs_value_fn", False):
                self.optimizer.value_fn = None
            if move0 is not None:
                cast(_HasMove, self.optimizer).move = move0     # leave the optimizer as it was given, also on an error
        if self.recorder is not None:
            self.recorder.finish(res)
        return res

    def _kkt_residual(self, x: np.ndarray, ev: Evaluation, lam: np.ndarray) -> float:
        """See ``Result.kkt_residual``: objective scale and multipliers as the optimizer reports them, constraint
        gradients those of ``Constraint.g`` (already scaled), as passed to ``Optimizer.step``."""
        s0 = float(self.optimizer.objective_scale(ev.f0)) if hasattr(self.optimizer, "objective_scale") else 1.0
        gl = s0 * ev.df0 + lam @ ev.dg
        r = np.where(x <= 1e-9, np.minimum(gl, 0.0), np.where(x >= 1.0 - 1e-9, np.maximum(gl, 0.0), gl))
        stat = float(np.max(np.abs(r))) / max(float(np.max(np.abs(s0 * ev.df0))), 1e-300) if r.size else 0.0
        comp = float(np.max(np.abs(lam * ev.g))) if lam.size else 0.0
        return max(stat, comp)

    def settings(self) -> dict[str, Any]:
        s: dict[str, Any] = {"domain": repr(self.domain), "optimizer": self.optimizer.name,
                             "objective": self.objective.name,
                             "constraints": [(c.key, c.upper) for c in self.constraints],
                             "constraint_keys": [c.key for c in self.constraints],
                             "links": [type(m).__name__ for m in self.chain.links],
                             "interp": type(self.interp).__name__}
        if self.n_fields > 1:
            s["phases"] = [p.name for p in getattr(self.interp, "phases", ())]
        for k in ("element", "nu"):
            if hasattr(self.physics, k):
                s[k] = getattr(self.physics, k)
        if hasattr(self.physics, "solver"):
            s["solver"] = getattr(self.physics.solver, "name", type(self.physics.solver).__name__)
        if hasattr(self.optimizer, "n_inner_total"):             # GCMMA / SiMPL: evaluations inside the steps
            s["inner_evaluations"] = int(self.optimizer.n_inner_total)
        if hasattr(self.optimizer, "subproblem_failures"):       # MMA: dual solves that did not converge this run
            s["subproblem_failures"] = int(self.optimizer.subproblem_failures)
        s["torch_threads"] = self.torch_threads
        if hasattr(self.physics, "model_epoch"):
            s["reset_on_model_change"] = self.reset_on_model_change
        return s

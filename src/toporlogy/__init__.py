"""toporlogy — modular density-based topology optimisation on regular grids (docs/mimari_plan.md).

Layers: :class:`Domain` → :class:`Chain` (design → physical density) → interpolation (:class:`SIMP`) →
physics (:class:`Elasticity`, autograd-aware linear solve) → responses (:class:`Compliance`, :class:`Volume`) →
optimizer (:class:`OC`, :class:`MMA`) → :class:`Study` loop → :class:`Recorder`.

CPU direct solver (PARDISO / SuperLU); the older drivers are used only as regression oracles in ``tests/toporlogy``.

Import order (measured 13 Nov 2026, Windows): torch and MKL PARDISO each ship ``libiomp5md.dll``. Loading MKL after
torch aborts the process ("OMP: Error #15"); loading MKL first works. This package therefore loads PARDISO before it
imports torch. If torch was imported earlier by the caller, PARDISO is not attempted and
:class:`~toporlogy.solvers.DirectSolver` falls back to SuperLU (:mod:`toporlogy.solvers.pardiso`).
"""
from toporlogy.solvers import pardiso as _pardiso

_pardiso.load()

from toporlogy.core.domain import Domain
from toporlogy.design.chain import Chain, DensityFilter, Embed, HeavisideProjection, Passive, standard_chain
from toporlogy.design.interp import RAMP, SIMP
from toporlogy.fem import Elasticity
from toporlogy.optimization.responses import Compliance, EvalContext, Response, Volume
from toporlogy.optimizers import MMA, OC, BetaContinuation, Optimizer, Schedule
from toporlogy.optimization.study import Constraint, Evaluation, Result, Study
from toporlogy.optimization.diagnostics import grey_fraction, gradcheck, study_gradcheck
from toporlogy.execution.record import Recorder

# Feature modules are imported from their own packages, e.g. ``from toporlogy.boundary import Fixed``,
# ``from toporlogy.optimization.connectivity import one_body``; only the core names above are flattened here.

__all__ = ["Domain", "Chain", "Embed", "DensityFilter", "HeavisideProjection", "Passive", "standard_chain",
           "SIMP", "RAMP", "Elasticity", "Response", "EvalContext", "Compliance", "Volume", "Optimizer", "OC", "MMA",
           "Schedule", "BetaContinuation", "Constraint", "Study", "Result", "Evaluation", "Recorder",
           "gradcheck", "study_gradcheck", "grey_fraction"]

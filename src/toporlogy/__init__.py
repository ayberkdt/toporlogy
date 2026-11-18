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

from toporlogy.design.chain import Chain, DensityFilter, Embed, HeavisideProjection, Passive, standard_chain
from toporlogy.optimization.checks import grey_fraction, gradcheck, study_gradcheck
from toporlogy.core.domain import Domain
from toporlogy.fem import Elasticity
from toporlogy.design.interp import RAMP, SIMP
from toporlogy.optim import MMA, OC, BetaContinuation, Optimizer, Schedule
from toporlogy.record import Recorder
from toporlogy.optimization.responses import Compliance, EvalContext, Response, Volume
from toporlogy.optimization.study import Constraint, Evaluation, Result, Study

# Feature modules (import as toporlogy.<module>; their classes are not flattened into this namespace):
# bc (supports, loads, checks), geometry (CSG / STL domains, bores), lengthscale (min/max thickness, measurement),
# robust (eroded/blueprint/dilated), connectivity (one-body gate, prevention, repair), manufacturing (AM, symmetry,
# extrusion, casting), dynamics (eigenfrequency responses), fem.heat, fem.modal, config / registry / cli.
from toporlogy import bc, boundary, connectivity, dynamics, geometry, lengthscale, manufacturing, robust  # noqa: F401

# Old module names that were package attributes before the G5 moves (docs/gelistirme_plani.md): importing their aliases
# keeps `import toporlogy; toporlogy.study.Study` working (tests/toporlogy/test_api_surface.py).
from toporlogy import chain, checks, domain, interp, responses, study  # noqa: E402, F401

__all__ = ["Domain", "Chain", "Embed", "DensityFilter", "HeavisideProjection", "Passive", "standard_chain",
           "SIMP", "RAMP", "Elasticity", "Response", "EvalContext", "Compliance", "Volume", "Optimizer", "OC", "MMA",
           "Schedule", "BetaContinuation", "Constraint", "Study", "Result", "Evaluation", "Recorder",
           "gradcheck", "study_gradcheck", "grey_fraction"]

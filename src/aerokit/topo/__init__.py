"""aerokit.topo — modular density-based topology optimisation on regular grids (docs/mimari_plan.md).

Layers: :class:`Domain` → :class:`Chain` (design → physical density) → interpolation (:class:`SIMP`) →
physics (:class:`Elasticity`, autograd-aware linear solve) → responses (:class:`Compliance`, :class:`Volume`) →
optimizer (:class:`OC`, :class:`MMA`) → :class:`Study` loop → :class:`Recorder`.

CPU direct solver (PARDISO / SuperLU); the older drivers are used only as regression oracles in ``tests/topo``.

Import order (measured 13 Nov 2026, Windows): torch and MKL PARDISO each ship ``libiomp5md.dll``. Loading MKL after
torch aborts the process ("OMP: Error #15"); loading MKL first works. This package therefore loads PARDISO before it
imports torch. If torch was imported earlier by the caller, PARDISO is not attempted and
:class:`~aerokit.topo.solvers.DirectSolver` falls back to SuperLU (:mod:`aerokit.topo.solvers.pardiso`).
"""
from aerokit.topo.solvers import pardiso as _pardiso

_pardiso.load()

from aerokit.topo.chain import Chain, DensityFilter, Embed, HeavisideProjection, Passive, standard_chain
from aerokit.topo.checks import grey_fraction, gradcheck, study_gradcheck
from aerokit.topo.domain import Domain
from aerokit.topo.fem import Elasticity
from aerokit.topo.interp import RAMP, SIMP
from aerokit.topo.optim import MMA, OC, BetaContinuation, Optimizer, Schedule
from aerokit.topo.record import Recorder
from aerokit.topo.responses import Compliance, EvalContext, Response, Volume
from aerokit.topo.study import Constraint, Evaluation, Result, Study

# Feature modules (import as aerokit.topo.<module>; their classes are not flattened into this namespace):
# bc (supports, loads, checks), geometry (CSG / STL domains, bores), lengthscale (min/max thickness, measurement),
# robust (eroded/blueprint/dilated), connectivity (one-body gate, prevention, repair), manufacturing (AM, symmetry,
# extrusion, casting), dynamics (eigenfrequency responses), fem.heat, fem.modal, config / registry / cli.
from aerokit.topo import bc, connectivity, dynamics, geometry, lengthscale, manufacturing, robust  # noqa: F401

__all__ = ["Domain", "Chain", "Embed", "DensityFilter", "HeavisideProjection", "Passive", "standard_chain",
           "SIMP", "RAMP", "Elasticity", "Response", "EvalContext", "Compliance", "Volume", "Optimizer", "OC", "MMA",
           "Schedule", "BetaContinuation", "Constraint", "Study", "Result", "Evaluation", "Recorder",
           "gradcheck", "study_gradcheck", "grey_fraction"]

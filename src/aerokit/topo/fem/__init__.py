"""Physics on the regular grid: linear elasticity (direct or multigrid solver), heat conduction, modal analysis."""
from typing import Any

from aerokit.topo.fem.elastic import Elasticity, element_matrix

__all__ = ["Elasticity", "ElasticityGMG", "element_matrix"]


def __getattr__(name: str) -> Any:
    if name == "ElasticityGMG":
        from aerokit.topo.fem.elastic_gmg import ElasticityGMG
        return ElasticityGMG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

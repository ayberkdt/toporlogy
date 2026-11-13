"""Physics on the regular grid (linear elasticity now; heat conduction later)."""
from aerokit.topo.fem.elastic import Elasticity, element_matrix

__all__ = ["Elasticity", "element_matrix"]

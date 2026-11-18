"""Declarative configuration: the component registry (``registry``), the component constructors (``components``),
shared helpers, the post steps and the boundary-condition mapping for re-analysis (docs/registry.md,
docs/gelistirme_plani.md G2). Importing the package registers every built-in component once.
"""
from __future__ import annotations

from toporlogy.configuration.components import register_all
from toporlogy.configuration.registry import REGISTRY

register_all()

__all__ = ["REGISTRY"]

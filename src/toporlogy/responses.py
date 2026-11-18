"""Compatibility alias (docs/gelistirme_plani.md G5).

Old name:   ``toporlogy.responses``
New module: ``toporlogy.optimization.responses``

The old name refers to the same module object, so every name, private ones and module state included, is shared.
"""
import sys

from toporlogy.optimization import responses as _module

sys.modules[__name__] = _module

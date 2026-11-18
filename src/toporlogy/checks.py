"""Compatibility alias (docs/gelistirme_plani.md G5).

Old name:   ``toporlogy.checks``
New module: ``toporlogy.optimization.checks``

The old name refers to the same module object, so every name, private ones and module state included, is shared.
"""
import sys

from toporlogy.optimization import checks as _module

sys.modules[__name__] = _module

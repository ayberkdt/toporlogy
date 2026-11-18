"""Compatibility alias (docs/gelistirme_plani.md G5).

Old name:   ``toporlogy.robust``
New module: ``toporlogy.optimization.robust``

The old name refers to the same module object, so every name, private ones and module state included, is shared.
"""
import sys

from toporlogy.optimization import robust as _module

sys.modules[__name__] = _module

"""Compatibility alias (docs/gelistirme_plani.md G5).

Old name:   ``toporlogy.manufacturing``
New module: ``toporlogy.design.manufacturing``

The old name refers to the same module object, so every name, private ones and module state included, is shared.
"""
import sys

from toporlogy.design import manufacturing as _module

sys.modules[__name__] = _module

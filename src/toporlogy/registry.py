"""Compatibility module: the component registry moved to :mod:`toporlogy.configuration` (docs/gelistirme_plani.md
G2). Every name that was importable from ``toporlogy.registry`` still is, and refers to the same object; importing it
registers the built-in components as before. New code imports from ``toporlogy.configuration.*``.

Where things are now: mechanism (``Registry``, ``Param``, ``Entry``, ``BuildContext``, ``REGISTRY``, ``register``)
in ``configuration.registry``; shared helpers in ``configuration.helpers``; constructors in
``configuration.components.*``; post steps in ``configuration.post_steps``; ``bc_mapping`` / ``bc_callables`` /
``selector_callable`` in ``configuration.boundary_mapping``.
"""
from __future__ import annotations

from typing import Any

import toporlogy.configuration  # noqa: F401  (registers the built-in components)
from toporlogy.configuration import post_steps as _post_steps
from toporlogy.configuration.registry import (  # noqa: F401
    BuildContext, ConfigError, Entry, KINDS, Param, REGISTRY, REQUIRED, Registry, _FALSE, _Required, _TRUE,
    _TYPE_NAMES, _annotation_type, _param_from_signature, _plain, _show, get, join, names, normalise_name,
    register, suggest,
)
from toporlogy.configuration.helpers import (  # noqa: F401
    SpreadForce, _axis, _bores, _bounds, _case_weights, _combine, _coord, _describe_bores, _design, _directions,
    _field_source, _interfaces, _is_cylinder, _nonempty, _point, _post_state, _reevaluate, _region_mask, _rel,
    _robust_set, _vector, as_selector, checked_selector, find_bore,
)
from toporlogy.configuration.boundary_mapping import (  # noqa: F401
    bc_callables, bc_mapping, selector_callable,
)
from toporlogy.configuration.post_steps import (  # noqa: F401
    _LAZY, _gate_dict, _make_checks, _make_finalize, _need_result, _post_shape, _roles, _signature_with,
    export_step, reverify_step,
)

__all__ = ["KINDS", "REQUIRED", "ConfigError", "Param", "Entry", "BuildContext", "Registry", "REGISTRY", "register",
           "get", "names", "normalise_name", "suggest", "as_selector", "checked_selector", "find_bore", "SpreadForce",
           "bc_callables", "bc_mapping", "selector_callable"]


def __getattr__(name: str) -> Any:                     # PEP 562: finalize_step / checks_step stay lazy
    return getattr(_post_steps, name)

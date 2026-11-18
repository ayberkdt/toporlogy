"""Compatibility module: ``toporlogy.config`` moved (docs/gelistirme_plani.md G3). Every name that was importable
from it still is and refers to the same object; new code imports from the modules below.

* ``toporlogy.configuration.schema``  — the config schema (its docstring is the reference), defaults, ``STATUS_OK``
* ``toporlogy.configuration.resolve`` — ``read_config``, ``resolve``, ``apply_overrides``, ``parse_sweep`` …
* ``toporlogy.configuration.build``   — ``Setup``, ``build``, ``build_setup``, ``study_from_dict``, ``load_config``
* ``toporlogy.execution.runner``      — ``run_setup``
* ``toporlogy.execution.runs``        — ``write_resolved``, ``status_of``, ``write_sweep_csv``, ``read_run``
"""
from __future__ import annotations

from toporlogy.configuration.registry import (  # noqa: F401  (config used to re-export these)
    REGISTRY, BuildContext, ConfigError, Param, join, normalise_name, suggest,
)
from toporlogy.configuration.schema import (  # noqa: F401
    SCHEMA_KEYS, STATUS_OK, _CASE, _COMBINATION, _CONNECTIVITY, _CONSTRAINT, _DEFAULTS, _MATERIAL,
    _NEEDS_FINALIZE, _OUTPUT, _REQUIRED, _ROBUST, _RUN, _TYING,
)
from toporlogy.configuration.resolve import (  # noqa: F401
    _h_dim, _list, _parse_value, _resolve_connectivity, _resolve_robust, _robust_settings, _section,
    _split_assignment, _split_values, _type_of, apply_overrides, combination_label, parse_sweep, read_config,
    resolve, set_path, sweep_combinations,
)
from toporlogy.configuration.build import (  # noqa: F401
    Setup, _as_load, _as_supports, _boundary_conditions, _check_bc, _default_interfaces, _explicit_interfaces,
    _loads, _resultant, _supports, build, build_setup, load_config, study_from_dict,
)
from toporlogy.execution.runs import (  # noqa: F401
    read_run, status_of, write_resolved, write_sweep_csv,
)
from toporlogy.execution.runner import (  # noqa: F401
    run_setup,
)

__all__ = ["ConfigError", "Setup", "SCHEMA_KEYS", "read_config", "resolve", "build", "build_setup",
           "study_from_dict", "load_config", "apply_overrides", "parse_sweep", "sweep_combinations", "run_setup",
           "write_resolved", "read_run", "status_of", "STATUS_OK"]

"""Component constructors, one module per area. ``register_all`` registers them in the order the single
registry module used (core, geometry, boundary, design, physics, responses, schedules, post): ``python -m toporlogy
list`` and the did-you-mean suggestions depend on that order.
"""
from __future__ import annotations

_DONE = False


def register_all() -> None:
    global _DONE
    if _DONE:
        return
    from toporlogy.configuration.components.boundary import register_boundary
    from toporlogy.configuration.components.core import register_core
    from toporlogy.configuration.components.design import register_design
    from toporlogy.configuration.components.geometry import register_geometry
    from toporlogy.configuration.components.physics import register_physics
    from toporlogy.configuration.components.responses import register_responses
    from toporlogy.configuration.components.schedules import register_schedules
    from toporlogy.configuration.post_steps import register_post_steps
    for fn in (register_core, register_geometry, register_boundary, register_design, register_physics,
               register_responses, register_schedules, register_post_steps):
        fn()
    _DONE = True

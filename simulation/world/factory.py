"""Create the continuous cooperative platform backend."""

from __future__ import annotations

from typing import Any

from simulation.world.environment import Environment
from simulation.world.level_builder import build_level


def create_environment(config: Any, agents: list[Any], simulation: Any) -> Environment:
    level = build_level(config.virtual_level, agents, level_number=1)
    environment = Environment(
        level,
        agents,
        simulation=simulation,
        level_type=config.virtual_level,
    )
    environment.level_type = config.virtual_level
    environment.level_name = level.title
    return environment

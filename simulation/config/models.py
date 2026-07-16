"""Typed configuration for interactive and multi-run ACT-R platform simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SPEED_PRESETS: tuple[tuple[str, float], ...] = (
    ("1/4 Realtime", 25.0),
    ("1/2 Realtime", 50.0),
    ("Realtime", 100.0),
    ("2x Realtime", 200.0),
    ("ASAP", -1.0),
)

ENVIRONMENT_MODES: tuple[tuple[str, str], ...] = (
    ("Continuous Physics", "virtual"),
)

VIRTUAL_LEVELS: tuple[tuple[str, str], ...] = (
    ("Procedural Cooperative Platforms", "cooperative_platforms"),
)


@dataclass(slots=True)
class AgentTypeConfig:
    count: int = 1
    print_agent_actions: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentTypeConfig":
        payload = payload or {}
        return cls(
            count=max(0, int(payload.get("count", 1))),
            print_agent_actions=bool(payload.get("print_agent_actions", True)),
        )


@dataclass(slots=True)
class SimulationConfig:
    focus_position: tuple[int, int] = (0, 0)
    print_middleman: bool = False
    speed_factor: float = 100.0
    print_agent_actions: bool = True
    experimental_pyactr_performance_boost: bool = False
    los: int = 32
    execution_mode: str = "single"
    environment_mode: str = "virtual"
    virtual_level: str = "cooperative_platforms"
    human_agent_enabled: bool = False
    human_agent_name: str = "Human Player"
    agent_type_config: dict[str, AgentTypeConfig] = field(
        default_factory=lambda: {
            "Example": AgentTypeConfig(count=2, print_agent_actions=True),
        }
    )

    @property
    def height(self) -> int:
        from simulation.world.level_builder import level_dimensions
        return level_dimensions(self.virtual_level)[0]

    @property
    def width(self) -> int:
        from simulation.world.level_builder import level_dimensions
        return level_dimensions(self.virtual_level)[1]

    @property
    def stepper(self) -> bool:
        return self.execution_mode == "single"

    @property
    def speed_label(self) -> str:
        for label, value in SPEED_PRESETS:
            if float(self.speed_factor) == float(value):
                return label
        return f"{self.speed_factor:g}%"

    @property
    def environment_label(self) -> str:
        labels = {value: label for label, value in ENVIRONMENT_MODES}
        return labels.get(self.environment_mode, self.environment_mode)

    @property
    def actr_agent_count(self) -> int:
        return sum(max(0, item.count) for item in self.agent_type_config.values())

    @property
    def spatial_agent_count(self) -> int:
        return self.actr_agent_count + int(self.human_agent_enabled)

    def validate(self) -> None:
        if float(self.speed_factor) not in {value for _, value in SPEED_PRESETS}:
            raise ValueError("The speed must use one of the predefined presets.")
        if self.los < 0:
            raise ValueError("The perception radius cannot be negative.")
        if self.execution_mode not in {"single", "automatic"}:
            raise ValueError("Unknown execution mode.")
        if self.environment_mode != "virtual":
            raise ValueError("This simulation supports only the continuous virtual backend.")
        if self.virtual_level not in {value for _, value in VIRTUAL_LEVELS}:
            raise ValueError("Unknown virtual level.")
        if self.human_agent_enabled and not self.human_agent_name.strip():
            raise ValueError("The human agent needs a name.")
        if self.actr_agent_count < 1:
            raise ValueError("At least one ACT-R agent must be enabled.")
        if self.spatial_agent_count != 2:
            raise ValueError(
                "The cooperative game requires exactly two agents in total. "
                "Use two ACT-R agents, or one ACT-R agent plus the human-controlled agent."
            )

    def without_human_agent(self) -> "SimulationConfig":
        """Replace a human player with a second ACT-R instance for batch runs."""
        payload = self.to_dict()
        payload["human_agent_enabled"] = False
        payload["human_agent_name"] = "Human Player"
        config = type(self).from_dict(payload)
        if config.actr_agent_count < 2:
            first_name = next(iter(config.agent_type_config), "Example")
            current = config.agent_type_config.setdefault(first_name, AgentTypeConfig(count=0))
            current.count += 2 - config.actr_agent_count
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_position": list(self.focus_position),
            "print_middleman": self.print_middleman,
            "width": self.width,
            "height": self.height,
            "speed_factor": self.speed_factor,
            "speed_label": self.speed_label,
            "print_agent_actions": self.print_agent_actions,
            "experimental_pyactr_performance_boost": self.experimental_pyactr_performance_boost,
            "los": self.los,
            "execution_mode": self.execution_mode,
            "stepper": self.stepper,
            "environment_mode": self.environment_mode,
            "environment_label": self.environment_label,
            "virtual_level": self.virtual_level,
            "human_agent_enabled": self.human_agent_enabled,
            "human_agent_name": self.human_agent_name,
            "agent_type_config": {
                name: config.to_dict()
                for name, config in self.agent_type_config.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SimulationConfig":
        payload = payload or {}
        focus = payload.get("focus_position", [0, 0])
        try:
            focus_position = (int(focus[0]), int(focus[1]))
        except Exception:
            focus_position = (0, 0)

        raw_agents = payload.get("agent_type_config", {})
        agent_types = {
            str(name): AgentTypeConfig.from_dict(value)
            for name, value in raw_agents.items()
            if isinstance(value, dict) and str(name).strip()
        }
        if not agent_types:
            agent_types = {"Example": AgentTypeConfig(count=2)}

        speed = float(payload.get("speed_factor", 100.0))
        if speed not in {value for _, value in SPEED_PRESETS}:
            speed = 100.0
        execution_mode = str(payload.get("execution_mode", "single"))
        if execution_mode not in {"single", "automatic"}:
            execution_mode = "single"

        human_enabled = bool(payload.get("human_agent_enabled", False))
        config = cls(
            focus_position=focus_position,
            print_middleman=bool(payload.get("print_middleman", False)),
            speed_factor=speed,
            print_agent_actions=bool(payload.get("print_agent_actions", True)),
            experimental_pyactr_performance_boost=bool(
                payload.get("experimental_pyactr_performance_boost", False)
            ),
            los=max(0, int(payload.get("los", 32))),
            execution_mode=execution_mode,
            environment_mode="virtual",
            virtual_level="cooperative_platforms",
            human_agent_enabled=human_enabled,
            human_agent_name=str(payload.get("human_agent_name", "Human Player")).strip() or "Human Player",
            agent_type_config=agent_types,
        )

        # Migrate old demo settings to the fixed two-player world instead of
        # making the application fail immediately after an upgrade.
        if config.spatial_agent_count != 2:
            positive_names = [name for name, row in config.agent_type_config.items() if row.count > 0]
            first_name = positive_names[0] if positive_names else next(iter(config.agent_type_config), "Example")
            for row in config.agent_type_config.values():
                row.count = 0
            config.agent_type_config.setdefault(first_name, AgentTypeConfig(count=0)).count = 1 if human_enabled else 2
        return config

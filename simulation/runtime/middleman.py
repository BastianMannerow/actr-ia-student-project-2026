"""Communication layer between ACT-R agents and the continuous world."""

from __future__ import annotations

import math
from typing import Any

from simulation.runtime.agent_construct import AgentConstruct
from simulation.world.entities import AvatarShape, Diamond, Platform, SpatialAgent


class Middleman:
    """Translate continuous world state into pyactr-compatible point stimuli."""

    POSITION_SCALE = 4.0  # quarter-world-unit visual coordinates
    INSPECTOR_COLUMNS = 25
    INSPECTOR_ROWS = 17

    def __init__(self, simulation: Any, print_middleman: bool):
        self.simulation = simulation
        self.experiment_environment = None
        self.print_middleman = print_middleman

    def set_game_environment(self, experiment_environment: Any) -> None:
        self.experiment_environment = experiment_environment

    def motor_input(self, key: str, current_agent: AgentConstruct) -> bool:
        if self.experiment_environment is None:
            return False
        moved = bool(self.experiment_environment.apply_action(current_agent, key))
        if self.print_middleman:
            shape = getattr(getattr(current_agent, "avatar_shape", None), "value", "unknown")
            print(
                f"{current_agent.name} [{shape}]: motor {key} -> "
                f"{'accepted' if moved else 'blocked'}"
            )
        return moved

    def get_agent_stimulus(self, agent: AgentConstruct):
        """Build a precise point projection without pretending the world is a matrix.

        Positions are quantized to quarter world units because pyactr visual
        positions must be compact numeric coordinates. Large platforms are
        represented by top and bottom edge samples, preserving their complete
        collision envelope while keeping the visual frame compact and deterministic.
        """
        environment = self.experiment_environment
        if environment is None or environment.find_agent(agent) is None:
            return [set()], [{}]

        body = agent.body
        configured_los = float(getattr(agent, "los", 0))
        los = 0.0 if bool(getattr(agent, "perfect_line_of_sight", False)) else configured_los
        records = environment.perceptual_objects(agent, los)
        agent_symbols = getattr(agent, "_agent_symbol_by_identity", {})
        trigger_symbols: set[str] = set()
        frame: dict[str, dict[str, Any]] = {}
        metadata: dict[str, dict[str, Any]] = {}

        # The inspector is a local raster projection only; the backend and ACT-R
        # coordinates remain continuous/quantized rather than cell based.
        inspector = [
            ["-" for _ in range(self.INSPECTOR_COLUMNS)]
            for _ in range(self.INSPECTOR_ROWS)
        ]
        view_radius_x = los if los > 0 else environment.width / 2.0
        view_radius_y = min(view_radius_x, environment.height / 2.0)
        left = body.x - view_radius_x
        top = body.y - view_radius_y

        for record_index, record in enumerate(records):
            entity = record["entity"]
            x = float(record["x"])
            y = float(record["y"])
            qx = int(round(x * self.POSITION_SCALE))
            qy = int(round(y * self.POSITION_SCALE))
            kind = str(record["kind"])
            symbol = self._symbol_for_record(record, agent_symbols)
            stimulus_id = self._stimulus_id(kind, entity, record, record_index, agent_symbols)
            frame[stimulus_id] = {"text": symbol, "position": (qx, qy)}
            trigger_symbols.add(symbol)

            metadata[stimulus_id] = {
                "entity_class": type(entity).__name__,
                "display_name": str(
                    getattr(entity, "display_name", getattr(entity, "name", type(entity).__name__))
                ),
                "world_position": (x, y),
                "quantized_position": (qx, qy),
                "kind": kind,
                "blocks_movement": bool(getattr(entity, "blocks_movement", False)),
                "is_target": bool(getattr(entity, "is_target", False)),
            }
            metadata[stimulus_id].update(
                {key: value for key, value in record.items() if key not in {"entity"}}
            )
            self._plot_inspector_symbol(
                inspector,
                symbol,
                x=x,
                y=y,
                left=left,
                top=top,
                width=max(0.1, 2.0 * view_radius_x),
                height=max(0.1, 2.0 * view_radius_y),
            )

            # Identity, shape, posture, and visible body bounds are separate
            # pyactr stimuli.  The adapter can therefore reconstruct geometry
            # from AgentConstruct.stimuli without reading the world backend or
            # the richer visual_metadata side channel.
            if kind == "agent":
                agent_symbol = str(
                    agent_symbols.get(id(entity), getattr(entity, "symbol", "A"))
                )
                shape_name = str(entity.avatar_shape.value)
                shape_symbol = "C" if entity.avatar_shape is AvatarShape.CIRCLE else "R"
                shape_id = f"agent__{agent_symbol}__shape__{shape_name}"
                frame[shape_id] = {"text": shape_symbol, "position": (qx, qy)}
                trigger_symbols.add(shape_symbol)
                metadata[shape_id] = {
                    **metadata[stimulus_id],
                    "perceptual_fact": "avatar_shape",
                    "shape": shape_name,
                }

                body_points = {
                    "left": (entity.body.left, entity.body.y),
                    "right": (entity.body.right, entity.body.y),
                    "top": (entity.body.x, entity.body.top),
                    "bottom": (entity.body.x, entity.body.bottom),
                }
                for edge_name, (edge_x, edge_y) in body_points.items():
                    edge_qx = int(round(float(edge_x) * self.POSITION_SCALE))
                    edge_qy = int(round(float(edge_y) * self.POSITION_SCALE))
                    edge_id = f"agent__{agent_symbol}__bound__{edge_name}"
                    frame[edge_id] = {
                        "text": agent_symbol,
                        "position": (edge_qx, edge_qy),
                    }
                    metadata[edge_id] = {
                        **metadata[stimulus_id],
                        "perceptual_fact": "body_bound",
                        "edge": edge_name,
                        "world_position": (float(edge_x), float(edge_y)),
                        "quantized_position": (edge_qx, edge_qy),
                    }

                if entity.avatar_shape is AvatarShape.RECTANGLE:
                    max_height_value = int(round(float(entity.body.effective_max_height) * self.POSITION_SCALE))
                    capability_id = (
                        f"agent__{agent_symbol}__capability__max_height__"
                        f"{max_height_value}"
                    )
                    frame[capability_id] = {
                        "text": "H",
                        "position": (qx, qy),
                    }
                    trigger_symbols.add("H")
                    metadata[capability_id] = {
                        **metadata[stimulus_id],
                        "perceptual_fact": "rectangle_max_height",
                        "max_height": float(entity.body.effective_max_height),
                    }
                    posture = (
                        "tall" if entity.body.transform >= 0.82
                        else "flat" if entity.body.transform <= 0.15
                        else "intermediate"
                    )
                    posture_symbol = {
                        "tall": "T",
                        "flat": "F",
                        "intermediate": "M",
                    }[posture]
                    posture_id = f"agent__{agent_symbol}__posture__{posture}"
                    frame[posture_id] = {
                        "text": posture_symbol,
                        "position": (qx, qy),
                    }
                    trigger_symbols.add(posture_symbol)
                    metadata[posture_id] = {
                        **metadata[stimulus_id],
                        "perceptual_fact": "rectangle_posture",
                        "posture": posture,
                    }

        agent.visual_stimuli = inspector
        agent.visual_metadata = metadata
        agent.visual_frame_origin = (top, left)
        agent.visual_frame_valid_positions = {
            tuple(value["quantized_position"])
            for value in metadata.values()
            if "quantized_position" in value
        }
        return [trigger_symbols], [frame]

    @staticmethod
    def _symbol_for_record(record: dict[str, Any], agent_symbols: dict[int, str]) -> str:
        entity = record["entity"]
        kind = record["kind"]
        if kind == "agent":
            return str(agent_symbols.get(id(entity), getattr(entity, "symbol", "A")))
        if isinstance(entity, Diamond):
            return "D"
        if isinstance(entity, Platform):
            return "P"
        return str(getattr(entity, "symbol", "?"))

    @staticmethod
    def _stimulus_id(
        kind: str,
        entity: Any,
        record: dict[str, Any],
        record_index: int,
        agent_symbols: dict[int, str],
    ) -> str:
        """Return a stable semantic identifier that is itself visual-frame data."""
        if kind == "agent":
            symbol = str(
                agent_symbols.get(id(entity), getattr(entity, "symbol", "A"))
            )
            return f"agent__{symbol}__body"
        if kind == "diamond":
            identifier = str(getattr(entity, "diamond_id", "") or record_index)
            role = str(record.get("role", getattr(entity, "role", "solo")))
            order = int(getattr(entity, "required_order", 0))
            return (
                f"diamond__{identifier}__role__{role}__order__{order}"
            )
        if kind == "platform":
            platform_kind = str(getattr(entity, "kind", "platform"))
            return (
                f"platform__{record.get('platform_index', record_index)}__kind__"
                f"{platform_kind}__sample__{record.get('sample', 'center')}"
            )
        return f"object__{record_index}__{type(entity).__name__}"

    @classmethod
    def _plot_inspector_symbol(
        cls,
        inspector: list[list[str]],
        symbol: str,
        *,
        x: float,
        y: float,
        left: float,
        top: float,
        width: float,
        height: float,
    ) -> None:
        column = int(math.floor((x - left) / width * cls.INSPECTOR_COLUMNS))
        row = int(math.floor((y - top) / height * cls.INSPECTOR_ROWS))
        if not (0 <= row < cls.INSPECTOR_ROWS and 0 <= column < cls.INSPECTOR_COLUMNS):
            return
        previous = inspector[row][column]
        inspector[row][column] = symbol if previous == "-" else f"{previous}{symbol}"

    def detect_bump(self, agent: AgentConstruct, *, reason: str = "obstacle") -> None:
        adapter = getattr(agent, "actr_adapter", None)
        callback = getattr(adapter, "on_bump_detected", None)
        if callable(callback):
            callback(reason=reason)
        if self.print_middleman:
            print(f"{agent.name}: bump detected ({reason})")

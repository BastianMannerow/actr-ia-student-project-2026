"""Visual-stimulus-to-chunk adapter for :mod:`agents.Example`.

The adapter has one communication partner: the paired ACT-R model.  It reads
requests and model state from ACT-R chunks, reads perception only through
``pyactrFunctionalityExtension.publish_visual_stimulus()``, uses regular Python arithmetic for derived metrics, and writes the resulting chunks
back to the model.  It never reads the simulation world, another AgentConstruct,
``visual_metadata``, or physical entity classes, and it never executes motor
commands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import pyactr as actr

from simulation import pyactrFunctionalityExtension as ext


YES = "yes"
NO = "no"
NONE = "none"


def _as_float(value: Any, default: float | None = None) -> float:
    try:
        raw = getattr(value, "values", value)
        return float(raw)
    except (TypeError, ValueError):
        if default is None:
            raise
        return float(default)


def _calc(operation: str, *operands: Any) -> float:
    values = [_as_float(value) for value in operands]
    if operation == "add":
        return sum(values)
    if operation == "subtract":
        if not values:
            return 0.0
        result = values[0]
        for value in values[1:]:
            result -= value
        return result
    if operation == "multiply":
        result = 1.0
        for value in values:
            result *= value
        return result
    if operation == "divide":
        if not values:
            return 0.0
        result = values[0]
        for value in values[1:]:
            result /= value
        return result
    if operation == "minimum":
        return min(values)
    if operation == "maximum":
        return max(values)
    if operation == "absolute":
        return abs(values[0])
    if operation == "square":
        return values[0] ** 2
    raise ValueError(f"Unsupported arithmetic operation: {operation}")


def _cmp(left: Any, operator: str, right: Any, tolerance: float = 1e-9) -> bool:
    a, b = _as_float(left), _as_float(right)
    if operator in {"=", "=="}:
        return abs(a - b) <= tolerance
    if operator in {"!=", "<>"}:
        return abs(a - b) > tolerance
    if operator == "<": return a < b - tolerance
    if operator == "<=": return a <= b + tolerance
    if operator == ">": return a > b + tolerance
    if operator == ">=": return a >= b - tolerance
    raise ValueError(f"Unsupported comparison: {operator}")


def _truth(value: Any) -> bool:
    raw = getattr(value, "values", value)
    if isinstance(raw, str):
        return raw.strip().casefold() not in {"", "0", "false", "no", "none", "null"}
    return bool(raw)


def _logic(operation: str, *values: Any) -> bool:
    flags = [_truth(value) for value in values]
    if operation == "and": return all(flags)
    if operation == "or": return any(flags)
    if operation == "not": return not flags[0]
    if operation == "xor": return sum(flags) == 1
    if operation == "nand": return not all(flags)
    if operation == "nor": return not any(flags)
    raise ValueError(f"Unsupported logical operation: {operation}")


def _clamp(value: Any, minimum: Any, maximum: Any) -> float:
    return max(_as_float(minimum), min(_as_float(maximum), _as_float(value)))


def _distance(x1: Any, y1: Any, x2: Any, y2: Any) -> float:
    return math.hypot(_as_float(x2) - _as_float(x1), _as_float(y2) - _as_float(y1))


def _direction(delta: Any, *, tolerance: float = 0.0) -> str:
    value = _as_float(delta)
    if value < -abs(tolerance): return "left"
    if value > abs(tolerance): return "right"
    return "level"


@dataclass(frozen=True, slots=True)
class VisualAgent:
    symbol: str
    shape: str
    x: float
    y: float
    left: float
    right: float
    top: float
    bottom: float
    posture: str = "none"
    max_height: float = 0.0

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class VisualTarget:
    target_id: str
    role: str
    required_order: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class VisualPlatform:
    platform_id: str
    kind: str
    left: float
    right: float
    top: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class VisualScene:
    self_agent: VisualAgent
    other_agent: VisualAgent
    targets: tuple[VisualTarget, ...]
    platforms: tuple[VisualPlatform, ...]
    upper_floor_y: float
    lower_floor_y: float
    drop_x: float
    revision: int

    @property
    def region_midpoint(self) -> float:
        return (self.upper_floor_y + self.lower_floor_y) / 2.0


@dataclass(frozen=True, slots=True)
class TargetAssessment:
    target: VisualTarget
    region: str
    reachability: str
    actor: str
    strategy: str
    cooperation: str
    priority: str
    status: str
    score: float
    distance: float
    dx: float
    dy: float
    staging_x: float
    staging_y: float
    reason: str



class ExampleAdapter:
    """Visual-to-imaginal bridge represented explicitly in the State Graph.

    All pyactr access goes through ``pyactrFunctionalityExtension``.  The
    adapter never issues motor commands.  It publishes the current visual
    frame, derives quantitative state, updates imaginals and declarative chunks,
    and moves the controller only into states whose productions perform the
    next retrieval or manual request.
    """

    def __init__(self, _actr_environment=None):
        self.agent_construct = None

    def extending_actr(self) -> None:
        production = ext.get_production_fired(self.agent_construct)
        if production is None:
            return
        try:
            if production == "P01_identify_self":
                self._identify()
            elif production == "P04_request_situational_assessment":
                self._assess()
            elif production == "P06_coordination_retrieved":
                self._apply_coordination_retrieval()
            elif production == "P11_strategy_schema_retrieved":
                self._apply_strategy_retrieval()
            elif production == "P14_recovery_schema_retrieved":
                self._apply_recovery_retrieval()
        except Exception as exc:
            self._adapter_error(exc)

    def _revision(self) -> int:
        progress = self._buffer_chunk("imaginal_progress")
        return self._integer(self._chunk_value(progress, "revision", "0"), 0) + 1

    def _identify(self) -> None:
        revision = self._revision()
        scene = self._scene(revision)
        self_chunk = self._self_chunk(scene, revision)
        other_chunk = self._other_chunk(
            scene, revision, inferred_intent="survey_environment",
            inferred_target=NONE, readiness=NO, blocking=NONE,
            self_blocking=NO, partner_blocking=NO, commitment=NO,
        )
        coordination_chunk = self._coordination_chunk(
            joint_target=NONE, phase="idle_monitor", self_role="observer",
            partner_role="observer", self_commitment=NO, partner_commitment=NO,
            route_conflict=NO, blocker=NONE, yield_direction="level",
            partner_direction=_direction(scene.other_agent.x - scene.self_agent.x, tolerance=1.2),
            separation=_distance(scene.self_agent.x, scene.self_agent.y, scene.other_agent.x, scene.other_agent.y),
            progress="initializing", timeout="0", association="perfect_los", revision=revision,
        )
        support = self._agent_support(scene.self_agent, scene, self._metric_parameters())
        progress_chunk = actr.makechunk(
            typename="progress_model", target_id=NONE,
            current_x=self._number(scene.self_agent.x), current_y=self._number(scene.self_agent.y),
            previous_x=self._number(scene.self_agent.x), previous_y=self._number(scene.self_agent.y),
            current_distance="999", previous_distance="999", route_distance="999",
            previous_route_distance="999", best_route_distance="999", displacement="0",
            distance_gain="0", route_gain="0", no_progress_cycles="0",
            repeated_action=NONE, repeated_count="0", last_direction="level",
            direction_changes="0", approach_side=NONE, support=support,
            previous_support=support, failed_attempts="0", irrational_cycle=NO,
            stuck=NO, stuck_kind=NONE, cause="initializing", recovery_required=NO,
            revision=str(revision),
        )
        agent = self._agent()
        ext.set_imaginal(agent, self_chunk, "imaginal_self")
        ext.set_imaginal(agent, other_chunk, "imaginal_other")
        ext.set_imaginal(agent, coordination_chunk, "imaginal_coordination")
        ext.set_imaginal(agent, progress_chunk, "imaginal_progress")
        identity_chunk = actr.makechunk(
            typename="identity_model", identity=scene.self_agent.symbol,
            shape=scene.self_agent.shape, status="identified", revision=str(revision),
        )
        self._replace_dynamic_memory("identity_model", (identity_chunk,))
        self._replace_dynamic_memory("self_model", (self_chunk,))
        self._replace_dynamic_memory("other_model", (other_chunk,))
        self._replace_dynamic_memory("coordination_model", (coordination_chunk,))
        self._set_goal("identity_lookup")

    def _assess(self) -> None:
        revision = self._revision()
        scene = self._scene(revision)
        parameters = self._metric_parameters()
        policies = self._target_policies()
        world = self._world_facts(scene, parameters)
        assessments = self._classify_targets(scene, world, parameters, policies)
        candidate = self._select_candidate(scene, world, assessments, policies)
        baseline_stagnation = self._stagnation(candidate)
        partner_intent, partner_target, partner_readiness = self._infer_other(
            scene, candidate, assessments, policies, parameters
        )
        coordination = self._derive_coordination(
            scene, world, candidate, partner_target, partner_readiness,
            baseline_stagnation, parameters,
        )
        progress_chunk = self._progress_chunk(
            scene, candidate, coordination, revision
        )
        stagnation = self._integer(
            self._chunk_value(progress_chunk, "no_progress_cycles", "0"), 0
        )
        coordination = self._derive_coordination(
            scene, world, candidate, partner_target, partner_readiness,
            stagnation, parameters,
        )
        affordances = self._derive_affordances(
            scene, world, candidate, partner_readiness, stagnation, parameters
        )
        self._ensure_actionable_affordance(affordances)

        self_chunk = self._self_chunk(scene, revision)
        other_chunk = self._other_chunk(
            scene, revision, inferred_intent=partner_intent, inferred_target=partner_target,
            readiness=partner_readiness, blocking=coordination["blocker"],
            self_blocking=coordination["self_blocks_partner"],
            partner_blocking=coordination["partner_blocks_self"],
            commitment=coordination["partner_commitment"],
        )
        world_chunk = self._world_chunk(scene, world, revision)
        candidate_chunk = self._candidate_chunk(candidate, stagnation, revision)
        affordance_chunk = self._affordance_chunk(affordances, revision)
        coordination_chunk = self._coordination_chunk(
            joint_target=coordination["joint_target"], phase=coordination["phase"],
            self_role=coordination["self_role"], partner_role=coordination["partner_role"],
            self_commitment=coordination["self_commitment"],
            partner_commitment=coordination["partner_commitment"],
            route_conflict=coordination["route_conflict"], blocker=coordination["blocker"],
            yield_direction=coordination["yield_direction"],
            partner_direction=coordination["partner_direction"],
            separation=coordination["separation"], progress=coordination["progress"],
            timeout=coordination["timeout"], association=coordination["association"],
            revision=revision,
        )

        agent = self._agent()
        for key, chunk in (
            ("imaginal_self", self_chunk), ("imaginal_other", other_chunk),
            ("imaginal_world", world_chunk), ("imaginal_target", candidate_chunk),
            ("imaginal_affordance", affordance_chunk),
            ("imaginal_coordination", coordination_chunk),
            ("imaginal_progress", progress_chunk),
        ):
            ext.set_imaginal(agent, chunk, key)

        self._replace_dynamic_memory("self_model", (self_chunk,))
        self._replace_dynamic_memory("other_model", (other_chunk,))
        self._replace_dynamic_memory("world_model", (world_chunk,))
        self._replace_dynamic_memory("coordination_model", (coordination_chunk,))
        self._replace_dynamic_memory("progress_model", (progress_chunk,))
        self._replace_target_memory(assessments, revision)
        self._replace_platform_memory(scene, revision)
        self._set_goal(
            "coordination_lookup",
            target_id=self._chunk_value(candidate_chunk, "target_id", NONE),
            strategy=self._chunk_value(candidate_chunk, "strategy", "idle_monitor"),
            cycle=str(revision),
            coordination_phase="none",
        )

    def _apply_coordination_retrieval(self) -> None:
        retrieval = self._buffer_chunk("retrieval")
        strategy = self._chunk_value(retrieval, "recommended_strategy", NONE)
        phase = self._chunk_value(retrieval, "phase", NONE)
        if strategy == NONE:
            strategy = self._chunk_value(self._buffer_chunk("imaginal_target"), "strategy", "idle_monitor")
        self._set_goal("strategy_lookup", strategy=strategy, coordination_phase=phase)

    def _apply_strategy_retrieval(self) -> None:
        retrieval = self._buffer_chunk("retrieval")
        strategy = self._chunk_value(retrieval, "strategy", NONE)
        if strategy == NONE:
            strategy = self._chunk_value(self._buffer_chunk("imaginal_target"), "strategy", "idle_monitor")
        progress = self._buffer_chunk("imaginal_progress")
        if self._chunk_value(progress, "recovery_required", NO) == YES:
            self._set_goal("recovery_lookup", strategy=strategy)
        else:
            self._set_goal("decide", strategy=strategy)

    def _apply_recovery_retrieval(self) -> None:
        revision = self._revision()
        retrieval = self._buffer_chunk("retrieval")
        progress = self._buffer_chunk("imaginal_progress")
        target = self._buffer_chunk("imaginal_target")
        previous_plan = self._buffer_chunk("imaginal_recovery")
        action = self._chunk_value(retrieval, "action", "reverse")
        direction = self._chunk_value(retrieval, "direction", "dynamic")
        last_action = self._chunk_value(progress, "repeated_action", NONE)
        if direction == "subgoal":
            self_model = self._buffer_chunk("imaginal_self")
            self_x = self._float_chunk(self_model, "x", 0.0)
            staging_x = self._float_chunk(target, "staging_x", self_x)
            direction = "left" if staging_x < self_x else "right"
        elif direction in {"dynamic", "opposite"}:
            if "left" in last_action:
                direction = "right"
            elif "right" in last_action:
                direction = "left"
            else:
                dx = self._float_chunk(target, "dx", 0.0)
                direction = "left" if dx >= 0 else "right"
        if action == "flatten":
            direction = "none"
        attempts = self._integer(self._chunk_value(previous_plan, "attempts", "0"), 0) + 1
        target_id = self._chunk_value(target, "target_id", NONE)
        target_x = self._float_chunk(target, "dx", 0.0)
        approach_side = "left" if target_x >= 0 else "right"
        plan = actr.makechunk(
            typename="recovery_plan",
            stuck_kind=self._chunk_value(retrieval, "stuck_kind", "immobile"),
            strategy=self._chunk_value(retrieval, "strategy", "escape"),
            action=action, direction=direction,
            association=self._chunk_value(retrieval, "association", "stagnation_requires_escape"),
            target_id=target_id, approach_side=approach_side,
            attempts=str(attempts), status="ready", revision=str(revision),
        )
        ext.set_imaginal(self._agent(), plan, "imaginal_recovery")
        self._replace_dynamic_memory("recovery_plan", (plan,))
        self._set_goal("recover_decide", strategy=self._chunk_value(retrieval, "strategy", "escape"))

    def _adapter_error(self, exc: Exception) -> None:
        revision = self._revision()
        plan = actr.makechunk(
            typename="recovery_plan", stuck_kind="adapter_error", strategy="reassess",
            action="none", direction="none", association=self._symbolic_error(exc),
            target_id=NONE, approach_side=NONE, attempts="0", status="error",
            revision=str(revision),
        )
        ext.set_imaginal(self._agent(), plan, "imaginal_recovery")
        self._set_goal("adapter_error")

    def _set_goal(
        self, state: str, *, strategy: str | None = None, target_id: str | None = None,
        cycle: str | None = None, coordination_phase: str | None = None,
    ) -> None:
        current = self._buffer_chunk("g")
        values = {
            "self_symbol": self._chunk_value(current, "self_symbol", "unknown"),
            "self_shape": self._chunk_value(current, "self_shape", "unknown"),
            "target_id": self._chunk_value(current, "target_id", NONE),
            "strategy": self._chunk_value(current, "strategy", NONE),
            "cycle": self._chunk_value(current, "cycle", "0"),
            "last_action": self._chunk_value(current, "last_action", NONE),
            "coordination_phase": self._chunk_value(current, "coordination_phase", NONE),
        }
        if strategy is not None: values["strategy"] = strategy
        if target_id is not None: values["target_id"] = target_id
        if cycle is not None: values["cycle"] = cycle
        if coordination_phase is not None: values["coordination_phase"] = coordination_phase
        ext.set_goal(self._agent(), actr.makechunk(typename="controller", state=state, **values))

    # ------------------------------------------------------------------
    # Perception: the frame returned by pyactrFunctionalityExtension only
    # ------------------------------------------------------------------
    def _scene(self, revision: int) -> VisualScene:
        frame = ext.publish_visual_stimulus(self._agent())
        if not frame:
            raise RuntimeError("visual_frame_empty")
        scale = self._metric_parameters()["visual_position_scale"]

        agent_parts: dict[str, dict[str, Any]] = {}
        targets: list[VisualTarget] = []
        platform_parts: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

        for stimulus_id, stimulus in frame.items():
            parts = stimulus_id.split("__")
            x = _calc("divide", stimulus["position"][0], scale)
            y = _calc("divide", stimulus["position"][1], scale)
            if not parts:
                continue
            if parts[0] == "agent" and len(parts) >= 3:
                symbol = parts[1]
                fact = parts[2]
                record = agent_parts.setdefault(symbol, {})
                if fact == "body":
                    record["x"] = x
                    record["y"] = y
                elif fact == "shape" and len(parts) >= 4:
                    record["shape"] = parts[3]
                elif fact == "bound" and len(parts) >= 4:
                    record[parts[3]] = x if parts[3] in {"left", "right"} else y
                elif fact == "posture" and len(parts) >= 4:
                    record["posture"] = parts[3]
                elif fact == "capability" and len(parts) >= 5 and parts[3] == "max_height":
                    record["max_height"] = _calc(
                        "divide", self._integer(parts[4], 0), scale
                    )
                continue
            if parts[0] == "diamond" and len(parts) >= 6:
                targets.append(
                    VisualTarget(
                        target_id=parts[1],
                        role=parts[3],
                        required_order=self._integer(parts[5], 0),
                        x=x,
                        y=y,
                    )
                )
                continue
            if parts[0] == "platform" and len(parts) >= 6:
                platform_id = parts[1]
                kind = parts[3]
                # platform__<id>__kind__<kind>__sample__<top/bottom edge sample>
                sample = parts[5]
                platform_parts.setdefault((platform_id, kind), {})[sample] = (x, y)

        agents = {
            symbol: self._visual_agent(symbol, values)
            for symbol, values in agent_parts.items()
            if "x" in values and "y" in values
        }
        self_agent = agents.get("A")
        other_agent = next(
            (value for symbol, value in sorted(agents.items()) if symbol != "A"),
            None,
        )
        if self_agent is None or other_agent is None:
            raise RuntimeError("self_or_partner_not_visible")

        platforms: list[VisualPlatform] = []
        for (platform_id, kind), samples in platform_parts.items():
            positions = list(samples.values())
            if not positions:
                continue
            left = samples.get("left", min(positions, key=lambda item: item[0]))[0]
            right = samples.get("right", max(positions, key=lambda item: item[0]))[0]
            top = samples.get("center", positions[0])[1]
            bottom = samples.get(
                "bottom_center",
                samples.get("bottom_left", samples.get("bottom_right", (0.0, top))),
            )[1]
            platforms.append(
                VisualPlatform(
                    platform_id=platform_id,
                    kind=kind,
                    left=min(left, right),
                    right=max(left, right),
                    top=min(top, bottom),
                    bottom=max(top, bottom),
                )
            )

        upper_floor = self._platform_of_kind(platforms, "upper_floor")
        lower_floor = self._platform_of_kind(platforms, "lower_floor")
        if upper_floor is None or lower_floor is None:
            raise RuntimeError("floor_landmarks_not_visible")

        return VisualScene(
            self_agent=self_agent,
            other_agent=other_agent,
            targets=tuple(sorted(targets, key=lambda item: item.target_id)),
            platforms=tuple(platforms),
            upper_floor_y=upper_floor.top,
            lower_floor_y=lower_floor.top,
            drop_x=upper_floor.right,
            revision=revision,
        )

    @staticmethod
    def _visual_agent(symbol: str, values: dict[str, Any]) -> VisualAgent:
        shape = str(values.get("shape", "unknown"))
        x = float(values["x"])
        y = float(values["y"])
        if shape == "circle":
            half_width = half_height = 1.65
        else:
            half_width = 3.5
            half_height = 1.3
        return VisualAgent(
            symbol=symbol,
            shape=shape,
            x=x,
            y=y,
            left=float(values.get("left", x - half_width)),
            right=float(values.get("right", x + half_width)),
            top=float(values.get("top", y - half_height)),
            bottom=float(values.get("bottom", y + half_height)),
            posture=str(values.get("posture", "none")),
            max_height=float(values.get("max_height", 0.0)),
        )

    @staticmethod
    def _platform_of_kind(
        platforms: list[VisualPlatform] | tuple[VisualPlatform, ...],
        kind: str,
    ) -> VisualPlatform | None:
        return next((platform for platform in platforms if platform.kind == kind), None)

    # ------------------------------------------------------------------
    # Cognitive knowledge supplied by Example chunks
    # ------------------------------------------------------------------
    def _metric_parameters(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for chunk in ext.get_declarative_chunk_type(
            self._agent(), "metric_parameter"
        ):
            key = self._chunk_value(chunk, "parameter", NONE)
            value = self._chunk_value(chunk, "value", NONE)
            if key != NONE and value != NONE:
                result[key] = _as_float(value)
        required = {
            "visual_position_scale",
            "position_tolerance",
            "stage_tolerance",
            "takeoff_tolerance",
            "jump_vertical_envelope",
            "cooperation_mount_window",
            "cooperation_center_tolerance",
            "cooperation_jump_window",
            "rectangle_min_height",
            "rectangle_max_height",
            "rectangle_flat_threshold",
            "rectangle_tall_threshold",
            "fall_risk_margin",
            "ground_contact_tolerance",
            "ledge_takeoff_offset",
            "jump_priming_clearance",
            "recovery_runway_increment",
            "route_progress_tolerance",
            "failed_jump_limit",
            "oscillation_limit",
            "cooperation_stage_offset",
            "blocking_distance",
            "corridor_vertical_tolerance",
            "yield_clearance",
            "reunion_distance",
        }
        missing = sorted(required.difference(result))
        if missing:
            raise RuntimeError("missing_metric_chunks_" + "_".join(missing))
        return result

    def _target_policies(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for chunk in ext.get_declarative_chunk_type(self._agent(), "target_policy"):
            role = self._chunk_value(chunk, "role", NONE)
            result[role] = {
                "circle": self._chunk_value(chunk, "circle_strategy", "support_partner"),
                "rectangle": self._chunk_value(chunk, "rectangle_strategy", "support_partner"),
                "weight": _as_float(self._chunk_value(chunk, "priority_weight", "0")),
                "order": self._integer(self._chunk_value(chunk, "order_bias", "0"), 0),
                "cooperation": self._chunk_value(chunk, "requires_cooperation", NO),
            }
        if not result:
            raise RuntimeError("target_policy_chunks_missing")
        return result

    # ------------------------------------------------------------------
    # Situation derivation
    # ------------------------------------------------------------------
    def _world_facts(
        self, scene: VisualScene, parameters: dict[str, float]
    ) -> dict[str, Any]:
        upper_targets = [
            target for target in scene.targets if self._region(target.y, scene) == "upper"
        ]
        lower_targets = [
            target for target in scene.targets if self._region(target.y, scene) == "lower"
        ]
        self_region = self._region(scene.self_agent.y, scene)
        other_region = self._region(scene.other_agent.y, scene)
        priority_pending = any(
            target.role == "priority_before_drop" for target in scene.targets
        )
        team_below = _logic(
            "and", self_region == "lower", other_region == "lower"
        )
        trapped = _logic("and", bool(upper_targets), team_below)
        safe_to_descend = not upper_targets
        separation = _distance(
            scene.self_agent.x,
            scene.self_agent.y,
            scene.other_agent.x,
            scene.other_agent.y,
        )
        room_barrier = self._room_barrier_between(scene)
        agents_separated = self_region != other_region or room_barrier
        fall_risk = _logic(
            "and",
            self_region == "upper",
            _cmp(
                scene.self_agent.x,
                ">=",
                _calc(
                    "subtract", scene.drop_x, parameters["fall_risk_margin"]
                ),
            ),
            bool(upper_targets),
        )
        return {
            "upper_targets": upper_targets,
            "lower_targets": lower_targets,
            "self_region": self_region,
            "other_region": other_region,
            "priority_pending": priority_pending,
            "safe_to_descend": safe_to_descend,
            "trapped": trapped,
            "fall_risk": fall_risk,
            "cooperation_possible": self_region == other_region and not room_barrier,
            "agents_separated": agents_separated,
            "reunion_required": self_region != other_region,
            "room_conflict": room_barrier,
            "separation": separation,
        }

    def _classify_targets(
        self,
        scene: VisualScene,
        world: dict[str, Any],
        parameters: dict[str, float],
        policies: dict[str, dict[str, Any]],
    ) -> list[TargetAssessment]:
        result: list[TargetAssessment] = []
        for target in scene.targets:
            policy = policies.get(target.role)
            if policy is None:
                continue
            strategy = str(policy.get(scene.self_agent.shape, "support_partner"))
            actor = "partner" if strategy == "support_partner" else "self"
            region = self._region(target.y, scene)
            dx = _calc("subtract", target.x, scene.self_agent.x)
            dy = _calc("subtract", target.y, scene.self_agent.y)
            distance = _distance(
                scene.self_agent.x,
                scene.self_agent.y,
                target.x,
                target.y,
            )
            staging_x, staging_y = self._staging_point(
                target, scene, parameters
            )
            reachability, status, reason, effective_strategy = self._reachability(
                scene,
                world,
                target,
                region,
                actor,
                strategy,
                parameters,
            )
            score = float(policy["weight"])
            score += 35.0 if actor == "self" else 0.0
            score += 45.0 if status == "actionable" else 0.0
            score += 12.0 if effective_strategy.startswith("cooperative") else 0.0
            score -= distance * 0.35
            if region == "lower" and world["upper_targets"]:
                score -= 250.0
            result.append(
                TargetAssessment(
                    target=target,
                    region=region,
                    reachability=reachability,
                    actor=actor,
                    strategy=effective_strategy,
                    cooperation=YES if policy["cooperation"] == YES else NO,
                    priority=YES if target.role == "priority_before_drop" else NO,
                    status=status,
                    score=score,
                    distance=distance,
                    dx=dx,
                    dy=dy,
                    staging_x=staging_x,
                    staging_y=staging_y,
                    reason=reason,
                )
            )
        return result

    def _reachability(
        self,
        scene: VisualScene,
        world: dict[str, Any],
        target: VisualTarget,
        region: str,
        actor: str,
        strategy: str,
        parameters: dict[str, float],
    ) -> tuple[str, str, str, str]:
        if world["trapped"]:
            return (
                "replan_required",
                "actionable",
                "team_below_with_upper_target",
                "recover_replan",
            )
        if region == "lower" and world["self_region"] == "upper" and world["upper_targets"]:
            return (
                "blocked_by_order",
                "deferred",
                "upper_targets_precede_irreversible_descent",
                strategy,
            )
        if region == "upper" and world["self_region"] == "lower":
            return (
                "unreachable_after_drop",
                "blocked",
                "irreversible_descent",
                strategy,
            )
        if actor == "partner":
            return (
                "partner_preferred",
                "observable",
                "partner_shape_has_actor_fit",
                strategy,
            )
        if strategy.startswith("cooperative"):
            if not world["cooperation_possible"]:
                return (
                    "await_partner_region",
                    "deferred",
                    "partner_not_in_same_region",
                    strategy,
                )
            return (
                "cooperative",
                "actionable",
                "requires_stack_and_timing",
                strategy,
            )
        if region != world["self_region"]:
            if region == "lower" and world["safe_to_descend"]:
                return (
                    "reachable_by_descent",
                    "actionable",
                    "upper_region_clear",
                    "descend_after_priority",
                )
            return (
                "different_region",
                "deferred",
                "region_transition_required",
                strategy,
            )

        vertical_gap = _calc(
            "subtract", scene.self_agent.bottom, target.y
        )
        if scene.self_agent.shape == "circle":
            if _cmp(
                vertical_gap, "<=", parameters["jump_vertical_envelope"]
            ):
                return (
                    "direct_circle",
                    "actionable",
                    "within_circle_or_ledge_envelope",
                    strategy,
                )
            return (
                "needs_cooperation",
                "deferred",
                "above_solo_jump_envelope",
                strategy,
            )
        if target.role in {"solo_flat_rectangle", "solo_ground"}:
            return (
                "direct_rectangle",
                "actionable",
                "rectangle_route",
                strategy,
            )
        if abs(target.y - scene.self_agent.y) <= 3.8:
            return (
                "direct_rectangle",
                "actionable",
                "same_height_route",
                strategy,
            )
        return (
            "partner_preferred",
            "observable",
            "rectangle_cannot_jump",
            "support_partner",
        )

    def _select_candidate(
        self,
        scene: VisualScene,
        world: dict[str, Any],
        assessments: list[TargetAssessment],
        _policies: dict[str, dict[str, Any]],
    ) -> TargetAssessment | None:
        if not assessments:
            return None
        if world["trapped"]:
            upper = [item for item in assessments if item.region == "upper"]
            base = min(upper or assessments, key=lambda item: item.distance)
            return replace(
                base,
                actor="self",
                strategy="recover_replan",
                status="actionable",
                reachability="replan_required",
                score=1000.0,
                reason="upper_route_irreversible_without_reset",
            )

        priority = [
            item for item in assessments if item.target.role == "priority_before_drop"
        ]
        if priority:
            return max(priority, key=lambda item: item.score)

        actionable = [
            item
            for item in assessments
            if item.actor == "self" and item.status == "actionable"
        ]
        if actionable:
            return max(actionable, key=lambda item: item.score)

        if world["safe_to_descend"] and world["self_region"] == "upper":
            lower = [item for item in assessments if item.region == "lower"]
            if lower:
                base = min(lower, key=lambda item: item.distance)
                return replace(
                    base,
                    actor="self",
                    strategy="descend_after_priority",
                    status="actionable",
                    reachability="reachable_by_descent",
                    reason="upper_region_clear",
                )

        partner_targets = [item for item in assessments if item.actor == "partner"]
        if partner_targets:
            return max(partner_targets, key=lambda item: item.score)
        return max(assessments, key=lambda item: item.score)

    # ------------------------------------------------------------------
    # Partner model and affordances
    # ------------------------------------------------------------------
    def _infer_other(
        self,
        scene: VisualScene,
        candidate: TargetAssessment | None,
        assessments: list[TargetAssessment],
        policies: dict[str, dict[str, Any]],
        parameters: dict[str, float],
    ) -> tuple[str, str, str]:
        other_shape = scene.other_agent.shape
        partner_options: list[tuple[float, TargetAssessment, str]] = []
        for item in assessments:
            policy = policies.get(item.target.role, {})
            strategy = str(policy.get(other_shape, "support_partner"))
            fit = 40.0 if strategy != "support_partner" else 0.0
            partner_distance = _distance(
                scene.other_agent.x,
                scene.other_agent.y,
                item.target.x,
                item.target.y,
            )
            partner_options.append((item.score + fit - partner_distance * 0.2, item, strategy))
        if partner_options:
            _, partner_target, partner_strategy = max(partner_options, key=lambda value: value[0])
        else:
            partner_target = None
            partner_strategy = "idle_monitor"

        readiness = NO
        intent = "survey_environment"
        inferred_target = partner_target.target.target_id if partner_target else NONE
        if candidate and candidate.strategy.startswith("cooperative"):
            if other_shape == "rectangle":
                staged = abs(scene.other_agent.x - candidate.staging_x) <= parameters["stage_tolerance"]
                tall = self._transform(scene.other_agent, parameters) >= parameters["rectangle_tall_threshold"]
                readiness = YES if staged else NO
                intent = "raise_circle" if staged and tall else "stage_mobile_platform"
            elif other_shape == "circle":
                mounted = self._agent_support(
                    scene.other_agent, scene, parameters
                ) == f"agent_{scene.self_agent.symbol}"
                near = abs(scene.other_agent.x - scene.self_agent.x) <= parameters["cooperation_mount_window"]
                readiness = YES if mounted or near else NO
                intent = "mount_or_jump_from_rectangle"
            inferred_target = candidate.target.target_id
        elif partner_target is not None:
            intent = (
                "collect_partner_target"
                if partner_strategy != "support_partner"
                else "monitor_team_progress"
            )
        return intent, inferred_target, readiness

    def _derive_coordination(
        self,
        scene: VisualScene,
        world: dict[str, Any],
        candidate: TargetAssessment | None,
        partner_target: str,
        partner_readiness: str,
        stagnation: int,
        parameters: dict[str, float],
    ) -> dict[str, str]:
        """Derive social facts from visual geometry; productions choose the response.

        The adapter does not decide a motor command here.  It only states whether
        the agents are separated, whether one body blocks the other's corridor,
        and which joint phase is currently visible.  ``Example`` retrieves an
        associated coordination schema and selects the actual strategy through
        productions.
        """
        separation = _distance(
            scene.self_agent.x,
            scene.self_agent.y,
            scene.other_agent.x,
            scene.other_agent.y,
        )
        partner_direction = _direction(
            scene.other_agent.x - scene.self_agent.x,
            tolerance=parameters["position_tolerance"],
        )
        self_blocks_partner = False
        partner_blocks_self = False
        destination_x = candidate.staging_x if candidate is not None else scene.other_agent.x
        if candidate is not None:
            same_corridor = abs(scene.self_agent.y - scene.other_agent.y) <= parameters[
                "corridor_vertical_tolerance"
            ]
            self_between = self._between(
                scene.self_agent.x,
                scene.other_agent.x,
                destination_x,
                margin=scene.self_agent.width * 0.35,
            )
            partner_between = self._between(
                scene.other_agent.x,
                scene.self_agent.x,
                destination_x,
                margin=scene.other_agent.width * 0.35,
            )
            close_enough = separation <= parameters["blocking_distance"]
            self_blocks_partner = bool(
                same_corridor
                and close_enough
                and candidate.actor == "partner"
                and self_between
            )
            partner_blocks_self = bool(
                same_corridor
                and close_enough
                and candidate.actor == "self"
                and partner_between
            )

        if world["reunion_required"]:
            phase = "reunite_team"
            self_role = "separated"
            partner_role = "partner"
            association = "separation_precedes_cooperation"
            if world["self_region"] == "upper" and world["other_region"] == "lower":
                partner_direction = _direction(
                    scene.drop_x + 2.0 - scene.self_agent.x,
                    tolerance=parameters["position_tolerance"],
                )
            elif world["self_region"] == "lower" and world["other_region"] == "upper":
                partner_direction = _direction(
                    scene.drop_x + 4.0 - scene.self_agent.x,
                    tolerance=parameters["reunion_distance"],
                )
        elif self_blocks_partner:
            phase = "yield_route"
            self_role = "support"
            partner_role = "collector"
            association = "blocking_requires_yield"
        elif candidate is None:
            phase = "idle_monitor"
            self_role = "observer"
            partner_role = "observer"
            association = "no_actionable_target"
        else:
            phase = candidate.strategy
            if candidate.strategy == "cooperative_rectangle":
                self_role, partner_role = "provider", "rider"
                association = "joint_target_binding"
            elif candidate.strategy == "cooperative_circle":
                self_role, partner_role = "rider", "provider"
                association = "joint_target_binding"
            elif candidate.strategy == "support_partner":
                self_role, partner_role = "support", "collector"
                association = "complementary_role"
            elif candidate.strategy == "descend_after_priority":
                self_role, partner_role = "team_member", "team_member"
                association = "ordered_descent"
            elif candidate.strategy == "recover_replan":
                self_role, partner_role = "team_member", "team_member"
                association = "trapped_team"
            else:
                self_role, partner_role = "collector", "monitor"
                association = "shape_target_fit"

        yield_direction = "level"
        if self_blocks_partner and candidate is not None:
            # A supporting agent must leave the segment between collector and
            # target.  Moving farther toward the target creates a recurring
            # chase/block cycle, so the associated social episode yields behind
            # the collector whenever the visual geometry permits it.
            yield_direction = _direction(
                scene.other_agent.x - scene.self_agent.x,
                tolerance=0.2,
            )
            if yield_direction == "level":
                yield_direction = _direction(
                    scene.self_agent.x - destination_x,
                    tolerance=parameters["position_tolerance"],
                )
        route_conflict = YES if (self_blocks_partner or partner_blocks_self) else NO
        blocker = (
            "self" if self_blocks_partner else "partner" if partner_blocks_self else NONE
        )
        self_commitment = YES if phase not in {"idle_monitor", "support_partner"} else NO
        partner_commitment = (
            YES
            if partner_readiness == YES
            or (candidate is not None and partner_target == candidate.target.target_id)
            else NO
        )
        progress = (
            "blocked"
            if route_conflict == YES
            else "stagnating" if stagnation >= 3
            else "advancing"
        )
        return {
            "joint_target": candidate.target.target_id if candidate is not None else NONE,
            "phase": phase,
            "self_role": self_role,
            "partner_role": partner_role,
            "self_commitment": self_commitment,
            "partner_commitment": partner_commitment,
            "route_conflict": route_conflict,
            "blocker": blocker,
            "self_blocks_partner": YES if self_blocks_partner else NO,
            "partner_blocks_self": YES if partner_blocks_self else NO,
            "yield_direction": yield_direction,
            "partner_direction": partner_direction,
            "separation": self._number(separation),
            "progress": progress,
            "timeout": str(max(0, stagnation)),
            "association": association,
        }

    @staticmethod
    def _between(value: float, start: float, end: float, *, margin: float = 0.0) -> bool:
        low, high = sorted((start, end))
        return low - margin <= value <= high + margin

    @staticmethod
    def _room_barrier_between(scene: VisualScene) -> bool:
        low_x, high_x = sorted((scene.self_agent.x, scene.other_agent.x))
        for platform in scene.platforms:
            if platform.kind != "room_wall_low_door":
                continue
            center = (platform.left + platform.right) / 2.0
            if low_x < center < high_x:
                # A low doorway is visually passable only while both agents are
                # near the shared floor.  Until then it represents separate rooms.
                floor_y = (
                    scene.upper_floor_y
                    if platform.top < scene.region_midpoint
                    else scene.lower_floor_y
                )
                if (
                    abs(scene.self_agent.bottom - floor_y) > 1.2
                    or abs(scene.other_agent.bottom - floor_y) > 1.2
                ):
                    return True
        return False

    def _derive_affordances(
        self,
        scene: VisualScene,
        world: dict[str, Any],
        candidate: TargetAssessment | None,
        partner_readiness: str,
        stagnation: int,
        parameters: dict[str, float],
    ) -> dict[str, str]:
        flags = {
            "transform_up": NO,
            "transform_down": NO,
            "move_left": NO,
            "move_right": NO,
            "prime_jump_left": NO,
            "prime_jump_right": NO,
            "fast_fall": NO,
            "wait": NO,
            "replan": NO,
            "partner_ready": partner_readiness,
            "circle_on_rectangle": NO,
            "aligned": NO,
            "target_above": NO,
            "unsafe_descent": YES if world["fall_risk"] else NO,
            "reason": "no_candidate",
        }
        if candidate is None:
            flags["wait"] = YES
            flags["reason"] = "no_visible_target"
            return flags

        dx = _calc("subtract", candidate.target.x, scene.self_agent.x)
        dy = _calc("subtract", candidate.target.y, scene.self_agent.y)
        grounded = self._is_grounded(scene.self_agent, scene, parameters)
        support = self._agent_support(scene.self_agent, scene, parameters)
        transform = self._transform(scene.self_agent, parameters)
        flags["target_above"] = YES if candidate.target.y < scene.self_agent.y - 0.7 else NO
        flags["aligned"] = YES if abs(dx) <= parameters["position_tolerance"] else NO
        flags["circle_on_rectangle"] = YES if support.startswith("agent_") else NO
        flags["reason"] = candidate.reason

        if candidate.strategy == "recover_replan":
            flags["replan"] = YES
            flags["reason"] = "irreversible_route_requires_cognitive_reclassification"
            return flags
        if candidate.strategy == "support_partner":
            flags["wait"] = YES
            flags["reason"] = "partner_executes_better_fitting_target"
            return flags
        if candidate.strategy == "descend_after_priority":
            destination = scene.drop_x + 2.0
            self._set_horizontal(flags, destination - scene.self_agent.x, 1.0)
            if scene.self_agent.x >= scene.drop_x + 0.8 and not grounded:
                flags["fast_fall"] = YES
            return flags
        if candidate.strategy == "direct_rectangle":
            if (
                candidate.target.role == "solo_flat_rectangle"
                and transform > parameters["rectangle_flat_threshold"]
            ):
                flags["transform_down"] = YES
            self._set_horizontal(flags, dx, parameters["position_tolerance"])
            if abs(dx) <= parameters["position_tolerance"]:
                flags["wait"] = YES
            return flags
        if candidate.strategy == "direct_circle":
            return self._direct_circle_affordances(
                flags,
                scene,
                candidate,
                grounded,
                support,
                stagnation,
                parameters,
            )
        if candidate.strategy == "cooperative_rectangle":
            return self._rectangle_cooperation_affordances(
                flags, scene, candidate, transform, parameters
            )
        if candidate.strategy == "cooperative_circle":
            return self._circle_cooperation_affordances(
                flags, scene, candidate, support, grounded, parameters
            )

        if not grounded and dy > 2.5 and abs(dx) < 2.5:
            flags["fast_fall"] = YES
        else:
            flags["replan"] = YES
        return flags

    def _direct_circle_affordances(
        self,
        flags: dict[str, str],
        scene: VisualScene,
        candidate: TargetAssessment,
        grounded: bool,
        support: str,
        stagnation: int,
        parameters: dict[str, float],
    ) -> dict[str, str]:
        dx = candidate.target.x - scene.self_agent.x
        dy = candidate.target.y - scene.self_agent.y
        above = candidate.target.y < scene.self_agent.y - 1.0
        target_support = self._target_support(candidate.target, scene)
        on_target_support = (
            target_support is not None
            and support == f"platform_{target_support.platform_id}"
        )
        partner_blocks_route = self._partner_blocks_route(
            scene,
            candidate.staging_x if above else candidate.target.x,
            parameters,
        )
        if partner_blocks_route and grounded:
            direction = "left" if dx < 0 else "right"
            flags[f"prime_jump_{direction}"] = YES
            flags["reason"] = "jump_over_partner_blocking_route"
            return flags
        if above and grounded and not on_target_support:
            stage_dx = candidate.staging_x - scene.self_agent.x
            takeoff_tolerance = parameters["takeoff_tolerance"]
            if abs(stage_dx) > takeoff_tolerance:
                self._set_horizontal(flags, stage_dx, takeoff_tolerance)
                flags["reason"] = (
                    "extend_runway_after_failed_jump"
                    if stagnation >= 2
                    else "approach_precise_ledge_takeoff_point"
                )
                return flags
            # Only the dedicated sub-unit take-off window may release the jump.
            # This prevents the previous 2.4-unit early launch at the ledge wall.
            landing_x = (
                (target_support.left + target_support.right) / 2.0
                if target_support is not None
                else candidate.target.x
            )
            jump_direction = "left" if landing_x < scene.self_agent.x else "right"
            flags[f"prime_jump_{jump_direction}"] = YES
            flags["reason"] = "jump_from_precise_runway_into_landing_window"
            return flags

        self._set_horizontal(flags, dx, parameters["position_tolerance"])
        if grounded and stagnation >= 3 and abs(dx) > 1.5:
            flags["prime_jump_left" if dx < 0 else "prime_jump_right"] = YES
            flags["reason"] = "jump_over_detected_route_obstacle"
        if not grounded and candidate.target.y > scene.self_agent.y + 2.5 and abs(dx) < 2.5:
            flags["fast_fall"] = YES
        if abs(dx) <= parameters["position_tolerance"] and abs(dy) <= 2.8:
            flags["wait"] = YES
        return flags

    def _rectangle_cooperation_affordances(
        self,
        flags: dict[str, str],
        scene: VisualScene,
        candidate: TargetAssessment,
        transform: float,
        parameters: dict[str, float],
    ) -> dict[str, str]:
        stage_dx = candidate.staging_x - scene.self_agent.x
        circle_on_top = self._agent_support(
            scene.other_agent, scene, parameters
        ) == f"agent_{scene.self_agent.symbol}"
        flags["circle_on_rectangle"] = YES if circle_on_top else NO
        staged = abs(stage_dx) <= parameters["stage_tolerance"]
        flags["aligned"] = YES if staged else NO
        if not staged:
            if transform > parameters["rectangle_flat_threshold"]:
                flags["transform_down"] = YES
            self._set_horizontal(flags, stage_dx, parameters["stage_tolerance"])
            return flags
        if not circle_on_top:
            if transform > parameters["rectangle_flat_threshold"]:
                flags["transform_down"] = YES
                flags["reason"] = "flatten_again_for_circle_mount"
            else:
                flags["wait"] = YES
                flags["reason"] = "staged_waiting_for_circle_mount"
            return flags
        center_error = scene.other_agent.x - scene.self_agent.x
        if abs(center_error) > parameters["cooperation_center_tolerance"]:
            flags["wait"] = YES
            flags["reason"] = "hold_flat_until_circle_centered"
            return flags
        rider_vertical_gap = scene.other_agent.bottom - candidate.target.y
        lift_sufficient = rider_vertical_gap <= parameters["jump_vertical_envelope"]
        if not lift_sufficient and transform < 1.0:
            flags["transform_up"] = YES
            flags["reason"] = "centered_circle_on_top_raise_until_target_reachable"
            return flags
        flags["wait"] = YES
        flags["reason"] = (
            "hold_at_reachable_height_until_circle_jumps"
            if lift_sufficient
            else "hold_at_maximum_clear_height_for_circle"
        )
        return flags

    def _circle_cooperation_affordances(
        self,
        flags: dict[str, str],
        scene: VisualScene,
        candidate: TargetAssessment,
        support: str,
        grounded: bool,
        parameters: dict[str, float],
    ) -> dict[str, str]:
        rectangle = scene.other_agent
        stage_ready = abs(rectangle.x - candidate.staging_x) <= parameters["stage_tolerance"]
        on_rectangle = support == f"agent_{rectangle.symbol}"
        flags["circle_on_rectangle"] = YES if on_rectangle else NO
        if not stage_ready:
            follow_dx = rectangle.x - scene.self_agent.x
            self._set_horizontal(flags, follow_dx, 2.2)
            if abs(follow_dx) <= 2.2:
                flags["wait"] = YES
            flags["reason"] = "follow_rectangle_to_staging_point"
            return flags
        if not on_rectangle:
            mount_dx = rectangle.x - scene.self_agent.x
            self._set_horizontal(flags, mount_dx, 1.2)
            # A wide flattened rectangle physically prevents the circle centre
            # from entering the old centre-distance mount window.  Use the
            # visually perceived body edges as the decisive affordance: when
            # the bodies touch (or are within one small approach step), prime a
            # jump toward the rectangle centre instead of pushing into its side.
            edge_gap = max(
                0.0,
                rectangle.left - scene.self_agent.right,
                scene.self_agent.left - rectangle.right,
            )
            near_mount_edge = edge_gap <= parameters["position_tolerance"]
            if grounded and (
                abs(mount_dx) <= parameters["cooperation_mount_window"]
                or near_mount_edge
            ):
                flags["move_left"] = NO
                flags["move_right"] = NO
                flags["prime_jump_left" if mount_dx < -0.2 else "prime_jump_right"] = YES
                flags["reason"] = "jump_from_visible_rectangle_edge_to_mount"
            else:
                flags["reason"] = "approach_visible_rectangle_edge"
            return flags
        center_dx = rectangle.x - scene.self_agent.x
        if abs(center_dx) > parameters["cooperation_center_tolerance"]:
            self._set_horizontal(
                flags, center_dx, parameters["cooperation_center_tolerance"]
            )
            flags["reason"] = "center_on_rectangle_before_raise"
            return flags
        rider_vertical_gap = scene.self_agent.bottom - candidate.target.y
        lift_sufficient = rider_vertical_gap <= parameters["jump_vertical_envelope"]
        if not lift_sufficient:
            flags["wait"] = YES
            flags["reason"] = "centered_wait_until_target_enters_jump_envelope"
            return flags
        jump_dx = candidate.target.x - scene.self_agent.x
        if abs(jump_dx) <= parameters["cooperation_jump_window"]:
            flags["prime_jump_left" if jump_dx < 0 else "prime_jump_right"] = YES
            flags["reason"] = "timed_jump_from_raised_rectangle"
        else:
            self._set_horizontal(flags, jump_dx, parameters["cooperation_jump_window"])
        return flags

    def _progress_chunk(
        self,
        scene: VisualScene,
        candidate: TargetAssessment | None,
        coordination: dict[str, Any],
        revision: int,
    ):
        """Encode progress toward the active subgoal and detect policy cycles.

        Mere displacement is not progress: walking away from a take-off point
        or returning to the same floor after a failed jump must accumulate
        evidence of irrational repetition.  Progress is therefore measured
        against the current route subgoal (take-off point or target) and the
        support surface reached after an action.
        """
        parameters = self._metric_parameters()
        previous = self._buffer_chunk("imaginal_progress")
        previous_other = self._buffer_chunk("imaginal_other")
        goal = self._buffer_chunk("g")

        previous_x = self._float_chunk(previous, "current_x", scene.self_agent.x)
        previous_y = self._float_chunk(previous, "current_y", scene.self_agent.y)
        displacement = _distance(
            previous_x, previous_y, scene.self_agent.x, scene.self_agent.y
        )
        partner_previous_x = self._float_chunk(
            previous_other, "x", scene.other_agent.x
        )
        partner_previous_y = self._float_chunk(
            previous_other, "y", scene.other_agent.y
        )
        partner_displacement = _distance(
            partner_previous_x,
            partner_previous_y,
            scene.other_agent.x,
            scene.other_agent.y,
        )

        target_id = candidate.target.target_id if candidate is not None else NONE
        current_distance = candidate.distance if candidate is not None else 999.0
        previous_target = self._chunk_value(previous, "target_id", NONE)
        same_target = target_id == previous_target
        previous_distance = self._float_chunk(previous, "current_distance", 999.0)
        distance_gain = (
            previous_distance - current_distance
            if same_target and candidate is not None
            else 0.0
        )

        current_support = self._agent_support(
            scene.self_agent, scene, parameters
        )
        previous_support = self._chunk_value(previous, "support", current_support)
        target_support = (
            self._target_support(candidate.target, scene)
            if candidate is not None
            else None
        )
        target_support_name = (
            f"platform_{target_support.platform_id}"
            if target_support is not None
            else NONE
        )
        on_target_support = current_support == target_support_name
        target_above = (
            candidate is not None
            and candidate.target.y < scene.self_agent.y - 1.0
            and not on_target_support
        )
        route_distance = (
            abs(candidate.staging_x - scene.self_agent.x)
            if target_above
            else current_distance
        ) if candidate is not None else 999.0
        previous_route_distance = self._float_chunk(
            previous, "route_distance", 999.0
        )
        previous_best = self._float_chunk(
            previous, "best_route_distance", 999.0
        )
        best_route_distance = (
            min(previous_best, route_distance)
            if same_target
            else route_distance
        )
        route_gain = (
            previous_route_distance - route_distance if same_target else 0.0
        )

        last_action = self._chunk_value(goal, "last_action", NONE)
        prior_action = self._chunk_value(previous, "repeated_action", NONE)
        prior_repeated = self._integer(
            self._chunk_value(previous, "repeated_count", "0"), 0
        )
        repeated_count = prior_repeated + 1 if last_action == prior_action else 1

        approach_side = NONE
        if target_support is not None and candidate is not None:
            support_center = (target_support.left + target_support.right) / 2.0
            approach_side = "left" if candidate.staging_x < support_center else "right"
        if candidate is not None:
            direction = _direction(
                candidate.staging_x - scene.self_agent.x,
                tolerance=parameters["takeoff_tolerance"],
            )
        else:
            direction = "level"
        prior_direction = self._chunk_value(previous, "last_direction", "level")
        prior_changes = self._integer(
            self._chunk_value(previous, "direction_changes", "0"), 0
        )
        direction_changes = prior_changes
        if (
            same_target
            and direction in {"left", "right"}
            and prior_direction in {"left", "right"}
            and direction != prior_direction
        ):
            direction_changes += 1
        elif route_gain >= 0.75 and last_action not in {"jump", "recovery_jump"}:
            direction_changes = max(0, direction_changes - 1)

        grounded = current_support != NONE
        previous_failed = self._integer(
            self._chunk_value(previous, "failed_attempts", "0"), 0
        )
        failed_jump = bool(
            same_target
            and grounded
            and last_action in {"jump", "recovery_jump"}
            and current_support == previous_support
            and not on_target_support
        )
        if not same_target or on_target_support:
            failed_attempts = 0
        elif failed_jump:
            failed_attempts = previous_failed + 1
        else:
            failed_attempts = previous_failed

        previous_no_progress = self._integer(
            self._chunk_value(previous, "no_progress_cycles", "0"), 0
        )
        route_progress = route_gain >= parameters["route_progress_tolerance"]
        support_progress = bool(
            on_target_support and current_support != previous_support
        )
        joint_progress = (
            coordination.get("phase") in {
                "cooperative_circle",
                "cooperative_rectangle",
                "reunite_team",
                "support_partner",
            }
            and partner_displacement >= parameters["route_progress_tolerance"]
        )
        intentionally_monitoring = last_action in {
            "wait",
            "monitor_reunion",
            "transform_up",
            "transform_down",
        } and joint_progress
        if not same_target:
            no_progress = 0
        elif route_progress or support_progress or intentionally_monitoring:
            no_progress = 0
        else:
            no_progress = previous_no_progress + 1

        route_conflict = coordination.get("route_conflict") == YES
        failed_limit = max(1, int(parameters["failed_jump_limit"]))
        oscillation_limit = max(2, int(parameters["oscillation_limit"]))
        previous_stuck = self._chunk_value(previous, "stuck", NO) == YES
        previous_stuck_kind = self._chunk_value(previous, "stuck_kind", NONE)
        irrational_cycle = bool(
            failed_attempts >= failed_limit
            or (direction_changes >= oscillation_limit and no_progress >= 2)
            or (previous_stuck and same_target and not route_progress)
        )
        stuck = False
        stuck_kind = NONE
        cause = "route_progress_observed" if route_progress else "monitoring_route"

        if failed_attempts >= failed_limit:
            stuck = True
            stuck_kind = "jump_loop"
            cause = "returned_to_same_support_after_repeated_jump"
        elif direction_changes >= oscillation_limit and no_progress >= 2:
            stuck = True
            stuck_kind = "oscillating"
            cause = "alternating_route_direction_without_subgoal_gain"
        elif route_conflict and no_progress >= 3:
            stuck = True
            stuck_kind = "partner_deadlock"
            cause = "route_conflict_without_joint_progress"
        elif (
            last_action in {"transform_up", "transform_down", "recovery_flatten"}
            and repeated_count >= 4
            and no_progress >= 4
        ):
            stuck = True
            stuck_kind = "transform_blocked"
            cause = "aspect_change_without_spatial_progress"
        elif candidate is None and no_progress >= 10:
            stuck = True
            stuck_kind = "no_target_progress"
            cause = "no_actionable_target_over_extended_window"
        elif grounded and displacement < parameters["route_progress_tolerance"] and no_progress >= 6 and repeated_count >= 4:
            stuck = True
            stuck_kind = "immobile"
            cause = "repeated_action_without_visual_displacement"
        elif (
            previous_stuck
            and same_target
            and not route_progress
            and not support_progress
            and not intentionally_monitoring
        ):
            # A retrieved recovery schema is a hypothesis, not evidence that
            # the failure has ended.  Preserve the diagnosed state until the
            # visual frame proves route or support progress.  This prevents a
            # one-cycle recovery from erasing the episode and re-entering the
            # same irrational policy immediately.
            stuck = True
            stuck_kind = previous_stuck_kind
            cause = "recovery_outcome_not_yet_observed"

        return actr.makechunk(
            typename="progress_model",
            target_id=target_id,
            current_x=self._number(scene.self_agent.x),
            current_y=self._number(scene.self_agent.y),
            previous_x=self._number(previous_x),
            previous_y=self._number(previous_y),
            current_distance=self._number(current_distance),
            previous_distance=self._number(previous_distance),
            route_distance=self._number(route_distance),
            previous_route_distance=self._number(previous_route_distance),
            best_route_distance=self._number(best_route_distance),
            displacement=self._number(displacement),
            distance_gain=self._number(distance_gain),
            route_gain=self._number(route_gain),
            no_progress_cycles=str(no_progress),
            repeated_action=last_action,
            repeated_count=str(repeated_count),
            last_direction=direction,
            direction_changes=str(direction_changes),
            approach_side=approach_side,
            support=current_support,
            previous_support=previous_support,
            failed_attempts=str(failed_attempts),
            irrational_cycle=YES if irrational_cycle else NO,
            stuck=YES if stuck else NO,
            stuck_kind=stuck_kind,
            cause=cause,
            recovery_required=YES if stuck else NO,
            revision=str(revision),
        )

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------
    def _self_chunk(self, scene: VisualScene, revision: int):
        old = self._buffer_chunk("imaginal_self")
        old_x = self._float_chunk(old, "x", scene.self_agent.x)
        old_y = self._float_chunk(old, "y", scene.self_agent.y)
        dx = scene.self_agent.x - old_x
        dy = scene.self_agent.y - old_y
        parameters = self._metric_parameters()
        return actr.makechunk(
            typename="self_model",
            identity=scene.self_agent.symbol,
            shape=scene.self_agent.shape,
            x=self._number(scene.self_agent.x),
            y=self._number(scene.self_agent.y),
            previous_x=self._number(old_x),
            previous_y=self._number(old_y),
            dx=self._number(dx),
            dy=self._number(dy),
            grounded=YES if self._is_grounded(scene.self_agent, scene, parameters) else NO,
            support=self._agent_support(scene.self_agent, scene, parameters),
            transform=self._number(self._transform(scene.self_agent, parameters)),
            region=self._region(scene.self_agent.y, scene),
            motion=self._motion(dx, dy),
            confidence="1.0",
            revision=str(revision),
        )

    def _other_chunk(
        self,
        scene: VisualScene,
        revision: int,
        *,
        inferred_intent: str,
        inferred_target: str,
        readiness: str,
        blocking: str,
        self_blocking: str,
        partner_blocking: str,
        commitment: str,
    ):
        old = self._buffer_chunk("imaginal_other")
        old_x = self._float_chunk(old, "x", scene.other_agent.x)
        old_y = self._float_chunk(old, "y", scene.other_agent.y)
        dx = scene.other_agent.x - old_x
        dy = scene.other_agent.y - old_y
        parameters = self._metric_parameters()
        relation = (
            f"{_direction(scene.other_agent.x - scene.self_agent.x, tolerance=1.2)}_"
            f"{self._vertical_direction(scene.other_agent.y - scene.self_agent.y, 1.2)}"
        )
        return actr.makechunk(
            typename="other_model",
            identity=scene.other_agent.symbol,
            shape=scene.other_agent.shape,
            x=self._number(scene.other_agent.x),
            y=self._number(scene.other_agent.y),
            previous_x=self._number(old_x),
            previous_y=self._number(old_y),
            dx=self._number(dx),
            dy=self._number(dy),
            grounded=YES if self._is_grounded(scene.other_agent, scene, parameters) else NO,
            support=self._agent_support(scene.other_agent, scene, parameters),
            transform=self._number(self._transform(scene.other_agent, parameters)),
            region=self._region(scene.other_agent.y, scene),
            motion=self._motion(dx, dy),
            inferred_intent=inferred_intent,
            inferred_target=inferred_target,
            readiness=readiness,
            relation=relation,
            blocking=blocking,
            self_blocking=self_blocking,
            partner_blocking=partner_blocking,
            commitment=commitment,
            confidence="0.95",
            revision=str(revision),
        )

    @staticmethod
    def _world_chunk(scene: VisualScene, world: dict[str, Any], revision: int):
        return actr.makechunk(
            typename="world_model",
            revision=str(revision),
            remaining=str(len(scene.targets)),
            upper_remaining=str(len(world["upper_targets"])),
            lower_remaining=str(len(world["lower_targets"])),
            region=world["self_region"],
            priority_pending=YES if world["priority_pending"] else NO,
            safe_to_descend=YES if world["safe_to_descend"] else NO,
            irreversible_drop=YES,
            fall_risk=YES if world["fall_risk"] else NO,
            cooperation_possible=YES if world["cooperation_possible"] else NO,
            agents_separated=YES if world["agents_separated"] else NO,
            reunion_required=YES if world["reunion_required"] else NO,
            room_conflict=YES if world["room_conflict"] else NO,
            status="trapped" if world["trapped"] else "active",
        )

    @staticmethod
    def _candidate_chunk(
        candidate: TargetAssessment | None,
        stagnation: int,
        revision: int,
    ):
        if candidate is None:
            return actr.makechunk(
                typename="candidate_model",
                status="none",
                target_id=NONE,
                role=NONE,
                reachability="none",
                actor="none",
                strategy="idle_monitor",
                cooperation=NO,
                priority=NO,
                score="0",
                distance="0",
                dx="0",
                dy="0",
                staging_x="0",
                staging_y="0",
                reason="no_visible_target",
                stagnation=str(stagnation),
                reservation=NONE,
                revision=str(revision),
            )
        return actr.makechunk(
            typename="candidate_model",
            status=candidate.status,
            target_id=candidate.target.target_id,
            role=candidate.target.role,
            reachability=candidate.reachability,
            actor=candidate.actor,
            strategy=candidate.strategy,
            cooperation=candidate.cooperation,
            priority=candidate.priority,
            score=ExampleAdapter._number(candidate.score),
            distance=ExampleAdapter._number(candidate.distance),
            dx=ExampleAdapter._number(candidate.dx),
            dy=ExampleAdapter._number(candidate.dy),
            staging_x=ExampleAdapter._number(candidate.staging_x),
            staging_y=ExampleAdapter._number(candidate.staging_y),
            reason=candidate.reason,
            stagnation=str(stagnation),
            reservation=(
                "joint" if candidate.cooperation == YES
                else "self" if candidate.actor == "self"
                else "partner"
            ),
            revision=str(revision),
        )

    @staticmethod
    def _coordination_chunk(
        *,
        joint_target: str,
        phase: str,
        self_role: str,
        partner_role: str,
        self_commitment: str,
        partner_commitment: str,
        route_conflict: str,
        blocker: str,
        yield_direction: str,
        partner_direction: str,
        separation: float | str,
        progress: str,
        timeout: str,
        association: str,
        revision: int,
    ):
        return actr.makechunk(
            typename="coordination_model",
            joint_target=joint_target,
            phase=phase,
            self_role=self_role,
            partner_role=partner_role,
            self_commitment=self_commitment,
            partner_commitment=partner_commitment,
            route_conflict=route_conflict,
            blocker=blocker,
            yield_direction=yield_direction,
            partner_direction=partner_direction,
            separation=ExampleAdapter._number(separation),
            progress=progress,
            timeout=timeout,
            association=association,
            revision=str(revision),
        )

    @staticmethod
    def _affordance_chunk(flags: dict[str, str], revision: int):
        return actr.makechunk(
            typename="affordance_model",
            transform_up=flags["transform_up"],
            transform_down=flags["transform_down"],
            move_left=flags["move_left"],
            move_right=flags["move_right"],
            prime_jump_left=flags["prime_jump_left"],
            prime_jump_right=flags["prime_jump_right"],
            fast_fall=flags["fast_fall"],
            wait=flags["wait"],
            replan=flags["replan"],
            partner_ready=flags["partner_ready"],
            circle_on_rectangle=flags["circle_on_rectangle"],
            aligned=flags["aligned"],
            target_above=flags["target_above"],
            unsafe_descent=flags["unsafe_descent"],
            reason=flags["reason"],
            revision=str(revision),
        )

    def _replace_target_memory(
        self, assessments: list[TargetAssessment], revision: int
    ) -> None:
        chunks = []
        for item in assessments:
            chunks.append(
                actr.makechunk(
                    typename="target_model",
                    target_id=item.target.target_id,
                    role=item.target.role,
                    x=self._number(item.target.x),
                    y=self._number(item.target.y),
                    region=item.region,
                    required_order=str(item.target.required_order),
                    reachability=item.reachability,
                    actor=item.actor,
                    strategy=item.strategy,
                    cooperation=item.cooperation,
                    priority=item.priority,
                    status=item.status,
                    score=self._number(item.score),
                    distance=self._number(item.distance),
                    dx=self._number(item.dx),
                    dy=self._number(item.dy),
                    staging_x=self._number(item.staging_x),
                    staging_y=self._number(item.staging_y),
                    reason=item.reason,
                    revision=str(revision),
                )
            )
        self._replace_dynamic_memory("target_model", tuple(chunks))

    def _replace_platform_memory(self, scene: VisualScene, revision: int) -> None:
        chunks = tuple(
            actr.makechunk(
                typename="platform_model",
                platform_id=platform.platform_id,
                kind=platform.kind,
                left=self._number(platform.left),
                right=self._number(platform.right),
                top=self._number(platform.top),
                bottom=self._number(platform.bottom),
                width=self._number(platform.width),
                height=self._number(platform.height),
                region=(
                    "global"
                    if platform.kind == "boundary"
                    else self._region(platform.top, scene)
                ),
                traversability=(
                    "solid_boundary"
                    if platform.kind == "boundary"
                    else "solid_surface"
                ),
                revision=str(revision),
            )
            for platform in scene.platforms
        )
        self._replace_dynamic_memory("platform_model", chunks)

    def _replace_dynamic_memory(self, typename: str, chunks: tuple[Any, ...]) -> None:
        ext.delete_declarative_chunk_type(self._agent(), typename)
        for chunk in chunks:
            ext.add_to_declarative_memory(self._agent(), chunk)

    # ------------------------------------------------------------------
    # Geometry and temporal helpers
    # ------------------------------------------------------------------
    def _staging_point(
        self,
        target: VisualTarget,
        scene: VisualScene,
        parameters: dict[str, float],
    ) -> tuple[float, float]:
        """Return a collision-aware and cognitively stable take-off point.

        The former implementation always used the left side of a ledge and
        accepted a 2.4-unit alignment window.  In the uploaded history that
        allowed the circle to jump while its bounding box was almost touching
        the priority ledge, so the jump repeatedly hit the side/underside and
        landed back on the same floor.  A narrow dedicated take-off tolerance
        is used by the affordance stage, while this method keeps a stable side
        and lengthens the runway after retrieved failed-jump episodes.
        """
        support = self._target_support(target, scene)
        if target.role == "cooperative_stack" and support is not None:
            floor_y = (
                scene.upper_floor_y
                if self._region(target.y, scene) == "upper"
                else scene.lower_floor_y
            )
            return (
                max(4.0, support.left - parameters["cooperation_stage_offset"]),
                floor_y,
            )
        if support is not None and support.kind not in {
            "upper_floor",
            "lower_floor",
            "boundary",
            "ceiling_boundary",
        }:
            floor_y = (
                scene.upper_floor_y
                if self._region(target.y, scene) == "upper"
                else scene.lower_floor_y
            )
            progress = self._buffer_chunk("imaginal_progress")
            recovery = self._buffer_chunk("imaginal_recovery")
            same_progress_target = (
                self._chunk_value(progress, "target_id", NONE) == target.target_id
            )
            approach_side = (
                self._chunk_value(progress, "approach_side", NONE)
                if same_progress_target
                else NONE
            )
            recovery_attempts = 0
            if self._chunk_value(recovery, "target_id", NONE) == target.target_id:
                recovery_attempts = self._integer(
                    self._chunk_value(recovery, "attempts", "0"), 0
                )
            runway = parameters["ledge_takeoff_offset"] + min(
                3.6,
                recovery_attempts * parameters["recovery_runway_increment"],
            )
            support_center = (support.left + support.right) / 2.0
            # The priority ledge is immediately before an irreversible drop.
            # Always approach it from the safe interior side.  Other ledges
            # retain the previous side to avoid left/right policy chatter.
            if target.role == "priority_before_drop":
                approach_side = "left"
            elif approach_side not in {"left", "right"}:
                approach_side = (
                    "left" if scene.self_agent.x <= support_center else "right"
                )
            half_width = max(0.1, scene.self_agent.width / 2.0)
            priming_clearance = parameters["jump_priming_clearance"]
            overhead_blockers = [
                platform
                for platform in scene.platforms
                if platform.kind in {"low_ceiling", "room_wall_low_door"}
                and platform.bottom <= floor_y - 0.05
                and platform.top >= support.top - 4.0
                and platform.left <= support.right + 1.0
                and platform.right >= support.left - 1.0
            ]
            if approach_side == "right":
                world_right = max(
                    (platform.right for platform in scene.platforms),
                    default=support.right + runway + 4.0,
                )
                stage_x = min(world_right - 4.0, support.right + runway)
                if overhead_blockers:
                    # The priming key moves the body before W fires.  Keep the
                    # complete circle clear of the nearest underside during
                    # that latency, otherwise the rise is cancelled by AABB
                    # collision before the jump can arc around the obstacle.
                    stage_x = max(
                        stage_x,
                        max(blocker.right for blocker in overhead_blockers)
                        + half_width + priming_clearance,
                    )
            else:
                stage_x = max(4.0, support.left - runway)
                if overhead_blockers:
                    stage_x = min(
                        stage_x,
                        min(blocker.left for blocker in overhead_blockers)
                        - half_width - priming_clearance,
                    )
                    stage_x = max(4.0, stage_x)
            return stage_x, floor_y
        return target.x, target.y

    @staticmethod
    def _target_support(
        target: VisualTarget, scene: VisualScene
    ) -> VisualPlatform | None:
        candidates = [
            platform
            for platform in scene.platforms
            if platform.left - 0.6 <= target.x <= platform.right + 0.6
            and abs((platform.top - 1.5) - target.y) <= 1.35
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda platform: abs((platform.top - 1.5) - target.y),
        )

    @staticmethod
    def _partner_blocks_route(
        scene: VisualScene,
        destination_x: float,
        parameters: dict[str, float],
    ) -> bool:
        partner = scene.other_agent
        self_agent = scene.self_agent
        if abs(partner.y - self_agent.y) > parameters["corridor_vertical_tolerance"]:
            return False
        if not ExampleAdapter._between(
            partner.x,
            self_agent.x,
            destination_x,
            margin=partner.width * 0.3,
        ):
            return False
        return abs(partner.x - self_agent.x) <= parameters["blocking_distance"]

    @staticmethod
    def _route_obstacle_nearby(scene: VisualScene, delta: float) -> bool:
        """Detect a visually represented low ceiling blocking floor travel."""
        body = scene.self_agent
        for platform in scene.platforms:
            if platform.kind != "low_ceiling":
                continue
            if delta > 0:
                gap = platform.left - body.right
            else:
                gap = body.left - platform.right
            if -0.75 <= gap <= 5.0:
                return True
        return False

    def _agent_support(
        self,
        agent: VisualAgent,
        scene: VisualScene,
        parameters: dict[str, float],
    ) -> str:
        tolerance = parameters["ground_contact_tolerance"]
        other = (
            scene.other_agent
            if agent.symbol == scene.self_agent.symbol
            else scene.self_agent
        )
        if (
            agent.right > other.left
            and agent.left < other.right
            and abs(agent.bottom - other.top) <= tolerance
        ):
            return f"agent_{other.symbol}"
        candidates = [
            platform
            for platform in scene.platforms
            if agent.right > platform.left
            and agent.left < platform.right
            and abs(agent.bottom - platform.top) <= tolerance
        ]
        if not candidates:
            return NONE
        support = min(candidates, key=lambda item: abs(agent.bottom - item.top))
        return f"platform_{support.platform_id}"

    def _is_grounded(
        self,
        agent: VisualAgent,
        scene: VisualScene,
        parameters: dict[str, float],
    ) -> bool:
        return self._agent_support(agent, scene, parameters) != NONE

    @staticmethod
    def _transform(
        agent: VisualAgent, parameters: dict[str, float]
    ) -> float:
        if agent.shape != "rectangle":
            return 0.0
        low = parameters["rectangle_min_height"]
        high = agent.max_height or parameters["rectangle_max_height"]
        return _clamp((agent.height - low) / max(0.001, high - low), 0.0, 1.0)

    def _stagnation(self, candidate: TargetAssessment | None) -> int:
        """Return prior route-subgoal stagnation, not raw target distance.

        Moving toward a take-off point can temporarily increase Euclidean
        distance to an elevated target.  Conversely, a failed jump can reduce
        target distance while still returning to the same support.  The
        progress imaginal already represents the appropriate route metric.
        """
        previous = self._buffer_chunk("imaginal_progress")
        if candidate is None or previous is None:
            return 0
        if self._chunk_value(previous, "target_id", NONE) != candidate.target.target_id:
            return 0
        return self._integer(
            self._chunk_value(previous, "no_progress_cycles", "0"), 0
        )

    @staticmethod
    def _ensure_actionable_affordance(flags: dict[str, str]) -> None:
        """Prevent a production dead end when perception catches an in-between frame."""
        action_slots = (
            "transform_up",
            "transform_down",
            "move_left",
            "move_right",
            "prime_jump_left",
            "prime_jump_right",
            "fast_fall",
            "wait",
            "replan",
        )
        if not any(flags.get(slot) == YES for slot in action_slots):
            flags["wait"] = YES
            flags["reason"] = "observe_transient_motion_before_reassessment"

    @staticmethod
    def _set_horizontal(
        flags: dict[str, str], delta: float, tolerance: float
    ) -> None:
        direction = _direction(delta, tolerance=tolerance)
        if direction == "left":
            flags["move_left"] = YES
        elif direction == "right":
            flags["move_right"] = YES
        else:
            flags["aligned"] = YES

    @staticmethod
    def _region(y: float, scene: VisualScene) -> str:
        return "upper" if y < scene.region_midpoint else "lower"

    @staticmethod
    def _motion(dx: float, dy: float) -> str:
        if abs(dy) > 0.25:
            return "rising" if dy < 0.0 else "falling"
        if abs(dx) > 0.2:
            return "moving_left" if dx < 0.0 else "moving_right"
        return "stationary"

    @staticmethod
    def _vertical_direction(delta: float, tolerance: float) -> str:
        if delta < -abs(tolerance):
            return "above"
        if delta > abs(tolerance):
            return "below"
        return "level"

    # ------------------------------------------------------------------
    # ACT-R access helpers: buffers and chunks only
    # ------------------------------------------------------------------
    def _agent(self):
        if self.agent_construct is None:
            raise RuntimeError("adapter_not_attached")
        return self.agent_construct

    def _buffer_chunk(self, name: str):
        buffer = ext.get_buffer(self._agent(), name)
        if buffer is None:
            return None
        try:
            return next(iter(buffer), None)
        except TypeError:
            return None

    @staticmethod
    def _chunk_value(chunk: Any, slot: str, default: str) -> str:
        if chunk is None:
            return default
        try:
            value = getattr(chunk, slot)
            value = getattr(value, "values", value)
            if isinstance(value, actr.chunks.Chunk.EmptyValue):
                return default
            return str(value)
        except (AttributeError, TypeError):
            return default

    def _float_chunk(self, chunk: Any, slot: str, default: float) -> float:
        value = self._chunk_value(chunk, slot, NONE)
        if value == NONE:
            return float(default)
        try:
            return _as_float(value)
        except ValueError:
            return float(default)

    @staticmethod
    def _integer(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _number(value: Any) -> str:
        return f"{float(value):.3f}"

    @staticmethod
    def _symbolic_error(exc: Exception) -> str:
        text = f"{type(exc).__name__}_{exc}".lower()
        return "".join(character if character.isalnum() else "_" for character in text)[:96]

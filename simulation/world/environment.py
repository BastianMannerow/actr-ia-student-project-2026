"""Continuous, gravity-driven cooperative platform backend."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable
from typing import Any, Optional

from simulation.world.entities import (
    AvatarShape,
    Diamond,
    LevelDescriptor,
    PhysicsBody,
    Platform,
    SpatialAgent,
)
from simulation.world.level_builder import build_level


_EPSILON = 1e-7


class Environment:
    """Small deterministic physics world optimized for two cooperative agents.

    The backend intentionally avoids a general-purpose physics dependency. The
    scene contains only two dynamic bodies and a small set of axis-aligned
    solids, so a fixed-step solver is both faster and easier to keep
    deterministic for ACT-R and batch execution.
    """

    backend_name = "continuous"
    PHYSICS_HZ = 120.0
    FIXED_DT = 1.0 / PHYSICS_HZ
    GRAVITY = 28.0
    MAX_FALL_SPEED = 32.0
    CIRCLE_MOVE_SPEED = 10.5
    RECTANGLE_MOVE_SPEED = 8.0
    GROUND_ACCELERATION = 65.0
    AIR_ACCELERATION = 31.0
    GROUND_FRICTION = 44.0
    RECTANGLE_MIN_WIDTH = 0.9
    RECTANGLE_MAX_HEIGHT = 20.0
    RECTANGLE_MIN_HEIGHT = 2.6
    TRANSFORM_RATE = 1.15

    def __init__(
        self,
        level: LevelDescriptor,
        agents: list[Any],
        gui: Optional[Any] = None,
        *,
        simulation: Any | None = None,
        level_type: str = "cooperative_platforms",
    ) -> None:
        if len(agents) != 2:
            raise ValueError("The cooperative platform world requires exactly two agents.")
        self.level_type = str(level_type)
        self.level = level
        self.gui = gui
        self.simulation = simulation
        self.level_number = 1
        self.level_name = level.title
        self.width = float(level.width)
        self.height = float(level.height)
        self.platforms: list[Platform] = list(level.platforms)
        self.diamonds: list[Diamond] = list(level.diamonds)
        self.agents: list[Any] = list(agents)
        self.world_revision = 0
        self.static_revision = 1
        self._render_full_redraw = True
        self._render_dirty = True
        self._accumulator = 0.0
        self._world_time = 0.0
        self._last_advance_wall = time.perf_counter()
        self._rng = random.Random(level.seed)
        self._pressed_keys: dict[int, set[str]] = {id(agent): set() for agent in agents}
        self._collected_total = 0
        self._level_completed_count = 0
        self._assign_random_roles_and_spawn()
        self._settle_initial_bodies()

    # ------------------------------------------------------------------
    # Compatibility and inspection API
    # ------------------------------------------------------------------
    @property
    def agent_count(self) -> int:
        return len(self.agents)

    @property
    def remaining_diamonds(self) -> int:
        return sum(not diamond.collected for diamond in self.diamonds)

    @property
    def total_diamonds(self) -> int:
        return len(self.diamonds)

    @property
    def collected_diamonds(self) -> int:
        return self.total_diamonds - self.remaining_diamonds

    @property
    def world_time(self) -> float:
        return self._world_time

    @property
    def level_matrix(self) -> None:
        """Explicitly signal that this backend is not matrix based."""
        return None

    def agent_by_name(self, name: str) -> Any | None:
        target = str(name)
        return next((agent for agent in self.agents if str(getattr(agent, "name", "")) == target), None)

    def positioned_agents(self) -> tuple[tuple[Any, tuple[float, float]], ...]:
        return tuple((agent, (agent.body.y, agent.body.x)) for agent in self.agents)

    def find_agent(self, agent: Any) -> tuple[float, float] | None:
        if agent not in self.agents:
            return None
        return (float(agent.body.y), float(agent.body.x))

    def target_positions(self) -> list[tuple[float, float]]:
        return [(diamond.y, diamond.x) for diamond in self.diamonds if not diamond.collected]

    def consume_render_changes(self) -> set[tuple[int, int]] | None:
        if self._render_full_redraw:
            self._render_full_redraw = False
            self._render_dirty = False
            return None
        if self._render_dirty:
            self._render_dirty = False
            return {(0, 0)}
        return set()

    def mark_static_changed(self) -> None:
        self.static_revision += 1
        self.world_revision += 1
        self._render_full_redraw = True
        self._mark_all_perception_dirty()
        self._update_gui()

    # ------------------------------------------------------------------
    # Level lifecycle
    # ------------------------------------------------------------------
    def _assign_random_roles_and_spawn(self) -> None:
        role_order = [AvatarShape.CIRCLE, AvatarShape.RECTANGLE]
        self._rng.shuffle(role_order)
        width_low, width_high = getattr(
            self.level, "rectangle_width_range", (8.0, 14.0)
        )
        rectangle_width = self._rng.uniform(float(width_low), float(width_high))
        rectangle_max_height = float(
            getattr(self.level, "rectangle_max_height", self.RECTANGLE_MAX_HEIGHT)
        )
        rectangle_min_width = float(
            getattr(self.level, "rectangle_min_width", self.RECTANGLE_MIN_WIDTH)
        )
        for index, agent in enumerate(self.agents):
            spawn_x, spawn_bottom = self.level.spawn_points[index]
            agent.configure_avatar(
                role_order[index],
                x=spawn_x,
                bottom=spawn_bottom,
                rectangle_width=rectangle_width,
                rectangle_max_height=rectangle_max_height,
                rectangle_min_width=rectangle_min_width,
            )
            # Wide rectangles must not overlap the full-height boundary at
            # spawn; otherwise vertical collision resolution places them on
            # top of the wall and the level immediately becomes unstable.
            left_wall = max(
                (platform.right for platform in self.platforms if platform.kind == "boundary" and platform.left <= 0.0 and platform.width <= 3.0),
                default=0.0,
            )
            right_wall = min(
                (platform.left for platform in self.platforms if platform.kind == "boundary" and platform.right >= self.width and platform.width <= 3.0),
                default=self.width,
            )
            agent.body.x = min(
                right_wall - agent.body.width / 2.0 - 0.2,
                max(left_wall + agent.body.width / 2.0 + 0.2, agent.body.x),
            )
            agent.body.previous_x = agent.body.x
            agent.collected_diamonds = 0
        self._mark_all_perception_dirty()

    def _settle_initial_bodies(self) -> None:
        for _ in range(5):
            self._physics_step(self.FIXED_DT, allow_level_advance=False)

    def start_next_level(self) -> None:
        self.level_number += 1
        next_seed = self._rng.randrange(1, 2**31)
        self.level = build_level(
            self.level_type,
            self.agents,
            level_number=self.level_number,
            seed=next_seed,
        )
        self.level_name = self.level.title
        self.width = float(self.level.width)
        self.height = float(self.level.height)
        self.platforms = list(self.level.platforms)
        self.diamonds = list(self.level.diamonds)
        self._pressed_keys = {id(agent): set() for agent in self.agents}
        self._assign_random_roles_and_spawn()
        self._accumulator = 0.0
        self.static_revision += 1
        self.world_revision += 1
        self._render_full_redraw = True
        self._settle_initial_bodies()
        callback = getattr(self.simulation, "on_level_started", None)
        if callable(callback):
            callback(self.level_number, self.level.seed)
        self._update_gui()

    def restart_current_level(self) -> None:
        self.level = build_level(
            self.level_type,
            self.agents,
            level_number=self.level_number,
            seed=self.level.seed,
        )
        self.level_name = self.level.title
        self.platforms = list(self.level.platforms)
        self.diamonds = list(self.level.diamonds)
        self._pressed_keys = {id(agent): set() for agent in self.agents}
        self._assign_random_roles_and_spawn()
        self._accumulator = 0.0
        self.static_revision += 1
        self.world_revision += 1
        self._render_full_redraw = True
        self._settle_initial_bodies()
        callback = getattr(self.simulation, "on_level_restarted", None)
        if callable(callback):
            callback(self.level_number, self.level.seed)
        self._update_gui()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def set_control_key(self, agent: Any, key: str, pressed: bool) -> bool:
        if agent not in self.agents:
            return False
        normalized = str(key).upper()
        if normalized not in {"W", "A", "S", "D"}:
            return False
        keys = self._pressed_keys.setdefault(id(agent), set())
        was_pressed = normalized in keys
        if pressed:
            keys.add(normalized)
            if (
                normalized == "W"
                and not was_pressed
                and agent.avatar_shape is AvatarShape.CIRCLE
            ):
                self._try_circle_jump(agent)
        else:
            keys.discard(normalized)
        return True

    def apply_action(self, agent: Any, key: str) -> bool:
        """Apply one discrete ACT-R motor command."""
        if agent not in self.agents:
            return False
        key = str(key).upper()
        body = agent.body
        if key == "R":
            # Cognitive agents are never allowed to reset the level through
            # the Manual buffer. Human/UI restart remains available through
            # Simulation.restart_level().
            return False
        if key == "A":
            body.input_x = -1.0
            body.input_until = self._world_time + 0.12
            return True
        if key == "D":
            body.input_x = 1.0
            body.input_until = self._world_time + 0.12
            return True
        if key == "W":
            if agent.avatar_shape is AvatarShape.CIRCLE:
                return self._try_circle_jump(agent)
            return self._change_rectangle_transform(agent, +0.20)
        if key == "S":
            if agent.avatar_shape is AvatarShape.CIRCLE:
                if not body.grounded:
                    body.vy = min(self.MAX_FALL_SPEED, body.vy + 5.5)
                else:
                    body.vx *= 0.35
                return True
            return self._change_rectangle_transform(agent, -0.20)
        return False

    def move_agent_top(self, agent: Any) -> bool:
        return self.apply_action(agent, "W")

    def move_agent_bottom(self, agent: Any) -> bool:
        return self.apply_action(agent, "S")

    def move_agent_left(self, agent: Any) -> bool:
        return self.apply_action(agent, "A")

    def move_agent_right(self, agent: Any) -> bool:
        return self.apply_action(agent, "D")

    def _try_circle_jump(self, agent: Any) -> bool:
        body = agent.body
        if agent.avatar_shape is not AvatarShape.CIRCLE or not body.grounded:
            self.register_bumping(agent, reason="jump_requires_ground")
            return False
        body.vy = -self._rng.uniform(20.8, 23.4)
        direction = self._desired_horizontal_input(agent)
        if abs(direction) < 0.1:
            direction = self._rng.choice((-1.0, 1.0))
            horizontal = self._rng.uniform(1.2, 3.6)
        else:
            horizontal = self._rng.uniform(2.0, 4.8)
        body.vx += direction * horizontal
        # Preserve directional air control long enough to clear the side of a
        # low platform.  Without this, horizontal collision zeroes the initial
        # impulse during the first rising frames and the circle repeatedly
        # jumps in place at tunnel/ledge edges.
        body.input_x = direction
        body.input_until = max(body.input_until, self._world_time + 0.62)
        body.grounded = False
        body.support = None
        self._touch_world()
        return True

    def _change_rectangle_transform(self, agent: Any, delta: float) -> bool:
        if agent.avatar_shape is not AvatarShape.RECTANGLE:
            return False
        body = agent.body
        old = (body.x, body.y, body.width, body.height, body.transform)
        bottom = body.bottom
        body.transform = min(1.0, max(0.0, body.transform + float(delta)))
        self._apply_rectangle_dimensions(body, bottom)
        if self._body_overlaps_static(body):
            body.x, body.y, body.width, body.height, body.transform = old
            self.register_bumping(agent, reason="transform_blocked")
            return False
        self._lift_circle_from_rectangle(agent, old_top=old[1] - old[3] / 2.0)
        self._touch_world()
        return True

    def _apply_rectangle_dimensions(self, body: PhysicsBody, bottom: float) -> None:
        """Change only the rectangle's aspect ratio while preserving area.

        ``transform`` controls height. Width is derived from the immutable
        rectangle area, so making the body taller necessarily makes it thinner.
        The effective height ceiling is whichever limit is reached first:
        ``max_height`` or ``min_width``.
        """
        t = min(1.0, max(0.0, float(body.transform)))
        base_height = max(0.1, float(body.base_height))
        area = max(0.1, float(body.rectangle_area))
        min_width = max(0.1, float(body.min_width))
        effective_max_height = body.effective_max_height
        body.height = base_height + (effective_max_height - base_height) * t
        body.width = area / body.height
        body.set_bottom(bottom)

    def _lift_circle_from_rectangle(self, rectangle_agent: Any, *, old_top: float) -> None:
        rectangle = rectangle_agent.body
        for candidate in self.agents:
            if candidate is rectangle_agent or candidate.avatar_shape is not AvatarShape.CIRCLE:
                continue
            circle = candidate.body
            horizontally_supported = circle.right > rectangle.left and circle.left < rectangle.right
            was_on_top = abs(circle.bottom - old_top) <= 0.7 or circle.support is rectangle_agent
            if horizontally_supported and was_on_top:
                circle.set_bottom(rectangle.top)
                circle.vy = min(circle.vy, 0.0)
                circle.grounded = True
                circle.support = rectangle_agent

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def advance(self, seconds: float) -> bool:
        """Advance with a fixed timestep; returns whether visible state changed."""
        seconds = max(0.0, min(float(seconds), 0.5))
        if seconds <= 0.0:
            return False
        self._accumulator = min(self._accumulator + seconds, 0.5)
        changed = False
        while self._accumulator + _EPSILON >= self.FIXED_DT:
            changed = self._physics_step(self.FIXED_DT) or changed
            self._accumulator = max(0.0, self._accumulator - self.FIXED_DT)
        if changed:
            self._update_gui()
        return changed

    def advance_wall_clock(self) -> bool:
        now = time.perf_counter()
        elapsed = min(0.05, max(0.0, now - self._last_advance_wall))
        self._last_advance_wall = now
        return self.advance(elapsed)

    def _physics_step(self, dt: float, *, allow_level_advance: bool = True) -> bool:
        before = [
            (agent.body.x, agent.body.y, agent.body.width, agent.body.height)
            for agent in self.agents
        ]
        self._world_time += dt
        for agent in self.agents:
            self._apply_continuous_controls(agent, dt)
            self._integrate_agent(agent, dt)
        self._resolve_agent_pair()
        self._collect_overlapping_diamonds()

        changed = any(
            abs(agent.body.x - previous[0]) > 1e-4
            or abs(agent.body.y - previous[1]) > 1e-4
            or abs(agent.body.width - previous[2]) > 1e-4
            or abs(agent.body.height - previous[3]) > 1e-4
            for agent, previous in zip(self.agents, before)
        )
        if changed:
            self._touch_world()

        if allow_level_advance and self.diamonds and self.remaining_diamonds == 0:
            self._level_completed_count += 1
            callback = getattr(self.simulation, "on_level_completed", None)
            if callable(callback):
                callback(self.level_number, self.level.seed)
            self.start_next_level()
            return True
        return changed

    def _apply_continuous_controls(self, agent: Any, dt: float) -> None:
        body = agent.body
        keys = self._pressed_keys.get(id(agent), set())
        keyboard_x = float("D" in keys) - float("A" in keys)
        if keyboard_x:
            desired = keyboard_x
        elif self._world_time <= body.input_until:
            desired = body.input_x
        else:
            # Stop active acceleration but retain the last intended direction.
            # A later W command can therefore form a deliberate left/right jump
            # after pyACT-R's manual module has completed the priming key.
            desired = 0.0

        target_speed = (
            self.CIRCLE_MOVE_SPEED
            if agent.avatar_shape is AvatarShape.CIRCLE
            else self.RECTANGLE_MOVE_SPEED
        ) * desired
        acceleration = self.GROUND_ACCELERATION if body.grounded else self.AIR_ACCELERATION
        body.vx = self._approach(body.vx, target_speed, acceleration * dt)
        if desired == 0.0 and body.grounded:
            body.vx = self._approach(body.vx, 0.0, self.GROUND_FRICTION * dt)

        if agent.avatar_shape is AvatarShape.RECTANGLE:
            transform_direction = float("W" in keys) - float("S" in keys)
            if transform_direction:
                self._change_rectangle_transform(agent, transform_direction * self.TRANSFORM_RATE * dt)
        elif "S" in keys and not body.grounded:
            body.vy = min(self.MAX_FALL_SPEED, body.vy + 22.0 * dt)

    def _integrate_agent(self, agent: Any, dt: float) -> None:
        body = agent.body
        body.previous_x = body.x
        body.previous_y = body.y
        body.vy = min(self.MAX_FALL_SPEED, body.vy + self.GRAVITY * dt)
        body.grounded = False
        body.support = None

        body.x += body.vx * dt
        self._resolve_static_horizontal(agent)
        body.y += body.vy * dt
        self._resolve_static_vertical(agent)

        if body.top > self.height + 10.0:
            # Out-of-bounds recovery is physical, not a level reset. This keeps
            # a failed agent inside the current episode so cognition can detect
            # and recover from the resulting lack of progress.
            floor = next(
                (
                    platform
                    for platform in self.platforms
                    if platform.kind == "boundary"
                    and platform.width >= self.width * 0.8
                    and platform.top >= self.height * 0.7
                ),
                None,
            )
            floor_top = floor.top if floor is not None else self.height - 2.0
            body.set_bottom(floor_top)
            body.vx = 0.0
            body.vy = 0.0
            body.grounded = True
            body.support = floor
            self.register_bumping(agent, reason="out_of_bounds_clamped")

    def _resolve_static_horizontal(self, agent: Any) -> None:
        body = agent.body
        for platform in self._candidate_platforms(body):
            if not self._aabb_overlap(body, platform):
                continue
            if body.vx > 0.0:
                body.x = platform.left - body.width / 2.0
            elif body.vx < 0.0:
                body.x = platform.right + body.width / 2.0
            else:
                continue
            body.vx = 0.0
            self.register_bumping(agent, reason="solid_surface")

    def _resolve_static_vertical(self, agent: Any) -> None:
        body = agent.body
        previous_top = body.previous_y - body.height / 2.0
        previous_bottom = body.previous_y + body.height / 2.0
        for platform in self._candidate_platforms(body):
            if not self._aabb_overlap(body, platform):
                continue
            descending_onto = body.vy >= 0.0 and previous_bottom <= platform.top + 0.35
            rising_into = body.vy < 0.0 and previous_top >= platform.bottom - 0.35
            if descending_onto:
                body.set_bottom(platform.top)
                body.vy = 0.0
                body.grounded = True
                body.support = platform
            elif rising_into:
                body.set_top(platform.bottom)
                body.vy = 0.0
            else:
                # Resolve the smallest vertical penetration as a fallback.
                down_penetration = body.bottom - platform.top
                up_penetration = platform.bottom - body.top
                if down_penetration <= up_penetration:
                    body.set_bottom(platform.top)
                    body.vy = 0.0
                    body.grounded = True
                    body.support = platform
                else:
                    body.set_top(platform.bottom)
                    body.vy = 0.0

    def _resolve_agent_pair(self) -> None:
        circle_agent = next(
            (agent for agent in self.agents if agent.avatar_shape is AvatarShape.CIRCLE),
            None,
        )
        rectangle_agent = next(
            (agent for agent in self.agents if agent.avatar_shape is AvatarShape.RECTANGLE),
            None,
        )
        if circle_agent is None or rectangle_agent is None:
            return
        circle = circle_agent.body
        rectangle = rectangle_agent.body
        if not self._aabb_overlap(circle, rectangle):
            return

        previous_circle_bottom = circle.previous_y + circle.height / 2.0
        descending_onto = circle.vy >= rectangle.vy and previous_circle_bottom <= rectangle.top + 0.55
        if descending_onto:
            circle.set_bottom(rectangle.top)
            circle.vy = min(0.0, rectangle.vy)
            circle.grounded = True
            circle.support = rectangle_agent
            # A moving rectangle carries the circle horizontally.
            circle.x += rectangle.vx * self.FIXED_DT
            return

        horizontal_penetration = min(circle.right - rectangle.left, rectangle.right - circle.left)
        vertical_penetration = min(circle.bottom - rectangle.top, rectangle.bottom - circle.top)
        if horizontal_penetration < vertical_penetration:
            direction = -1.0 if circle.x < rectangle.x else 1.0
            circle.x += direction * horizontal_penetration
            circle.vx = rectangle.vx
        elif circle.y < rectangle.y:
            circle.set_bottom(rectangle.top)
            circle.vy = min(circle.vy, 0.0)
            circle.grounded = True
            circle.support = rectangle_agent
        else:
            circle.set_top(rectangle.bottom)
            circle.vy = max(circle.vy, 0.0)

    def _collect_overlapping_diamonds(self) -> None:
        for diamond in self.diamonds:
            if diamond.collected:
                continue
            collector = next(
                (agent for agent in self.agents if self._diamond_overlap(agent.body, diamond)),
                None,
            )
            if collector is None:
                continue
            diamond.collected = True
            collector.collected_diamonds = int(getattr(collector, "collected_diamonds", 0)) + 1
            self._collected_total += 1
            self.world_revision += 1
            self._render_dirty = True
            self._mark_all_perception_dirty()
            callback = getattr(self.simulation, "on_diamond_collected", None)
            if callable(callback):
                callback(collector, diamond)

    # ------------------------------------------------------------------
    # Perception helpers
    # ------------------------------------------------------------------
    def perceptual_objects(self, observer: Any, los: float) -> list[dict[str, Any]]:
        """Return exact objects plus representative points for large solids."""
        if observer not in self.agents:
            return []
        body = observer.body
        radius = float(los)
        full_world = radius <= 0.0
        records: list[dict[str, Any]] = []

        for agent in self.agents:
            candidate = agent.body
            if not full_world and self._distance(body.x, body.y, candidate.x, candidate.y) > radius:
                continue
            records.append(
                {
                    "kind": "agent",
                    "entity": agent,
                    "x": candidate.x,
                    "y": candidate.y,
                    "width": candidate.width,
                    "height": candidate.height,
                    "shape": agent.avatar_shape.value,
                }
            )

        for diamond in self.diamonds:
            if diamond.collected:
                continue
            if not full_world and self._distance(body.x, body.y, diamond.x, diamond.y) > radius:
                continue
            records.append(
                {
                    "kind": "diamond",
                    "entity": diamond,
                    "x": diamond.x,
                    "y": diamond.y,
                    "role": diamond.role,
                }
            )

        for platform_index, platform in enumerate(self.platforms):
            nearest_x = min(max(body.x, platform.left), platform.right)
            nearest_y = min(max(body.y, platform.top), platform.bottom)
            if not full_world and self._distance(body.x, body.y, nearest_x, nearest_y) > radius:
                continue
            # Top and bottom samples preserve the complete collision envelope in
            # the point-based ACT-R visual environment.  The adapter therefore
            # receives the same axis-aligned bounds used by the physics solver,
            # rather than inferring platform thickness from its semantic kind.
            center_x = (platform.left + platform.right) / 2.0
            for sample_name, sample_x, sample_y in (
                ("left", platform.left, platform.top),
                ("center", center_x, platform.top),
                ("right", platform.right, platform.top),
                ("bottom_left", platform.left, platform.bottom),
                ("bottom_center", center_x, platform.bottom),
                ("bottom_right", platform.right, platform.bottom),
            ):
                records.append(
                    {
                        "kind": "platform",
                        "entity": platform,
                        "x": sample_x,
                        "y": sample_y,
                        "sample": sample_name,
                        "platform_index": platform_index,
                    }
                )
        return records

    # ------------------------------------------------------------------
    # GUI and lifecycle
    # ------------------------------------------------------------------
    def set_gui(self, gui: Any) -> None:
        self.gui = gui
        self._update_gui()

    def _update_gui(self) -> None:
        if self.gui is None:
            return
        refresh = getattr(self.gui, "refresh", None)
        if callable(refresh):
            refresh()
            return
        update = getattr(self.gui, "update", None)
        if callable(update):
            update()

    def close(self) -> None:
        self._pressed_keys.clear()

    def remove_agent_from_game(self, agent: Any) -> None:
        if agent in self.agents:
            self.agents.remove(agent)
            self._pressed_keys.pop(id(agent), None)
            self._touch_world()

    def register_bumping(self, agent: Any, *, reason: str = "obstacle") -> None:
        middleman = getattr(agent, "middleman", None)
        detect = getattr(middleman, "detect_bump", None)
        if callable(detect):
            detect(agent, reason=reason)

    # ------------------------------------------------------------------
    # Serialization and utility
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "level_type": self.level_type,
            "level_number": self.level_number,
            "level_name": self.level_name,
            "scenario": getattr(self.level, "scenario", "classic"),
            "seed": self.level.seed,
            "width": self.width,
            "height": self.height,
            "drop_x": self.level.drop_x,
            "diamonds": [
                {
                    "id": diamond.diamond_id,
                    "x": diamond.x,
                    "y": diamond.y,
                    "role": diamond.role,
                    "required_order": diamond.required_order,
                    "collected": diamond.collected,
                }
                for diamond in self.diamonds
            ],
            "platforms": [
                {
                    "x": platform.x,
                    "y": platform.y,
                    "width": platform.width,
                    "height": platform.height,
                    "kind": platform.kind,
                }
                for platform in self.platforms
            ],
            "agents": [
                {
                    "name": getattr(agent, "name", None),
                    "shape": agent.avatar_shape.value,
                    "human_controlled": bool(getattr(agent, "is_human_controlled", False)),
                    "x": agent.body.x,
                    "y": agent.body.y,
                    "vx": agent.body.vx,
                    "vy": agent.body.vy,
                    "width": agent.body.width,
                    "height": agent.body.height,
                    "rectangle_max_height": (
                        agent.body.effective_max_height
                        if agent.avatar_shape is AvatarShape.RECTANGLE
                        else None
                    ),
                    "grounded": agent.body.grounded,
                    "collected_diamonds": int(getattr(agent, "collected_diamonds", 0)),
                }
                for agent in self.agents
            ],
            "guarantees": list(self.level.guarantees),
        }

    def _touch_world(self) -> None:
        self.world_revision += 1
        self._render_dirty = True
        self._mark_all_perception_dirty()

    def _mark_all_perception_dirty(self) -> None:
        candidates = list(getattr(self.simulation, "agent_list", ()) or ())
        for agent in candidates:
            marker = getattr(agent, "mark_perception_dirty", None)
            if callable(marker):
                marker()

    def _body_overlaps_static(self, body: PhysicsBody) -> bool:
        return any(self._aabb_overlap(body, platform) for platform in self._candidate_platforms(body))

    def _candidate_platforms(self, body: PhysicsBody) -> Iterable[Platform]:
        # With <20 platforms a linear scan is faster than maintaining a broad
        # phase index and avoids allocation in the 120 Hz hot path.
        del body
        return self.platforms

    @staticmethod
    def _aabb_overlap(first: Any, second: Any) -> bool:
        return (
            first.right > second.left + _EPSILON
            and first.left < second.right - _EPSILON
            and first.bottom > second.top + _EPSILON
            and first.top < second.bottom - _EPSILON
        )

    @staticmethod
    def _diamond_overlap(body: PhysicsBody, diamond: Diamond) -> bool:
        nearest_x = min(max(diamond.x, body.left), body.right)
        nearest_y = min(max(diamond.y, body.top), body.bottom)
        dx = diamond.x - nearest_x
        dy = diamond.y - nearest_y
        return dx * dx + dy * dy <= (diamond.radius + 0.28) ** 2

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _approach(value: float, target: float, maximum_delta: float) -> float:
        if value < target:
            return min(target, value + maximum_delta)
        return max(target, value - maximum_delta)

    def _desired_horizontal_input(self, agent: Any) -> float:
        keys = self._pressed_keys.get(id(agent), set())
        if "A" in keys and "D" not in keys:
            return -1.0
        if "D" in keys and "A" not in keys:
            return 1.0
        body = agent.body
        if self._world_time <= body.input_until:
            return body.input_x
        if abs(body.vx) > 0.4:
            return math.copysign(1.0, body.vx)
        if abs(body.input_x) > 0.1:
            return math.copysign(1.0, body.input_x)
        return 0.0

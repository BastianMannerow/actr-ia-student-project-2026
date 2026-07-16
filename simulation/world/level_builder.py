"""Validated procedural generation for diverse cooperative platform scenarios.

The generator uses scenario grammars rather than unconstrained scatter.  Every
scenario randomizes geometry while retaining explicit solvability invariants for
both random avatar assignments.  The two agents can start together, opposed,
in separate chambers, or on different vertical tiers before reuniting.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from simulation.world.entities import Diamond, LevelDescriptor, Platform


LEVEL_DIMENSIONS: dict[str, tuple[int, int]] = {
    "cooperative_platforms": (68, 120),  # (height, width) in world units
}

SCENARIOS = (
    "paired_left",
    "paired_right",
    "opposed_upper",
    "separate_chambers",
    "central_split",
    "giant_lift",
    "vertical_reunion",
    "vertical_reunion_far",
)


def level_dimensions(level_type: str) -> tuple[int, int]:
    try:
        return LEVEL_DIMENSIONS[level_type]
    except KeyError as exc:
        raise ValueError(f"Unknown virtual level: {level_type!r}") from exc


def build_level(
    level_type: str,
    agents: Sequence[Any] | None = None,
    rng: Optional[random.Random] = None,
    *,
    level_number: int = 1,
    seed: int | None = None,
) -> LevelDescriptor:
    del agents
    if level_type != "cooperative_platforms":
        raise ValueError(f"Unknown virtual level: {level_type!r}")
    local_seed = int(seed if seed is not None else random.SystemRandom().randrange(1, 2**31))
    generator = rng or random.Random(local_seed)
    descriptor = _cooperative_platform_level(generator, local_seed, level_number)
    validate_level(descriptor)
    return descriptor


def _cooperative_platform_level(
    rng: random.Random,
    seed: int,
    level_number: int,
) -> LevelDescriptor:
    # A seed-dependent offset plus the level number prevents successive levels
    # from repeatedly selecting the same start arrangement while remaining
    # deterministic for replay/history exports.
    scenario_offset = seed % len(SCENARIOS)
    scenario = SCENARIOS[(scenario_offset + max(0, level_number - 1)) % len(SCENARIOS)]
    if scenario.startswith("vertical_reunion"):
        return _vertical_reunion_level(rng, seed, level_number, scenario)
    return _upper_route_level(rng, seed, level_number, scenario)


def _boundaries(width: float, height: float, wall: float = 2.0) -> list[Platform]:
    return [
        Platform(0.0, 0.0, wall, height, kind="boundary"),
        Platform(width - wall, 0.0, wall, height, kind="boundary"),
        Platform(0.0, height - wall, width, wall, kind="boundary"),
        Platform(0.0, 0.0, width, wall, kind="ceiling_boundary"),
    ]


def _upper_route_level(
    rng: random.Random,
    seed: int,
    level_number: int,
    scenario: str,
) -> LevelDescriptor:
    height_i, width_i = level_dimensions("cooperative_platforms")
    height, width = float(height_i), float(width_i)
    giant_lift = scenario == "giant_lift"
    upper_y = rng.uniform(32.0, 35.5) if giant_lift else rng.uniform(24.5, 29.0)
    lower_y = rng.uniform(63.0, 65.0) if giant_lift else rng.uniform(59.0, 63.5)
    drop_x = rng.uniform(57.0, 64.0)
    rectangle_max_height = (
        rng.uniform(34.0, 42.0) if giant_lift else rng.uniform(28.0, 36.0)
    )
    rectangle_width_range = (
        (rng.uniform(9.0, 10.0), rng.uniform(11.5, 12.5))
        if giant_lift
        else (rng.uniform(9.5, 11.5), rng.uniform(14.0, 18.0))
    )
    rectangle_min_width = rng.uniform(0.75, 1.15)
    guaranteed_rectangle_height = min(
        rectangle_max_height,
        rectangle_width_range[0] * 2.6 / rectangle_min_width,
    )

    platforms = _boundaries(width, height)
    platforms.extend(
        [
            Platform(0.0, upper_y, drop_x, 3.0, kind="upper_floor"),
            Platform(drop_x - 3.0, lower_y, width - drop_x + 3.0, 3.0, kind="lower_floor"),
        ]
    )

    solo_x = rng.uniform(4.8, 6.5) if giant_lift else rng.uniform(8.0, 13.0)
    solo_h = rng.uniform(6.5, 9.0) if giant_lift else rng.uniform(5.8, 8.2)
    solo_w = rng.uniform(5.0, 6.0) if giant_lift else rng.uniform(6.0, 8.5)
    platforms.append(Platform(solo_x, upper_y - solo_h, solo_w, 1.4, kind="solo_ledge"))

    coop_x = (
        rng.uniform(39.0, 41.0)
        if giant_lift
        else rng.uniform(max(29.0, solo_x + solo_w + 9.5), 36.0)
    )
    coop_gap = rng.uniform(
        16.0,
        min(
            upper_y - 4.0,
            27.0 if giant_lift else 22.0,
            guaranteed_rectangle_height + 8.5,
        ),
    )
    coop_w = rng.uniform(6.5, 9.0)
    platforms.append(
        Platform(coop_x, upper_y - coop_gap, coop_w, 1.5, kind="cooperation_ledge")
    )

    tunnel_x = rng.uniform(max(42.0, coop_x + coop_w + 2.5), 49.0)
    tunnel_w = rng.uniform(8.0, 11.0)
    platforms.append(Platform(tunnel_x, upper_y - 4.4, tunnel_w, 1.0, kind="low_ceiling"))

    priority_x = min(drop_x - 5.5, max(tunnel_x + 1.5, drop_x - rng.uniform(10.0, 15.0)))
    priority_y = upper_y - rng.uniform(5.8, 8.0)
    priority_w = rng.uniform(5.5, 7.5)
    platforms.append(
        Platform(priority_x, priority_y, priority_w, 1.3, kind="priority_ledge")
    )

    lower_solo_x = rng.uniform(drop_x + 10.0, drop_x + 18.0)
    lower_solo_gap = rng.uniform(5.5, 8.0)
    lower_solo_w = rng.uniform(7.0, 9.5)
    platforms.append(
        Platform(
            lower_solo_x,
            lower_y - lower_solo_gap,
            lower_solo_w,
            1.4,
            kind="lower_solo_ledge",
        )
    )
    lower_coop_x = rng.uniform(max(91.0, lower_solo_x + lower_solo_w + 9.0), 104.0)
    lower_coop_gap = rng.uniform(
        25.0 if giant_lift else 18.0,
        min(
            36.0 if giant_lift else 30.0,
            guaranteed_rectangle_height + 8.5,
        ),
    )
    lower_coop_w = rng.uniform(6.5, 9.0)
    platforms.append(
        Platform(
            lower_coop_x,
            lower_y - lower_coop_gap,
            lower_coop_w,
            1.5,
            kind="lower_cooperation_ledge",
        )
    )

    # Chamber variants retain a floor-level exit that both the circle and a
    # fully flattened rectangle can traverse.  The room is therefore initially
    # enclosed in the visual scene without making random role assignment fatal.
    if scenario in {"opposed_upper", "separate_chambers", "central_split"}:
        divider_x = rng.uniform(21.0, 25.0)
        doorway_clearance = rng.uniform(4.5, 5.2)
        platforms.append(
            Platform(
                divider_x,
                2.0,
                rng.uniform(1.6, 2.3),
                upper_y - doorway_clearance - 2.0,
                kind="room_wall_low_door",
            )
        )
    if scenario in {"separate_chambers", "central_split"}:
        second_x = rng.uniform(39.0, 42.0)
        platforms.append(
            Platform(
                second_x,
                2.0,
                rng.uniform(1.6, 2.2),
                upper_y - rng.uniform(4.7, 5.4) - 2.0,
                kind="room_wall_low_door",
            )
        )

    diamonds = [
        Diamond(solo_x + solo_w * 0.52, upper_y - solo_h - 1.5, role="solo_circle", diamond_id="upper_solo"),
        Diamond(coop_x + coop_w * 0.5, upper_y - coop_gap - 1.55, role="cooperative_stack", diamond_id="upper_coop"),
        Diamond(tunnel_x + tunnel_w * 0.55, upper_y - 1.35, role="solo_flat_rectangle", diamond_id="tunnel"),
        Diamond(priority_x + priority_w * 0.48, priority_y - 1.45, role="priority_before_drop", required_order=1, diamond_id="priority"),
        Diamond(lower_solo_x + lower_solo_w * 0.45, lower_y - lower_solo_gap - 1.45, role="solo_lower", required_order=2, diamond_id="lower_solo"),
        Diamond(lower_coop_x + lower_coop_w * 0.48, lower_y - lower_coop_gap - 1.55, role="cooperative_stack", required_order=2, diamond_id="lower_coop"),
        Diamond(rng.uniform(drop_x + 7.0, drop_x + 18.0), lower_y - 1.25, role="solo_ground", required_order=2, diamond_id="lower_ground"),
    ]

    if scenario == "paired_left":
        ranges = ((5.0, 14.0), (18.0, 29.0))
    elif scenario == "paired_right":
        ranges = ((23.0, 35.0), (39.0, max(45.0, drop_x - 4.5)))
    elif scenario == "opposed_upper":
        ranges = ((5.0, 15.0), (29.0, 41.0))
    elif scenario == "central_split":
        ranges = ((7.0, 18.0), (44.0, max(48.0, drop_x - 4.5)))
    elif scenario == "giant_lift":
        ranges = ((19.0, 22.0), (29.0, 32.5))
    else:  # separate_chambers
        ranges = ((5.0, 15.0), (29.0, 39.0))
    first_x = _safe_spawn_x(
        rng,
        upper_y,
        ranges[0],
        platforms,
        diamonds,
        max_body_width=rectangle_width_range[1],
    )
    second_x = _safe_spawn_x(
        rng,
        upper_y,
        ranges[1],
        platforms,
        diamonds,
        max_body_width=rectangle_width_range[1],
        avoid=(first_x,),
    )
    spawn_points = ((first_x, upper_y), (second_x, upper_y))
    if scenario in {"opposed_upper", "central_split", "giant_lift"} and rng.random() < 0.5:
        spawn_points = tuple(reversed(spawn_points))

    return LevelDescriptor(
        width=width,
        height=height,
        platforms=platforms,
        diamonds=diamonds,
        spawn_points=spawn_points,
        drop_x=drop_x,
        upper_floor_y=upper_y,
        lower_floor_y=lower_y,
        seed=seed,
        title=f"Intelligent Agents · Level {level_number} · {scenario.replace('_', ' ').title()}",
        scenario=scenario,
        rectangle_width_range=rectangle_width_range,
        rectangle_max_height=rectangle_max_height,
        rectangle_min_width=rectangle_min_width,
        guarantees=(
            "randomized start topology",
            "solo and shape-specific collectibles",
            "mobile-platform cooperation",
            "priority collectible before irreversible descent",
            "large rectangle transformation envelope",
        ),
    )


def _vertical_reunion_level(
    rng: random.Random,
    seed: int,
    level_number: int,
    scenario: str,
) -> LevelDescriptor:
    """One agent starts above and one below; all joint work occurs after reunion."""
    height_i, width_i = level_dimensions("cooperative_platforms")
    height, width = float(height_i), float(width_i)
    upper_y = rng.uniform(23.5, 27.0)
    lower_y = rng.uniform(58.5, 62.5)
    drop_x = rng.uniform(47.0, 56.0)
    rectangle_max_height = rng.uniform(34.0, 42.0)
    rectangle_width_range = (rng.uniform(10.0, 12.0), rng.uniform(14.5, 18.0))
    rectangle_min_width = rng.uniform(0.75, 1.15)
    guaranteed_rectangle_height = min(
        rectangle_max_height,
        rectangle_width_range[0] * 2.6 / rectangle_min_width,
    )

    platforms = _boundaries(width, height)
    platforms.extend(
        [
            Platform(0.0, upper_y, drop_x, 3.0, kind="upper_floor"),
            Platform(drop_x - 3.0, lower_y, width - drop_x + 3.0, 3.0, kind="lower_floor"),
            # A visually closed upper chamber with a floor-level exit toward
            # the drop.  Either random avatar can leave it.
            Platform(rng.uniform(17.0, 21.0), 2.0, 2.0, upper_y - 6.7, kind="room_wall_low_door"),
        ]
    )

    priority_x = rng.uniform(drop_x - 12.0, drop_x - 7.0)
    # Ground-level priority target is traversable by either shape, so role
    # randomization cannot strand the upper agent.
    diamonds = [
        Diamond(priority_x, upper_y - 1.3, role="priority_before_drop", required_order=1, diamond_id="upper_exit_priority"),
    ]

    lower_solo_x = rng.uniform(drop_x + 8.0, drop_x + 15.0)
    lower_solo_gap = rng.uniform(5.0, 7.5)
    lower_solo_w = rng.uniform(7.0, 9.0)
    platforms.append(Platform(lower_solo_x, lower_y - lower_solo_gap, lower_solo_w, 1.4, kind="lower_solo_ledge"))

    lower_tunnel_x = rng.uniform(drop_x + 20.0, drop_x + 28.0)
    lower_tunnel_w = rng.uniform(9.0, 12.0)
    platforms.append(Platform(lower_tunnel_x, lower_y - 4.5, lower_tunnel_w, 1.0, kind="low_ceiling"))

    lower_coop_x = rng.uniform(88.0, 98.0)
    lower_coop_gap = rng.uniform(
        24.0, min(34.0, guaranteed_rectangle_height + 8.5)
    )
    lower_coop_w = rng.uniform(7.0, 9.5)
    platforms.append(Platform(lower_coop_x, lower_y - lower_coop_gap, lower_coop_w, 1.5, kind="lower_cooperation_ledge"))

    second_coop_x = rng.uniform(103.0, 110.0)
    second_gap = rng.uniform(
        19.0, min(29.0, guaranteed_rectangle_height + 8.5)
    )
    platforms.append(Platform(second_coop_x, lower_y - second_gap, 6.5, 1.5, kind="lower_cooperation_ledge_secondary"))

    diamonds.extend(
        [
            Diamond(lower_solo_x + lower_solo_w * 0.5, lower_y - lower_solo_gap - 1.45, role="solo_lower", required_order=2, diamond_id="lower_solo"),
            Diamond(lower_tunnel_x + lower_tunnel_w * 0.5, lower_y - 1.35, role="solo_flat_rectangle", required_order=2, diamond_id="lower_tunnel"),
            Diamond(lower_coop_x + lower_coop_w * 0.5, lower_y - lower_coop_gap - 1.55, role="cooperative_stack", required_order=2, diamond_id="lower_coop_high"),
            Diamond(second_coop_x + 3.2, lower_y - second_gap - 1.55, role="cooperative_stack", required_order=2, diamond_id="lower_coop_far"),
            Diamond(rng.uniform(drop_x + 5.0, drop_x + 13.0), lower_y - 1.25, role="solo_ground", required_order=2, diamond_id="reunion_ground"),
        ]
    )

    upper_spawn = (
        _safe_spawn_x(
            rng,
            upper_y,
            (5.0, 13.0),
            platforms,
            diamonds,
            max_body_width=rectangle_width_range[1],
        ),
        upper_y,
    )
    lower_range = (
        (drop_x + 4.0, drop_x + 18.0)
        if scenario == "vertical_reunion"
        else (max(drop_x + 18.0, 78.0), 114.0)
    )
    lower_spawn = (
        _safe_spawn_x(
            rng,
            lower_y,
            lower_range,
            platforms,
            diamonds,
            max_body_width=rectangle_width_range[1],
        ),
        lower_y,
    )
    # Randomize which runtime agent occupies the upper/lower spawn while the
    # avatar assignment remains independently random in Environment.
    spawn_points = (upper_spawn, lower_spawn)
    if rng.random() < 0.5:
        spawn_points = (lower_spawn, upper_spawn)

    return LevelDescriptor(
        width=width,
        height=height,
        platforms=platforms,
        diamonds=diamonds,
        spawn_points=spawn_points,
        drop_x=drop_x,
        upper_floor_y=upper_y,
        lower_floor_y=lower_y,
        seed=seed,
        title=f"Intelligent Agents · Level {level_number} · {scenario.replace('_', ' ').title()}",
        scenario=scenario,
        rectangle_width_range=rectangle_width_range,
        rectangle_max_height=rectangle_max_height,
        rectangle_min_width=rectangle_min_width,
        guarantees=(
            "agents start on different vertical tiers",
            "upper agent can exit regardless of avatar shape",
            "team reunites before cooperative targets",
            "two lower-region cooperation challenges",
            "large rectangle transformation envelope",
        ),
    )


def _safe_spawn_x(
    rng: random.Random,
    floor_y: float,
    x_range: tuple[float, float],
    platforms: list[Platform],
    diamonds: list[Diamond],
    *,
    max_body_width: float,
    required_vertical_clearance: float = 12.0,
    avoid: tuple[float, ...] = (),
) -> float:
    """Choose a floor position with room for either random avatar assignment."""
    start, end = sorted((float(x_range[0]), float(x_range[1])))
    half_width = max_body_width / 2.0 + 0.35
    start = max(2.2 + half_width, start)
    end = min(117.8 - half_width, end)
    if end <= start:
        raise ValueError("Spawn range is too narrow for the rectangle envelope.")
    candidates = [start + (end - start) * index / 24.0 for index in range(25)]
    rng.shuffle(candidates)
    for x in candidates:
        # Only one of the two random avatars can be the wide rectangle; the
        # other is the 3.3-unit circle.  Requiring two maximum-width bodies
        # made valid separated-room spawns impossible for large rectangles.
        minimum_pair_distance = max_body_width / 2.0 + 1.65 + 1.0
        if any(abs(x - previous) < minimum_pair_distance for previous in avoid):
            continue
        body_left, body_right = x - half_width, x + half_width
        blocked = any(
            platform.right > body_left
            and platform.left < body_right
            and platform.top < floor_y - 0.05
            and platform.bottom > floor_y - required_vertical_clearance
            for platform in platforms
            if platform.kind not in {"upper_floor", "lower_floor", "boundary", "ceiling_boundary"}
        )
        if blocked:
            continue
        overlaps_target = any(
            abs(diamond.x - x) <= half_width + diamond.radius + 0.5
            and abs(diamond.y - (floor_y - 1.3)) <= 3.0
            for diamond in diamonds
        )
        if overlaps_target:
            continue
        return x
    # A low-ceiling/chamber scenario may intentionally spawn the rectangle
    # where it must first move horizontally before growing.  Preserve a
    # deterministic in-range fallback rather than rejecting an otherwise
    # solvable level.
    return (start + end) / 2.0


def validate_level(level: LevelDescriptor) -> None:
    """Reject generator regressions before a level reaches the simulation."""
    if level.width <= 0 or level.height <= 0:
        raise ValueError("Level dimensions must be positive.")
    if level.scenario not in SCENARIOS:
        raise ValueError(f"Unknown generated scenario: {level.scenario!r}")
    if len(level.spawn_points) != 2:
        raise ValueError("A cooperative level needs exactly two spawn points.")
    if level.rectangle_max_height < 24.0:
        raise ValueError("The rectangle transformation envelope is not sufficiently large.")
    if not (0.65 <= level.rectangle_min_width <= 1.5):
        raise ValueError("The rectangle minimum width is outside the aspect-ratio limit.")
    low_width, high_width = level.rectangle_width_range
    if not (6.0 <= low_width <= high_width <= 22.0):
        raise ValueError("Invalid rectangle base-width range.")

    roles = [diamond.role for diamond in level.diamonds]
    if not any(role.startswith("solo") for role in roles):
        raise ValueError("The level has no solo collectible.")
    if roles.count("cooperative_stack") < 1:
        raise ValueError("The level has no cooperative collectible.")
    priority = [diamond for diamond in level.diamonds if diamond.role == "priority_before_drop"]
    if len(priority) != 1 or priority[0].x >= level.drop_x:
        raise ValueError("The priority collectible must be before the irreversible drop.")
    if level.lower_floor_y - level.upper_floor_y < 25.0:
        raise ValueError("The descent is not sufficiently irreversible.")

    for spawn_x, spawn_bottom in level.spawn_points:
        if not 2.0 < spawn_x < level.width - 2.0:
            raise ValueError(f"Spawn x-position {spawn_x:.2f} is outside the world.")
        if not 2.0 < spawn_bottom < level.height - 1.0:
            raise ValueError(f"Spawn bottom {spawn_bottom:.2f} is outside the world.")
        supported = any(
            platform.left + 0.1 <= spawn_x <= platform.right - 0.1
            and abs(platform.top - spawn_bottom) <= 0.2
            for platform in level.platforms
            if platform.kind in {"upper_floor", "lower_floor"}
        )
        if not supported:
            raise ValueError("Every spawn must start on a certified floor surface.")

    guaranteed_rectangle_height = min(
        level.rectangle_max_height,
        level.rectangle_width_range[0] * 2.6 / level.rectangle_min_width,
    )
    for platform in level.platforms:
        if "cooperation_ledge" not in platform.kind:
            continue
        floor_y = (
            level.lower_floor_y
            if platform.kind.startswith("lower_")
            else level.upper_floor_y
        )
        gap = floor_y - platform.top
        if not 11.0 < gap < guaranteed_rectangle_height + 9.0:
            raise ValueError(
                f"Invalid constant-area cooperation gap: {gap:.2f}"
            )

    for platform in level.platforms:
        if platform.kind not in {"low_ceiling", "room_wall_low_door"}:
            continue
        if platform.kind == "low_ceiling":
            floor_y = (
                level.upper_floor_y
                if platform.top < (level.upper_floor_y + level.lower_floor_y) / 2.0
                else level.lower_floor_y
            )
            clearance = floor_y - platform.bottom
            if clearance < 3.1:
                raise ValueError(
                    f"Low route leaves only {clearance:.2f} units of clearance."
                )

    if level.scenario in {"opposed_upper", "separate_chambers", "central_split"}:
        separation = abs(level.spawn_points[0][0] - level.spawn_points[1][0])
        minimum = 24.0 if level.scenario == "central_split" else 16.0
        if separation < minimum:
            raise ValueError("Separated-room scenario did not separate the agents.")
        wall_count = sum(
            platform.kind == "room_wall_low_door" for platform in level.platforms
        )
        required_walls = 2 if level.scenario in {"separate_chambers", "central_split"} else 1
        if wall_count < required_walls:
            raise ValueError("Separated-room scenario is missing its chamber wall.")
    if level.scenario == "giant_lift":
        if level.rectangle_max_height < 34.0 or level.upper_floor_y < 30.0:
            raise ValueError("Giant-lift scenario lacks its enlarged transformation volume.")
    if level.scenario.startswith("vertical_reunion"):
        vertical_separation = abs(level.spawn_points[0][1] - level.spawn_points[1][1])
        if vertical_separation < 20.0:
            raise ValueError("Vertical-reunion spawns are not on different tiers.")

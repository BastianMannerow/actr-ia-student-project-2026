"""Continuous entities used by the cooperative platform simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from typing import Any


class AvatarShape(str, Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


class SpatialEntity:
    """Base class for visible world entities."""

    display_name = "Entity"
    symbol = "?"
    blocks_movement = False
    is_target = False


@dataclass(slots=True)
class PhysicsBody:
    """Mutable body state kept separate from cognitive agent state."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    width: float = 3.2
    height: float = 3.2
    radius: float = 1.6
    shape: AvatarShape = AvatarShape.CIRCLE
    grounded: bool = False
    support: Any | None = None
    transform: float = 0.0
    base_width: float = 7.0
    base_height: float = 2.6
    rectangle_area: float = 18.2
    max_height: float = 20.0
    min_width: float = 0.9
    input_x: float = 0.0
    input_until: float = 0.0
    previous_x: float = 0.0
    previous_y: float = 0.0

    @property
    def effective_max_height(self) -> float:
        if self.shape is not AvatarShape.RECTANGLE:
            return float(self.height)
        area = max(0.1, float(self.rectangle_area))
        min_width = max(0.1, float(self.min_width))
        return max(
            float(self.base_height),
            min(float(self.max_height), area / min_width),
        )

    @property
    def left(self) -> float:
        return self.x - self.width / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width / 2.0

    @property
    def top(self) -> float:
        return self.y - self.height / 2.0

    @property
    def bottom(self) -> float:
        return self.y + self.height / 2.0

    def set_bottom(self, bottom: float) -> None:
        self.y = float(bottom) - self.height / 2.0

    def set_top(self, top: float) -> None:
        self.y = float(top) + self.height / 2.0

    def speed(self) -> float:
        return hypot(self.vx, self.vy)


class SpatialAgent(SpatialEntity):
    """Base class for ACT-R and human-controlled platform avatars."""

    is_human_controlled = False
    symbol = "A"

    def __init__(self, name: str) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("A spatial agent needs a non-empty name.")
        self.name = normalized
        self.name_number = normalized
        self.display_name = normalized
        self.avatar_shape = AvatarShape.CIRCLE
        self.body = PhysicsBody()
        self.collected_diamonds = 0

    def configure_avatar(
        self,
        shape: AvatarShape,
        *,
        x: float,
        bottom: float,
        rectangle_width: float = 7.0,
        rectangle_max_height: float = 20.0,
        rectangle_min_width: float = 0.9,
    ) -> None:
        """Assign a shape and reset its physical state at a spawn point."""
        self.avatar_shape = AvatarShape(shape)
        body = self.body
        body.x = float(x)
        body.vx = 0.0
        body.vy = 0.0
        body.grounded = False
        body.support = None
        body.input_x = 0.0
        body.input_until = 0.0
        body.previous_x = body.x
        body.previous_y = body.y
        if self.avatar_shape is AvatarShape.CIRCLE:
            body.shape = AvatarShape.CIRCLE
            body.radius = 1.65
            body.width = body.radius * 2.0
            body.height = body.radius * 2.0
            body.transform = 0.0
        else:
            body.shape = AvatarShape.RECTANGLE
            body.base_width = float(rectangle_width)
            body.base_height = 2.6
            body.rectangle_area = body.base_width * body.base_height
            body.max_height = max(body.base_height, float(rectangle_max_height))
            body.min_width = max(0.65, float(rectangle_min_width))
            body.transform = 0.0
            body.width = body.base_width
            body.height = body.base_height
            body.radius = 0.0
        body.set_bottom(bottom)
        body.previous_y = body.y

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"shape={self.avatar_shape.value!r})"
        )


@dataclass(slots=True)
class Platform(SpatialEntity):
    """Static axis-aligned solid surface in world coordinates."""

    x: float
    y: float
    width: float
    height: float
    kind: str = "platform"
    display_name: str = "Solid platform"
    symbol: str = "P"
    blocks_movement: bool = True

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(slots=True)
class Diamond(SpatialEntity):
    """Collectible with generation metadata describing its intended role."""

    x: float
    y: float
    role: str = "solo"
    required_order: int = 0
    collected: bool = False
    diamond_id: str = ""
    radius: float = 1.05
    display_name: str = "Diamond"
    symbol: str = "D"
    blocks_movement: bool = False
    is_target: bool = True

    @property
    def left(self) -> float:
        return self.x - self.radius

    @property
    def right(self) -> float:
        return self.x + self.radius

    @property
    def top(self) -> float:
        return self.y - self.radius

    @property
    def bottom(self) -> float:
        return self.y + self.radius


@dataclass(slots=True)
class LevelDescriptor:
    """Generated level state consumed by the runtime and the renderer."""

    width: float
    height: float
    platforms: list[Platform]
    diamonds: list[Diamond]
    spawn_points: tuple[tuple[float, float], tuple[float, float]]
    drop_x: float
    upper_floor_y: float
    lower_floor_y: float
    seed: int
    title: str
    scenario: str = "classic"
    rectangle_width_range: tuple[float, float] = (8.0, 14.0)
    rectangle_max_height: float = 20.0
    rectangle_min_width: float = 0.9
    guarantees: tuple[str, ...] = field(default_factory=tuple)


# Compatibility aliases retained for generic inspection code and older agents.
Wall = Platform
Goal = Diamond
Target = Diamond
Checkpoint = Diamond
DefinitelyAWall = Platform
FakeWall = Platform
BurningTree = Platform
FireTarget = Diamond

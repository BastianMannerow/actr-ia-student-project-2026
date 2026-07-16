"""Cached rendering of the continuous cooperative platform environment."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from gui.environment_symbols import EnvironmentSymbol, draw_environment_symbol
from simulation.world.entities import AvatarShape


class PlatformerCanvas(QWidget):
    """Render static platforms once and draw dynamic bodies at native precision."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.environment: Any | None = None
        self._world_rect = QRectF()
        self._scale = 1.0
        self._static_cache: QPixmap | None = None
        self._static_cache_key: tuple[Any, ...] | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(720, 500)

    def set_environment(self, environment: Any) -> None:
        self.environment = environment
        self.invalidate_static_cache()
        consume = getattr(environment, "consume_render_changes", None)
        if callable(consume):
            consume()
        self.update()

    def refresh_from_environment(self) -> None:
        consume = getattr(self.environment, "consume_render_changes", None)
        if callable(consume):
            dirty = consume()
            if dirty is None:
                self.invalidate_static_cache()
            elif not dirty:
                return
        self.update()

    def invalidate_static_cache(self) -> None:
        self._static_cache = None
        self._static_cache_key = None

    def resizeEvent(self, event) -> None:  # noqa: N802
        self.invalidate_static_cache()
        super().resizeEvent(event)

    def _geometry(self, world_width: float, world_height: float) -> tuple[float, float, float]:
        margin = 18.0
        available_width = max(1.0, self.width() - 2.0 * margin)
        available_height = max(1.0, self.height() - 2.0 * margin)
        scale = min(available_width / world_width, available_height / world_height)
        draw_width = world_width * scale
        draw_height = world_height * scale
        origin_x = (self.width() - draw_width) / 2.0
        origin_y = (self.height() - draw_height) / 2.0
        self._scale = scale
        self._world_rect = QRectF(origin_x, origin_y, draw_width, draw_height)
        return origin_x, origin_y, scale

    def _world_to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(
            self._world_rect.left() + x * self._scale,
            self._world_rect.top() + y * self._scale,
        )

    def _world_box_to_screen(self, left: float, top: float, width: float, height: float) -> QRectF:
        point = self._world_to_screen(left, top)
        return QRectF(point.x(), point.y(), width * self._scale, height * self._scale)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        environment = self.environment
        if environment is None:
            painter.fillRect(self.rect(), QColor("#0b0f16"))
            painter.setPen(QColor("#8d98aa"))
            painter.setFont(QFont(self.font().family(), 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Environment is initializing…")
            return

        world_width = float(getattr(environment, "width", 120.0))
        world_height = float(getattr(environment, "height", 68.0))
        self._geometry(world_width, world_height)
        cache_key = (
            self.width(),
            self.height(),
            round(world_width, 4),
            round(world_height, 4),
            int(getattr(environment, "static_revision", 0)),
        )
        if self._static_cache is None or self._static_cache_key != cache_key:
            self._static_cache = self._build_static_cache(environment)
            self._static_cache_key = cache_key
        painter.drawPixmap(0, 0, self._static_cache)

        for diamond in getattr(environment, "diamonds", ()):  # dynamic because collectibles disappear
            if getattr(diamond, "collected", False):
                continue
            radius = float(getattr(diamond, "radius", 1.0))
            rect = self._world_box_to_screen(
                diamond.x - radius,
                diamond.y - radius,
                radius * 2.0,
                radius * 2.0,
            )
            symbol = (
                EnvironmentSymbol.PRIORITY_DIAMOND
                if getattr(diamond, "role", "") == "priority_before_drop"
                else EnvironmentSymbol.DIAMOND
            )
            draw_environment_symbol(painter, rect, symbol)

        for agent_index, agent in enumerate(getattr(environment, "agents", ())):
            body = agent.body
            rect = self._world_box_to_screen(body.left, body.top, body.width, body.height)
            symbol = (
                EnvironmentSymbol.CIRCLE_AGENT
                if agent.avatar_shape is AvatarShape.CIRCLE
                else EnvironmentSymbol.RECTANGLE_AGENT
            )
            draw_environment_symbol(
                painter,
                rect,
                symbol,
                human_controlled=bool(getattr(agent, "is_human_controlled", False)),
            )
            self._draw_agent_label(painter, rect, agent, agent_index)

        self._draw_world_hud(painter, environment)

    def _build_static_cache(self, environment: Any) -> QPixmap:
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#0b0f16"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gradient = QLinearGradient(self._world_rect.topLeft(), self._world_rect.bottomLeft())
        gradient.setColorAt(0.0, QColor("#111827"))
        gradient.setColorAt(1.0, QColor("#071018"))
        painter.fillRect(self._world_rect, gradient)

        # Fine coordinate guide: visual aid only, not a movement matrix.
        painter.setPen(QPen(QColor(51, 65, 85, 90), 1.0))
        grid_step = 5.0
        x = grid_step
        while x < float(getattr(environment, "width", 120.0)):
            start = self._world_to_screen(x, 0.0)
            end = self._world_to_screen(x, float(getattr(environment, "height", 68.0)))
            painter.drawLine(start, end)
            x += grid_step
        y = grid_step
        while y < float(getattr(environment, "height", 68.0)):
            start = self._world_to_screen(0.0, y)
            end = self._world_to_screen(float(getattr(environment, "width", 120.0)), y)
            painter.drawLine(start, end)
            y += grid_step

        for platform in getattr(environment, "platforms", ()):
            rect = self._world_box_to_screen(platform.left, platform.top, platform.width, platform.height)
            draw_environment_symbol(painter, rect, EnvironmentSymbol.PLATFORM)

        drop_x = float(getattr(getattr(environment, "level", None), "drop_x", -1.0))
        if drop_x >= 0.0:
            x_screen = self._world_to_screen(drop_x, 0.0).x()
            painter.setPen(QPen(QColor("#f472b6"), 1.3, Qt.PenStyle.DashLine))
            painter.drawLine(
                QPointF(x_screen, self._world_rect.top()),
                QPointF(x_screen, self._world_rect.bottom()),
            )

        painter.setPen(QPen(QColor("#64748b"), 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._world_rect)
        painter.end()
        return pixmap

    def _draw_agent_label(
        self, painter: QPainter, rect: QRectF, agent: Any, agent_index: int
    ) -> None:
        name = str(getattr(agent, "name", "Agent"))
        shape = getattr(getattr(agent, "avatar_shape", None), "value", "")
        shape_abbreviation = "C" if shape == "circle" else "R"
        text = f"{name} [{shape_abbreviation}]"
        painter.save()
        font = QFont("Sans Serif", 8)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = min(180.0, float(metrics.horizontalAdvance(text) + 14))
        height = 18.0
        center_x = rect.center().x()
        proposed_left = center_x - width / 2.0
        left = min(
            max(self._world_rect.left() + 4.0, proposed_left),
            self._world_rect.right() - width - 4.0,
        )
        top = rect.top() - 22.0 - agent_index * 18.0
        if top < self._world_rect.top() + 26.0:
            top = rect.bottom() + 4.0 + agent_index * 18.0
        label_rect = QRectF(left, top, width, height)
        painter.setPen(QPen(QColor("#64748b"), 0.8))
        painter.setBrush(QColor(7, 16, 24, 220))
        painter.drawRoundedRect(label_rect, 4.0, 4.0)
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _draw_world_hud(self, painter: QPainter, environment: Any) -> None:
        text = (
            f"Level {getattr(environment, 'level_number', 1)}   "
            f"Diamonds {getattr(environment, 'collected_diamonds', 0)}/"
            f"{getattr(environment, 'total_diamonds', 0)}   "
            f"Seed {getattr(getattr(environment, 'level', None), 'seed', '-') }"
        )
        rect = self._world_rect.adjusted(10.0, 8.0, -10.0, -8.0)
        painter.save()
        painter.setFont(QFont("Sans Serif", 9, QFont.Weight.DemiBold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, text)
        painter.restore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        environment = self.environment
        if environment is None or not self._world_rect.contains(event.position()):
            QToolTip.hideText()
            return
        world_x = (event.position().x() - self._world_rect.left()) / self._scale
        world_y = (event.position().y() - self._world_rect.top()) / self._scale
        details: list[str] = []
        for agent_index, agent in enumerate(getattr(environment, "agents", ())):
            body = agent.body
            if body.left <= world_x <= body.right and body.top <= world_y <= body.bottom:
                details.append(
                    f"{agent.name}: {agent.avatar_shape.value}, "
                    f"v=({body.vx:.1f}, {body.vy:.1f}), grounded={body.grounded}"
                )
        for diamond in getattr(environment, "diamonds", ()):
            if not diamond.collected and abs(diamond.x - world_x) <= 1.4 and abs(diamond.y - world_y) <= 1.4:
                details.append(f"Diamond: {diamond.role}")
        for platform in getattr(environment, "platforms", ()):
            if platform.left <= world_x <= platform.right and platform.top <= world_y <= platform.bottom:
                details.append(f"Solid: {platform.kind}")
                break
        text = f"World x={world_x:.2f}, y={world_y:.2f}"
        if details:
            text += "\n" + "\n".join(details)
        QToolTip.showText(event.globalPosition().toPoint(), text, self)


# Backward-compatible import name used by EnvironmentView.
GridCanvas = PlatformerCanvas

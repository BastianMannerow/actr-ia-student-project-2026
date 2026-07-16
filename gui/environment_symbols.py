"""Shared visual language for the cooperative platform environment."""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget


class EnvironmentSymbol(str, Enum):
    PLATFORM = "platform"
    DIAMOND = "diamond"
    PRIORITY_DIAMOND = "priority_diamond"
    CIRCLE_AGENT = "circle_agent"
    RECTANGLE_AGENT = "rectangle_agent"


ACTR_AGENT_COLOR = QColor("#38bdf8")
HUMAN_AGENT_COLOR = QColor("#f59e0b")
PLATFORM_COLOR = QColor("#334155")
DIAMOND_COLOR = QColor("#22d3ee")
PRIORITY_COLOR = QColor("#f472b6")


def draw_environment_symbol(
    painter: QPainter,
    rect: QRectF,
    symbol: EnvironmentSymbol,
    *,
    label: str | None = None,
    human_controlled: bool = False,
) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    if symbol == EnvironmentSymbol.PLATFORM:
        painter.setPen(QPen(QColor("#94a3b8"), max(1.0, rect.height() * 0.14)))
        painter.setBrush(PLATFORM_COLOR)
        painter.drawRoundedRect(rect, min(3.0, rect.height() * 0.25), min(3.0, rect.height() * 0.25))
    elif symbol in {EnvironmentSymbol.DIAMOND, EnvironmentSymbol.PRIORITY_DIAMOND}:
        color = PRIORITY_COLOR if symbol == EnvironmentSymbol.PRIORITY_DIAMOND else DIAMOND_COLOR
        points = QPolygonF(
            [
                QPointF(rect.center().x(), rect.top()),
                QPointF(rect.right(), rect.center().y()),
                QPointF(rect.center().x(), rect.bottom()),
                QPointF(rect.left(), rect.center().y()),
            ]
        )
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#e0f2fe"), max(1.0, rect.width() * 0.06)))
        painter.drawPolygon(points)
        painter.setPen(QPen(QColor("#ffffff"), max(0.7, rect.width() * 0.035)))
        painter.drawLine(points[0], points[2])
        painter.drawLine(points[1], points[3])
    else:
        color = HUMAN_AGENT_COLOR if human_controlled else ACTR_AGENT_COLOR
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#f8fafc"), max(1.0, min(rect.width(), rect.height()) * 0.06)))
        if symbol == EnvironmentSymbol.CIRCLE_AGENT:
            painter.drawEllipse(rect)
        else:
            painter.drawRoundedRect(rect, min(4.0, rect.width() * 0.12), min(4.0, rect.height() * 0.12))

    if label and rect.width() >= 20 and rect.height() >= 14:
        font = QFont("Sans Serif")
        font.setBold(True)
        font.setPointSizeF(max(5.5, min(10.0, min(rect.width(), rect.height()) * 0.24)))
        painter.setFont(font)
        painter.setPen(QColor("#071018"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
    painter.restore()


class EnvironmentLegendMarker(QWidget):
    """Legend marker rendered by the same painter as the environment."""

    def __init__(
        self,
        symbol: EnvironmentSymbol,
        parent=None,
        *,
        human_controlled: bool = False,
    ) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self.human_controlled = human_controlled
        self.setFixedSize(30, 30)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        draw_environment_symbol(
            painter,
            QRectF(3.0, 3.0, self.width() - 6.0, self.height() - 6.0),
            self.symbol,
            human_controlled=self.human_controlled,
        )

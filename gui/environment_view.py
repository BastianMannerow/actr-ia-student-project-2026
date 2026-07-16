"""Panel-level presentation of the cooperative platform environment."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.environment_canvas import PlatformerCanvas
from gui.environment_symbols import EnvironmentLegendMarker, EnvironmentSymbol


class EnvironmentView(QFrame):
    """Panel containing continuous-world metadata, legend, and renderer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.environment: Any | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        title = QLabel("Cooperative Platform Environment")
        title.setObjectName("sectionTitle")
        self.info_label = QLabel("Not initialized")
        self.info_label.setObjectName("muted")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.info_label)
        layout.addLayout(heading)

        self.canvas = PlatformerCanvas(self)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self._build_legend())

    def _build_legend(self) -> QWidget:
        legend = QFrame(self)
        legend.setObjectName("toolbar")
        row = QHBoxLayout(legend)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(14)
        row.addWidget(self._legend_item(EnvironmentSymbol.PLATFORM, "Solid surface"))
        row.addWidget(self._legend_item(EnvironmentSymbol.DIAMOND, "Diamond"))
        row.addWidget(self._legend_item(EnvironmentSymbol.PRIORITY_DIAMOND, "Collect before drop"))
        row.addWidget(self._legend_item(EnvironmentSymbol.CIRCLE_AGENT, "Circle avatar"))
        row.addWidget(self._legend_item(EnvironmentSymbol.RECTANGLE_AGENT, "Rectangle avatar"))
        row.addStretch(1)
        controls = QLabel(
            "Human: A/D move · Circle W jump, S fast-fall · Rectangle W tall/narrow, S flat/wide · R restart"
        )
        controls.setObjectName("muted")
        row.addWidget(controls)
        return legend

    @staticmethod
    def _legend_item(symbol: EnvironmentSymbol, text: str) -> QWidget:
        item = QWidget()
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(5)
        marker = EnvironmentLegendMarker(symbol, item)
        label = QLabel(text)
        label.setObjectName("muted")
        item_layout.addWidget(marker)
        item_layout.addWidget(label)
        return item

    def set_environment(self, environment: Any) -> None:
        self.environment = environment
        self.canvas.set_environment(environment)
        self.refresh()

    def refresh(self) -> None:
        environment = self.environment
        if environment is None:
            self.info_label.setText("Not initialized")
            return
        role_text = " · ".join(
            f"{getattr(agent, 'name', 'Agent')}: {getattr(getattr(agent, 'avatar_shape', None), 'value', '?')}"
            for agent in getattr(environment, "agents", ())
        )
        self.info_label.setText(
            f"LEVEL {getattr(environment, 'level_number', 1)} · "
            f"{getattr(environment, 'collected_diamonds', 0)}/"
            f"{getattr(environment, 'total_diamonds', 0)} DIAMONDS · "
            f"{role_text}"
        )
        self.canvas.refresh_from_environment()

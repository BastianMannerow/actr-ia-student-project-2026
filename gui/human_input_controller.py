"""Keyboard mapping for the optional human-controlled platform avatar."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QSlider,
    QTabBar,
    QTextEdit,
)


class HumanInputController(QObject):
    """Track held WASD keys and expose R as a level restart."""

    _KEYS = {
        Qt.Key.Key_W: "W",
        Qt.Key.Key_Up: "W",
        Qt.Key.Key_A: "A",
        Qt.Key.Key_Left: "A",
        Qt.Key.Key_S: "S",
        Qt.Key.Key_Down: "S",
        Qt.Key.Key_D: "D",
        Qt.Key.Key_Right: "D",
    }

    _INTERACTIVE_CONTROLS = (
        QLineEdit,
        QPlainTextEdit,
        QTextEdit,
        QComboBox,
        QAbstractSpinBox,
        QAbstractButton,
        QAbstractItemView,
        QSlider,
        QTabBar,
    )

    def __init__(
        self,
        simulation: Any,
        *,
        enabled_predicate: Callable[[], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.simulation = simulation
        self.enabled_predicate = enabled_predicate

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        del watched
        if event.type() not in {QEvent.Type.KeyPress, QEvent.Type.KeyRelease}:
            return False
        if not getattr(self.simulation, "initialized", False):
            return False

        pressed = event.type() == QEvent.Type.KeyPress
        key = self._KEYS.get(event.key())

        # Releases must always clear held state, even after focus or tab changes.
        if not pressed and key is not None and getattr(self.simulation, "human_agent", None) is not None:
            self.simulation.set_human_control(key, False)
            return True

        if self.enabled_predicate is not None and not self.enabled_predicate():
            return False
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, self._INTERACTIVE_CONTROLS):
            return False

        if event.key() == Qt.Key.Key_R and pressed and not event.isAutoRepeat():
            return bool(self.simulation.restart_level())
        if key is None or getattr(self.simulation, "human_agent", None) is None:
            return False
        self.simulation.set_human_control(key, True)
        return True

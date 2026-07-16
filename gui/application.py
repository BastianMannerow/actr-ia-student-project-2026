"""QApplication construction and global application configuration."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from gui.resources import (
    APPLICATION_NAME,
    build_application_icon,
    configure_windows_process_identity,
)
from gui.styles import APP_STYLESHEET


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the active QApplication or create and configure one."""
    configure_windows_process_identity()
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        existing.setApplicationName(APPLICATION_NAME)
        existing.setApplicationDisplayName(APPLICATION_NAME)
        existing.setWindowIcon(build_application_icon())
        return existing

    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName(APPLICATION_NAME)
    app.setApplicationDisplayName(APPLICATION_NAME)
    app.setOrganizationName("Intelligent Agents")
    app.setOrganizationDomain("intelligent-agents.local")
    app.setWindowIcon(build_application_icon())
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)
    return app

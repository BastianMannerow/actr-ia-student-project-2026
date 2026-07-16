"""Application identity and resource paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


APPLICATION_NAME = "Intelligent Agents"
WINDOWS_APP_USER_MODEL_ID = "IntelligentAgents.PlatformSimulation.2026"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def application_icon_candidates() -> tuple[Path, ...]:
    """Return platform-preferred icon files with a cross-platform fallback."""
    assets = project_root() / "assets"
    preferred = assets / ("actr_icon.ico" if os.name == "nt" else "actr_icon.png")
    fallback = assets / ("actr_icon.png" if os.name == "nt" else "actr_icon.ico")
    return preferred, fallback


def application_icon_path() -> Path:
    for candidate in application_icon_candidates():
        if candidate.exists():
            return candidate
    return application_icon_candidates()[0]


def application_icon_pixmap_path() -> Path:
    """Prefer the PNG asset for an icon rendered inside the GUI header."""
    png = project_root() / "assets" / "actr_icon.png"
    return png if png.exists() else application_icon_path()


def build_application_icon():
    """Build the application icon, using the multi-resolution ICO on Windows."""
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QIcon, QPixmap

    ico = project_root() / "assets" / "actr_icon.ico"
    if sys.platform == "win32" and ico.exists():
        # Passing the ICO directly lets Qt/Windows select the appropriate native
        # image for the title bar, Alt-Tab view, and taskbar button.
        native_icon = QIcon(str(ico))
        if not native_icon.isNull():
            return native_icon

    icon = QIcon()
    png = application_icon_pixmap_path()
    if png.exists():
        source = QPixmap(str(png))
        if not source.isNull():
            for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
                icon.addPixmap(source.scaled(QSize(size, size)))
    if ico.exists():
        ico_icon = QIcon(str(ico))
        for size in ico_icon.availableSizes():
            icon.addPixmap(ico_icon.pixmap(size))
    return icon


def configure_windows_process_identity() -> None:
    """Set the AppUserModelID before QApplication creates a native window."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            WINDOWS_APP_USER_MODEL_ID
        )
    except Exception:
        pass


def apply_native_windows_icon(window: Any) -> None:
    """Set the ICO as the native HWND and window-class icon on Windows."""
    if sys.platform != "win32":
        return
    icon_path = project_root() / "assets" / "actr_icon.ico"
    if not icon_path.exists():
        return
    try:
        import ctypes
        from ctypes import wintypes

        # Ensure the QMainWindow already owns a native HWND.
        hwnd = wintypes.HWND(int(window.winId()))
        user32 = ctypes.windll.user32
        user32.LoadImageW.argtypes = (
            wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
            ctypes.c_int, ctypes.c_int, wintypes.UINT,
        )
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.SendMessageW.argtypes = (
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        )
        user32.SendMessageW.restype = wintypes.LPARAM

        pointer_size = ctypes.sizeof(ctypes.c_void_p)
        set_class_long = (
            user32.SetClassLongPtrW if pointer_size == 8 else user32.SetClassLongW
        )
        set_class_long.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_void_p)
        set_class_long.restype = ctypes.c_void_p

        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        GCLP_HICON = -14
        GCLP_HICONSM = -34

        handles = []
        for size, icon_kind, class_index in (
            (32, ICON_SMALL, GCLP_HICONSM),
            (256, ICON_BIG, GCLP_HICON),
        ):
            handle = user32.LoadImageW(
                None, str(icon_path), IMAGE_ICON, size, size,
                LR_LOADFROMFILE | LR_DEFAULTSIZE,
            )
            if not handle:
                continue
            handle_value = int(handle)
            user32.SendMessageW(hwnd, WM_SETICON, icon_kind, handle_value)
            set_class_long(hwnd, class_index, ctypes.c_void_p(handle_value))
            handles.append(handle)
        # HICONs loaded from file must remain alive for the window lifetime.
        window._native_icon_handles = handles
    except Exception:
        pass


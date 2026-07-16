"""ACT-R simulation runtime, world model, inspection, export, and batch tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["pyactr_extension", "pyactrFunctionalityExtension"]


def __getattr__(name: str) -> Any:
    """Load pyactr integration lazily so world utilities stay independently usable."""
    if name in {"pyactr_extension", "pyactrFunctionalityExtension"}:
        module = import_module("simulation.integrations.pyactr_extension")
        globals()["pyactr_extension"] = module
        globals()["pyactrFunctionalityExtension"] = module
        return module
    raise AttributeError(name)

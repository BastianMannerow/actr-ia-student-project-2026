"""Runtime-only compatibility patches for upstream pyactr event scheduling.

This module deliberately stays outside ``pyactrFunctionalityExtension``.  The
extension remains the narrow public helper API supplied to adapters, while
runtime compatibility is owned by the simulation engine.
"""

from __future__ import annotations

import pyactr.simulation as actr_simulation
import simpy

_PATCHED = False


def install_runtime_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    def safe_localprocess(self, name, generator):
        del name
        while True:
            try:
                event = next(generator)
            except StopIteration:
                return
            if not isinstance(event, actr_simulation.Event):
                return event
            try:
                simulation = self._Simulation__simulation
                yield simulation.timeout(event.time - round(simulation.now, 4))
            except simpy.Interrupt:
                break
            else:
                self.__printevent__(event)
                self.__activate__(event)
            try:
                environment = self._Simulation__env
                procedural = self._Simulation__pr
                interaction = procedural.env_interaction
                if environment.trigger and interaction.intersection(environment.trigger):
                    activation = self._Simulation__environment_activate
                    if not activation.triggered:
                        activation.succeed(value=(environment.trigger, interaction))
                procedural.env_interaction = set()
            except AttributeError:
                pass

    actr_simulation.Simulation.__localprocess__ = safe_localprocess

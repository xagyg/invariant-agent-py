"""The capability registry."""

from __future__ import annotations

from typing import Iterable, Optional

from ..core.abstractions import ICapability, ICapabilityRegistry


class CapabilityRegistry(ICapabilityRegistry):
    """Default :class:`ICapabilityRegistry`: indexes capabilities by name and aliases (case-insensitive).

    Later registrations win if names collide.
    """

    def __init__(self, capabilities: Iterable[ICapability]) -> None:
        # lower-cased name -> (name as first registered, capability)
        self._capabilities: dict[str, tuple[str, ICapability]] = {}
        for capability in capabilities:
            self._register(capability.name, capability)  # primary name
            for alias in capability.aliases or ():  # aliases
                self._register(alias, capability)

    def _register(self, name: str, capability: ICapability) -> None:
        key = name.lower()
        display = self._capabilities[key][0] if key in self._capabilities else name
        self._capabilities[key] = (display, capability)

    def get(self, capability_name: str) -> Optional[ICapability]:
        entry = self._capabilities.get((capability_name or "").lower())
        return entry[1] if entry else None

    def get_capability_names(self) -> list[str]:
        return [display for display, _ in self._capabilities.values()]

    @property
    def capabilities(self) -> list[ICapability]:
        """Every distinct registered capability, in registration order."""
        seen: list[ICapability] = []
        for _, capability in self._capabilities.values():
            if not any(c is capability for c in seen):
                seen.append(capability)
        return seen

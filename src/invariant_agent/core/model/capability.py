"""Capability requests and results, including proposed (not yet applied) self-modifications."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .._util import utc_now
from .data import CapabilityData


@dataclass
class SelfModificationRequest:
    """A capability's proposal to change agent state (e.g. ``memory.set goal=...``).

    Proposals are evaluated by self-modification invariants and, if approved, applied by the reducer. Capabilities
    never apply them directly.
    """

    #: What is being modified (currently ``memory``; future targets may include policy or planner).
    target: str = ""
    #: The operation to perform (currently ``set``).
    operation: str = ""
    #: The key to modify.
    key: str = ""
    #: The new value.
    value: str = ""


@dataclass
class CapabilityRequest:
    """Input passed to ``ICapability.execute``."""

    #: Raw input text selected by the planner.
    input: str = ""


@dataclass
class CapabilityResult:
    """The result returned by a capability, optionally carrying a proposed self-modification.

    Capabilities cannot modify runtime state directly. Instead they attach a :class:`SelfModificationRequest` to
    :attr:`proposed_modification`, which the runtime evaluates against self-modification invariants before the
    reducer applies it.
    """

    #: ``True`` if the capability executed successfully.
    success: bool = False
    #: Name of the capability that produced the result.
    capability: str = ""
    #: Typed payload produced on success.
    data: Optional[CapabilityData] = None
    #: Failure reason when :attr:`success` is ``False``.
    error: str = ""
    #: UTC time the result was created.
    timestamp: datetime = field(default_factory=utc_now)
    #: A proposed change to agent state, subject to governance. ``None`` if none.
    proposed_modification: Optional[SelfModificationRequest] = None

    @staticmethod
    def ok(capability: str, data: CapabilityData,
           proposed: Optional[SelfModificationRequest] = None) -> "CapabilityResult":
        """Creates a successful result."""
        return CapabilityResult(success=True, capability=capability, data=data, proposed_modification=proposed)

    @staticmethod
    def fail(capability: str, error: str) -> "CapabilityResult":
        """Creates a failed result."""
        return CapabilityResult(success=False, capability=capability, error=error)

    def __str__(self) -> str:
        """Diagnostic string showing either the data or the error."""
        if not self.success:
            return f"Capability={self.capability}, Error={self.error}"
        return f"Capability={self.capability}, Data={_format(self.data)}"


def _format(value: Any) -> str:
    # Recursively formats strings, collections, primitives and simple objects (via their fields).
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, CapabilityData):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    if isinstance(value, (bool, int, float)):
        return str(value)
    if dataclasses.is_dataclass(value):
        parts = [f"{f.name}={_format(getattr(value, f.name))}" for f in dataclasses.fields(value)]
        return "{ " + ", ".join(parts) + " }"
    return str(value)

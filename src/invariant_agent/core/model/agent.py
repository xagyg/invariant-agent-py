"""Agent-level model: proposed actions, outcomes, runtime state and its bounded projection."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .._util import is_empty, new_id, utc_now


@dataclass
class AgentAction:
    """An action proposed by a planner: the capability to invoke and the input to pass to it (``Aₜ`` in the IBAAM
    transition model). A proposal only - it is executed only after plan invariants pass."""

    #: Name or alias of the capability to invoke.
    capability: Optional[str] = None
    #: Input passed to the capability.
    input: Optional[str] = None
    #: Planner failure reason, if planning failed (``None`` = success).
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        """``True`` if the planner reported an error."""
        return not is_empty(self.error)


class AgentEvent(ABC):
    """Base class for domain events recorded in :attr:`AgentState.events`."""

    def __init__(self) -> None:
        #: Unique event identifier.
        self.id: uuid.UUID = new_id()
        #: UTC time the event was created.
        self.timestamp: datetime = utc_now()

    @property
    @abstractmethod
    def type(self) -> str:
        """Event type discriminator."""

    @abstractmethod
    def to_observation(self) -> str:
        """Renders the event as a short observation suitable for planner context."""


@dataclass
class AgentOutcome:
    """The observed result of executing a proposed action (``Oₜ`` in the IBAAM transition model)."""

    #: ``True`` if the capability executed successfully.
    success: bool = False
    #: The capability that produced this outcome.
    capability: str = ""
    #: String form of the capability's data on success; empty on failure.
    result: str = ""
    #: Failure reason on unsuccessful execution.
    error: str = ""


@dataclass
class AgentState:
    """The agent's runtime state ``Sₜ``.

    A new instance is produced by the reducer for each committed transition, and only committed after reduction
    invariants pass.
    """

    #: Monotonic state version; incremented by one on every committed transition.
    version: int = 0
    #: Adaptive key/value memory (e.g. ``goal``, ``lastOutcome``). Changed via governed self-modification.
    memory: dict[str, Any] = field(default_factory=dict)
    #: Domain events recorded against this state. Polymorphic events are not persisted (as in the C# store).
    events: list[Any] = field(default_factory=list)
    #: Current operating mode. Changes contribute to behavioural drift.
    mode: str = ""
    #: Active behavioural policies. Changes contribute to behavioural drift.
    policies: list[str] = field(default_factory=list)
    #: Explicit goal. When empty, drift and stability logic fall back to the ``goal`` memory entry.
    goal: Optional[str] = None

    def add_event(self, event: AgentEvent) -> None:
        """Appends a domain event."""
        self.events.append(event)


@dataclass
class StateProjection:
    """The bounded projection ``π(Sₜ)`` of runtime state exposed to planners.

    Planners never see the full :class:`AgentState`; this controlled interface limits what adaptive cognition can
    observe about governed execution.
    """

    #: Current state version.
    version: int = 0
    #: Current explicit goal.
    goal: str = ""
    #: Summary of a whitelisted subset of memory keys.
    memory_summary: str = ""
    #: Result of the last successful non-introspective capability.
    last_outcome: str = ""
    #: Currently active policies.
    active_policies: list[str] = field(default_factory=list)

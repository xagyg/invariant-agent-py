"""What a planner is allowed to see."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .agent import AgentAction, StateProjection


@dataclass
class PlannerRevision:
    """Conscience feedback handed back to the planner with a request to re-plan.

    This is what makes tension a cost on action selection rather than only a gate.
    """

    #: Revision attempt being requested (1 for the first re-plan).
    attempt: int = 0
    #: The proposal the conscience asked to revise.
    previous: Optional[AgentAction] = None
    #: Combined tension of the previous proposal.
    tension: float = 0.0
    #: The conscience's justification, generated from its tension components.
    justification: str = ""
    #: Protected values the previous proposal put at stake.
    values_at_stake: list[str] = field(default_factory=list)


@dataclass
class PlannerContext:
    """Everything a planner is allowed to see: user input plus the bounded state projection ``π(Sₜ)``."""

    #: Raw user input ``Iₜ``.
    input: str = ""
    #: Bounded projection of runtime state.
    state: StateProjection = field(default_factory=StateProjection)
    #: The ``goal`` memory value, if any.
    goal: str = ""
    #: The ``lastOutcome`` memory value, if any.
    last_outcome: str = ""
    #: Whitelisted memory entries exposed to the planner.
    relevant_memory: dict[str, Any] = field(default_factory=dict)
    #: Set when the conscience has asked for a revised plan: the appraisal of the previous proposal, returned to the
    #: planner as context (IBAAM 6, Section 6.3). ``None`` for a first proposal.
    revision: Optional[PlannerRevision] = None

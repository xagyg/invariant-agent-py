"""Transitions: the fundamental unit of governed execution, with their events, conflict records and context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .._util import new_id, utc_now
from .agent import AgentAction, AgentOutcome, AgentState
from .capability import SelfModificationRequest
from .conscience import ConscienceAppraisalRecord
from .control import GovernanceCandidate, InvariantEvaluationReport
from .drift import DriftBaselineSnapshot
from .enums import (
    InvariantScope,
    OperationalContext,
    TransitionEventStage,
    TransitionPhase,
    TransitionStatus,
)


@dataclass
class TransitionEvent:
    """A single entry in a transition's audit trail."""

    #: UTC time of the event.
    timestamp: datetime = field(default_factory=utc_now)
    #: Pipeline stage that raised the event.
    stage: TransitionEventStage = TransitionEventStage.Input
    #: Human-readable message (also printed by the REPL).
    message: str = ""
    #: Structured data (e.g. invariant name, scope, pass/fail) used by observability tools.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictRecord:
    """A compact, persisted summary of one resolved invariant conflict (IBAAM 3 revised, Section 7.4).

    The full decision record (every candidate profile and witness) is in the transition's Control events.
    """

    #: The checkpoint at which the conflict was resolved.
    scope: InvariantScope = InvariantScope.Transition
    #: Operational context the decision was made under.
    operational_context: OperationalContext = OperationalContext.Normal
    #: The selected candidate: ``proposed``, ``alternative-N`` or ``decline``.
    selected: str = GovernanceCandidate.PROPOSED_ID
    #: Capability of the selected candidate; empty when it is the decline option.
    selected_capability: str = ""
    #: The selected candidate's violation profile.
    profile: str = "none"
    #: Invariants the selected action overrides.
    overridden: list[str] = field(default_factory=list)
    #: ``True`` when the resolution prescribes different behaviour from simply carrying out the proposal (an
    #: alternative, a decline or overrides), i.e. a behavioural change is expected.
    changes_behaviour: bool = False
    #: ``True`` when the conflict was held for human review.
    escalated: bool = False
    #: Plain-words reason for the selection.
    reason: str = ""

    @staticmethod
    def from_report(report: InvariantEvaluationReport) -> Optional["ConflictRecord"]:
        """Builds a record from an evaluation report; ``None`` when nothing was resolved."""
        resolution = report.resolution
        if resolution is None or (
            not resolution.conflict and not report.overrides and report.selected_alternative is None
        ):
            return None
        selected = next((c for c in resolution.candidates if c.id == resolution.selected_id), None)
        return ConflictRecord(
            scope=report.scope,
            operational_context=resolution.operational_context,
            selected=resolution.selected_id,
            selected_capability=(selected.action.capability if selected and selected.action else None) or "",
            profile=selected.profile if selected else "none",
            overridden=[o.overridden_violation.invariant for o in report.overrides],
            changes_behaviour=resolution.selected_id != GovernanceCandidate.PROPOSED_ID or len(report.overrides) > 0,
            escalated=report.requires_escalation,
            reason=resolution.reason,
        )


@dataclass
class Transition:
    """The fundamental unit of governed execution: one input processed through the full lifecycle (planning,
    pre-control, execution, post-control, self-modification check, reduction, commit).

    Formally a transition maps ``(Sₜ, Iₜ)`` to ``Sₜ₊₁`` via a proposed action ``Aₜ`` and an observed outcome ``Oₜ``.
    Every transition - completed or rejected - is persisted with its full event trail for replay, audit and drift
    analysis. Fields are declared in the same order as the C# class so the JSONL records match.
    """

    #: Unique transition identifier.
    id: uuid.UUID = field(default_factory=new_id)
    #: UTC creation time.
    timestamp: datetime = field(default_factory=utc_now)
    #: Current lifecycle phase. Change only via ``TransitionPhases.move_to`` to enforce legal moves.
    phase: TransitionPhase = TransitionPhase.Created
    #: The input ``Iₜ`` that initiated this transition.
    input: str = ""
    #: The prior state ``Sₜ``.
    before: Optional[AgentState] = None
    #: The planner's proposed action ``Aₜ``; executed only if plan invariants pass.
    proposed_action: Optional[AgentAction] = None
    #: The observed outcome ``Oₜ``; ``None`` if rejected before execution.
    outcome: Optional[AgentOutcome] = None
    #: The next state ``Sₜ₊₁`` proposed by the reducer; committed only if reduction invariants pass. ``None`` if
    #: reduction was never reached.
    after: Optional[AgentState] = None
    #: Current or final status of the transition.
    status: TransitionStatus = TransitionStatus.Completed
    #: Governance summary explaining a rejection; empty otherwise.
    reason: str = ""
    #: Self-modification proposed by the executed capability, if any.
    self_modification: Optional[SelfModificationRequest] = None
    #: HIG-C operational context in force when this transition ran (IBAAM 3 revised, Section 5). Recorded so replay
    #: resolves conflicts under the same context the live run used; older history deserialises as Normal.
    operational_context: OperationalContext = OperationalContext.Normal
    #: The drift baseline in force when reduction invariants were evaluated. Recorded so replay scores behavioural
    #: drift against the same baseline the live run used. ``None`` means it was not recorded.
    drift_baseline: Optional[DriftBaselineSnapshot] = None
    #: Invariants whose unresolved violations rejected this transition. Empty when it completed, or when it was
    #: stopped by the conscience rather than by a control. Boundary labels for the conscience.
    blocking_invariants: list[str] = field(default_factory=list)
    #: Invariant conflicts resolved during this transition (IBAAM 3 revised, Section 7.4). Behavioural change on a
    #: transition that carries a conflict record is *attributed* drift; change without one is unattributed.
    conflicts: list[ConflictRecord] = field(default_factory=list)
    #: ``True`` when HIG-C held this transition for human review because every candidate, including declining,
    #: violated a fundamental invariant and no pre-authorised protocol covered the case (Section 7.2).
    governance_escalated: bool = False
    #: Conscience appraisals made during this transition, in order (plan appraisals per revision attempt, then the
    #: answer appraisal). Recorded so replay reproduces the live decision (IBAAM 6, Section 6.4.4).
    appraisals: list[ConscienceAppraisalRecord] = field(default_factory=list)
    #: Ordered, auditable event trail for this transition.
    events: list[TransitionEvent] = field(default_factory=list)

    def add_event(self, stage: TransitionEventStage, message: str,
                  metadata: Optional[dict[str, Any]] = None) -> None:
        """Appends an event to the transition's audit trail."""
        self.events.append(TransitionEvent(stage=stage, message=message, metadata=dict(metadata or {})))


class TransitionContext:
    """The context passed through the pipeline (invariants, executor, reducer) for one transition.

    The operational context is stored on the transition itself (:attr:`Transition.operational_context`) so it is
    persisted and replay re-resolves conflicts under the context the live run used. A context set before the
    transition is attached is carried onto it.
    """

    def __init__(self, transition: Optional[Transition] = None,
                 operational_context: Optional[OperationalContext] = None) -> None:
        self._transition: Optional[Transition] = None
        self._pending_context: Optional[OperationalContext] = None
        if operational_context is not None:
            self.operational_context = operational_context
        if transition is not None:
            self.transition = transition

    @property
    def transition(self) -> Optional[Transition]:
        """The transition being governed."""
        return self._transition

    @transition.setter
    def transition(self, value: Optional[Transition]) -> None:
        self._transition = value
        if value is not None and self._pending_context is not None:
            value.operational_context = self._pending_context
            self._pending_context = None

    @property
    def operational_context(self) -> OperationalContext:
        """HIG-C operational context in effect for this transition (default Normal)."""
        if self._transition is not None:
            return self._transition.operational_context
        if self._pending_context is not None:
            return self._pending_context
        return OperationalContext.Normal

    @operational_context.setter
    def operational_context(self, value: OperationalContext) -> None:
        if self._transition is not None:
            self._transition.operational_context = value
        else:
            self._pending_context = value

    @property
    def is_rejected(self) -> bool:
        """``True`` if the transition has been rejected."""
        return self._transition is not None and self._transition.status == TransitionStatus.Rejected

    @property
    def is_completed(self) -> bool:
        """``True`` if the transition has completed."""
        return self._transition is not None and self._transition.status == TransitionStatus.Completed

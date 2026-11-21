"""Functional conscience model (IBAAM 6): commitments, tension components, appraisals and their records."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .._util import utc_now
from .agent import AgentAction, AgentOutcome, AgentState
from .enums import (
    AppraisalPoint,
    CommitmentKind,
    CommitmentOrigin,
    CommitmentScope,
    ConscienceEffect,
    ConscienceStance,
    PrecedentOutcome,
    TensionKind,
)
from .stability import StabilityAssessment


@dataclass(frozen=True)
class Commitment:
    """A record of conduct the agent has declined or undertaken, against which later plans are appraised
    (IBAAM 6, Section 6.4.2). This is the operational form of the agent's self-narrative.

    Commitments live in an ``ICommitmentStore`` rather than in ``AgentState``: a rejected transition can never change
    agent state, yet a Veto or Escalate (which rejects the transition) must still leave a commitment behind. Like drift
    baselines, commitments are governance state, persisted beyond any context window. Commitments about individuals are
    sensitive data in their own right (Section 9).

    Use :func:`dataclasses.replace` for the C# ``with`` expression.
    """

    #: Commitment identifier.
    id: uuid.UUID
    #: The transition where the commitment arose.
    source_transition: uuid.UUID
    #: Protected-value tag (same tags as invariants).
    value: str
    #: Where the commitment applies.
    scope: CommitmentScope
    #: The conduct declined or undertaken, in words.
    statement: str
    #: Agent or reviewer.
    origin: CommitmentOrigin
    #: Creation time.
    created: datetime
    #: Optional expiry; ``None`` means it lasts until superseded or its scope ends.
    expires: Optional[datetime] = None
    #: Declined or undertaken.
    kind: CommitmentKind = CommitmentKind.Declined
    #: Capability of the conduct the commitment is about (used to match later plans).
    capability: str = ""
    #: Normalised input of the conduct the commitment is about (used to match later plans).
    subject: str = ""
    #: Session the commitment arose in (matters for ``CommitmentScope.Conversation``).
    conversation_id: str = ""
    #: User the commitment arose for (matters for ``CommitmentScope.User``).
    user_id: str = ""
    #: For a reviewer commitment, the escalated (or vetoed) transition it resolves.
    resolves: Optional[uuid.UUID] = None


@dataclass(frozen=True)
class TensionComponent:
    """One decomposed part of a conscience appraisal.

    The justification is generated from these parts, so the stated reason always matches the computed score
    (IBAAM 6, Section 6.4.3).
    """

    #: Which tension component this is.
    kind: TensionKind
    #: Component value in 0..1.
    value: float
    #: Weight used when combining components into the overall tension.
    weight: float
    #: Human-readable account of how the value was computed.
    detail: str
    #: Protected-value tags this component concerns (from invariants or commitments).
    values: list[str] = field(default_factory=list)
    #: ``False`` when the component does not apply at this appraisal point and was not combined.
    applicable: bool = True


@dataclass(frozen=True)
class ConscienceAppraisal:
    """The result of a conscience appraisal (IBAAM 6, Section 6.1): graded, decomposed tension plus a stance."""

    #: Combined tension in 0..1. A weighted score used to select a stance; it is not a probability (only the
    #: violation-risk part is calibrated).
    tension: float
    #: The decomposed tension components.
    parts: list[TensionComponent]
    #: Proceed, Revise, Escalate or Veto.
    stance: ConscienceStance
    #: Generated from :attr:`parts` and the cited cases.
    justification: str
    #: Ids of the past transitions the appraisal reasons from.
    cited_precedents: list[uuid.UUID]
    #: Ids of the live commitments the plan was found to conflict with (or be cleared by).
    cited_commitments: list[uuid.UUID] = field(default_factory=list)
    #: Protected-value tags at stake, most significant first.
    values_at_stake: list[str] = field(default_factory=list)
    #: Name of the conscience implementation that produced the appraisal.
    conscience: str = ""


@dataclass
class ConscienceAppraisalRecord:
    """An appraisal as recorded on its transition, so deterministic replay reproduces the live decision even where a
    component is non-deterministic (IBAAM 6, Section 6.4.4)."""

    #: Plan or answer appraisal.
    point: AppraisalPoint = AppraisalPoint.Plan
    #: Revision attempt the appraisal was made on (0 = the planner's first proposal).
    attempt: int = 0
    #: Capability of the action that was appraised.
    capability: str = ""
    #: Input of the action that was appraised.
    input: str = ""
    #: The appraisal itself.
    appraisal: Optional[ConscienceAppraisal] = None
    #: What the runtime did with it. Can differ from the stance (e.g. Revise at the limit becomes Escalated).
    effect: ConscienceEffect = ConscienceEffect.Proceeded
    #: Why the effect differs from the stance, if it does.
    note: str = ""
    #: Commitment created from this appraisal, if any.
    commitment_id: Optional[uuid.UUID] = None
    #: When the appraisal was made.
    timestamp_utc: datetime = field(default_factory=utc_now)

    @property
    def stopped(self) -> bool:
        """``True`` when the effect stopped the transition (Escalated or Vetoed)."""
        return self.effect in (ConscienceEffect.Escalated, ConscienceEffect.Vetoed)


@dataclass(frozen=True)
class Precedent:
    """A similar past transition retrieved for precedent-based reasoning (IBAAM 6, Section 6.4.1): "this plan
    resembles transition X, blocked by Y for protecting Z"."""

    #: The past transition.
    transition_id: uuid.UUID
    #: Plan similarity in 0..1.
    similarity: float
    #: How the controls treated it.
    outcome: PrecedentOutcome
    #: Its capability.
    capability: str
    #: Its capability input.
    input: str
    #: Invariants that blocked it (when :attr:`outcome` is Blocked).
    blocked_by: list[str] = field(default_factory=list)
    #: Protected-value tags of the invariants that blocked it.
    values: list[str] = field(default_factory=list)
    #: Behavioural drift score its resulting state produced, if it reached reduction.
    realised_drift: Optional[int] = None


@dataclass(frozen=True)
class ConscienceInput:
    """Everything a conscience may see when appraising a plan (IBAAM 6, Section 6.1).

    The appraisal must be a function of this input alone, so the same plan and the same history always give the same
    appraisal (determinism serves replay and audit), while a different history can give a different one (history
    sensitivity, the property that separates a conscience from an invariant).
    """

    #: Plan or answer appraisal.
    point: AppraisalPoint
    #: The action that will actually run (after plan invariants).
    proposed: Optional[AgentAction]
    #: The last committed agent state.
    state: Optional[AgentState]
    #: Behavioural stability over recent history.
    trend: Optional[StabilityAssessment]
    #: Similar past transitions, most similar first.
    precedents: list[Precedent]
    #: Live commitments in scope for this plan.
    commitments: list[Commitment]
    #: Smoothed rate at which the controls have blocked plans in this deployment (the cold-start prior).
    base_rate: float = 0.25
    #: Number of labelled transitions (passed or blocked) the base rate was computed from.
    labelled_history: int = 0
    #: Answer point only: the drift score of the proposed next state against the baseline in force (a dry run of the
    #: commit). ``None`` at the plan point, where drift is forecast from precedents.
    forecast_drift_score: Optional[int] = None
    #: Drift score at which the drift-bound invariant rejects; used to normalise drift.
    drift_bound: int = 70
    #: Answer point only: the observed outcome.
    outcome: Optional[AgentOutcome] = None
    #: Revision attempt (0 for the planner's first proposal).
    attempt: int = 0

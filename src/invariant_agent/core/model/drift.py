"""Drift baselines, records, scores and the aggregate drift report."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .._util import new_id, utc_now
from .agent import AgentState
from .enums import DriftType, InvariantCategory, InvariantSeverity, TransitionPhase
from .stability import StabilityAssessment


@dataclass
class DriftBaseline:
    """An operator-approved reference state used for drift scoring."""

    #: Baseline identifier.
    id: uuid.UUID = field(default_factory=new_id)
    #: UTC time of approval.
    approved_at_utc: datetime = field(default_factory=utc_now)
    #: Version of the approved state.
    state_version: int = 0
    #: Justification recorded at approval.
    reason: str = ""
    #: Snapshot of the approved state.
    state: AgentState = field(default_factory=AgentState)


@dataclass
class DriftBaselineSnapshot:
    """The drift baseline a transition was scored against, captured on the transition itself.

    Self-contained (it carries the baseline state, not just its id) so replay is deterministic even if the baseline
    store is in-memory, has been reset, or has since approved a different baseline.
    """

    #: Id of the approved baseline, or ``None`` if none was approved (the transition's ``before`` state was used).
    baseline_id: Optional[uuid.UUID] = None
    #: State version of the approved baseline, if any.
    state_version: Optional[int] = None
    #: When the baseline was approved, if any.
    approved_at_utc: Optional[datetime] = None
    #: The approved baseline state, or ``None`` when no baseline was approved.
    state: Optional[AgentState] = None

    @property
    def has_approved_baseline(self) -> bool:
        """True when an operator-approved baseline was in force (not persisted)."""
        return self.baseline_id is not None

    @staticmethod
    def from_baseline(baseline: Optional[DriftBaseline]) -> "DriftBaselineSnapshot":
        """Captures ``baseline`` (which may be ``None``, meaning "no approved baseline")."""
        if baseline is None:
            return DriftBaselineSnapshot()
        return DriftBaselineSnapshot(
            baseline_id=baseline.id,
            state_version=baseline.state_version,
            approved_at_utc=baseline.approved_at_utc,
            state=baseline.state,
        )


@dataclass
class DriftRecord:
    """A single observed drift event."""

    #: Kind of drift observed.
    type: DriftType = DriftType.None_
    #: Explanation of what changed.
    reason: str = ""
    #: Id of the transition in which drift was observed.
    transition_id: str = ""
    #: UTC time of the observation.
    timestamp_utc: Optional[datetime] = None
    #: Lifecycle phase of the transition when drift was observed.
    phase: TransitionPhase = TransitionPhase.Created
    #: Severity band of the drift.
    severity: InvariantSeverity = InvariantSeverity.Info
    #: Numeric drift score.
    score: int = 0


@dataclass
class DriftScore:
    """A behavioural drift score with its severity band and explanation."""

    #: Numeric score (0 means no adaptive divergence).
    score: int = 0
    #: Severity band derived from the score.
    severity: InvariantSeverity = InvariantSeverity.Info
    #: Human-readable description of what changed.
    explanation: str = ""


@dataclass
class DriftAttribution:
    """Count and total score of a subset of behavioural drift records."""

    #: Number of drift records.
    events: int = 0
    #: Sum of their scores.
    score: int = 0
    #: Share of all behavioural drift score (0 when there is none).
    share: float = 0.0


@dataclass
class DriftReport:
    """Aggregate observability report over transition history: drift, invariant failures, capability usage and the
    behavioural stability assessment."""

    #: Total number of transitions analysed.
    transition_count: int = 0
    #: Number of rejected transitions.
    rejected_transitions: int = 0
    #: Sum of all drift record scores.
    total_drift_score: int = 0
    #: Highest severity among drift records.
    highest_drift_severity: InvariantSeverity = InvariantSeverity.Info
    #: Number of times each capability was proposed.
    capability_usage: dict[str, int] = field(default_factory=dict)
    #: Failure counts keyed by ``"InvariantName [Category]"``.
    invariant_failures: dict[str, int] = field(default_factory=dict)
    #: Failure counts grouped by invariant category.
    invariant_failures_by_category: dict[InvariantCategory, int] = field(default_factory=dict)
    #: Number of drift records per drift type.
    drift_counts: dict[DriftType, int] = field(default_factory=dict)
    #: The most recent drift records (up to 10).
    recent_drift: list[DriftRecord] = field(default_factory=list)
    #: Behavioural stability assessment for the same history.
    stability: StabilityAssessment = field(default_factory=StabilityAssessment)
    #: Transitions that carry at least one HIG-C conflict record.
    conflict_transitions: int = 0
    #: Transitions held for human review because of a fundamental conflict.
    escalated_transitions: int = 0
    #: Behavioural drift attributed to governance (IBAAM 3 revised, Section 9): drift on transitions whose conflict
    #: record prescribed a change of behaviour. Expected; replaying the log reproduces it.
    attributed_drift: DriftAttribution = field(default_factory=DriftAttribution)
    #: Behavioural drift the governance process does not explain: drift on transitions with no conflict record that
    #: prescribed a change. A governance signal even when every invariant is satisfied.
    unattributed_drift: DriftAttribution = field(default_factory=DriftAttribution)
    #: Missing expected change: conflict records whose prescribed outcome did not take effect (a declined transition
    #: that still completed, an alternative that was not the one executed, or an overridden invariant that still
    #: blocked). Suggests a resolution is not taking effect; feeds review.
    missing_expected_change: list[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        """``True`` if any drift was recorded."""
        return len(self.drift_counts) > 0

    @property
    def total_drift_events(self) -> int:
        """Total number of drift records."""
        return sum(self.drift_counts.values())

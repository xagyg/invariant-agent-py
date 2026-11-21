"""Behavioural drift scoring, the in-memory baseline store and the runtime drift tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ._util import cs_str, is_blank
from .abstractions import IDriftBaselineStore
from .model.agent import AgentState
from .model.drift import DriftBaseline, DriftRecord, DriftScore
from .model.enums import DriftType, InvariantSeverity, TransitionStatus
from .model.transition import Transition

# Memory keys that change on almost every transition and so are excluded from drift scoring.
VOLATILE_MEMORY_KEYS = frozenset({"lastoutcome", "last_action"})


@dataclass
class _AdaptiveStateSnapshot:
    """Normalised, order-independent view of the adaptive parts of an :class:`AgentState`."""

    goal: str = ""
    mode: str = ""
    policies: list[str] = field(default_factory=list)
    #: Memory keyed by lower-cased key (case-insensitive), values as strings.
    memory: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def of(state: AgentState) -> "_AdaptiveStateSnapshot":
        memory: dict[str, str] = {}
        for key in sorted((k for k in state.memory if k.lower() not in VOLATILE_MEMORY_KEYS), key=str.lower):
            memory[key.lower()] = cs_str(state.memory[key])
        # Goals may be stored on AgentState.goal or as a "goal" memory entry; prefer the explicit field.
        memory_goal = memory.get("goal", "")
        return _AdaptiveStateSnapshot(
            goal=memory_goal if is_blank(state.goal) else state.goal,
            mode=state.mode or "",
            policies=sorted(state.policies or [], key=str.lower),
            memory=memory,
        )


class BehaviouralDriftDetector:
    """Scores behavioural drift by comparing the adaptive parts of two agent states (goal, mode, policies and
    non-volatile memory).

    Scoring weights: goal change +30, mode change +20, policy changes +10 each (max 30), memory changes +10 per
    added/removed key and +5 per changed value (max 40).

    Scores are mapped to severity: at or above :attr:`DEFAULT_BOUND_THRESHOLD` is Error, 40 or more is Warning,
    otherwise Info. The same scores feed both the ``BehaviouralDriftBoundInvariant`` (governance) and drift/stability
    reports (observability).
    """

    #: Number of transitions used as a comparison window by horizon-based measurements.
    DEFAULT_HORIZON = 3
    #: Minimum per-transition score that is reported as a drift record.
    REPORTING_THRESHOLD = 25
    #: Default maximum admissible drift score before the drift-bound invariant rejects.
    DEFAULT_BOUND_THRESHOLD = 70

    def __init__(self, baseline_store: Optional[IDriftBaselineStore] = None) -> None:
        """Creates a detector. With a ``baseline_store`` it prefers an approved drift baseline when one exists;
        without one it always scores against the transition's prior state."""
        self._baseline_store = baseline_store

    def detect(self, transitions: Sequence[Transition]) -> list[DriftRecord]:
        """Produces a drift record for each completed transition whose local (before vs after) drift score reaches
        :attr:`REPORTING_THRESHOLD`."""
        records = []
        for transition in transitions:
            # Only completed transitions with a resulting state can exhibit behavioural drift.
            if transition.after is None or transition.status != TransitionStatus.Completed:
                continue
            delta = self.score_states(transition.before or AgentState(), transition.after)
            if delta.score == 0 or delta.score < self.REPORTING_THRESHOLD:
                continue
            records.append(DriftRecord(
                type=DriftType.BehaviouralDrift,
                reason=delta.explanation,
                transition_id=str(transition.id),
                timestamp_utc=transition.timestamp,
                phase=transition.phase,
                severity=delta.severity,
                score=delta.score,
            ))
        return records

    def score(self, transition: Transition) -> DriftScore:
        """Scores the proposed state of ``transition`` against the approved baseline if one exists now, otherwise
        against the transition's prior state. Governance uses :meth:`score_as_recorded`."""
        if transition.after is None:
            return DriftScore(score=0, severity=InvariantSeverity.Info,
                              explanation="Transition does not contain proposed adaptive state.")
        return self.score_states(self._baseline_for(transition), transition.after)

    def score_as_recorded(self, transition: Transition) -> DriftScore:
        """Scores ``transition`` against the baseline recorded on it (``Transition.drift_baseline``), falling back to
        :meth:`score` when none was recorded.

        This is what governance uses. During a live run the recorded baseline is the current one, so the result is
        identical to :meth:`score`; during replay it keeps the score independent of baselines approved after the
        transition ran.
        """
        recorded = transition.drift_baseline
        if recorded is None or transition.after is None:
            return self.score(transition)
        return self.score_states(recorded.state or transition.before or AgentState(), transition.after)

    def score_states(self, baseline: AgentState, current: AgentState) -> DriftScore:
        """Scores the adaptive difference between two states."""
        score, reason = self._compare(_AdaptiveStateSnapshot.of(baseline), _AdaptiveStateSnapshot.of(current))
        return DriftScore(score=score, severity=self._to_severity(score), explanation=reason)

    # Prefer the operator-approved baseline; fall back to the transition's prior state.
    def _baseline_for(self, transition: Transition) -> AgentState:
        current = self._baseline_store.current if self._baseline_store is not None else None
        if current is not None and current.state is not None:
            return current.state
        return transition.before or AgentState()

    # Weighted comparison of goal, mode, policies and memory; returns total score and reasons.
    @staticmethod
    def _compare(baseline: _AdaptiveStateSnapshot, current: _AdaptiveStateSnapshot) -> tuple[int, str]:
        score = 0
        reasons = []
        if baseline.goal != current.goal:
            score += 30
            reasons.append(f"goal changed from '{_display(baseline.goal)}' to '{_display(current.goal)}'")
        if baseline.mode != current.mode:
            score += 20
            reasons.append(f"mode changed from '{_display(baseline.mode)}' to '{_display(current.mode)}'")
        added_policies = [p for p in dict.fromkeys(current.policies) if p not in baseline.policies]
        removed_policies = [p for p in dict.fromkeys(baseline.policies) if p not in current.policies]
        if added_policies or removed_policies:
            score += min(30, (len(added_policies) + len(removed_policies)) * 10)
            reasons.append(f"policies changed (+{len(added_policies)}/-{len(removed_policies)})")
        memory_score, memory_reason = BehaviouralDriftDetector._compare_memory(baseline.memory, current.memory)
        if memory_score > 0:
            score += memory_score
            reasons.append(memory_reason)
        return score, ("No adaptive state divergence detected." if not reasons else "; ".join(reasons))

    # Counts added, removed and changed memory keys (case-insensitive) and caps the score at 40.
    @staticmethod
    def _compare_memory(baseline: dict[str, str], current: dict[str, str]) -> tuple[int, str]:
        added = sum(1 for k in current if k not in baseline)
        removed = sum(1 for k in baseline if k not in current)
        changed = sum(1 for k, v in baseline.items() if k in current and current[k] != v)
        if added + removed + changed == 0:
            return 0, ""
        return (min(40, added * 10 + removed * 10 + changed * 5),
                f"adaptive memory changed (+{added}/-{removed}/~{changed})")

    # Maps a drift score to an invariant severity band.
    @classmethod
    def _to_severity(cls, score: int) -> InvariantSeverity:
        if score >= cls.DEFAULT_BOUND_THRESHOLD:
            return InvariantSeverity.Error
        if score >= 40:
            return InvariantSeverity.Warning
        return InvariantSeverity.Info


def _display(value: Optional[str]) -> str:
    return "<empty>" if is_blank(value) else value


def copy_state(state: AgentState) -> AgentState:
    """Defensive copy so later changes to the live state cannot alter an approved baseline.

    Memory values are copied by reference (shallow copy), as in the C# stores.
    """
    return AgentState(
        version=state.version,
        mode=state.mode,
        goal=state.goal,
        policies=list(state.policies),
        events=list(state.events),
        memory=dict(state.memory),
    )


class InMemoryDriftBaselineStore(IDriftBaselineStore):
    """Process-local :class:`IDriftBaselineStore`. The approved baseline is lost when the process exits."""

    def __init__(self) -> None:
        self._current: Optional[DriftBaseline] = None

    @property
    def current(self) -> Optional[DriftBaseline]:
        return self._current

    def approve(self, state: AgentState, reason: str) -> DriftBaseline:
        self._current = DriftBaseline(state_version=state.version, reason=reason, state=copy_state(state))
        return self._current


class DriftTracker:
    """In-memory log of drift records raised at runtime (e.g. governance drift from replay validation).

    Merged with behavioural drift by the drift analyzer.
    """

    def __init__(self) -> None:
        self._records: list[DriftRecord] = []

    @property
    def records(self) -> list[DriftRecord]:
        """All recorded drift events."""
        return self._records

    def record(self, record: DriftRecord) -> None:
        """Records a drift event."""
        self._records.append(record)

    def count(self, type: DriftType) -> int:
        """Counts records of the given type."""
        return sum(1 for r in self._records if r.type == type)

    def get_recent(self, count: int) -> list[DriftRecord]:
        """Returns the most recent records (at most ``count``)."""
        return self._records[-count:] if count > 0 else []

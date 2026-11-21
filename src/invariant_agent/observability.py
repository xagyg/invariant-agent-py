"""IA2 Observability: drift analysis and behavioural stability evaluation (IBAAM 2)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from .core._util import cs_str, is_blank
from .core.abstractions import IDriftAnalyzer, IStabilityEvaluator
from .core.drift import VOLATILE_MEMORY_KEYS, BehaviouralDriftDetector, DriftTracker
from .core.model.agent import AgentAction, AgentState
from .core.model.control import GovernanceCandidate
from .core.model.drift import DriftAttribution, DriftRecord, DriftReport
from .core.model.enums import (
    DriftType,
    InvariantCategory,
    InvariantSeverity,
    MeasurementConfidence,
    StabilityRegion,
    TransitionEventStage,
    TransitionStatus,
)
from .core.model.stability import StabilityAssessment, StabilityVector
from .core.model.transition import Transition, TransitionEvent


def _is_failed_invariant_event(e: TransitionEvent) -> bool:
    # Prefers structured "Passed" metadata; falls back to message text for older events.
    if e.stage != TransitionEventStage.Invariant:
        return False
    if "Passed" in e.metadata:
        passed = e.metadata["Passed"]
        return isinstance(passed, bool) and not passed
    return "Failed" in e.message


class SimpleDriftAnalyzer(IDriftAnalyzer):
    """Default :class:`IDriftAnalyzer`. Combines runtime drift records (e.g. governance drift from replay validation)
    with behavioural drift detected from history, and attaches a stability assessment."""

    def __init__(self, drift_tracker: DriftTracker, behavioural_drift_detector: BehaviouralDriftDetector,
                 stability_evaluator: IStabilityEvaluator) -> None:
        self._drift_tracker = drift_tracker
        self._detector = behavioural_drift_detector
        self._stability = stability_evaluator

    def analyze(self, transitions: Sequence[Transition]) -> DriftReport:
        transitions = list(transitions)
        capability_usage = dict(Counter(
            t.proposed_action.capability for t in transitions if t.proposed_action is not None))
        failed_events = [e for t in transitions for e in t.events if _is_failed_invariant_event(e)]
        invariant_failures = dict(Counter(
            f"{_extract_invariant_name(e.message)} [{_extract_invariant_category(e).name}]" for e in failed_events))
        invariant_failures_by_category = dict(Counter(_extract_invariant_category(e) for e in failed_events))

        # Merge tracked (runtime) drift with behavioural drift recomputed from history.
        records = sorted(list(self._drift_tracker.records) + self._detector.detect(transitions),
                         key=lambda r: _sort_key(r))
        drift_counts = dict(Counter(r.type for r in records))
        total_drift_score = sum(r.score for r in records)
        highest = max((r.severity for r in records), default=InvariantSeverity.Info)

        # HIG-C (IBAAM 3 revised, Section 9): split behavioural drift by whether governance explains it.
        by_id = {str(t.id): t for t in transitions}
        behavioural = [r for r in records if r.type == DriftType.BehaviouralDrift]
        attributed = [r for r in behavioural if r.transition_id in by_id and _prescribes_change(by_id[r.transition_id])]
        attributed_ids = {id(r) for r in attributed}
        unattributed = [r for r in behavioural if id(r) not in attributed_ids]
        behavioural_score = sum(r.score for r in behavioural)

        return DriftReport(
            transition_count=len(transitions),
            rejected_transitions=sum(1 for t in transitions if t.status == TransitionStatus.Rejected),
            capability_usage=capability_usage,
            invariant_failures=invariant_failures,
            invariant_failures_by_category=invariant_failures_by_category,
            drift_counts=drift_counts,
            recent_drift=records[-10:],
            total_drift_score=total_drift_score,
            highest_drift_severity=highest,
            conflict_transitions=sum(1 for t in transitions if t.conflicts),
            escalated_transitions=sum(1 for t in transitions if t.governance_escalated),
            attributed_drift=_attribution(attributed, behavioural_score),
            unattributed_drift=_attribution(unattributed, behavioural_score),
            missing_expected_change=[item for t in transitions for item in _missing_expected_change(t)],
            stability=self._stability.evaluate(transitions),
        )


def _sort_key(record: DriftRecord):
    return record.timestamp_utc.timestamp() if record.timestamp_utc is not None else float("-inf")


# A transition whose conflict record prescribes different behaviour from simply carrying out the proposal.
def _prescribes_change(transition: Transition) -> bool:
    return any(c.changes_behaviour for c in transition.conflicts or [])


def _attribution(subset: list[DriftRecord], total: int) -> DriftAttribution:
    score = sum(r.score for r in subset)
    return DriftAttribution(events=len(subset), score=score, share=0.0 if total == 0 else score / total)


def _missing_expected_change(transition: Transition) -> Iterable[str]:
    """Conflict records whose prescribed outcome did not take effect on the transition (Section 9, "missing expected
    change"): a decline on a transition that still completed, a selected alternative that was not the capability
    executed, or an overridden invariant that still appears among the blocking invariants."""
    for conflict in transition.conflicts or []:
        short = transition.id.hex[:8]
        if conflict.selected == GovernanceCandidate.DECLINE_ID:
            if transition.status == TransitionStatus.Completed:
                yield f"{short} {conflict.scope}: declined, but the transition completed."
            continue
        if (conflict.selected != GovernanceCandidate.PROPOSED_ID and transition.outcome is not None
                and (transition.outcome.capability or "").lower() != (conflict.selected_capability or "").lower()):
            yield (f"{short} {conflict.scope}: selected {conflict.selected} ({conflict.selected_capability}), "
                   f"but '{transition.outcome.capability}' executed.")
        blocking = transition.blocking_invariants or []
        still_blocking = [name for name in dict.fromkeys(conflict.overridden) if name in blocking]
        if still_blocking:
            yield f"{short} {conflict.scope}: overrode {', '.join(still_blocking)}, but it still blocked the transition."


# Invariant events are formatted "Name: Passed|Failed reason"; take the text before the colon.
def _extract_invariant_name(message: str) -> str:
    index = message.find(":")
    return "Unknown" if index <= 0 else message[:index]


# Reads the "Category" metadata; defaults to Integrity when missing.
def _extract_invariant_category(e: TransitionEvent) -> InvariantCategory:
    category = InvariantCategory.parse(cs_str(e.metadata.get("Category"))) if "Category" in e.metadata else None
    return category if category is not None else InvariantCategory.Integrity


class TransitionHistoryStabilityEvaluator(IStabilityEvaluator):
    """Default :class:`IStabilityEvaluator`. Computes Agent Identity Evaluation (AIE) metrics over windows of
    completed transitions (IBAAM 2: Measuring Behavioural Stability).

    The latest ``DEFAULT_HORIZON`` completed transitions form the current window and the window before that forms the
    previous one. Drift magnitude is the Euclidean distance between the two stability vectors.

    These measurements indicate behavioural change; they do not reject transitions. Invariants remain responsible for
    deciding whether change is acceptable.
    """

    def __init__(self, drift_detector: BehaviouralDriftDetector) -> None:
        self._detector = drift_detector

    def evaluate(self, transitions: Sequence[Transition]) -> StabilityAssessment:
        transitions = list(transitions)
        completed = [t for t in transitions if _is_completed(t)]
        if not completed:
            return StabilityAssessment(
                current=StabilityVector(), confidence=MeasurementConfidence.Low, region=StabilityRegion.Stable,
                recommendations=["Collect completed transitions before interpreting stability scores."])
        horizon = BehaviouralDriftDetector.DEFAULT_HORIZON
        # Current window = last DEFAULT_HORIZON completed transitions; prior window = the DEFAULT_HORIZON before it.
        current_window = completed[-min(horizon, len(completed)):]
        earlier = completed[: max(0, len(completed) - len(current_window))]
        prior_window = earlier[-min(horizon, len(earlier)):] if earlier else []
        current = self._measure(transitions, current_window, is_latest=True)
        previous = None if not prior_window else self._measure(transitions, prior_window, is_latest=False)
        drift_magnitude = 0 if previous is None else _distance(previous, current)
        region = _classify_region(transitions, current, drift_magnitude)
        return StabilityAssessment(
            current=current,
            previous=previous,
            drift_magnitude=drift_magnitude,
            region=region,
            confidence=_to_confidence(len(completed)),
            recommendations=_recommend(region, current, drift_magnitude),
        )

    def _measure(self, all_transitions: list[Transition], window: list[Transition], is_latest: bool) -> StabilityVector:
        """Computes the five AIE dimensions for a window of completed transitions.

        Identifiability looks at the window's span of history, rejected transitions included (the window itself holds
        only completed ones, so scoring it directly could never see a rejection). Consistency and Recovery are
        cumulative up to the end of the span, so the previous window's vector reflects the history as it stood then.
        Scoring both windows over the whole history made the two values identical, so they never contributed to drift
        magnitude.
        """
        baseline = window[0].before or AgentState()
        current = window[-1].after
        drift_score = self._detector.score_states(baseline, current).score
        span, history_to_date = _span(all_transitions, window, is_latest)
        return StabilityVector(
            identifiability=_score_identifiability(span),
            continuity=_clamp(100 - drift_score),
            consistency=_score_consistency(history_to_date),
            persistence=_score_persistence(baseline, current),
            recovery=_score_recovery(history_to_date),
        )


def _is_completed(t: Transition) -> bool:
    return t.status == TransitionStatus.Completed and t.after is not None


def _index_of(transitions: list[Transition], transition: Transition) -> int:
    return next((i for i, t in enumerate(transitions) if t is transition), -1)


def _span(all_transitions: list[Transition], window: list[Transition],
          is_latest: bool) -> tuple[list[Transition], list[Transition]]:
    """The span of a window: every transition (any status) after the completed transition that precedes the window,
    up to the window's last completed transition, or to the end of history for the latest window so that trailing
    rejections count. Also returns the history from the start up to the end of that span."""
    first = _index_of(all_transitions, window[0])
    last = len(all_transitions) - 1 if is_latest else _index_of(all_transitions, window[-1])
    if first < 0 or last < 0:
        return window, all_transitions
    start = first
    while start > 0 and not _is_completed(all_transitions[start - 1]):
        start -= 1
    return all_transitions[start:last + 1], all_transitions[:last + 1]


# Identifiability: -15 per rejection and -5 per distinct capability beyond 3.
def _score_identifiability(transitions: list[Transition]) -> int:
    actions = [t for t in transitions if t.proposed_action is not None]
    if not actions:
        return 100
    distinct_capabilities = len({(t.proposed_action.capability or "").lower() for t in actions})
    rejection_penalty = sum(1 for t in transitions if t.status == TransitionStatus.Rejected) * 15
    spread_penalty = max(0, distinct_capabilities - 3) * 5
    return _clamp(100 - rejection_penalty - spread_penalty)


# Consistency: share of repeated (capability, input) pairs that always produced the same outcome.
def _score_consistency(transitions: list[Transition]) -> int:
    groups: dict[str, list[Transition]] = {}
    for t in transitions:
        if t.proposed_action is None or t.outcome is None:
            continue
        groups.setdefault(_normalize_comparable_action(t.proposed_action).lower(), []).append(t)
    comparable = [g for g in groups.values() if len(g) > 1]
    if not comparable:
        return 100
    inconsistent = sum(1 for g in comparable if len({_outcome_signature(t).lower() for t in g}) > 1)
    return _clamp(100 - (inconsistent * 100 // len(comparable)))


# Persistence: penalises lost goal (-30), mode change (-20), lost policies (-10 each, max 30) and lost stable memory
# keys (-10 each, max 20).
def _score_persistence(baseline: AgentState, current: AgentState) -> int:
    score = 100
    if _effective_goal(baseline) != _effective_goal(current):
        score -= 30
    if (baseline.mode or "") != (current.mode or ""):
        score -= 20
    current_policies = {p.lower() for p in current.policies}
    lost_policies = len({p.lower() for p in baseline.policies if p.lower() not in current_policies})
    score -= min(30, lost_policies * 10)
    current_memory = _stable_memory(current)
    lost_memory = sum(1 for k in _stable_memory(baseline) if k not in current_memory)
    score -= min(20, lost_memory * 10)
    return _clamp(score)


# Recovery: share of rejected transitions followed later by at least one completed transition.
def _score_recovery(transitions: list[Transition]) -> int:
    rejected = [t for t in transitions if t.status == TransitionStatus.Rejected]
    if not rejected:
        return 100
    recovered = sum(1 for r in rejected if any(t.timestamp > r.timestamp and t.status == TransitionStatus.Completed
                                               for t in transitions))
    return _clamp(recovered * 100 // len(rejected))


# Critical: invariant failures with low average or high drift; Intervention: average below 60 or drift 40+;
# Watch: average below 85, drift 20+, or any invariant failure; otherwise Stable.
def _classify_region(transitions: list[Transition], vector: StabilityVector, drift_magnitude: int) -> StabilityRegion:
    has_failure = any(_is_failed_invariant_event(e) for t in transitions for e in t.events)
    if has_failure and (vector.average_score < 75 or drift_magnitude >= 25):
        return StabilityRegion.Critical
    if vector.average_score < 60 or drift_magnitude >= 40:
        return StabilityRegion.Intervention
    if vector.average_score < 85 or drift_magnitude >= 20 or has_failure:
        return StabilityRegion.Watch
    return StabilityRegion.Stable


# Maps the region (and notable drift or poor recovery) to governance recommendations.
def _recommend(region: StabilityRegion, vector: StabilityVector, drift_magnitude: int) -> list[str]:
    recommendations = [{
        StabilityRegion.Stable: "Continue normal monitoring.",
        StabilityRegion.Watch: "Increase monitoring and review recent adaptive changes.",
        StabilityRegion.Intervention: "Run enhanced validation before approving further adaptation.",
    }.get(region, "Trigger human review before accepting additional behavioural changes.")]
    if drift_magnitude >= 20:
        recommendations.append("Compare the latest stability vector with the approved baseline.")
    if vector.recovery < 80:
        recommendations.append("Review rollback or recovery behaviour after rejected transitions.")
    return recommendations


# Confidence grows with the number of completed transitions relative to DEFAULT_HORIZON.
def _to_confidence(completed: int) -> MeasurementConfidence:
    horizon = BehaviouralDriftDetector.DEFAULT_HORIZON
    if completed >= horizon * 2:
        return MeasurementConfidence.High
    if completed >= horizon:
        return MeasurementConfidence.Medium
    return MeasurementConfidence.Low


# Euclidean distance between two stability vectors, clamped to 0-100 (rounded half to even, like Math.Round).
def _distance(left: StabilityVector, right: StabilityVector) -> int:
    total = ((left.identifiability - right.identifiability) ** 2
             + (left.continuity - right.continuity) ** 2
             + (left.consistency - right.consistency) ** 2
             + (left.persistence - right.persistence) ** 2
             + (left.recovery - right.recovery) ** 2)
    return _clamp(int(round(math.sqrt(total))))


# Key used to group repeated actions for the consistency score.
def _normalize_comparable_action(action: AgentAction) -> str:
    return f"{action.capability or ''}:{(action.input or '').strip()}"


# Comparable representation of an outcome (success plus result, or failure plus error).
def _outcome_signature(transition: Transition) -> str:
    if transition.outcome is None:
        return "missing"
    return f"ok:{transition.outcome.result}" if transition.outcome.success else f"fail:{transition.outcome.error}"


# Explicit goal if set, otherwise the "goal" memory entry.
def _effective_goal(state: AgentState) -> str:
    if not is_blank(state.goal):
        return state.goal
    return _stable_memory(state).get("goal", "")


# Memory without volatile keys, keyed case-insensitively, values as strings.
def _stable_memory(state: AgentState) -> dict[str, str]:
    return {k.lower(): cs_str(v) for k, v in state.memory.items() if k.lower() not in VOLATILE_MEMORY_KEYS}


def _clamp(score: int) -> int:
    return max(0, min(100, score))


__all__ = ["SimpleDriftAnalyzer", "TransitionHistoryStabilityEvaluator"]

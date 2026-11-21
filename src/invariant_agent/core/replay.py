"""Replay validation: re-evaluates invariants against recorded transitions to verify governance is reproducible."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._util import utc_now
from .abstractions import IInvariantEvaluator
from .drift import DriftTracker
from .model.conscience import ConscienceAppraisalRecord
from .model.drift import DriftRecord
from .model.enums import (
    AppraisalPoint,
    DriftType,
    InvariantScope,
    InvariantSeverity,
    TransitionEventStage,
    TransitionPhase,
    TransitionStatus,
)
from .model.transition import Transition, TransitionContext

S = InvariantScope


@dataclass
class ReplayComparison:
    """Side-by-side comparison of a transition's original governance decision and the decision obtained by
    re-evaluating invariants on replay."""

    #: Status recorded when the transition originally ran.
    original_status: TransitionStatus = TransitionStatus.Completed
    #: Status implied by re-evaluation (Rejected or Completed).
    replay_status: TransitionStatus = TransitionStatus.Completed
    #: Phase recorded when the transition originally ran.
    original_phase: TransitionPhase = TransitionPhase.Created
    #: Phase implied by re-evaluation.
    replay_phase: TransitionPhase = TransitionPhase.Created
    #: Failed-invariant messages from the original event trail.
    original_violations: list[str] = field(default_factory=list)
    #: Unresolved violations found on replay.
    replay_violations: list[str] = field(default_factory=list)
    #: True if the transition carries the drift baseline it was originally scored against. When false (history
    #: recorded before baselines were captured), replay used the currently approved baseline, so a drift-related
    #: mismatch may reflect a baseline change rather than a rule change.
    drift_baseline_recorded: bool = False
    #: True if the conscience stopped the original transition (Escalate or Veto) and replay applied that recorded
    #: appraisal rather than re-appraising.
    conscience_stop_recorded: bool = False


@dataclass
class ReplayValidationResult:
    """Result of replay validation for a single transition."""

    #: ``True`` if replay reproduced the original accept/reject decision.
    passed: bool = False
    #: ``GovernanceDrift`` on mismatch, otherwise ``None_``.
    drift_type: DriftType = DriftType.None_
    #: Severity derived from the mismatch score.
    severity: InvariantSeverity = InvariantSeverity.Info
    #: Mismatch score: status change +100, phase change +50, +10 per difference in violation count.
    score: int = 0
    #: Explanation of the mismatch.
    reason: str = ""
    #: Detailed original-vs-replay comparison.
    comparison: Optional[ReplayComparison] = None


class ReplayValidator:
    """Re-evaluates invariants against a recorded transition to verify that governance decisions are reproducible.
    A different accept/reject outcome is reported as governance drift.

    Callers should pass a cloned transition, because invariant evaluation appends events to it. Behavioural drift is
    scored against the baseline recorded on the transition, so the result depends only on the recorded transition and
    the current rules, not on baselines approved since. Transitions recorded before baselines were captured fall back
    to the current baseline (see :attr:`ReplayComparison.drift_baseline_recorded`). A transition stopped by the
    conscience (Escalate or Veto) is replayed by re-evaluating invariants up to the point where it was stopped and
    then applying the *recorded* appraisal, so replay reproduces the live decision without re-running the conscience
    (IBAAM 6, Section 6.4.4).
    """

    def __init__(self, evaluator: IInvariantEvaluator, drift_tracker: DriftTracker) -> None:
        self._evaluator = evaluator
        self._drift_tracker = drift_tracker

    def validate(self, context: TransitionContext, record_drift: bool = False) -> ReplayValidationResult:
        """Replays governance for one transition and compares the result with the original.

        :param record_drift: when true, a mismatch is also written to the drift tracker as governance drift. Defaults
            to false so that replay is report-only: repeatedly replaying the same history must not change drift
            statistics.
        """
        transition = context.transition
        original_violations = _extract_violations(transition)
        reports = [self._evaluator.evaluate(context, scope) for scope in _scopes_for_replay(transition)]
        replay_violations = [f"{v.invariant}: {v.reason}" for r in reports if not r.passed for v in r.violations]
        conscience_stop = _recorded_conscience_stop(transition)
        # Invariants decide first; a recorded conscience stop applies only where the controls let the plan through.
        replay_rejected = any(not r.passed for r in reports) or conscience_stop is not None
        original_rejected = transition.status == TransitionStatus.Rejected
        comparison = ReplayComparison(
            original_status=transition.status,
            replay_status=TransitionStatus.Rejected if replay_rejected else TransitionStatus.Completed,
            original_phase=transition.phase,
            replay_phase=TransitionPhase.Rejected if replay_rejected else transition.phase,
            original_violations=original_violations,
            replay_violations=replay_violations,
            drift_baseline_recorded=transition.drift_baseline is not None,
            conscience_stop_recorded=conscience_stop is not None,
        )
        # Mismatch score: status change +100, phase change +50, +10 per difference in violation count.
        score = 0
        if comparison.original_status != comparison.replay_status:
            score += 100
        if comparison.original_phase != comparison.replay_phase:
            score += 50
        score += abs(len(comparison.original_violations) - len(comparison.replay_violations)) * 10
        severity = (InvariantSeverity.Critical if score >= 100 else
                    InvariantSeverity.Error if score >= 50 else
                    InvariantSeverity.Warning if score > 0 else InvariantSeverity.Info)
        # A flip in the accept/reject decision is governance drift.
        if replay_rejected != original_rejected:
            result = ReplayValidationResult(passed=False, drift_type=DriftType.GovernanceDrift,
                                            reason="Replay validation mismatch.", comparison=comparison,
                                            severity=severity, score=score)
            if record_drift:
                self._drift_tracker.record(DriftRecord(
                    type=result.drift_type, reason=result.reason, transition_id=str(transition.id),
                    timestamp_utc=utc_now(), phase=transition.phase, severity=severity, score=score))
            return result
        return ReplayValidationResult(passed=True, drift_type=DriftType.None_, comparison=comparison,
                                      severity=severity, score=score)


# Failed-invariant messages from the original event trail.
def _extract_violations(transition: Transition) -> list[str]:
    return [e.message for e in transition.events
            if e.stage == TransitionEventStage.Invariant and "Failed" in e.message]


def _scopes_for_replay(transition: Transition) -> list[InvariantScope]:
    """Chooses which scopes to re-evaluate. For rejected transitions, evaluation stops at the scope that originally
    failed (mirroring the runtime pipeline); otherwise all checkpoints are replayed."""
    everything = [S.Plan, S.Execution, S.SelfModification, S.Reduction]
    # Stopped by the conscience at the plan point: nothing after plan validation ever ran.
    stop = _recorded_conscience_stop(transition)
    if stop is not None and stop.point == AppraisalPoint.Plan and not _has_any_invariant_failure(transition):
        return [S.Plan]
    if transition.status == TransitionStatus.Rejected:
        for i, scope in enumerate(everything):
            if _has_invariant_failure(transition, scope):
                return everything[: i + 1]
    return everything


# The recorded appraisal that stopped the transition, if the conscience stopped it.
def _recorded_conscience_stop(transition: Transition) -> Optional[ConscienceAppraisalRecord]:
    return next((a for a in reversed(transition.appraisals or []) if a.stopped), None)


def _has_any_invariant_failure(transition: Transition) -> bool:
    return any(e.stage == TransitionEventStage.Invariant and "Failed" in e.message for e in transition.events)


# True if the original trail contains a failed invariant event for the given scope.
def _has_invariant_failure(transition: Transition, scope: InvariantScope) -> bool:
    for e in transition.events:
        if e.stage != TransitionEventStage.Invariant or "Failed" not in e.message:
            continue
        value = e.metadata.get("Scope")
        if value is not None and str(value).lower() == scope.name.lower():
            return True
    return False

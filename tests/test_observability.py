"""Observability tests: drift baselines, drift reports, explain, replay (determinism and history) and behavioural
stability evaluation (AIE metrics)."""

from __future__ import annotations

from invariant_agent.core._util import utc_now
from invariant_agent.core.control import InvariantEvaluator
from invariant_agent.core.drift import BehaviouralDriftDetector, DriftTracker, InMemoryDriftBaselineStore
from invariant_agent.core.model import (
    AgentAction,
    AgentState,
    DriftBaselineSnapshot,
    DriftRecord,
    DriftType,
    InvariantCategory,
    InvariantSeverity,
    StabilityRegion,
    Transition,
    TransitionContext,
    TransitionPhase,
    TransitionStatus,
)
from invariant_agent.core.rendering import TransitionFormatter
from invariant_agent.core.replay import ReplayValidator
from invariant_agent.core.serialization import clone
from invariant_agent.safety import BehaviouralDriftBoundInvariant, NoDeleteInvariant

# ---------------------------------------------------------------------------------------------- drift baselines


def test_baseline_approve_stores_current_committed_state(fixture):
    fixture.run("memory-set goal=stable baseline")
    t = fixture.run("baseline-approve after review").transition
    baseline = fixture.baseline_store.current
    assert t.status == TransitionStatus.Completed
    assert baseline is not None
    assert baseline.state_version == 1
    assert baseline.state.memory["goal"] == "stable baseline"
    assert baseline.reason == "after review"


def test_baseline_show_returns_approved_baseline_details(fixture):
    fixture.run("memory-set goal=visible baseline")
    fixture.run("baseline-approve reviewer accepted")
    t = fixture.run("baseline-show").transition
    assert t.status == TransitionStatus.Completed
    for expected in ("==== DRIFT BASELINE ====", "StateVersion: 1", "Reason: reviewer accepted", "goal=visible baseline"):
        assert expected in t.outcome.result


def test_detector_uses_approved_baseline_for_transition_scoring(fixture):
    fixture.run("memory-set goal=approved goal")
    fixture.run("baseline-approve approved goal")
    transition = next(t for t in fixture.transition_store.get_all() if t.input == "memory-set goal=approved goal")
    assert BehaviouralDriftDetector(fixture.baseline_store).score(transition).score == 0


# ------------------------------------------------------------------------------------------------- drift reports


def test_analyze_with_recorded_drift_returns_drift_counts(fixture):
    fixture.drift_tracker.record(DriftRecord(type=DriftType.GovernanceDrift, reason="Replay validation mismatch.",
                                             transition_id="test-transition", timestamp_utc=utc_now(),
                                             phase=TransitionPhase.ExecutionValidation,
                                             severity=InvariantSeverity.Critical))
    report = fixture.drift_analyzer.analyze(fixture.transition_store.get_all())
    assert report.drift_counts[DriftType.GovernanceDrift] == 1
    assert len(report.recent_drift) == 1


def test_analyze_with_goal_change_returns_behavioural_drift(fixture):
    fixture.run("memory-set goal=ship behavioural drift")
    report = fixture.drift_analyzer.analyze(fixture.transition_store.get_all())
    assert report.drift_counts[DriftType.BehaviouralDrift] == 1
    assert report.total_drift_score > 0
    assert any(d.type == DriftType.BehaviouralDrift and "goal changed" in d.reason for d in report.recent_drift)


def test_analyze_with_repeated_drift_tool_does_not_report_read_only_transitions_as_new_behavioural_drift(fixture):
    fixture.run("echo before-goal")
    fixture.run("memory-set goal=go to shop")
    fixture.run("drift")
    fixture.run("drift")
    report = fixture.drift_analyzer.analyze(fixture.transition_store.get_all())
    assert report.drift_counts[DriftType.BehaviouralDrift] == 1
    (drift,) = [d for d in report.recent_drift if d.type == DriftType.BehaviouralDrift]
    (source,) = [t for t in fixture.transition_store.get_all() if str(t.id) == drift.transition_id]
    assert source.proposed_action.capability == "memory-set"


def test_analyze_with_sequential_goal_changes_reports_transition_local_deltas(fixture):
    for goal in ("sss", "we", "ppppp", "iuewiuer23423"):
        fixture.run(f"memory-set goal={goal}")
    reasons = [d.reason for d in fixture.drift_analyzer.analyze(fixture.transition_store.get_all()).recent_drift]
    for expected in ("goal changed from '<empty>' to 'sss'", "goal changed from 'sss' to 'we'",
                     "goal changed from 'we' to 'ppppp'", "goal changed from 'ppppp' to 'iuewiuer23423'"):
        assert any(expected in r for r in reasons)


def test_analyze_with_only_volatile_outcome_memory_does_not_return_behavioural_drift(fixture):
    fixture.run("echo hello")
    fixture.run("calc 10+20")
    report = fixture.drift_analyzer.analyze(fixture.transition_store.get_all())
    assert DriftType.BehaviouralDrift not in report.drift_counts


def test_analyze_with_invariant_failures_groups_failures_by_category(fixture):
    fixture.run("does-not-exist thing")
    fixture.run("memory-set password=secret")
    report = fixture.drift_analyzer.analyze(fixture.transition_store.get_all())
    assert report.invariant_failures_by_category[InvariantCategory.Safety] == 1
    assert report.invariant_failures_by_category[InvariantCategory.SelfModification] == 1
    assert report.invariant_failures["AllowedCapabilityInvariant [Safety]"] == 1
    assert report.invariant_failures["AllowedMemoryKeyInvariant [SelfModification]"] == 1


def test_drift_tool_with_invariant_failures_shows_category_grouping(fixture):
    fixture.run("does-not-exist thing")
    fixture.run("memory-set password=secret")
    result = fixture.run("drift").transition.outcome.result
    assert "Invariant failures by category:" in result
    assert "Safety: 1" in result
    assert "SelfModification: 1" in result


# --------------------------------------------------------------------------------------------------------- explain


def test_format_explain_includes_reduction_invariants(fixture):
    report = TransitionFormatter.format_explain(fixture.run("memory-set goal=show reduction governance").transition)
    assert "Reduction invariants:" in report
    assert "BehaviouralDriftBoundInvariant" in report


# -------------------------------------------------------------------------------------------- replay determinism

# A baseline far enough from the transition's state to exceed the drift bound (score 100 vs bound 70).
def distant_baseline_state() -> AgentState:
    return AgentState(goal="a different objective", mode="strict", policies=["p1", "p2", "p3"],
                      memory={"notes": "unrelated"})


# Completed transition whose After state is close to Before (score 40, under the bound).
def completed_transition(recorded_baseline) -> Transition:
    return Transition(input="memory-set goal=ship", before=AgentState(),
                      after=AgentState(version=1, memory={"goal": "ship"}),
                      proposed_action=AgentAction(capability="memory-set", input="goal=ship"),
                      phase=TransitionPhase.Completed, status=TransitionStatus.Completed,
                      drift_baseline=recorded_baseline)


def create_validator():
    baselines = InMemoryDriftBaselineStore()
    tracker = DriftTracker()
    evaluator = InvariantEvaluator([BehaviouralDriftBoundInvariant(BehaviouralDriftDetector(baselines))])
    return ReplayValidator(evaluator, tracker), baselines, tracker


def test_replay_uses_recorded_baseline_not_one_approved_later():
    validator, baselines, _ = create_validator()
    # Ran with no approved baseline; an operator approves a very different one afterwards.
    transition = completed_transition(DriftBaselineSnapshot.from_baseline(None))
    baselines.approve(distant_baseline_state(), "approved after the transition ran")
    result = validator.validate(TransitionContext(transition=clone(transition)))
    assert result.passed
    assert result.comparison.replay_status == TransitionStatus.Completed
    assert result.comparison.drift_baseline_recorded


def test_replay_uses_recorded_approved_baseline_after_baseline_is_replaced():
    validator, baselines, _ = create_validator()
    # Ran under a baseline matching its own state; later replaced by a distant one.
    original = baselines.approve(AgentState(memory={"goal": "ship"}), "original")
    transition = completed_transition(DriftBaselineSnapshot.from_baseline(original))
    baselines.approve(distant_baseline_state(), "replacement")
    result = validator.validate(TransitionContext(transition=clone(transition)))
    assert result.passed
    assert result.score == 0


def test_replay_without_recorded_baseline_falls_back_to_current_and_says_so():
    validator, baselines, _ = create_validator()
    # Legacy history: no baseline captured, so replay can only use the current one.
    transition = completed_transition(None)
    baselines.approve(distant_baseline_state(), "approved after the transition ran")
    result = validator.validate(TransitionContext(transition=clone(transition)))
    assert not result.passed
    assert not result.comparison.drift_baseline_recorded


def test_replay_is_report_only_by_default():
    tracker = DriftTracker()
    validator = ReplayValidator(InvariantEvaluator([NoDeleteInvariant()]), tracker)
    # Recorded as completed, but current rules reject it: a mismatch.
    result = validator.validate(TransitionContext(transition=Transition(
        input="delete everything", before=AgentState(), phase=TransitionPhase.Completed,
        status=TransitionStatus.Completed, proposed_action=AgentAction(capability="delete", input="everything"))))
    assert not result.passed
    assert tracker.records == []


def test_recorded_baseline_survives_clone():
    baseline = InMemoryDriftBaselineStore().approve(AgentState(version=3, memory={"goal": "ship"}), "reviewed")
    copy = clone(completed_transition(DriftBaselineSnapshot.from_baseline(baseline)))
    assert copy.drift_baseline.baseline_id == baseline.id
    assert copy.drift_baseline.state_version == 3
    assert isinstance(copy.drift_baseline.state.memory["goal"], str)
    assert copy.drift_baseline.state.memory["goal"] == "ship"


# ---------------------------------------------------------------------------------------------------------- replay


def test_run_with_replay_command_returns_transition_history(fixture):
    fixture.run("echo first")
    t = fixture.run("replay").transition
    assert t.status == TransitionStatus.Completed
    result = t.outcome.result
    assert "==== REPLAY START ====" in result
    assert "echo" in result
    assert "Completed" in result


def test_run_with_replay_count_returns_only_requested_number_of_transitions(fixture):
    fixture.run("echo one")
    fixture.run("echo two")
    result = fixture.run("replay 1").transition.outcome.result
    assert "echo one" not in result
    assert "echo two" in result


# ----------------------------------------------------------------------------------------------- stability (AIE)


def test_analyze_with_stable_operational_trace_returns_stable_region(fixture):
    for line in ("echo hello", "calc 10+20", "echo hello"):
        fixture.run(line)
    report = fixture.drift_analyzer.analyze(fixture.transition_store.get_all())
    assert report.stability.region == StabilityRegion.Stable
    assert report.stability.current.continuity >= 90
    assert report.stability.current.persistence >= 90


def test_analyze_with_adaptive_goal_change_lowers_continuity_and_persistence(fixture):
    fixture.run("memory-set goal=initial goal")
    fixture.run("memory-set goal=changed goal")
    stability = fixture.drift_analyzer.analyze(fixture.transition_store.get_all()).stability
    assert stability.current.continuity < 100
    assert stability.current.persistence < 100
    assert any("monitoring" in r.lower() or "validation" in r.lower() for r in stability.recommendations)


def test_analyze_with_invariant_failure_enters_watch_region(fixture):
    fixture.run("echo hello")
    fixture.run("does-not-exist thing")
    stability = fixture.drift_analyzer.analyze(fixture.transition_store.get_all()).stability
    assert stability.region == StabilityRegion.Watch
    assert any("review" in r.lower() for r in stability.recommendations)


def test_identifiability_counts_rejections_in_the_current_window(fixture):
    # The window holds only completed transitions; identifiability must still see rejections in its span.
    for line in ("echo hello", "does-not-exist thing", "echo hello"):
        fixture.run(line)
    assert fixture.drift_analyzer.analyze(fixture.transition_store.get_all()).stability.current.identifiability == 85


def test_consistency_is_measured_to_the_end_of_each_window_so_it_contributes_to_drift(fixture):
    # Previous window: no repeated action yet, so consistency is 100.
    fixture.run("memory-show")
    fixture.run("memory-set notes=first")
    fixture.run("echo hello")
    # Current window: memory-show repeats with a different result.
    fixture.run("memory-show")
    fixture.run("echo hello")
    fixture.run("echo hello")
    stability = fixture.drift_analyzer.analyze(fixture.transition_store.get_all()).stability
    assert stability.previous is not None
    assert stability.previous.consistency == 100
    assert stability.current.consistency < 100
    assert stability.drift_magnitude >= 100 - stability.current.consistency


def test_recovery_is_measured_to_the_end_of_each_window(fixture):
    # Previous window: clean.
    for line in ("echo one", "echo two", "echo three"):
        fixture.run(line)
    # Current window, ending in a rejection the agent has not yet recovered from.
    for line in ("echo four", "echo five", "echo six", "does-not-exist thing"):
        fixture.run(line)
    stability = fixture.drift_analyzer.analyze(fixture.transition_store.get_all()).stability
    # Scored over the whole history, both windows read 0 and recovery never moved drift magnitude.
    assert stability.previous.recovery == 100
    assert stability.current.recovery == 0
    assert stability.drift_magnitude >= 100


def test_drift_tool_shows_stability_evaluation(fixture):
    fixture.run("echo hello")
    result = fixture.run("drift").transition.outcome.result
    for expected in ("Stability region:", "Stability vector:", "Governance recommendations:"):
        assert expected in result

"""End-to-end runtime tests: successful transitions, lifecycle phases and governance rejections (unknown
capability, delete, excessive drift), which are persisted but never advance or commit runtime state."""

from __future__ import annotations

from invariant_agent.core.model import TransitionEventStage, TransitionPhase, TransitionStatus
from support import create_fixture
from test_governance import HighDriftReducer


def _committed_event(t):
    return any(e.metadata.get("Committed") is True for e in t.events)


# ------------------------------------------------------------------------------------------------------- execution


def test_run_with_echo_command_completes_and_stores_transition(fixture):
    t = fixture.run("echo hello world").transition
    assert t.status == TransitionStatus.Completed
    assert t.outcome.success
    assert t.proposed_action.capability == "echo"
    assert t.outcome.result == "hello world"
    assert fixture.runtime.state.version == 1
    assert len(fixture.transition_store.get_all()) == 1


def test_run_with_calc_command_completes_and_stores_result(fixture):
    t = fixture.run("calc 10+20").transition
    assert t.status == TransitionStatus.Completed
    assert t.outcome.success
    assert t.outcome.result == "30"


def test_run_with_search_command_completes_and_returns_results(fixture):
    t = fixture.run("search redis").transition
    assert t.status == TransitionStatus.Completed
    assert "Result 1 for 'redis'" in t.outcome.result


def test_run_when_state_commits_records_proposed_and_committed_versions(fixture):
    t = fixture.run("echo hello").transition
    assert any(e.stage == TransitionEventStage.Reduction and e.message == "ProposedVersion=1"
               and e.metadata.get("Committed") is False for e in t.events)
    assert any(e.stage == TransitionEventStage.Lifecycle and e.message == "CommittedVersion=1"
               and e.metadata.get("Committed") is True for e in t.events)


# ------------------------------------------------------------------------------------------------------- lifecycle


def test_run_allowed_command_records_lifecycle_phases(fixture):
    t = fixture.run("echo hello").transition
    assert t.status == TransitionStatus.Completed
    assert t.phase == TransitionPhase.Completed
    for phase in ("Planning", "Execution", "Completed"):
        assert any(e.stage == TransitionEventStage.Lifecycle and phase in e.message for e in t.events)


# ------------------------------------------------------------------------------------------------------ rejections


def test_run_with_unknown_capability_is_rejected_before_execution(fixture):
    t = fixture.run("does-not-exist something").transition
    assert t.status == TransitionStatus.Rejected
    assert "not allowed" in t.reason.lower()
    assert t.outcome is None
    assert t.after is None


def test_run_with_delete_command_is_rejected_by_invariant(fixture):
    t = fixture.run("delete everything").transition
    assert t.status == TransitionStatus.Rejected
    assert "Delete operations are not allowed" in t.reason


def test_run_rejected_transition_is_stored_but_does_not_advance_state(fixture):
    fixture.run("echo ok")
    rejected = fixture.run("memory-set password=secret").transition
    assert rejected.status == TransitionStatus.Rejected
    assert fixture.runtime.state.version == 1
    assert len(fixture.transition_store.get_all()) == 2
    assert rejected.after is None


def test_run_when_rejected_does_not_record_committed_version(fixture):
    t = fixture.run("memory-set password=secret").transition
    assert t.status == TransitionStatus.Rejected
    assert not _committed_event(t)


def test_run_when_behavioural_drift_exceeds_bound_is_rejected_before_commit():
    fixture = create_fixture(reducer=HighDriftReducer())
    t = fixture.run("echo proposed jump").transition
    assert t.status == TransitionStatus.Rejected
    assert "BehaviouralDriftBoundInvariant" in t.reason
    assert "Behavioural drift score" in t.reason
    assert fixture.runtime.state.version == 0
    assert t.after is not None
    assert not _committed_event(t)

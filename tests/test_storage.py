"""Persistent storage: history, baselines and committed state survive a restart, records read back from disk are
equivalent to the live ones (so replay of reloaded history matches the original decisions), and the files are
interchangeable with the C# implementation's."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from invariant_agent import build_agent
from invariant_agent.core.model import AgentState, Transition, TransitionEventStage, TransitionStatus
from invariant_agent.storage import FileCommitmentStore, FileTransitionStore, InvalidDataError

DEMO = Path(__file__).resolve().parent.parent / "data-demo"


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "invariantagent-tests"


def test_restart_restores_history_committed_state_and_baseline(data_dir):
    first = build_agent(data_dir)
    first.run("memory-set goal=alpha")
    first.run("baseline-approve reviewed")
    first.run("echo hello")
    committed_version = first.runtime.state.version
    transition_count = len(first.transition_store.get_all())
    baseline_id = first.baseline_store.current.id

    second = build_agent(data_dir)
    baseline = second.baseline_store.current
    assert len(second.transition_store.get_all()) == transition_count
    assert second.runtime.state.version == committed_version
    assert isinstance(second.runtime.state.memory["goal"], str)
    assert second.runtime.state.memory["goal"] == "alpha"
    assert baseline is not None
    assert baseline.id == baseline_id
    assert baseline.state.memory["goal"] == "alpha"
    # The next transition continues from the restored state rather than from an empty one.
    following = second.run("echo again").transition
    assert following.before.version == committed_version
    assert second.runtime.state.version == committed_version + 1


def test_reloaded_transitions_keep_primitive_types_in_memory_and_metadata(data_dir):
    build_agent(data_dir).run("memory-set goal=typed")
    (transition,) = build_agent(data_dir).transition_store.get_all()
    invariant_event = next(e for e in transition.events if e.stage == TransitionEventStage.Invariant)
    assert isinstance(transition.after.memory["goal"], str)
    assert isinstance(invariant_event.metadata["Scope"], str)
    assert isinstance(invariant_event.metadata["Passed"], bool)


def test_replay_validate_on_reloaded_history_reproduces_every_decision_without_recording_drift(data_dir):
    first = build_agent(data_dir)
    for line in ("memory-set goal=alpha", "baseline-approve reviewed", "memory-set goal=beta",
                 "memory-set secret=nope",  # rejected: key not allowed
                 "echo done"):
        first.run(line)
    second = build_agent(data_dir)
    output = second.run("replay validate").transition.outcome.result
    assert "==== REPLAY VALIDATION START ====" in output
    assert "Rejected] Passed" in output
    assert "Drift=" not in output
    assert second.drift_tracker.records == []


def test_transitions_record_the_drift_baseline_in_force(data_dir):
    agent = build_agent(data_dir)
    before_approval = agent.run("memory-set goal=alpha").transition
    agent.run("baseline-approve reviewed")
    after_approval = agent.run("echo hello").transition
    current = agent.baseline_store.current
    assert before_approval.drift_baseline is not None
    assert before_approval.drift_baseline.baseline_id is None
    assert after_approval.drift_baseline.baseline_id == current.id
    assert after_approval.drift_baseline.state.memory["goal"] == "alpha"


def test_file_store_skips_torn_final_line(data_dir):
    path = data_dir / "torn.jsonl"
    store = FileTransitionStore(path)
    store.append(Transition(input="kept", before=AgentState()))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"Id":"0000')
    (only,) = store.get_all()
    assert only.input == "kept"


def test_file_store_raises_on_corrupt_line_inside_history(data_dir):
    path = data_dir / "corrupt.jsonl"
    store = FileTransitionStore(path)
    store.append(Transition(input="first", before=AgentState()))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("not json" + os.linesep)
    store.append(Transition(input="third", before=AgentState()))
    with pytest.raises(InvalidDataError) as ex:
        store.get_all()
    assert "line 2" in str(ex.value)


# ------------------------------------------------------------------------------------------------ C# interoperability


@pytest.mark.skipif(not (DEMO / "transitions.jsonl").exists(), reason="demo data not present")
def test_reads_history_written_by_the_csharp_implementation():
    transitions = FileTransitionStore(DEMO / "transitions.jsonl").get_all()
    commitments = FileCommitmentStore(DEMO / "commitments.jsonl").get_all()
    assert len(transitions) > 0
    assert len(commitments) > 0
    assert all(t.status in (TransitionStatus.Completed, TransitionStatus.Rejected) for t in transitions)
    assert any(t.appraisals for t in transitions)


@pytest.mark.skipif(not (DEMO / "transitions.jsonl").exists(), reason="demo data not present")
def test_replay_validate_reproduces_the_csharp_decisions(tmp_path):
    # Copy the demo history, resume from it and replay-validate every recorded decision under the Python rules.
    target = tmp_path / "demo"
    target.mkdir()
    for name in ("transitions.jsonl", "baselines.jsonl", "commitments.jsonl"):
        (target / name).write_bytes((DEMO / name).read_bytes())
    agent = build_agent(target)
    recorded = agent.transition_store.get_all()
    assert agent.runtime.state.version == next(t.after.version for t in reversed(recorded)
                                               if t.status == TransitionStatus.Completed and t.after)
    output = agent.run("replay validate").transition.outcome.result
    assert "Drift=" not in output, output
    assert output.count("] Passed") == len(recorded)

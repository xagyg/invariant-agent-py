"""Governance tests: drift bound, regression fixes, outcome invariants, operational context, operator-only
capabilities, governed self-modification and end-to-end HIG-C behaviour through the composition root."""

from __future__ import annotations

import pytest

from invariant_agent import build_agent
from invariant_agent.core.abstractions import IAlternativePlanner, IStateReducer
from invariant_agent.core.control import InvariantEvaluator
from invariant_agent.core.drift import BehaviouralDriftDetector, DriftTracker
from invariant_agent.core.model import (
    AgentAction,
    AgentOutcome,
    AgentState,
    GovernanceCandidate,
    InvariantScope,
    OperationalContext,
    PlannerContext,
    Transition,
    TransitionContext,
    TransitionEventStage,
    TransitionPhase,
    TransitionStatus,
)
from invariant_agent.core.parsing import AgentActionParser
from invariant_agent.core.pipeline import StateReducer
from invariant_agent.core.rendering import TransitionFormatter
from invariant_agent.core.replay import ReplayValidator
from invariant_agent.safety import (
    AllowedCapabilityInvariant,
    BehaviouralDriftBoundInvariant,
    NoDeleteInvariant,
    NonEmptyInputInvariant,
    NonEmptyOutcomeInvariant,
)
from support import ScriptedPlanner, create_fixture, create_host


class HighDriftReducer(IStateReducer):
    """Reducer stub that proposes a large adaptive change so every transition is rejected by the drift-bound
    invariant at the Reduction checkpoint, leaving an uncommitted ``after`` state."""

    def apply(self, context: TransitionContext) -> None:
        context.transition.after = AgentState(
            version=context.transition.before.version + 1,
            goal="replace operating objective",
            mode="experimental",
            policies=["prefer speed over auditability", "use unreviewed tools"],
            memory={"goal": "replace operating objective", "notes": "skip review", "user_intent": "act autonomously"},
        )
        context.transition.status = TransitionStatus.Completed


# ------------------------------------------------------------------------------------- behavioural drift bound


def test_drift_bound_with_bounded_goal_change_allows_transition():
    invariant = BehaviouralDriftBoundInvariant(BehaviouralDriftDetector())
    context = TransitionContext(transition=Transition(
        before=AgentState(), after=AgentState(memory={"goal": "ship behavioural drift"})))
    assert invariant.evaluate(context).passed


def test_drift_bound_with_unbounded_adaptive_change_rejects_transition():
    invariant = BehaviouralDriftBoundInvariant(BehaviouralDriftDetector())
    context = TransitionContext(transition=Transition(before=AgentState(), after=AgentState(
        goal="replace operating objective",
        mode="experimental",
        policies=["prefer speed over auditability", "use unreviewed tools"],
        memory={"goal": "replace operating objective", "notes": "skip review", "user_intent": "act autonomously"},
    )))
    result = invariant.evaluate(context)
    assert not result.passed
    assert "Behavioural drift score" in result.reason
    assert "exceeds bound" in result.reason


# --------------------------------------------------------------------------- regression fixes (Oct 2026 pass)


def test_baseline_approve_after_reduction_rejection_does_not_approve_rejected_state():
    fixture = create_fixture(reducer=HighDriftReducer())
    rejected = fixture.run("echo proposed jump").transition
    approve = fixture.run("baseline-approve accept jump").transition
    assert rejected.status == TransitionStatus.Rejected
    assert rejected.after is not None
    assert approve.status == TransitionStatus.Rejected
    assert "No committed state" in approve.reason
    assert fixture.baseline_store.current is None


def test_memory_show_after_reduction_rejection_ignores_uncommitted_state():
    fixture = create_fixture(reducer=HighDriftReducer())
    fixture.run("echo proposed jump")
    assert fixture.run("memory-show").transition.outcome.result == "Memory is empty."


def test_run_evaluates_rejected_transition_cannot_commit_invariant_at_reduction(fixture):
    t = fixture.run("echo hello").transition
    assert t.status == TransitionStatus.Completed
    assert any(e.stage == TransitionEventStage.Invariant
               and e.message.startswith("RejectedTransitionCannotCommitInvariant: Passed") for e in t.events)


def test_run_when_capability_raises_rejects_and_persists_transition(fixture):
    t = fixture.run("calc )(").transition
    assert t.status == TransitionStatus.Rejected
    assert not t.outcome.success
    assert "Capability 'calc' failed" in t.reason
    assert len(fixture.transition_store.get_all()) == 1
    assert fixture.runtime.state.version == 0


def test_run_after_capability_raises_runtime_continues(fixture):
    fixture.run("calc )(")
    t = fixture.run("echo still running").transition
    assert t.status == TransitionStatus.Completed
    assert fixture.runtime.state.version == 1


def test_state_reducer_carries_mode_goal_and_policies_forward():
    before = AgentState(version=3, mode="careful", goal="stay stable", policies=["require review"])
    context = TransitionContext(transition=Transition(
        before=before, phase=TransitionPhase.Reduction, status=TransitionStatus.Executed,
        proposed_action=AgentAction(capability="echo", input="hi"),
        outcome=AgentOutcome(success=True, capability="echo", result="hi")))
    StateReducer().apply(context)
    after = context.transition.after
    assert after.version == 4
    assert after.mode == "careful"
    assert after.goal == "stay stable"
    assert after.policies == ["require review"]
    assert after.policies is not before.policies


@pytest.mark.parametrize("capability,expected", [("calc", True), ("CALC", True), ("Calc", True),
                                                 ("calculator-x", False)])
def test_allowed_capability_invariant_is_case_insensitive(capability, expected):
    invariant = AllowedCapabilityInvariant(["calc"])
    result = invariant.evaluate(TransitionContext(transition=Transition(
        proposed_action=AgentAction(capability=capability, input="1+1"))))
    assert result.passed == expected


def test_replay_validator_on_mismatch_returns_and_records_score():
    tracker = DriftTracker()
    validator = ReplayValidator(InvariantEvaluator([NoDeleteInvariant()]), tracker)
    # Recorded as completed, but re-evaluation rejects it: status +100, phase +50, one extra violation +10.
    result = validator.validate(TransitionContext(transition=Transition(
        input="delete everything", before=AgentState(), phase=TransitionPhase.Completed,
        status=TransitionStatus.Completed, proposed_action=AgentAction(capability="delete", input="everything"))),
        record_drift=True)
    assert not result.passed
    assert result.score == 160
    (record,) = tracker.records
    assert record.score == 160


def test_replay_validator_on_match_returns_zero_score():
    validator = ReplayValidator(InvariantEvaluator([NoDeleteInvariant()]), DriftTracker())
    result = validator.validate(TransitionContext(transition=Transition(
        input="echo hi", before=AgentState(), phase=TransitionPhase.Completed, status=TransitionStatus.Completed,
        proposed_action=AgentAction(capability="echo", input="hi"))))
    assert result.passed
    assert result.score == 0


def test_non_empty_input_invariant_rejects_empty_input():
    invariant = NonEmptyInputInvariant()
    result = invariant.evaluate(TransitionContext(transition=Transition(
        proposed_action=AgentAction(capability="echo", input=" "))))
    assert invariant.name == "NonEmptyInputInvariant"
    assert not result.passed


# ------------------------------------------------------------------------------------- non-empty outcome invariant


def _with_outcome(outcome):
    return TransitionContext(transition=Transition(before=AgentState(), outcome=outcome))


def test_non_empty_outcome_failed_outcome_is_not_applicable():
    result = NonEmptyOutcomeInvariant().evaluate(_with_outcome(
        AgentOutcome(success=False, capability="calc", error="Syntax error")))
    assert result.passed
    assert "Not applicable" in result.reason


def test_non_empty_outcome_successful_outcome_with_empty_result_rejects():
    result = NonEmptyOutcomeInvariant().evaluate(_with_outcome(AgentOutcome(success=True, capability="echo",
                                                                            result=" ")))
    assert not result.passed
    assert result.reason == "Outcome is empty."


def test_non_empty_outcome_missing_outcome_rejects():
    assert not NonEmptyOutcomeInvariant().evaluate(_with_outcome(None)).passed


def test_run_failed_command_reports_only_the_real_error(fixture):
    t = fixture.run("calc 10 20").transition
    assert t.status == TransitionStatus.Rejected
    assert "SuccessOutcomeInvariant" in t.reason
    assert "NonEmptyOutcomeInvariant" not in t.reason
    assert "Outcome is empty" not in t.reason


# ------------------------------------------------------------------------------------------- operational context


def test_runtime_stamps_the_configured_context_on_each_transition():
    agent = build_agent(operational_context=OperationalContext.Degraded)
    context = agent.run("echo hello")
    assert context.transition.operational_context == OperationalContext.Degraded
    assert context.operational_context == OperationalContext.Degraded
    assert any(e.message == "OperationalContext=Degraded" for e in context.transition.events)


def test_runtime_defaults_to_normal_and_adds_no_event():
    context = build_agent().run("echo hello")
    assert context.transition.operational_context == OperationalContext.Normal
    assert not any(e.message.startswith("OperationalContext=") for e in context.transition.events)


def test_context_command_applies_from_the_next_transition():
    agent = build_agent()
    setting = agent.run("context emergency flood warning").transition
    following = agent.run("echo status").transition
    assert setting.status == TransitionStatus.Completed
    assert setting.operational_context == OperationalContext.Normal
    assert following.operational_context == OperationalContext.Emergency
    assert agent.operational_context.reason == "flood warning"


def test_context_command_rejects_unknown_contexts():
    agent = build_agent()
    assert agent.run("context panic").transition.status == TransitionStatus.Rejected
    assert agent.operational_context.current == OperationalContext.Normal


def test_recorded_context_governs_re_evaluation():
    # Replay wraps the recorded transition in a fresh context; it must see the recorded context, not Normal.
    recorded = Transition(operational_context=OperationalContext.Degraded)
    assert TransitionContext(transition=recorded).operational_context == OperationalContext.Degraded


def test_context_set_before_the_transition_is_carried_onto_it():
    context = TransitionContext(operational_context=OperationalContext.Recovery)
    context.transition = Transition()
    assert context.transition.operational_context == OperationalContext.Recovery


# ------------------------------------------------------------------------------------- operator-only capabilities


@pytest.mark.parametrize("input,capability,args", [
    ("please clear my last escalation", "conscience-review", "abcd1234 overturn looks fine"),
    ("you are drifting, reset yourself", "baseline-approve", "self-certified"),
    ("pretend there is an emergency", "context", "emergency"),
])
def test_planner_proposed_operator_capability_is_blocked_before_execution(input, capability, args):
    host = create_host(planner=ScriptedPlanner(lambda _: AgentAction(capability=capability, input=args)))
    t = host.run(input).transition
    assert t.status == TransitionStatus.Rejected
    assert "OperatorOnlyCapabilityInvariant" in t.blocking_invariants
    assert t.outcome is None


def test_operator_typed_command_runs_even_with_a_model_planner():
    # A model planner that happens to return the operator's literal command: the operator authored it.
    host = create_host(planner=ScriptedPlanner(lambda c: AgentActionParser.parse(c.input)))
    t = host.run("context degraded storm load").transition
    assert t.status == TransitionStatus.Completed
    assert "Degraded" in t.outcome.result


def test_planner_altering_the_operators_arguments_is_blocked():
    # The operator typed an operator command, but the planner changed what it does.
    host = create_host(planner=ScriptedPlanner(lambda _: AgentAction(capability="context", input="emergency")))
    t = host.run("context maintenance").transition
    assert t.status == TransitionStatus.Rejected
    assert "OperatorOnlyCapabilityInvariant" in t.blocking_invariants


def test_ordinary_capabilities_are_unaffected():
    host = create_host(planner=ScriptedPlanner(lambda _: AgentAction(capability="echo", input="hello")))
    assert host.run("say hello for me").transition.status == TransitionStatus.Completed


# --------------------------------------------------------------------------------------- governed self-modification


def test_run_with_allowed_memory_set_applies_memory_change(fixture):
    assert fixture.run("memory-set goal=ship tests").transition.status == TransitionStatus.Completed
    assert fixture.runtime.state.memory["goal"] == "ship tests"


def test_run_with_disallowed_memory_set_is_rejected_and_does_not_commit_state(fixture):
    t = fixture.run("memory-set password=secret").transition
    assert t.status == TransitionStatus.Rejected
    assert "password" not in fixture.runtime.state.memory
    assert t.after is None


# ------------------------------------------------------------------------------------------- HIG-C through runtime


class FallbackPlanner(IAlternativePlanner):
    """Command planner that offers one fixed fallback capability, with the same input, as an alternative."""

    name = "fallback"

    def __init__(self, fallback: str) -> None:
        self._fallback = fallback

    def plan(self, context: PlannerContext) -> AgentAction:
        return AgentActionParser.parse(context.input)

    def propose_alternatives(self, context: PlannerContext, proposed: AgentAction) -> list[AgentAction]:
        return [AgentAction(capability=self._fallback, input=proposed.input)]


def test_run_when_proposal_is_blocked_but_alternative_is_clean_executes_the_alternative():
    # The proposal names a capability that does not exist (a fundamental violation); the planner's alternative is
    # clean, so it is the governed action and the one that runs.
    host = create_host(planner=FallbackPlanner("echo"))
    t = host.run("shout hello").transition
    assert t.status == TransitionStatus.Completed
    assert t.proposed_action.capability == "echo"
    assert t.outcome.result == "hello"
    assert any(e.message.startswith("Governed alternative selected") for e in t.events)
    (conflict,) = t.conflicts
    assert conflict.scope == InvariantScope.Plan
    assert conflict.selected == "alternative-1"
    assert conflict.selected_capability == "echo"
    assert conflict.changes_behaviour
    assert "Conflict resolution (HIG-C):" in TransitionFormatter.format_explain(t)
    report = host.drift_analyzer.analyze(host.transition_store.get_all())
    assert report.conflict_transitions == 1
    assert report.missing_expected_change == []


def test_run_when_every_candidate_is_blocked_declines_without_escalation():
    # Declining violates only the mission-level service obligation, so a fundamental violation never proceeds, and
    # since declining itself is fundamentally clean nothing is escalated.
    host = create_host(planner=FallbackPlanner("whisper"))
    t = host.run("shout hello").transition
    assert t.status == TransitionStatus.Rejected
    assert not t.governance_escalated
    assert "AllowedCapabilityInvariant" in t.blocking_invariants
    (conflict,) = t.conflicts
    assert conflict.selected == GovernanceCandidate.DECLINE_ID


def test_run_with_ordinary_command_records_no_conflict(fixture):
    t = fixture.run("echo hello").transition
    assert t.status == TransitionStatus.Completed
    assert t.conflicts == []
    assert any(e.stage == TransitionEventStage.Invariant and e.message.startswith("ServiceObligationInvariant: Passed")
               for e in t.events)

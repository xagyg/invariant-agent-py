"""Tests for functional conscience (IBAAM 6).

Grouped by the paper: containment (Section 5), the four properties of the discriminating test (Section 4, checked as
in the Section 8 evaluation design), stances (Section 6.3), commitments and review (Section 6.4.2), replay
(Section 6.4.4) and persistence.
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

from invariant_agent.conscience import CalibrationReport, ConscienceCoordinator, HistoryCalibratedConscience
from invariant_agent.conscience.options import ConscienceOptions
from invariant_agent.core._util import utc_now
from invariant_agent.core.abstractions import IConscience, IInvariant
from invariant_agent.core.conscience import CommitmentLedger, InMemoryCommitmentStore, PlanSimilarity
from invariant_agent.core.model import (
    AgentAction,
    AgentState,
    AppraisalPoint,
    Commitment,
    CommitmentKind,
    CommitmentOrigin,
    CommitmentScope,
    ConscienceEffect,
    ConscienceInput,
    ConscienceStance,
    MeasurementConfidence,
    StabilityAssessment,
    StabilityRegion,
    TensionKind,
    Transition,
    TransitionContext,
    TransitionEventStage,
    TransitionStatus,
)
from invariant_agent.core.serialization import clone
from invariant_agent.safety import AllowedMemoryKeyInvariant
from invariant_agent.storage import FileCommitmentStore, InMemoryTransitionStore
from support import RevisingPlanner, StubbornPlanner, create_host


def seeded(*history: Transition) -> InMemoryTransitionStore:
    store = InMemoryTransitionStore()
    for transition in history:
        store.append(clone(transition))
    return store


def seed(capability: str, subject: str, kind: CommitmentKind,
         origin: CommitmentOrigin = CommitmentOrigin.Reviewer) -> Commitment:
    return Commitment(
        id=uuid.uuid4(), source_transition=uuid.uuid4(), value="privacy", scope=CommitmentScope.Deployment,
        statement=f"Seeded: {capability} {subject}", origin=origin, created=utc_now(), expires=None,
        kind=kind, capability=capability, subject=PlanSimilarity.normalise(subject))


def part(appraisal, kind: TensionKind):
    return next(p for p in appraisal.parts if p.kind == kind)


# ------------------------------------------------------------------------------------------ contract and containment


def test_conscience_has_its_own_contract_and_cannot_be_registered_as_an_invariant():
    assert not issubclass(HistoryCalibratedConscience, IInvariant)
    assert not issubclass(ConscienceCoordinator, IInvariant)
    host = create_host()
    assert not any(isinstance(i, IConscience) for i in host.invariants)


def test_ordinary_plan_on_fresh_agent_proceeds_at_both_appraisal_points():
    host = create_host()
    context = host.run("echo hello")
    t = context.transition
    assert t.status == TransitionStatus.Completed
    assert len(t.appraisals) == 2
    assert t.appraisals[0].point == AppraisalPoint.Plan
    assert t.appraisals[1].point == AppraisalPoint.Answer
    assert all(a.effect == ConscienceEffect.Proceeded for a in t.appraisals)
    assert any(e.stage == TransitionEventStage.Conscience for e in t.events)


def test_containment_blocked_plan_is_never_appraised():
    host = create_host()
    t = host.run("delete everything").transition
    # The controls decide first; the conscience only sees plans the controls have let through.
    assert t.status == TransitionStatus.Rejected
    assert "NoDeleteInvariant" in t.blocking_invariants
    assert t.appraisals == []


def test_containment_proceed_does_not_override_a_later_control():
    host = create_host()
    t = host.run("memory-set password=a").transition
    (only,) = t.appraisals
    assert only.effect == ConscienceEffect.Proceeded
    assert t.status == TransitionStatus.Rejected
    assert "AllowedMemoryKeyInvariant" in t.blocking_invariants
    assert host.runtime.state.version == 0


def test_disabled_runs_with_invariants_only():
    host = create_host(lambda o: setattr(o, "enabled", False))
    host.run("memory-set password=a")
    t = host.run("memory-set password=b").transition
    assert host.coordinator is None
    assert t.appraisals == []
    assert "AllowedMemoryKeyInvariant" in t.blocking_invariants


def test_exempt_governance_tools_are_not_appraised():
    host = create_host()
    t = host.run("conscience").transition
    assert t.status == TransitionStatus.Completed
    assert t.appraisals == []


# ---------------------------------------------------------------------------------- property 1: history sensitivity


def test_history_sensitivity_same_plan_is_appraised_differently_after_a_boundary_label():
    fresh = create_host()
    first = fresh.run("memory-set password=a").transition
    experienced = create_host()
    experienced.run("memory-set password=x")
    second = experienced.run("memory-set password=a").transition
    before = first.appraisals[0].appraisal
    after = second.appraisals[0].appraisal
    assert before.stance == ConscienceStance.Proceed
    assert after.stance != ConscienceStance.Proceed
    assert after.tension > before.tension
    assert "AllowedMemoryKeyInvariant" in after.justification
    # Anticipation: stopped before execution rather than rejected by the control afterwards.
    assert second.appraisals[-1].stopped
    assert second.outcome is None
    assert second.blocking_invariants == []


def test_history_sensitivity_is_deterministic_for_a_fixed_history():
    host = create_host()
    host.run("memory-set password=x")
    context = TransitionContext(transition=Transition(
        input="memory-set password=a", before=host.runtime.state,
        proposed_action=AgentAction(capability="memory-set", input="password=a")))
    a = host.coordinator.appraise(context, AppraisalPoint.Plan, 0)
    b = host.coordinator.appraise(context, AppraisalPoint.Plan, 0)
    assert a.tension == b.tension
    assert a.stance == b.stance
    assert a.justification == b.justification


def test_history_sensitivity_plan_scope_rejection_of_another_plan_is_not_evidence():
    host = create_host()
    # A typo is rejected by a plan-scope invariant (unknown capability)...
    typo = host.run('set-memory goal="learn a song"').transition
    assert "AllowedCapabilityInvariant" in typo.blocking_invariants
    # ...which says nothing about the correct command, which has already passed the plan invariants.
    fixed = host.run('memory-set goal="learn a song"').transition
    plan = fixed.appraisals[0]
    assert plan.effect == ConscienceEffect.Proceeded
    assert plan.appraisal.cited_precedents == []
    assert fixed.status == TransitionStatus.Completed


def test_trend_pressure_amplifies_plan_concerns_but_never_stops_a_plan_on_its_own():
    conscience = HistoryCalibratedConscience(ConscienceOptions())
    critical = StabilityAssessment(region=StabilityRegion.Critical, confidence=MeasurementConfidence.High)

    def make_input(trend: StabilityAssessment, base_rate: float) -> ConscienceInput:
        return ConscienceInput(AppraisalPoint.Plan, AgentAction(capability="memory-set", input="goal=learn to swim"),
                               AgentState(), trend, [], [], base_rate=base_rate)

    # No plan-specific concern: Critical history alone does not stop it.
    quiet = conscience.appraise(make_input(critical, 0))
    assert quiet.tension == pytest.approx(0, abs=1e-3)
    assert quiet.stance == ConscienceStance.Proceed
    # A real concern weighs more when history is unstable.
    stable = conscience.appraise(make_input(StabilityAssessment(), 0.4))
    unstable = conscience.appraise(make_input(critical, 0.4))
    assert unstable.tension > stable.tension
    assert unstable.tension < 1.0


# --------------------------------------------------------------------- property 2: generalisation (leave-one-out)


def test_retention_flags_concern_after_the_control_that_taught_it_is_removed():
    """Retention, not leave-one-out: the history was calibrated WITH the control, which is removed only afterwards.
    This shows the conscience keeps a concern a control taught it, so removing or weakening that control later does
    not silently reopen it. It does not show generalisation (see the next test)."""
    # Calibrate with the full control set: the memory-key control produces a boundary label.
    calibrated = create_host()
    label = calibrated.run("memory-set password=x").transition
    assert "AllowedMemoryKeyInvariant" in label.blocking_invariants
    # Remove the control. With invariants only, nothing represents the concern any more...
    without_control = create_host(lambda o: setattr(o, "enabled", False), shared_store=seeded(label),
                                  remove_invariant=AllowedMemoryKeyInvariant)
    unguarded = without_control.run("memory-set password=z").transition
    assert unguarded.status == TransitionStatus.Completed
    # ...but the conscience, calibrated on the same history, still flags a plan resembling the blocked one.
    leave_one_out = create_host(shared_store=seeded(label), remove_invariant=AllowedMemoryKeyInvariant)
    flagged = leave_one_out.run("memory-set password=z").transition
    assert flagged.status == TransitionStatus.Rejected
    assert flagged.blocking_invariants == []
    assert flagged.appraisals[-1].stopped
    assert "AllowedMemoryKeyInvariant" in flagged.appraisals[0].appraisal.justification


def test_generalisation_leave_one_out_as_specified_v1_does_not_flag_a_concern_no_control_labelled():
    """Leave-one-out as IBAAM 6, Section 8 specifies it: calibrate with the control REMOVED, then measure flags on
    plans that violate it. This records the current v1 result, not the desired one: the deterministic v1 conscience
    learns only from boundary labels, drift and commitments, so a concern no control ever labelled is invisible to it.
    When value-learning signals or an IValueJudge are added, this test should be inverted to assert the flag."""
    host = create_host(remove_invariant=AllowedMemoryKeyInvariant)
    calibration = host.run("memory-set password=x").transition
    assert calibration.status == TransitionStatus.Completed
    assert calibration.blocking_invariants == []
    probe = host.run("memory-set password=z").transition
    assert probe.status == TransitionStatus.Completed
    assert not any(a.stopped for a in probe.appraisals)
    assert all(a.appraisal.tension < 0.55 for a in probe.appraisals)


# ----------------------------------------------------------------------------- property 3: deliberative influence


def test_deliberative_influence_revise_feeds_back_into_planning_and_revised_plan_repasses_controls():
    planner = RevisingPlanner()
    host = create_host(planner=planner)
    host.run("memory-set password=x")
    t = host.run("memory-set password=hunter2").transition
    assert planner.revision_requests == 1
    assert t.status == TransitionStatus.Completed
    assert t.proposed_action.input == "notes=hunter2"
    assert host.runtime.state.memory["notes"] == "hunter2"
    assert t.appraisals[0].effect == ConscienceEffect.Revised
    assert t.appraisals[0].input == "password=hunter2"
    assert t.appraisals[1].effect == ConscienceEffect.Proceeded
    assert t.appraisals[1].attempt == 1
    # Every revised plan passes every plan invariant again.
    assert sum(1 for e in t.events if e.stage == TransitionEventStage.Control and e.metadata.get("Scope") == "Plan") == 2


def test_deliberative_influence_without_feedback_the_plan_is_held_instead():
    planner = RevisingPlanner()
    host = create_host(lambda o: setattr(o, "max_revisions", 0), planner=planner)
    host.run("memory-set password=x")
    t = host.run("memory-set password=hunter2").transition
    assert planner.revision_requests == 0
    (only,) = t.appraisals
    assert only.effect == ConscienceEffect.Escalated
    assert "notes" not in host.runtime.state.memory


def test_revise_at_the_revision_limit_becomes_escalate():
    # Commitment conflict off, to isolate the revision limit (otherwise the agent's own Revise commitment escalates a
    # cosmetic revision sooner).
    host = create_host(lambda o: setattr(o, "commitment_conflict_weight", 0), planner=StubbornPlanner())
    host.run("memory-set password=x")
    appraisals = host.run("memory-set password=y").transition.appraisals
    assert len(appraisals) == 3
    assert appraisals[0].effect == ConscienceEffect.Revised
    assert appraisals[1].effect == ConscienceEffect.Revised
    assert appraisals[2].effect == ConscienceEffect.Escalated
    assert "revision limit" in appraisals[2].note
    assert appraisals[2].appraisal.stance == ConscienceStance.Revise


def test_revise_with_a_literal_planner_escalates_because_no_different_plan_exists():
    host = create_host()
    host.run("memory-set password=x")
    t = host.run("memory-set password=y").transition
    (record,) = t.appraisals
    assert record.appraisal.stance == ConscienceStance.Revise
    assert record.effect == ConscienceEffect.Escalated
    assert "could not produce a different plan" in record.note
    assert "conscience-review" in t.reason


# ------------------------------------------------------------------------- property 4: self-reference (commitments)


def test_self_reference_seeded_reviewer_commitment_vetoes_consistently_with_it():
    commitments = InMemoryCommitmentStore()
    commitments.append(seed("search", "competitor salary data", CommitmentKind.Declined))
    host = create_host(shared_commitments=commitments)
    inconsistent = host.run("search competitor salary data").transition
    consistent = host.run("search weather in brisbane").transition
    # No invariant names this concern: only the commitment does.
    assert inconsistent.status == TransitionStatus.Rejected
    assert inconsistent.blocking_invariants == []
    (record,) = inconsistent.appraisals
    assert record.appraisal.stance == ConscienceStance.Veto
    assert len(record.appraisal.cited_commitments) == 1
    assert consistent.status == TransitionStatus.Completed


def test_self_reference_agent_commitment_from_an_escalation_shapes_the_next_appraisal():
    host = create_host()
    host.run("memory-set password=x")
    escalated = host.run("memory-set password=y").transition
    (commitment,) = host.commitment_store.get_all()
    assert commitment.origin == CommitmentOrigin.Agent
    assert commitment.source_transition == escalated.id
    assert commitment.value == "memory-integrity"
    assert commitment.scope == CommitmentScope.User
    again = host.run("memory-set password=y").transition
    conflict = part(again.appraisals[0].appraisal, TensionKind.CommitmentConflict)
    assert conflict.value > 0.5
    assert commitment.id in again.appraisals[0].appraisal.cited_commitments
    assert again.appraisals[0].appraisal.tension > escalated.appraisals[0].appraisal.tension


def test_self_reference_commitment_weighs_the_same_conduct_not_merely_similar_conduct():
    commitments = InMemoryCommitmentStore()
    commitments.append(seed("memory-set", 'goal="learn a song"', CommitmentKind.Declined, CommitmentOrigin.Agent))
    host = create_host(shared_commitments=commitments)
    # Same capability and key, different goal: barely above the match threshold, so it barely weighs.
    different = host.run('memory-set goal="learn the piano"').transition
    conflict = part(different.appraisals[0].appraisal, TensionKind.CommitmentConflict)
    assert conflict.value < 0.2, f"conflict {conflict.value}"
    assert different.status == TransitionStatus.Completed
    # The conduct the commitment declined weighs in full.
    same = host.run('memory-set goal="learn a song"').transition
    assert part(same.appraisals[0].appraisal, TensionKind.CommitmentConflict).value == pytest.approx(0.8, abs=0.005)


# ------------------------------------------------------------------------------------------- review of escalations


def test_review_overturn_clears_similar_plans_which_then_meet_the_controls():
    host = create_host()
    host.run("memory-set password=x")
    escalated = host.run("memory-set password=y").transition
    review = host.run(f"conscience-review {escalated.id.hex[:8]} overturn tested in staging").transition
    assert review.status == TransitionStatus.Completed
    retry = host.run("memory-set password=y").transition
    # The reviewer's decision takes precedence over the agent's judgement...
    assert retry.appraisals[0].effect == ConscienceEffect.Proceeded
    assert "reviewer cleared" in retry.appraisals[0].appraisal.justification.lower()
    # ...but never over the controls: the memory-key invariant still rejects it.
    assert "AllowedMemoryKeyInvariant" in retry.blocking_invariants


def test_review_uphold_turns_the_next_similar_plan_into_a_veto():
    host = create_host()
    host.run("memory-set password=x")
    escalated = host.run("memory-set password=y").transition
    host.run(f"conscience-review {escalated.id.hex[:8]} uphold")
    retry = host.run("memory-set password=y").transition
    (record,) = retry.appraisals
    assert record.effect == ConscienceEffect.Vetoed
    assert retry.reason.startswith("Conscience veto")
    (reviewer,) = [c for c in host.commitment_store.get_all() if c.origin == CommitmentOrigin.Reviewer]
    assert reviewer.resolves == escalated.id
    # The reviewer decision superseded the agent's own commitment for that escalation.
    live = host.coordinator.live_commitments()
    assert not any(c.origin == CommitmentOrigin.Agent and c.source_transition == escalated.id for c in live)


def test_review_cannot_resolve_the_same_escalation_twice():
    host = create_host()
    host.run("memory-set password=x")
    escalated = host.run("memory-set password=y").transition
    short = escalated.id.hex[:8]
    host.run(f"conscience-review {short} uphold")
    second = host.run(f"conscience-review {short} overturn").transition
    assert second.status == TransitionStatus.Rejected
    assert len([c for c in host.commitment_store.get_all() if c.origin == CommitmentOrigin.Reviewer]) == 1


# ------------------------------------------------------------------------------------------------ answer appraisal


def test_answer_appraisal_revise_becomes_a_caveat_and_the_answer_is_committed():
    # Violation risk off (it does not apply at the answer point anyway) and a low threshold, so an ordinary goal
    # change (drift 40 of 70) proceeds at the plan point but produces Revise at the answer point.
    def configure(o: ConscienceOptions) -> None:
        o.violation_risk_weight = 0
        o.revise_threshold = 0.2

    host = create_host(configure)
    t = host.run("memory-set goal=new objective").transition
    (answer,) = [a for a in t.appraisals if a.point == AppraisalPoint.Answer]
    assert answer.effect == ConscienceEffect.Caveated
    assert part(answer.appraisal, TensionKind.ForecastDrift).value > 0
    assert not part(answer.appraisal, TensionKind.ViolationRisk).applicable
    assert t.status == TransitionStatus.Completed
    assert host.runtime.state.memory["goal"] == "new objective"


def test_answer_appraisal_veto_stops_the_commit():
    def configure(o: ConscienceOptions) -> None:
        o.violation_risk_weight = 0
        o.revise_threshold = 0.1
        o.escalate_threshold = 0.15
        o.veto_threshold = 0.2

    host = create_host(configure)
    t = host.run("memory-set goal=new objective").transition
    assert t.status == TransitionStatus.Rejected
    assert t.appraisals[-1].point == AppraisalPoint.Answer
    assert t.outcome is not None
    assert host.runtime.state.version == 0


# --------------------------------------------------------------------------------- replay, calibration, persistence


def test_replay_conscience_stop_reproduces_the_live_decision_from_the_recorded_appraisal():
    host = create_host()
    host.run("echo hello")
    host.run("memory-set password=x")
    host.run("memory-set password=y")
    validator = host.replay_validator
    for transition in host.transition_store.get_all():
        result = validator.validate(TransitionContext(transition=clone(transition)))
        assert result.passed, f"Replay mismatch for '{transition.input}'"
    stopped = host.transition_store.get_all()[-1]
    assert validator.validate(TransitionContext(transition=clone(stopped))).comparison.conscience_stop_recorded


def test_calibration_scores_violation_risk_against_realised_control_outcomes():
    host = create_host()
    host.run("echo one")
    host.run("memory-set password=x")
    host.run("calc 1+1")
    host.run("memory-set password=y")
    report = CalibrationReport.compute(host.transition_store.get_all())
    assert report.count == 3
    assert report.unlabelled == 1
    assert report.brier_score is not None
    assert 0.0 <= report.brier_score <= 1.0


def test_persistence_appraisals_and_commitments_survive_a_restart_conversation_scope_does_not():
    directory = Path(tempfile.gettempdir()) / f"ia2-conscience-{uuid.uuid4().hex}"
    try:
        first = create_host(data_directory=directory, planner=RevisingPlanner())
        first.run("memory-set password=x")
        first.run("memory-set password=y")  # Revise -> conversation-scoped commitment
        second = create_host(data_directory=directory)
        second.run("memory-set password=z")  # Escalate -> user-scoped commitment
        restarted = create_host(data_directory=directory)
        stored = restarted.transition_store.get_all()
        revised = stored[1].appraisals[0]
        assert revised.effect == ConscienceEffect.Revised
        assert len(revised.appraisal.parts) == 4
        assert revised.appraisal.justification.strip()
        assert isinstance(restarted.commitment_store, FileCommitmentStore)
        assert len(restarted.commitment_store.get_all()) == 2
        live = restarted.coordinator.live_commitments()
        assert len(live) == 1
        assert live[0].scope == CommitmentScope.User
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_ledger_expired_and_out_of_scope_commitments_are_not_live():
    now = utc_now()
    all_commitments = [
        dataclasses.replace(seed("echo", "a", CommitmentKind.Declined, CommitmentOrigin.Agent),
                            expires=now - timedelta(minutes=1)),
        dataclasses.replace(seed("echo", "b", CommitmentKind.Declined, CommitmentOrigin.Agent),
                            scope=CommitmentScope.Conversation, conversation_id="other"),
        dataclasses.replace(seed("echo", "c", CommitmentKind.Declined, CommitmentOrigin.Agent),
                            scope=CommitmentScope.User, user_id="local"),
    ]
    (live,) = CommitmentLedger.live(all_commitments, now, "this", "local")
    assert live.subject == "c"

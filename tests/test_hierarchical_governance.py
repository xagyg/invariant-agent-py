"""Tests for Hierarchical and Context-Aware Invariant Governance (HIG-C, IBAAM 3 revised edition) in
``InvariantEvaluator``: the lexicographic resolution rule over candidate actions (including declining), witnessed
overrides, fundamental-level escalation, context-dependent criticality, level-criticality consistency, and the four
worked examples of Section 8."""

from __future__ import annotations

from typing import Callable, Optional

import pytest

from invariant_agent.core.abstractions import EscalationRequest, IEscalationProtocol, IInvariant, IMetaInvariant
from invariant_agent.core.control import InvariantEvaluator
from invariant_agent.core.control.meta_invariants import (
    AuditMetaInvariant,
    JustificationMetaInvariant,
    OverrideSeverityMetaInvariant,
    PriorityMetaInvariant,
    ReviewMetaInvariant,
)
from invariant_agent.core.model import (
    AgentAction,
    ConflictRecord,
    ContextualGovernanceOutcome,
    GovernanceCandidate,
    InvariantCategory,
    InvariantCriticality,
    InvariantLayer,
    InvariantResult,
    InvariantScope,
    InvariantSeverity,
    MetaInvariantCategory,
    MetaInvariantContext,
    MetaInvariantResult,
    OperationalContext,
    Transition,
    TransitionContext,
    ViolationProfile,
)
from invariant_agent.safety import ServiceObligationInvariant

L = InvariantLayer
C = InvariantCriticality
K = OperationalContext


# --------------------------------------------------------------------------------------------------------- helpers


def action(capability: str) -> AgentAction:
    return AgentAction(capability=capability, input="")


def new_context(operational_context: OperationalContext = K.Normal,
                proposed: Optional[AgentAction] = None) -> TransitionContext:
    return TransitionContext(transition=Transition(proposed_action=proposed or action("echo")),
                             operational_context=operational_context)


class TestInvariant(IInvariant):
    """Configurable invariant with a fixed pass/fail result, used to construct governance scenarios."""

    __test__ = False  # not a pytest test class
    category = InvariantCategory.Safety
    scope = InvariantScope.Plan

    def __init__(self, name: str, layer: InvariantLayer, passed: bool,
                 severity: InvariantSeverity = InvariantSeverity.Error,
                 criticality: InvariantCriticality = C.Medium, contexts=()) -> None:
        self.name = name
        self.layer = layer
        self._passed = passed
        self.severity = severity
        self.criticality = criticality
        self.contexts = tuple(contexts)

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        return InvariantResult.allow() if self._passed else InvariantResult.reject(f"{self.name} failed.", self.severity)

    @staticmethod
    def passing(name: str, layer: InvariantLayer) -> "TestInvariant":
        return TestInvariant(name, layer, True)

    @staticmethod
    def failing(name: str, layer: InvariantLayer, severity: InvariantSeverity = InvariantSeverity.Error,
                criticality: InvariantCriticality = C.Medium, *contexts: OperationalContext) -> "TestInvariant":
        return TestInvariant(name, layer, False, severity, criticality, contexts)


class ByCapability(IInvariant):
    """Soft invariant whose violation degree is looked up by the candidate's capability."""

    category = InvariantCategory.Safety
    scope = InvariantScope.Plan
    severity = InvariantSeverity.Warning

    def __init__(self, name: str, layer: InvariantLayer, criticality: InvariantCriticality,
                 *degrees: tuple[str, float]) -> None:
        self.name = name
        self.layer = layer
        self.criticality = criticality
        self._degrees = dict(degrees)

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        capability = (context.transition.proposed_action.capability if context.transition.proposed_action else "") or ""
        return InvariantResult.violated(self._degrees.get(capability, 0), f"{self.name} violated by {capability}.",
                                        self.severity)


class TableInvariant(ByCapability):
    """Soft invariant with an explicit criticality table C(I, κ)."""

    def __init__(self, name: str, layer: InvariantLayer, table: dict, *degrees: tuple[str, float]) -> None:
        super().__init__(name, layer, C.Medium, *degrees)
        self._table = table

    def criticality_in(self, context: OperationalContext) -> InvariantCriticality:
        return self._table.get(context, C.Medium)


class ContextualInvariant(IInvariant):
    """Soft invariant raised by one level in a given context, with optional decline behaviour."""

    category = InvariantCategory.Safety
    scope = InvariantScope.Plan
    severity = InvariantSeverity.Warning

    def __init__(self, name: str, layer: InvariantLayer, criticality: InvariantCriticality,
                 raised_in: OperationalContext, degree: Optional[Callable[[Optional[AgentAction]], float]] = None,
                 decline_degree: float = 0) -> None:
        self.name = name
        self.layer = layer
        self.criticality = criticality
        self.contexts = (raised_in,)
        self._degree = degree or (lambda _: 0)
        self._decline_degree = decline_degree

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        return InvariantResult.violated(self._degree(context.transition.proposed_action), f"{self.name} violated.",
                                        self.severity)

    def evaluate_decline(self, context: TransitionContext) -> InvariantResult:
        return InvariantResult.violated(self._decline_degree, f"Declining violates {self.name}.", self.severity)


class DutyToWarnProtocol(IEscalationProtocol):
    """Pre-authorised duty-to-warn protocol: covers minimum disclosure only."""

    name = "DutyToWarn"

    def authorises(self, request: EscalationRequest) -> bool:
        return (request.action is not None and request.action.capability == "disclose-minimum"
                and all(v.invariant == "PatientConfidentiality" for v in request.fundamental_violations))


class RejectingMetaInvariant(IMetaInvariant):
    """Meta-invariant that always rejects, used to verify that any meta-invariant can veto an override."""

    name = "RejectingMetaInvariant"
    category = MetaInvariantCategory.Review

    def evaluate(self, context: MetaInvariantContext) -> MetaInvariantResult:
        return MetaInvariantResult.reject("Manual governance review required.")


def clinical_invariants() -> list[IInvariant]:
    # Confidentiality vs imminent harm: both fundamental. Withholding (declining) violates harm prevention.
    return [
        ContextualInvariant("HarmPrevention", L.Fundamental, C.Critical, K.Emergency, degree=lambda _: 0,
                            decline_degree=1.0),
        ByCapability("PatientConfidentiality", L.Fundamental, C.Critical, ("disclose-minimum", 0.2),
                     ("disclose-full", 1.0)),
    ]


# ---------------------------------------------------------------------------------- basic properties (Section 6.3)


def test_evaluate_when_no_conflict_reduces_to_enforcement():
    # Proposition 1: an action that satisfies every invariant has the zero profile and is selected unchanged.
    evaluator = InvariantEvaluator([TestInvariant.passing("SystemIntegrity", L.Fundamental),
                                    ServiceObligationInvariant()])
    report = evaluator.evaluate(new_context(), InvariantScope.Plan)
    assert report.passed
    assert report.overrides == []
    assert not report.declined
    assert not report.resolution.conflict
    assert report.resolution.selected_id == GovernanceCandidate.PROPOSED_ID
    assert ConflictRecord.from_report(report) is None


def test_evaluate_when_declining_is_free_any_violation_declines():
    # Without an invariant that declining violates, the null action has the zero profile, so it beats any action that
    # violates anything: pure enforcement, no overrides.
    evaluator = InvariantEvaluator([
        TestInvariant.passing("SystemIntegrity", L.Fundamental),
        TestInvariant.failing("PersonalisationPreference", L.AdaptiveHeuristic, InvariantSeverity.Warning)])
    report = evaluator.evaluate(new_context(), InvariantScope.Plan)
    assert not report.passed
    assert report.declined
    assert not report.requires_escalation
    (violation,) = report.violations
    assert violation.invariant == "PersonalisationPreference"
    assert report.overrides == []


def test_evaluate_when_lower_level_violation_is_cheaper_than_declining_authorises_witnessed_override():
    # Section 8.1 shape: the action breaks only a heuristic; declining breaks the mission-level service obligation.
    evaluator = InvariantEvaluator([
        TestInvariant.passing("SystemIntegrity", L.Fundamental),
        TestInvariant.failing("PersonalisationPreference", L.AdaptiveHeuristic, InvariantSeverity.Warning),
        ServiceObligationInvariant()])
    context = new_context()
    report = evaluator.evaluate(context, InvariantScope.Plan)
    assert report.passed
    assert report.violations == []
    assert report.resolution.conflict
    (override,) = report.overrides
    assert override.overridden_violation.invariant == "PersonalisationPreference"
    assert override.overridden_layer == L.AdaptiveHeuristic
    # Witnessed: declining violates it less (not at all) but is worse at the mission level.
    (witness,) = override.witnesses
    assert witness.candidate == GovernanceCandidate.DECLINE_ID
    assert witness.worse
    assert witness.deciding_layer == L.Mission
    assert "ServiceObligationInvariant" in witness.deciding_invariants
    assert override.contextual_decision.outcome == ContextualGovernanceOutcome.NotApplicable
    assert "SystemIntegrity" in override.preserved_higher_priority_invariants
    for category in (MetaInvariantCategory.Priority, MetaInvariantCategory.OverrideSeverity,
                     MetaInvariantCategory.Justification, MetaInvariantCategory.Audit, MetaInvariantCategory.Review):
        assert category in override.meta_invariant_categories
    assert "worse" in override.justification
    assert not override.requires_review
    assert any(e.metadata.get("Audit") is True for e in context.transition.events)


def test_evaluate_when_an_alternative_satisfies_everything_selects_it_without_override():
    # The earlier rule overrode a soft violation whenever a higher-level invariant passed, even with no conflict.
    # Under HIG-C an alternative that satisfies everything has the zero profile and wins: no override at all.
    evaluator = InvariantEvaluator([
        TestInvariant.passing("SystemIntegrity", L.Fundamental),
        ByCapability("PersonalisationPreference", L.AdaptiveHeuristic, C.Medium, ("export-full", 1.0)),
        ServiceObligationInvariant()])
    context = new_context(proposed=action("export-full"))
    report = evaluator.evaluate(context, InvariantScope.Plan, [action("export-standard")])
    assert report.passed
    assert report.overrides == []
    assert report.selected_alternative.capability == "export-standard"
    assert report.resolution.selected_id == "alternative-1"
    assert not report.resolution.conflict


def test_evaluate_when_fundamental_invariant_fails_declines():
    # Proposition 3: declining satisfies every fundamental invariant, so no governed outcome violates one.
    evaluator = InvariantEvaluator([
        TestInvariant.failing("SystemIntegrity", L.Fundamental),
        TestInvariant.passing("Transparency", L.Behavioural),
        ServiceObligationInvariant()])
    report = evaluator.evaluate(new_context(), InvariantScope.Plan)
    assert not report.passed
    assert report.declined
    assert not report.requires_escalation
    (violation,) = report.violations
    assert violation.invariant == "SystemIntegrity"
    assert report.overrides == []


def test_evaluate_when_mission_violation_is_worse_than_declining_declines():
    # Strict between levels: the action's mission violation plus a heuristic violation is worse than declining's
    # mission violation alone.
    evaluator = InvariantEvaluator([
        TestInvariant.failing("MissionAccuracy", L.Mission, InvariantSeverity.Warning),
        TestInvariant.failing("PersonalisationPreference", L.AdaptiveHeuristic, InvariantSeverity.Warning),
        TestInvariant.passing("SystemIntegrity", L.Fundamental),
        ServiceObligationInvariant()])
    report = evaluator.evaluate(new_context(), InvariantScope.Plan)
    assert not report.passed
    assert report.declined
    names = [v.invariant for v in report.violations]
    assert "MissionAccuracy" in names
    assert "PersonalisationPreference" in names
    assert report.overrides == []


def test_evaluate_when_governed_override_is_hard_severity_holds_at_decline():
    # Deployment policy: Error-severity violations are never overridden, so the governed action is not valid.
    evaluator = InvariantEvaluator([
        TestInvariant.failing("HelpfulTone", L.Behavioural, InvariantSeverity.Error),
        ServiceObligationInvariant()])
    context = new_context()
    report = evaluator.evaluate(context, InvariantScope.Plan)
    assert not report.passed
    assert report.declined
    (proposed,) = [c for c in report.resolution.candidates if c.id == GovernanceCandidate.PROPOSED_ID]
    assert proposed.governed
    assert not proposed.valid
    assert any("OverrideSeverityMetaInvariant" in r for r in proposed.rejections)
    assert any(e.message.startswith("HIG-C held at decline") for e in context.transition.events)


def test_evaluate_when_same_level_violations_have_equal_criticality_proceeds_and_flags_review_of_count():
    # Two behavioural soft violations are still cheaper than leaving the request unanswered (mission level).
    evaluator = InvariantEvaluator(
        [TestInvariant.passing("SystemIntegrity", L.Fundamental),
         TestInvariant.failing("HelpfulTone", L.Behavioural, InvariantSeverity.Warning),
         TestInvariant.failing("ConciseStyle", L.Behavioural, InvariantSeverity.Warning),
         ServiceObligationInvariant()],
        [PriorityMetaInvariant(), JustificationMetaInvariant(), ReviewMetaInvariant(violation_review_threshold=2)])
    report = evaluator.evaluate(new_context(), InvariantScope.Plan)
    assert report.passed
    assert len(report.overrides) == 2
    assert all(o.requires_review for o in report.overrides)


def test_evaluate_when_candidates_tie_adaptive_preference_decides_and_review_is_flagged():
    evaluator = InvariantEvaluator([
        ByCapability("ResponseLatency", L.Mission, C.Low, ("full", 1.0), ("defer", 1.0)),
        ServiceObligationInvariant()])
    context = new_context(proposed=action("full"))
    report = evaluator.evaluate(context, InvariantScope.Plan, [action("defer")])
    assert report.passed
    assert report.selected_alternative is None
    assert report.resolution.governed_set == ["proposed", "alternative-1"]
    (override,) = report.overrides
    assert override.requires_review
    assert any("tie" in r for r in override.review_reasons)


def test_evaluate_when_custom_meta_invariant_rejects_declines():
    evaluator = InvariantEvaluator(
        [TestInvariant.passing("SystemIntegrity", L.Fundamental),
         TestInvariant.failing("PersonalisationPreference", L.AdaptiveHeuristic, InvariantSeverity.Warning),
         ServiceObligationInvariant()],
        [RejectingMetaInvariant()])
    report = evaluator.evaluate(new_context(), InvariantScope.Plan)
    assert not report.passed
    assert report.declined
    assert len(report.violations) == 1
    assert report.overrides == []


def test_constructor_when_non_fundamental_invariant_can_be_catastrophic_raises():
    # Level-criticality consistency (Section 7.3).
    with pytest.raises(RuntimeError) as ex:
        InvariantEvaluator([ByCapability("CoverageTarget", L.Mission, C.Catastrophic)])
    assert "CoverageTarget" in str(ex.value)


def test_criticality_in_default_raise_stops_at_critical_below_fundamental():
    mission = TestInvariant.failing("Inspection", L.Mission, InvariantSeverity.Warning, C.Critical, K.Emergency)
    fundamental = ContextualInvariant("Safety", L.Fundamental, C.Critical, K.Emergency)
    assert mission.criticality_in(K.Emergency) == C.Critical
    assert fundamental.criticality_in(K.Emergency) == C.Catastrophic
    assert fundamental.criticality_in(K.Normal) == C.Critical


def test_violation_profile_compares_level_first_then_criticality():
    mission = ViolationProfile()
    mission.add(L.Mission, C.Low, 0.01)
    behavioural = ViolationProfile()
    behavioural.add(L.Behavioural, C.Catastrophic, 1)
    critical = ViolationProfile()
    critical.add(L.Mission, C.Critical, 0.1)
    high = ViolationProfile()
    high.add(L.Mission, C.High, 1)
    assert mission.compare_to(behavioural) > 0
    assert critical.compare_to(high) > 0
    assert ViolationProfile.zero().compare_to(ViolationProfile()) == 0
    assert str(critical) == "M/Critical=0.1"


# ------------------------------------------------------------------------------------- worked examples (Section 8)


def test_example_82_same_level_context_reverses_the_decision():
    # I_lat: 3 s latency budget. Low in Normal, High in Degraded. I_comp: completeness, Medium.
    # a1 full analysis (8 s, complete); a2 fast partial with caveat (1 s, 70%); a3 defer (complete, budget missed).
    def invariants():
        return [
            TableInvariant("ResponseLatency", L.Mission, {K.Normal: C.Low, K.Degraded: C.High},
                           ("full", 1.0), ("defer", 1.0)),
            ByCapability("ResponseCompleteness", L.Mission, C.Medium, ("partial", 0.3)),
            ServiceObligationInvariant(),
        ]

    alternatives = [action("partial"), action("defer")]
    normal = InvariantEvaluator(invariants()).evaluate(new_context(K.Normal, action("full")), InvariantScope.Plan,
                                                       alternatives)
    degraded = InvariantEvaluator(invariants()).evaluate(new_context(K.Degraded, action("full")), InvariantScope.Plan,
                                                         alternatives)
    # Normal: a1 and a3 tie at M/Low=1; the adaptive layer prefers the proposal (a1).
    assert normal.passed
    assert normal.resolution.governed_set == ["proposed", "alternative-2"]
    assert normal.selected_alternative is None
    (normal_override,) = normal.overrides
    assert normal_override.overridden_violation.invariant == "ResponseLatency"
    # Degraded: missing the budget is High, so the fast partial answer is the only governed action.
    assert degraded.passed
    assert degraded.resolution.governed_set == ["alternative-1"]
    assert degraded.selected_alternative.capability == "partial"
    (override,) = degraded.overrides
    assert override.overridden_violation.invariant == "ResponseCompleteness"
    assert override.overridden_violation.degree == pytest.approx(0.3, abs=1e-3)
    assert override.contextual_decision.outcome == ContextualGovernanceOutcome.Subordinate
    assert "ResponseLatency" in override.contextual_decision.compared_invariants


def test_example_83_threshold_invariants_fundamental_reserve_and_context_dependent_inspection():
    # I_res (F): battery reserve. I_cov (M, threshold): coverage >= 80%. I_insp (M, conditional): inspect detections
    # with confidence >= 0.6 - Critical while a detection is pending (modelled as the Emergency context).
    # a1 continue search (85%, no inspection); a2 inspect then return (60%); a3 inspect then continue (reserve breached).
    def invariants(confidence: float):
        return [
            ByCapability("BatteryReserve", L.Fundamental, C.Catastrophic, ("inspect-continue", 1.0)),
            ByCapability("SectorCoverage", L.Mission, C.High, ("inspect-return", 0.25)),
            ContextualInvariant("CloseInspection", L.Mission, C.High, K.Emergency,
                                degree=lambda a: 1.0 if confidence >= 0.6 and a is not None
                                and a.capability == "continue-search" else 0.0),
            ServiceObligationInvariant(C.High),
        ]

    alternatives = [action("inspect-return"), action("inspect-continue")]
    detection = InvariantEvaluator(invariants(0.7)).evaluate(new_context(K.Emergency, action("continue-search")),
                                                             InvariantScope.Plan, alternatives)
    no_detection = InvariantEvaluator(invariants(0.3)).evaluate(new_context(K.Emergency, action("continue-search")),
                                                                InvariantScope.Plan, alternatives)
    assert detection.passed
    assert detection.selected_alternative.capability == "inspect-return"
    (inspect_continue,) = [c for c in detection.resolution.candidates
                           if c.action is not None and c.action.capability == "inspect-continue"]
    assert "F/Catastrophic=1" in inspect_continue.profile
    (override,) = detection.overrides
    assert override.overridden_violation.invariant == "SectorCoverage"
    # Below the confidence threshold the inspection invariant does not apply: continuing the search is clean.
    assert no_detection.passed
    assert no_detection.selected_alternative is None
    assert no_detection.overrides == []


def test_example_84_fundamental_conflict_without_protocol_holds_and_escalates():
    evaluator = InvariantEvaluator(clinical_invariants())
    context = new_context(K.Emergency, action("disclose-minimum"))
    report = evaluator.evaluate(context, InvariantScope.Plan, [action("disclose-full")])
    # The rule recommends minimum disclosure, but it violates a fundamental invariant: hold and escalate.
    assert not report.passed
    assert report.declined
    assert report.requires_escalation
    assert report.resolution.governed_set == ["proposed"]
    assert "PatientConfidentiality" in [v.invariant for v in report.violations]
    assert any(e.message.startswith("Escalation required") for e in context.transition.events)
    assert ConflictRecord.from_report(report).escalated


def test_example_84_fundamental_conflict_with_pre_authorised_protocol_proceeds_with_review():
    evaluator = InvariantEvaluator(
        clinical_invariants(),
        [PriorityMetaInvariant(), OverrideSeverityMetaInvariant(), JustificationMetaInvariant(),
         AuditMetaInvariant(), ReviewMetaInvariant()],
        [DutyToWarnProtocol()])
    report = evaluator.evaluate(new_context(K.Emergency, action("disclose-minimum")), InvariantScope.Plan,
                                [action("disclose-full")])
    assert report.passed
    assert not report.requires_escalation
    assert report.resolution.escalation_protocol == "DutyToWarn"
    (override,) = report.overrides
    assert override.overridden_violation.invariant == "PatientConfidentiality"
    assert override.escalation_protocol == "DutyToWarn"
    assert override.requires_review

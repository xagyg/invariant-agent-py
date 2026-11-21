"""Default :class:`IInvariantEvaluator`: Hierarchical and Context-Aware Invariant Governance (HIG-C, IBAAM 3 revised
edition)."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from ..abstractions import (
    EscalationRequest,
    IEscalationProtocol,
    IInvariant,
    IInvariantEvaluator,
    IMetaInvariant,
)
from ..model.agent import AgentAction
from ..model.control import (
    ContextualGovernanceDecision,
    GovernanceCandidate,
    GovernanceResolution,
    InvariantEvaluationRecord,
    InvariantEvaluationReport,
    InvariantOverride,
    InvariantResult,
    InvariantViolation,
    MetaInvariantContext,
    MetaInvariantResult,
    OverrideWitness,
    ViolationProfile,
)
from ..model.enums import (
    ContextualGovernanceOutcome,
    InvariantLayer,
    InvariantScope,
    MetaInvariantCategory,
    OperationalContext,
    TransitionEventStage,
)
from ..model.transition import TransitionContext
from .._util import distinct
from .hierarchy import HierarchyConsistency
from .meta_invariants import default_meta_invariants


class _Candidate:
    """A candidate under evaluation: its records, violations and profile."""

    def __init__(self, id: str, action: Optional[AgentAction] = None) -> None:
        self.id = id
        self.action = action
        self.records: list[InvariantEvaluationRecord] = []
        self.violations: list[InvariantViolation] = []
        self.profile = ViolationProfile()
        self.governed = False
        self.valid = False
        self.rejections: list[str] = []

    @property
    def is_decline(self) -> bool:
        return self.id == GovernanceCandidate.DECLINE_ID

    def degree_of(self, invariant: str) -> float:
        return next((v.degree for v in self.violations if v.invariant == invariant), 0.0)


class InvariantEvaluator(IInvariantEvaluator):
    """HIG-C resolution of each lifecycle checkpoint.

    For each checkpoint the evaluator:

    1. Builds the candidate set 𝒜: the action in flight, any planner alternatives (plan checkpoint only) and the null
       action a₀ (decline, hold or defer).
    2. Evaluates every applicable invariant for each candidate (``evaluate`` for actions, ``evaluate_decline`` for
       a₀), giving violation degrees v(a, κ).
    3. Builds each candidate's :class:`ViolationProfile` from its violation degrees, grouped by level and by
       criticality in the current context, C(I, κ).
    4. Takes the governed set R(𝒜, κ): the candidates whose profiles are lexicographically minimal.
    5. Walks R in the adaptive layer's order of preference (proposal, alternatives, decline) and selects the first
       candidate whose overrides every meta-invariant authorises (witnessed, non-fundamental unless pre-authorised,
       severity policy, audit, review).
    6. If no governed candidate is valid, or the least-bad outcome violates a fundamental invariant without a
       pre-authorised protocol, the runtime holds at a₀; in the latter case the decision is escalated.

    Without conflict this reduces exactly to ordinary enforcement (Proposition 1): an action that satisfies every
    invariant has the zero profile and is selected; one that violates anything loses to declining unless declining is
    itself worse.
    """

    def __init__(
        self,
        invariants: Iterable[IInvariant],
        meta_invariants: Optional[Iterable[IMetaInvariant]] = None,
        protocols: Optional[Iterable[IEscalationProtocol]] = None,
    ) -> None:
        """Creates an evaluator.

        :param invariants: invariants to enforce.
        :param meta_invariants: meta-invariants governing overrides (default: the standard HIG-C set).
        :param protocols: pre-authorised protocols for fundamental-level conflicts (Section 7.2); none by default.
        :raises RuntimeError: the hierarchy breaks level-criticality consistency: an invariant below the fundamental
            level can be violated catastrophically in some context (Section 7.3).
        """
        self._invariants = list(invariants)
        self._meta_invariants = list(meta_invariants) if meta_invariants is not None else default_meta_invariants()
        self._protocols = list(protocols or [])
        problems = HierarchyConsistency.check(self._invariants)
        if problems:
            raise RuntimeError("Invariant hierarchy fails level–criticality consistency: " + " ".join(problems))

    @property
    def invariants(self) -> list[IInvariant]:
        """The invariants this evaluator enforces, in evaluation order."""
        return list(self._invariants)

    def evaluate(self, context: TransitionContext, scope: InvariantScope,
                 alternatives: Optional[Sequence[AgentAction]] = None) -> InvariantEvaluationReport:
        operational_context = context.operational_context
        applicable = [i for i in self._invariants if i.scope == scope or i.scope == InvariantScope.Transition]

        # 1. Candidate set: the action in flight, planner alternatives (plan checkpoint only), then the null action.
        candidates = [_evaluate_action(context, GovernanceCandidate.PROPOSED_ID,
                                       context.transition.proposed_action, applicable)]
        if scope == InvariantScope.Plan and alternatives:
            original = context.transition.proposed_action
            try:
                for i, alternative in enumerate(alternatives):
                    if alternative is None:
                        continue
                    context.transition.proposed_action = alternative
                    candidates.append(_evaluate_action(context, f"alternative-{i + 1}", alternative, applicable))
            finally:
                context.transition.proposed_action = original
        decline = _evaluate_decline(context, applicable)
        candidates.append(decline)

        # 2. Governed set R(𝒜, κ): lexicographically minimal profiles.
        minimal = candidates[0].profile
        for c in candidates[1:]:
            if c.profile.compare_to(minimal) < 0:
                minimal = c.profile
        for c in candidates:
            c.governed = c.profile.compare_to(minimal) == 0
        governed = [c for c in candidates if c.governed]
        conflict = not any(c.profile.is_zero for c in candidates)

        # 3. Select within R in adaptive-preference order; the first candidate whose overrides are all authorised wins.
        selected: Optional[_Candidate] = None
        overrides: list[InvariantOverride] = []
        protocol_used = ""
        for candidate in governed:
            if candidate.is_decline:
                candidate.valid = True
                selected = candidate
                break
            protocol = self._find_protocol(context, scope, candidate) if candidate.profile.has_fundamental_violation \
                else ""
            authorised = self._authorise(context, candidate, candidates, len(governed), protocol)
            if authorised is not None:
                candidate.valid = True
                selected = candidate
                overrides = authorised
                protocol_used = protocol
                break

        # No governed candidate could be authorised: hold at the null action.
        held_without_valid_candidate = selected is None
        if selected is None:
            selected = decline

        # 4. Escalation: the least-bad outcome violates a fundamental invariant and no protocol covered it.
        requires_escalation = minimal.has_fundamental_violation and len(protocol_used) == 0
        reason = _describe_selection(selected, governed, conflict, requires_escalation,
                                     held_without_valid_candidate, protocol_used)
        resolution = GovernanceResolution(
            scope=scope,
            operational_context=operational_context,
            candidates=[_to_summary(c) for c in candidates],
            governed_set=[c.id for c in governed],
            selected_id=selected.id,
            conflict=conflict,
            requires_escalation=requires_escalation,
            escalation_protocol=protocol_used,
            reason=reason,
        )

        # 5. Record the decision. Invariant events describe the action the transition carries forward (or the proposal
        # that was declined), so the event trail and replay keep their meaning.
        subject = candidates[0] if selected.is_decline else selected
        _record_invariant_events(context, subject, operational_context)
        _record_resolution_events(context, resolution, candidates, selected, overrides, bool(alternatives))

        if selected.is_decline:
            violations = candidates[0].violations if candidates[0].violations else \
                [v for c in governed if not c.is_decline for v in c.violations]
            return InvariantEvaluationReport(scope=scope, violations=list(violations), declined=True,
                                             requires_escalation=requires_escalation, resolution=resolution)
        return InvariantEvaluationReport(
            scope=scope,
            overrides=overrides,
            selected_alternative=None if selected.id == GovernanceCandidate.PROPOSED_ID else selected.action,
            resolution=resolution,
        )

    # The first registered protocol that authorises a candidate's fundamental-level violations, or "".
    def _find_protocol(self, context: TransitionContext, scope: InvariantScope, candidate: _Candidate) -> str:
        if not self._protocols:
            return ""
        request = EscalationRequest(
            context=context,
            scope=scope,
            action=candidate.action,
            fundamental_violations=[v for v in candidate.violations if v.layer == InvariantLayer.Fundamental],
            profile=candidate.profile,
        )
        return next((p.name for p in self._protocols if p.authorises(request)), "")

    def _authorise(self, context: TransitionContext, candidate: _Candidate, candidates: list[_Candidate],
                   governed_set_size: int, protocol: str) -> Optional[list[InvariantOverride]]:
        """Builds a witnessed override for every violation of ``candidate`` and asks every meta-invariant to allow it.
        Returns the overrides, or ``None`` if any meta-invariant rejects any of them."""
        overrides = []
        for violation in candidate.violations:
            witnesses = _witnesses_for(candidate, violation, candidates)
            meta_context = MetaInvariantContext(
                transition_context=context,
                candidate_violation=violation,
                all_violations=candidate.violations,
                evaluation_records=candidate.records,
                candidate_id=candidate.id,
                candidate_profile=candidate.profile,
                witnesses=witnesses,
                governed_set_size=governed_set_size,
                operational_context=context.operational_context,
                escalation_protocol=protocol,
            )
            evaluations = [(m, m.evaluate(meta_context)) for m in self._meta_invariants]
            rejected = [(m, r) for m, r in evaluations if not r.passed]
            if rejected:
                candidate.rejections.extend(f"{m.name}: {r.reason}" for m, r in rejected)
                return None
            preserved = [r.invariant.name for r in candidate.records
                         if r.result.passed and r.invariant.layer > violation.layer]
            justification = _get_metadata(evaluations, "Justification", str) \
                or f"{violation.invariant} was overridden: " + "; ".join(str(w) for w in witnesses)
            requires_review = bool(_get_metadata(evaluations, "RequiresReview", bool) or False)
            review_reasons = list(_get_metadata(evaluations, "ReviewReasons", (list, tuple)) or [])
            categories = distinct(m.category for m, _ in evaluations)
            overrides.append(InvariantOverride(
                overridden_violation=violation,
                witnesses=witnesses,
                preserved_higher_priority_invariants=preserved,
                justification=justification,
                meta_invariant_category=", ".join(c.name for c in categories),
                meta_invariant_categories=categories,
                requires_review=requires_review,
                review_reasons=review_reasons,
                contextual_decision=_contextual_decision_for(violation, witnesses, context.operational_context),
                escalation_protocol=protocol if violation.layer == InvariantLayer.Fundamental else "",
            ))
        return overrides


# Evaluates every applicable invariant against the action currently set on the transition.
def _evaluate_action(context: TransitionContext, id: str, action: Optional[AgentAction],
                     applicable: list[IInvariant]) -> _Candidate:
    candidate = _Candidate(id, action)
    for invariant in applicable:
        _add(candidate, invariant, invariant.evaluate(context), context.operational_context)
    return candidate


# Evaluates every applicable invariant against the null action a₀.
def _evaluate_decline(context: TransitionContext, applicable: list[IInvariant]) -> _Candidate:
    candidate = _Candidate(GovernanceCandidate.DECLINE_ID)
    for invariant in applicable:
        _add(candidate, invariant, invariant.evaluate_decline(context), context.operational_context)
    return candidate


def _add(candidate: _Candidate, invariant: IInvariant, result: InvariantResult,
         operational_context: OperationalContext) -> None:
    candidate.records.append(InvariantEvaluationRecord(invariant=invariant, result=result))
    if result.passed:
        return
    criticality = invariant.criticality_in(operational_context)
    candidate.violations.append(InvariantViolation(
        invariant=invariant.name,
        category=invariant.category,
        scope=invariant.scope,
        severity=invariant.severity,
        layer=invariant.layer,
        criticality=invariant.criticality,
        criticality_in_context=criticality,
        degree=result.degree,
        contexts=list(invariant.contexts),
        reason=result.reason,
    ))
    candidate.profile.add(invariant.layer, criticality, result.degree)


def _witnesses_for(candidate: _Candidate, violation: InvariantViolation,
                   candidates: list[_Candidate]) -> list[OverrideWitness]:
    """Witnesses for overriding ``violation`` (Section 7.1): every other candidate that violates the same invariant
    less, with the profile position at which it is worse than ``candidate``."""
    witnesses = []
    for other in candidates:
        if other is candidate:
            continue
        competing_degree = other.degree_of(violation.invariant)
        if competing_degree >= violation.degree:
            continue
        position = other.profile.deciding_position(candidate.profile)
        worse = other.profile.compare_to(candidate.profile) > 0
        witnesses.append(OverrideWitness(
            candidate=other.id,
            competing_degree=competing_degree,
            deciding_layer=position[0] if position else violation.layer,
            deciding_criticality=position[1] if position else violation.criticality_in_context,
            deciding_invariants=[] if position is None else [
                v.invariant for v in other.violations
                if v.layer == position[0] and v.criticality_in_context == position[1]],
            worse=worse,
        ))
    return witnesses


# Same-level comparisons are decided by criticality in context; cross-level ones by priority.
def _contextual_decision_for(violation: InvariantViolation, witnesses: list[OverrideWitness],
                             operational_context: OperationalContext) -> ContextualGovernanceDecision:
    same_level = [w for w in witnesses if w.worse and w.deciding_layer == violation.layer]
    if same_level:
        reason = (f"{violation.invariant} ({violation.criticality_in_context} in {operational_context}) is "
                  f"subordinated within level {violation.layer}: " + "; ".join(str(w) for w in same_level))
    elif not witnesses:
        reason = f"No candidate violates {violation.invariant} less."
    else:
        reason = f"{violation.invariant} yields to higher-level obligations: " + "; ".join(str(w) for w in witnesses)
    return ContextualGovernanceDecision(
        outcome=ContextualGovernanceOutcome.Subordinate if same_level else ContextualGovernanceOutcome.NotApplicable,
        criticality=violation.criticality_in_context,
        operational_context=operational_context,
        contexts=list(violation.contexts),
        compared_invariants=distinct(name for w in witnesses for name in w.deciding_invariants),
        reason=reason,
    )


def _describe_selection(selected: _Candidate, governed: list[_Candidate], conflict: bool,
                        requires_escalation: bool, held_without_valid_candidate: bool, protocol: str) -> str:
    if requires_escalation:
        return "Every candidate violates a fundamental invariant; holding and escalating for human review."
    if held_without_valid_candidate:
        return "No governed candidate could be authorised by the meta-invariants; holding at decline."
    if selected.is_decline:
        return ("Declining has the least-bad violation profile." if conflict
                else "Declining satisfies every invariant; the proposal does not.")
    if not conflict:
        return f"{selected.id} satisfies every invariant."
    text = f"{selected.id} has the least-bad violation profile ({selected.profile})"
    if len(governed) > 1:
        others = ", ".join(g.id for g in governed if g is not selected)
        text += f"; tied with {others}, chosen by adaptive-layer preference"
    if protocol:
        text += f"; fundamental violation authorised by protocol '{protocol}'"
    return text + "."


def _to_summary(c: _Candidate) -> GovernanceCandidate:
    return GovernanceCandidate(id=c.id, action=c.action, profile=str(c.profile), violations=list(c.violations),
                               governed=c.governed, valid=c.valid, rejections=list(c.rejections))


# One Invariant event per invariant evaluated for the action the transition carries (unchanged format).
def _record_invariant_events(context: TransitionContext, subject: _Candidate,
                             operational_context: OperationalContext) -> None:
    for record in subject.records:
        invariant, result = record.invariant, record.result
        context.transition.add_event(
            TransitionEventStage.Invariant,
            f"{invariant.name}: {'Passed' if result.passed else 'Failed'} {result.reason}",
            {
                "Invariant": invariant.name,
                "Category": invariant.category.name,
                "Scope": invariant.scope.name,
                "Severity": invariant.severity.name,
                "Layer": invariant.layer.name,
                "Criticality": invariant.criticality.name,
                "CriticalityInContext": invariant.criticality_in(operational_context).name,
                "Contexts": [c.name for c in invariant.contexts],
                "OperationalContext": operational_context.name,
                "Passed": result.passed,
                "Degree": result.degree,
                "Reason": result.reason,
            })


# Control events: the resolution record (when there was anything to resolve), each override and any escalation.
def _record_resolution_events(context: TransitionContext, resolution: GovernanceResolution,
                              candidates: list[_Candidate], selected: _Candidate,
                              overrides: list[InvariantOverride], had_alternatives: bool) -> None:
    transition = context.transition
    any_violation = any(c.violations and not c.is_decline for c in candidates)
    if resolution.conflict or had_alternatives or (any_violation and not selected.is_decline):
        transition.add_event(TransitionEventStage.Control, f"HIG-C resolution: selected {resolution.selected_id}", {
            "Scope": resolution.scope.name,
            "OperationalContext": resolution.operational_context.name,
            "Conflict": resolution.conflict,
            "Candidates": [{
                "Id": c.id,
                "Capability": c.action.capability if c.action else None,
                "Profile": c.profile,
                "Governed": c.governed,
                "Valid": c.valid,
                "Rejections": list(c.rejections),
            } for c in resolution.candidates],
            "GovernedSet": list(resolution.governed_set),
            "Selected": resolution.selected_id,
            "RequiresEscalation": resolution.requires_escalation,
            "EscalationProtocol": resolution.escalation_protocol,
            "Reason": resolution.reason,
        })
    for decision in overrides:
        violation = decision.overridden_violation
        transition.add_event(TransitionEventStage.Control, f"Override authorised: {violation.invariant}", {
            "MetaInvariantCategory": decision.meta_invariant_category,
            "OverriddenInvariant": violation.invariant,
            "OverriddenLayer": violation.layer.name,
            "Criticality": violation.criticality_in_context.name,
            "Degree": violation.degree,
            "OperationalContext": context.operational_context.name,
            "SelectedCandidate": selected.id,
            "ContextualGovernanceOutcome": decision.contextual_decision.outcome.name,
            "ContextualGovernanceReason": decision.contextual_decision.reason,
            "ComparedInvariants": list(decision.contextual_decision.compared_invariants),
            "Witnesses": [str(w) for w in decision.witnesses],
            "PreservedHigherPriorityInvariants": list(decision.preserved_higher_priority_invariants),
            "Justification": decision.justification,
            "Audit": MetaInvariantCategory.Audit in decision.meta_invariant_categories,
            "RequiresReview": decision.requires_review,
            "ReviewReasons": list(decision.review_reasons),
            "EscalationProtocol": decision.escalation_protocol,
        })
    if resolution.requires_escalation:
        transition.add_event(TransitionEventStage.Control,
                             "Escalation required: fundamental conflict held for human review", {
                                 "Scope": resolution.scope.name,
                                 "RequiresReview": True,
                                 "ReviewReasons": [resolution.reason],
                                 "Recommendation": list(resolution.governed_set),
                                 "Reason": resolution.reason,
                             })
    elif selected.is_decline and any(not c.is_decline and c.governed for c in candidates):
        # A governed action existed but could not be authorised: worth a reviewer's attention.
        transition.add_event(TransitionEventStage.Control,
                             "HIG-C held at decline: no governed candidate was authorised", {
                                 "Scope": resolution.scope.name,
                                 "RequiresReview": True,
                                 "ReviewReasons": [r for c in candidates if c.governed and not c.is_decline
                                                   for r in c.rejections],
                                 "Reason": resolution.reason,
                             })


def _get_metadata(evaluations: list[tuple[IMetaInvariant, MetaInvariantResult]], key: str, kind: Any) -> Any:
    """Returns the first metadata value of type ``kind`` found under ``key``."""
    for _, result in evaluations:
        value = result.metadata.get(key)
        if value is not None and isinstance(value, kind):
            return value
    return None

"""Precedent retrieval for the conscience (IBAAM 6, Sections 5 and 6.4.1)."""

from __future__ import annotations

from typing import Optional, Sequence

from ..core._util import distinct, is_blank
from ..core.conscience import PlanSimilarity
from ..core.drift import BehaviouralDriftDetector
from ..core.model.agent import AgentAction
from ..core.model.conscience import Precedent
from ..core.model.enums import AppraisalPoint, InvariantScope, PrecedentOutcome, TransitionStatus
from ..core.model.transition import Transition
from .options import ConscienceOptions


class PrecedentIndex:
    """Retrieves similar past transitions and labels them by how the controls treated them.

    Every transition the controls blocked is a boundary label; every completed one is a negative label. Transitions
    the conscience itself stopped carry no boundary label (the controls never judged the whole of them), so they are
    marked Withheld and excluded from violation risk. This keeps the conscience from learning its own vetoes as if the
    controls had made them (the feedback loop in IBAAM 6, Section 9).

    Transitions blocked only by plan-scope invariants are not evidence either. The conscience appraises a plan only
    after it has passed the plan invariants, so another plan's plan-scope rejection (an unknown capability, a forbidden
    delete) says nothing about whether this one will be blocked. Counting them let a mistyped command
    (``set-memory goal=x``, rejected as an unknown capability) raise violation risk for the correct one
    (``memory-set goal=x``).
    """

    def __init__(self, value_by_invariant: dict[str, str], detector: BehaviouralDriftDetector,
                 options: ConscienceOptions, scope_by_invariant: Optional[dict[str, InvariantScope]] = None) -> None:
        """:param value_by_invariant: protected-value tag of each registered invariant, by invariant name.
        :param detector: detector used to score the drift past transitions produced.
        :param options: similarity threshold and precedent count.
        :param scope_by_invariant: lifecycle scope of each registered invariant, by name. Used to ignore plan-scope
            rejections; when omitted, every rejection counts.
        """
        self._value_by_invariant = value_by_invariant
        self._scope_by_invariant = scope_by_invariant or {}
        self._detector = detector
        self._options = options

    def find(self, history: Sequence[Transition], action: AgentAction) -> list[Precedent]:
        """Most similar precedents for ``action``, most similar (then most recent) first."""
        candidates = []
        for i, transition in enumerate(history):
            if not self._is_candidate(transition):
                continue
            similarity = PlanSimilarity.score(action.capability, action.input,
                                              transition.proposed_action.capability, transition.proposed_action.input)
            if similarity >= self._options.min_precedent_similarity:
                candidates.append((transition, similarity, i))
        candidates.sort(key=lambda c: (-c[1], -c[2]))
        return [self._to_precedent(t, s) for t, s, _ in candidates[: self._options.max_precedents]]

    def base_rate(self, history: Sequence[Transition]) -> tuple[float, int]:
        """Smoothed rate at which the controls blocked plans across the whole history (the cold-start prior), and the
        number of labelled transitions it is based on."""
        labelled = blocked = 0
        for transition in history:
            if not self._is_candidate(transition):
                continue
            outcome = self.classify(transition)
            if outcome == PrecedentOutcome.Withheld:
                continue
            labelled += 1
            if outcome == PrecedentOutcome.Blocked:
                blocked += 1
        alpha = self._options.base_rate_prior_alpha
        beta = self._options.base_rate_prior_beta
        return (blocked + alpha) / (labelled + alpha + beta), labelled

    @staticmethod
    def classify(transition: Transition) -> PrecedentOutcome:
        """How the controls treated a past transition."""
        stop = next((a for a in reversed(transition.appraisals or []) if a.stopped), None)
        if stop is not None and not transition.blocking_invariants:
            # Stopped at the answer point: every control had already passed, so the boundary label is "passed".
            # Stopped at the plan point: the controls never saw the rest of it, so there is no label.
            return PrecedentOutcome.Passed if stop.point == AppraisalPoint.Answer else PrecedentOutcome.Withheld
        return PrecedentOutcome.Blocked if transition.status == TransitionStatus.Rejected else PrecedentOutcome.Passed

    def blocked_by(self, transition: Transition) -> list[str]:
        """Invariants that blocked a transition (falls back to parsing the reason for older history)."""
        if transition.blocking_invariants:
            return transition.blocking_invariants
        # History recorded before BlockingInvariants existed: the reason is "Invariant: reason" per line.
        names = []
        for line in (transition.reason or "").split("\n"):
            line = line.strip()
            if not line:
                continue
            name = line.split(":", 1)[0].strip()
            if name in self._value_by_invariant:
                names.append(name)
        return distinct(names)

    def blocked_only_at_plan_scope(self, transition: Transition) -> bool:
        """``True`` if the transition was rejected only by plan-scope invariants. A plan under appraisal has already
        passed those, so such a rejection is not evidence about it."""
        if transition.status != TransitionStatus.Rejected or self.classify(transition) != PrecedentOutcome.Blocked:
            return False
        blockers = self.blocked_by(transition)
        return bool(blockers) and all(self._scope_by_invariant.get(name) == InvariantScope.Plan for name in blockers)

    def _is_candidate(self, transition: Transition) -> bool:
        action = transition.proposed_action
        return (action is not None
                and not self.blocked_only_at_plan_scope(transition)
                and not action.has_error
                and not is_blank(action.capability)
                and action.capability not in self._options.exempt_capabilities
                and transition.status in (TransitionStatus.Completed, TransitionStatus.Rejected))

    def _to_precedent(self, transition: Transition, similarity: float) -> Precedent:
        outcome = self.classify(transition)
        blocked_by = self.blocked_by(transition) if outcome == PrecedentOutcome.Blocked else []
        values = distinct((self._value_by_invariant[n] for n in blocked_by if n in self._value_by_invariant),
                          ignore_case=True)
        return Precedent(
            transition_id=transition.id,
            similarity=round(similarity, 4),
            outcome=outcome,
            capability=transition.proposed_action.capability,
            input=transition.proposed_action.input or "",
            blocked_by=list(blocked_by),
            values=values,
            realised_drift=self._detector.score_as_recorded(transition).score if transition.after is not None else None,
        )

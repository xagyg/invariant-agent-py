"""Connects an :class:`IConscience` to the governed runtime (IBAAM 6, Sections 5 and 6)."""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

from ..core._util import fmt_fixed, is_blank, is_empty, new_id, utc_now
from ..core.abstractions import ICommitmentStore, IConscience, IInvariant, IStabilityEvaluator, ITransitionStore
from ..core.conscience import CommitmentLedger, PlanSimilarity
from ..core.drift import BehaviouralDriftDetector
from ..core.model.agent import AgentAction
from ..core.model.conscience import Commitment, ConscienceAppraisal, ConscienceAppraisalRecord, ConscienceInput
from ..core.model.control import ProtectedValues
from ..core.model.enums import (
    AppraisalPoint,
    CommitmentKind,
    CommitmentOrigin,
    ConscienceEffect,
    InvariantScope,
    PrecedentOutcome,
    TensionKind,
    TransitionEventStage,
)
from ..core.model.planning import PlannerRevision
from ..core.model.transition import Transition, TransitionContext
from .options import ConscienceOptions, ConscienceSession
from .precedent_index import PrecedentIndex


class ConscienceCoordinator:
    """Assembles the conscience's input from governance history, records each appraisal on the transition, and
    creates commitments.

    The runtime decides what each stance does to the pipeline and calls this class only after the relevant invariants
    have passed, so the conscience can never authorise anything the controls block.
    """

    def __init__(
        self,
        conscience: IConscience,
        transitions: ITransitionStore,
        commitments: ICommitmentStore,
        stability: IStabilityEvaluator,
        detector: BehaviouralDriftDetector,
        invariants: Iterable[IInvariant],
        options: ConscienceOptions,
        session: Optional[ConscienceSession] = None,
    ) -> None:
        """:param invariants: the registered invariants, for their protected-value tags.
        :param session: current session (for conversation-scoped commitments); a new one if omitted."""
        self._conscience = conscience
        self._transitions = transitions
        self._commitments = commitments
        self._stability = stability
        self._detector = detector
        #: Conscience options in force.
        self.options = options
        #: Identifies this runtime session for conversation-scoped commitments. A restart starts a new conversation,
        #: while user- and deployment-scoped commitments carry over.
        self.conversation_id = (session or ConscienceSession()).conversation_id
        value_by_invariant: dict[str, str] = {}
        scope_by_invariant: dict[str, InvariantScope] = {}
        for invariant in invariants:
            value_by_invariant[invariant.name] = invariant.protected_value
            scope_by_invariant[invariant.name] = invariant.scope
        self._precedents = PrecedentIndex(value_by_invariant, detector, options, scope_by_invariant)

    def should_appraise(self, action: Optional[AgentAction]) -> bool:
        """``True`` if ``action`` should be appraised (not exempt and not a planner error)."""
        return (action is not None and not action.has_error and not is_blank(action.capability)
                and action.capability not in self.options.exempt_capabilities)

    def live_commitments(self) -> list[Commitment]:
        """The commitments live for this session right now."""
        return CommitmentLedger.live(self._commitments.get_all(), utc_now(), self.conversation_id,
                                     self.options.user_id, self.options.commitment_match_threshold)

    def build_input(self, context: TransitionContext, point: AppraisalPoint, attempt: int) -> ConscienceInput:
        """Assembles the conscience input for the transition's current action."""
        transition = context.transition
        history = self._transitions.get_all()
        action = transition.proposed_action
        precedents = self._precedents.find(history, action)
        base_rate, labelled = self._precedents.base_rate(history)
        forecast_drift = None
        # Answer point: the reducer has proposed the next state, so drift is a dry run of the commit, scored against
        # the baseline recorded on the transition (the same one the drift-bound invariant used).
        if point == AppraisalPoint.Answer and transition.after is not None:
            forecast_drift = self._detector.score_as_recorded(transition).score
        # Trend is measured without the conscience's own plan-point interventions: counting them as instability would
        # let each stop raise the pressure for the next (the self-reinforcing loop of IBAAM 6, Section 9).
        trend_history = [t for t in history if PrecedentIndex.classify(t) != PrecedentOutcome.Withheld]
        return ConscienceInput(
            point=point,
            proposed=action,
            state=transition.before,
            trend=self._stability.evaluate(trend_history),
            precedents=precedents,
            commitments=self.live_commitments(),
            base_rate=base_rate,
            labelled_history=labelled,
            forecast_drift_score=forecast_drift,
            drift_bound=BehaviouralDriftDetector.DEFAULT_BOUND_THRESHOLD,
            outcome=transition.outcome if point == AppraisalPoint.Answer else None,
            attempt=attempt,
        )

    def appraise(self, context: TransitionContext, point: AppraisalPoint, attempt: int) -> ConscienceAppraisal:
        """Appraises the transition's current action (``attempt`` 0 for the first proposal)."""
        appraisal = self._conscience.appraise(self.build_input(context, point, attempt))
        if is_empty(appraisal.conscience):
            appraisal = dataclasses.replace(appraisal, conscience=self._conscience.name)
        return appraisal

    def record(self, context: TransitionContext, point: AppraisalPoint, attempt: int,
               appraised: Optional[AgentAction], appraisal: ConscienceAppraisal, effect: ConscienceEffect,
               note: str = "") -> ConscienceAppraisalRecord:
        """Records an appraisal on the transition (for replay) with a Conscience event, and creates a commitment when
        the effect was Revised, Escalated or Vetoed."""
        transition = context.transition
        commitment = self._create_commitment(transition, appraised, appraisal, effect)
        if commitment is not None:
            self._commitments.append(commitment)
        record = ConscienceAppraisalRecord(
            point=point,
            attempt=attempt,
            capability=(appraised.capability if appraised else None) or "",
            input=(appraised.input if appraised else None) or "",
            appraisal=appraisal,
            effect=effect,
            note=note or "",
            commitment_id=commitment.id if commitment else None,
        )
        transition.appraisals.append(record)
        message = (f"{point} appraisal: {appraisal.stance} (tension {fmt_fixed(appraisal.tension, 2)}) -> {effect}"
                   + ("" if is_blank(note) else f" [{note}]"))
        transition.add_event(TransitionEventStage.Conscience, message, {
            "Point": point.name,
            "Attempt": attempt,
            "Capability": record.capability,
            "Stance": appraisal.stance.name,
            "Effect": effect.name,
            "Tension": appraisal.tension,
            "ViolationRisk": _part_value(appraisal, TensionKind.ViolationRisk),
            "ForecastDrift": _part_value(appraisal, TensionKind.ForecastDrift),
            "TrendPressure": _part_value(appraisal, TensionKind.TrendPressure),
            "CommitmentConflict": _part_value(appraisal, TensionKind.CommitmentConflict),
            "ValuesAtStake": ",".join(appraisal.values_at_stake),
            "Justification": appraisal.justification,
            "Commitment": str(commitment.id) if commitment else "",
        })
        return record

    @staticmethod
    def to_revision(attempt: int, previous: Optional[AgentAction], appraisal: ConscienceAppraisal) -> PlannerRevision:
        """Conscience feedback for the planner when a revised plan is requested."""
        return PlannerRevision(attempt=attempt, previous=previous, tension=appraisal.tension,
                               justification=appraisal.justification, values_at_stake=list(appraisal.values_at_stake))

    @staticmethod
    def same_plan(a: Optional[AgentAction], b: Optional[AgentAction]) -> bool:
        """True if two actions are materially the same plan."""
        return (a is not None and b is not None
                and PlanSimilarity.normalise(a.capability) == PlanSimilarity.normalise(b.capability)
                and PlanSimilarity.normalise(a.input) == PlanSimilarity.normalise(b.input))

    def _create_commitment(self, transition: Transition, appraised: Optional[AgentAction],
                           appraisal: ConscienceAppraisal, effect: ConscienceEffect) -> Optional[Commitment]:
        if appraised is None or effect not in (ConscienceEffect.Revised, ConscienceEffect.Escalated,
                                               ConscienceEffect.Vetoed):
            return None
        o = self.options
        scope = o.revise_commitment_scope if effect == ConscienceEffect.Revised else o.stop_commitment_scope
        now = utc_now()
        value = appraisal.values_at_stake[0] if appraisal.values_at_stake else ProtectedValues.ControlBoundary
        verb = {ConscienceEffect.Revised: "Asked to revise",
                ConscienceEffect.Escalated: "Held for human review"}.get(effect, "Declined")
        return Commitment(
            id=new_id(),
            source_transition=transition.id,
            value=value,
            scope=scope,
            statement=f"{verb}: {appraised.capability} {appraised.input}".strip(),
            origin=CommitmentOrigin.Agent,
            created=now,
            expires=now + o.agent_commitment_lifetime if o.agent_commitment_lifetime is not None else None,
            kind=CommitmentKind.Declined,
            capability=appraised.capability or "",
            subject=PlanSimilarity.normalise(appraised.input),
            conversation_id=self.conversation_id,
            user_id=o.user_id,
        )


def _part_value(appraisal: ConscienceAppraisal, kind: TensionKind) -> float:
    return next((p.value for p in appraisal.parts if p.kind == kind), 0.0)

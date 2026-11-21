"""The governed agent runtime: orchestrates every input through the invariant-bounded transition lifecycle."""

from __future__ import annotations

from typing import Optional

from ..conscience.coordinator import ConscienceCoordinator
from ..core.abstractions import (
    IAlternativePlanner,
    IDriftBaselineStore,
    IExecutor,
    IInvariantEvaluator,
    IPlanner,
    IStateReducer,
    ITransitionStore,
)
from ..core.control.hierarchy import OperationalContextState
from ..core.model.agent import AgentAction, AgentState
from ..core.model.conscience import ConscienceAppraisal
from ..core.model.control import GovernanceCandidate, InvariantEvaluationReport
from ..core.model.drift import DriftBaselineSnapshot
from ..core.model.enums import (
    AppraisalPoint,
    ConscienceEffect,
    ConscienceStance,
    InvariantScope,
    OperationalContext,
    TransitionEventStage,
    TransitionPhase,
    TransitionStatus,
)
from ..core.model.planning import PlannerContext
from ..core.model.transition import ConflictRecord, Transition, TransitionContext
from ..core.pipeline import PlannerContextProjector
from ..core.transitioning import TransitionPhases
from ..core._util import distinct, short_id


class GovernedAgentRuntime:
    """Orchestrates every input through the invariant-bounded transition lifecycle.

    Runtime hierarchy: invariants, then runtime governance, then planner, then capabilities.

    Pipeline for each input:

    1. Input received and the planner proposes an action from the bounded projection ``π(Sₜ)``.
    2. Pre-control: HIG-C resolves the plan checkpoint over the proposal, any planner alternatives
       (:class:`IAlternativePlanner`) and declining; the governed action is carried forward.
    3. Conscience plan appraisal (IBAAM 6), when a conscience is configured: Proceed, Revise (re-plan, re-check plan
       invariants, re-appraise; bounded), Escalate or Veto.
    4. Capability execution.
    5. Post-control: Execution invariants.
    6. Self-modification invariants.
    7. Reduction: the reducer proposes ``Sₜ₊₁``, then Reduction invariants (e.g. drift bound) are checked.
    8. Conscience answer appraisal: Proceed, caveat (Revise), Escalate or Veto before commit.
    9. Commit the new state and persist the transition.

    The conscience sits strictly inside the invariants: it is only consulted after they pass, and none of its stances
    can authorise anything they block. Every revised plan passes every plan invariant again.

    At every checkpoint the governed outcome is either to continue (possibly with witnessed overrides) or to decline.
    Declining rejects the transition; when every candidate violates a fundamental invariant the transition is also
    marked as escalated for human review. Rejected transitions are persisted too, but never change runtime state.

    The C# runtime exposes ``RunAsync``; this port is synchronous (:meth:`run`).
    """

    def __init__(
        self,
        planner: IPlanner,
        invariant_evaluator: IInvariantEvaluator,
        executor: IExecutor,
        reducer: IStateReducer,
        store: ITransitionStore,
        baselines: Optional[IDriftBaselineStore] = None,
        conscience: Optional[ConscienceCoordinator] = None,
        operational_context: Optional[OperationalContextState] = None,
    ) -> None:
        """Creates the runtime from its governed components.

        :param planner: untrusted planner that proposes actions.
        :param invariant_evaluator: evaluator enforcing invariants at each checkpoint.
        :param executor: executes approved actions.
        :param reducer: assimilates approved outcomes into a proposed next state.
        :param store: persists every transition, completed or rejected. With a persistent store the runtime resumes
            from the last committed state, so a restart does not silently reset the agent while its history survives.
        :param baselines: approved drift baselines. When supplied, the baseline in force is recorded on each
            transition before reduction invariants run, so replay can score drift against the same baseline.
        :param conscience: functional conscience (IBAAM 6). When ``None`` the runtime behaves exactly as without one.
        :param operational_context: HIG-C operational context in force (IBAAM 3 revised). Stamped on each transition
            when it starts, so conflicts are resolved under it and replay reproduces the decision. When ``None`` every
            transition runs as Normal.
        """
        self._planner = planner
        self._evaluator = invariant_evaluator
        self._executor = executor
        self._reducer = reducer
        self._store = store
        self._baselines = baselines
        self._conscience = conscience
        self._operational_context = operational_context
        # The last committed state. Replaced only after all governance checkpoints pass.
        self._state = self._restore_committed_state(store)

    @property
    def state(self) -> AgentState:
        """The last committed agent state."""
        return self._state

    # The After state of the most recent completed transition is the last state that was committed.
    @staticmethod
    def _restore_committed_state(store: ITransitionStore) -> AgentState:
        for t in reversed(store.get_all()):
            if t.status == TransitionStatus.Completed and t.after is not None:
                return t.after
        return AgentState()

    def run(self, input: str) -> TransitionContext:
        """Processes one input through the full governed transition lifecycle.

        Returns the transition context, including the full event trail and final status.
        """
        transition = Transition(input=input, before=self._state)
        TransitionPhases.move_to(transition, TransitionPhase.InputReceived)
        transition.add_event(TransitionEventStage.Input, input, {"RawInput": input})
        # HIG-C: the context in force when the transition starts governs all of its checkpoints.
        context = TransitionContext(
            transition=transition,
            operational_context=(self._operational_context.current if self._operational_context is not None
                                 else OperationalContext.Normal))
        if context.operational_context != OperationalContext.Normal:
            transition.add_event(TransitionEventStage.Lifecycle, f"OperationalContext={context.operational_context}", {
                "OperationalContext": context.operational_context.name,
                "Reason": self._operational_context.reason if self._operational_context is not None else "",
            })
        TransitionPhases.move_to(transition, TransitionPhase.Planning)

        # PLAN
        planner_context = PlannerContextProjector.project(self._state, input)
        proposed_action = self._planner.plan(planner_context)
        transition.proposed_action = proposed_action
        transition.status = TransitionStatus.Proposed
        capability = proposed_action.capability if proposed_action else None
        transition.add_event(TransitionEventStage.Planning, f"Capability={capability or ''}", {
            "Capability": capability,
            "Input": transition.input,
            "Planner": self._planner.name,
        })
        TransitionPhases.move_to(transition, TransitionPhase.PlanValidation)

        # PRE-CONTROL: HIG-C resolves over the proposal, any planner alternatives and declining.
        alternatives = self._propose_alternatives(planner_context, proposed_action)
        decision = self._evaluator.evaluate(context, InvariantScope.Plan, alternatives)
        if not self._try_apply_governance_outcome(context, decision):
            self._reject_by_controls(transition, decision)
            self._store.append(transition)
            return context

        # CONSCIENCE: plan appraisal, inside the boundary the plan invariants just confirmed.
        if self._conscience is not None and not self._appraise_plan(context, input):
            self._store.append(transition)
            return context
        TransitionPhases.move_to(transition, TransitionPhase.Execution)

        # EXECUTION
        self._executor.execute(context)
        outcome = transition.outcome
        transition.add_event(
            TransitionEventStage.Execution,
            outcome.result if outcome is not None and outcome.success else (outcome.error if outcome else "Unknown"),
            {
                "Capability": outcome.capability if outcome else None,
                "Success": outcome.success if outcome else None,
                "Error": outcome.error if outcome else "",
            })
        TransitionPhases.move_to(transition, TransitionPhase.ExecutionValidation)

        # POST-CONTROL
        decision = self._evaluator.evaluate(context, InvariantScope.Execution)
        if not self._try_apply_governance_outcome(context, decision):
            self._reject_by_controls(transition, decision)
            self._store.append(transition)
            return context
        TransitionPhases.move_to(transition, TransitionPhase.SelfModificationValidation)

        # SELF-MODIFICATION CHECK
        decision = self._evaluator.evaluate(context, InvariantScope.SelfModification)
        if not self._try_apply_governance_outcome(context, decision):
            self._reject_by_controls(transition, decision)
            self._store.append(transition)
            return context
        # Do not allow the reducer to run if the transition was rejected.
        if transition.status == TransitionStatus.Rejected:
            self._store.append(transition)
            return context
        TransitionPhases.move_to(transition, TransitionPhase.Reduction)

        # STATE ASSIMILATION
        self._reducer.apply(context)
        transition.add_event(
            TransitionEventStage.Reduction,
            f"ProposedVersion={transition.after.version if transition.after else ''}",
            {
                "BeforeVersion": transition.before.version if transition.before else None,
                "ProposedAfterVersion": transition.after.version if transition.after else None,
                "Committed": False,
            })
        # Record the drift baseline these invariants are about to use, for deterministic replay. Without a baseline
        # store we cannot know which baseline the detector uses, so record nothing.
        if self._baselines is not None:
            transition.drift_baseline = DriftBaselineSnapshot.from_baseline(self._baselines.current)

        # REDUCTION INVARIANTS
        decision = self._evaluator.evaluate(context, InvariantScope.Reduction)
        if not self._try_apply_governance_outcome(context, decision):
            self._reject_by_controls(transition, decision)
            self._store.append(transition)
            return context

        # CONSCIENCE: answer appraisal, after every control has passed and before anything is committed.
        if self._conscience is not None and not self._appraise_answer(context):
            self._store.append(transition)
            return context

        # COMMIT NEW STATE
        if transition.after is not None:
            self._state = transition.after
            transition.add_event(TransitionEventStage.Lifecycle, f"CommittedVersion={self._state.version}",
                                 {"CommittedVersion": self._state.version, "Committed": True})
        TransitionPhases.move_to(transition, TransitionPhase.Completed)
        self._store.append(transition)
        return context

    def _appraise_plan(self, context: TransitionContext, input: str) -> bool:
        """Plan appraisal loop (IBAAM 6, Section 6.3). Returns ``False`` if the transition was stopped.

        Revise hands the appraisal back to the planner for a bounded number of re-plans. Each revised plan goes
        through every plan invariant again and is re-appraised, so the action appraised is always the action executed.
        When the revision limit is reached, or the planner cannot produce a different plan, the stance becomes
        Escalate.
        """
        transition = context.transition
        conscience = self._conscience
        attempt = 0
        while True:
            action = transition.proposed_action
            if not conscience.should_appraise(action):
                return True
            appraisal = conscience.appraise(context, AppraisalPoint.Plan, attempt)
            if appraisal.stance == ConscienceStance.Proceed:
                conscience.record(context, AppraisalPoint.Plan, attempt, action, appraisal, ConscienceEffect.Proceeded)
                return True
            if appraisal.stance == ConscienceStance.Veto:
                conscience.record(context, AppraisalPoint.Plan, attempt, action, appraisal, ConscienceEffect.Vetoed)
                TransitionPhases.reject(transition, f"Conscience veto: {appraisal.justification}")
                return False
            if appraisal.stance == ConscienceStance.Escalate:
                self._escalate(context, AppraisalPoint.Plan, attempt, action, appraisal, "")
                return False
            # REVISE
            if attempt >= conscience.options.max_revisions:
                self._escalate(context, AppraisalPoint.Plan, attempt, action, appraisal,
                               f"revision limit {conscience.options.max_revisions} reached")
                return False
            revision = ConscienceCoordinator.to_revision(attempt + 1, action, appraisal)
            revised = self._planner.plan(PlannerContextProjector.project(self._state, input, revision))
            if ConscienceCoordinator.same_plan(action, revised):
                self._escalate(context, AppraisalPoint.Plan, attempt, action, appraisal,
                               f"planner '{self._planner.name}' could not produce a different plan")
                return False
            conscience.record(context, AppraisalPoint.Plan, attempt, action, appraisal, ConscienceEffect.Revised)
            transition.proposed_action = revised
            transition.add_event(TransitionEventStage.Planning,
                                 f"Revised: Capability={revised.capability if revised else ''}", {
                                     "Capability": revised.capability if revised else None,
                                     "Input": revised.input if revised else None,
                                     "Planner": self._planner.name,
                                     "RevisionAttempt": attempt + 1,
                                     "PreviousCapability": action.capability if action else None,
                                     "PreviousInput": action.input if action else None,
                                 })
            # Every revised plan passes every plan invariant again.
            decision = self._evaluator.evaluate(context, InvariantScope.Plan)
            if not self._try_apply_governance_outcome(context, decision):
                self._reject_by_controls(transition, decision)
                return False
            attempt += 1

    def _appraise_answer(self, context: TransitionContext) -> bool:
        """Answer appraisal. Returns ``False`` if the transition was stopped before commit. Revise at this point cannot
        re-plan an action that has already run, so it is recorded as a caveat on the answer."""
        transition = context.transition
        conscience = self._conscience
        action = transition.proposed_action
        if not conscience.options.appraise_answers or not conscience.should_appraise(action):
            return True
        attempt = max(0, sum(1 for a in transition.appraisals if a.point == AppraisalPoint.Plan) - 1)
        appraisal = conscience.appraise(context, AppraisalPoint.Answer, attempt)
        if appraisal.stance == ConscienceStance.Revise:
            conscience.record(context, AppraisalPoint.Answer, attempt, action, appraisal, ConscienceEffect.Caveated,
                              "answer already produced; caveat recorded instead of re-planning")
            return True
        if appraisal.stance == ConscienceStance.Escalate:
            self._escalate(context, AppraisalPoint.Answer, attempt, action, appraisal, "")
            return False
        if appraisal.stance == ConscienceStance.Veto:
            conscience.record(context, AppraisalPoint.Answer, attempt, action, appraisal, ConscienceEffect.Vetoed)
            TransitionPhases.reject(transition, f"Conscience veto: {appraisal.justification}")
            return False
        conscience.record(context, AppraisalPoint.Answer, attempt, action, appraisal, ConscienceEffect.Proceeded)
        return True

    # Holds the transition for human review: nothing further runs and nothing is committed.
    def _escalate(self, context: TransitionContext, point: AppraisalPoint, attempt: int,
                  action: Optional[AgentAction], appraisal: ConscienceAppraisal, note: str) -> None:
        self._conscience.record(context, point, attempt, action, appraisal, ConscienceEffect.Escalated, note)
        TransitionPhases.reject(
            context.transition,
            f"Conscience escalated for human review (conscience-review {short_id(context.transition.id)} "
            f"uphold|overturn): {appraisal.justification}")

    # Rejects through the controls and records which invariants blocked it (boundary labels for the conscience). A
    # fundamental conflict that no pre-authorised protocol covers is held for human review (HIG-C, Section 7.2).
    @staticmethod
    def _reject_by_controls(transition: Transition, report: InvariantEvaluationReport) -> None:
        transition.blocking_invariants = distinct(v.invariant for v in report.violations)
        if report.requires_escalation:
            transition.governance_escalated = True
        TransitionPhases.reject(transition, report.summary)

    # Asks an alternative-capable planner for further candidates; other planners contribute only their proposal.
    def _propose_alternatives(self, planner_context: PlannerContext,
                              proposed: Optional[AgentAction]) -> list[AgentAction]:
        if not isinstance(self._planner, IAlternativePlanner) or proposed is None:
            return []
        alternatives = self._planner.propose_alternatives(planner_context, proposed) or []
        return [a for a in alternatives if a is not None and not ConscienceCoordinator.same_plan(proposed, a)]

    def _try_apply_governance_outcome(self, context: TransitionContext, report: InvariantEvaluationReport) -> bool:
        """Records a Control event describing the evaluation report (violations and overrides) and returns whether the
        transition may proceed."""
        transition = context.transition
        # HIG-C may select a planner alternative over the proposal: from here on the transition carries that action.
        if report.passed and report.selected_alternative is not None:
            previous = transition.proposed_action
            transition.proposed_action = report.selected_alternative
            transition.add_event(
                TransitionEventStage.Planning,
                f"Governed alternative selected: Capability={report.selected_alternative.capability}", {
                    "Capability": report.selected_alternative.capability,
                    "Input": report.selected_alternative.input,
                    "Selected": report.resolution.selected_id if report.resolution else "",
                    "PreviousCapability": previous.capability if previous else None,
                    "PreviousInput": previous.input if previous else None,
                    "Reason": report.resolution.reason if report.resolution else "",
                })
        conflict = ConflictRecord.from_report(report)
        if conflict is not None:
            transition.conflicts.append(conflict)
        transition.add_event(
            TransitionEventStage.Control,
            f"{report.scope} invariants passed" if report.passed else f"{report.scope} invariants failed: {report.summary}",
            {
                "Scope": report.scope.name,
                "Allowed": report.passed,
                "Reason": report.summary,
                "ViolationCount": len(report.violations),
                "OverrideCount": len(report.overrides),
                "Declined": report.declined,
                "RequiresEscalation": report.requires_escalation,
                "Selected": report.resolution.selected_id if report.resolution else GovernanceCandidate.PROPOSED_ID,
                "GovernedSet": list(report.resolution.governed_set) if report.resolution else [],
                "Violations": [{
                    "Invariant": v.invariant,
                    "Category": v.category.name,
                    "Scope": v.scope.name,
                    "Severity": v.severity.name,
                    "Layer": v.layer.name,
                    "Criticality": v.criticality_in_context.name,
                    "Degree": v.degree,
                    "Reason": v.reason,
                } for v in report.violations],
                "Overrides": [{
                    "OverriddenInvariant": o.overridden_violation.invariant,
                    "OverriddenLayer": o.overridden_layer.name,
                    "PreservedHigherPriorityInvariants": list(o.preserved_higher_priority_invariants),
                    "Witnesses": [str(w) for w in o.witnesses],
                    "Justification": o.justification,
                    "MetaInvariantCategory": o.meta_invariant_category,
                    "RequiresReview": o.requires_review,
                    "ReviewReasons": list(o.review_reasons),
                } for o in report.overrides],
            })
        return report.passed

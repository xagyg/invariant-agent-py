"""Every enumeration of the model, in one module.

Numeric values match the C# enums exactly because the JSONL history persists enums as numbers. New members must
only ever be appended.
"""

from __future__ import annotations

from .._util import CsEnum

# --------------------------------------------------------------------------------------------------------- control


class InvariantCategory(CsEnum):
    """The behavioural concern an invariant protects. Used for grouping in drift reports."""

    #: Prevents harmful or disallowed actions.
    Safety = 0
    #: Protects correctness of execution, outcomes and lifecycle.
    Integrity = 1
    #: Bounds behavioural drift so the agent's behaviour remains recognisably stable.
    Identity = 2
    #: Governs changes the agent proposes to its own state.
    SelfModification = 3


class InvariantCriticality(CsEnum):
    """HIG-C criticality: the potential severity of consequences if an invariant is violated.

    Within a level, a more critical violation outweighs any amount of less critical violation (HIG-C).
    """

    #: Minor consequences.
    Low = 1
    #: Moderate consequences (default).
    Medium = 2
    #: Significant consequences.
    High = 3
    #: Severe consequences.
    Critical = 4
    #: Irreversible or catastrophic consequences.
    Catastrophic = 5


class InvariantLayer(CsEnum):
    """HIG governance layers. Higher numeric values take precedence:
    Fundamental, then Mission, then Behavioural, then AdaptiveHeuristic."""

    #: Learned or tunable heuristics; lowest priority and most readily overridden.
    AdaptiveHeuristic = 1
    #: Behavioural preferences and norms.
    Behavioural = 2
    #: Obligations arising from the agent's mission or task.
    Mission = 3
    #: Non-negotiable constraints. Never overridden.
    Fundamental = 4


class InvariantScope(CsEnum):
    """The lifecycle checkpoint at which an invariant is evaluated."""

    #: Evaluated at every checkpoint.
    Transition = 0
    #: Pre-control: validates the planner's proposed action before execution.
    Plan = 1
    #: Post-control: validates the execution outcome.
    Execution = 2
    #: Validates agent state (reserved).
    State = 3
    #: Validates memory contents (reserved).
    Memory = 4
    #: Validates the reducer's proposed next state before it is committed.
    Reduction = 5
    #: Validates proposed self-modifications before they are assimilated.
    SelfModification = 6
    #: Validates observability concerns (reserved).
    Observability = 7


class InvariantSeverity(CsEnum):
    """Severity of an invariant violation or drift measurement. Error and Critical are blocking by default."""

    #: Informational; overrideable.
    Info = 0
    #: Soft violation; overrideable under HIG.
    Warning = 1
    #: Hard violation; blocking.
    Error = 2
    #: Severe violation; blocking.
    Critical = 3


class MetaInvariantCategory(CsEnum):
    """The governance concern addressed by a meta-invariant."""

    #: Hierarchy precedence between layers.
    Priority = 0
    #: Which severities may be overridden.
    OverrideSeverity = 1
    #: Requirement for an inspectable override rationale.
    Justification = 2
    #: Marking overrides for governance audit.
    Audit = 3
    #: Flagging ambiguous overrides for human review.
    Review = 4


class OperationalContext(CsEnum):
    """HIG-C operational circumstances κ: they set each invariant's criticality C(I, κ) and so shape conflict
    resolution."""

    #: Normal operation (default).
    Normal = 0
    #: Operating with reduced capability or resources.
    Degraded = 1
    #: Responding to an emergency.
    Emergency = 2
    #: Recovering from a failure.
    Recovery = 3
    #: Planned maintenance.
    Maintenance = 4


class ContextualGovernanceOutcome(CsEnum):
    """How criticality and context entered an override decision under HIG-C."""

    #: Priority between levels decided the override; criticality and context did not.
    NotApplicable = 0
    #: Retained for compatibility with records written before the revised HIG-C rule; no longer produced.
    Preserve = 1
    #: A same-level comparison decided the override: every competing candidate is worse at the invariant's own
    #: level, by criticality in the current context.
    Subordinate = 2
    #: Retained for compatibility with records written before the revised HIG-C rule; no longer produced.
    Unresolved = 3


class ViolationType(CsEnum):
    """Violation classification mirroring :class:`InvariantCategory` (not currently referenced)."""

    Safety = 0
    Integrity = 1
    Identity = 2
    SelfModification = 3


# ------------------------------------------------------------------------------------------------------ transition


class TransitionPhase(CsEnum):
    """Ordered lifecycle phases of a transition.

    Ordering matters: ``TransitionLifecycle`` compares phases numerically, and legal moves are defined in
    ``TransitionPhases``.
    """

    #: Transition object created.
    Created = 0
    #: User input recorded.
    InputReceived = 1
    #: Planner is proposing an action.
    Planning = 2
    #: Pre-control: plan invariants are evaluated.
    PlanValidation = 3
    #: Capability is executed.
    Execution = 4
    #: Post-control: execution invariants are evaluated.
    ExecutionValidation = 5
    #: Proposed self-modifications are evaluated.
    SelfModificationValidation = 6
    #: Reducer builds the next state and reduction invariants are evaluated.
    Reduction = 7
    #: Terminal: state committed.
    Completed = 8
    #: Terminal: rejected by governance; no state committed.
    Rejected = 9


class TransitionStatus(CsEnum):
    """Classification of a transition's progress or result."""

    #: The planner has proposed an action.
    Proposed = 0
    #: Allowed by governance (reserved).
    Allowed = 1
    #: Rejected by governance.
    Rejected = 2
    #: The capability has executed.
    Executed = 3
    #: The transition completed (or the reducer produced the next state).
    Completed = 4
    #: Execution could not proceed (e.g. no proposed action).
    Failed = 5


class TransitionEventStage(CsEnum):
    """The pipeline stage that raised a ``TransitionEvent``."""

    Input = 0
    Planning = 1
    Invariant = 2
    Control = 3
    Execution = 4
    Reduction = 5
    SelfModification = 6
    Lifecycle = 7
    #: Conscience appraisal (IBAAM 6). Appended last because stages are persisted as numbers.
    Conscience = 8


# ----------------------------------------------------------------------------------------------------------- drift


class DriftType(CsEnum):
    """Kinds of drift tracked by the observability layer."""

    #: No drift.
    None_ = 0
    #: Re-evaluating invariants on replay produced a different decision than originally recorded.
    GovernanceDrift = 1
    #: Adaptive state (goal, mode, policies, memory) diverged.
    BehaviouralDrift = 2
    #: Transition lifecycle behaviour diverged (reserved).
    LifecycleDrift = 3
    #: Execution outcomes diverged (reserved).
    OutcomeDrift = 4
    #: Invariant results diverged (reserved).
    InvariantDrift = 5

    def __str__(self) -> str:  # "None" is a Python keyword, so the member is None_ but displays as None.
        return "None" if self is DriftType.None_ else self.name

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)


# ------------------------------------------------------------------------------------------------------- stability


class MeasurementConfidence(CsEnum):
    """Confidence in a stability measurement, based on how many completed transitions were available."""

    Low = 0
    Medium = 1
    High = 2


class StabilityRegion(CsEnum):
    """Stability classification used to drive governance recommendations."""

    #: Behaviour within normal bounds; continue normal monitoring.
    Stable = 0
    #: Early signs of instability; increase monitoring.
    Watch = 1
    #: Significant instability; enhanced validation required before further adaptation.
    Intervention = 2
    #: Instability combined with invariant failures; human review required.
    Critical = 3


# ------------------------------------------------------------------------------------------------------ conscience


class AppraisalPoint(CsEnum):
    """Where in the transition lifecycle the conscience appraises (IBAAM 6, Section 5 and Figure 1)."""

    #: Plan appraisal: after plan invariants pass and before execution, so the conscience sees the action that will
    #: actually run.
    Plan = 0
    #: Answer appraisal: after execution, self-modification and reduction invariants pass and before the new state is
    #: committed, so the conscience can weigh the actual outcome and resulting state.
    Answer = 1


class CommitmentScope(CsEnum):
    """The scope a commitment applies to (IBAAM 6, Section 6.4.2)."""

    #: Only the runtime session (conversation) in which it arose.
    Conversation = 0
    #: Every session of the same user.
    User = 1
    #: The whole deployment.
    Deployment = 2


class CommitmentOrigin(CsEnum):
    """Who made a commitment. Reviewer commitments take precedence over the agent's own."""

    #: Created by the conscience when it adopted a Revise, Escalate or Veto stance.
    Agent = 0
    #: Created when a human reviewer resolved an escalation.
    Reviewer = 1


class CommitmentKind(CsEnum):
    """Whether the commitment records conduct declined or conduct undertaken (cleared)."""

    #: The agent (or a reviewer) declined this kind of conduct. Similar plans raise commitment conflict.
    Declined = 0
    #: A reviewer cleared this kind of conduct. Supersedes earlier declined commitments on the same subject and lets
    #: similar plans proceed to the controls.
    Undertaken = 1


class ConscienceEffect(CsEnum):
    """What the runtime did with an appraisal."""

    #: The pipeline continued unchanged.
    Proceeded = 0
    #: The appraisal was returned to the planner and a revised plan was requested.
    Revised = 1
    #: The transition was held for human review; nothing further ran.
    Escalated = 2
    #: The transition was rejected by the conscience.
    Vetoed = 3
    #: Answer point: the answer was allowed with a caveat recorded (Revise at the answer point).
    Caveated = 4


class ConscienceStance(CsEnum):
    """The four stances a functional conscience can take on a plan (IBAAM 6, Section 6.3).

    Every stance is contained by the controls: none of them can authorise an action an invariant blocks.
    """

    #: No change to the pipeline.
    Proceed = 0
    #: Return the appraisal to the planner for a bounded number of re-plans. Every revised plan passes every plan
    #: invariant again and is re-appraised. At the answer appraisal point Revise becomes a caveat.
    Revise = 1
    #: Route the decision to human review; nothing is executed or committed meanwhile.
    Escalate = 2
    #: Reject through the same path as a failed control, with the justification logged.
    Veto = 3


class PrecedentOutcome(CsEnum):
    """How a precedent transition ended, as far as the controls are concerned."""

    #: Every control passed and the transition completed.
    Passed = 0
    #: A control (invariant) blocked or rejected it. A boundary label.
    Blocked = 1
    #: The conscience stopped it before the controls saw all of it, so there is no boundary label. Excluded from
    #: violation risk to avoid the feedback loop in IBAAM 6, Section 9.
    Withheld = 2


class TensionKind(CsEnum):
    """The four components of prospective tension (IBAAM 6, Section 6.2)."""

    #: Probability that similar plans were later blocked or rejected by the controls. The only component that is a
    #: calibrated probability.
    ViolationRisk = 0
    #: Drift the plan's resulting state would produce, relative to the drift bound.
    ForecastDrift = 1
    #: Remaining tolerance given recent behaviour (the current stability region).
    TrendPressure = 2
    #: Inconsistency with the agent's own live commitments and reviewer decisions.
    CommitmentConflict = 3

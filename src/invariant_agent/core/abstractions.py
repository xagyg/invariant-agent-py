"""Contracts between the IBAAM layers.

The C# interface names (``IInvariant``, ``IConscience``, ``ICapability`` ...) are kept so the code lines up with the
IBAAM papers and the C# reference implementation, which refer to them by name. Python members are snake_case.

Implementations may provide ``name``, ``category``, ``layer`` and similar members either as class attributes, instance
attributes or properties.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from .model.control import ProtectedValues, ViolationProfile
from .model.enums import (
    InvariantCategory,
    InvariantCriticality,
    InvariantLayer,
    InvariantScope,
    InvariantSeverity,
    MetaInvariantCategory,
    OperationalContext,
)

if TYPE_CHECKING:  # pragma: no cover
    from .model.agent import AgentAction, AgentState
    from .model.capability import CapabilityRequest, CapabilityResult
    from .model.conscience import Commitment, ConscienceAppraisal, ConscienceInput, TensionComponent
    from .model.control import (
        InvariantEvaluationReport,
        InvariantResult,
        InvariantViolation,
        MetaInvariantContext,
        MetaInvariantResult,
    )
    from .model.drift import DriftBaseline, DriftReport
    from .model.planning import PlannerContext
    from .model.stability import StabilityAssessment
    from .model.transition import Transition, TransitionContext


# ------------------------------------------------------------------------------------------------------- adaptive


class IPlanner(ABC):
    """Proposes the next action for the agent (IBAAM Adaptive layer).

    Planners - including LLM planners - are treated as untrusted cognitive generators. They only see a bounded
    projection of runtime state and their output is always subject to invariant evaluation before anything is
    executed.
    """

    #: Planner identifier recorded on planning events (e.g. "command", "openai", "gemini").
    name: str

    @abstractmethod
    def plan(self, context: "PlannerContext") -> "AgentAction":
        """Produces a proposed action from the bounded planner context (user input plus ``π(Sₜ)``)."""


class IAlternativePlanner(IPlanner):
    """A planner that can offer alternatives to its proposal, enlarging the candidate set 𝒜 that HIG-C resolves over
    (IBAAM 3 revised, Section 6).

    The runtime always adds the null action itself. Alternatives are listed in the planner's order of preference;
    when several candidates tie under the resolution rule, that order decides (the adaptive layer chooses within the
    governed set). Alternatives are proposals like any other: they are untrusted and pass every plan invariant before
    anything runs.
    """

    @abstractmethod
    def propose_alternatives(self, context: "PlannerContext", proposed: "AgentAction") -> list["AgentAction"]:
        """Proposes alternatives to ``proposed``, most preferred first, from the same bounded planner context."""


# ------------------------------------------------------------------------------------------------------ execution


class ICapability(ABC):
    """A unit of execution (tool or service) that the runtime can invoke on behalf of the agent.

    In IBAAM terms a capability belongs to the Execution layer. Capabilities never mutate runtime state directly:
    they return a ``CapabilityResult`` which may carry a proposed self-modification that is only applied after
    invariant approval and reduction.
    """

    #: Primary name used by planners and the capability registry to select this capability.
    name: str
    #: Optional alternative names that also resolve to this capability.
    aliases: Sequence[str] = ()
    #: ``True`` for governance capabilities only a human operator may invoke (resolving conscience escalations,
    #: re-certifying the drift baseline, changing the operational context).
    #:
    #: Enforced by ``OperatorOnlyCapabilityInvariant``, not by planner prompts: an operator-only capability runs only
    #: when the operator typed that exact command, so a planner (an LLM in particular) can never propose one itself.
    #: Otherwise the agent could clear its own escalations or move the baseline its drift is measured against, and
    #: the control layer would no longer be outside the agent's reach (IBAAM 1, Section 4.3).
    operator_only: bool = False

    @abstractmethod
    def execute(self, request: "CapabilityRequest") -> "CapabilityResult":
        """Executes the capability; returns the result, including any proposed (not yet applied) self-modification."""


class ICapabilityRegistry(ABC):
    """Resolves capability names (and aliases) to :class:`ICapability` instances."""

    @abstractmethod
    def get(self, capability_name: str) -> Optional[ICapability]:
        """Looks up a capability by name or alias; ``None`` if none is registered."""

    @abstractmethod
    def get_capability_names(self) -> list[str]:
        """Returns every registered name and alias. Used to seed the allowed-capability invariant."""


class IExecutor(ABC):
    """Executes the planner's proposed action for a transition (IBAAM Execution layer)."""

    @abstractmethod
    def execute(self, context: "TransitionContext") -> None:
        """Executes ``context.transition.proposed_action`` and records the outcome and any proposed self-modification
        on the transition. Implementations must not mutate agent state."""


class IStateReducer(ABC):
    """Assimilates an approved transition into a proposed next state ``Sₜ₊₁`` (IBAAM state assimilation)."""

    @abstractmethod
    def apply(self, context: "TransitionContext") -> None:
        """Builds ``context.transition.after`` from the prior state, the outcome and any approved self-modification.
        The proposed state is only committed by the runtime after reduction invariants pass."""


# -------------------------------------------------------------------------------------------------------- control


class IInvariant(ABC):
    """A behavioural constraint evaluated by the runtime at a specific point in the transition lifecycle.

    Invariants are the core of IBAAM's Control layer: they define admissible behaviour and are enforced externally to
    the (untrusted) planner. Under Hierarchical and Context-Aware Invariant Governance (HIG-C, IBAAM 3 revised
    edition) each invariant has a governance :attr:`layer`, a violation degree (``InvariantResult.degree``) and a
    criticality that depends on the operational context (:meth:`criticality_in`). The evaluator compares candidate
    actions, including declining to act, by their violation profiles and selects the lexicographically least-bad
    ones.
    """

    #: Unique, human-readable invariant name used in events, reports and override records.
    name: str
    #: The behavioural concern this invariant protects (safety, integrity, identity, self-modification).
    category: InvariantCategory
    #: The lifecycle checkpoint at which this invariant is evaluated. ``Transition`` scope means every checkpoint.
    scope: InvariantScope
    #: Severity of a violation. Under the default deployment policy Error and above can never be overridden (see
    #: ``OverrideSeverityMetaInvariant``).
    severity: InvariantSeverity
    #: HIG governance level. Defaults to Fundamental, so an invariant nobody classified can never be violated
    #: autonomously.
    layer: InvariantLayer = InvariantLayer.Fundamental
    #: Base criticality: the severity of consequences if this invariant is violated in normal operation. The
    #: evaluator uses :meth:`criticality_in`, which may raise it in particular contexts.
    criticality: InvariantCriticality = InvariantCriticality.Medium
    #: Operational contexts in which violating this invariant is more critical than usual. Used by the default
    #: :meth:`criticality_in`.
    contexts: Sequence[OperationalContext] = ()

    def criticality_in(self, context: OperationalContext) -> InvariantCriticality:
        """Criticality of violating this invariant in the given operational context: C(I, κ).

        The default raises :attr:`criticality` by one level when ``context`` is listed in :attr:`contexts`. Below the
        fundamental level the default raise stops at Critical, because only fundamental invariants may carry
        Catastrophic criticality (level-criticality consistency, Section 7.3). Override this for a full criticality
        table.
        """
        if context not in self.contexts:
            return self.criticality
        ceiling = (InvariantCriticality.Catastrophic if self.layer == InvariantLayer.Fundamental
                   else InvariantCriticality.Critical)
        return self.criticality if self.criticality >= ceiling else InvariantCriticality(self.criticality + 1)

    @property
    def protected_value(self) -> str:
        """Tag for the value this invariant protects (IBAAM 6, Section 6.4.1).

        The conscience reasons from precedent using these tags ("this resembles a plan blocked for exposing restricted
        data"). Defaults to a tag derived from :attr:`category`; override to be more specific.
        """
        return ProtectedValues.for_category(self.category)

    @abstractmethod
    def evaluate(self, context: "TransitionContext") -> "InvariantResult":
        """Evaluates the invariant against the current transition (the action in flight)."""

    def evaluate_decline(self, context: "TransitionContext") -> "InvariantResult":
        """Evaluates the null action a₀: declining, holding or deferring at this checkpoint instead of continuing.

        Most invariants constrain what an action does, so declining satisfies them (the default). Invariants that
        declining itself breaks, such as a duty to answer the request, report the violation here.
        """
        from .model.control import InvariantResult

        return InvariantResult.allow()


class IMetaInvariant(ABC):
    """A meta-invariant governs *how* invariant conflicts are resolved (HIG).

    Meta-invariants are consulted for each candidate violation that might be overridden. If any meta-invariant
    rejects, the violation stays blocking. Meta-invariants may also contribute metadata (justification text, audit
    flags, review reasons) to the resulting override record.
    """

    #: Unique meta-invariant name recorded with override decisions.
    name: str
    #: The governance concern addressed (priority, severity, justification, audit, review).
    category: MetaInvariantCategory

    @abstractmethod
    def evaluate(self, context: "MetaInvariantContext") -> "MetaInvariantResult":
        """Decides whether the candidate violation in ``context`` may be overridden."""


@dataclass
class EscalationRequest:
    """A governed action that needs fundamental-level authorisation."""

    #: The transition being governed.
    context: Optional["TransitionContext"] = None
    #: The checkpoint.
    scope: InvariantScope = InvariantScope.Transition
    #: The governed action that would be carried out (``None`` if it is the decline option).
    action: Optional["AgentAction"] = None
    #: The fundamental invariants the action violates.
    fundamental_violations: list["InvariantViolation"] = field(default_factory=list)
    #: The action's violation profile.
    profile: ViolationProfile = field(default_factory=ViolationProfile)


class IEscalationProtocol(ABC):
    """A pre-authorised protocol: a rule, approved in advance by the responsible governance body, that authorises a
    governed action which violates a fundamental invariant (IBAAM 3 revised, Section 7.2).

    Without a protocol, an action that violates a fundamental invariant is never carried out autonomously: the agent
    holds at the null action and the decision is escalated for human review. Protocols are governance artefacts:
    register them in the composition root, never let the agent create or change them.
    """

    #: Protocol name recorded on the decision.
    name: str

    @abstractmethod
    def authorises(self, request: EscalationRequest) -> bool:
        """Returns ``True`` if this protocol covers the case and authorises the action."""


class IInvariantEvaluator(ABC):
    """Resolves a lifecycle checkpoint under HIG-C: evaluates the applicable invariants for each candidate action and
    selects a governed outcome."""

    @abstractmethod
    def evaluate(self, context: "TransitionContext", scope: InvariantScope,
                 alternatives: Optional[Sequence["AgentAction"]] = None) -> "InvariantEvaluationReport":
        """Resolves ``scope`` for the action in flight. The candidates are that action, any ``alternatives`` (plan
        checkpoint only, in the adaptive layer's order of preference) and the null action (declining to continue)."""


# ---------------------------------------------------------------------------------------------- conscience & storage


class IConscience(ABC):
    """Functional conscience (IBAAM 6): weighs what the agent *should* do inside the boundary the invariants (Agent
    Behavioural Controls) set on what it *may* do.

    This contract is deliberately separate from :class:`IInvariant`, so a conscience cannot be registered as an
    invariant. An invariant is a fixed, stateless predicate returning pass/fail; a conscience returns graded,
    decomposed tension and a stance, is calibrated from governance history, appraises plans against the agent's own
    commitments, and feeds back into planning (the four properties of IBAAM 6, Section 4).

    Containment: the runtime only ever calls the conscience after the relevant invariants have passed, and no stance
    has a path to an action an invariant blocks. A Proceed changes nothing downstream.
    """

    #: Implementation name recorded on each appraisal.
    name: str

    @abstractmethod
    def appraise(self, input: "ConscienceInput") -> "ConscienceAppraisal":
        """Appraises a plan (or answer). Must depend only on ``input``."""


class IValueJudge(ABC):
    """Optional language-model value judge, consulted only for plans with no close precedent (IBAAM 6, Section 6.2).

    Its estimate stands in for violation risk when history has nothing to say. The judge's output is recorded on the
    transition with the rest of the appraisal, so replay reproduces the decision even though the judge is not
    deterministic.
    """

    @abstractmethod
    def judge(self, input: "ConscienceInput") -> Optional["TensionComponent"]:
        """Estimates the risk that the plan breaches a protected value, or returns ``None`` to abstain."""


class ICommitmentStore(ABC):
    """Append-only store of conscience commitments (IBAAM 6, Section 6.4.2).

    Commitments are never edited or deleted. Which ones are live (unexpired, in scope, not superseded by a later
    reviewer decision) is computed by ``CommitmentLedger``, so the store is also the audit history.
    """

    @abstractmethod
    def append(self, commitment: "Commitment") -> None:
        """Appends a commitment."""

    @abstractmethod
    def get_all(self) -> list["Commitment"]:
        """Every commitment, oldest first."""


class ITransitionStore(ABC):
    """Persists transitions (both completed and rejected) for replay, auditing and drift analysis."""

    @abstractmethod
    def append(self, transition: "Transition") -> None:
        """Appends a finished transition to the store."""

    @abstractmethod
    def get_all(self) -> list["Transition"]:
        """Returns all stored transitions in the order they were appended."""


class IDriftBaselineStore(ABC):
    """Holds the human-approved reference state against which behavioural drift is measured.

    Approving a baseline lets an operator "accept" an evolved state so that subsequent drift is scored relative to it
    rather than to the previous transition.
    """

    @property
    @abstractmethod
    def current(self) -> Optional["DriftBaseline"]:
        """The currently approved baseline, or ``None`` if none has been approved."""

    @abstractmethod
    def approve(self, state: "AgentState", reason: str) -> "DriftBaseline":
        """Approves a snapshot of ``state`` as the new drift baseline, recording a human-readable justification."""


# ---------------------------------------------------------------------------------------------------- observability


class IDriftAnalyzer(ABC):
    """Produces an aggregate ``DriftReport`` from transition history (IBAAM Observability layer)."""

    @abstractmethod
    def analyze(self, transitions: Sequence["Transition"]) -> "DriftReport":
        """Analyses the transitions for drift, invariant failures, capability usage and stability."""


class IStabilityEvaluator(ABC):
    """Measures behavioural stability over transition history using Agent Identity Evaluation (AIE) style metrics
    (IBAAM 2). Measurement indicates change; invariants still decide acceptability."""

    @abstractmethod
    def evaluate(self, transitions: Sequence["Transition"]) -> "StabilityAssessment":
        """Computes the current stability vector, drift magnitude, region and recommendations."""

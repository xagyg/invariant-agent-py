"""IA2 Safety: the concrete invariant set (Agent Behavioural Controls).

Grouped as in the C# project: Action, Control, Identity, Mission, Outcome, Runtime and SelfModification invariants.
Invariants that do not declare a layer are Fundamental (the :class:`IInvariant` default), so nothing unclassified can
ever be violated autonomously.
"""

from __future__ import annotations

from typing import Iterable

from .core._util import is_blank, lower_set
from .core.abstractions import IInvariant
from .core.drift import BehaviouralDriftDetector
from .core.model.control import InvariantResult, ProtectedValues
from .core.model.enums import (
    InvariantCategory,
    InvariantCriticality,
    InvariantLayer,
    InvariantScope,
    InvariantSeverity,
    TransitionPhase,
    TransitionStatus,
)
from .core.model.transition import TransitionContext
from .core.parsing import AgentActionParser
from .core.transitioning import TransitionLifecycle

# ----------------------------------------------------------------------------------------------------------- action


class AllowedCapabilityInvariant(IInvariant):
    """Plan-scope safety invariant: the proposed capability must be on the allow-list.

    The allow-list is seeded from the capability registry, so this also rejects unknown capabilities and planner error
    actions before any execution takes place.
    """

    name = "AllowedCapabilityInvariant"
    category = InvariantCategory.Safety
    scope = InvariantScope.Plan
    severity = InvariantSeverity.Error

    def __init__(self, allowed_capabilities: Iterable[str]) -> None:
        # Case-insensitive to match CapabilityRegistry, so LLM planners that return "Calc" are not rejected for a
        # capability the registry would resolve.
        self._allowed = lower_set(allowed_capabilities)

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        action = context.transition.proposed_action
        if action is None:
            return InvariantResult.reject("No proposed action.", self.severity)
        if is_blank(action.capability):
            return InvariantResult.reject("No capability selected.", self.severity)
        if action.capability.lower() in self._allowed:
            return InvariantResult.allow()
        return InvariantResult.reject(f"Capability '{action.capability}' is not allowed or unknown.", self.severity)


class CapabilityMustExistInvariant(IInvariant):
    """Plan-scope integrity invariant: a capability must have been selected. Not registered by default
    (:class:`AllowedCapabilityInvariant` covers this case)."""

    name = "CapabilityMustExistInvariant"
    category = InvariantCategory.Integrity
    scope = InvariantScope.Plan
    severity = InvariantSeverity.Error

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        action = context.transition.proposed_action
        if action is None:
            return InvariantResult.reject("No proposed action.")
        if is_blank(action.capability):
            return InvariantResult.reject("No tool selected.")
        return InvariantResult.allow()


class NoDeleteInvariant(IInvariant):
    """Plan-scope safety invariant: forbids the ``delete`` capability. An example hard safety constraint."""

    name = "NoDeleteInvariant"
    category = InvariantCategory.Safety
    scope = InvariantScope.Plan
    severity = InvariantSeverity.Error
    #: Protects stored data from destruction.
    protected_value = "data-preservation"

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        action = context.transition.proposed_action
        if action is None:
            return InvariantResult.reject("No proposed action.")
        if action.capability == "delete":
            return InvariantResult.reject("Delete operations are not allowed.")
        return InvariantResult.allow()


class NonEmptyInputInvariant(IInvariant):
    """Plan-scope integrity invariant: the proposed action must have non-empty input. Not registered by default."""

    name = "NonEmptyInputInvariant"
    category = InvariantCategory.Integrity
    scope = InvariantScope.Plan
    severity = InvariantSeverity.Error

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        action = context.transition.proposed_action
        if action is None:
            return InvariantResult.reject("No proposed action.")
        if is_blank(action.input):
            return InvariantResult.reject("Empty input not allowed.")
        return InvariantResult.allow()


# ---------------------------------------------------------------------------------------------------------- control


class OperatorOnlyCapabilityInvariant(IInvariant):
    """Plan-scope control-integrity invariant: an operator-only capability (``ICapability.operator_only``) may run
    only when the operator typed exactly that command.

    IBAAM 1 (Section 4.3) requires that the control layer "must not be freely modifiable by the agent itself".
    Resolving conscience escalations, re-certifying the drift baseline and changing the HIG-C operational context are
    control actions. If a planner could propose them, an LLM planner could overturn its own escalations or move the
    baseline its drift is scored against, and the reviewer signal IBAAM 6 relies on (Section 5.2) would no longer be
    independent.

    The check is structural, not a prompt instruction: the proposed action must equal the literal parse of the raw
    input. With the command planner that is always true, so the REPL works as before. With an LLM planner the operator
    can still type the command verbatim, but a model that proposes one on its own (from a paraphrase, an injected
    instruction or a revision) is blocked. The decision depends only on the recorded input and action, so replay
    reproduces it.
    """

    name = "OperatorOnlyCapabilityInvariant"
    category = InvariantCategory.Safety
    scope = InvariantScope.Plan
    severity = InvariantSeverity.Critical
    protected_value = ProtectedValues.ControlIntegrity

    def __init__(self, operator_only_capabilities: Iterable[str]) -> None:
        self._operator_only_names = list(operator_only_capabilities)
        self._operator_only = lower_set(self._operator_only_names)

    @property
    def operator_only_capabilities(self) -> list[str]:
        """The operator-only capability names this invariant guards."""
        return list(self._operator_only_names)

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        transition = context.transition
        action = transition.proposed_action
        if action is None or is_blank(action.capability) or action.capability.strip().lower() not in self._operator_only:
            return InvariantResult.allow()
        if self._is_operator_authored(transition.input, action.capability, action.input):
            return InvariantResult.allow()
        return InvariantResult.reject(
            f"Capability '{action.capability}' is operator-only and the planner proposed it; "
            "the operator must type the command directly.",
            self.severity)

    @staticmethod
    def _is_operator_authored(raw_input: str, capability: str, input: str) -> bool:
        # The operator authored the action when it is exactly what they typed (capability case-insensitive, input
        # verbatim apart from surrounding whitespace).
        if is_blank(raw_input):
            return False
        typed = AgentActionParser.parse(raw_input)
        return ((typed.capability or "").lower() == capability.strip().lower()
                and (typed.input or "").strip() == (input or "").strip())


# --------------------------------------------------------------------------------------------------------- identity


class BehaviouralDriftBoundInvariant(IInvariant):
    """Reduction-scope identity invariant: rejects a proposed next state whose behavioural drift score reaches the
    configured bound.

    This is where measurement becomes governance: drift is scored against the approved baseline (or the prior state)
    before the new state is committed, bounding how far the agent may evolve in one step.
    """

    name = "BehaviouralDriftBoundInvariant"
    category = InvariantCategory.Identity
    scope = InvariantScope.Reduction
    severity = InvariantSeverity.Error

    def __init__(self, detector: BehaviouralDriftDetector,
                 max_score: int = BehaviouralDriftDetector.DEFAULT_BOUND_THRESHOLD) -> None:
        self._detector = detector
        self._max_score = max_score

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        transition = context.transition
        if transition.after is None:
            return InvariantResult.allow()
        # Use the baseline recorded on the transition so replay reproduces the live decision.
        score = self._detector.score_as_recorded(transition)
        if score.score < self._max_score:
            return InvariantResult.allow()
        return InvariantResult.reject(
            f"Behavioural drift score {score.score} exceeds bound {self._max_score}: {score.explanation}",
            self.severity)


# ---------------------------------------------------------------------------------------------------------- mission


class ServiceObligationInvariant(IInvariant):
    """Mission-level duty to act on the user's request. Only the null action violates it: declining, holding or
    deferring leaves the request unanswered.

    This is what gives declining a cost under HIG-C (IBAAM 3 revised, Section 8.1). Without it, declining satisfies
    every invariant, so any violation at all makes declining the governed outcome and no override is ever possible
    (pure enforcement). With it, an action that violates only lower-level obligations (behavioural or heuristic) is
    better than declining and may proceed with witnessed overrides; an action that violates a mission obligation more
    critically than this one, or anything fundamental, still loses to declining.

    Severity is Warning: declining is always a permitted, safe hold, so its cost never needs authorising.
    """

    name = "ServiceObligationInvariant"
    category = InvariantCategory.Integrity
    #: Evaluated at every checkpoint: declining at any of them leaves the request unanswered.
    scope = InvariantScope.Transition
    severity = InvariantSeverity.Warning
    layer = InvariantLayer.Mission
    protected_value = "service"

    def __init__(self, criticality: InvariantCriticality = InvariantCriticality.Medium) -> None:
        """:param criticality: how critical leaving a request unanswered is (default Medium)."""
        self.criticality = criticality

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        """Acting on the request satisfies the obligation."""
        return InvariantResult.allow()

    def evaluate_decline(self, context: TransitionContext) -> InvariantResult:
        """Declining leaves the request unanswered."""
        return InvariantResult.reject("Declining leaves the user's request unanswered.", self.severity)


# ---------------------------------------------------------------------------------------------------------- outcome


class NonEmptyOutcomeInvariant(IInvariant):
    """Execution-scope integrity invariant: a successful execution outcome must have a non-empty result.

    Failed outcomes are out of scope: they carry no result by design and are already rejected by
    :class:`SuccessOutcomeInvariant`, so rejecting them here would report the same failure twice.
    """

    name = "NonEmptyOutcomeInvariant"
    category = InvariantCategory.Integrity
    scope = InvariantScope.Execution
    severity = InvariantSeverity.Error

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        outcome = context.transition.outcome
        if outcome is None:
            return InvariantResult.reject("No execution outcome.")
        # Not applicable to failed executions; SuccessOutcomeInvariant reports those.
        if not outcome.success:
            return InvariantResult(passed=True, reason="Not applicable: execution failed.")
        if is_blank(outcome.result):
            return InvariantResult.reject("Outcome is empty.")
        return InvariantResult.allow()


class SuccessOutcomeInvariant(IInvariant):
    """Execution-scope integrity invariant: the capability must report success. Failed executions are rejected, so
    they never reach state assimilation."""

    name = "SuccessOutcomeInvariant"
    category = InvariantCategory.Integrity
    scope = InvariantScope.Execution
    severity = InvariantSeverity.Error

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        outcome = context.transition.outcome
        if outcome is None:
            return InvariantResult.reject("No execution outcome.")
        if not outcome.success:
            return InvariantResult.reject("Execution failed." if is_blank(outcome.error) else outcome.error)
        return InvariantResult.allow()


# ---------------------------------------------------------------------------------------------------------- runtime


class ExecutionRequiresPlanValidationInvariant(IInvariant):
    """Execution-scope lifecycle invariant: execution must not occur unless the transition passed through plan
    validation. Not registered by default."""

    name = "ExecutionRequiresPlanValidationInvariant"
    category = InvariantCategory.Integrity
    scope = InvariantScope.Execution
    severity = InvariantSeverity.Critical

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        if not TransitionLifecycle.has_reached(context.transition, TransitionPhase.PlanValidation):
            return InvariantResult.reject("Execution requires successful plan validation.", self.severity)
        return InvariantResult.allow()


class RejectedTransitionCannotCommitInvariant(IInvariant):
    """Reduction-scope lifecycle invariant: a rejected transition must not carry a proposed next state.

    Registered by default as a backstop alongside the runtime's own rejection handling.
    """

    name = "RejectedTransitionCannotCommitInvariant"
    category = InvariantCategory.Integrity
    scope = InvariantScope.Reduction
    severity = InvariantSeverity.Critical

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        transition = context.transition
        if transition.status != TransitionStatus.Rejected:
            return InvariantResult.allow()
        if transition.after is not None:
            return InvariantResult.reject("Rejected transitions cannot produce committed state.", self.severity)
        return InvariantResult.allow()


# -------------------------------------------------------------------------------------------------- self-modification


class AllowedMemoryKeyInvariant(IInvariant):
    """Self-modification invariant: memory updates may only target allow-listed keys.

    Bounds the agent's ability to rewrite its own memory (governed self-modification).
    """

    name = "AllowedMemoryKeyInvariant"
    scope = InvariantScope.SelfModification
    severity = InvariantSeverity.Error
    category = InvariantCategory.SelfModification
    #: Protects the integrity of the agent's own memory.
    protected_value = "memory-integrity"

    #: Memory keys that capabilities may propose to change (case-insensitive).
    ALLOWED_KEYS = frozenset({"goal", "user_intent", "last_action", "notes"})

    def evaluate(self, context: TransitionContext) -> InvariantResult:
        modification = context.transition.self_modification
        if modification is None or modification.target != "memory":
            return InvariantResult.allow()
        if (modification.key or "").lower() in self.ALLOWED_KEYS:
            return InvariantResult.allow()
        return InvariantResult.reject(f"Memory key '{modification.key}' is not allowed.")


__all__ = [
    "AllowedCapabilityInvariant",
    "AllowedMemoryKeyInvariant",
    "BehaviouralDriftBoundInvariant",
    "CapabilityMustExistInvariant",
    "ExecutionRequiresPlanValidationInvariant",
    "NoDeleteInvariant",
    "NonEmptyInputInvariant",
    "NonEmptyOutcomeInvariant",
    "OperatorOnlyCapabilityInvariant",
    "RejectedTransitionCannotCommitInvariant",
    "ServiceObligationInvariant",
    "SuccessOutcomeInvariant",
]

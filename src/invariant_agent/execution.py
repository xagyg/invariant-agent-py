"""IA2 Execution layer: capability execution and mediation."""

from __future__ import annotations

from .core.abstractions import ICapabilityRegistry, IExecutor
from .core.model.agent import AgentAction, AgentOutcome
from .core.model.capability import CapabilityRequest, CapabilityResult
from .core.model.enums import TransitionEventStage, TransitionStatus
from .core.model.transition import TransitionContext


class CapabilityExecutor(IExecutor):
    """Default :class:`IExecutor` (IBAAM Execution layer). Resolves the proposed capability, executes it and records
    the outcome plus any proposed self-modification on the transition.

    The executor mediates capability access but never applies self-modifications; those are evaluated by invariants
    and assimilated by the reducer. Exceptions raised by capabilities are converted into failed results, so the
    transition is rejected by execution invariants and still persisted instead of crashing the runtime.
    """

    def __init__(self, capabilities: ICapabilityRegistry) -> None:
        self._capabilities = capabilities

    def execute(self, context: TransitionContext) -> None:
        transition = context.transition
        action = transition.proposed_action
        if action is None:
            transition.status = TransitionStatus.Failed
            transition.outcome = AgentOutcome(success=False, error="No proposed action.")
            return
        result = self._execute_capability(action)
        transition.outcome = AgentOutcome(
            success=result.success,
            capability=result.capability,
            result=(str(result.data) if result.data is not None else "") if result.success else "",
            error=result.error,
        )
        # Record the proposal only; SelfModification invariants decide whether it is applied.
        transition.self_modification = result.proposed_modification
        transition.status = TransitionStatus.Executed
        modification = transition.self_modification
        if modification is not None:
            transition.add_event(TransitionEventStage.SelfModification,
                                 f"{modification.target}.{modification.operation} {modification.key}")

    # Resolves and invokes the capability. An unknown capability or a capability that raises yields a failed result
    # rather than an exception.
    def _execute_capability(self, action: AgentAction) -> CapabilityResult:
        capability = self._capabilities.get(action.capability or "")
        if capability is None:
            return CapabilityResult(success=False, capability=action.capability or "",
                                    error=f"Unknown capability '{action.capability}'.")
        try:
            return capability.execute(CapabilityRequest(input=action.input if action.input is not None else ""))
        except Exception as ex:  # noqa: BLE001 - every capability failure is contained
            return CapabilityResult.fail(action.capability, f"Capability '{action.capability}' failed: {ex}")

"""Pipeline stages around the planner and the reducer: the bounded projection ``π(Sₜ)`` and state assimilation."""

from __future__ import annotations

from typing import Any, Optional

from ._util import cs_str
from .abstractions import IStateReducer
from .model.agent import AgentState, StateProjection
from .model.enums import TransitionEventStage, TransitionPhase, TransitionStatus
from .model.planning import PlannerContext, PlannerRevision
from .model.transition import Transition, TransitionContext


class StateProjector:
    """Produces the bounded :class:`StateProjection` ``π(Sₜ)`` that planners operate on."""

    # Only these memory keys are summarised, so planners cannot see arbitrary memory.
    IMPORTANT_KEYS = frozenset({"goal", "user_intent", "last_action", "lastOutcome", "error_state"})

    @staticmethod
    def project(state: AgentState) -> StateProjection:
        """Creates a read-only, summarised view of the state."""
        return StateProjection(
            version=state.version,
            goal=state.goal or "",
            memory_summary=StateProjector._summarise_memory(state.memory),
            last_outcome=cs_str(state.memory.get("lastOutcome")) if "lastOutcome" in state.memory else "",
            active_policies=list(state.policies or []),
        )

    @staticmethod
    def _summarise_memory(memory: Optional[dict[str, Any]]) -> str:
        # Summarises only an allow-list of memory keys (max 10).
        if not memory:
            return ""
        prioritized = [(k, v) for k, v in memory.items() if k in StateProjector.IMPORTANT_KEYS][:10]
        return "; ".join(f"{k}={cs_str(v)}" for k, v in prioritized)


class PlannerContextProjector:
    """Builds the :class:`PlannerContext` handed to planners, enforcing the bounded state projection ``π(Sₜ)``
    rather than exposing full runtime state."""

    @staticmethod
    def project(state: AgentState, input: str, revision: Optional[PlannerRevision] = None) -> PlannerContext:
        """Projects ``state`` and ``input`` into a planner context exposing only whitelisted memory (``goal``,
        ``lastOutcome``), attaching conscience feedback for a revised plan when ``revision`` is given."""
        memory = state.memory
        return PlannerContext(
            input=input,
            state=StateProjector.project(state),
            goal=cs_str(memory.get("goal")) if "goal" in memory else "",
            last_outcome=cs_str(memory.get("lastOutcome")) if "lastOutcome" in memory else "",
            relevant_memory={k: v for k, v in memory.items() if k in ("goal", "lastOutcome")},
            revision=revision,
        )


class StateReducer(IStateReducer):
    """Default :class:`IStateReducer`: assimilates an approved transition into a new proposed state.

    The reducer:

    * runs only in the Reduction phase and never for rejected transitions;
    * creates a new state with ``version + 1``, copying prior memory;
    * records ``lastOutcome`` for successful, non-introspective capabilities;
    * applies an approved ``memory.set`` self-modification and records a SelfModification event.

    The resulting ``after`` state is a proposal: the runtime commits it only after reduction invariants (e.g. the
    behavioural drift bound) pass.
    """

    # Introspective and governance tools are excluded so that observing the agent does not itself change its
    # adaptive state.
    INTROSPECTIVE_CAPABILITIES = frozenset({
        "explain", "drift", "replay", "memory-show", "baseline-approve", "baseline-show",
        "conscience", "conscience-show", "conscience-review",
    })

    def apply(self, context: TransitionContext) -> None:
        transition = context.transition
        # Reject if the transition is in the wrong phase or state.
        if transition.phase != TransitionPhase.Reduction or transition.status == TransitionStatus.Rejected:
            return
        before = transition.before
        # Build a fresh state rather than mutating Before, so the prior state stays intact for audit. Mode, Goal and
        # Policies are carried forward unchanged; Policies is copied so the two states never share a mutable list.
        after = AgentState(
            version=before.version + 1,
            mode=before.mode,
            goal=before.goal,
            policies=list(before.policies or []),
        )
        after.memory.update(before.memory)  # shallow copy for now
        if self._should_store_last_outcome(transition):
            after.memory["lastOutcome"] = transition.outcome.result
        transition.after = after
        modification = transition.self_modification
        # Self-modification has already passed SelfModification-scope invariants at this point.
        if modification is not None and modification.target == "memory" and modification.operation == "set":
            old_value = after.memory.get(modification.key)
            after.memory[modification.key] = modification.value
            transition.add_event(TransitionEventStage.SelfModification, f"Set {modification.key}", {
                # Later: "Policy", "Planner"
                "Target": "Memory",
                "Operation": modification.operation,
                "Key": modification.key,
                "OldValue": old_value,
                "NewValue": modification.value,
            })
        transition.status = TransitionStatus.Completed

    @classmethod
    def _should_store_last_outcome(cls, transition: Transition) -> bool:
        """``True`` for successful outcomes of capabilities that act on the world."""
        capability = transition.proposed_action.capability if transition.proposed_action else None
        return (transition.outcome is not None and transition.outcome.success
                and capability not in cls.INTROSPECTIVE_CAPABILITIES)

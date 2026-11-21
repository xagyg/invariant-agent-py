"""The transition lifecycle state machine and phase comparisons."""

from __future__ import annotations

from .model.enums import TransitionEventStage, TransitionPhase, TransitionStatus
from .model.transition import Transition

P = TransitionPhase


class TransitionLifecycle:
    """Helpers for comparing a transition's current phase against the ordered :class:`TransitionPhase` sequence."""

    @staticmethod
    def has_reached(transition: Transition, phase: TransitionPhase) -> bool:
        """Returns ``True`` if the transition is at or beyond ``phase``."""
        return transition.phase >= phase

    @staticmethod
    def is_before(transition: Transition, phase: TransitionPhase) -> bool:
        """Returns ``True`` if the transition has not yet reached ``phase``."""
        return transition.phase < phase

    @staticmethod
    def is_after(transition: Transition, phase: TransitionPhase) -> bool:
        """Returns ``True`` if the transition is strictly beyond ``phase``."""
        return transition.phase > phase


class TransitionPhases:
    """Lifecycle state machine for transitions. Enforces legal phase moves and records a Lifecycle event for each
    move."""

    # Legal next phases from each phase. Every non-terminal phase may move to Rejected; ExecutionValidation may skip
    # straight to Reduction; Completed and Rejected are terminal.
    ALLOWED_MOVES: dict[TransitionPhase, tuple[TransitionPhase, ...]] = {
        P.Created: (P.InputReceived, P.Rejected),
        P.InputReceived: (P.Planning, P.Rejected),
        P.Planning: (P.PlanValidation, P.Rejected),
        P.PlanValidation: (P.Execution, P.Rejected),
        P.Execution: (P.ExecutionValidation, P.Rejected),
        P.ExecutionValidation: (P.SelfModificationValidation, P.Reduction, P.Rejected),
        P.SelfModificationValidation: (P.Reduction, P.Rejected),
        P.Reduction: (P.Completed, P.Rejected),
        P.Completed: (),
        P.Rejected: (),
    }

    @staticmethod
    def move_to(transition: Transition, next_phase: TransitionPhase) -> None:
        """Moves the transition to ``next_phase`` and records a Lifecycle event.

        :raises RuntimeError: the move is not allowed from the current phase (C# ``InvalidOperationException``).
        """
        current = transition.phase
        if next_phase not in TransitionPhases.ALLOWED_MOVES.get(current, ()):
            raise RuntimeError(f"Illegal transition phase move: {current} -> {next_phase}")
        transition.phase = next_phase
        transition.add_event(TransitionEventStage.Lifecycle, f"Phase: {next_phase}",
                             {"PreviousPhase": current.name, "Phase": next_phase.name})

    @staticmethod
    def reject(transition: Transition, reason: str) -> None:
        """Moves the transition to Rejected and records the governance summary explaining the rejection."""
        TransitionPhases.move_to(transition, TransitionPhase.Rejected)
        transition.status = TransitionStatus.Rejected
        transition.reason = reason

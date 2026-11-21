"""IA2 Core: contracts, models, transitions, projections, HIG-C control, drift, replay and rendering."""

from . import abstractions, model
from .conscience import CommitmentLedger, InMemoryCommitmentStore, PlanSimilarity
from .control import InvariantEvaluator, OperationalContextState
from .drift import BehaviouralDriftDetector, DriftTracker, InMemoryDriftBaselineStore
from .parsing import AgentActionParser
from .pipeline import PlannerContextProjector, StateProjector, StateReducer
from .rendering import TransitionFormatter
from .replay import ReplayComparison, ReplayValidationResult, ReplayValidator
from .serialization import clone
from .transitioning import TransitionLifecycle, TransitionPhases

__all__ = [
    "AgentActionParser",
    "BehaviouralDriftDetector",
    "CommitmentLedger",
    "DriftTracker",
    "InMemoryCommitmentStore",
    "InMemoryDriftBaselineStore",
    "InvariantEvaluator",
    "OperationalContextState",
    "PlanSimilarity",
    "PlannerContextProjector",
    "ReplayComparison",
    "ReplayValidationResult",
    "ReplayValidator",
    "StateProjector",
    "StateReducer",
    "TransitionFormatter",
    "TransitionLifecycle",
    "TransitionPhases",
    "abstractions",
    "clone",
    "model",
]

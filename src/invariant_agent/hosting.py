"""Composition root: wires every InvariantAgent component together (the C# ``AddInvariantAgent`` DI registration).

:func:`build_agent` returns an :class:`InvariantAgent` holding the runtime and every service it was built from, so
callers (and tests) can inspect the stores, the analyzer, the replay validator and so on. Keyword overrides replace
individual components, the way the C# tests re-register services in the container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from .adaptive import CommandPlanner
from .capabilities import (
    BaselineApproveTool,
    BaselineShowTool,
    CalculatorTool,
    CapabilityRegistry,
    ConscienceReviewTool,
    ConscienceShowTool,
    ContextTool,
    DriftTool,
    EchoTool,
    ExampleHttpService,
    ExplainTool,
    MemorySetTool,
    MemoryShowTool,
    ReplayTool,
    SearchTool,
)
from .conscience import ConscienceCoordinator, ConscienceOptions, ConscienceSession, HistoryCalibratedConscience
from .core.abstractions import (
    ICapability,
    ICommitmentStore,
    IConscience,
    IDriftAnalyzer,
    IDriftBaselineStore,
    IEscalationProtocol,
    IExecutor,
    IInvariant,
    IInvariantEvaluator,
    IMetaInvariant,
    IPlanner,
    IStabilityEvaluator,
    IStateReducer,
    ITransitionStore,
    IValueJudge,
)
from .core.conscience import InMemoryCommitmentStore
from .core.control import InvariantEvaluator, OperationalContextState
from .core.control.meta_invariants import (
    AuditMetaInvariant,
    JustificationMetaInvariant,
    OverrideSeverityMetaInvariant,
    PriorityMetaInvariant,
    ReviewMetaInvariant,
)
from .core.drift import BehaviouralDriftDetector, DriftTracker, InMemoryDriftBaselineStore
from .core.model.enums import OperationalContext
from .core.pipeline import StateReducer
from .core.replay import ReplayValidator
from .execution import CapabilityExecutor
from .observability import SimpleDriftAnalyzer, TransitionHistoryStabilityEvaluator
from .runtime import GovernedAgentRuntime
from .safety import (
    AllowedCapabilityInvariant,
    AllowedMemoryKeyInvariant,
    BehaviouralDriftBoundInvariant,
    NoDeleteInvariant,
    NonEmptyOutcomeInvariant,
    OperatorOnlyCapabilityInvariant,
    RejectedTransitionCannotCommitInvariant,
    ServiceObligationInvariant,
    SuccessOutcomeInvariant,
)
from .storage import FileCommitmentStore, FileDriftBaselineStore, FileTransitionStore, InMemoryTransitionStore

#: File name of the transition history inside the data directory.
TRANSITIONS_FILE_NAME = "transitions.jsonl"
#: File name of the drift baseline approval history inside the data directory.
BASELINES_FILE_NAME = "baselines.jsonl"
#: File name of the conscience commitment history inside the data directory.
COMMITMENTS_FILE_NAME = "commitments.jsonl"


@dataclass
class InvariantAgent:
    """A fully wired agent: the runtime plus every service it was composed from."""

    runtime: GovernedAgentRuntime
    transition_store: ITransitionStore
    baseline_store: IDriftBaselineStore
    commitment_store: ICommitmentStore
    drift_tracker: DriftTracker
    drift_detector: BehaviouralDriftDetector
    stability_evaluator: IStabilityEvaluator
    drift_analyzer: IDriftAnalyzer
    replay_validator: ReplayValidator
    evaluator: IInvariantEvaluator
    invariants: list[IInvariant]
    meta_invariants: list[IMetaInvariant]
    protocols: list[IEscalationProtocol]
    capabilities: list[ICapability]
    registry: CapabilityRegistry
    executor: IExecutor
    reducer: IStateReducer
    planner: IPlanner
    conscience_options: ConscienceOptions
    session: ConscienceSession
    operational_context: OperationalContextState
    conscience: Optional[IConscience] = None
    coordinator: Optional[ConscienceCoordinator] = None
    data_directory: Optional[str] = field(default=None)

    def run(self, input: str):
        """Shortcut for ``runtime.run(input)``."""
        return self.runtime.run(input)


def build_agent(
    data_directory: Optional[str | os.PathLike] = None,
    configure_conscience: Optional[Callable[[ConscienceOptions], None]] = None,
    operational_context: OperationalContext = OperationalContext.Normal,
    *,
    planner: Optional[IPlanner] = None,
    reducer: Optional[IStateReducer] = None,
    transition_store: Optional[ITransitionStore] = None,
    baseline_store: Optional[IDriftBaselineStore] = None,
    commitment_store: Optional[ICommitmentStore] = None,
    remove_invariants: Iterable[type] = (),
    extra_invariants: Sequence[IInvariant] = (),
    extra_capabilities: Sequence[ICapability] = (),
    protocols: Sequence[IEscalationProtocol] = (),
    value_judge: Optional[IValueJudge] = None,
) -> InvariantAgent:
    """Builds the storage, observability, planner, executor, reducer, meta-invariants, invariants, conscience,
    capabilities and the governed runtime.

    :param data_directory: directory for persistent storage. When set, transitions, drift baselines and conscience
        commitments are kept in ``transitions.jsonl``, ``baselines.jsonl`` and ``commitments.jsonl`` there and survive
        restarts (the runtime resumes from the last committed state). When ``None``, everything is in memory.
    :param configure_conscience: configures the functional conscience (IBAAM 6). It is enabled by default; set
        ``ConscienceOptions.enabled`` to false to run with invariants only.
    :param operational_context: HIG-C operational context in force at start-up; an operator changes it at run time
        with the ``context`` command.
    :param planner: replaces the default :class:`CommandPlanner` (e.g. an :class:`~invariant_agent.adaptive.OpenAiPlanner`).
    :param reducer: replaces the default :class:`StateReducer`.
    :param transition_store: replaces the transition store chosen from ``data_directory``.
    :param baseline_store: replaces the drift baseline store chosen from ``data_directory``.
    :param commitment_store: replaces the commitment store chosen from ``data_directory``.
    :param remove_invariants: invariant types to leave out (e.g. for leave-one-out experiments).
    :param extra_invariants: further invariants, evaluated after the standard set.
    :param extra_capabilities: further capabilities, registered after the standard set.
    :param protocols: pre-authorised protocols for fundamental-level conflicts (Section 7.2). None by default, so a
        conflict between fundamental invariants is always held for human review. Register protocols only with the
        approval of the responsible governance body.
    :param value_judge: optional language-model value judge consulted only for plans with no close precedent.
    """
    options = ConscienceOptions()
    if configure_conscience is not None:
        configure_conscience(options)
    directory = None if data_directory is None or str(data_directory).strip() == "" else str(data_directory)

    # STORAGE: transition history, approved drift baselines and conscience commitments.
    if directory is None:
        transitions = transition_store or InMemoryTransitionStore()
        baselines = baseline_store or InMemoryDriftBaselineStore()
        commitments = commitment_store or InMemoryCommitmentStore()
    else:
        transitions = transition_store or FileTransitionStore(os.path.join(directory, TRANSITIONS_FILE_NAME))
        baselines = baseline_store or FileDriftBaselineStore(os.path.join(directory, BASELINES_FILE_NAME))
        commitments = commitment_store or FileCommitmentStore(os.path.join(directory, COMMITMENTS_FILE_NAME))

    # OBSERVABILITY: drift analysis and behavioural stability evaluation.
    detector = BehaviouralDriftDetector(baselines)
    tracker = DriftTracker()
    stability = TransitionHistoryStabilityEvaluator(detector)
    analyzer = SimpleDriftAnalyzer(tracker, detector, stability)

    # HIG-C OPERATIONAL CONTEXT: stamped on every transition; changed only by an operator (the context capability).
    context_state = OperationalContextState(
        operational_context,
        "default" if operational_context == OperationalContext.Normal else "configured at start-up")

    # HIG-C META-INVARIANTS (IBAAM 3 revised, Section 7): fundamental violations need escalation (Priority); hard
    # severities stay blocking (deployment policy); overrides must be witnessed (Justification); every override is
    # audited; ambiguous ones are flagged for review.
    meta_invariants: list[IMetaInvariant] = [
        PriorityMetaInvariant(),
        OverrideSeverityMetaInvariant(),
        JustificationMetaInvariant(),
        AuditMetaInvariant(),
        ReviewMetaInvariant(),
    ]

    # CONSCIENCE (IBAAM 6) settings and session; the session exists even when the conscience is disabled.
    session = ConscienceSession()

    # CAPABILITIES: user-facing tools followed by internal introspection tools. The replay tool resolves its validator
    # lazily, because the validator depends on the invariants, which depend on the registry.
    holder: dict[str, ReplayValidator] = {}
    capabilities: list[ICapability] = [
        EchoTool(),
        CalculatorTool(),
        SearchTool(),
        ReplayTool(transitions, lambda: holder["validator"]),
        DriftTool(transitions, analyzer),
        ExampleHttpService(),
        MemorySetTool(),
        MemoryShowTool(transitions),
        ExplainTool(transitions),
        BaselineApproveTool(transitions, baselines),
        BaselineShowTool(baselines),
        ConscienceShowTool(transitions, commitments, options, session),
        ConscienceReviewTool(transitions, commitments, options),
        ContextTool(context_state),
        *extra_capabilities,
    ]
    registry = CapabilityRegistry(capabilities)

    # CONTROL: the active invariant set, in evaluation order. Each invariant declares its own scope (lifecycle
    # checkpoint). Registered individually so the conscience can read the value each one protects.
    operator_only = [n for n in registry.get_capability_names() if getattr(registry.get(n), "operator_only", False)]
    invariants: list[IInvariant] = [
        NoDeleteInvariant(),
        AllowedCapabilityInvariant(registry.get_capability_names()),
        SuccessOutcomeInvariant(),
        AllowedMemoryKeyInvariant(),
        NonEmptyOutcomeInvariant(),
        BehaviouralDriftBoundInvariant(detector),
        # Control integrity: governance capabilities (review, baseline, context) run only when the operator typed them.
        OperatorOnlyCapabilityInvariant(operator_only),
        # Backstop: a rejected transition must never carry state into a commit.
        RejectedTransitionCannotCommitInvariant(),
        # HIG-C: the duty to act on the request gives declining its cost (IBAAM 3 revised, Section 8.1). Without it,
        # declining satisfies everything and any violation at all rejects the transition.
        ServiceObligationInvariant(),
        *extra_invariants,
    ]
    removed = tuple(remove_invariants)
    if removed:
        invariants = [i for i in invariants if type(i) not in removed]

    evaluator = InvariantEvaluator(invariants, meta_invariants, list(protocols))
    validator = ReplayValidator(evaluator, tracker)
    holder["validator"] = validator

    # CONSCIENCE: weighs what the agent should do inside what the invariants say it may do. Deliberately not an
    # IInvariant: it has its own contract and runs only after the invariants pass.
    conscience: Optional[IConscience] = None
    coordinator: Optional[ConscienceCoordinator] = None
    if options.enabled:
        conscience = HistoryCalibratedConscience(options, value_judge)
        coordinator = ConscienceCoordinator(conscience, transitions, commitments, stability, detector, invariants,
                                            options, session)

    # EXECUTION and STATE ASSIMILATION; ADAPTIVE: swap in OpenAiPlanner/GeminiPlanner to use an LLM planner.
    executor = CapabilityExecutor(registry)
    reducer = reducer or StateReducer()
    planner = planner or CommandPlanner()

    # RUNTIME: orchestrates the governed transition lifecycle.
    runtime = GovernedAgentRuntime(planner, evaluator, executor, reducer, transitions, baselines, coordinator,
                                   context_state)
    return InvariantAgent(
        runtime=runtime,
        transition_store=transitions,
        baseline_store=baselines,
        commitment_store=commitments,
        drift_tracker=tracker,
        drift_detector=detector,
        stability_evaluator=stability,
        drift_analyzer=analyzer,
        replay_validator=validator,
        evaluator=evaluator,
        invariants=invariants,
        meta_invariants=meta_invariants,
        protocols=list(protocols),
        capabilities=capabilities,
        registry=registry,
        executor=executor,
        reducer=reducer,
        planner=planner,
        conscience_options=options,
        session=session,
        operational_context=context_state,
        conscience=conscience,
        coordinator=coordinator,
        data_directory=directory,
    )


__all__ = [
    "BASELINES_FILE_NAME",
    "COMMITMENTS_FILE_NAME",
    "InvariantAgent",
    "TRANSITIONS_FILE_NAME",
    "build_agent",
]

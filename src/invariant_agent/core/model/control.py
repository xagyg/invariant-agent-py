"""Control-layer model: invariant results, violations, HIG-C violation profiles, resolutions and overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .._util import clamp, fmt_max
from .agent import AgentAction
from .enums import (
    ContextualGovernanceOutcome,
    InvariantCategory,
    InvariantCriticality,
    InvariantLayer,
    InvariantScope,
    InvariantSeverity,
    MetaInvariantCategory,
    OperationalContext,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..abstractions import IInvariant
    from .transition import TransitionContext


# ------------------------------------------------------------------------------------------------- protected values


class ProtectedValues:
    """Protected-value tags shared by invariants and conscience commitments (IBAAM 6, Section 6.4)."""

    #: Stability of the agent's behavioural identity (drift).
    IdentityStability = "identity-stability"
    #: Staying inside the controls: used when the concern is anticipated rejection with no more specific tag.
    ControlBoundary = "control-boundary"
    #: Consistency with the agent's own prior commitments.
    Consistency = "consistency"
    #: The control layer stays outside the agent's reach (IBAAM 1, Section 4.3).
    ControlIntegrity = "control-integrity"

    @staticmethod
    def for_category(category: InvariantCategory) -> str:
        """Default tag for an invariant category."""
        return {
            InvariantCategory.Safety: "safety",
            InvariantCategory.Integrity: "integrity",
            InvariantCategory.Identity: ProtectedValues.IdentityStability,
            InvariantCategory.SelfModification: "self-modification",
        }.get(category, category.name.lower())


# ----------------------------------------------------------------------------------------------------------- results


class InvariantResult:
    """The result of evaluating a single invariant: pass/fail plus a violation degree.

    HIG-C (IBAAM 3, revised edition, Section 3.2) grades violations: :attr:`degree` is 0 when the invariant holds and
    in (0, 1] when it is violated. Boolean invariants use 0 or 1 (:meth:`allow` / :meth:`reject`); threshold
    invariants report the normalised shortfall with :meth:`violated`.
    """

    __slots__ = ("passed", "_degree", "reason", "severity")

    def __init__(
        self,
        passed: bool = False,
        degree: Optional[float] = None,
        reason: str = "",
        severity: InvariantSeverity = InvariantSeverity.Error,
    ) -> None:
        #: ``True`` if the invariant holds.
        self.passed = passed
        self._degree = None if degree is None else clamp(degree, 0.0, 1.0)
        #: Explanation for a failure.
        self.reason = reason
        #: Severity reported for this result. Note: the evaluator uses the invariant's declared severity for
        #: violations.
        self.severity = severity

    @property
    def degree(self) -> float:
        """Violation degree v ∈ [0, 1]: 0 when the invariant holds, 1 for a full violation, and a value in between
        for a partial (threshold) violation. Defaults to 0 for a pass and 1 for a failure."""
        if self.passed:
            return 0.0
        return 1.0 if self._degree is None else self._degree

    @staticmethod
    def allow() -> "InvariantResult":
        """Creates a passing result."""
        return InvariantResult(passed=True)

    @staticmethod
    def reject(reason: str, severity: InvariantSeverity = InvariantSeverity.Error) -> "InvariantResult":
        """Creates a failing result with violation degree 1."""
        return InvariantResult(passed=False, reason=reason, severity=severity)

    @staticmethod
    def violated(degree: float, reason: str,
                 severity: InvariantSeverity = InvariantSeverity.Error) -> "InvariantResult":
        """Creates a graded result. A degree of 0 (or less) is a pass; anything greater is a violation of that degree,
        e.g. ``max(0, (0.8 - coverage) / 0.8)`` for "coverage is at least 80%"."""
        if degree <= 0:
            return InvariantResult.allow()
        return InvariantResult(passed=False, degree=degree, reason=reason, severity=severity)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"InvariantResult(passed={self.passed}, degree={self.degree}, reason={self.reason!r})"


@dataclass
class InvariantViolation:
    """A failed invariant evaluation, carrying the invariant's governance metadata at the time of failure."""

    #: Name of the violated invariant.
    invariant: str = ""
    #: Category of the violated invariant.
    category: InvariantCategory = InvariantCategory.Safety
    #: Scope at which the invariant was evaluated.
    scope: InvariantScope = InvariantScope.Transition
    #: Severity of the violation; under the default policy Error and Critical can never be overridden.
    severity: InvariantSeverity = InvariantSeverity.Info
    #: HIG level of the violated invariant.
    layer: InvariantLayer = InvariantLayer.Fundamental
    #: Base criticality of the violated invariant.
    criticality: InvariantCriticality = InvariantCriticality.Medium
    #: Criticality of this violation in the operational context it occurred in, C(I, κ).
    criticality_in_context: InvariantCriticality = InvariantCriticality.Medium
    #: Violation degree v ∈ (0, 1].
    degree: float = 1.0
    #: Operational contexts in which violating the invariant is more critical than usual.
    contexts: list[OperationalContext] = field(default_factory=list)
    #: Explanation returned by the invariant.
    reason: str = ""


@dataclass
class InvariantEvaluationRecord:
    """Pairs an evaluated invariant with its result; consumed by meta-invariants."""

    #: The invariant that was evaluated.
    invariant: "IInvariant"
    #: The evaluation result.
    result: InvariantResult


# ------------------------------------------------------------------------------------------------- violation profile


def level_code(layer: InvariantLayer) -> str:
    """One-letter level code used in profile strings: F, M, B, H."""
    return {
        InvariantLayer.Fundamental: "F",
        InvariantLayer.Mission: "M",
        InvariantLayer.Behavioural: "B",
    }.get(layer, "H")


class ViolationProfile:
    """The violation profile P(a, κ) of a candidate action (IBAAM 3 revised, Section 6.1).

    Component n(ℓ, k) is the total violation degree of the invariants at level ℓ whose violation has criticality k in
    the current context. Components are ordered by level first (Fundamental, Mission, Behavioural, Heuristic) and
    criticality second (Catastrophic down to Low), most severe first.

    Profiles are compared lexicographically: any violation at a higher level outweighs any violation at a lower level,
    and within a level a more critical violation outweighs any amount of less critical violation.
    """

    #: Levels in profile order, most authoritative first.
    LEVEL_ORDER = (
        InvariantLayer.Fundamental,
        InvariantLayer.Mission,
        InvariantLayer.Behavioural,
        InvariantLayer.AdaptiveHeuristic,
    )
    #: Criticalities in profile order, most severe first.
    CRITICALITY_ORDER = (
        InvariantCriticality.Catastrophic,
        InvariantCriticality.Critical,
        InvariantCriticality.High,
        InvariantCriticality.Medium,
        InvariantCriticality.Low,
    )
    # Differences smaller than this are treated as equal, so floating-point noise never decides a resolution.
    _TOLERANCE = 1e-9

    def __init__(self) -> None:
        self._mass = [0.0] * (len(self.LEVEL_ORDER) * len(self.CRITICALITY_ORDER))

    @classmethod
    def zero(cls) -> "ViolationProfile":
        """The zero profile: no violations."""
        return cls()

    def add(self, layer: InvariantLayer, criticality: InvariantCriticality, degree: float) -> None:
        """Adds a violation of degree ``degree`` at the given level and criticality."""
        if degree <= 0:
            return
        self._mass[self._index_of(layer, criticality)] += degree

    def __getitem__(self, key: tuple[InvariantLayer, InvariantCriticality]) -> float:
        """Violation mass n(ℓ, k)."""
        return self._mass[self._index_of(*key)]

    @property
    def components(self) -> list[float]:
        """The profile as a vector in comparison order."""
        return list(self._mass)

    @property
    def is_zero(self) -> bool:
        """``True`` when nothing is violated."""
        return all(m <= self._TOLERANCE for m in self._mass)

    @property
    def has_fundamental_violation(self) -> bool:
        """``True`` when any fundamental invariant is violated (escalation is then required, Section 7.2)."""
        return self.fundamental_mass > self._TOLERANCE

    @property
    def fundamental_mass(self) -> float:
        """Total violation degree at the fundamental level."""
        return sum(self[(InvariantLayer.Fundamental, k)] for k in self.CRITICALITY_ORDER)

    def compare_to(self, other: Optional["ViolationProfile"]) -> int:
        """Lexicographic comparison: negative when this profile is less bad than ``other``."""
        if other is None:
            return 1
        for mine, theirs in zip(self._mass, other._mass):
            diff = mine - theirs
            if abs(diff) > self._TOLERANCE:
                return -1 if diff < 0 else 1
        return 0

    def __lt__(self, other: "ViolationProfile") -> bool:
        return self.compare_to(other) < 0

    def deciding_position(self, other: "ViolationProfile") -> Optional[tuple[InvariantLayer, InvariantCriticality]]:
        """The first position (level, criticality) at which this profile and ``other`` differ, i.e. the component
        that decides the comparison. ``None`` when the profiles are equal."""
        for i, (mine, theirs) in enumerate(zip(self._mass, other._mass)):
            if abs(mine - theirs) > self._TOLERANCE:
                return self._position_of(i)
        return None

    def __str__(self) -> str:
        """Compact, human-readable form listing only non-zero components, e.g. ``M/Critical=1; M/High=0.25``."""
        parts = []
        for i, mass in enumerate(self._mass):
            if mass > self._TOLERANCE:
                layer, criticality = self._position_of(i)
                parts.append(f"{level_code(layer)}/{criticality.name}={fmt_max(mass)}")
        return "none" if not parts else "; ".join(parts)

    __repr__ = __str__

    @staticmethod
    def short(layer: InvariantLayer) -> str:
        """One-letter level code used in profile strings: F, M, B, H."""
        return level_code(layer)

    @classmethod
    def _index_of(cls, layer: InvariantLayer, criticality: InvariantCriticality) -> int:
        try:
            level = cls.LEVEL_ORDER.index(layer)
        except ValueError:
            raise ValueError(f"Unknown governance level: {layer!r}") from None
        try:
            crit = cls.CRITICALITY_ORDER.index(criticality)
        except ValueError:
            raise ValueError(f"Unknown criticality: {criticality!r}") from None
        return level * len(cls.CRITICALITY_ORDER) + crit

    @classmethod
    def _position_of(cls, index: int) -> tuple[InvariantLayer, InvariantCriticality]:
        n = len(cls.CRITICALITY_ORDER)
        return cls.LEVEL_ORDER[index // n], cls.CRITICALITY_ORDER[index % n]


# ------------------------------------------------------------------------------------------------- meta-invariants


@dataclass
class OverrideWitness:
    """Evidence that an override was necessary (Section 7.1): a candidate that would have violated the overridden
    invariant less, and the profile position at which that candidate is worse."""

    #: Id of the competing candidate.
    candidate: str = ""
    #: The competing candidate's violation degree for the overridden invariant.
    competing_degree: float = 0.0
    #: Level of the profile component at which the competing candidate is worse.
    deciding_layer: InvariantLayer = InvariantLayer.Fundamental
    #: Criticality of the profile component at which the competing candidate is worse.
    deciding_criticality: InvariantCriticality = InvariantCriticality.Medium
    #: Invariants the competing candidate violates at the deciding position.
    deciding_invariants: list[str] = field(default_factory=list)
    #: ``False`` when the competing candidate has an equal profile, i.e. it is not worse and the override is not
    #: witnessed.
    worse: bool = True

    def __str__(self) -> str:
        if not self.worse:
            return f"{self.candidate} is equally good and violates it less"
        text = f"{self.candidate} is worse at {level_code(self.deciding_layer)}/{self.deciding_criticality.name}"
        if self.deciding_invariants:
            text += f" ({', '.join(self.deciding_invariants)})"
        return text


@dataclass
class MetaInvariantContext:
    """Input to a meta-invariant: one violation the governed candidate would carry, plus the resolution around it."""

    #: The transition being governed.
    transition_context: Optional["TransitionContext"] = None
    #: The violation being considered for override.
    candidate_violation: InvariantViolation = field(default_factory=InvariantViolation)
    #: All violations of the candidate action in the same checkpoint.
    all_violations: list[InvariantViolation] = field(default_factory=list)
    #: All invariant evaluations (passed and failed) of the candidate action in the same checkpoint.
    evaluation_records: list[InvariantEvaluationRecord] = field(default_factory=list)
    #: Id of the candidate being authorised (see :class:`GovernanceCandidate`).
    candidate_id: str = "proposed"
    #: The candidate's violation profile.
    candidate_profile: ViolationProfile = field(default_factory=ViolationProfile)
    #: Candidates that would have violated :attr:`candidate_violation`'s invariant less, with the position at which
    #: each is worse (Section 7.1).
    witnesses: list[OverrideWitness] = field(default_factory=list)
    #: Number of candidates in the governed set (more than one means a tie left to the adaptive layer).
    governed_set_size: int = 1
    #: Operational context the decision is made under.
    operational_context: OperationalContext = OperationalContext.Normal
    #: Pre-authorised protocol covering a fundamental-level violation, or empty when none does.
    escalation_protocol: str = ""


@dataclass
class MetaInvariantResult:
    """Result of a meta-invariant evaluation, with optional metadata for the override record."""

    #: ``True`` if the meta-invariant allows the override.
    passed: bool = False
    #: Explanation, typically for a rejection.
    reason: str = ""
    #: Values contributed to the override record. Well-known keys: ``PreservedHigherPriorityInvariants``,
    #: ``Justification``, ``Audit``, ``RequiresReview``, ``ReviewReasons``.
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def allow(reason: str = "", metadata: Optional[dict[str, Any]] = None) -> "MetaInvariantResult":
        """Creates an allowing result."""
        return MetaInvariantResult(passed=True, reason=reason, metadata=dict(metadata or {}))

    @staticmethod
    def reject(reason: str) -> "MetaInvariantResult":
        """Creates a rejecting result; the candidate violation stays blocking."""
        return MetaInvariantResult(passed=False, reason=reason)


# ------------------------------------------------------------------------------------------------------ resolutions


@dataclass
class ContextualGovernanceDecision:
    """How priority, criticality and context decided one override under HIG-C.

    HIG-C keeps hierarchy as the primary mechanism; criticality and operational context only decide within a level
    (IBAAM 3 revised, Section 6.2).
    """

    #: The contextual resolution outcome.
    outcome: ContextualGovernanceOutcome = ContextualGovernanceOutcome.NotApplicable
    #: Criticality of the violation being decided.
    criticality: InvariantCriticality = InvariantCriticality.Medium
    #: Operational context in effect when the decision was made.
    operational_context: OperationalContext = OperationalContext.Normal
    #: Contexts in which the violated invariant is especially relevant.
    contexts: list[OperationalContext] = field(default_factory=list)
    #: Invariants violated by the competing candidates at the deciding profile position.
    compared_invariants: list[str] = field(default_factory=list)
    #: Human-readable explanation of the decision.
    reason: str = ""


@dataclass
class GovernanceCandidate:
    """One candidate action and its evaluation under the resolution rule."""

    #: Id of the planner's proposal (the action in flight at later checkpoints).
    PROPOSED_ID = "proposed"
    #: Id of the null action a₀: decline, hold or defer.
    DECLINE_ID = "decline"

    #: Candidate id: ``proposed``, ``alternative-N`` or ``decline``.
    id: str = "proposed"
    #: The candidate action; ``None`` for the decline option.
    action: Optional[AgentAction] = None
    #: The candidate's violation profile, rendered (see :meth:`ViolationProfile.__str__`).
    profile: str = "none"
    #: The invariants this candidate violates, with degree and criticality in context.
    violations: list[InvariantViolation] = field(default_factory=list)
    #: ``True`` when the candidate is in the governed set.
    governed: bool = False
    #: ``True`` when the candidate is governed and every override it needs is authorised by the meta-invariants.
    valid: bool = False
    #: Why a governed candidate was not valid (meta-invariant rejections), if it was not.
    rejections: list[str] = field(default_factory=list)

    @property
    def is_decline(self) -> bool:
        """``True`` for the null action a₀."""
        return self.id == GovernanceCandidate.DECLINE_ID


@dataclass
class GovernanceResolution:
    """The decision record for one governed checkpoint (IBAAM 3 revised, Section 7.4): every candidate and its
    violation profile, the governed set R(𝒜, κ), the action selected within it and any escalation.

    Because the resolution rule is deterministic given the context, the violation degrees and the criticality table, a
    reviewer can recompute the decision from this record.
    """

    #: The checkpoint that was resolved.
    scope: InvariantScope = InvariantScope.Transition
    #: Operational context κ the decision was made under.
    operational_context: OperationalContext = OperationalContext.Normal
    #: Every candidate considered, in adaptive-preference order (proposal first, decline last).
    candidates: list[GovernanceCandidate] = field(default_factory=list)
    #: Ids of the candidates in the governed set R(𝒜, κ): those with lexicographically minimal profiles.
    governed_set: list[str] = field(default_factory=list)
    #: Id of the candidate the runtime will act on (``decline`` when it holds).
    selected_id: str = GovernanceCandidate.PROPOSED_ID
    #: ``True`` when no candidate satisfies every invariant.
    conflict: bool = False
    #: ``True`` when the minimal profile has a positive fundamental component, so no governed action may be carried
    #: out autonomously (Section 7.2).
    requires_escalation: bool = False
    #: The pre-authorised protocol that covered the escalation, or empty when none did.
    escalation_protocol: str = ""
    #: Why this candidate was selected, in plain words.
    reason: str = ""


@dataclass
class InvariantOverride:
    """An auditable record of a violation the governed action carries (IBAAM 3 revised, Section 7).

    An override exists only when the selected action is in the governed set and every meta-invariant allows the
    override. The record carries the witnesses showing that every candidate which would have violated the invariant
    less has a worse violation profile, the generated justification and review flags, so the override can be
    recomputed and inspected later.
    """

    #: The violation that was overridden.
    overridden_violation: InvariantViolation = field(default_factory=InvariantViolation)
    #: Candidates that would have violated the invariant less, each with the profile position at which it is worse.
    witnesses: list[OverrideWitness] = field(default_factory=list)
    #: Names of higher-level invariants the governed action satisfies (context for reviewers).
    preserved_higher_priority_invariants: list[str] = field(default_factory=list)
    #: Generated rationale explaining why the override was necessary.
    justification: str = ""
    #: Comma-separated list of meta-invariant categories that participated (kept for display/back-compat).
    meta_invariant_category: str = "Priority"
    #: Distinct meta-invariant categories that evaluated and allowed this override.
    meta_invariant_categories: list[MetaInvariantCategory] = field(default_factory=list)
    #: ``True`` when the override situation is ambiguous or noisy and should be reviewed by a human.
    requires_review: bool = False
    #: Reasons the override was flagged for review (e.g. same-level conflicts, ties).
    review_reasons: list[str] = field(default_factory=list)
    #: How criticality and context decided the override: ``Subordinate`` when a same-level comparison decided it,
    #: ``NotApplicable`` when priority between levels decided it.
    contextual_decision: ContextualGovernanceDecision = field(default_factory=ContextualGovernanceDecision)
    #: Pre-authorised protocol that authorised a fundamental-level override, if any.
    escalation_protocol: str = ""

    @property
    def overridden_layer(self) -> InvariantLayer:
        """Convenience accessor for the governance level of the overridden invariant."""
        return self.overridden_violation.layer


@dataclass
class InvariantEvaluationReport:
    """The outcome of resolving one lifecycle checkpoint under HIG-C."""

    #: The lifecycle scope that was evaluated.
    scope: InvariantScope = InvariantScope.Transition
    #: Violations that stop the transition: the in-flight action's violations when the governed outcome is to
    #: decline or to escalate.
    violations: list[InvariantViolation] = field(default_factory=list)
    #: Violations carried by the governed action, each witnessed and authorised by the meta-invariants.
    overrides: list[InvariantOverride] = field(default_factory=list)
    #: The alternative the resolution selected instead of the planner's proposal (plan checkpoint only); ``None``
    #: when the proposal itself was selected or the outcome is to decline.
    selected_alternative: Optional[AgentAction] = None
    #: ``True`` when the governed outcome is the null action: the transition does not continue.
    declined: bool = False
    #: ``True`` when the least-bad outcome violates a fundamental invariant and no pre-authorised protocol covers it,
    #: so the decision is held for human review (Section 7.2).
    requires_escalation: bool = False
    #: The full decision record: candidates, profiles, governed set and selection.
    resolution: Optional[GovernanceResolution] = None

    @property
    def passed(self) -> bool:
        """``True`` when the runtime may continue with a governed action. Violations carried by that action appear as
        :attr:`overrides`; they do not prevent the report from passing."""
        return len(self.violations) == 0

    @property
    def summary(self) -> str:
        """Human-readable summary used as the rejection reason and in control events."""
        if self.passed:
            if not self.overrides:
                return "All invariants passed."
            return "Governed action selected; its violations were witnessed and overridden."
        prefix = ("Escalation required: the least-bad outcome violates a fundamental invariant.\n"
                  if self.requires_escalation else "")
        return prefix + "\n".join(f"{v.invariant}: {v.reason}" for v in self.violations)


@dataclass
class ControlDecision:
    """A simple allow/block control decision, optionally derived from an :class:`InvariantEvaluationReport`."""

    #: ``True`` if the transition may proceed.
    allowed: bool = False
    #: Reason for blocking.
    reason: str = ""
    #: Whether the decision was escalated (reserved for future human-in-the-loop handling).
    escalated: bool = False
    #: Violations that caused the decision.
    violations: list[InvariantViolation] = field(default_factory=list)

    @staticmethod
    def allow() -> "ControlDecision":
        """Creates an allowing decision."""
        return ControlDecision(allowed=True)

    @staticmethod
    def block(reason: str) -> "ControlDecision":
        """Creates a blocking decision."""
        return ControlDecision(allowed=False, reason=reason)

    @staticmethod
    def from_report(report: InvariantEvaluationReport) -> "ControlDecision":
        """Converts an evaluation report into an allow/block decision."""
        return ControlDecision.allow() if report.passed else ControlDecision.block(report.summary)

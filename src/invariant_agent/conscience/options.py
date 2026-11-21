"""Conscience tuning and session identity."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from ..core.conscience import CommitmentLedger
from ..core.model.enums import CommitmentScope


def _default_exempt() -> set[str]:
    return {
        "explain",
        "drift",
        "replay",
        "memory-show",
        "baseline-approve",
        "baseline-show",
        "conscience",
        "conscience-show",
        "conscience-review",
        "context",
    }


class _CaseInsensitiveSet(set):
    """A ``set`` of strings with ``StringComparer.OrdinalIgnoreCase`` membership."""

    def __init__(self, values=()):
        super().__init__(v.lower() for v in values)

    def __contains__(self, item) -> bool:  # type: ignore[override]
        return isinstance(item, str) and super().__contains__(item.lower())

    def add(self, item: str) -> None:  # type: ignore[override]
        super().add(item.lower())

    def discard(self, item: str) -> None:  # type: ignore[override]
        super().discard(item.lower())


@dataclass
class ConscienceOptions:
    """Tuning for the history-calibrated conscience.

    Defaults are conservative placeholders; IBAAM 6 (Section 6.2) tunes the thresholds against a human-labelled set,
    which is the next empirical step.
    """

    #: When false the runtime runs without a conscience (pure IBAAM 1-4 behaviour).
    enabled: bool = True
    #: Whether to run the answer appraisal (after reduction invariants, before commit).
    appraise_answers: bool = True
    #: Maximum number of re-plans a Revise stance may request before it becomes Escalate.
    max_revisions: int = 2
    #: Tension at or above which the stance is Revise.
    revise_threshold: float = 0.55
    #: Tension at or above which the stance is Escalate.
    escalate_threshold: float = 0.75
    #: Tension at or above which the stance is Veto.
    veto_threshold: float = 0.9
    #: Minimum plan similarity for a past transition to count as a precedent.
    min_precedent_similarity: float = 0.6
    #: Maximum number of precedents retrieved per appraisal.
    max_precedents: int = 10
    #: Maximum number of precedents (and commitments) cited in a justification.
    max_citations: int = 3
    #: Pseudo-count pulling violation risk towards the deployment base rate. Larger values need more precedent before
    #: the conscience trusts it.
    prior_pseudo_count: float = 1.0
    #: Beta prior (alpha) for the deployment base rate of control rejections. With :attr:`base_rate_prior_beta` the
    #: cold-start base rate is alpha / (alpha + beta) = 0.25, below the Revise threshold, so a new deployment does not
    #: intervene on ordinary plans before it has history.
    base_rate_prior_alpha: float = 1.0
    #: Beta prior (beta) for the deployment base rate of control rejections.
    base_rate_prior_beta: float = 3.0
    #: Weight of violation risk when combining components.
    violation_risk_weight: float = 1.0
    #: Weight of forecast drift when combining components.
    forecast_drift_weight: float = 0.8
    #: How strongly trend pressure amplifies the plan-specific tension: at full pressure (Critical region, high
    #: confidence) the exponent becomes ``1 + weight``. Trend pressure never raises tension on its own.
    trend_pressure_weight: float = 0.5
    #: Weight of commitment conflict when combining components.
    commitment_conflict_weight: float = 1.0
    #: How much an agent's own commitment counts relative to a reviewer's (reviewer = 1).
    agent_commitment_strength: float = 0.8
    #: Similarity at or above which a plan matches a commitment.
    commitment_match_threshold: float = CommitmentLedger.DEFAULT_MATCH_THRESHOLD
    #: Lifetime of the agent's own commitments; ``None`` for no expiry. Reviewer commitments never expire.
    agent_commitment_lifetime: Optional[timedelta] = timedelta(days=30)
    #: Scope of commitments created by a Revise stance.
    revise_commitment_scope: CommitmentScope = CommitmentScope.Conversation
    #: Scope of commitments created by Escalate and Veto. User scope lets a trajectory spread across sessions by one
    #: user stay visible (IBAAM 6, Case 1).
    stop_commitment_scope: CommitmentScope = CommitmentScope.User
    #: Scope of reviewer commitments.
    reviewer_commitment_scope: CommitmentScope = CommitmentScope.Deployment
    #: Identity used for user-scoped commitments. IA2 is single-user, so this is a fixed id.
    user_id: str = "local"
    #: Capabilities that are never appraised: introspection, and the governance tools a human uses to review the
    #: conscience itself (a conscience must not be able to block its own review). Case-insensitive.
    exempt_capabilities: set[str] = field(default_factory=lambda: _CaseInsensitiveSet(_default_exempt()))

    def __post_init__(self) -> None:
        if not isinstance(self.exempt_capabilities, _CaseInsensitiveSet):
            self.exempt_capabilities = _CaseInsensitiveSet(self.exempt_capabilities)


class ConscienceSession:
    """Identifies the current runtime session for conversation-scoped commitments.

    A restart starts a new conversation, while user- and deployment-scoped commitments carry over.
    """

    def __init__(self, conversation_id: Optional[str] = None) -> None:
        #: The conversation id.
        self.conversation_id: str = conversation_id or uuid.uuid4().hex

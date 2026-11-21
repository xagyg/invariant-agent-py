"""Core conscience helpers: plan similarity, the commitment ledger and the in-memory commitment store."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Optional, Sequence

from ._util import is_blank
from .abstractions import ICommitmentStore
from .model.conscience import Commitment
from .model.enums import CommitmentOrigin, CommitmentScope


class PlanSimilarity:
    """Deterministic similarity between two plans, used to retrieve precedents and to match plans to commitments.

    ``0.4 x [same capability] + 0.6 x Jaccard(character trigrams of the inputs)``, giving a value in 0..1. Trigrams
    make near-identical inputs (``password=a`` and ``password=b``) similar while unrelated inputs to the same
    capability (``echo hi`` and ``echo weather``) stay below the default precedent threshold of 0.6. Deliberately
    simple and inspectable: every flag cites cases a reviewer can compare by eye.
    """

    CAPABILITY_WEIGHT = 0.4
    INPUT_WEIGHT = 0.6

    @staticmethod
    def score(capability_a: Optional[str], input_a: Optional[str],
              capability_b: Optional[str], input_b: Optional[str]) -> float:
        """Similarity of two plans in 0..1."""
        same = 1.0 if PlanSimilarity.normalise(capability_a) == PlanSimilarity.normalise(capability_b) else 0.0
        return (PlanSimilarity.CAPABILITY_WEIGHT * same
                + PlanSimilarity.INPUT_WEIGHT * PlanSimilarity.input_similarity(input_a, input_b))

    @staticmethod
    def input_similarity(a: Optional[str], b: Optional[str]) -> float:
        """Jaccard similarity of the character trigrams of two inputs (1 when both are empty)."""
        na = PlanSimilarity.normalise(a)
        nb = PlanSimilarity.normalise(b)
        if not na and not nb:
            return 1.0
        if na == nb:
            return 1.0
        ta = PlanSimilarity._trigrams(na)
        tb = PlanSimilarity._trigrams(nb)
        if not ta or not tb:
            return 0.0
        intersection = len(ta & tb)
        union = len(ta) + len(tb) - intersection
        return 0.0 if union == 0 else intersection / union

    @staticmethod
    def normalise(value: Optional[str]) -> str:
        """Lower-cases, trims and collapses whitespace."""
        if is_blank(value):
            return ""
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _trigrams(value: str) -> set[str]:
        # Trigrams of the padded string, so short inputs still produce some.
        padded = f" {value} "
        return {padded[i:i + 3] for i in range(len(padded) - 2)}


class CommitmentLedger:
    """Computes which commitments are live (IBAAM 6, Section 6.4.2): unexpired, in scope, and not superseded by a
    later reviewer decision.

    Reviewer commitments take precedence over the agent's own. A reviewer commitment supersedes every earlier
    commitment that it resolves (same source transition) or that concerns the same value and a similar subject.
    Agent commitments never supersede anything. Liveness is computed from the append-only store, so nothing is lost.
    """

    #: Default similarity at or above which two commitments (or a plan and a commitment) are "about the same thing".
    DEFAULT_MATCH_THRESHOLD = 0.6

    @staticmethod
    def live(all: Sequence[Commitment], now: datetime, conversation_id: str, user_id: str,
             match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> list[Commitment]:
        """Returns the live commitments, oldest first."""
        result = []
        for i, commitment in enumerate(all):
            if commitment.expires is not None and commitment.expires <= now:
                continue
            if not CommitmentLedger._in_scope(commitment, conversation_id, user_id):
                continue
            if CommitmentLedger._is_superseded(all, i, match_threshold):
                continue
            result.append(commitment)
        return result

    @staticmethod
    def for_transition(all: Sequence[Commitment], transition_id: uuid.UUID) -> list[Commitment]:
        """Commitments created from (or resolving) the given transition."""
        return [c for c in all if c.source_transition == transition_id or c.resolves == transition_id]

    @staticmethod
    def is_resolved(all: Sequence[Commitment], transition_id: uuid.UUID) -> bool:
        """``True`` if a reviewer has already resolved the given transition."""
        return any(c.origin == CommitmentOrigin.Reviewer and c.resolves == transition_id for c in all)

    @staticmethod
    def match(commitment: Commitment, capability: str, input: str) -> float:
        """Similarity between a plan and a commitment's subject."""
        return PlanSimilarity.score(commitment.capability, commitment.subject, capability, input)

    @staticmethod
    def _in_scope(commitment: Commitment, conversation_id: str, user_id: str) -> bool:
        if commitment.scope == CommitmentScope.Deployment:
            return True
        if commitment.scope == CommitmentScope.User:
            return commitment.user_id == user_id
        if commitment.scope == CommitmentScope.Conversation:
            return commitment.conversation_id == conversation_id
        return False

    @staticmethod
    def _is_superseded(all: Sequence[Commitment], index: int, match_threshold: float) -> bool:
        # Superseded when a later reviewer commitment resolves it or concerns the same value and a similar subject.
        commitment = all[index]
        for later in all[index + 1:]:
            if later.origin != CommitmentOrigin.Reviewer:
                continue
            if later.resolves is not None and later.resolves == commitment.source_transition:
                return True
            if ((later.value or "").lower() == (commitment.value or "").lower() and
                    PlanSimilarity.score(later.capability, later.subject,
                                         commitment.capability, commitment.subject) >= match_threshold):
                return True
        return False


class InMemoryCommitmentStore(ICommitmentStore):
    """In-memory :class:`ICommitmentStore` (tests and non-persistent runs)."""

    def __init__(self) -> None:
        self._commitments: list[Commitment] = []
        self._lock = threading.Lock()

    def append(self, commitment: Commitment) -> None:
        with self._lock:
            self._commitments.append(commitment)

    def get_all(self) -> list[Commitment]:
        with self._lock:
            return list(self._commitments)

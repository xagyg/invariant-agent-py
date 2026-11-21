"""Behavioural stability measurement (IBAAM 2, Agent Identity Evaluation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .enums import MeasurementConfidence, StabilityRegion


@dataclass
class StabilityVector:
    """Agent Identity Evaluation (AIE) dimensions, each scored 0-100 (higher is more stable)."""

    #: How recognisable behaviour is: penalised by rejections and by a wide spread of capabilities.
    identifiability: int = 100
    #: How smoothly state evolves: 100 minus the behavioural drift score across the window.
    continuity: int = 100
    #: Whether identical actions yield identical outcomes.
    consistency: int = 100
    #: Whether goal, mode, policies and stable memory are retained across the window.
    persistence: int = 100
    #: Whether the agent returns to successful operation after rejected transitions.
    recovery: int = 100

    @property
    def average_score(self) -> float:
        """Unweighted mean of the five dimensions."""
        return (self.identifiability + self.continuity + self.consistency + self.persistence + self.recovery) / 5.0


@dataclass
class StabilityAssessment:
    """Result of behavioural stability evaluation (IBAAM 2).

    Indicates how behaviour is changing; invariants remain responsible for deciding whether change is acceptable.
    """

    #: Stability vector for the most recent window of completed transitions.
    current: StabilityVector = field(default_factory=StabilityVector)
    #: Stability vector for the preceding window, or ``None`` if there is not enough history.
    previous: Optional[StabilityVector] = None
    #: Euclidean distance between :attr:`previous` and :attr:`current` (0-100).
    drift_magnitude: int = 0
    #: Classified stability region.
    region: StabilityRegion = StabilityRegion.Stable
    #: Confidence in the measurement.
    confidence: MeasurementConfidence = MeasurementConfidence.Low
    #: Governance recommendations for the region.
    recommendations: list[str] = field(default_factory=list)

    @property
    def has_prior_measurement(self) -> bool:
        """``True`` if a previous window was available for comparison."""
        return self.previous is not None

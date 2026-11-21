"""Calibration of the violation-risk component against realised control outcomes (IBAAM 6, Section 8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..core._util import fmt_fixed
from ..core.model.enums import AppraisalPoint, ConscienceEffect, TensionKind, TransitionStatus
from ..core.model.transition import Transition


@dataclass(frozen=True)
class CalibrationBin:
    """One reliability-curve bin."""

    #: Inclusive lower forecast bound.
    lower: float
    #: Exclusive upper forecast bound (inclusive for the last bin).
    upper: float
    #: Forecasts in the bin.
    count: int
    #: Mean forecast violation risk.
    mean_forecast: float
    #: Fraction actually blocked by the controls.
    observed_rate: float


@dataclass(frozen=True)
class CalibrationReport:
    """Brier score and a reliability curve for the violation-risk forecasts.

    Only plans the conscience let through to the controls are scored, because only they have a realised outcome.
    Plans it revised away or stopped have none, which is the label-thinning effect of IBAAM 6, Section 9: the more
    the conscience intervenes, the less of its own forecast it can check.
    """

    #: Number of scored forecasts.
    count: int
    #: Mean squared error of the forecasts (0 is perfect), or ``None`` with no forecasts.
    brier_score: Optional[float]
    #: Observed rate of control rejections among scored plans.
    base_rate: float
    #: Appraised plans with no realised control outcome (revised away or stopped by the conscience).
    unlabelled: int
    #: Reliability curve.
    bins: list[CalibrationBin] = field(default_factory=list)

    @staticmethod
    def compute(transitions: Iterable[Transition], bin_count: int = 5) -> "CalibrationReport":
        """Computes the report from transition history with recorded appraisals."""
        pairs: list[tuple[float, float]] = []
        unlabelled = 0
        for transition in transitions:
            appraisals = transition.appraisals or []
            plan = next((a for a in reversed(appraisals) if a.point == AppraisalPoint.Plan), None)
            if plan is None or plan.appraisal is None:
                continue
            risk = next((p for p in plan.appraisal.parts if p.kind == TensionKind.ViolationRisk), None)
            if risk is None or not risk.applicable:
                continue
            # Only the final plan that proceeded to the controls has a realised outcome.
            if plan.effect != ConscienceEffect.Proceeded:
                unlabelled += 1
                continue
            if transition.status == TransitionStatus.Completed:
                outcome: Optional[float] = 0.0
            elif transition.status == TransitionStatus.Rejected and transition.blocking_invariants:
                outcome = 1.0
            else:
                outcome = None
            if outcome is None:
                # e.g. passed every control and was then stopped at the answer point.
                if transition.status == TransitionStatus.Rejected and any(a.stopped for a in appraisals):
                    pairs.append((risk.value, 0.0))
                else:
                    unlabelled += 1
                continue
            pairs.append((risk.value, outcome))
        if not pairs:
            return CalibrationReport(0, None, 0.0, unlabelled, [])
        brier = sum((f - o) ** 2 for f, o in pairs) / len(pairs)
        bins = []
        for b in range(bin_count):
            lower = b / bin_count
            upper = (b + 1) / bin_count
            last = b == bin_count - 1
            in_bin = [(f, o) for f, o in pairs if f >= lower and (f <= upper if last else f < upper)]
            if not in_bin:
                continue
            bins.append(CalibrationBin(lower, upper, len(in_bin),
                                       sum(f for f, _ in in_bin) / len(in_bin),
                                       sum(o for _, o in in_bin) / len(in_bin)))
        return CalibrationReport(len(pairs), brier, sum(o for _, o in pairs) / len(pairs), unlabelled, bins)

    def format(self) -> str:
        """Plain-text rendering."""
        lines = ["Violation-risk calibration (forecast vs realised control outcome):"]
        if self.count == 0:
            lines.append("  No scored forecasts yet.")
        else:
            lines.append(f"  Forecasts scored: {self.count}")
            lines.append(f"  Brier score: {fmt_fixed(self.brier_score, 3)} (0 = perfect; always forecasting the base "
                         f"rate scores {fmt_fixed(self.base_rate * (1 - self.base_rate), 3)})")
            lines.append(f"  Observed rejection rate: {fmt_fixed(self.base_rate, 2)}")
            lines.append("  Reliability:")
            for b in self.bins:
                lines.append(f"    [{fmt_fixed(b.lower, 1)}-{fmt_fixed(b.upper, 1)}] n={b.count} "
                             f"forecast {fmt_fixed(b.mean_forecast, 2)} observed {fmt_fixed(b.observed_rate, 2)}")
        lines.append(f"  Appraised plans with no realised outcome (revised or stopped): {self.unlabelled}")
        return "\n".join(lines).rstrip()

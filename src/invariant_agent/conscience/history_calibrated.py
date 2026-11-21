"""The deterministic, history-calibrated conscience (IBAAM 6, Section 6.2)."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence

from ..core._util import distinct, fmt_fixed, short_id
from ..core.abstractions import IConscience, IValueJudge
from ..core.conscience import CommitmentLedger
from ..core.model.conscience import Commitment, ConscienceAppraisal, ConscienceInput, TensionComponent
from ..core.model.control import ProtectedValues
from ..core.model.enums import (
    AppraisalPoint,
    CommitmentKind,
    CommitmentOrigin,
    ConscienceStance,
    MeasurementConfidence,
    PrecedentOutcome,
    StabilityRegion,
    TensionKind,
)
from ..core.model.stability import StabilityAssessment
from .options import ConscienceOptions


class HistoryCalibratedConscience(IConscience):
    """Deterministic first version of functional conscience: prospective tension calibrated from the agent's own
    governance history.

    Four components, each in 0..1:

    * **Violation risk**: similarity-weighted rate at which similar past plans were blocked by the controls, pulled
      towards the deployment base rate by a pseudo-count. The only component that is a calibrated probability.
    * **Forecast drift**: at the plan point, the drift similar past plans produced; at the answer point, the drift
      score of the proposed state (a dry run of the commit). Expressed as (score / bound)^2.
    * **Trend pressure**: the current stability region (halved when measurement confidence is low).
    * **Commitment conflict**: closeness of the plan to live declined commitments (similarity rescaled from the match
      threshold to 1), reviewer commitments counting in full and the agent's own at
      :attr:`ConscienceOptions.agent_commitment_strength`.

    The plan-specific components combine as ``p = 1 - product((1 - v_i)^w_i)``, so one strong component is enough to
    raise tension while weak ones accumulate. Trend pressure then amplifies it:
    ``tension = 1 - (1 - p)^(1 + w x trend)``. Unstable history makes real concerns weigh more but never stops a plan
    on its own. Tension is a score for stance selection, not a probability.

    Same plan + same history gives the same appraisal; a different history can give a different one. That history
    dependence, not non-determinism, is what separates this from an invariant.
    """

    name = "history-calibrated-v1"

    def __init__(self, options: ConscienceOptions, value_judge: Optional[IValueJudge] = None) -> None:
        """:param value_judge: optional value judge consulted only when no close precedent exists."""
        self._options = options
        self._value_judge = value_judge

    def appraise(self, input: ConscienceInput) -> ConscienceAppraisal:
        violation = self._violation_risk(input)
        drift = self._forecast_drift(input)
        trend = self._trend_pressure(input)
        conflict, conflict_cited, cleared_by = self._commitment_conflict(input)
        parts = [violation, drift, trend, conflict]
        tension = _combine(parts)
        stance = self._select_stance(tension)
        # Reviewer decisions take precedence over the agent's own judgement (IBAAM 6, Section 6.4.2). A reviewer who
        # cleared this conduct lets it proceed to the controls, which still decide what is permitted.
        if cleared_by is not None and stance != ConscienceStance.Proceed:
            stance = ConscienceStance.Proceed
        blocked = sorted((p for p in input.precedents if p.outcome == PrecedentOutcome.Blocked),
                         key=lambda p: -p.similarity)
        cited = [p.transition_id for p in blocked[: self._options.max_citations]]
        cited_commitments = list(conflict_cited)
        if cleared_by is not None:
            cited_commitments.insert(0, cleared_by.id)
        return ConscienceAppraisal(
            tension=round(tension, 4),
            parts=parts,
            stance=stance,
            justification=_justify(stance, tension, parts, cleared_by),
            cited_precedents=cited,
            cited_commitments=distinct(cited_commitments),
            values_at_stake=_values_at_stake(parts),
            conscience=self.name,
        )

    def _violation_risk(self, input: ConscienceInput) -> TensionComponent:
        o = self._options
        if input.point == AppraisalPoint.Answer:
            return TensionComponent(TensionKind.ViolationRisk, 0.0, o.violation_risk_weight,
                                    "not applicable at the answer point: every control has already passed",
                                    applicable=False)
        labelled = [p for p in input.precedents if p.outcome != PrecedentOutcome.Withheld]
        if not labelled and self._value_judge is not None:
            judged = self._value_judge.judge(input)
            if judged is not None:
                return dataclasses.replace(judged, kind=TensionKind.ViolationRisk, weight=o.violation_risk_weight,
                                           value=_clamp(judged.value),
                                           detail="value judge (no close precedent): " + judged.detail)
        weight = sum(p.similarity for p in labelled)
        blocked = [p for p in labelled if p.outcome == PrecedentOutcome.Blocked]
        blocked_weight = sum(p.similarity for p in blocked)
        alpha = o.prior_pseudo_count
        risk = (blocked_weight + alpha * input.base_rate) / (weight + alpha) if (weight + alpha) else 0.0
        if not labelled:
            detail = (f"no close precedent; deployment base rate {_f(input.base_rate)} over "
                      f"{input.labelled_history} labelled transitions")
        else:
            cites = []
            for p in sorted(blocked, key=lambda p: -p.similarity)[: o.max_citations]:
                values = f" [{', '.join(p.values)}]" if p.values else ""
                cites.append(f"#{short_id(p.transition_id)} blocked by {'+'.join(p.blocked_by)}{values}, "
                             f"sim {_f(p.similarity)}")
            detail = (f"{len(blocked)} of {len(labelled)} similar plans were blocked by the controls"
                      + (f" ({'; '.join(cites)})" if blocked else "")
                      + f"; base rate {_f(input.base_rate)}")
        return TensionComponent(TensionKind.ViolationRisk, _clamp(risk), o.violation_risk_weight, detail,
                                values=distinct((v for p in blocked for v in p.values), ignore_case=True))

    def _forecast_drift(self, input: ConscienceInput) -> TensionComponent:
        o = self._options
        bound = max(1, input.drift_bound)
        if input.point == AppraisalPoint.Answer:
            score = input.forecast_drift_score or 0
            value = _drift_pressure(score, bound)
            return TensionComponent(TensionKind.ForecastDrift, value, o.forecast_drift_weight,
                                    f"committing would produce drift score {score} against bound {bound}",
                                    values=[ProtectedValues.IdentityStability] if value > 0 else [])
        with_drift = [p for p in input.precedents if p.realised_drift is not None]
        if not with_drift:
            return TensionComponent(TensionKind.ForecastDrift, 0.0, o.forecast_drift_weight,
                                    "no similar plan has reached reduction, so no drift is forecast")
        weight = sum(p.similarity for p in with_drift)
        mean = sum(p.similarity * p.realised_drift for p in with_drift) / weight
        forecast = _drift_pressure(mean, bound)
        return TensionComponent(
            TensionKind.ForecastDrift, forecast, o.forecast_drift_weight,
            f"similar plans produced drift {fmt_fixed(mean, 0)} on average ({len(with_drift)} precedents) "
            f"against bound {bound}",
            values=[ProtectedValues.IdentityStability] if forecast > 0 else [])

    def _trend_pressure(self, input: ConscienceInput) -> TensionComponent:
        trend = input.trend or StabilityAssessment()
        value = {
            StabilityRegion.Stable: 0.0,
            StabilityRegion.Watch: 0.34,
            StabilityRegion.Intervention: 0.67,
            StabilityRegion.Critical: 1.0,
        }.get(trend.region, 0.0)
        # Little history means little to go on: halve the pressure rather than act on a noisy region.
        if trend.confidence == MeasurementConfidence.Low:
            value *= 0.5
        return TensionComponent(
            TensionKind.TrendPressure, _clamp(value), self._options.trend_pressure_weight,
            f"stability region {trend.region} ({trend.confidence} confidence, drift magnitude {trend.drift_magnitude})",
            values=[ProtectedValues.IdentityStability] if value > 0 else [])

    def _commitment_conflict(self, input: ConscienceInput):
        o = self._options
        capability = (input.proposed.capability if input.proposed else None) or ""
        plan_input = (input.proposed.input if input.proposed else None) or ""
        matches = [(c, CommitmentLedger.match(c, capability, plan_input)) for c in input.commitments]
        matches = [(c, s) for c, s in matches if s >= o.commitment_match_threshold]
        conflicts = [
            (c, s, self._closeness(s) * (1.0 if c.origin == CommitmentOrigin.Reviewer else o.agent_commitment_strength))
            for c, s in matches if c.kind == CommitmentKind.Declined]
        conflicts.sort(key=lambda m: -m[2])
        clearing = sorted(((c, s) for c, s in matches
                           if c.kind == CommitmentKind.Undertaken and c.origin == CommitmentOrigin.Reviewer),
                          key=lambda m: -m[1])
        # A reviewer clearance applies unless a reviewer decline matches the plan at least as closely.
        strongest_reviewer_decline = max((s for c, s, _ in conflicts if c.origin == CommitmentOrigin.Reviewer),
                                         default=0.0)
        cleared_by: Optional[Commitment] = None
        if clearing and clearing[0][1] > strongest_reviewer_decline:
            cleared_by = clearing[0][0]
        cited = [c.id for c, _, _ in conflicts[: o.max_citations]]
        if not conflicts:
            detail = f"no live commitment conflicts with this plan ({len(input.commitments)} live)"
            if cleared_by is not None:
                detail += f"; cleared by reviewer commitment #{short_id(cleared_by.id)}"
            return (TensionComponent(TensionKind.CommitmentConflict, 0.0, o.commitment_conflict_weight, detail),
                    cited, cleared_by)
        top, top_similarity, top_score = conflicts[0]
        detail = (f"conflicts with {top.origin.name.lower()} commitment #{short_id(top.id)} "
                  f"(\"{top.statement}\", sim {_f(top_similarity)})"
                  + (f" and {len(conflicts) - 1} more" if len(conflicts) > 1 else ""))
        component = TensionComponent(TensionKind.CommitmentConflict, _clamp(top_score), o.commitment_conflict_weight,
                                     detail, values=distinct((c.value for c, _, _ in conflicts), ignore_case=True))
        return component, cited, cleared_by

    # How close a match is, rescaled so a plan just at the match threshold contributes almost nothing and the
    # identical plan contributes in full. Without this, anything that merely cleared the threshold (same capability,
    # same memory key) scored about 0.5 straight away, so one commitment about "goal=learn a song" escalated
    # "goal=learn the piano".
    def _closeness(self, similarity: float) -> float:
        threshold = min(0.99, self._options.commitment_match_threshold)
        return _clamp((similarity - threshold) / (1.0 - threshold))

    def _select_stance(self, tension: float) -> ConscienceStance:
        o = self._options
        if tension >= o.veto_threshold:
            return ConscienceStance.Veto
        if tension >= o.escalate_threshold:
            return ConscienceStance.Escalate
        if tension >= o.revise_threshold:
            return ConscienceStance.Revise
        return ConscienceStance.Proceed


# Quadratic in the fraction of the bound used: ordinary adaptation (a goal change scores 40 of 70, giving 0.33) stays
# mild, while drift close to the bound (60 of 70, giving 0.73) weighs heavily. Remaining tolerance shrinks fastest
# near the bound.
def _drift_pressure(score: float, bound: int) -> float:
    fraction = _clamp(score / bound)
    return _clamp(fraction * fraction)


def _combine(parts: Sequence[TensionComponent]) -> float:
    """Plan-specific components (violation risk, forecast drift, commitment conflict) combine as
    ``1 - product((1 - v)^w)``. Trend pressure is remaining tolerance, not a concern about this plan, so it amplifies
    that tension rather than adding its own: ``tension = 1 - (1 - planTension)^(1 + w x trend)``. A plan with no
    specific concern stays at zero however unstable recent history is; how to respond to instability as such is the
    stability evaluator's business (IBAAM 2), not the conscience's."""
    remaining = 1.0
    amplification = 1.0
    for part in parts:
        if not part.applicable or part.weight <= 0:
            continue
        if part.kind == TensionKind.TrendPressure:
            amplification += part.weight * _clamp(part.value)
            continue
        remaining *= math.pow(1.0 - _clamp(part.value), part.weight)
    return _clamp(1.0 - math.pow(remaining, amplification))


# Values ordered by how much their component contributed; falls back to the dominant component's tag.
def _values_at_stake(parts: Sequence[TensionComponent]) -> list[str]:
    ordered = sorted((p for p in parts if p.applicable), key=lambda p: -(p.value * p.weight))
    values = distinct((v for p in ordered if p.value > 0 for v in p.values), ignore_case=True)
    if values:
        return values
    dominant = ordered[0] if ordered else None
    kind = dominant.kind if dominant is not None else None
    if kind in (TensionKind.ForecastDrift, TensionKind.TrendPressure):
        return [ProtectedValues.IdentityStability]
    if kind == TensionKind.CommitmentConflict:
        return [ProtectedValues.Consistency]
    return [ProtectedValues.ControlBoundary]


# The justification is generated from the parts, so the stated reason always matches the computed score.
def _justify(stance: ConscienceStance, tension: float, parts: Sequence[TensionComponent],
             cleared_by: Optional[Commitment]) -> str:
    text = f"{stance} (tension {_f(tension)}): " + "; ".join(
        f"{_label(p.kind)} {_f(p.value) if p.applicable else 'n/a'} - {p.detail}" for p in parts)
    if cleared_by is not None:
        text += (f". A reviewer cleared this conduct (commitment #{short_id(cleared_by.id)}: "
                 f"\"{cleared_by.statement}\"), so it proceeds to the controls")
    return text + "."


def _label(kind: TensionKind) -> str:
    return {
        TensionKind.ViolationRisk: "violation risk",
        TensionKind.ForecastDrift: "forecast drift",
        TensionKind.TrendPressure: "trend pressure",
        TensionKind.CommitmentConflict: "commitment conflict",
    }.get(kind, kind.name)


def _clamp(value: float) -> float:
    return 0.0 if value != value else max(0.0, min(1.0, value))


def _f(value: float) -> str:
    return fmt_fixed(value, 2)

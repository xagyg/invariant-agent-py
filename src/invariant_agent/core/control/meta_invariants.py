"""The standard HIG-C meta-invariants (IBAAM 3 revised, Section 7)."""

from __future__ import annotations

from .._util import fmt_max, is_empty
from ..abstractions import IMetaInvariant
from ..model.control import MetaInvariantContext, MetaInvariantResult
from ..model.enums import InvariantLayer, InvariantSeverity, MetaInvariantCategory


class PriorityMetaInvariant(IMetaInvariant):
    """Fundamental invariants and escalation (IBAAM 3 revised, Section 7.2): no action that violates a fundamental
    invariant may be carried out autonomously.

    Priority between levels is enforced by the resolution rule itself: a higher-level violation always outweighs a
    lower-level one in the violation profile, so this meta-invariant no longer needs the earlier rules ("no override
    while a higher-level violation is unresolved", "override authority from a preserved higher-level invariant").

    What remains is the fundamental rule. A fundamental violation is authorised only when a pre-authorised protocol
    covers it; otherwise the agent holds at the null action and the evaluator escalates. Changing priorities is a
    change to the hierarchy (versioned and approved), never a runtime override.
    """

    name = "PriorityMetaInvariant"
    category = MetaInvariantCategory.Priority

    def evaluate(self, context: MetaInvariantContext) -> MetaInvariantResult:
        violation = context.candidate_violation
        if violation.layer == InvariantLayer.Fundamental:
            if is_empty(context.escalation_protocol):
                return MetaInvariantResult.reject(
                    f"{violation.invariant} is fundamental: it cannot be violated autonomously and requires "
                    "escalation.")
            return MetaInvariantResult.allow(
                f"Fundamental violation authorised by pre-authorised protocol '{context.escalation_protocol}'.",
                {"EscalationProtocol": context.escalation_protocol})
        return MetaInvariantResult.allow()


class OverrideSeverityMetaInvariant(IMetaInvariant):
    """Deployment policy: Error and Critical violations are never overridden. Only Info and Warning violations are
    eligible.

    The HIG-C paper has no severity dimension; this is an additional, deployment-level meta-invariant kept so that
    invariants declared as hard constraints stay hard regardless of level. A governed candidate that would need a hard
    override is not valid, and the runtime falls back to the next governed candidate or to declining. Remove it from
    the meta-invariant set to let level, criticality and context alone decide.
    """

    name = "OverrideSeverityMetaInvariant"
    category = MetaInvariantCategory.OverrideSeverity

    def evaluate(self, context: MetaInvariantContext) -> MetaInvariantResult:
        violation = context.candidate_violation
        if violation.severity >= InvariantSeverity.Error:
            return MetaInvariantResult.reject(
                f"{violation.invariant} has severity {violation.severity} and remains blocking by policy.")
        return MetaInvariantResult.allow()


class JustificationMetaInvariant(IMetaInvariant):
    """Witnessed override (IBAAM 3 revised, Section 7.1): an override is authorised only when every candidate that
    would have violated the invariant less has a worse violation profile.

    The witnesses are computed by the evaluator and supplied in :attr:`MetaInvariantContext.witnesses`. Because the
    candidate is in the governed set, every competing candidate is at least as bad; the override is rejected when a
    competitor is *equally* good and violates the invariant less, since the override is then not necessary.

    This replaces the earlier rule that justified an override whenever some higher-level invariant had passed, which
    allowed overrides with no conflict at all. The generated justification is returned as ``Justification``
    metadata.
    """

    name = "JustificationMetaInvariant"
    category = MetaInvariantCategory.Justification

    def evaluate(self, context: MetaInvariantContext) -> MetaInvariantResult:
        violation = context.candidate_violation
        unwitnessed = [w for w in context.witnesses if not w.worse]
        if unwitnessed:
            return MetaInvariantResult.reject(
                f"{violation.invariant} override is not witnessed: "
                + ", ".join(w.candidate for w in unwitnessed)
                + " is equally good and violates it less.")
        if not context.witnesses:
            justification = (f"{violation.invariant} (degree {fmt_max(violation.degree)}) is overridden: "
                             "no candidate violates it less.")
        else:
            justification = (f"{violation.invariant} (degree {fmt_max(violation.degree)}, "
                             f"{violation.criticality_in_context} in {context.operational_context}) is overridden "
                             "because every alternative that violates it less is worse: "
                             + "; ".join(str(w) for w in context.witnesses) + ".")
        return MetaInvariantResult.allow(metadata={"Justification": justification})


class AuditMetaInvariant(IMetaInvariant):
    """Marks every authorised override for governance audit. Never rejects; contributes ``Audit = True``."""

    name = "AuditMetaInvariant"
    category = MetaInvariantCategory.Audit

    def evaluate(self, context: MetaInvariantContext) -> MetaInvariantResult:
        return MetaInvariantResult.allow(metadata={"Audit": True})


class ReviewMetaInvariant(IMetaInvariant):
    """Flags overrides that warrant human review without blocking them (IBAAM 3 revised, Section 7.5).

    Review is requested when:

    * the override was decided within a level, by criticality in context (a same-level conflict);
    * several candidates tie in the governed set, so the adaptive layer's preference decided;
    * a fundamental violation was authorised by a pre-authorised protocol;
    * the number of simultaneous violations reaches the configured threshold.

    Repeated overrides of the same invariant across transitions are an observability concern and are reported by the
    drift tooling rather than here.
    """

    name = "ReviewMetaInvariant"
    category = MetaInvariantCategory.Review

    def __init__(self, violation_review_threshold: int = 3) -> None:
        """:param violation_review_threshold: number of simultaneous violations that triggers review (default 3)."""
        self._violation_review_threshold = violation_review_threshold

    def evaluate(self, context: MetaInvariantContext) -> MetaInvariantResult:
        reasons = []
        violation = context.candidate_violation
        same_level = [w for w in context.witnesses if w.worse and w.deciding_layer == violation.layer]
        if same_level:
            reasons.append(f"{violation.invariant} was decided within level {violation.layer} by criticality in "
                           f"{context.operational_context}: " + "; ".join(str(w) for w in same_level))
        if context.governed_set_size > 1:
            reasons.append(f"{context.governed_set_size} candidates tie in the governed set; the adaptive layer's "
                           "preference decided.")
        if violation.layer == InvariantLayer.Fundamental and not is_empty(context.escalation_protocol):
            reasons.append(f"Fundamental invariant {violation.invariant} overridden under protocol "
                           f"'{context.escalation_protocol}'.")
        if len(context.all_violations) >= self._violation_review_threshold:
            reasons.append(f"Override considered while {len(context.all_violations)} invariants were violated.")
        return MetaInvariantResult.allow(metadata={"RequiresReview": len(reasons) > 0, "ReviewReasons": reasons})


def default_meta_invariants() -> list[IMetaInvariant]:
    """The standard HIG-C meta-invariant set."""
    return [
        PriorityMetaInvariant(),
        OverrideSeverityMetaInvariant(),
        JustificationMetaInvariant(),
        AuditMetaInvariant(),
        ReviewMetaInvariant(),
    ]

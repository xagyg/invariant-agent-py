"""Internal introspection and governance tools: explain, replay, drift, memory, baselines, conscience and context.

``baseline-approve``, ``conscience-review`` and ``context`` change the controls rather than act within them, so they
are operator-only (``ICapability.operator_only``).
"""

from __future__ import annotations

from typing import Callable, Optional

from ..conscience.calibration import CalibrationReport
from ..conscience.options import ConscienceOptions, ConscienceSession
from ..core._util import (
    cs_str,
    fmt_fixed,
    fmt_percent0,
    fmt_round_trip,
    fmt_universal,
    is_blank,
    new_id,
    short_id,
    utc_now,
)
from ..core.abstractions import ICapability, ICommitmentStore, IDriftAnalyzer, IDriftBaselineStore, ITransitionStore
from ..core.conscience import CommitmentLedger, PlanSimilarity
from ..core.control.hierarchy import OperationalContextState
from ..core.model.capability import CapabilityRequest, CapabilityResult, SelfModificationRequest
from ..core.model.conscience import Commitment, ConscienceAppraisalRecord
from ..core.model.control import ProtectedValues
from ..core.model.data import TextData
from ..core.model.enums import (
    CommitmentKind,
    CommitmentOrigin,
    ConscienceEffect,
    OperationalContext,
    TransitionStatus,
)
from ..core.model.transition import Transition, TransitionContext
from ..core.rendering import TransitionFormatter
from ..core.replay import ReplayValidator
from ..core.serialization import clone


def _latest_committed(transitions: list[Transition]) -> Optional[Transition]:
    return next((t for t in reversed(transitions) if t.status == TransitionStatus.Completed and t.after is not None),
                None)


# ----------------------------------------------------------------------------------------------------------- memory


class MemorySetTool(ICapability):
    """``memory-set key=value``: proposes a governed memory update.

    The tool does not change memory itself. It returns a ``SelfModificationRequest`` which must pass
    self-modification invariants (e.g. ``AllowedMemoryKeyInvariant``) before the reducer applies it.
    """

    name = "memory-set"

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        parts = [p.strip() for p in (request.input or "").split("=", 1)]
        if len(parts) != 2:
            return CapabilityResult.fail(self.name, "Expected format: key=value")
        modification = SelfModificationRequest(target="memory", operation="set", key=parts[0], value=parts[1])
        return CapabilityResult.ok(self.name, TextData(value=f"Proposed memory update: {parts[0]}"), modification)


class MemoryShowTool(ICapability):
    """``memory-show``: lists memory from the latest committed state.

    Only completed transitions are considered. A transition rejected at the Reduction checkpoint still has a proposed
    ``after`` state, but that state was never committed and is ignored here.
    """

    name = "memory-show"

    def __init__(self, store: ITransitionStore) -> None:
        self._store = store

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        latest = _latest_committed(self._store.get_all())
        state = latest.after if latest else None
        if state is None or not state.memory:
            return CapabilityResult.ok(self.name, TextData(value="Memory is empty."))
        lines = ["\n==== MEMORY ===="] + [f"{k}={cs_str(v)}" for k, v in state.memory.items()]
        return CapabilityResult.ok(self.name, TextData(value="\n".join(lines) + "\n==== END MEMORY ===="))


# -------------------------------------------------------------------------------------------------------- baselines


class BaselineApproveTool(ICapability):
    """``baseline-approve [reason]``: approves the latest committed state as the drift baseline.

    After approval, behavioural drift (and the drift-bound invariant) is measured against this state. Only completed
    transitions are considered, so a state that governance rejected (for example at the Reduction checkpoint) can
    never be approved as a baseline. Approval is the human-in-the-loop mechanism for accepting intentional behavioural
    evolution.
    """

    name = "baseline-approve"
    #: Re-certifying the baseline moves what drift is measured against, so only an operator may do it.
    operator_only = True

    def __init__(self, transitions: ITransitionStore, baselines: IDriftBaselineStore) -> None:
        self._transitions = transitions
        self._baselines = baselines

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        latest = _latest_committed(self._transitions.get_all())
        if latest is None:
            return CapabilityResult.fail(self.name, "No committed state is available to approve as a drift baseline.")
        reason = "Approved via baseline-approve." if is_blank(request.input) else request.input.strip()
        baseline = self._baselines.approve(latest.after, reason)
        return CapabilityResult.ok(self.name, TextData(
            value=f"Approved drift baseline {baseline.id} at state version {baseline.state_version}."))


class BaselineShowTool(ICapability):
    """``baseline-show``: displays the currently approved drift baseline."""

    name = "baseline-show"

    def __init__(self, baselines: IDriftBaselineStore) -> None:
        self._baselines = baselines

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        baseline = self._baselines.current
        if baseline is None:
            return CapabilityResult.ok(self.name, TextData(value="No drift baseline has been approved."))
        lines = [
            "\n==== DRIFT BASELINE ====",
            f"Id: {baseline.id}",
            f"ApprovedUtc: {fmt_round_trip(baseline.approved_at_utc)}",
            f"StateVersion: {baseline.state_version}",
            f"Reason: {baseline.reason}",
        ]
        if baseline.state.memory:
            lines.append("Memory:")
            lines += [f"  {k}={cs_str(v)}" for k, v in sorted(baseline.state.memory.items())]
        lines.append("==== END DRIFT BASELINE ====")
        return CapabilityResult.ok(self.name, TextData(value="\n".join(lines)))


# ---------------------------------------------------------------------------------------------------------- context


class ContextTool(ICapability):
    """``context [normal|degraded|emergency|recovery|maintenance] [reason]``: shows or sets the HIG-C operational
    context (IBAAM 3 revised, Section 5). Operator-only: a planner can never change the context its own conflicts are
    judged under.

    A new context applies from the next transition; the transition that sets it keeps the old one.
    """

    name = "context"
    operator_only = True

    def __init__(self, state: OperationalContextState) -> None:
        self._state = state

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        parts = [p.strip() for p in (request.input or "").strip().split(" ", 1) if p.strip()]
        names = ", ".join(c.name for c in OperationalContext)
        if not parts:
            return CapabilityResult.ok(self.name, TextData(
                value=f"Operational context: {self._state.current} (since {fmt_universal(self._state.since)}; "
                      f"{self._state.reason}). Options: {names}."))
        context = OperationalContext.parse(parts[0])
        if context is None:
            return CapabilityResult.fail(self.name, f"Unknown context '{parts[0]}'. Expected one of: {names}.")
        reason = parts[1] if len(parts) > 1 else "set by operator"
        previous = self._state.set(context, reason)
        return CapabilityResult.ok(self.name, TextData(
            value=f"Operational context {previous} -> {context} ({reason}). Applies from the next transition."))


# ------------------------------------------------------------------------------------------------- explain & replay


class ExplainTool(ICapability):
    """``explain [last | id-prefix]``: explains a transition's plan, invariant decisions and state impact."""

    name = "explain"

    def __init__(self, store: ITransitionStore) -> None:
        self._store = store

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        transitions = self._store.get_all()
        if not transitions:
            return CapabilityResult.ok(self.name, TextData(value="No transitions recorded."))
        transition = self._resolve(transitions, request.input)
        if transition is None:
            return CapabilityResult.fail(self.name, "Transition not found.")
        return CapabilityResult.ok(self.name, TextData(value=TransitionFormatter.format_explain(transition)))

    # Empty input or "last" selects the latest transition; otherwise matches by id prefix. The explain transition that
    # is currently running is not stored yet, so "last" means the previous transition.
    @staticmethod
    def _resolve(transitions: list[Transition], input: str) -> Optional[Transition]:
        if is_blank(input) or input.lower() == "last":
            return transitions[-1]
        return next((t for t in reversed(transitions) if str(t.id).lower().startswith(input.lower())), None)


class ReplayTool(ICapability):
    """``replay [N] [verbose] [validate]`` (aliases: history, audit, trace): replays recorded transitions.

    * ``N`` limits output to the last N transitions.
    * ``verbose`` includes each transition's full event trail.
    * ``validate`` re-evaluates invariants on a copy of each transition and reports governance drift.
    """

    name = "replay"
    aliases = ("history", "audit", "trace")

    def __init__(self, store: ITransitionStore, validator: Callable[[], ReplayValidator]) -> None:
        """:param validator: returns the replay validator lazily (avoids a construction-time dependency cycle through
        the invariant evaluator and capability registry)."""
        self._store = store
        self._validator = validator

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        transitions = self._store.get_all()
        if not transitions:
            return CapabilityResult.ok(self.name, TextData(
                value="==== REPLAY START ====\nNo transitions recorded.\n==== REPLAY END ===="))
        text = (request.input or "").strip()
        validate = "validate" in text.lower()
        verbose = "verbose" in text.lower()
        count = _parse_count(text, len(transitions))
        selected = transitions[len(transitions) - count:] if count > 0 else []
        result = (TransitionFormatter.format_replay_verbose(selected) if verbose
                  else TransitionFormatter.format_replay(selected))
        if validate:
            result += self._format_validation(selected, verbose)
        return CapabilityResult.ok(self.name, TextData(value=result))

    # Runs replay validation for each transition and formats the comparison. Details are shown when verbose, or
    # always for mismatches.
    def _format_validation(self, transitions: list[Transition], verbose: bool) -> str:
        lines = ["", "==== REPLAY VALIDATION START ===="]
        validator = self._validator()
        for transition in transitions:
            # Do not use the real transition: this is for reporting only.
            context = TransitionContext(transition=clone(transition))
            # Report-only: replaying must not add records to the drift tracker.
            result = validator.validate(context, record_drift=False)
            if (verbose or not result.passed) and result.comparison is not None:
                c = result.comparison
                lines += [f"  OriginalStatus={c.original_status}", f"  ReplayStatus={c.replay_status}",
                          f"  OriginalPhase={c.original_phase}", f"  ReplayPhase={c.replay_phase}"]
                if not c.drift_baseline_recorded and transition.after is not None:
                    lines.append("  DriftBaseline=not recorded (scored against current baseline)")
                if c.original_violations:
                    lines.append(f"  OriginalViolations={len(c.original_violations)}")
                    lines += [f"    - {v}" for v in c.original_violations]
                if c.replay_violations:
                    lines.append(f"  ReplayViolations={len(c.replay_violations)}")
                    lines += [f"    - {v}" for v in c.replay_violations]
            lines.append(f"[{transition.id} {transition.status}] Passed" if result.passed
                         else f"[{transition.id} {transition.status}] Drift={result.drift_type}, Reason={result.reason}")
        lines.append("==== REPLAY VALIDATION END ====")
        return "\n".join(lines) + "\n"


# Returns the first integer token in the input (capped at the history length), or all transitions.
def _parse_count(text: str, default: int) -> int:
    for part in text.split():
        try:
            return min(int(part), default)
        except ValueError:
            continue
    return default


# ------------------------------------------------------------------------------------------------------------ drift


class DriftTool(ICapability):
    """``drift``: renders a drift report over the full transition history, including the behavioural stability
    vector, region and governance recommendations."""

    name = "drift"

    def __init__(self, store: ITransitionStore, analyzer: IDriftAnalyzer) -> None:
        self._store = store
        self._analyzer = analyzer

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        report = self._analyzer.analyze(self._store.get_all())
        s = report.stability
        lines = [
            "\n==== DRIFT REPORT ====",
            f"Transitions: {report.transition_count}",
            f"Rejected: {report.rejected_transitions}",
            f"Total drift score: {report.total_drift_score}",
            f"Highest drift severity: {report.highest_drift_severity}",
            f"Stability region: {s.region}",
            f"Stability confidence: {s.confidence}",
            f"Stability drift magnitude: {s.drift_magnitude}",
            "",
            "Conflict resolution (HIG-C):",
            f"  Transitions with conflicts: {report.conflict_transitions}",
            f"  Escalated for human review: {report.escalated_transitions}",
            f"  Attributed drift: {report.attributed_drift.events} events, score {report.attributed_drift.score} "
            f"({fmt_percent0(report.attributed_drift.share)})",
            f"  Unattributed drift: {report.unattributed_drift.events} events, score "
            f"{report.unattributed_drift.score} ({fmt_percent0(report.unattributed_drift.share)})",
        ]
        if report.missing_expected_change:
            lines.append("  Missing expected change (review):")
            lines += [f"    {item}" for item in report.missing_expected_change]
        v = s.current
        lines += [
            "",
            "Stability vector:",
            f"  Identifiability: {v.identifiability}",
            f"  Continuity: {v.continuity}",
            f"  Consistency: {v.consistency}",
            f"  Persistence: {v.persistence}",
            f"  Recovery: {v.recovery}",
            f"  Average: {fmt_fixed(v.average_score, 1)}",
            "",
            "Governance recommendations:",
        ]
        lines += [f"  {r}" for r in s.recommendations] or ["  none"]
        for title, items in (("Capability usage:", report.capability_usage),
                             ("Invariant failures:", report.invariant_failures),
                             ("Invariant failures by category:", report.invariant_failures_by_category),
                             ("Drift events:", report.drift_counts)):
            lines += ["", title]
            lines += [f"  {k}: {count}" for k, count in items.items()] or ["  none"]
        lines += ["", "Recent drift:"]
        if not report.recent_drift:
            lines.append("  none")
        for d in report.recent_drift:
            lines.append(f"  [{d.severity}] {d.type} Score={d.score} Phase={d.phase} Transition={d.transition_id}")
            lines.append(f"    {d.reason}")
        lines.append("==== END DRIFT REPORT ====")
        return CapabilityResult.ok(self.name, TextData(value="\n".join(lines)))


# ------------------------------------------------------------------------------------------------------- conscience


def pending_escalations(history: list[Transition],
                        commitments: list[Commitment]) -> list[tuple[Transition, ConscienceAppraisalRecord]]:
    """Escalated transitions that no reviewer has resolved yet."""
    pending = []
    for t in history:
        record = next((a for a in reversed(t.appraisals or []) if a.effect == ConscienceEffect.Escalated), None)
        if record is not None and not CommitmentLedger.is_resolved(commitments, t.id):
            pending.append((t, record))
    return pending


class ConscienceShowTool(ICapability):
    """``conscience [commitments | pending | calibration | last | id-prefix]``: inspects the functional conscience
    (IBAAM 6): live commitments, escalations awaiting review, violation-risk calibration and recorded appraisals.

    Read-only and exempt from appraisal, so the conscience can always be inspected.
    """

    name = "conscience"
    aliases = ("conscience-show",)

    def __init__(self, transitions: ITransitionStore, commitments: ICommitmentStore, options: ConscienceOptions,
                 session: ConscienceSession) -> None:
        self._transitions = transitions
        self._commitments = commitments
        self._options = options
        self._session = session

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        arg = (request.input or "").strip()
        history = self._transitions.get_all()
        key = arg.lower()
        if key == "":
            text = self._summary(history)
        elif key == "commitments":
            text = self._commitments_text()
        elif key == "pending":
            text = self._pending(history)
        elif key == "calibration":
            text = CalibrationReport.compute(history).format()
        else:
            text = self._appraisals(history, arg)
        return CapabilityResult.ok(self.name, TextData(value=text))

    def _summary(self, history: list[Transition]) -> str:
        all_commitments = self._commitments.get_all()
        live = self._live(all_commitments)
        appraised = [t for t in history if t.appraisals]
        lines = [f"Conscience: {'enabled' if self._options.enabled else 'disabled'}",
                 f"Appraised transitions: {len(appraised)}"]
        counts: dict[ConscienceEffect, int] = {}
        for record in (a for t in appraised for a in t.appraisals):
            counts[record.effect] = counts.get(record.effect, 0) + 1
        lines += [f"  {effect}: {count}" for effect, count in sorted(counts.items())]
        lines += [f"Commitments: {len(live)} live of {len(all_commitments)} recorded",
                  f"Escalations awaiting review: {len(pending_escalations(history, all_commitments))}",
                  "",
                  "Usage: conscience commitments | pending | calibration | last | <transition-id-prefix>",
                  "       conscience-review <transition-id-prefix> uphold|overturn [note]"]
        return "\n".join(lines).rstrip()

    def _commitments_text(self) -> str:
        all_commitments = self._commitments.get_all()
        if not all_commitments:
            return "No commitments recorded."
        live = self._live(all_commitments)
        lines = [f"Live commitments ({len(live)} of {len(all_commitments)}):"]
        for c in live:
            expires = f", expires {c.expires:%Y-%m-%d}" if c.expires is not None else ""
            lines.append(f"  #{short_id(c.id)} {c.origin} {c.kind} [{c.value}] {c.scope}: {c.statement}"
                         f" (from #{short_id(c.source_transition)}{expires})")
        return "\n".join(lines).rstrip()

    def _pending(self, history: list[Transition]) -> str:
        pending = pending_escalations(history, self._commitments.get_all())
        if not pending:
            return "No escalations awaiting review."
        lines = ["Escalations awaiting review:"]
        for transition, record in pending:
            lines.append(f"  {short_id(transition.id)} {record.point}: {record.capability} {record.input}".rstrip())
            lines.append(f"    {record.appraisal.justification}")
        lines += ["", "Resolve with: conscience-review <id> uphold|overturn [note]"]
        return "\n".join(lines).rstrip()

    @staticmethod
    def _appraisals(history: list[Transition], arg: str) -> str:
        if arg.lower() == "last":
            transition = next((t for t in reversed(history) if t.appraisals), None)
        else:
            prefix = arg.replace("-", "").lower()
            transition = next((t for t in reversed(history) if t.id.hex.startswith(prefix)), None)
        if transition is None:
            return "Transition not found (or never appraised)."
        if not transition.appraisals:
            return (f"Transition {short_id(transition.id)} was not appraised (exempt capability or conscience "
                    "disabled).")
        lines = [f"Transition {short_id(transition.id)}: {transition.input} -> {transition.status}"]
        for record in transition.appraisals:
            a = record.appraisal
            lines += ["", f"{record.point} appraisal, attempt {record.attempt}: {record.capability} {record.input}".rstrip()]
            note = "" if is_blank(record.note) else f" ({record.note})"
            lines.append(f"  Stance: {a.stance} -> {record.effect}{note}")
            lines.append(f"  Tension: {fmt_fixed(a.tension, 2)}  Values at stake: {', '.join(a.values_at_stake)}")
            for part in a.parts:
                value = fmt_fixed(part.value, 2) if part.applicable else "n/a"
                lines.append(f"  - {part.kind}: {value} (w {fmt_fixed(part.weight, 1)}) {part.detail}")
            if record.commitment_id is not None:
                lines.append(f"  Commitment created: #{short_id(record.commitment_id)}")
        return "\n".join(lines).rstrip()

    def _live(self, all_commitments: list[Commitment]) -> list[Commitment]:
        return CommitmentLedger.live(all_commitments, utc_now(), self._session.conversation_id,
                                     self._options.user_id, self._options.commitment_match_threshold)


class ConscienceReviewTool(ICapability):
    """``conscience-review <transition-id-prefix> uphold|overturn [note]``: a human reviewer resolves a conscience
    escalation (or appeals a veto).

    The decision is recorded as a reviewer commitment, which takes precedence over the agent's own commitments
    (IBAAM 6, Section 6.4.2) and is the value-learning signal the controls cannot provide (Section 5.2):

    * **uphold**: the conduct is declined. Similar plans now meet a reviewer-strength commitment conflict.
    * **overturn**: the conduct is cleared. Earlier declined commitments on the same subject are superseded and
      similar plans proceed to the controls, which still decide what is permitted.

    Nothing is re-run: the user retries the request if they still want it.
    """

    name = "conscience-review"
    #: A reviewer decision is the value-learning signal the controls cannot provide, so it must come from a human,
    #: never from a planner reviewing its own escalation.
    operator_only = True

    def __init__(self, transitions: ITransitionStore, commitments: ICommitmentStore,
                 options: ConscienceOptions) -> None:
        self._transitions = transitions
        self._commitments = commitments
        self._options = options

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        parts = [p.strip() for p in (request.input or "").strip().split(None, 2) if p.strip()]
        if len(parts) < 2:
            return CapabilityResult.fail(self.name,
                                         "Expected: conscience-review <transition-id-prefix> uphold|overturn [note]")
        prefix = parts[0].replace("-", "").lower()
        decision = parts[1].lower()
        note = parts[2] if len(parts) > 2 else ""
        if decision not in ("uphold", "overturn"):
            return CapabilityResult.fail(self.name, "Decision must be 'uphold' or 'overturn'.")
        transition = next((t for t in reversed(self._transitions.get_all()) if t.id.hex.startswith(prefix)), None)
        stop = next((a for a in reversed(transition.appraisals or []) if a.stopped), None) if transition else None
        if transition is None or stop is None:
            return CapabilityResult.fail(self.name, f"No escalated or vetoed transition matches '{parts[0]}'.")
        if CommitmentLedger.is_resolved(self._commitments.get_all(), transition.id):
            return CapabilityResult.fail(self.name, f"Transition {short_id(transition.id)} has already been reviewed.")
        upheld = decision == "uphold"
        value = (stop.appraisal.values_at_stake[0] if stop.appraisal and stop.appraisal.values_at_stake
                 else ProtectedValues.ControlBoundary)
        conduct = f"{stop.capability} {stop.input}".strip()
        statement = (f"Reviewer declined: {conduct}" if upheld else f"Reviewer cleared: {conduct}") \
            + ("" if is_blank(note) else f" ({note})")
        commitment = Commitment(
            id=new_id(),
            source_transition=transition.id,
            value=value,
            scope=self._options.reviewer_commitment_scope,
            statement=statement,
            origin=CommitmentOrigin.Reviewer,
            created=utc_now(),
            expires=None,
            kind=CommitmentKind.Declined if upheld else CommitmentKind.Undertaken,
            capability=stop.capability,
            subject=PlanSimilarity.normalise(stop.input),
            user_id=self._options.user_id,
            resolves=transition.id,
        )
        self._commitments.append(commitment)
        effect = stop.effect.name.lower()
        if upheld:
            text = (f"Upheld {effect} of {short_id(transition.id)} ('{conduct}'). Reviewer commitment "
                    f"#{short_id(commitment.id)} [{value}] declines similar plans.")
        else:
            text = (f"Overturned {effect} of {short_id(transition.id)} ('{conduct}'). Reviewer commitment "
                    f"#{short_id(commitment.id)} [{value}] clears similar plans; retry the request to run it.")
        return CapabilityResult.ok(self.name, TextData(value=text))


__all__ = [
    "BaselineApproveTool",
    "BaselineShowTool",
    "ConscienceReviewTool",
    "ConscienceShowTool",
    "ContextTool",
    "DriftTool",
    "ExplainTool",
    "MemorySetTool",
    "MemoryShowTool",
    "ReplayTool",
    "pending_escalations",
]

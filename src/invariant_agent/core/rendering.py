"""Renders transitions as human-readable text for the replay and explain capabilities."""

from __future__ import annotations

from typing import Iterable, Optional

from ._util import cs_str, is_blank, is_empty
from .model.enums import InvariantScope, TransitionEventStage, TransitionStatus
from .model.transition import Transition, TransitionEvent


class _Builder:
    """Minimal ``StringBuilder`` so the formatting reads like the C# original."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def line(self, text: str = "") -> "_Builder":
        self._parts.append(text + "\n")
        return self

    def append(self, text: str) -> "_Builder":
        self._parts.append(text)
        return self

    def __str__(self) -> str:
        return "".join(self._parts)


def _version(state) -> str:
    return str(state.version) if state is not None else "?"


class TransitionFormatter:
    """Renders transitions as text."""

    @staticmethod
    def format_replay(transitions: Iterable[Transition]) -> str:
        """Compact replay: status, input, state version change and rejection reason per transition."""
        sb = _Builder().line("\n==== REPLAY START ====")
        for t in transitions:
            sb.line()
            sb.line(f"[{t.status}] {t.input}")
            after = str(t.after.version) if t.after is not None else "not applied"
            sb.line(f"State: {_version(t.before)} -> {after}")
            if not is_blank(t.reason):
                sb.line(f"Reason: {t.reason}")
        sb.line()
        sb.line("==== REPLAY END ====")
        return str(sb)

    @staticmethod
    def format_replay_verbose(transitions: Iterable[Transition]) -> str:
        """Verbose replay: as :meth:`format_replay` plus the full event trail of each transition."""
        sb = _Builder().line("\n==== REPLAY VERBOSE START ====")
        for t in transitions:
            sb.line()
            sb.line(f"Transition Id={t.id}")
            after = str(t.after.version) if t.after is not None else "not applied"
            sb.line(f"State = {_version(t.before)} -> {after}")
            sb.line(f"Status={t.status}")
            sb.line(f"Input={t.input}")
            if not is_blank(t.reason):
                sb.line(f"Reason={t.reason}")
            for e in t.events:
                sb.line(f"[{e.stage}] {e.message}")
        sb.line()
        sb.line("==== REPLAY VERBOSE END ====")
        return str(sb)

    @staticmethod
    def format_explain(transition: Transition) -> str:
        """Explains a single transition: summary, plan, invariant results grouped by scope, conscience appraisals and
        state impact (version change, memory diff and self-modifications)."""
        sb = _Builder().line("\n==== EXPLAIN START ====")
        sb.line()
        sb.line(f"Input: {transition.input}")
        sb.line(f"Outcome: {transition.status}")
        sb.line()
        _append_summary(sb, transition)
        _append_plan(sb, transition)
        _append_governance(sb, transition)
        _append_conscience(sb, transition)
        _append_state_impact(sb, transition)
        sb.line()
        sb.line("==== EXPLAIN END ====")
        return str(sb)


# One-paragraph narrative of what the runtime did and how the transition ended.
def _append_summary(sb: _Builder, t: Transition) -> None:
    capability = (t.proposed_action.capability if t.proposed_action else None) or "unknown"
    sb.line("Summary:")
    if t.status == TransitionStatus.Rejected:
        stopped_by_conscience = any(a.stopped for a in t.appraisals or []) and not t.blocking_invariants
        if stopped_by_conscience:
            sb.line(f"  The runtime planned to use '{capability}'; every control passed, but the conscience "
                    "stopped the transition.")
        elif t.outcome is None:
            sb.line(f"  The runtime planned to use '{capability}', but the transition was rejected before execution.")
        else:
            sb.line(f"  The runtime used '{capability}', but the transition was rejected before its state was "
                    "committed.")
        if not is_blank(t.reason):
            sb.line(f"  {t.reason}")
    elif t.status == TransitionStatus.Completed:
        sb.line(f"  The runtime used '{capability}' and completed the transition successfully.")
    else:
        sb.line(f"  The runtime used '{capability}' and ended with status '{t.status}'.")
    sb.line()


# The planner's proposed capability and input.
def _append_plan(sb: _Builder, t: Transition) -> None:
    sb.line("Plan:")
    if t.proposed_action is None:
        sb.line("  No action was proposed.")
    else:
        sb.line(f"  Capability: {t.proposed_action.capability}")
        if not is_blank(t.proposed_action.input):
            sb.line(f"  Input: {t.proposed_action.input}")
        # HIG-C may have replaced the planner's proposal with a governed alternative.
        replaced = next((e for e in t.events if e.stage == TransitionEventStage.Planning
                         and e.message.startswith("Governed alternative selected")), None)
        if replaced is not None and "PreviousCapability" in (replaced.metadata or {}):
            previous = cs_str(replaced.metadata["PreviousCapability"])
            sb.line(f"  Governed alternative to the planner's proposal '{previous}'.")
    sb.line()


# Invariant results for each lifecycle checkpoint, in pipeline order.
def _append_governance(sb: _Builder, t: Transition) -> None:
    _append_invariant_events(sb, t, InvariantScope.Plan, "Plan invariants")
    _append_invariant_events(sb, t, InvariantScope.Execution, "Execution invariants")
    _append_invariant_events(sb, t, InvariantScope.SelfModification, "Self-modification invariants")
    _append_invariant_events(sb, t, InvariantScope.Reduction, "Reduction invariants")
    _append_conflicts(sb, t)


# HIG-C conflict resolutions (IBAAM 3 revised, Section 7.4): selected candidate, its profile, overrides, escalation.
def _append_conflicts(sb: _Builder, t: Transition) -> None:
    if not t.conflicts:
        return
    sb.line("Conflict resolution (HIG-C):")
    for c in t.conflicts:
        text = (f"  - {c.scope} [{c.operational_context}]: selected {c.selected}"
                + (" " if is_empty(c.selected_capability) else f" '{c.selected_capability}' ")
                + f"(profile {c.profile})"
                + (f"; overrides {', '.join(c.overridden)}" if c.overridden else "")
                + ("; ESCALATED for human review" if c.escalated else ""))
        sb.line(text)
        if not is_blank(c.reason):
            sb.line(f"    {c.reason}")
    for e in t.events:
        if e.stage == TransitionEventStage.Control and e.message.startswith("Override authorised:"):
            if e.metadata and "Justification" in e.metadata:
                sb.line(f"    {cs_str(e.metadata['Justification'])}")
    sb.line()


# Conscience appraisals (IBAAM 6): stance, effect and the justification generated from the tension components.
def _append_conscience(sb: _Builder, t: Transition) -> None:
    if not t.appraisals:
        return
    sb.line("Conscience:")
    for record in t.appraisals:
        appraisal = record.appraisal
        stance = appraisal.stance.name if appraisal is not None else ""
        sb.line(f"  - {record.point} (attempt {record.attempt}) {record.capability}: {stance} -> {record.effect}"
                + ("" if is_blank(record.note) else f" [{record.note}]"))
        if appraisal is not None:
            sb.line(f"    {appraisal.justification}")
    sb.line()


# Lists Invariant-stage events whose "Scope" metadata matches the requested scope.
def _append_invariant_events(sb: _Builder, t: Transition, scope: InvariantScope, title: str) -> None:
    events = [e for e in t.events if e.stage == TransitionEventStage.Invariant and _has_scope(e, scope)]
    if not events:
        return
    sb.line(title + ":")
    for e in events:
        sb.line(f"  - {e.message}")


# True if the event's "Scope" metadata equals the given scope (case-insensitive).
def _has_scope(e: TransitionEvent, scope: InvariantScope) -> bool:
    return bool(e.metadata) and "Scope" in e.metadata and cs_str(e.metadata["Scope"]).lower() == scope.name.lower()


# Version change, memory diff and any self-modification events.
def _append_state_impact(sb: _Builder, t: Transition) -> None:
    sb.line("State Impact:")
    if t.after is None:
        sb.line("  No state change was applied.")
        return
    sb.line(f"  Version changed from {_version(t.before)} to {t.after.version}.")
    _append_memory_diff(sb, t)
    for e in t.events:
        if e.stage == TransitionEventStage.SelfModification:
            target = _metadata_string(e, "Target") or "Unknown"
            operation = _metadata_string(e, "Operation") or "Changed"
            key = _metadata_string(e, "Key") or ""
            sb.line(f"  Self modification: {target}.{operation} {key}")


def _metadata_string(e: TransitionEvent, key: str) -> Optional[str]:
    if key not in e.metadata:
        return None
    value = e.metadata[key]
    return None if value is None else cs_str(value)


# Lists added (+), removed (-) and changed (~) memory keys between Before and After.
def _append_memory_diff(sb: _Builder, t: Transition) -> None:
    if t.before is None or t.after is None:
        return
    before, after = t.before.memory, t.after.memory
    changes = []
    for key in sorted(set(before) | set(after)):
        had_before, has_after = key in before, key in after
        if not had_before and has_after:
            changes.append(f"  + {key}={cs_str(after[key])}")
        elif had_before and not has_after:
            changes.append(f"  - {key}")
        elif before[key] != after[key]:
            changes.append(f"  ~ {key}={cs_str(before[key])} -> {cs_str(after[key])}")
    if not changes:
        return
    sb.line("Memory Changes:")
    for change in changes:
        sb.line(change)

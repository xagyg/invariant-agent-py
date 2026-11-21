"""Level-criticality consistency (IBAAM 3 revised, Section 7.3) and the operational context in force."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Iterable

from .._util import is_blank, utc_now
from ..abstractions import IInvariant
from ..model.enums import InvariantCriticality, InvariantLayer, OperationalContext


class HierarchyConsistency:
    """Every invariant below the fundamental level must have criticality below Catastrophic in every operational
    context.

    An invariant whose violation could be catastrophic belongs at the fundamental level. Enforcing this rules out the
    case where strict layering lets a low-criticality fundamental violation outrank a catastrophic mission violation.
    The check runs when the evaluator is built (i.e. whenever the hierarchy or criticality table changes), not at
    every decision.
    """

    @staticmethod
    def check(invariants: Iterable[IInvariant]) -> list[str]:
        """Returns one message per inconsistency; empty when the hierarchy is consistent."""
        problems = []
        for invariant in invariants:
            if invariant.layer == InvariantLayer.Fundamental:
                continue
            catastrophic_in = [c for c in OperationalContext
                               if invariant.criticality_in(c) >= InvariantCriticality.Catastrophic]
            if catastrophic_in:
                problems.append(
                    f"{invariant.name} is at level {invariant.layer} but has Catastrophic criticality in "
                    f"{', '.join(c.name for c in catastrophic_in)}; move it to Fundamental or lower its criticality.")
        return problems


class OperationalContextState:
    """The HIG-C operational context currently in force (IBAAM 3 revised, Section 5): Normal, Degraded, Emergency,
    Recovery or Maintenance.

    The runtime stamps :attr:`current` onto every transition when it starts, so conflicts are resolved under the
    context that was in force and replay reproduces that decision. Context is governance state, like the approved
    drift baseline: it is set by an operator (the ``context`` capability, which planners cannot invoke) or by
    configuration (``--operational-context Degraded``), never by the agent itself.

    The value lives in memory; on restart it returns to the configured default.
    """

    def __init__(self, initial: OperationalContext = OperationalContext.Normal, reason: str = "default") -> None:
        self._gate = threading.Lock()
        self._current = initial
        self._reason = reason
        self._since = utc_now()

    @property
    def current(self) -> OperationalContext:
        """The context currently in force."""
        with self._gate:
            return self._current

    @property
    def reason(self) -> str:
        """Why the current context was set."""
        with self._gate:
            return self._reason

    @property
    def since(self) -> datetime:
        """When the current context was set (UTC)."""
        with self._gate:
            return self._since

    def set(self, context: OperationalContext, reason: str) -> OperationalContext:
        """Changes the context in force. Takes effect from the next transition. Returns the context replaced."""
        with self._gate:
            previous = self._current
            self._current = context
            self._reason = "set by operator" if is_blank(reason) else reason.strip()
            self._since = utc_now()
            return previous

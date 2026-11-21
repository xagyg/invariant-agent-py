"""Interactive console REPL for the governed runtime (the C# ``GovernedAgentRuntimeHostedService``)."""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional, TextIO

from ..core._util import is_blank, short_id
from ..core.model.enums import ConscienceEffect, TransitionStatus
from ..core.model.transition import Transition
from .governed_agent_runtime import GovernedAgentRuntime

_RESET = "\x1b[0m"
_COLOURS = {
    "Input": "\x1b[96m",            # cyan
    "Planning": "\x1b[96m",
    "Invariant": "\x1b[93m",        # yellow
    "Control": "\x1b[93m",
    "Execution": "\x1b[92m",        # green
    "Reduction": "\x1b[96m",
    "SelfModification": "\x1b[94m",  # blue
    "Lifecycle": "\x1b[97m",        # white
    "Conscience": "\x1b[91m",       # red
}
_GREY = "\x1b[37m"
_RED = "\x1b[91m"
_CYAN = "\x1b[96m"


def _supports_colour(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        os.system("")  # enables ANSI escape sequences in the Windows console
    return True


class GovernedAgentRepl:
    """Each line of input runs one governed transition; the transition's events are printed with stage-specific
    colours. ``exit``/``quit`` stops the loop; ``clear`` clears the screen."""

    def __init__(self, runtime: GovernedAgentRuntime, input_fn: Callable[[str], str] = input,
                 out: Optional[TextIO] = None, colour: Optional[bool] = None) -> None:
        self._runtime = runtime
        self._input = input_fn
        self._out = out or sys.stdout
        self._colour = _supports_colour(self._out) if colour is None else colour

    def _paint(self, text: str, colour: str) -> str:
        return f"{colour}{text}{_RESET}" if self._colour else text

    def _print(self, text: str = "") -> None:
        print(text, file=self._out)

    def run(self) -> None:
        """Runs the REPL loop until end of input or the user exits."""
        self._print("=== InvariantAgent REPL ===")
        self._print("Commands:")
        self._print("  exit  - quit")
        self._print("  clear - clear screen")
        self._print(f"Resumed at state version {self._runtime.state.version}.")
        self._print()
        while True:
            try:
                line = self._input(self._paint("agent> ", _CYAN))
            except (EOFError, KeyboardInterrupt):
                self._print()
                return
            if is_blank(line):
                continue
            line = line.strip()
            if line in ("exit", "quit"):
                return
            if line == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            self.run_one(line)

    def run_one(self, line: str) -> Transition:
        """Runs one input and prints its event trail, id, status and (for rejections) the reason."""
        context = self._runtime.run(line)
        transition = context.transition
        for e in transition.events:
            self._print(self._paint(f"[{e.stage}] ", _COLOURS.get(e.stage.name, _GREY)) + e.message)
        self._print(f"Transition Id: {transition.id}")
        self._print(f"Status: {transition.status}")
        if transition.status == TransitionStatus.Rejected and not is_blank(transition.reason):
            self._print(f"Reason: {transition.reason}")
        self._print_review_commands(transition)
        self._print()
        return transition

    # When the conscience escalated, prints the review commands ready to copy. Deliberately print-only: escalation
    # means someone decides separately, so the REPL never prompts the requester to approve their own request.
    def _print_review_commands(self, transition: Transition) -> None:
        escalation = transition.appraisals[-1] if transition.appraisals else None
        if escalation is None or escalation.effect != ConscienceEffect.Escalated:
            return
        id = short_id(transition.id)
        self._print(self._paint("Held for human review. To resolve, run one of:", _RED))
        self._print(f"  conscience-review {id} uphold <note>     (decline this and similar requests)")
        self._print(f"  conscience-review {id} overturn <note>   (clear it; then retry the request)")
        self._print("  conscience pending                           (see everything awaiting review)")

"""Parses command-style input (``"capability arguments"``) into an :class:`AgentAction`."""

from __future__ import annotations

from .model.agent import AgentAction


class AgentActionParser:
    """Parses command-style input into an action."""

    @staticmethod
    def parse(input: str) -> AgentAction:
        """Splits input on the first space: the first token (lower-cased) is the capability and the remainder is passed
        through unchanged as the capability input.

        Assumes non-empty input; the REPL filters blank lines before calling the runtime. Empty input yields an action
        with an empty capability, which the plan invariants reject.
        """
        trimmed = (input or "").strip()
        parts = [p for p in trimmed.split(" ", 1) if p != ""]
        if not parts:
            return AgentAction(capability="", input="")
        capability = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        return AgentAction(capability=capability, input=args)


def parse_action(input: str) -> AgentAction:
    """Function form of :meth:`AgentActionParser.parse`."""
    return AgentActionParser.parse(input)

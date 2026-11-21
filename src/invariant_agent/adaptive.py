"""IA2 Adaptive layer: planners. Every planner is an untrusted cognitive generator whose proposals are governed."""

from __future__ import annotations

import json
import os
import urllib.request
from abc import abstractmethod
from typing import Optional

from ._version import USER_AGENT
from .core._util import is_blank
from .core.abstractions import IPlanner
from .core.model.agent import AgentAction, StateProjection
from .core.model.planning import PlannerContext, PlannerRevision
from .core.parsing import AgentActionParser


class Planner(IPlanner):
    """Base class for planners (IBAAM Adaptive layer). Wraps plan generation and converts planner exceptions into an
    error action instead of crashing the runtime.

    A failed plan yields capability ``"planner"``, which is not a registered capability, so the transition is rejected
    by ``AllowedCapabilityInvariant`` during plan validation.
    """

    def plan(self, context: PlannerContext) -> AgentAction:
        try:
            return self.generate_plan(context)
        except Exception as ex:  # noqa: BLE001 - planner failures are governed, not fatal
            return self.handle_planner_error(ex, context)

    @abstractmethod
    def generate_plan(self, context: PlannerContext) -> AgentAction:
        """Generates the proposed action for the given context."""

    def handle_planner_error(self, ex: Exception, context: PlannerContext) -> AgentAction:
        """Converts a planning exception into an error action that governance will reject."""
        return AgentAction(capability="planner", input=f"Planner error: {ex}", error=str(ex))


class CommandPlanner(Planner):
    """Deterministic planner that treats input as a literal command (``capability arguments``).

    The default planner; useful for reproducible runs, tests and replay.
    """

    name = "command"

    def generate_plan(self, context: PlannerContext) -> AgentAction:
        return AgentActionParser.parse(context.input)


class RulePlanner(Planner):
    """Rule-based planner placeholder. Currently identical to :class:`CommandPlanner`; intended as an extension point
    for deterministic rule-based action selection."""

    name = "rule"

    def generate_plan(self, context: PlannerContext) -> AgentAction:
        return AgentActionParser.parse(context.input)


class LlmPlanner(Planner):
    """Base class for LLM-backed planners. Builds a prompt from the bounded state projection, sends it to the model
    and parses a ``capability:`` / ``input:`` response into an :class:`AgentAction`.

    The LLM is treated as an untrusted cognitive generator: whatever it returns is still subject to full invariant
    governance before execution.
    """

    def generate_plan(self, context: PlannerContext) -> AgentAction:
        prompt = self.build_prompt(context.state, context.input)
        if context.revision is not None:
            prompt += self.build_revision_prompt(context.revision)
        return self.parse_action(self.complete(prompt))

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Sends the prompt to the underlying model and returns its raw text response."""

    def build_prompt(self, state: StateProjection, input: str) -> str:
        """Builds the planner prompt: role, available capabilities, required output format, the projected runtime
        state and the user input."""
        lines = [
            "You are a governed agent planner.",
            "Your role is to select the single best capability",
            "and input for the current request.",
            "Available capabilities:",
            "- echo",
            "- search",
            "- calc",
            "- replay",
            "- drift",
            "- memory-set",
            "- memory-show",
            "You operate under invariant governance.",
            "Return ONLY the following format:",
            "capability: <capability-name>",
            "input: <capability-input>",
            "Do not explain your reasoning.",
            "Do not return markdown.",
            "Do not return JSON.",
            "",
            "Current runtime state:",
            f"Version: {state.version}",
        ]
        if not is_blank(state.goal):
            lines.append(f"Goal: {state.goal}")
        if not is_blank(state.memory_summary):
            lines.append(f"Memory: {state.memory_summary}")
        if not is_blank(state.last_outcome):
            lines.append(f"Last outcome: {state.last_outcome}")
        if state.active_policies:
            lines.append(f"Policies: {', '.join(state.active_policies)}")
        lines += ["", "User input:", input]
        return "\n".join(lines) + "\n"

    def build_revision_prompt(self, revision: PlannerRevision) -> str:
        """Appends the conscience's appraisal of the previous proposal (IBAAM 6, Section 6.3), so the model can re-plan
        around the concern rather than only being blocked by it."""
        previous = revision.previous
        lines = [
            "",
            "Conscience feedback on your previous proposal:",
            f"Previous capability: {previous.capability if previous else ''}",
            f"Previous input: {previous.input if previous else ''}",
        ]
        if revision.values_at_stake:
            lines.append(f"Values at stake: {', '.join(revision.values_at_stake)}")
        lines += [
            f"Appraisal: {revision.justification}",
            "",
            "Propose a different action that still serves the user's request but addresses this concern.",
            "If no such action exists, propose the most conservative option (for example echo an explanation).",
            "Use the same output format.",
        ]
        return "\n".join(lines) + "\n"

    def parse_action(self, text: str) -> AgentAction:
        """Parses the model response (``capability:`` and ``input:`` lines) into an action.

        :raises StopIteration: either line is missing. :class:`Planner` turns this into an error action that is then
            rejected.
        """
        lines = [line.strip() for line in text.split("\n") if not is_blank(line)]
        capability_line = next((x for x in lines if x.startswith("capability:")), None)
        input_line = next((x for x in lines if x.startswith("input:")), None)
        if capability_line is None or input_line is None:
            raise ValueError("Sequence contains no matching element")
        return AgentAction(capability=capability_line[len("capability:"):].strip(),
                           input=input_line[len("input:"):].strip())


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https endpoints
        return json.loads(response.read().decode("utf-8"))


class OpenAiPlanner(LlmPlanner):
    """LLM planner backed by the OpenAI Chat Completions API (standard library HTTP, no SDK required)."""

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5",
                 base_url: str = "https://api.openai.com/v1") -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")

    def complete(self, prompt: str) -> str:
        response = _post_json(
            f"{self._base_url}/chat/completions",
            {"model": self._model, "messages": [{"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {self._api_key}"},
        )
        return response["choices"][0]["message"]["content"] or ""


class GeminiPlanner(LlmPlanner):
    """LLM planner backed by Google Gemini (``generateContent`` REST API, no SDK required)."""

    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-pro") -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self._model = model

    def complete(self, prompt: str) -> str:
        response = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent",
            {"contents": [{"parts": [{"text": prompt}]}]},
            {"x-goog-api-key": self._api_key},
        )
        parts = response["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)

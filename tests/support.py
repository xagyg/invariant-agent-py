"""Test helpers: fixtures built from the production composition root, and test planners.

Mirrors ``TestFixture`` and ``ConscienceTestHost`` in the C# test project. Each call creates an isolated agent.
"""

from __future__ import annotations

from typing import Callable

from invariant_agent import build_agent
from invariant_agent.core.abstractions import IPlanner
from invariant_agent.core.model import AgentAction, PlannerContext
from invariant_agent.core.parsing import AgentActionParser


def create_fixture(**kwargs):
    """A fresh, isolated, in-memory agent (the C# ``TestFixture.Create()``)."""
    return build_agent(**kwargs)


def create_host(configure=None, shared_store=None, shared_commitments=None, planner=None, remove_invariant=None,
                data_directory=None):
    """The C# ``ConscienceTestHost.Create``: conscience options, a shared transition or commitment store, a custom
    planner, or a reduced invariant set (leave-one-out)."""
    return build_agent(
        data_directory,
        configure,
        planner=planner,
        transition_store=shared_store,
        commitment_store=shared_commitments,
        remove_invariants=(remove_invariant,) if remove_invariant else (),
    )


class RevisingPlanner(IPlanner):
    """Command planner that responds to conscience feedback: when asked to revise a memory write, it proposes writing
    the same value to an allowed key instead (``notes``). Used to test deliberative influence."""

    name = "revising"

    def __init__(self) -> None:
        self.revision_requests = 0

    def plan(self, context: PlannerContext) -> AgentAction:
        action = AgentActionParser.parse(context.input)
        if context.revision is not None:
            self.revision_requests += 1
            value = action.input.split("=", 1)[1] if "=" in action.input else action.input
            return AgentAction(capability="memory-set", input=f"notes={value}")
        return action


class StubbornPlanner(IPlanner):
    """Planner that, when asked to revise, only makes cosmetic changes (so revision never resolves the concern)."""

    name = "stubborn"

    def plan(self, context: PlannerContext) -> AgentAction:
        action = AgentActionParser.parse(context.input)
        if context.revision is not None:
            action.input += "!" * context.revision.attempt
        return action


class ScriptedPlanner(IPlanner):
    """Planner whose proposal is a fixed function of the context, standing in for a model planner."""

    name = "scripted"

    def __init__(self, plan: Callable[[PlannerContext], AgentAction]) -> None:
        self._plan = plan

    def plan(self, context: PlannerContext) -> AgentAction:
        return self._plan(context)


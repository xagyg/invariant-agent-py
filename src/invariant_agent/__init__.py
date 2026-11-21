"""InvariantAgent (Python): a reference implementation of the Invariant-Bounded Agent Alignment Model (IBAAM).

A governed agent runtime where behavioural constraints remain external to adaptive cognition. The planner (LLM or
otherwise) is an untrusted component that proposes actions; invariants, runtime governance and state transition
control stay outside it.

Quick start::

    from invariant_agent import build_agent

    agent = build_agent()                 # in memory; build_agent("data") persists to ./data
    context = agent.run("calc 10+20")
    print(context.transition.status, context.transition.outcome.result)

Package layout (mirroring the C# projects of InvariantAgent2):

* :mod:`invariant_agent.core` - contracts, models, transitions, projections, HIG-C control, drift, replay
* :mod:`invariant_agent.safety` - the invariant set (Agent Behavioural Controls)
* :mod:`invariant_agent.conscience` - functional conscience (IBAAM 6)
* :mod:`invariant_agent.capabilities` - tools and the capability registry
* :mod:`invariant_agent.execution` - capability execution and mediation
* :mod:`invariant_agent.adaptive` - planners
* :mod:`invariant_agent.observability` - drift analysis and stability evaluation
* :mod:`invariant_agent.storage` - JSONL persistence (compatible with the C# data files)
* :mod:`invariant_agent.runtime` - the governed runtime and the REPL
* :mod:`invariant_agent.hosting` - the composition root (:func:`build_agent`)
"""

from ._version import __version__
from .hosting import InvariantAgent, build_agent
from .runtime import GovernedAgentRepl, GovernedAgentRuntime

__all__ = ["GovernedAgentRepl", "GovernedAgentRuntime", "InvariantAgent", "__version__", "build_agent"]

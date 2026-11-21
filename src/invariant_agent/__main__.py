"""Console entry point: hosts the governed runtime as an interactive REPL.

Usage::

    python -m invariant_agent [--data-dir PATH] [--no-conscience] [--operational-context Degraded]
                              [--planner command|openai|gemini] [--memory]

Each option can also be set through the environment, mirroring the C# configuration keys:
``INVARIANTAGENT__DATADIRECTORY``, ``INVARIANTAGENT__CONSCIENCE__ENABLED=false`` and
``INVARIANTAGENT__OPERATIONALCONTEXT``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from ._version import __version__
from .adaptive import CommandPlanner, GeminiPlanner, OpenAiPlanner
from .core.model.enums import OperationalContext
from .hosting import build_agent
from .runtime import GovernedAgentRepl


def _env(name: str) -> Optional[str]:
    for key, value in os.environ.items():
        if key.upper() == name:
            return value
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Configures the services, then runs the REPL until the user exits."""
    parser = argparse.ArgumentParser(prog="invariant-agent", description=(
        "Invariant-bounded agent runtime (IBAAM reference implementation, Python port of InvariantAgent2)."))
    parser.add_argument("--data-dir", dest="data_dir", default=None,
                        help="persistent data directory (default: ./data). History, baselines and commitments "
                             "survive restarts and can be replayed later.")
    parser.add_argument("--memory", action="store_true", help="keep everything in memory (no persistence)")
    parser.add_argument("--no-conscience", action="store_true",
                        help="run with invariants only (disable the IBAAM 6 functional conscience)")
    parser.add_argument("--operational-context", default=None,
                        help="HIG-C operational context at start-up: " +
                             ", ".join(c.name for c in OperationalContext))
    parser.add_argument("--planner", choices=("command", "openai", "gemini"), default="command",
                        help="planner to use (LLM planners read OPENAI_API_KEY / GEMINI_API_KEY)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    data_dir = None
    if not args.memory:
        data_dir = args.data_dir or _env("INVARIANTAGENT__DATADIRECTORY") or os.path.join(os.getcwd(), "data")

    # Functional conscience (IBAAM 6) is on by default.
    conscience_enabled = not args.no_conscience and (_env("INVARIANTAGENT__CONSCIENCE__ENABLED") or "").lower() != "false"

    # HIG-C operational context in force at start-up (default Normal).
    context = OperationalContext.parse(args.operational_context or _env("INVARIANTAGENT__OPERATIONALCONTEXT")) \
        or OperationalContext.Normal

    planner = {"command": CommandPlanner, "openai": OpenAiPlanner, "gemini": GeminiPlanner}[args.planner]()

    def configure(options):
        options.enabled = conscience_enabled

    agent = build_agent(data_dir, configure, context, planner=planner)
    GovernedAgentRepl(agent.runtime).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

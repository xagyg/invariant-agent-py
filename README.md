# invariant-agent-py

A Python port of [InvariantAgent2 (IA2)](https://github.com/xagyg/InvariantAgent2), the reference implementation of
the **Invariant-Bounded Agent Alignment Model (IBAAM)**.

IA2 explores governed agent runtimes where behavioural constraints remain external to adaptive cognition. Rather than
allowing an LLM to directly govern system behaviour, all actions, state transitions and self-modifications are
evaluated under invariant-based runtime control. The planner (LLM or otherwise) is an untrusted cognitive component
that proposes actions; runtime governance, invariant enforcement and state transition control stay outside it.

This port covers the whole C# solution: the governed runtime, HIG-C conflict resolution (IBAAM 3 revised), drift and
stability measurement (IBAAM 2), the functional conscience (IBAAM 6), JSONL persistence, replay validation, the tools
and the REPL. It has no third-party runtime dependencies and runs on Python 3.10+.

## Quick start

```bash
pip install -e ".[test]"        # or run from the source tree with PYTHONPATH=src
python -m invariant_agent       # REPL; history is persisted to ./data
```

```text
agent> calc 10+20
[Lifecycle] Phase: InputReceived
[Input] calc 10+20
[Lifecycle] Phase: Planning
[Planning] Capability=calc
[Lifecycle] Phase: PlanValidation
[Invariant] NoDeleteInvariant: Passed
[Invariant] AllowedCapabilityInvariant: Passed
[Invariant] OperatorOnlyCapabilityInvariant: Passed
[Invariant] ServiceObligationInvariant: Passed
[Control] Plan invariants passed
[Conscience] Plan appraisal: Proceed (tension 0.25) -> Proceeded
[Lifecycle] Phase: Execution
[Execution] 30
[Lifecycle] Phase: ExecutionValidation
[Invariant] SuccessOutcomeInvariant: Passed
[Invariant] NonEmptyOutcomeInvariant: Passed
[Invariant] ServiceObligationInvariant: Passed
[Control] Execution invariants passed
[Lifecycle] Phase: SelfModificationValidation
[Invariant] AllowedMemoryKeyInvariant: Passed
[Invariant] ServiceObligationInvariant: Passed
[Control] SelfModification invariants passed
[Lifecycle] Phase: Reduction
[Reduction] ProposedVersion=1
[Invariant] BehaviouralDriftBoundInvariant: Passed
[Invariant] RejectedTransitionCannotCommitInvariant: Passed
[Invariant] ServiceObligationInvariant: Passed
[Control] Reduction invariants passed
[Conscience] Answer appraisal: Proceed (tension 0.00) -> Proceeded
[Lifecycle] CommittedVersion=1
[Lifecycle] Phase: Completed
Transition Id: 6e0b0755-3e47-46cd-90cd-81209c1434e3
Status: Completed
```

The output is the same as the C# REPL's, line for line, tensions included.

### Command-line options

| Option | Environment variable | Meaning |
|---|---|---|
| `--data-dir PATH` | `INVARIANTAGENT__DATADIRECTORY` | Persistent data directory (default `./data`) |
| `--memory` | | Keep everything in memory |
| `--no-conscience` | `INVARIANTAGENT__CONSCIENCE__ENABLED=false` | Run with invariants only |
| `--operational-context Degraded` | `INVARIANTAGENT__OPERATIONALCONTEXT` | HIG-C context at start-up |
| `--planner command\|openai\|gemini` | `OPENAI_API_KEY`, `GEMINI_API_KEY` | Planner (default: the deterministic command planner) |

### As a library

```python
from invariant_agent import build_agent

agent = build_agent()                       # in memory; build_agent("data") persists to ./data
context = agent.run("memory-set goal=learn IBAAM")
t = context.transition
print(t.status, agent.runtime.state.memory["goal"])  # Completed learn IBAAM

report = agent.drift_analyzer.analyze(agent.transition_store.get_all())
print(report.stability.region)
```

`build_agent()` is the composition root (the C# `AddInvariantAgent`). It returns an `InvariantAgent` holding the
runtime and every service it was built from, and accepts overrides for the planner, reducer, stores, conscience
options, escalation protocols, a value judge, extra invariants or capabilities, and invariants to leave out:

```python
from invariant_agent.adaptive import OpenAiPlanner
from invariant_agent.safety import AllowedMemoryKeyInvariant

agent = build_agent(
    "data",
    configure_conscience=lambda o: setattr(o, "max_revisions", 1),
    planner=OpenAiPlanner(),                       # reads OPENAI_API_KEY
    remove_invariants=[AllowedMemoryKeyInvariant], # leave-one-out experiments
)
```

## Core idea

LLMs are treated as **cognitive generators**, not governing authorities. Every proposed action passes through
invariant-bounded runtime control before execution or state mutation occurs.

```text
invariants  >  runtime governance  >  planner  >  capabilities
```

Invariants define admissible behavioural boundaries. The runtime governs transitions. Planners propose actions.
Capabilities perform execution. Reducers assimilate approved state evolution.

### Governed transition lifecycle

```text
Input
 -> Planning (proposal, plus alternatives from an IAlternativePlanner)
 -> Pre-Control (HIG-C: proposal vs alternatives vs declining)
 -> Conscience: plan appraisal (Proceed | Revise -> re-plan -> Pre-Control again | Escalate | Veto)
 -> Capability Execution
 -> Post-Control
 -> State Assimilation
 -> Conscience: answer appraisal (Proceed | caveat | Escalate | Veto)
 -> Transition Persistence
```

Every transition, completed or rejected, records the prior state, proposed action, execution outcome, invariant
decisions, HIG-C conflict records, conscience appraisals, adaptive modifications and resulting state, and is available
for replay, auditing, drift analysis and governance inspection.

### Hierarchical and Context-Aware Invariant Governance (HIG-C)

Implements IBAAM 3 (revised edition). Invariants have a level (`Fundamental > Mission > Behavioural >
AdaptiveHeuristic`), a violation degree in [0, 1], and a criticality that can depend on the operational context. At
each checkpoint the evaluator compares the proposed action, any planner alternatives and declining, and selects the
least-bad by level first, then criticality. `ServiceObligationInvariant` gives declining a cost, so minor lower-level
violations can proceed as witnessed, audited overrides; fundamental violations never proceed autonomously and are
escalated for human review when every option breaks one.

### Functional conscience (IBAAM 6)

The conscience weighs what the agent **should** do inside the boundary the invariants set on what it **may** do. It
has its own contract (`IConscience`, not `IInvariant`), runs only after the relevant invariants pass, and can never
authorise anything they block. `HistoryCalibratedConscience` is deterministic and calibrated from governance history:
violation risk, forecast drift and commitment conflict, amplified by trend pressure, select **Proceed**, **Revise**,
**Escalate** (held for `conscience-review <id> uphold|overturn <note>`) or **Veto**.

```text
agent> memory-set password=hunter2
[Conscience] Plan appraisal: Proceed (tension 0.20) -> Proceeded
[Control] SelfModification invariants failed: AllowedMemoryKeyInvariant: Memory key 'password' is not allowed.

agent> memory-set password=letmein
[Conscience] Plan appraisal: Revise (tension 0.64) -> Escalated [planner 'command' could not produce a different plan]
Held for human review. To resolve, run one of:
  conscience-review 3e7a2a5d uphold <note>     (decline this and similar requests)
  conscience-review 3e7a2a5d overturn <note>   (clear it; then retry the request)
  conscience pending                           (see everything awaiting review)
```

### Operator-only governance commands

`conscience-review`, `baseline-approve` and `context` change the controls rather than act within them, so they are
operator-only (`ICapability.operator_only`). `OperatorOnlyCapabilityInvariant` lets one run only when the proposed
action is exactly what the operator typed, so a model planner can never clear its own escalation, re-certify its own
drift baseline or declare an emergency. The guard is a control, not a prompt instruction.

## Capabilities

`echo`, `search`, `calc`, `explain`, `replay [N] [verbose] [validate]`, `drift`, `memory-set key=value`, `memory-show`,
`baseline-show`, `baseline-approve` (operator-only), `conscience [commitments | pending | calibration | last | <id>]`,
`conscience-review <id> uphold|overturn [note]` (operator-only), `context [normal|degraded|emergency|recovery|maintenance]
[reason]` (operator-only), and the `http_example` external-service sample. Add your own by passing
`extra_capabilities=[...]` to `build_agent()`.

## Package layout

| Python package | C# project | IBAAM layer |
|---|---|---|
| `invariant_agent.core` | `InvariantAgent.Core` | Contracts, models, transitions, projections, HIG-C control, replay |
| `invariant_agent.safety` | `InvariantAgent.Safety` | Control layer (Agent Behavioural Controls) |
| `invariant_agent.conscience` | `InvariantAgent.Conscience` | Functional conscience, contained by the controls |
| `invariant_agent.execution` | `InvariantAgent.Execution` | Execution layer |
| `invariant_agent.adaptive` | `InvariantAgent.Adaptive` | Adaptive layer (planners) |
| `invariant_agent.capabilities` | `InvariantAgent.Capabilities` | Tools and services |
| `invariant_agent.observability` | `InvariantAgent.Observability` | Drift analysis, stability evaluation |
| `invariant_agent.storage` | `InvariantAgent.Storage` | Persistence |
| `invariant_agent.runtime` | `InvariantAgent.Runtime` | Governed transition lifecycle, REPL |
| `invariant_agent.hosting` / `__main__` | `InvariantAgent.Hosting` / `ConsoleApp` | Composition root, entry point |

## Persistence and interoperability

With a data directory the runtime keeps `transitions.jsonl`, `baselines.jsonl` and `commitments.jsonl` (append-only,
one JSON object per line) and resumes from the last committed state on restart. The files use the same format as the
C# implementation (PascalCase names, numeric enums, ISO timestamps), so the two runtimes can share a data directory.
`data-demo/` is a history recorded by the C# REPL; the test suite reads it and replay-validates every decision.

`replay validate` re-evaluates recorded history against the current invariants (report-only). Each transition records
the drift baseline and operational context in force, and conscience appraisals are recorded, so replay reproduces the
live decisions.

## Tests

```bash
pip install -e ".[test]"
pytest
```

All 117 tests from the C# xUnit suite are ported (`tests/`), plus two interoperability tests against `data-demo/`.

## Differences from the C# implementation

* **Synchronous API.** `runtime.run(input)` replaces `RunAsync`, and planners implement `plan(context)`.
* **Naming.** Members are snake_case; the interface names (`IInvariant`, `IConscience`, `ICapability` ...) are kept so
  the code lines up with the IBAAM papers. Invariants, capabilities and planners can declare `name`, `scope`,
  `layer`, `operator_only` and so on as plain class attributes.
* **Composition.** `build_agent()` replaces .NET dependency injection and hosting.
* **Calculator.** `DataTable.Compute` is replaced by a small, safe arithmetic evaluator with the same semantics for
  `+ - * / %`, parentheses and decimals (`0.1+0.2` is `0.3`, `7/2` is `3.5`).
* **LLM planners** call the OpenAI and Gemini REST APIs with the standard library, so no SDK is needed.
* **Timestamps** are strictly increasing within a process, so timestamp comparisons stay unambiguous on platforms
  with a coarse clock.
* **Line endings.** Rejection summaries join violations with `\n` rather than `Environment.NewLine`.

## Citing

If you use this Python port in your research, please cite it:

```bibtex
@software{invariantagentpy2026,
  author  = {Long, Brad},
  title   = {Invariant Agent 2 (IA2), Python port},
  year    = {2026},
  url     = {https://github.com/xagyg/invariant-agent-py}
}
```

This is a port of InvariantAgent2; please also cite the reference implementation:

```bibtex
@software{invariantagent2026,
  author  = {Long, Brad},
  title   = {Invariant Agent 2 (IA2)},
  year    = {2026},
  url     = {https://github.com/xagyg/InvariantAgent2}
}
```

The project remains intentionally experimental and architecture-focused. Thresholds are placeholders pending
evaluation against a labelled set.

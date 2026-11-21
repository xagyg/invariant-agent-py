"""IA2 Storage: transition, drift-baseline and commitment persistence (JSON Lines, C#-compatible)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Generic, Optional, Type, TypeVar

from .core.abstractions import ICommitmentStore, IDriftBaselineStore, ITransitionStore
from .core.drift import copy_state
from .core.model.agent import AgentState
from .core.model.conscience import Commitment
from .core.model.drift import DriftBaseline
from .core.model.transition import Transition
from .core.serialization import dumps, from_json_value

T = TypeVar("T")


class InvalidDataError(ValueError):
    """A line other than the last in a JSON Lines history cannot be parsed (C# ``InvalidDataException``)."""


class JsonLinesFile(Generic[T]):
    """Append-only JSON Lines file shared by the durable stores. One record per line, serialised with
    :mod:`invariant_agent.core.serialization` so records read back match what was written."""

    def __init__(self, file_path: str | os.PathLike, record_type: Type[T]) -> None:
        """Opens (creating if needed) the file and its directory."""
        self._path = Path(file_path).resolve()
        self._type = record_type
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    @property
    def file_path(self) -> str:
        """Full path of the backing file."""
        return str(self._path)

    def append(self, record: T) -> None:
        """Appends one record as a single line."""
        line = dumps(record)
        with self._lock:
            with open(self._path, "a", encoding="utf-8", newline="") as handle:
                handle.write(line + os.linesep)

    def read_all(self) -> list[T]:
        """Reads every record in file order.

        An unreadable *final* line is skipped: it is what a crash part-way through an append leaves behind. An
        unreadable line anywhere else means the history itself is damaged, so it raises rather than silently dropping
        governance records.

        :raises InvalidDataError: a line other than the last cannot be parsed.
        """
        with self._lock:
            text = self._path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        content = [i for i, line in enumerate(lines) if line.strip()]
        last_content_line = content[-1] if content else -1
        result: list[T] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = from_json_value(self._type, json.loads(line))
            except (ValueError, TypeError, KeyError) as ex:
                if index == last_content_line:
                    continue  # torn write from an interrupted append; the record was never fully stored
                raise InvalidDataError(
                    f"Unreadable record on line {index + 1} of '{self._path}': {ex}") from ex
            if record is not None:
                result.append(record)
        return result


class InMemoryTransitionStore(ITransitionStore):
    """Process-local :class:`ITransitionStore`; history is lost when the process exits. The default store."""

    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    def append(self, transition: Transition) -> None:
        self._transitions.append(transition)

    def get_all(self) -> list[Transition]:
        return self._transitions


class FileTransitionStore(ITransitionStore):
    """Durable :class:`ITransitionStore` that appends each transition as one JSON line (JSONL).

    Thread-safe within one process. ``Any``-typed memory and metadata values come back as strings, numbers and
    booleans, so a reloaded transition replays the same as the live one. Polymorphic members (e.g. ``AgentEvent``)
    are not supported.
    """

    def __init__(self, file_path: str | os.PathLike) -> None:
        self._file = JsonLinesFile(file_path, Transition)

    @property
    def file_path(self) -> str:
        """Full path of the backing file."""
        return self._file.file_path

    def append(self, transition: Transition) -> None:
        self._file.append(transition)

    def get_all(self) -> list[Transition]:
        """Reads and deserialises the whole file on every call."""
        return self._file.read_all()


class FileDriftBaselineStore(IDriftBaselineStore):
    """Durable :class:`IDriftBaselineStore`. Every approval is appended to a JSONL file, so the file is also the audit
    history of baseline approvals; the current baseline is the most recent one."""

    def __init__(self, file_path: str | os.PathLike) -> None:
        self._file = JsonLinesFile(file_path, DriftBaseline)
        self._lock = threading.Lock()
        history = self._file.read_all()
        self._current: Optional[DriftBaseline] = history[-1] if history else None

    @property
    def file_path(self) -> str:
        """Full path of the backing file."""
        return self._file.file_path

    @property
    def current(self) -> Optional[DriftBaseline]:
        with self._lock:
            return self._current

    @property
    def history(self) -> list[DriftBaseline]:
        """Every approved baseline, oldest first."""
        return self._file.read_all()

    def approve(self, state: AgentState, reason: str) -> DriftBaseline:
        baseline = DriftBaseline(state_version=state.version, reason=reason, state=copy_state(state))
        with self._lock:
            self._file.append(baseline)
            self._current = baseline
        return baseline


class FileCommitmentStore(ICommitmentStore):
    """Durable :class:`ICommitmentStore`: every conscience commitment is appended to a JSONL file, so commitments
    outlive restarts and the file is also their audit history.

    Commitments about individuals are sensitive in their own right (IBAAM 6, Section 9) and this file deserves the
    same protection as the transition history.
    """

    def __init__(self, file_path: str | os.PathLike) -> None:
        self._file = JsonLinesFile(file_path, Commitment)
        self._commitments: list[Commitment] = list(self._file.read_all())
        self._lock = threading.Lock()

    @property
    def file_path(self) -> str:
        """Full path of the backing file."""
        return self._file.file_path

    def append(self, commitment: Commitment) -> None:
        with self._lock:
            self._file.append(commitment)
            self._commitments.append(commitment)

    def get_all(self) -> list[Commitment]:
        with self._lock:
            return list(self._commitments)


__all__ = [
    "FileCommitmentStore",
    "FileDriftBaselineStore",
    "FileTransitionStore",
    "InMemoryTransitionStore",
    "InvalidDataError",
    "JsonLinesFile",
]

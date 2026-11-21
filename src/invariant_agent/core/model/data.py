"""Typed capability payloads. Their string form becomes the transition outcome's ``result``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class CapabilityData(ABC):
    """Base type for typed capability payloads."""

    @abstractmethod
    def __str__(self) -> str:
        """Renders the payload as text; this becomes the transition outcome's ``result``."""


@dataclass
class JsonData(CapabilityData):
    """A JSON string payload."""

    #: The raw JSON text.
    json: str = ""

    def __str__(self) -> str:
        return self.json


@dataclass
class ListData(CapabilityData):
    """A list-of-rows payload, rendered as rows separated by ``" | "``."""

    #: The rows.
    rows: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return " | ".join(self.rows)


@dataclass
class TableData(CapabilityData):
    """A tabular payload; cells are comma-joined and rows separated by ``" | "``."""

    #: Table rows, each a list of cells.
    rows: list[list[str]] = field(default_factory=list)

    def __str__(self) -> str:
        return " | ".join(",".join(row) for row in self.rows)


@dataclass
class TextData(CapabilityData):
    """A plain-text payload."""

    #: The text.
    value: str = ""

    def __str__(self) -> str:
        return self.value

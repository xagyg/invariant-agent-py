"""Small helpers that reproduce C#/.NET behaviour the port relies on.

The C# implementation leans on a handful of framework behaviours (enum names in string interpolation,
``string.IsNullOrWhiteSpace``, ``Guid.ToString("N")``, custom numeric formats such as ``0.###``). Keeping them in one
place makes the rest of the port read like the original.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any, Iterable, Optional


class CsEnum(IntEnum):
    """An ``IntEnum`` that renders as its member name, like a C# enum in string interpolation.

    Values are the C# numeric values, because the JSONL history stores enums as numbers.
    """

    def __str__(self) -> str:  # noqa: D105
        return self.name

    def __format__(self, spec: str) -> str:  # noqa: D105
        return format(self.name, spec)

    @classmethod
    def parse(cls, text: Optional[str], ignore_case: bool = True):
        """Equivalent of ``Enum.TryParse``: returns the member named ``text`` (or with that number), else ``None``."""
        if text is None:
            return None
        value = text.strip()
        if not value:
            return None
        for member in cls:
            if member.name == value or (ignore_case and member.name.lower() == value.lower()):
                return member
        try:
            return cls(int(value))
        except (ValueError, TypeError):
            return None


_clock_lock = threading.Lock()
_last_now: Optional[datetime] = None


def utc_now() -> datetime:
    """Current UTC time, strictly increasing within the process.

    ``DateTime.UtcNow`` is precise on .NET; Python's clock can be coarse on some platforms, and the stability
    evaluator compares transition timestamps with ``>``. Nudging equal readings forward by a microsecond keeps the
    order of events unambiguous without changing any observable behaviour.
    """
    global _last_now
    with _clock_lock:
        now = datetime.now(timezone.utc)
        if _last_now is not None and now <= _last_now:
            now = _last_now + timedelta(microseconds=1)
        _last_now = now
        return now


def new_id() -> uuid.UUID:
    """Equivalent of ``Guid.NewGuid()``."""
    return uuid.uuid4()


def short_id(value: uuid.UUID) -> str:
    """``id.ToString("N")[..8]``: the first eight hex digits, used in operator-facing ids."""
    return value.hex[:8]


def is_blank(value: Optional[str]) -> bool:
    """``string.IsNullOrWhiteSpace``."""
    return value is None or value.strip() == ""


def is_empty(value: Optional[str]) -> bool:
    """``string.IsNullOrEmpty``."""
    return value is None or value == ""


def cs_str(value: Any) -> str:
    """Renders a value the way C# string interpolation (``object.ToString()``) would.

    ``null`` becomes the empty string, booleans become ``True``/``False`` and integral doubles drop their ``.0``.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if value != value:  # NaN
            return "NaN"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, CsEnum):
        return value.name
    return str(value)


def fmt_max(value: float, decimals: int = 3) -> str:
    """Custom numeric format ``0.###`` (up to ``decimals`` places, trailing zeros removed)."""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def fmt_fixed(value: float, decimals: int) -> str:
    """Custom numeric formats ``0``, ``0.0``, ``0.00``, ``0.000`` (invariant culture)."""
    text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]
    return text


def fmt_percent0(value: float) -> str:
    """Standard numeric format ``P0`` (e.g. ``0.4`` -> ``40%``)."""
    return f"{value * 100:.0f}%"


def fmt_round_trip(value: datetime) -> str:
    """Date format ``O`` (ISO 8601 round-trip) for a UTC time."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "0Z"


def fmt_universal(value: datetime) -> str:
    """Date format ``u`` (universal sortable), e.g. ``2026-11-21 03:49:00Z``."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def lower_set(values: Iterable[str]) -> set[str]:
    """A case-insensitive (``StringComparer.OrdinalIgnoreCase``) set, represented by lower-cased members."""
    return {v.lower() for v in values if v is not None}


def distinct(values: Iterable[Any], ignore_case: bool = False) -> list[Any]:
    """LINQ ``Distinct()``: first occurrence wins, order preserved."""
    seen: set[Any] = set()
    result = []
    for value in values:
        key = value.lower() if ignore_case and isinstance(value, str) else value
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def clamp(value: float, low: float, high: float) -> float:
    """``Math.Clamp``."""
    return max(low, min(high, value))

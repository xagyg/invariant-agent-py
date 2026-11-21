"""Shared JSON settings for persisting and cloning transitions, states, baselines and commitments.

Every persistence path (transition store, baseline store, commitment store, :func:`clone`) goes through these
functions, so a record read back from disk is equivalent to the one that was written.

The wire format is the one the C# implementation writes with ``System.Text.Json`` defaults: PascalCase property names,
enums as numbers, GUIDs as hyphenated strings and ISO 8601 timestamps. Histories are therefore interchangeable: the
Python runtime can resume from (and replay) a ``data`` directory written by IA2, and the reverse.

Values held in ``Any``-typed slots (``AgentState.memory`` and ``TransitionEvent.metadata``) come back as JSON
primitives (``str``, ``bool``, ``int``, ``float``, ``None``); arrays and objects come back as lists and dicts, the
equivalent of C#'s ``InferredObjectConverter``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import types
import typing
import uuid
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Union

_SNAKE = re.compile(r"_([a-z0-9])")


def json_name(field_name: str) -> str:
    """``approved_at_utc`` -> ``ApprovedAtUtc`` (the C# property name)."""
    return field_name[:1].upper() + _SNAKE.sub(lambda m: m.group(1).upper(), field_name[1:])


# ---------------------------------------------------------------------------------------------------------- writing


def format_datetime(value: datetime) -> str:
    """UTC ISO 8601 with a ``Z`` suffix, as ``DateTime.UtcNow`` serialises."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def to_json_value(value: Any) -> Any:
    """Converts a model object into JSON-compatible Python values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return format_datetime(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {json_name(f.name): to_json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_value(v) for v in value]
    if hasattr(value, "to_json"):
        return value.to_json()
    # A bare object has no members (C# writes {}); anything else falls back to its string form.
    if type(value) is object:
        return {}
    return str(value)


def dumps(value: Any) -> str:
    """Serialises a record as one compact JSON line."""
    return json.dumps(to_json_value(value), ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------------------------------------- reading

_FRACTION = re.compile(r"\.(\d+)")


def parse_datetime(text: str) -> datetime:
    """Parses ``DateTime``/``DateTimeOffset`` JSON as an aware UTC time.

    .NET writes up to 7 fractional digits and trims trailing zeros (``.90605``); the fraction is normalised to the
    6 digits every supported Python version accepts.
    """
    value = text.strip()
    if value.endswith("Z") or value.endswith("z"):
        value = value[:-1] + "+00:00"
    value = _FRACTION.sub(lambda m: "." + (m.group(1) + "000000")[:6], value, count=1)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    module = sys.modules.get(cls.__module__)
    return typing.get_type_hints(cls, globalns=vars(module) if module else None)


def _strip_optional(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin is Union or (sys.version_info >= (3, 10) and origin is types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
        return Any
    return tp


def from_json_value(tp: Any, value: Any) -> Any:
    """Converts JSON-compatible values back into an instance of ``tp``."""
    if value is None:
        return None
    tp = _strip_optional(tp)
    if tp is Any or tp is object:
        return value
    origin = typing.get_origin(tp)
    if origin in (list, typing.List, tuple, set, frozenset, typing.Sequence) or tp in (list, tuple):
        args = typing.get_args(tp)
        item_type = args[0] if args else Any
        return [from_json_value(item_type, v) for v in value]
    if origin in (dict, typing.Dict) or tp is dict:
        args = typing.get_args(tp)
        key_type, value_type = (args if len(args) == 2 else (str, Any))
        return {from_json_value(key_type, k): from_json_value(value_type, v) for k, v in value.items()}
    if isinstance(tp, type):
        if issubclass(tp, Enum):
            if isinstance(value, str):
                member = next((m for m in tp if m.name.lower() == value.lower()), None)
                if member is not None:
                    return member
            return tp(int(value))
        if tp is bool:
            return bool(value)
        if tp is int:
            return int(value)
        if tp is float:
            return float(value)
        if tp is str:
            return value if isinstance(value, str) else str(value)
        if tp is uuid.UUID:
            return uuid.UUID(str(value))
        if tp is datetime:
            return parse_datetime(value)
        if dataclasses.is_dataclass(tp):
            return _dataclass_from_json(tp, value)
    return value


def _dataclass_from_json(cls: type, data: dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object for {cls.__name__}, got {type(data).__name__}.")
    hints = _hints(cls)
    lookup = {k.lower(): v for k, v in data.items()}
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        key = json_name(f.name).lower()
        if key not in lookup:
            continue  # missing members keep their defaults, as System.Text.Json does
        raw = lookup[key]
        if raw is None and f.default is not dataclasses.MISSING and f.default is not None:
            continue  # null for a non-nullable member with a default: keep the default
        if raw is None and f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            continue
        kwargs[f.name] = from_json_value(hints.get(f.name, Any), raw)
    return cls(**kwargs)


def loads(cls: type, text: str) -> Any:
    """Parses one JSON line into an instance of ``cls``."""
    return from_json_value(cls, json.loads(text))


def clone(value: Any) -> Any:
    """Deep copy via a JSON round-trip, so a clone matches what the file store would read back.

    Used, for example, by replay validation so invariants can be re-evaluated on a copy without mutating the stored
    transition. Polymorphic or non-serialisable members (e.g. ``AgentEvent``) may be lost.
    """
    if value is None:
        return None
    return from_json_value(type(value), json.loads(dumps(value)))


__all__ = ["clone", "dumps", "from_json_value", "json_name", "loads", "parse_datetime", "to_json_value"]

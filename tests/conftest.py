"""Pytest configuration: makes the shared helpers in ``support.py`` importable and provides a fresh agent."""

from __future__ import annotations

import pytest

from support import create_fixture


@pytest.fixture
def fixture():
    """A fresh, isolated, in-memory agent (the C# ``TestFixture.Create()``)."""
    return create_fixture()

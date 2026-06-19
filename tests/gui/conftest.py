"""
tests/gui/conftest.py
pytest-Konfiguration fuer GUI-Tests.

Fixtures:
  - fresh_theme: isolierter ThemeManager ohne globalen Zustand
"""

from __future__ import annotations

import os

import pytest

# Offscreen-Rendering falls kein Display vorhanden (CI/Headless-Umgebungen)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui.design.theme import ThemeManager, ThemeMode, reset_theme_manager


@pytest.fixture(autouse=True)
def reset_global_theme():
    """Setzt den globalen ThemeManager nach jedem Test zurueck."""
    yield
    reset_theme_manager()


@pytest.fixture
def fresh_theme() -> ThemeManager:
    """Gibt einen frischen, isolierten ThemeManager zurueck."""
    return ThemeManager(mode=ThemeMode.DARK)

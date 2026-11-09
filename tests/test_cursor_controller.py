"""
Unit tests for Windows Cursor Controller and Safety Mechanisms.
"""

import pytest
from config import SmoothingConfig
from core.cursor_controller import CursorController


def test_cursor_initial_state_disabled():
    cfg = SmoothingConfig()
    cursor = CursorController(cfg)
    # Cursor control must be disabled initially
    assert cursor.is_active is False


def test_cursor_enable_disable_toggle():
    cfg = SmoothingConfig()
    cursor = CursorController(cfg)

    cursor.enable()
    assert cursor.is_active is True

    cursor.disable()
    assert cursor.is_active is False

    new_state = cursor.toggle()
    assert new_state is True
    assert cursor.is_active is True

    new_state = cursor.toggle()
    assert new_state is False
    assert cursor.is_active is False


def test_update_position_when_disabled():
    cfg = SmoothingConfig()
    cursor = CursorController(cfg)

    # When disabled, update_position must return None and not dispatch movement
    res = cursor.update_position((500.0, 500.0))
    assert res is None


def test_update_position_when_none_coords():
    cfg = SmoothingConfig()
    cursor = CursorController(cfg)
    cursor.enable()

    # When coordinates are None (outside zone / no hand), cursor is frozen
    res = cursor.update_position(None)
    assert res is None


def test_cursor_clamping_within_screen_boundaries():
    cfg = SmoothingConfig()
    cursor = CursorController(cfg)
    cursor.enable()

    # Pass out-of-bound coords; must clamp to valid screen range
    w = cursor.screen_width
    h = cursor.screen_height

    # Coordinates beyond right/bottom
    pos = cursor.update_position((w + 500.0, h + 500.0))
    assert pos is not None
    assert pos[0] == w - 1
    assert pos[1] == h - 1

    # Coordinates below zero
    pos2 = cursor.update_position((-100.0, -200.0))
    assert pos2 is not None
    assert pos2[0] == 0
    assert pos2[1] == 0

    cursor.disable()


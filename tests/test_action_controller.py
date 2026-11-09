"""
Unit tests for ActionController native Windows dispatches and safety release.
"""

import pytest
from config import SmoothingConfig
from core.action_controller import ActionController


def test_action_controller_initial_state():
    cfg = SmoothingConfig()
    controller = ActionController(cfg)
    assert controller.is_active is False
    assert controller.is_dragging is False


def test_action_controller_toggle_and_release():
    cfg = SmoothingConfig()
    controller = ActionController(cfg)

    controller.enable()
    assert controller.is_active is True

    controller.disable()
    assert controller.is_active is False
    assert controller.is_dragging is False


def test_action_controller_drag_and_drop_lifecycle():
    cfg = SmoothingConfig()
    controller = ActionController(cfg)
    controller.enable()

    controller.start_drag()
    assert controller.is_dragging is True

    # Re-calling start_drag when already dragging should be idempotent
    controller.start_drag()
    assert controller.is_dragging is True

    controller.stop_drag()
    assert controller.is_dragging is False


def test_action_controller_safety_release_all():
    cfg = SmoothingConfig()
    controller = ActionController(cfg)
    controller.enable()

    controller.start_drag()
    assert controller.is_dragging is True

    # Calling release_all must release drag state immediately
    controller.release_all()
    assert controller.is_dragging is False


def test_action_controller_position_clamping():
    cfg = SmoothingConfig()
    controller = ActionController(cfg)
    controller.enable()

    w = controller.screen_width
    h = controller.screen_height

    pos = controller.update_position((w + 1000, h + 1000))
    assert pos is not None
    assert pos[0] == w - 1
    assert pos[1] == h - 1

    pos_neg = controller.update_position((-500, -500))
    assert pos_neg is not None
    assert pos_neg[0] == 0
    assert pos_neg[1] == 0

    controller.disable()


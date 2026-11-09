"""
Unit tests for Active Rectangle Zone and Coordinate Mapping.
"""

import pytest
from config import ActiveZoneConfig
from core.coordinate_mapper import CoordinateMapper


def test_zone_config_bounds():
    # Width 0.6, Height 0.6 centered at (0.5, 0.5)
    cfg = ActiveZoneConfig(width_ratio=0.6, height_ratio=0.6, center_x_offset=0.0, center_y_offset=0.0)
    assert abs(cfg.x_min - 0.2) < 1e-5
    assert abs(cfg.x_max - 0.8) < 1e-5
    assert abs(cfg.y_min - 0.2) < 1e-5
    assert abs(cfg.y_max - 0.8) < 1e-5


def test_is_inside_zone():
    cfg = ActiveZoneConfig(width_ratio=0.6, height_ratio=0.6, center_x_offset=0.0, center_y_offset=0.0)
    mapper = CoordinateMapper(zone_config=cfg, screen_width=1920, screen_height=1080)

    # Center is inside
    assert mapper.is_inside_zone(0.5, 0.5) is True

    # Extreme corners outside
    assert mapper.is_inside_zone(0.05, 0.05) is False
    assert mapper.is_inside_zone(0.95, 0.95) is False
    assert mapper.is_inside_zone(0.10, 0.50) is False


def test_map_to_screen_proportionality():
    cfg = ActiveZoneConfig(width_ratio=0.5, height_ratio=0.5, center_x_offset=0.0, center_y_offset=0.0)
    # x_min = 0.25, x_max = 0.75, y_min = 0.25, y_max = 0.75
    mapper = CoordinateMapper(zone_config=cfg, screen_width=1920, screen_height=1080, sensitivity=1.0)

    # Center of active rectangle maps to exact center of screen
    center_coords = mapper.map_to_screen(0.5, 0.5)
    assert center_coords is not None
    cx, cy = center_coords
    assert abs(cx - (1919 / 2)) < 1.0
    assert abs(cy - (1079 / 2)) < 1.0

    # Top-left of active rectangle maps to (0, 0)
    tl_coords = mapper.map_to_screen(0.25, 0.25)
    assert tl_coords is not None
    assert abs(tl_coords[0] - 0.0) < 1e-4
    assert abs(tl_coords[1] - 0.0) < 1e-4

    # Bottom-right of active rectangle maps to (screen_w-1, screen_h-1)
    br_coords = mapper.map_to_screen(0.75, 0.75)
    assert br_coords is not None
    assert abs(br_coords[0] - 1919.0) < 1e-4
    assert abs(br_coords[1] - 1079.0) < 1e-4


def test_freeze_outside_zone():
    cfg = ActiveZoneConfig(width_ratio=0.5, height_ratio=0.5)
    mapper = CoordinateMapper(zone_config=cfg, screen_width=1920, screen_height=1080)

    # Coordinates outside active rectangle return None (cursor freezes)
    assert mapper.map_to_screen(0.1, 0.1) is None
    assert mapper.map_to_screen(0.9, 0.5) is None


def test_directional_mapping():
    cfg = ActiveZoneConfig(width_ratio=0.6, height_ratio=0.6)
    mapper = CoordinateMapper(zone_config=cfg, screen_width=1920, screen_height=1080)

    p1 = mapper.map_to_screen(0.4, 0.5)
    p2 = mapper.map_to_screen(0.6, 0.5)
    # Moving finger right increases screen X
    assert p2[0] > p1[0]

    p3 = mapper.map_to_screen(0.5, 0.4)
    p4 = mapper.map_to_screen(0.5, 0.6)
    # Moving finger down increases screen Y
    assert p4[1] > p3[1]


def test_pixel_rect_calculation():
    cfg = ActiveZoneConfig(width_ratio=0.5, height_ratio=0.5, center_x_offset=0.0, center_y_offset=0.0)
    mapper = CoordinateMapper(zone_config=cfg, screen_width=1920, screen_height=1080)

    x, y, w, h = mapper.get_pixel_rect(640, 480)
    assert x == 160
    assert y == 120
    assert w == 320
    assert h == 240


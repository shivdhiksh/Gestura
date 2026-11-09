"""
GESTURA - Cursor and Action Controller
Extends ActionController to provide backward compatibility with existing CursorController imports.
"""

from core.action_controller import ActionController


class CursorController(ActionController):
    """
    Unified controller managing hardware cursor positioning, clicks, drags,
    scrolling, volume, brightness, media, and window shortcuts.
    """
    pass

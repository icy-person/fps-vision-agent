"""Default first-person-shooter control scheme.

Swap this out (or add a new games/*.py alongside it) for a different genre —
nothing in src/ hardcodes FPS controls, they all import whichever profile
you pass them. See games/platformer_default.py for a concrete example of how
different a non-FPS profile looks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from action_schema import ActionProfile  # noqa: E402

PROFILE = ActionProfile(
    name="fps_default",
    movement_combos=(
        (), ("w",), ("s",), ("a",), ("d",),
        ("w", "a"), ("w", "d"), ("s", "a"), ("s", "d"),
    ),
    buttons=("jump", "crouch", "fire"),
    has_mouse_look=True,
)

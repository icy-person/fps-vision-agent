"""Example 2D-platformer control scheme — no mouse look, different movement
set entirely. Demonstrates that games/ profiles are genuinely swappable: a
platformer or top-down game reuses the exact same src/ pipeline
(record_demo.py, policy_model.py, dataset.py, train_bc.py, play.py), it just
gets a smaller/different action space and no mouse-related model outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from action_schema import ActionProfile  # noqa: E402

PROFILE = ActionProfile(
    name="platformer_default",
    movement_combos=((), ("a",), ("d",)),  # left, right, or standing still
    buttons=("jump", "attack"),
    has_mouse_look=False,
)

"""The shared action contract between recording, training, and live play.

An `ActionProfile` is what makes the small model modular across games: it's
the one thing that changes per game (or per game genre — an FPS and a
twin-stick shooter can share a profile, a platformer needs a different one),
while `record_demo.py`, `policy_model.py`, `dataset.py`, and `play.py` are
all written against this shared interface and never hardcode any single
game's controls.

A trained checkpoint is tied to the ActionProfile it was trained with (head
sizes depend on it) — checkpoints are not interchangeable across profiles.
Two games that use the *same* profile CAN share a checkpoint, and training
on demonstrations from several such games at once is how the small model
generalizes across them — see the "how multi-game reuse actually works"
section in README.md before expecting zero-shot generalization to a genre
it has no data for.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionProfile:
    name: str
    # Each entry is the set of movement keys held together for that combo,
    # e.g. ("w", "a") = moving forward-left. Index 0 should always be the
    # empty tuple (no movement keys held).
    movement_combos: tuple[tuple[str, ...], ...]
    # Extra binary buttons beyond movement, e.g. "jump", "crouch", "fire",
    # "interact". Each becomes one binary output head.
    buttons: tuple[str, ...]
    # Whether this game uses continuous mouse-look (first-person camera).
    # Platformers/top-down games typically set this False.
    has_mouse_look: bool
    mouse_bins: int = 11  # odd, center bin = no movement
    max_mouse_delta_px: int = 25


# Concrete profiles live under games/ (e.g. games/fps_default.py), not here —
# this module only defines the shared shape. All keys any profile references
# must be present in input_sim._KEY_MAP.


def button_index(profile: ActionProfile, name: str) -> int:
    return profile.buttons.index(name)


# Shared between input_sim.py (injecting a button press) and record_demo.py
# (reading a button's current held state from human input) so both sides of
# the recording/playing round-trip agree on what each button name means.
# Values are either a key name (see input_sim._KEY_MAP) or the literal
# string "click" for the mouse's left button.
BUTTON_TO_KEY = {
    "jump": "space",
    "crouch": "ctrl",
    "attack": "click",
    "fire": "click",
    "interact": "e",
    "reload": "r",
}


def movement_combo_index(held_movement_keys: set[str], profile: ActionProfile) -> int:
    """Matches the currently-held wasd-style keys to one of the profile's
    predefined combos. Falls back to index 0 (no movement keys) if the held
    set doesn't exactly match any combo — e.g. holding 3+ movement keys at
    once, which none of the default profiles define a combo for."""
    for index, combo in enumerate(profile.movement_combos):
        if set(combo) == held_movement_keys:
            return index
    return 0


def quantize_mouse_delta(delta_px: int, profile: ActionProfile) -> int:
    """Maps a raw pixel delta to a bin index in [0, mouse_bins). Bin
    `mouse_bins // 2` is the center (no movement); bins are evenly spaced
    across [-max_mouse_delta_px, +max_mouse_delta_px], with values beyond
    that range clamped to the outermost bin rather than wrapping."""
    center = profile.mouse_bins // 2
    clamped = max(-profile.max_mouse_delta_px, min(profile.max_mouse_delta_px, delta_px))
    step = max(1, profile.max_mouse_delta_px // center)
    return center + round(clamped / step)


def dequantize_mouse_bin(bin_index: int, profile: ActionProfile) -> int:
    """Inverse of `quantize_mouse_delta`, used at play time to turn a
    predicted bin back into a pixel delta to inject."""
    center = profile.mouse_bins // 2
    step = max(1, profile.max_mouse_delta_px // center)
    return (bin_index - center) * step

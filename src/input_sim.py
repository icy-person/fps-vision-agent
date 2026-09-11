"""Keyboard/mouse simulation for live play.

Uses `python-uinput`, which creates a *virtual input device* at the kernel
level (via /dev/uinput). This is the important choice for FPS-style games
specifically: most first-person games read raw relative mouse motion for
camera look, not the X11 cursor position, so X11-level tools (xdotool,
PyAutoGUI) that warp the cursor often do nothing in-game. A uinput device
emits real `REL_X`/`REL_Y` relative-motion events indistinguishable from a
physical mouse, so games that read raw input see it correctly. It's also
the reason `record_demo.py` reads raw evdev events rather than X11 mouse
position — recorded labels and injected actions use the same units.

Setup required before this will work (see README.md for the full version):
  1. `sudo modprobe uinput` (usually already loaded on modern kernels)
  2. Your user needs write access to /dev/uinput — either run as root (fine
     for local experimentation) or add a udev rule + join the `input` group.
  3. `pip install python-uinput`

This talks directly to the kernel input subsystem: there is no confirmation
step, no dry-run — whatever action the model picks gets sent immediately.
Keep the game in a safe, non-destructive state (offline/practice mode) while
testing a freshly trained model.
"""
from __future__ import annotations

from typing import Iterable

try:
    import uinput
except ImportError as exc:  # pragma: no cover - import-time guidance only
    raise ImportError(
        "python-uinput is required for input simulation. Install it with: "
        "pip install python-uinput (and see README.md for the /dev/uinput "
        "permission setup — it will not work out of the box)."
    ) from exc

from action_schema import BUTTON_TO_KEY

# Extend this map (and a game's ActionProfile) together if you add new keys.
_KEY_MAP = {
    "w": uinput.KEY_W,
    "a": uinput.KEY_A,
    "s": uinput.KEY_S,
    "d": uinput.KEY_D,
    "space": uinput.KEY_SPACE,
    "ctrl": uinput.KEY_LEFTCTRL,
    "shift": uinput.KEY_LEFTSHIFT,
    "r": uinput.KEY_R,
    "e": uinput.KEY_E,
}


class InputController:
    """Owns one virtual keyboard+mouse device for the process's lifetime.

    Not meant to be shared across threads — create one instance in the play
    loop and use it from there only.
    """

    def __init__(self, key_names: Iterable[str]):
        key_events = [_KEY_MAP[name] for name in key_names]
        self._device = uinput.Device(
            [*key_events, uinput.BTN_LEFT, uinput.BTN_RIGHT, uinput.REL_X, uinput.REL_Y]
        )
        self._held: set[str] = set()

    def set_key_state(self, key_name: str, held: bool) -> None:
        """Idempotent: calling with the same state twice in a row is a
        no-op, matching how a real held key behaves."""
        is_held = key_name in self._held
        if held == is_held:
            return
        self._device.emit(_KEY_MAP[key_name], 1 if held else 0)
        if held:
            self._held.add(key_name)
        else:
            self._held.discard(key_name)

    def set_button(self, button_name: str, held: bool) -> None:
        """Applies an ActionProfile button by name, e.g. "jump" -> space key,
        "fire"/"attack" -> left click (held, not a tap — for guns that need
        held-fire; call click() instead for a single tap)."""
        target = BUTTON_TO_KEY.get(button_name)
        if target is None:
            raise KeyError(f"No key/click mapping for button {button_name!r}; add one to action_schema.BUTTON_TO_KEY")
        if target == "click":
            if held:
                self._device.emit(uinput.BTN_LEFT, 1)
            else:
                self._device.emit(uinput.BTN_LEFT, 0)
        else:
            self.set_key_state(target, held)

    def release_all(self) -> None:
        """Releases every currently-held key. Call this on shutdown/error so
        a crash mid-run doesn't leave a key stuck down in the game."""
        for key_name in list(self._held):
            self.set_key_state(key_name, False)
        self._device.emit(uinput.BTN_LEFT, 0)

    def move_mouse_relative(self, dx: int, dy: int) -> None:
        if dx:
            self._device.emit(uinput.REL_X, dx, syn=False)
        if dy:
            self._device.emit(uinput.REL_Y, dy)
        elif dx:
            self._device.syn()

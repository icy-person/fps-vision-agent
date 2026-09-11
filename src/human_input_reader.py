"""Reads real human input during demonstration recording.

Uses `python-evdev` to read raw kernel input events directly from the
keyboard and mouse device nodes, rather than any X11-level API. This
matters for the same reason `input_sim.py` uses uinput instead of X11 cursor
warping: an FPS-style game reading raw relative mouse motion sees exactly
the REL_X/REL_Y delta evdev reports, so recorded labels are in the same
units `input_sim.py` later injects — there's no X11-coordinate-vs-raw-input
mismatch to worry about.

Requires read access to the device nodes (same `input` group setup as
uinput — see README.md) and the exact device paths, which you get from
scripts/list_input_devices.py.
"""
from __future__ import annotations

import threading

try:
    from evdev import InputDevice, ecodes
except ImportError as exc:  # pragma: no cover - import-time guidance only
    raise ImportError(
        "python-evdev is required for recording. Install it with: pip install evdev"
    ) from exc

# Maps evdev key codes to the same short names used throughout this project
# (input_sim._KEY_MAP, ActionProfile movement_combos). Extend both together.
_CODE_TO_KEY = {
    ecodes.KEY_W: "w",
    ecodes.KEY_A: "a",
    ecodes.KEY_S: "s",
    ecodes.KEY_D: "d",
    ecodes.KEY_SPACE: "space",
    ecodes.KEY_LEFTCTRL: "ctrl",
    ecodes.KEY_LEFTSHIFT: "shift",
    ecodes.KEY_R: "r",
    ecodes.KEY_E: "e",
}


class HumanInputReader:
    """Background-thread reader exposing thread-safe, poll-based access to
    current held keys/buttons and accumulated mouse motion.

    Call `consume_mouse_delta()` once per recorded frame — it returns the
    motion accumulated since the previous call and resets to zero, so each
    recorded frame's label reflects exactly the motion that happened during
    that frame's time slice, matching how it will be replayed frame-by-frame
    at play time.
    """

    def __init__(self, keyboard_device_path: str, mouse_device_path: str):
        if not keyboard_device_path or not mouse_device_path:
            raise ValueError(
                "keyboard_device_path and mouse_device_path must be set "
                "(see scripts/list_input_devices.py) before recording."
            )
        self._keyboard = InputDevice(keyboard_device_path)
        self._mouse = InputDevice(mouse_device_path)
        self._lock = threading.Lock()
        self._held_keys: set[str] = set()
        self._left_button_down = False
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def _read_keyboard(self) -> None:
        for event in self._keyboard.read_loop():
            if self._stop_event.is_set():
                return
            if event.type != ecodes.EV_KEY:
                continue
            key_name = _CODE_TO_KEY.get(event.code)
            if key_name is None:
                continue
            with self._lock:
                if event.value == 1:  # key down
                    self._held_keys.add(key_name)
                elif event.value == 0:  # key up
                    self._held_keys.discard(key_name)
                # value == 2 is "repeat", ignored — held state is unaffected

    def _read_mouse(self) -> None:
        for event in self._mouse.read_loop():
            if self._stop_event.is_set():
                return
            if event.type == ecodes.EV_REL:
                with self._lock:
                    if event.code == ecodes.REL_X:
                        self._mouse_dx += event.value
                    elif event.code == ecodes.REL_Y:
                        self._mouse_dy += event.value
            elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_LEFT:
                with self._lock:
                    self._left_button_down = event.value == 1

    def start(self) -> None:
        for target in (self._read_keyboard, self._read_mouse):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop_event.set()
        # read_loop() blocks on the device fd until the next event; there is
        # no clean way to interrupt it from outside without writing to the
        # device, so threads are daemonized and left to exit with the process
        # rather than joined here.

    def held_keys(self) -> set[str]:
        with self._lock:
            return set(self._held_keys)

    def fire_held(self) -> bool:
        with self._lock:
            return self._left_button_down

    def consume_mouse_delta(self) -> tuple[int, int]:
        with self._lock:
            dx, dy = self._mouse_dx, self._mouse_dy
            self._mouse_dx = 0
            self._mouse_dy = 0
        return dx, dy

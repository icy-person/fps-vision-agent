#!/usr/bin/env python3
"""Lists available input devices and their /dev/input/eventN paths, to fill
in RecordingConfig.keyboard_device_path / mouse_device_path.

Needs read access to /dev/input/event* — same `input` group setup as
uinput (see README.md).

Usage:
    python scripts/list_input_devices.py
"""
from __future__ import annotations

try:
    from evdev import InputDevice, list_devices
except ImportError:
    print("python-evdev is required. Install it with: pip install evdev")
    raise SystemExit(1)


def main() -> None:
    paths = list_devices()
    if not paths:
        print(
            "No devices found. If you're not in the 'input' group yet, "
            "either add yourself (see README.md) or run this with sudo "
            "just to check paths/names, then fix permissions before recording."
        )
        return

    print("Available input devices:\n")
    for path in paths:
        try:
            device = InputDevice(path)
            print(f"  {path}  {device.name!r}")
        except PermissionError:
            print(f"  {path}  <no permission to read — fix /dev/input group access>")

    print(
        "\nPick your keyboard and mouse from the list above and set "
        "RecordingConfig.keyboard_device_path / mouse_device_path in "
        "src/config.py accordingly."
    )


if __name__ == "__main__":
    main()

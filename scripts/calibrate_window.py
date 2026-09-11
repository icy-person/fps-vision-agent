#!/usr/bin/env python3
"""Prints the screen region of a window, to fill in src/config.py's
CaptureConfig. Only works under X11/XWayland - see capture.py's docstring
about pure-Wayland sessions.

Requires `wmctrl` (install with your distro's package manager, e.g.
`sudo apt install wmctrl` on Debian/Ubuntu).

Usage:
    1. Open your game in windowed or borderless-windowed mode (not
       exclusive fullscreen - exclusive fullscreen can prevent screen
       capture tools from reading the frame on some setups).
    2. python scripts/calibrate_window.py
    3. Pick your game's window from the printed list.
    4. Copy the printed left/top/width/height into CaptureConfig in
       src/config.py.
"""
from __future__ import annotations

import subprocess
import sys


def list_windows() -> list[tuple[str, int, int, int, int, str]]:
    try:
        output = subprocess.check_output(["wmctrl", "-lG"], text=True)
    except FileNotFoundError:
        print("wmctrl not found. Install it first, e.g.: sudo apt install wmctrl", file=sys.stderr)
        sys.exit(1)

    windows = []
    for line in output.splitlines():
        # Format: <id> <desktop> <x> <y> <width> <height> <host> <title...>
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        win_id, _desktop, x, y, width, height, _host, title = parts
        windows.append((win_id, int(x), int(y), int(width), int(height), title))
    return windows


def main() -> None:
    windows = list_windows()
    if not windows:
        print("No windows found (or wmctrl returned nothing).")
        return

    print("Windows currently open:\n")
    for i, (win_id, x, y, w, h, title) in enumerate(windows):
        print(f"  [{i}] {title!r}  region=({x}, {y}, {w}x{h})")

    choice = input("\nPick a window index: ").strip()
    try:
        index = int(choice)
        _win_id, x, y, w, h, title = windows[index]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    print(f"\nSelected: {title!r}")
    print("Paste this into CaptureConfig in src/config.py:\n")
    print(f"    left: int = {x}")
    print(f"    top: int = {y}")
    print(f"    width: int = {w}")
    print(f"    height: int = {h}")


if __name__ == "__main__":
    main()

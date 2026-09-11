#!/usr/bin/env python3
"""Measures the achievable capture -> input loop rate on this machine,
BEFORE you commit to a `target_fps` in config.py. 30-60fps is an upper
bound for a small CNN, not a guarantee - actual achievable rate depends on
your CPU, screen resolution, and capture backend, so measure it here first.

Usage:
    python scripts/benchmark_loop.py --seconds 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from capture import ScreenCapture  # noqa: E402
from config import AgentConfig  # noqa: E402
from input_sim import InputController  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--skip-input", action="store_true", help="benchmark capture only, skip uinput setup")
    args = parser.parse_args()

    cfg = AgentConfig()
    capture = ScreenCapture(cfg.capture)
    controller = InputController(set()) if not args.skip_input else None

    print(f"Running for {args.seconds}s...")
    frame_count = 0
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            _frame = capture.grab_for_policy()
            if controller is not None:
                # Harmless no-op movement so the benchmark includes real
                # input-emission cost, without pressing any game keys.
                controller.move_mouse_relative(0, 0)
            frame_count += 1
    finally:
        capture.close()
        if controller is not None:
            controller.release_all()

    elapsed = time.monotonic() - start
    fps = frame_count / elapsed
    print(f"\n{frame_count} frames in {elapsed:.2f}s = {fps:.1f} fps")
    print(
        f"Suggested target_fps for config.py: {int(fps * 0.7)} "
        "(70% of measured max, to leave headroom for the model's own "
        "inference time, which this benchmark does not include)"
    )


if __name__ == "__main__":
    main()

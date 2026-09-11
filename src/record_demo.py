#!/usr/bin/env python3
"""Records demonstration episodes while YOU play the game normally.

This is the data collection step for imitation learning (behavior cloning):
no reward function, no trial-and-error, no agent acting on its own — you
play correctly, and every frame is saved alongside the action you actually
took (and the planner's goal at that moment), for train_bc.py to later
learn to imitate.

Usage:
    python record_demo.py --game fps_default --episode-name run1
    python record_demo.py --game platformer_default --episode-name level1 --no-planner

Press Ctrl+C to stop and save the episode. `--no-planner` skips loading the
big model (faster startup) for quick tests of the capture/input-reading
pipeline; goal is recorded as a constant 0 in that case, which is fine for
testing but not for a training set you intend to actually use, since the
model won't learn any goal-conditioning from data where the goal never
varies.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

from action_schema import ActionProfile, movement_combo_index, quantize_mouse_delta
from capture import ScreenCapture
from config import AgentConfig
from human_input_reader import HumanInputReader
from planner import Planner


def load_profile(game_name: str) -> ActionProfile:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    module = importlib.import_module(f"games.{game_name}")
    return module.PROFILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, help="name of a module under games/, e.g. fps_default")
    parser.add_argument("--episode-name", required=True)
    parser.add_argument("--no-planner", action="store_true", help="skip the big model, record goal=0 always")
    args = parser.parse_args()

    profile = load_profile(args.game)
    cfg = AgentConfig()
    print(
        f"[record] capture region: {cfg.capture.left},{cfg.capture.top} "
        f"{cfg.capture.width}x{cfg.capture.height} -- verify this matches "
        f"your game window (scripts/calibrate_window.py) before continuing."
    )

    capture = ScreenCapture(cfg.capture)
    reader = HumanInputReader(cfg.recording.keyboard_device_path, cfg.recording.mouse_device_path)
    reader.start()

    current_goal = [0]
    planner = None
    if not args.no_planner:
        planner = Planner(cfg, on_goal=lambda index: current_goal.__setitem__(0, index))
        planner.start()
        print(f"[record] waiting {cfg.timing.planner_interval_seconds}s for the first planner goal...")
        time.sleep(cfg.timing.planner_interval_seconds)

    frames: list[np.ndarray] = []
    goals: list[int] = []
    movements: list[int] = []
    buttons: dict[str, list[int]] = {name: [] for name in profile.buttons}
    yaws: list[int] = []
    pitches: list[int] = []

    frame_period = 1.0 / cfg.timing.target_fps
    print(f"[record] recording at target {cfg.timing.target_fps} fps — play now. Ctrl+C to stop and save.")

    try:
        while True:
            loop_start = time.monotonic()

            frame = capture.grab_for_policy()
            held = reader.held_keys()
            movement_keys_held = held & {"w", "a", "s", "d"}

            frames.append(frame)
            goals.append(current_goal[0])
            movements.append(movement_combo_index(movement_keys_held, profile))
            for name in profile.buttons:
                if name in ("fire", "attack"):
                    buttons[name].append(int(reader.fire_held()))
                else:
                    key = {"jump": "space", "crouch": "ctrl"}.get(name)
                    buttons[name].append(int(key in held) if key else 0)
            if profile.has_mouse_look:
                dx, dy = reader.consume_mouse_delta()
                yaws.append(quantize_mouse_delta(dx, profile))
                pitches.append(quantize_mouse_delta(dy, profile))

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    except KeyboardInterrupt:
        print(f"\n[record] stopped after {len(frames)} frames")
    finally:
        capture.close()
        reader.stop()
        if planner is not None:
            planner.stop()

    if not frames:
        print("[record] no frames recorded, nothing saved")
        return

    out_dir = Path(cfg.recording.data_dir) / args.game
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.episode_name}.npz"

    save_kwargs = {
        "frames": np.stack(frames),
        "goals": np.array(goals, dtype=np.int64),
        "movements": np.array(movements, dtype=np.int64),
        **{f"button_{name}": np.array(values, dtype=np.int64) for name, values in buttons.items()},
    }
    if profile.has_mouse_look:
        save_kwargs["yaws"] = np.array(yaws, dtype=np.int64)
        save_kwargs["pitches"] = np.array(pitches, dtype=np.int64)

    np.savez_compressed(out_path, **save_kwargs)
    print(f"[record] saved {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()

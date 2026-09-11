#!/usr/bin/env python3
"""Runs the full two-layer agent live: trained small model + LM Studio planner.

    python play.py --game fps_default --checkpoint checkpoints/fps_default.pt

The planner thread starts first and picks an initial goal before the fast
loop begins, so the small model is never conditioned on an undefined goal.
After that, the planner keeps updating the active goal in the background on
its own schedule while this loop calls the small model every frame with
whatever goal is currently active.

This does not train or record anything - it only runs a checkpoint produced
by train_bc.py. Sampling (not greedy argmax) is used for movement/buttons by
default so play doesn't look robotically deterministic; pass
--deterministic for reproducible/debuggable runs.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from action_schema import BUTTON_TO_KEY, dequantize_mouse_bin
from capture import ScreenCapture
from config import AgentConfig
from input_sim import InputController
from planner import Planner
from policy_model import PolicyModel


def load_profile(game_name: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    module = importlib.import_module(f"games.{game_name}")
    return module.PROFILE


def sample_or_argmax(logits: torch.Tensor, deterministic: bool) -> int:
    if deterministic:
        return int(logits.argmax(dim=-1).item())
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-planner", action="store_true", help="run the small model alone with a fixed goal=0, skip LM Studio")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many frames (default: run until Ctrl+C)")
    args = parser.parse_args()

    profile = load_profile(args.game)
    cfg = AgentConfig()
    num_goals = len(cfg.planner.goals)

    model = PolicyModel(profile, cfg.capture.policy_frame_size, num_goals)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[play] loaded checkpoint {args.checkpoint} (trained for game={checkpoint.get('game')})")

    capture = ScreenCapture(cfg.capture)
    # InputController needs the flat set of every key any movement combo or
    # button in this profile might press.
    all_keys = {key for combo in profile.movement_combos for key in combo}
    for button in profile.buttons:
        mapped = BUTTON_TO_KEY.get(button)
        if mapped and mapped != "click":
            all_keys.add(mapped)
    controller = InputController(all_keys)

    current_goal = [0]
    planner = None
    if not args.no_planner:
        planner = Planner(cfg, on_goal=lambda index: current_goal.__setitem__(0, index))
        planner.start()
        print(f"[play] waiting {cfg.timing.planner_interval_seconds}s for the first planner goal...")
        time.sleep(cfg.timing.planner_interval_seconds)

    frame_period = 1.0 / cfg.timing.target_fps
    steps = 0
    print("[play] running - Ctrl+C to stop")
    try:
        while args.max_steps is None or steps < args.max_steps:
            loop_start = time.monotonic()

            frame = capture.grab_for_policy()
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            goal_t = torch.zeros(1, num_goals)
            goal_t[0, current_goal[0]] = 1.0

            with torch.no_grad():
                outputs = model(frame_t, goal_t)

            movement_idx = sample_or_argmax(outputs["movement"][0], args.deterministic)
            pressed = set(profile.movement_combos[movement_idx])
            for key in ("w", "a", "s", "d"):
                if key in all_keys:
                    controller.set_key_state(key, key in pressed)

            for button in profile.buttons:
                held = bool(sample_or_argmax(outputs[button][0], args.deterministic))
                controller.set_button(button, held)

            if profile.has_mouse_look:
                yaw_bin = sample_or_argmax(outputs["yaw"][0], args.deterministic)
                pitch_bin = sample_or_argmax(outputs["pitch"][0], args.deterministic)
                dx = dequantize_mouse_bin(yaw_bin, profile)
                dy = dequantize_mouse_bin(pitch_bin, profile)
                controller.move_mouse_relative(dx, dy)

            steps += 1
            elapsed = time.monotonic() - loop_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    except KeyboardInterrupt:
        print("\n[play] stopped")
    finally:
        controller.release_all()
        capture.close()
        if planner is not None:
            planner.stop()


if __name__ == "__main__":
    main()

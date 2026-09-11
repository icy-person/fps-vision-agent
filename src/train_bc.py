#!/usr/bin/env python3
"""Trains the small model by imitation (behavior cloning) on recorded
demonstrations — plain supervised classification, no reward function, no
environment, no trial-and-error.

Usage:
    python train_bc.py --game fps_default --epochs 20

Loads every episode under data/<game>/ (recorded with matching --game via
record_demo.py) and trains one checkpoint against that game's ActionProfile.
Multiple episodes from different sessions (or, if they share the same
ActionProfile, from different games of the same genre) are combined into one
training set automatically — that's how the model generalizes across
sessions/games, by being trained on demonstrations from all of them, not by
zero-shot transfer to genres it has never seen.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import AgentConfig
from dataset import load_dataset
from policy_model import PolicyModel


def load_profile(game_name: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    module = importlib.import_module(f"games.{game_name}")
    return module.PROFILE


def compute_loss(outputs: dict, labels: dict, profile) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs["movement"], labels["movement"])
    for name in profile.buttons:
        loss = loss + loss_fn(outputs[name], labels[name])
    if profile.has_mouse_look:
        loss = loss + loss_fn(outputs["yaw"], labels["yaw"])
        loss = loss + loss_fn(outputs["pitch"], labels["pitch"])
    return loss


def accuracy(outputs: dict, labels: dict, profile) -> dict[str, float]:
    accs = {}
    for key in outputs:
        preds = outputs[key].argmax(dim=1)
        accs[key] = (preds == labels[key]).float().mean().item()
    return accs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--save-path", type=str, default=None, help="default: checkpoints/<game>.pt")
    args = parser.parse_args()

    profile = load_profile(args.game)
    cfg = AgentConfig()
    num_goals = len(cfg.planner.goals)

    data_dir = Path(cfg.recording.data_dir) / args.game
    full_dataset = load_dataset(data_dir, profile, num_goals)
    val_size = max(1, int(len(full_dataset) * args.val_fraction))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    print(f"[train_bc] {len(full_dataset)} frames total ({train_size} train / {val_size} val)")

    # num_workers=0: dataset workers forking after mmap'd numpy arrays are
    # open can be flaky on some setups; single-process loading is slower but
    # dependable. Raise this once you've confirmed multi-worker loading
    # works reliably in your environment.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PolicyModel(profile, cfg.capture.policy_frame_size, num_goals)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_path = Path(args.save_path) if args.save_path else Path("checkpoints") / f"{args.game}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for frames, goals, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(frames, goals)
            loss = compute_loss(outputs, labels, profile)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * frames.size(0)
        train_loss = total_loss / train_size

        model.eval()
        val_accs: dict[str, list[float]] = {}
        with torch.no_grad():
            for frames, goals, labels in val_loader:
                outputs = model(frames, goals)
                for key, value in accuracy(outputs, labels, profile).items():
                    val_accs.setdefault(key, []).append(value)
        val_acc_str = ", ".join(f"{k}={sum(v) / len(v):.2f}" for k, v in val_accs.items())
        print(f"[train_bc] epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc: {val_acc_str}")

        torch.save({"model_state": model.state_dict(), "game": args.game}, save_path)

    print(f"[train_bc] saved to {save_path}")


if __name__ == "__main__":
    main()

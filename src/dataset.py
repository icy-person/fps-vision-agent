"""Loads recorded demonstration episodes (from record_demo.py) for
supervised training in train_bc.py.

Episodes are memory-mapped (`np.load(..., mmap_mode="r")`) rather than
loaded fully into RAM — with an 8GB machine, a few dozen episodes of frames
would otherwise be the single biggest memory consumer in this whole
project. Mapped arrays are read lazily from disk per-item, which is slower
per-sample than fully in-RAM data but keeps total memory bounded regardless
of how much demonstration data you record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from action_schema import ActionProfile


class _EpisodeDataset(Dataset):
    def __init__(self, path: Path, profile: ActionProfile, num_goals: int):
        self.profile = profile
        self.num_goals = num_goals
        self._data = np.load(path, mmap_mode="r")
        self._length = self._data["frames"].shape[0]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        frame = np.array(self._data["frames"][index])  # copy out of the mmap
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        goal_index = int(self._data["goals"][index])
        goal_t = torch.zeros(self.num_goals)
        goal_t[goal_index] = 1.0

        labels = {"movement": torch.tensor(int(self._data["movements"][index]), dtype=torch.long)}
        for name in self.profile.buttons:
            labels[name] = torch.tensor(int(self._data[f"button_{name}"][index]), dtype=torch.long)
        if self.profile.has_mouse_look:
            labels["yaw"] = torch.tensor(int(self._data["yaws"][index]), dtype=torch.long)
            labels["pitch"] = torch.tensor(int(self._data["pitches"][index]), dtype=torch.long)

        return frame_t, goal_t, labels


def load_dataset(episode_dir: Path, profile: ActionProfile, num_goals: int) -> Dataset:
    """`episode_dir` is typically `data/<game_name>/` — every `.npz` file in
    it (as written by record_demo.py) is treated as one episode and
    concatenated into a single dataset."""
    paths = sorted(episode_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz episodes found in {episode_dir} — run record_demo.py first.")
    return ConcatDataset([_EpisodeDataset(path, profile, num_goals) for path in paths])

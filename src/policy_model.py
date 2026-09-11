"""The small model: a CNN over the frame, conditioned on the planner's goal,
with one output head per part of the action (movement combo, each button,
and — if the profile uses mouse look — yaw/pitch bins).

Deliberately small (a handful of conv layers, ~1-3M parameters depending on
frame size) so it can run in well under a frame period on CPU, unlike the
planner. Trained with plain supervised learning (behavior cloning) in
train_bc.py — no reward function, no simulation, no trial-and-error: it
learns to imitate the (frame, goal) -> action pairs recorded in
record_demo.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from action_schema import ActionProfile


class PolicyModel(nn.Module):
    def __init__(self, profile: ActionProfile, frame_size: tuple[int, int], num_goals: int):
        super().__init__()
        self.profile = profile
        width, height = frame_size

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width)
            conv_out_size = self.conv(dummy).flatten(1).shape[1]

        goal_embed_size = 32
        self.goal_embed = nn.Sequential(nn.Linear(num_goals, goal_embed_size), nn.ReLU())

        trunk_in = conv_out_size + goal_embed_size
        self.trunk = nn.Sequential(nn.Linear(trunk_in, 256), nn.ReLU())

        self.movement_head = nn.Linear(256, len(profile.movement_combos))
        self.button_heads = nn.ModuleDict({name: nn.Linear(256, 2) for name in profile.buttons})
        if profile.has_mouse_look:
            self.yaw_head = nn.Linear(256, profile.mouse_bins)
            self.pitch_head = nn.Linear(256, profile.mouse_bins)

    def forward(self, frame: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        """`frame`: (B, 3, H, W) float in [0, 1]. `goal`: (B, num_goals) one-hot.
        Returns a dict of logits, one entry per action-profile output head —
        same keys every call, only the tensor shapes reflect the profile."""
        features = self.conv(frame).flatten(1)
        goal_features = self.goal_embed(goal)
        hidden = self.trunk(torch.cat([features, goal_features], dim=1))

        outputs = {"movement": self.movement_head(hidden)}
        for name, head in self.button_heads.items():
            outputs[name] = head(hidden)
        if self.profile.has_mouse_look:
            outputs["yaw"] = self.yaw_head(hidden)
            outputs["pitch"] = self.pitch_head(hidden)
        return outputs

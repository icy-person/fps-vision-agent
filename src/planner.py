"""Slow-path strategic planner (the "big model" in the two-layer design).

Talks to LM Studio's local server (OpenAI-compatible chat completions API
with image_url content, same shape as OpenAI's vision API) rather than
loading the model directly - this matches actually running the
Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled GGUF (Q8_0) through LM
Studio's llama.cpp backend, which is both faster and lighter than loading
raw weights through `transformers`/PyTorch.

Before using this: open LM Studio, load the model, go to the "Developer"
tab and start the local server (default http://localhost:1234). GET
{lm_studio_base_url}/models with the server running to confirm the exact
model identifier string for `PlannerConfig.lm_studio_model`.

Runs on its own background thread, on its own schedule
(`planner_interval_seconds`), completely decoupled from the fast per-frame
loop - during both recording and live play.

Important: this specific checkpoint is a reasoning-distilled model trained
to answer inside a `<think>...</think>` block before its final response
(see the model's card). That reasoning costs real generation time even for
a one-word goal choice - `_choose_goal` strips the `<think>` block before
matching a goal keyword, and logs how long each decision actually took so
you can tune `planner_interval_seconds` to what your hardware really
achieves rather than guessing.
"""
from __future__ import annotations

import base64
import io
import re
import threading
import time
from typing import Callable

import numpy as np
import requests
from PIL import Image

from capture import ScreenCapture
from config import AgentConfig

_PROMPT_TEMPLATE = (
    "You are the high-level strategist for a game-playing agent. Look at "
    "this game frame and choose exactly one word from this list that best "
    "describes what the agent should do next: {goals}. After your "
    "reasoning, give the final answer as just that one word on its own "
    "line, nothing else."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _frame_to_data_url(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class Planner:
    """Owns its own screen-capture instance (capture is explicitly not
    thread-safe - see capture.py - so this must not share a ScreenCapture
    with the fast loop). The model itself lives in LM Studio's process, not
    in this one - there's nothing to load here."""

    def __init__(self, cfg: AgentConfig, on_goal: Callable[[int], None]):
        self.cfg = cfg
        self.on_goal = on_goal
        self._capture = ScreenCapture(cfg.capture)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_index = 0
        self._base_url = cfg.planner.lm_studio_base_url.rstrip("/")

        try:
            response = requests.get(f"{self._base_url}/models", timeout=5)
            response.raise_for_status()
            model_ids = [m["id"] for m in response.json().get("data", [])]
            print(f"[planner] connected to LM Studio, loaded model(s): {model_ids}")
            if cfg.planner.lm_studio_model not in model_ids:
                print(
                    f"[planner] WARNING: configured model "
                    f"{cfg.planner.lm_studio_model!r} not in the list above - "
                    f"copy the exact id from there into PlannerConfig.lm_studio_model"
                )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Could not reach LM Studio server at {self._base_url}. "
                "Open LM Studio, load the model, and start the local server "
                "from the Developer tab before running this."
            ) from exc

    def _choose_goal(self, frame: np.ndarray) -> int:
        goals = self.cfg.planner.goals
        prompt = _PROMPT_TEMPLATE.format(goals=", ".join(goals))
        payload = {
            "model": self.cfg.planner.lm_studio_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _frame_to_data_url(frame)}},
                ],
            }],
            "max_tokens": self.cfg.planner.max_response_tokens,
            "temperature": 0.0,
        }
        response = requests.post(f"{self._base_url}/chat/completions", json=payload, timeout=60)
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]

        # Strip the reasoning block (if present - some server/template combos
        # already split it into a separate field, but handle it inline too)
        # before matching, so a goal word mentioned mid-reasoning doesn't get
        # picked up instead of the actual final answer.
        final_text = _THINK_BLOCK.sub("", raw_text).strip().lower()

        for index, goal in enumerate(goals):
            if goal in final_text:
                return index
        print(f"[planner] could not parse goal from output {final_text!r}, keeping previous goal")
        return self._current_index

    def current_goal_index(self) -> int:
        return self._current_index

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            frame = self._capture.grab_for_policy()
            elapsed = 0.0
            try:
                self._current_index = self._choose_goal(frame)
                elapsed = time.monotonic() - loop_start
                print(f"[planner] goal -> {self.cfg.planner.goals[self._current_index]}  ({elapsed:.1f}s)")
                self.on_goal(self._current_index)
            except Exception as exc:  # noqa: BLE001 - keep the thread alive on any single failure
                elapsed = time.monotonic() - loop_start
                print(f"[planner] error during goal selection ({exc}), keeping previous goal")
            remaining = self.cfg.planner.planner_interval_seconds - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)
            elif elapsed > self.cfg.planner.planner_interval_seconds * 1.5:
                print(
                    f"[planner] decision took {elapsed:.1f}s, well over your "
                    f"{self.cfg.planner.planner_interval_seconds}s interval - "
                    "consider raising planner_interval_seconds in config.py"
                )

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="planner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._capture.close()

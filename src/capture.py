"""Screen capture for the recording and play loops.

Uses `mss`, which on Linux talks to X11 (via XShm when available) and is one
of the faster pure-Python capture options. Two things to know before relying
on this in production:

- Under a pure Wayland session (no XWayland), `mss` generally cannot see
  other applications' windows at all — this is a Wayland security
  restriction, not a bug in this code. If `grab()` returns a black/empty
  frame, check `echo $XDG_SESSION_TYPE`; if it says "wayland", you likely
  need to run the game (or your whole session) under XWayland/X11 instead.
- The capture region is in physical pixels of the X11 screen, not logical/
  scaled pixels; on HiDPI setups these can differ.
"""
from __future__ import annotations

import numpy as np

from config import CaptureConfig

try:
    import mss
except ImportError as exc:  # pragma: no cover - import-time guidance only
    raise ImportError(
        "mss is required for screen capture. Install it with: pip install mss"
    ) from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python is required for frame resizing. Install it with: "
        "pip install opencv-python"
    ) from exc


class ScreenCapture:
    """Grabs the configured screen region and returns it as an RGB array.

    One `mss` instance must be created and used from a single thread — it is
    not thread-safe. The planner (which also wants frames, but only every
    few seconds) creates its own `ScreenCapture` instance rather than
    sharing this one across threads.
    """

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._sct = mss.mss()
        self._region = {
            "left": cfg.left,
            "top": cfg.top,
            "width": cfg.width,
            "height": cfg.height,
        }

    def grab_raw(self) -> np.ndarray:
        """Full-resolution capture as an HxWx3 uint8 RGB array."""
        shot = self._sct.grab(self._region)
        frame = np.array(shot)[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        return np.ascontiguousarray(frame)

    def grab_for_policy(self) -> np.ndarray:
        """Downscaled, model-ready frame: HxWx3 uint8 RGB at policy_frame_size."""
        frame = self.grab_raw()
        resized = cv2.resize(frame, self.cfg.policy_frame_size, interpolation=cv2.INTER_AREA)
        return resized

    def close(self) -> None:
        self._sct.close()

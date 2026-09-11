"""Central configuration for the agent.

Nothing here is auto-detected — capture region, and the recording device
paths, MUST be set for your setup via scripts/calibrate_window.py and
scripts/list_input_devices.py before recording or playing. The game's
*control scheme* (which keys exist, mouse or no mouse) is NOT here — that
lives in games/*.py as a swappable ActionProfile, imported by whichever
script needs it via a --game argument.
"""
from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    # Top-left corner + size of the screen region to capture, in pixels.
    # Fill these in with scripts/calibrate_window.py — there is no safe default.
    left: int = 0
    top: int = 0
    width: int = 1280
    height: int = 720
    # Frame fed to the model is downscaled to this size (keeps the CNN small
    # and fast). 84x84 is the classic Atari-DQN size and a reasonable
    # starting point for a first-person view; platformers with more
    # horizontal detail may want something wider, e.g. (128, 72).
    policy_frame_size: tuple[int, int] = (84, 84)


@dataclass
class TimingConfig:
    # Target loop rate for recording AND for live play — these should
    # match, since the model is trained on frames spaced at this rate.
    # Start lower (e.g. 20-30) and measure your actual achievable rate with
    # scripts/benchmark_loop.py before assuming 60 is reachable.
    target_fps: int = 30
    # How often the slow planner (the big model) re-evaluates and issues a
    # new high-level goal. Runs in its own background thread; the fast loop
    # just keeps using the last goal it has, whether recording or playing.
    # The reasoning-distilled planner model spends real tokens on a <think>
    # block before answering (see planner.py) — at ~12 tok/s this can take
    # several seconds on its own, so this default is set well above the
    # bare minimum; measure your actual goal-decision latency (planner.py
    # logs it) and tune from there rather than assuming faster.
    planner_interval_seconds: float = 8.0


@dataclass
class RecordingConfig:
    # Linux evdev device paths for the keyboard and mouse you play with,
    # e.g. "/dev/input/event5". Find yours with scripts/list_input_devices.py.
    # Reading raw evdev events (rather than X11-level input) is what lets
    # recorded demonstrations use the exact same relative mouse-motion units
    # that input_sim.py later injects via uinput — the model trains on and
    # produces the same units throughout.
    keyboard_device_path: str = ""
    mouse_device_path: str = ""
    # Where recorded episodes are written, one .npz file per episode, under
    # data/<game_name>/.
    data_dir: str = "data"


@dataclass
class PlannerConfig:
    # LM Studio's local server (OpenAI-compatible). Start it from LM
    # Studio's "Developer" tab -> Local Server -> Start Server, with the
    # Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled GGUF (Q8_0) loaded.
    lm_studio_base_url: str = "http://localhost:1234/v1"
    # The model identifier LM Studio shows for the loaded model — check the
    # server logs or GET {lm_studio_base_url}/models once the server is
    # running to get the exact string.
    lm_studio_model: str = "qwen3.5-2b-claude-4.6-opus-reasoning-distilled"
    # This checkpoint is a reasoning model — it answers inside a <think>...
    # </think> block before the final short answer (see planner.py), which
    # costs real tokens/time even for a one-word goal choice. Budget enough
    # planner_interval_seconds for a full reasoning pass at your measured
    # tokens/sec, not just for a one-word answer.
    max_response_tokens: int = 300
    # Coarse, fixed goal vocabulary. A classification head over a small fixed
    # set is far more robust here than free-text parsing or raw embeddings:
    # the small model only needs to condition on *which* of a few behaviors
    # to bias towards, not on open-ended language.
    goals: tuple[str, ...] = ("advance", "retreat", "search", "engage", "hold_position")


@dataclass
class AgentConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

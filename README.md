# fps-vision-agent

A modular, CPU-only agent that learns to play games from raw screen pixels
by **imitation** (behavior cloning on recorded human demonstrations) - not
reinforcement learning. Two models at two different speeds, connected by a
narrow, well-defined interface:

| | Small model | Big model (planner) |
|---|---|---|
| What | CNN, a few million params, plain PyTorch | Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled (Q8_0 GGUF), served by LM Studio |
| Rate | Every frame (`TimingConfig.target_fps`) | Every few seconds (`planner_interval_seconds`) |
| Learns via | Supervised learning on your recorded demonstrations (`train_bc.py`) | Not trained here - used as-is via LM Studio's local server |
| Output | Immediate action (move/buttons/mouse look) | One goal from a small fixed vocabulary |
| Runs in | `play.py`'s frame loop | Its own background thread (`planner.py`), talks to LM Studio over HTTP |

The small model is *goal-conditioned*: every frame it sees both the current
frame and the planner's latest goal (a one-hot vector), and its whole job is
to act well given that goal. The planner never blocks the fast loop - it
just periodically overwrites which goal is active.

## Why imitation learning instead of an RL agent

Trial-and-error RL (e.g. PPO) needs a reward signal, and there is no
generic way to read score/health/kills from arbitrary game pixels - a real
reward function needs per-game work (HUD OCR, a memory-read hook, etc.).
Recording yourself playing correctly and training the small model to
imitate those (frame, goal, action) examples sidesteps that problem
entirely: no reward function needed, and it directly encodes "play the way
I actually play" rather than whatever behavior happens to maximize a proxy
reward.

## How multi-game reuse actually works

The modularity is in `games/*.py`: each game (or genre) gets an
`ActionProfile` - its control scheme (which keys exist, whether it has
mouse look) - completely separate from `src/`, which never hardcodes any
one game's controls. `games/fps_default.py` and `games/platformer_default.py`
are two concrete, very different examples.

Two games that share a profile CAN share one trained checkpoint, and
training on demonstrations from several such games at once is how the small
model generalizes across them. This is **not** zero-shot generalization to
a genre it has no data for - a checkpoint trained only on FPS demonstrations
will not know how to play a platformer. If you want it to handle both, you
record demonstrations for both (they can even go in the same training run
if they share a profile) and train on the combined set.

## Setup

```bash
# System packages
sudo apt install wmctrl          # for scripts/calibrate_window.py
sudo modprobe uinput             # usually already loaded

# /dev/uinput and /dev/input/event* permission - either run as root for
# local experiments, or:
sudo tee /etc/udev/rules.d/99-uinput.rules <<'RULES'
KERNEL=="uinput", MODE="0660", GROUP="input"
RULES
sudo usermod -aG input "$USER"   # log out/in for this to take effect
sudo udevadm control --reload-rules && sudo udevadm trigger

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**LM Studio**: install it, download/load
`Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` (Q8_0), and
from the "Developer" tab start the local server (default
`http://localhost:1234`). `GET http://localhost:1234/v1/models` with the
server running to get the exact model id string for
`PlannerConfig.lm_studio_model` in `src/config.py`.

## Usage

```bash
cd src

# 1. Find your game window's screen region
python ../scripts/calibrate_window.py
# 2. Find your keyboard/mouse device paths
python ../scripts/list_input_devices.py
# Paste both into src/config.py (CaptureConfig, RecordingConfig)

# 3. Measure your machine's actual achievable capture/input loop rate,
#    adjust TimingConfig.target_fps accordingly - don't assume 60 is
#    reachable, measure it
python ../scripts/benchmark_loop.py --seconds 10

# 4. Record yourself playing (repeat for as many episodes/sessions as you
#    want - more, varied demonstrations make a better dataset)
python record_demo.py --game fps_default --episode-name run1

# 5. Train the small model on everything recorded so far
python train_bc.py --game fps_default --epochs 20

# 6. Watch it play, with the planner attached
python play.py --game fps_default --checkpoint ../checkpoints/fps_default.pt
```

## Reality checks

- **X11 only** (or XWayland) for screen capture - see `src/capture.py`.
  Check `echo $XDG_SESSION_TYPE`.
- **Input simulation is real and immediate** once `play.py` is running - no
  confirmation step. Test with an undertrained checkpoint in an
  offline/practice mode, not a live match.
- **The planner reasons before answering.** This specific checkpoint
  answers inside a `<think>...</think>` block even for a one-word goal
  choice - budget `planner_interval_seconds` for a full reasoning pass at
  your measured tokens/sec (`planner.py` logs actual decision latency), not
  for an instant one-word reply.
- **Demonstration quality is the ceiling.** Behavior cloning learns to
  imitate what you recorded, including mistakes and hesitation - a small
  number of clean, deliberate demonstration episodes will train a better
  model than a large number of messy ones.

## Project layout

```
src/
  config.py            All tunable runtime settings (capture, timing, LM
                        Studio connection, recording device paths).
  action_schema.py      The shared action contract (ActionProfile) between
                        recording, the model, and play - this is what makes
                        the small model swappable across games.
  capture.py            Screen capture (mss).
  input_sim.py          Keyboard/mouse simulation via a virtual uinput device.
  human_input_reader.py Reads YOUR real input via evdev during recording.
  record_demo.py        Records (frame, goal, action) demonstration episodes.
  dataset.py             Memory-mapped loading of recorded episodes.
  policy_model.py        The small CNN, goal-conditioned, PyTorch nn.Module.
  train_bc.py            Supervised training on recorded demonstrations.
  planner.py              LM Studio HTTP client for the big model.
  play.py                 Live play: trained small model + planner together.
games/
  fps_default.py          First-person control scheme.
  platformer_default.py   2D platformer control scheme (no mouse look) -
                          shows how different a non-FPS profile looks.
scripts/
  calibrate_window.py      Find a window's screen region.
  list_input_devices.py    Find keyboard/mouse evdev device paths.
  benchmark_loop.py        Measure your machine's actual achievable fps.
```

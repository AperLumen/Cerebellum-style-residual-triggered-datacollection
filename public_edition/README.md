# Cerebellum-Style Residual-Triggered Data Collection

An always-on collection layer for camera + IMU rigs that stores only what the
rig **cannot predict**. Inspired by corollary discharge in cerebellum-like
circuits: the brain cancels the sensory consequences of its own movements; what
remains in the residual is genuinely new information in the world.

**核心等式 / the core equation:** `residual = observation − prediction(self-motion)`
→ 残差里亮起来的，才值得占存储。/ whatever lights up in the residual deserves storage.

```
frontend (pluggable)      cancellation                triggering
HWC uint8 frames      ->  feature maps  ->  gyro warp + z-residual  ->  adaptive
                          (frozen MobileNetV3)  (self-motion cancelled)     trigger
                                                                             |
                                                                             v
                                          per-frame {keep, in_clip, surprise, z_map, clips}
```

- **Cancellation**: the IMU stream acts as a corollary discharge. Feature maps
  are warped by the gyro-derived shift; running statistics turn the leftover
  into a z-scored residual map. Predictable self-motion cancels out.
- **Residual triggering**: surprise (max |z|) drives a self-calibrating trigger
  with hysteresis — it stores short clips around unpredicted events and skips
  frames the rig's own motion already explains.
- **Acquisition layer** (`cerebellum_cancel/acquisition.py`): pluggable
  `step(frame, gyro, t) → {keep, in_clip, surprise, z_map, clips}` API; swap the
  frozen frontend for any callable with the same signature.

Scope: this layer does **motion deduplication / bandwidth pre-screening**.
"Kinetically unpredictable ≠ semantically important" — semantic arbitration
belongs to the vision model upstream, not here.

## Quickstart

Python 3.9+, then:

```bash
pip install -r requirements.txt

# 1. generate a synthetic rig recording (textured scene + self-rotation from
#    IMU + sparse injected events) — no external data needed
PYTHONPATH=. python scripts/make_synthetic.py --out demo_scene

# 2. feasibility run: cerebellum trigger vs naive-diff / uniform baselines
#    (recall / precision / stored-fraction on the ground-truth events)
PYTHONPATH=. python scripts/run_experiment.py demo_scene

# 3. render the side-by-side demo video (raw frame | residual heatmap | HUD)
PYTHONPATH=. python scripts/demo_video.py --src demo_scene --out demo_video.mp4
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

Covers the cancellation statistics, the trigger (gates, hysteresis,
self-calibration), the gyro/IMU IO helpers, the flow variant, end-to-end
evaluation, and the acquisition layer contract.

## Repository layout

```
cerebellum_cancel/   the package: io, core (cancel + trigger), featmap_cancel,
                     flow_cancel, adapter, pipeline, eval, acquisition
scripts/             make_synthetic.py (scene generator), run_experiment.py
                     (baselines comparison), demo_video.py (visual demo)
tests/               unit + integration tests, no external data required
```

Everything is generated locally; the repository contains no datasets.

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
evaluation, the acquisition layer contract, the full-rotation warp
(`rotwarp`), and the input contracts.

## Repository layout

```
cerebellum_cancel/   the package:
                       io                frame / IMU readers
                       core              cancellation statistics + adaptive trigger
                       featmap_cancel    feature-map cancellation ("shift" | "rotation")
                       rotwarp           full rotation-matrix warp (yaw+pitch+roll)
                       flow_cancel       optical-flow variant (quality upper bound)
                       adapter           cerebellum as an add-on to a frozen backbone
                       acquisition       the pluggable collection layer (public API)
                       pipeline          video + IMU -> clips, end to end
                       eval              trigger evaluation helpers
                       contracts         input contracts / explicit failure modes
scripts/             make_synthetic.py (scene generator), run_experiment.py
                     (baselines comparison), demo_video.py (visual demo)
tests/               unit + integration tests, no external data required
```

Everything is generated locally; the repository contains no datasets.

## What is deliberately NOT in this subset

This edition is the **online acquisition layer**. Two neighbours in the full
project are omitted, because they belong to the offline dataset layer and none
of the claims above depend on them:

| Omitted | What it is | Why it is out of scope |
| --- | --- | --- |
| `integrity.py` | acquisition-link integrity sentinel | operates on whole recordings, not per-frame acquisition |
| `dataset.py` | multimodal dataset container (trainable, publishable) | offline training assets, not collection |

Their contract tests are also omitted from `tests/`. The full versions live in
the parent repository.

## `rotwarp`: the full-rotation warp

`featmap_cancel` supports two prediction modes. `pred="shift"` is the original:
gyro → `(dx, dy)` constant translation. `pred="rotation"` uses a real rotation
matrix homography (yaw + pitch + **roll**) via `rotwarp.warp_by_rotation`.

On real phone IMU data the difference is not cosmetic — the same gyro stream
gives:

| predictor | residual ratio (< 1 = useful) |
| --- | --- |
| naive (no compensation) | 1.000 |
| `shift` (tan-approximation translation) | 1.433 |
| `rotation` (full matrix, with roll) | **0.614** |

Two independent causes: roll was dropped entirely, and a constant translation
approximates a field that is actually linear (roll) / quadratic (pitch) in
position, so the error grows with distance from the centre. Default is still
`"shift"` so previously published numbers are unchanged.

Two conventions must be declared per dataset (`rotwarp.signs_for` /
`rotwarp.lag_for`): the sign mapping and the gyro pairing lag. Getting either
wrong flips compensation into anti-compensation, and **the residual value alone
will not tell you** — it just looks "worse". See the module docstring.

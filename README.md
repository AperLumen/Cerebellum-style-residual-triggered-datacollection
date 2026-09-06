# Cerebellum-style residual-triggered data collection

A minimal prototype of the idea discussed from Perks et al. (Nature 2026,
doi:10.1038/s41586-026-10690-6): **a camera+IMU rig should not store what it
can predict**. The pipeline learns a "negative image" of the sensory
consequences of the rig's own motion (corollary discharge = IMU) and only
records the residual — the unexpected.

## The mapping

| Electric fish ELL (paper) | This prototype |
|---|---|
| Electric organ discharge (EOD) | Rig's own motion |
| Corollary discharge of EOD command | IMU gyro stream |
| Granule-cell temporal basis | Gyro features `[1, \|gx\|, \|gy\|, \|gz\|]` |
| Negative image (predictable response) | Per-cell linear prediction `w_fast + w_slow` + slow negative image `neg` |
| Associative depression/potentiation | Local anti-Hebbian updates, no backprop |
| Fast/slow plasticity pairing | `w_fast` (seconds) + `w_slow` (long), `w_fast` pulled toward `w_slow` |
| Equilibrium-rate plasticity gate | Per-cell gate: a cell that is *surprising right now* does not learn |
| Output cell residual | Per-cell z-scored residual; max over interior cells = surprise |
| Prey detection | Triggered clip saving (with hysteresis) |

## Current demo results (synthetic 60 s rig, 6 surprise events)

```
method                recall  precision   stored  clips
cerebellum            0.83    1.00        0.093    5
naive_diff            0.00    0.00        0.034    1
uniform(matched)      0.17    0.20        0.083    5
uniform(5%)           0.00    0.00        0.050    3
```

Naive frame-differencing drowns in self-motion (recall 0 even though the rig's
rotation is smooth); uniform sampling at the *same storage budget* as the
cerebellum method misses 5 of 6 events. One event is still missed by the
cerebellum pipeline (the earliest one, while statistics are still converging)
— see Limitations.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install numpy opencv-python-headless
.venv/bin/python scripts/make_synthetic.py --seconds 60 --out data/synthetic
PYTHONPATH=. .venv/bin/python scripts/run_experiment.py data/synthetic
```

## Real capture with an iPhone

1. **Video**: record with the Camera app (1080p30 is plenty).
2. **IMU**: run *Sensor Logger* (or any app exporting CSV) with gyroscope at
   ≥100 Hz. Expected columns: `time,gyroscope.x,gyroscope.y,gyroscope.z`.
3. **Sync**: before the scene starts, shake the phone sharply once. The spike
   is visible in both the video and the gyro norm; compute
   `imu_offset = t_shake_video - t_shake_imu` and pass `--imu-offset`.
4. **Ground truth**: annotate `events.json`:
   `{"events": [[t0, t1], ...], "fps": 30}` with the timestamps of anything
   that should have been recorded.
5. Run: `PYTHONPATH=. .venv/bin/python scripts/run_experiment.py data/my_rec`

## Adapter experiment: cerebellum as a frozen-backbone add-on

Extends the mapping from the signal level to **modern CV stacks**: freeze an
ImageNet-pretrained MobileNetV3-small (576-d pooled features), insert a
cerebellum adapter behind it, and ask whether the adapter's residual helps a
downstream event head.

Adapter (`cerebellum_cancel/adapter.py`): fixed random granule expansion
`phi([f_{t-1}, a_t])` -> linear predictor `W_slow` (ridge on train set) +
`W_fast` (online, unit-norm-gated anti-Hebbian, decays toward zero); frames
that are currently surprising do not update anything (equilibrium-rate gate);
output = z-scored residual `r_t = (f_t - f_hat_t) / sd`.

Benchmark (`data/dataset`): 50 synthetic sequences, textured scene + smooth
self-rotation + IMU; train at motion scale 1x, val at 3x (a regime shift).
Event = an object appearing/moving. Slow predictor reaches R^2 = 0.971.

```
variant            AP@1    AP@4   AP@16   AP@40   evRec@10%
A_static          0.536   0.871   0.842   0.815        1.00   (head on raw features)
B_slow_only       0.223   0.185   0.157   0.162        1.00   (head on residual, W_slow)
C_cerebellum      0.193   0.179   0.171   0.186        1.00   (head on residual, W_slow+W_fast)
C_energy             --      --      --   0.325        0.76   (untrained sum(z^2) readout)
```

**Honest negative result.** On this benchmark the adapter does not beat raw
frozen features, in any configuration (trained/few-shot/untrained readout).
Reasons, in order of how fundamental they are:

1. **Pooled deep features are already motion-robust.** A 1-4s motion regime
   shift does not hurt A at all — the trained head finds blob-sensitive dims
   and motion variation is learnable background. Cancellation has nothing to
   remove that the head cannot already ignore.
2. **Residual noise is not learnable-away.** Raw features = signal + stable
   background (a trained head subtracts the background for free); residual =
   signal + motion-prediction noise (a linear map on *pooled* features cannot
   express "content shifted by k px", so ego-motion leaks into the residual).
   Unless the predictor is near-perfect, cancellation trades removable
   background for unremovable noise — here that trade loses.
3. **The event is too salient.** A white blob on textured background is
   trivially separable in feature space; the ELL's advantage (surprise pops
   out of a cancelled background) needs events that are subtle relative to
   the predictable structure.

What *did* hold up: B/C AP is unchanged across the 1x->3x motion regime
shift (0.15-0.19 both regimes) while being trained only at 1x — the residual
is genuinely motion-invariant, and W_fast now learns stably (an initial
LMS-divergence bug: `fast_lr=0.02` vs stability bound `2/||phi||^2 ~ 0.016`,
fixed by unit-normalizing granule outputs). The strong version of the
mechanism remains the pixel-space one (Current demo results above), and its
feature-space form is the geometric variant in the next section.
*(Measured on the salient-event dataset; the current default dataset is the
open-set variant used below.)*

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python scripts/make_dataset.py --n-train 40 --n-val 10 --seconds 20
PYTHONPATH=. .venv/bin/python scripts/extract_features.py data/dataset   # needs torch+torchvision
PYTHONPATH=. .venv/bin/python scripts/train_adapter.py data/dataset
```

## Feature-map cancellation (geometric, untrained)

The pooled-feature adapter above loses because a linear map on *pooled*
features cannot express "content shifted by k px" — ego-motion leaks into the
residual as unlearnable noise. This module is the spatially structured fix,
and the closer analogue of the ELL negative image:

- backbone feature maps `F_t` (MobileNetV3-small `features[:3]`, 24x28x28,
  frozen, spatial structure kept);
- **predict** `F_hat_t(x) = F_{t-1}(x - dx_t)` where `dx_t` comes from the
  IMU gyro (corollary discharge) converted to map pixels —
  `dx = f_map * tan(radians(gyro) / fps)` (sign verified empirically);
- **residual** `r_t = F_t - F_hat_t`, z-scored by per-cell running statistics
  that update on calm frames only (equilibrium-rate gate; warmup frames
  bootstrap the stats unconditionally);
- **readout**: untrained — pooled z-energy, or strongest interior cell.
  Nothing anywhere is trained.

Benchmark: same 50 sequences but now the open-set version — train events are
faint blobs (alpha 0.3), val events at 3x motion are **faint bars in a
different screen region: a shape no trained head has ever seen**.

```
                          val AP   evRec@10%      (25 val sequences)
gyro_warp + maxcell        0.304       0.64   (untrained — best overall)
gyro_warp + energy         0.209       0.49   (untrained)
C cerebellum (pooled)      0.182       0.91   (trained on residual)
B slow-only  (pooled)      0.165       0.91
naive_diff + maxcell       0.171       0.51   (untrained, no warp)
naive_diff + energy        0.172       0.34
A static head (pooled)     0.119       0.64   (trained on raw features — collapsed)
```

Three claims, in increasing strength:

1. **The gyro warp works.** Calm-frame residual drops 2.6x (0.48 -> 0.18)
   and every gyro_warp readout beats its naive_diff twin at 3x motion.
2. **Surprise representations transfer; appearance representations do not.**
   A's head trained on raw features collapses on unseen event types (recall
   0.46); heads trained on *residuals* hold 0.8-0.9 recall — they learned the
   noise model, not the event.
3. **A fixed, untrained front-end wins the open-set regime outright.**
   gyro_warp + maxcell (0.271 AP) beats every trained-head variant, with zero
   training and zero event-type assumptions — the architecture固化:
   alignment, local gating, and a trivial readout are all the machinery.

Caveats: absolute APs are modest (faint bars under 3x motion are hard for
everything); the open-set margin held when val was expanded 10 -> 25
sequences (gyro_warp+maxcell 0.271 -> 0.304 AP, ordering unchanged);
"untrained" still assumes the gyro-to-shift calibration (f_map) is known.

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python scripts/make_dataset.py --n-train 40 --n-val 25 \
    --seconds 20 --event-alpha 0.3 --val-event-shape bar
PYTHONPATH=. .venv/bin/python scripts/extract_featmaps.py data/dataset
PYTHONPATH=. .venv/bin/python scripts/run_featmap_eval.py data/dataset
```

## Honest limitations

- **Screen recording is a dead end for this method (measured, not assumed).**
  A `screencapture -v` recording of the scene was tested end-to-end: static
  frames are clean (residual 4e-5), but under motion even an *oracle*
  compensator (perfect knowledge of the true shift) leaves residual ~0.066,
  and the best integer-aligned frame pair differs by 1.16/255 vs 0.009/255
  for the raw frames — a ~130x floor. ProMotion compositor frame-blending /
  HEVC B-frame smoothing destroys exactly the signal this method needs
  (sharp frame-to-frame temporal sampling). Conclusion: validate on real
  cameras (iPhone / webcam), which sample sharp instants, not on screen
  captures.
- **Rotation-only compensation**: translation/parallax is not modeled (no
  accelerometer fusion); the learned weights absorb some of it, but a walking
  rig will produce larger residuals than this synthetic demo.
- **The synthetic demo is favorable**: real scenes have autofocus breathing,
  exposure adaptation — all of which look like "surprise" until the negative
  image absorbs them. That absorption is exactly what the multi-timescale
  plasticity is for, but it needs real-data tuning.
- **Thresholds are hand-set** (`TriggerConfig`): a real deployment needs the
  per-cell statistics to be the only calibrated quantity.
- **Downstream value not yet measured**: the headline claim — residual clips
  train downstream models better per byte than uniform data — needs a
  training experiment. That is the next milestone, not an afterthought.

## Next steps

1. **iPhone field test** (the real experiment): record handheld video + IMU
   per README; annotate events; run the same eval. This is now the critical
   path — the screen-recording shortcut is disproven.
2. Webcam-facing-screen variant if you want a desk-bound intermediate: the
   camera channel is temporally sharp, but needs a mirror or external camera
   pointed at the screen (the built-in one films the user).
3. Downstream training comparison: train a small detector on
   cerebellum-clips vs uniform-sampled clips at matched byte budgets.
4. Add accelerometer features to the action vector (translation prediction).

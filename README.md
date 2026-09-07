# Cerebellum-style residual-triggered data collection

[English](README.md) | [中文](README.zh.md)

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
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/make_synthetic.py --seconds 60 --out data/synthetic
PYTHONPATH=. .venv/bin/python scripts/run_experiment.py data/synthetic
```

Run the unit tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/
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

## First real capture (webcam, desk scene)

One 65 s desk recording with an external webcam (30 fps) + phone gyro stream
(synced by the shake trick, `imu_offset` 11.342 s), 6 annotated events:

```
method                recall  precision   stored  clips
cerebellum            0.33    0.50        0.102    4
naive_diff            0.00    0.00        0.042    2
uniform(matched)      0.17    0.17        0.092    6
uniform(5%)           0.17    0.33        0.046    3
```

The relative ordering survives on real data (cerebellum > baselines at matched
budget), but absolute recall drops from 0.83 to 0.33: the two late events are
caught, the four earlier ones are missed, and one early clip (5.5–8.9 s) is a
false trigger. A 60 s desk scene is exactly the regime the README's
limitations predict to be hard — autofocus breathing, exposure adaptation,
small background motion — all of which read as "surprise" until the negative
image absorbs them. This recording predates the multi-timescale tuning and
still uses the hand-set thresholds; treat it as a sanity check (real sensor
noise does not break the pipeline) rather than a performance measurement.

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_experiment.py data/webcam_rec
```

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

## Downstream training comparison (the headline claim, measured)

Does a rig that stores only cerebellum-triggered clips train a downstream
model **better per stored byte** than uniform sampling? Training frames are
sampled from the 40 train sequences by each method at matched storage
budgets; the downstream model is the same LogisticHead on frozen MobileNetV3
features as in the adapter experiment; eval is on the full 25-sequence val
split (frame-level AP, event recall at a top-10% budget). Clip source is the
feature-map cancellation front-end above (gyro warp + maxcell z^2 readout) —
the pixel-space trigger was measured to never fire on this dataset (train
events are faint, alpha 0.3; its z < 6 persistence gate needs salient
events: recall 0.13 even at threshold 2.5).

Triggering is **self-calibrating** (`AdaptiveTrigger`): the raw z^2 score is
normalized by its own calm-frame running statistics before the dimensionless
6/2.5 hysteresis gate, so no hand-set score threshold is needed. The
previous hand-set operating point (T=300) is kept as a reference row.

```
method                stored  frames   val AP   evRec@10%
cerebellum (adaptive)  0.034     826    0.298       0.96
uniform(0.5x)          0.061    1457    0.135       0.58
uniform(1x)            0.074    1766    0.131       0.59
uniform(2x)            0.091    2181    0.125       0.58
uniform(4x)            0.121    2894    0.120       0.57
oracle(adaptive)       0.019     449    0.191       0.72
--
cerebellum(T=300)      0.150    3593    0.210       0.92   (hand-set reference)
```

Claims, strongest last:

1. **Cerebellum clips beat uniform at 4x the bytes.** At 3.4% stored (826
   frames), AP 0.298 / evRec 0.96; uniform needs 3.5x the frames (12%) and
   still loses by a wide margin (0.120 / 0.57).
2. **The win is the data, not the budget.** Scaling uniform's budget up does
   not close the gap (it gets worse past 0.5x — more dilute background), so
   the advantage is event enrichment, not storage volume.
3. **Cerebellum clips even beat the oracle** (budget spent only on
   ground-truth event frames) on both metrics (0.96 vs 0.72 evRec). The
   trigger keeps event onsets, disappearances, and motion-boundary context —
   exactly the frames a discriminative head needs — where oracle keeps only
   mid-event frames.
4. **Self-calibration beats the hand-set threshold.** The adaptive trigger
   stores 4.3x fewer frames than the T=300 operating point AND trains a
   better head (0.298 vs 0.210 AP) — normalizing by the trace's own calm
   statistics is a sharper criterion than any fixed raw threshold.

Regime check (val split, 3x motion, mean stored fraction): adaptive 0.339
vs T300 0.379. Both inflate under the regime shift — the calm gate refuses
to absorb motion bursts into the baseline (by design), so a 3x noisier rig
still stores proportionally more. Self-calibration buys precision at the
operating regime, not regime invariance; the slow-timescale absorption of a
persistent new motion level would need longer streams than 20 s.

Head robustness (same clips, stronger downstream model): swapping the
LogisticHead for a 128-hidden MLP does **not** raise val AP — it lowers it
(cerebellum adaptive: 0.298 -> 0.200; a 5-config regularization sweep —
hidden/l2/dropout/iters — tops out at 0.247). The val regime (3x motion +
unseen bar shape) punishes capacity: the MLP overfits motion-specific
correlations in the training clips, the same failure the adapter section
documented for appearance-based heads ("surprise representations transfer;
appearance representations do not"). What the MLP does improve is the top of
the ranking (evRec@10% 0.96 -> 0.99-1.00 on cerebellum clips). The
data-value claim is unchanged across every head and config: cerebellum
clips beat uniform at all matched budgets, and beat the oracle on event
recall. On this benchmark, AP is data-limited, not model-limited.

Caveats: byte budget is proxied by frame count (bitrate per frame is similar
across methods by construction); the frozen feature extractor is assumed to
ship with the rig (the adapter section's premise); 6/2.5 are still chosen
constants, just dimensionless ones; and the downstream head is deliberately
small — the claim is about *data value per byte*, not peak detector
performance (a stronger MLP head is measured below: it does not raise AP).

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_downstream.py data/dataset
```

## Optical-flow baseline and cost profile

The CV-native alternative to the gyro warp is dense optical flow: compensate
with what the image content itself says, no IMU needed. Measured end-to-end
on this benchmark (`cerebellum_cancel/flow_cancel.py`, DIS preset MEDIUM;
15 train + 5 val sequences):

```
                    calm residual (mean |err|)     stored   evRec
                    train (1x)   val (3x)          val(3x)  val(3x)
naive (no warp)        0.093        0.221            --       --
gyro warp              0.079        0.129           0.113    0.30
DIS flow warp          0.070        0.103           0.192    0.33
```

Median per-frame latency on this machine (320x240 frames, 24x28x28 maps):

```
stage                     ms/frame
gyro shift (24x28x28)      0.032
pixel pipeline core        0.171   (warp + cells + plasticity + trigger)
frozen backbone fwd        0.827   (MobileNetV3 features[:3], 224x2, MPS)
DIS flow (320x240)         3.813
full flow-cancel step      4.456
```

Takeaways:

1. **Flow does buy residual quality** — 11% lower calm residual than the
   gyro warp at 1x, 20% at 3x (DIS tracks the actual content displacement,
   including what the gyro approximation misses). This is real, and the
   largest honest advantage of the flow baseline.
2. **It costs 26x the entire pixel pipeline** (4.5 ms vs 0.17 ms per frame,
   ~120x at the warp step alone) — on a 30 fps budget the pixel pipeline
   uses 0.5% of the frame time, the flow step 13%. And these timings are on
   a laptop; on-device (phone CPU, no MPS) DIS is typically several times
   slower still.
3. **It needs texture and stability**: DIS loses the target under fast
   motion (one 3x val sequence stores 59% of the stream vs gyro's 13% —
   flow noise reads as surprise) and has no answer for textureless or
   low-light patches, where the gyro warp is unaffected. Event regions
   corrupt the local flow field, leaking into the residual.
4. **Architecture reading**: the gyro warp is a ~30 µs matrix op that
   turns the IMU into a corollary discharge; flow is a per-frame computer
   vision computation that must run before the compensation it feeds. For
   a rig whose whole point is *cheap always-on* surprisingness, the IMU is
   the right layer — flow is the quality ceiling to compare against, and
   the residual-quality gap (11-20%) is the honest price of the 100x
   cost ratio. A sub-pixel gyro shift and a better f_map calibration would
   close part of that gap.

Reproduce:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/dataset
```

## Resolution: the knob that decides whether the gyro warp pays

The gyro warp can only remove motion the feature map can *see*. The same
rotation produces a content shift of `f_map * tan(omega/fps)` map px, and
`f_map` scales with backbone input size. Re-running the benchmark at three
input sizes (same 40 train + 25 val sequences, MobileNetV3 features[:3],
stride 8; `f_map` 24.25 at 224 px is the rig default) isolates resolution
as the variable:

```
input   map      f_map   calm residual (val)   gyro gain   gyro shift   DIS flow
        size           naive   gyro   flow    vs naive    ms/frame     ms/frame
224     28x28    24.25   0.221   0.130  0.107   -41%        0.032        3.86
336     42x42    36.38   0.187   0.089  0.069   -52%        0.061        4.20
448     56x56    48.50   0.169   0.072  0.052   -58%        0.087        4.35
```

(backbone forward is ~0.6 ms/frame at every size on MPS — shallow stride-8
layers are memory-bound; the protocol cost of resolution is in storage and
bandwidth, not compute. Same-session re-measurement, `runs/flow_eval_res*.json`.)

The gain is monotone and has not saturated: every 1.5x input size removes
another ~6-10 points of calm residual that the naive predictor cannot
touch. The flow gap narrows with resolution too (flow/gyro 0.82 -> 0.73):
the gyro warp's main error source — sub-pixel quantization of the shift —
shrinks as the map gets finer. This also retroactively explains both real
regimes: desk-egocentric Open-AoE at 224 px is f_map ~11-19 with rotation
under ~5 deg/s (shift < 0.05 map px — nothing to remove, warp neutral),
while the high-motion pair at p95 |yaw| 31-52 deg/s reaches ~0.5-0.7 map
px — inside the window where model error (uncompensated roll, calibration)
starts to *add* residual instead of removing it.

Design rule for the protocol layer: **corollary-discharge cancellation is
a resolution-gated primitive** — pair it with fine-grained features
(DINOv2 patch tokens, or stride-8 maps at >= 448 px input) and it removes
half the predictable residual for ~0.09 ms/frame; at coarse maps it is
free but neutral. The same variable decides whether the *trigger* keeps
working: at 448 px the gated self-normalizing trigger froze to zero clips
on all 5 synthetic val sequences (baseline surprise grows and
heavy-tails with resolution), while winsorized robust stats fired on all
5 — the real-video gate failure, reproduced on synthetic data.

Reproduce:

```bash
# featmaps at other input sizes; f_map scales linearly (24.25 * size/224)
for S in 336 448; do
  .venv/bin/python scripts/extract_featmaps.py data/dataset_$S --size $S
  PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/dataset_$S \
      --n-train 40 --n-val 25 --size $S
done
```
(The `data/dataset_$S` dirs hold symlinks to the frames, `imu.csv` copies
and `events.json` with the scaled `f_map`; delete/rebuild at will — the
featmaps re-extract in minutes.)

## Flow-as-arbiter cascade

A middle architecture between "gyro only" and "flow every frame": the gyro
warp runs always-on; DIS flow is consulted **only on trigger candidates**, as
an arbiter that vetoes the ones it can explain. Composition per candidate
peak frame (`scripts/run_arbiter_eval.py`): the gyro shift pre-warps both the
gray pair (so DIS only searches the small residual the gyro model
mispredicts — without this, DIS loses track at large rotations, exactly where
triggers fire) and F_{t-1}; DIS then refines along the residual flow. A
candidate run is vetoed when the flow-refined residual keeps < `explain_ratio`
(0.5) of the gyro residual's interior energy — i.e. the residual was
ego-motion misprediction, not surprise.

Measured on the full benchmark (15 train + 5 val, run-level readout —
fixed-threshold runs, which unlike `AdaptiveTrigger`'s adaptive clips are
monotone in the trace):

```
                 stored   precision   evRec   DIS calls/seq   amortized ms/frame
train gyro        0.401      0.257      1.00        --               --
train cascade     0.401      0.257      1.00       21.3            0.135
val   gyro        0.520      0.146      1.00        --               --
val   cascade     0.498      0.155      1.00       23.6            0.150
```

Honest read: on this synthetic rig the cascade is almost a no-op (6 of 118
val runs vetoed). That is a *measured* property of the benchmark, not a
weakness of the arbiter — three mechanisms explain it, each verified:

1. The synthetic rig is pure global yaw/pitch translation (roll < 1 deg/s),
   which is exactly the model class of the gyro warp, so the warp is
   near-exact and calm-frame residuals are feature noise, not modeling error
   (flow-refined residual energy ≈ gyro residual energy, ratio 1.03–1.27).
2. At 28x28 maps, subpixel miscalibration stays under the feature-noise
   floor: stressing f_map by 10–50% still leaves `rho ≈ 1`, so no veto fires
   (`--fmap-scale` in the eval script).
3. What the arbiter *would* catch — structured residuals the global-shift
   model cannot express (real roll, parallax/depth, large calibration
   error) — does not exist in this data.

So the cascade's status is: mechanism and cost validated (~0.15 ms/frame
amortized, ~25x cheaper than running DIS every frame, veto power retained
when it matters), value pending real-camera validation — it is the right
safety net for exactly the failure modes the synthetic rig lacks. The
run-level readout also exposed that `AdaptiveTrigger`'s running normalization
is non-monotone (zeroing a vetoed run can *re-sensitize* later clips), which
the cascade evaluation sidesteps by comparing at fixed-threshold run level.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_arbiter_eval.py data/dataset
```

## Real egocentric validation (Open-AoE)

The synthetic rig is pure global yaw/pitch, so a first real-world check
uses [Open-AoE](https://github.com/ant-research/Open-AoE) (inclusionAI/
OpenAoE-2000h on HF): ~700h of consumer-phone egocentric video with a
visual-SLAM camera trajectory and time-aligned atomic-action labels. Nine
diverse segments (bathroom / bedroom / childs-room / dining / kitchen /
living-room / workshop / workspace / shop, 134–301 s each, 67.4k frames
total) were selected from a 60-segment probe by `scripts/fetch_aoe.py`
(greedy scene/verb coverage, duration and head-motion filters) and
prepared by `scripts/prep_aoe.py`, which differentiates `cam_c2w` into
the gyro-like signal the pipeline expects (per-axis signs
auto-calibrated against DIS flow; |r| = 0.18–0.93, worst on two
near-static clips where there is no motion to correlate) and writes the
standard sequence layout, with each phone's real focal length
(`f_map` = 10.95–19.17, fx 751–1315, not the synthetic rig's 24.25).

Measured first on one segment (seq_000), then generalized across all nine —
three findings that the synthetic rig could not produce:

1. **Real ego-motion is sub-pixel at 28x28.** Median content shift is
   0.013 map px (p95 0.05, max 0.16), so the gyro warp is physically
   neutral here: calm residual naive 0.0323 vs gyro 0.0328. The
   corollary-discharge warp only earns its keep at finer resolution,
   deeper features, or walking-scale motion — not at a desk-egocentric
   clip at 224 px input.
2. **The equilibrium-rate gate freezes on real data.** 96.4% of frames
   have surprise > 4, so after warmup the running stats update on only
   3.6% of frames and the sd ratchets down (0.92 -> 0.17); z saturates
   (median max|z| = 13) and the two readouts degenerate in opposite
   directions: fixed-threshold runs cover 92% of all frames, while
   `AdaptiveTrigger`'s relative normalization fires **zero** clips. The
   calm-gate assumption (surprise is sparse) holds for pop-in synthetic
   events and fails for real video. The fix is winsorized always-update
   stats (`CancelConfig.robust`): outliers enter the running stats
   clipped at 3 sigma, so they can neither inflate the scale nor freeze
   it through the gate. With robust stats the trigger fires again on
   this segment: **9.9% stored at 0.79 event recall (zero clips
   before)**. Telling cross-check: the DIS flow baseline
   (`FlowWarpCancel`) shares the same gated stats and also fires zero
   clips on real video — the gate, not the predictor, was the failure.
3. **Residual spikes are real but misaligned with action labels.** The
   top residual frames are fast hand/object motion with motion blur
   (`runs/aoe_peaks/`), yet the score is *anti*-correlated with the
   annotation's action intervals (AUC 0.47): atomic actions are extended
   intervals with varying motion energy, while the residual indexes
   motion transients. Frame-level event-recall against human action
   labels is therefore the wrong metric on real data; for the
   residual-triggered *storage-selection* architecture the meaningful
   metric is downstream value of the selected clips (as in the
   downstream section above), not label alignment.

Across all nine segments the three findings hold: gyro-warp calm residual
never beats naive by more than noise (sub-pixel ego-motion everywhere,
median content shift < 0.05 map px); the robust-stats trigger fires on
every clip (stored 1.2–13.1%, median 9.2%, event recall 0.14–0.79, median
0.46 vs action labels — the seq_000 spread, now with error bars); and the
DIS flow baseline still mostly fires zero clips under the same gated
stats, except where flow itself hallucinates triggers (seq_004: 99%
stored). Per-sequence numbers: `runs/flow_eval.json`.

At the runs level the arbiter generalizes too (`runs/arbiter_eval.json`):
across the nine clips, 140 candidate runs/seq under robust stats, the
cascade vetoes 8–31% of them (flow calls amortize to 0.89 ms/frame),
cutting storage 0.89 -> 0.72 of flagged frames at precision 0.994 with
event recall 1.00 -> 0.94; at the clip level, AdaptiveTrigger stored
7.9% -> 20.6% with recall 0.49 -> 0.55 — the vetoed runs were dominated
by ego-motion modeling error even at sub-pixel shifts.

**The high-motion end** (2 segments at the top of the 259-probed pool,
p95 |yaw| 31–52 deg/s vs 1–41 on the desk clips; `data/aoe_motion`,
`runs/flow_eval_motion.json`) finally leaves the sub-pixel regime — and
the ranking of predictors flips, with caveats. Calm residual: gyro warp
is now *worse* than naive (0.090 vs 0.080; the 2D-shift model leaves
roll uncompensated and the sign-calibration is weaker there, y r = 0.15
on the warehouse clip), while DIS flow clearly wins on quality (0.052) —
and loses on cost by ~250x (8.4 ms vs 0.033 ms per frame). The robust
trigger also thins out (stored 1.4%/1.2%, evRec 0.09/0.04): continuous
real motion makes the residual heavy-tailed, the winsorized sd inflates,
and the z-bar rises. The arbiter is exactly the bridge for this regime:
it vetoes 11% of runs on the liveliest clip (stored 0.92 -> 0.62 at
precision 1.00, amortized 0.58 ms/frame). Honest bottom line: on this
corpus even the top-decile rotation stays ~0.5 map px/frame at 28x28, so
the gyro warp's quality window is narrow at this resolution — its case
is cost, and the flow arbiter is the paid upgrade when the cheap model
breaks. (n=2; the axis-calibration caveat above applies.)

```bash
# pick + download segments (probe -> select -> fetch, each cached)
.venv/bin/python scripts/fetch_aoe.py probe --n 60
.venv/bin/python scripts/fetch_aoe.py select --k 8
# high-motion regime: filter by trajectory-derived rotation before selecting
.venv/bin/python scripts/fetch_aoe.py select --k 3 --motion-min 60
.venv/bin/python scripts/fetch_aoe.py fetch
# convert to the standard sequence layout, one dir per segment
for d in data/aoe_src/<segment>; do
  PYTHONPATH=. .venv/bin/python scripts/prep_aoe.py "$d" --out data/aoe/val/seq_XXX
done
PYTHONPATH=. .venv/bin/python scripts/extract_featmaps.py data/aoe
PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/aoe --n-train 0 --n-val 9 --robust
PYTHONPATH=. .venv/bin/python scripts/run_arbiter_eval.py data/aoe --n-train 0 --n-val 9 --robust
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
- **Thresholds**: the featmap/downstream path now self-calibrates
  (`AdaptiveTrigger` — dimensionless 6/2.5 gate on the trace's own calm
  statistics). The pixel-space demo path still uses hand-set `TriggerConfig`
  values; and the regime check shows self-calibration buys precision at the
  operating regime, not regime invariance.
- ~~Downstream value not yet measured~~ **done** (section above): cerebellum
  clips train better detectors per stored byte than uniform sampling — and
  better than an oracle that spends its budget only on event frames.

## Next steps

1. **iPhone field test** (the real experiment): record handheld video + IMU
   per README; annotate events; run the same eval. This is now the critical
   path — the screen-recording shortcut is disproven.
2. Webcam-facing-screen variant if you want a desk-bound intermediate: the
   camera channel is temporally sharp, but needs a mirror or external camera
   pointed at the screen (the built-in one films the user).
3. ~~Downstream training comparison~~ **done** (section above): cerebellum
   clips beat uniform at 4x bytes, and beat the oracle on event recall.
4. Add accelerometer features to the action vector (translation prediction).

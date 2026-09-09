"""Probe 2 (read-only): isolate WHY the DINOv2 score loses temporal support.

(a) consecutive-ness of trigger-level z>6 / z>12 frames (persistence gate)
(b) what those burst frames coincide with (event? gyro spike?)
(c) robust=True (winsorized always-update) cancel stats: does the event
    structure and multi-frame support recover? Does the trigger fire?
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cerebellum_cancel.core import AdaptiveTrigger, TriggerConfig, SurpriseTrigger
from cerebellum_cancel.featmap_cancel import CancelConfig, FeatureMapCancel, maxcell_score
from cerebellum_cancel.io import load_imu_csv

sys.path.insert(0, "scripts")
from train_downstream import seq_dirs


def run(seq, name, robust):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/{name}").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    f_map = 24.25 * fm.shape[-1] / 28.0
    canc = FeatureMapCancel(CancelConfig(f_map=f_map, robust=robust))
    zs, sur = [], []
    for t in range(n):
        out = canc.step(fm[t], gyro[t])
        zs.append(out["z"]); sur.append(out["surprise"])
    return maxcell_score(np.stack(zs)), np.array(sur), fps, n


def trig_z(score, fps, n):
    cfg = TriggerConfig()
    inner = SurpriseTrigger(cfg, fps)
    mu, var, cnt = 0.0, 1.0, 0
    zs = np.zeros(n)
    for t in range(1, n):
        cnt += 1
        s = float(score[t])
        if cnt == 1:
            mu = s
        sd = float(np.sqrt(var)) + 1e-3 * (abs(mu) + 1.0)
        z = (s - mu) / sd
        zs[t] = z
        if cnt <= 30:
            d = s - mu
            mu += 0.02 * d
            var += 0.02 * (d * d - var)
            continue
        on = inner.update(t / fps, z, 1.0)
        if not on and z < cfg.threshold_k:
            d = s - mu
            mu += 0.02 * d
            var += 0.02 * (d * d - var)
    return zs, inner.finalize(n / fps)


for seq in seq_dirs("data/dataset", "train")[:2]:
    labels = np.load(f"{seq}/labels.npy")
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    ev = labels > 0
    print(f"\n=== {seq} ===")
    for name in ("featmaps.npy", "featmaps_dino.npy"):
        score, sur, fps, n = run(seq, name, robust=False)
        z, clips = trig_z(score, fps, n)
        hi = np.where(z > 6)[0]
        # consecutive-run structure of high-z frames
        runs = []
        for i in hi:
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        print(f"\n-- {name} (gated)")
        print(f"  z>6 frames: {len(hi)}, runs: {[len(r) for r in runs]}; "
              f"event-coincidence: {[bool(ev[i]) for i in hi]}")
        print(f"  clips: {len(clips)}")
        # robust variant
        score_r, sur_r, _, _ = run(seq, name, robust=True)
        z_r, clips_r = trig_z(score_r, fps, n)
        evr = ev[:n]
        if evr.sum() and (~evr).sum():
            calm_p99 = np.percentile(score_r[~evr], 99)
            frac = (score_r[evr] > calm_p99).mean()
        else:
            frac = float("nan")
        print(f"-- {name} (robust winsor) surprise mean={sur_r.mean():.2f} "
              f"(gated was {sur.mean():.2f}); "
              f"events>calm-p99: {frac:.2f} (gated was "
              f"{(score[ev[:n]] > np.percentile(score[~ev[:n]], 99)).mean():.2f}); "
              f"clips: {len(clips_r)}")

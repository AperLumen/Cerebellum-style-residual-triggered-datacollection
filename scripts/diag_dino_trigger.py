"""Diagnostic (read-only, single process): why does the AdaptiveTrigger fire
0 clips on DINOv2 featmaps while MobileNetV3 works?

For a few train sequences, per backbone (featmaps.npy vs featmaps_dino.npy):
  - raw maxcell z^2 score stats (mean/std/quantiles, event-frame vs calm)
  - featmap-cancel internals: per-frame surprise (max |z|), stat-update gate
    block rate (the 448px "gated freeze" precedent)
  - instrumented AdaptiveTrigger: z trace, running mu/sd, onset-block reason

No files written; prints only. Usage:
  PYTHONPATH=. .venv/bin/python scripts/diag_dino_trigger.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cerebellum_cancel.core import (AdaptiveTrigger, SurpriseTrigger,
                                    TriggerConfig)
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                              maxcell_score)
from cerebellum_cancel.io import load_imu_csv

sys.path.insert(0, "scripts")
from train_downstream import seq_dirs  # noqa: E402


def score_seq(seq, featmaps_name):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/{featmaps_name}").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    f_map = 24.25 * fm.shape[-1] / 28.0
    canc = FeatureMapCancel(CancelConfig(f_map=f_map))
    z, sur = [], []
    for t in range(n):
        out = canc.step(fm[t], gyro[t])
        z.append(out["z"]); sur.append(out["surprise"])
    z = np.stack(z); sur = np.array(sur)
    return maxcell_score(z), sur, z, fps, n


def cancel_gate_stats(seq, featmaps_name):
    """Re-run the cancel with gate instrumentation: how often is the calm
    gate (surprise < calm_tau=4) closed, i.e. stats frozen?"""
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/{featmaps_name}").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    f_map = 24.25 * fm.shape[-1] / 28.0
    canc = FeatureMapCancel(CancelConfig(f_map=f_map))
    blocked = 0
    for t in range(n):
        out = canc.step(fm[t], gyro[t])
        if canc.n > canc.cfg.warmup and not out["learned"]:
            blocked += 1
    return blocked, n


def instrumented_trigger(score, fps, n):
    """Mirror AdaptiveTrigger.update but record everything."""
    cfg = TriggerConfig()
    inner = SurpriseTrigger(cfg, fps)
    stat_ema, warmup = 0.02, 30
    mu, var, cnt = 0.0, 1.0, 0
    zs = np.zeros(n)
    onsets = 0
    for t in range(1, n):
        cnt += 1
        s = float(score[t])
        if cnt == 1:
            mu = s
        sd = float(np.sqrt(var)) + 1e-3 * (abs(mu) + 1.0)
        z = (s - mu) / sd
        zs[t] = z
        if cnt <= warmup:
            d = s - mu
            mu += stat_ema * d
            var += stat_ema * (d * d - var)
            continue
        was_on = inner.on
        on = inner.update(t / fps, z, 1.0)
        if not was_on and on:
            onsets += 1
        if not on and z < cfg.threshold_k:
            d = s - mu
            mu += stat_ema * d
            var += stat_ema * (d * d - var)
    return zs, onsets, mu, var


def main():
    seqs = seq_dirs("data/dataset", "train")[:3]
    for seq in seqs:
        labels = np.load(f"{seq}/labels.npy")
        print(f"\n=== {seq} ===")
        for name in ("featmaps.npy", "featmaps_dino.npy"):
            score, sur, z, fps, n = score_seq(seq, name)
            score = score[:n]
            ev = labels[:n] > 0
            zs, onsets, mu, var = instrumented_trigger(score, fps, n)
            blocked, nn = cancel_gate_stats(seq, name)
            c = 384 if "dino" in name else 24
            print(f"\n-- {name} (C={c}) n={n} events={ev.sum()}")
            print(f"  score: mean={score.mean():.1f} sd={score.std():.1f} "
                  f"p50={np.percentile(score,50):.0f} p99={np.percentile(score,99):.0f} "
                  f"max={score.max():.1f}")
            if ev.sum() and (~ev).sum():
                rk = (score < score[ev][:, None]).mean(axis=1)  # calm-fraction below each event score
                print(f"  event-frame score: mean={score[ev].mean():.1f} "
                      f"median-rank-vs-calm={np.median(rk):.3f} "
                      f"(1.0=above all calm, 0.5=at calm median)")
                calm_hi = np.percentile(score[~ev], 99)
                print(f"  calm p99={calm_hi:.0f}; events above calm-p99: "
                      f"{(score[ev] > calm_hi).mean():.2f}")
            print(f"  cancel internals: mean surprise(max|z|)={sur.mean():.2f} "
                  f"p99={np.percentile(sur,99):.1f} max={sur.max():.0f}; "
                  f"stat-gate blocked {blocked}/{nn} "
                  f"({100*blocked/nn:.0f}% after warmup)")
            print(f"  trigger: final mu={mu:.1f} sd={np.sqrt(var):.1f}; "
                  f"z p99={np.percentile(zs,99):.2f} z max={zs.max():.2f}; "
                  f"frames z>6: {(zs>6).sum()} z>12: {(zs>12).sum()}; "
                  f"onsets={onsets}")


if __name__ == "__main__":
    main()

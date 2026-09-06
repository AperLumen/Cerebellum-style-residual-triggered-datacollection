"""Downstream training comparison: do cerebellum-triggered clips train an
event detector better per stored byte than uniform sampling?

Setup (the README's headline claim, measured):
  - byte budget = fraction of train-stream frames retained for downstream
    training (frame count is the proxy; encoding bitrate per frame is similar
    across methods by construction)
  - clip source: the feature-map cancellation pipeline (gyro warp + maxcell
    z^2 readout), the best-performing front-end on this benchmark (README).
    The pixel-space trigger was measured to never fire here: train events are
    faint (alpha 0.3) and its z < 6 persistence gate requires salient events.
  - triggering: AdaptiveTrigger (default) — the raw score is normalized by
    its own calm-frame statistics, so the dimensionless 6/2.5 gate needs no
    hand-set score threshold and survives the 1x -> 3x motion regime shift.
    --fixed-thr 300 keeps the previous hand-set operating point for
    comparison.
  - baselines per operating point: uniform 1 s clips at k x the cerebellum
    budget; oracle = budget spent only on event frames
  - downstream model: LogisticHead on frozen MobileNetV3 features (the same
    head as the adapter experiment, so results are directly comparable)
  - eval: mean per-sequence frame-level AP on the FULL val split, plus
    event recall at a top-10% frame budget

Usage: PYTHONPATH=. .venv/bin/python scripts/train_downstream.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.adapter import LogisticHead, average_precision
from cerebellum_cancel.core import AdaptiveTrigger, SurpriseTrigger, TriggerConfig
from cerebellum_cancel.eval import uniform_baseline
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               maxcell_score, rollout)
from cerebellum_cancel.io import load_imu_csv

CACHE = "runs/downstream_clips.json"
FPS_FALLBACK = 30.0


def seq_dirs(dataset, split):
    return sorted(glob.glob(f"{dataset}/{split}/seq_*"))


def score_seq(seq):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", FPS_FALLBACK))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    canc = FeatureMapCancel(CancelConfig())
    z, _ = rollout(canc, fm[:n], gyro[:n])
    return maxcell_score(z), fps, n


def clips_for_score(score, fps, n, mode):
    if mode == "adaptive":
        trig = AdaptiveTrigger()
        for t in range(1, n):
            trig.update(t / fps, score[t], fps)
        return trig.finalize(n / fps)
    onset = float(mode[1:])          # "T300" -> 300.0
    trig = SurpriseTrigger(TriggerConfig(threshold_k=onset, off_k=onset / 2),
                           fps)
    for t in range(1, n):
        trig.update(t / fps, score[t], 1.0)
    return trig.finalize(n / fps)


def load_or_compute_clips(dataset, modes):
    if os.path.exists(CACHE):
        with open(CACHE) as fp:
            cache = json.load(fp)
        if all(all(m in cache.get(seq, {}) for m in modes)
               for seq in seq_dirs(dataset, "train")):
            return cache
    cache = {}
    for seq in seq_dirs(dataset, "train"):
        cache[seq] = {}
        score, fps, n = score_seq(seq)
        for mode in modes:
            clips = clips_for_score(score, fps, n, mode)
            cache[seq][mode] = {"clips": clips, "fps": fps, "n_frames": n}
            stored = sum(b - a for a, b in clips) / (n / fps)
            print(f"  {seq} {mode}: {len(clips)} clips stored={stored:.3f}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fp:
        json.dump(cache, fp)
    return cache


def clips_to_frames(clips, fps, n):
    keep = np.zeros(n, dtype=bool)
    for a, b in clips:
        lo, hi = int(a * fps), min(int(b * fps) + 1, n)
        keep[lo:hi] = True
    return keep


def uniform_frames(budget_frac, fps, n):
    duration = n / fps
    clips = uniform_baseline(duration, budget_frac)
    return clips_to_frames(clips, fps, n)


def oracle_frames(budget_frac, labels):
    """Budget spent only on ground-truth event frames, evenly spread."""
    n = len(labels)
    k = int(budget_frac * n)
    if k <= 0:
        return np.zeros(n, dtype=bool)
    pos = np.where(labels > 0)[0]
    if len(pos) <= k:
        keep = np.zeros(n, dtype=bool)
        keep[pos] = True
        return keep
    step = len(pos) / k
    keep = np.zeros(n, dtype=bool)
    keep[pos[np.floor(np.arange(k) * step).astype(int)]] = True
    return keep


def eval_head(head, val):
    aps, recalls = [], []
    for feats, labels in val:
        scores = np.abs(head.scores(feats))
        aps.append(average_precision(labels, scores))
        k = max(int(0.1 * len(labels)), 1)
        keep = np.argsort(-scores)[:k]
        idx = np.where(labels > 0)[0]
        groups = (np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                  if len(idx) else [])
        recalls.append(sum(1 for g in groups
                           if len(np.intersect1d(g, keep)) > 0)
                       / max(len(groups), 1))
    return float(np.mean(aps)), float(np.mean(recalls))


def regime_check(dataset, fixed_mode):
    """The 1x -> 3x motion regime shift: a fixed score threshold calibrated
    at 1x must blow up at 3x; the adaptive trigger should hold."""
    rows = []
    for seq in seq_dirs(dataset, "val")[:10]:
        score, fps, n = score_seq(seq)
        for mode in ("adaptive", fixed_mode):
            clips = clips_for_score(score, fps, n, mode)
            rows.append((mode, sum(b - a for a, b in clips) / (n / fps)))
    out = {}
    for mode, frac in rows:
        out.setdefault(mode, []).append(frac)
    print("\nregime check (val split, 3x motion):")
    for mode, fracs in out.items():
        print(f"  {mode:<10} mean stored fraction: {np.mean(fracs):.3f}")
    return {m: float(np.mean(f)) for m, f in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    ap.add_argument("--modes", default="adaptive,T300",
                    help="clip triggering modes: adaptive and/or fixed "
                         "thresholds like T300 (comma-separated)")
    ap.add_argument("--uniform-mults", default="0.5,1,2,4",
                    help="uniform budgets as multiples of the cerebellum budget")
    args = ap.parse_args()
    modes = args.modes.split(",")
    mults = [float(m) for m in args.uniform_mults.split(",")]

    print("computing featmap-cancellation clips on train split (cached) ...")
    cache = load_or_compute_clips(args.dataset, modes)

    train = []
    for seq in seq_dirs(args.dataset, "train"):
        feats = np.load(f"{seq}/features.npy").astype(np.float32)
        labels = np.load(f"{seq}/labels.npy")
        fps = cache[seq][modes[0]]["fps"]
        seq_masks = {}
        for mode in modes:
            info = cache[seq][mode]
            n = min(info["n_frames"], len(feats), len(labels))
            seq_masks[mode] = clips_to_frames([tuple(c) for c in info["clips"]],
                                              fps, n)
        train.append((seq, feats, labels, fps, seq_masks))

    val = []
    for seq in seq_dirs(args.dataset, "val"):
        feats = np.load(f"{seq}/features.npy").astype(np.float32)
        labels = np.load(f"{seq}/labels.npy")
        val.append((feats, labels))

    results = {}
    print(f"\nval: {len(val)} sequences; training LogisticHead per method\n")
    for mode in modes:
        label = "cerebellum" if mode == "adaptive" else f"cerebellum({mode})"
        specs = [(label, "cb", 1.0)]
        specs += [(f"uniform({m:g}x)", "uniform", m) for m in mults]
        specs += [(f"oracle({mode})", "oracle", 1.0)]
        print(f"{'method':<20}{'stored':>8}{'frames':>8}{'val AP':>9}{'evRec@10%':>11}")
        for name, kind, mult in specs:
            Xs, ys, fracs = [], [], []
            for seq, feats, labels, fps, seq_masks in train:
                n = min(len(feats), len(labels))
                if kind == "cb":
                    mask = seq_masks[mode]
                elif kind == "oracle":
                    mask = oracle_frames(seq_masks[mode].mean(), labels[:n])
                else:
                    mask = uniform_frames(mult * seq_masks[mode].mean(), fps, n)
                Xs.append(feats[:n][mask])
                ys.append(labels[:n][mask])
                fracs.append(mask.mean())
            X = np.vstack(Xs)
            y = np.concatenate(ys)
            head = LogisticHead(X.shape[1]).fit(X, y)
            ap_mean, rec = eval_head(head, val)
            results[name] = {"stored_frac": float(np.mean(fracs)),
                             "n_frames": int(len(X)),
                             "val_ap": ap_mean, "ev_rec_at_10": rec}
            print(f"{name:<20}{np.mean(fracs):>8.3f}{len(X):>8}"
                  f"{ap_mean:>9.3f}{rec:>11.2f}")
        print()

    results["regime_check"] = regime_check(args.dataset, "T300")

    out_path = "runs/downstream_results.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()

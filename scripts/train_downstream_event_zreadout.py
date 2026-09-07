"""Readout upgrade test (GUIDE roadmap): does the SPATIAL pattern of the
residual z-map carry event information that the pooled amplitude does not?

§2.3.7 follow-up: frame residual / surprise are ~equal inside and outside
annotated events, so the trigger signal carries little event information.
But that was the pooled readout. This script trains the same logistic-head
protocol on spatial readouts of the z-map and asks:

  zamp      — 1-d: global mean of channel-summed z^2 (pooled amplitude,
              sanity control: should reproduce the "no information" result)
  zpyr      — 21-d: log1p spatial pyramid of the z^2 map (1 global + 2x2
              + 4x4 cell means) — the minimal "where is the residual" readout
  feat      — 576-d backbone pooled features (the §2.3.7 baseline)
  feat+zpyr — concat: does z-map spatial info complement backbone features?

Methods (storage selection, label-free): cerebellum clips vs uniform(1x,
equal budget) vs oracle(event frames, less storage). LOO x seeds protocol
identical to train_downstream_event.py.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train_downstream_event_zreadout.py \
      data/aoe_event --seeds $(seq 1 25)
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.adapter import LogisticHead
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               rollout)
from cerebellum_cancel.io import load_imu_csv
from train_downstream import clips_to_frames, eval_head, oracle_frames, uniform_frames
from train_downstream_aoe import seq_dirs
from train_downstream_event import load_or_compute_clips

READOUTS = ("zamp", "zpyr", "feat", "feat+zpyr")


def zmap_pyramid(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (zamp, zpyr) from z maps (T,C,H,W).

    Energy map e = sum_c z^2; pyramid = log1p of [global, 2x2, 4x4 means].
    """
    e = (z ** 2).sum(axis=1)                     # (T,H,W)
    T, H, W = e.shape
    g = e.mean(axis=(1, 2), keepdims=True)       # (T,1,1)
    p2 = e.reshape(T, 2, H // 2, 2, W // 2).mean(axis=(2, 4)).reshape(T, 4)
    p4 = e.reshape(T, 4, H // 4, 4, W // 4).mean(axis=(2, 4)).reshape(T, 16)
    amp = np.log1p(g[:, 0, 0])[:, None]
    pyr = np.log1p(np.concatenate([g.reshape(T, 1), p2, p4], axis=1))
    return amp.astype(np.float32), pyr.astype(np.float32)


def compute_z_features(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """Rollout the robust cancel on stored featmaps and return (zamp, zpyr)."""
    meta = json.load(open(f"{seq}/events.json"))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    cfg = CancelConfig(fps=float(meta.get("fps", 30.0)), robust=True)
    if "f_map" in meta:
        cfg.f_map = float(meta["f_map"])
    canc = FeatureMapCancel(cfg)
    z, _ = rollout(canc, fm[:n], gyro[:n])
    return zmap_pyramid(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe_event")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    seqs = seq_dirs(args.dataset)
    print(f"z-readout LOO x {len(args.seeds)} seeds over {len(seqs)} segments")
    print("cerebellum clips (cached) ...")
    cache = load_or_compute_clips(args.dataset)
    zfeat = {}
    for s in seqs:
        amp, pyr = compute_z_features(s)
        zfeat[s] = {"zamp": amp, "zpyr": pyr}
        print(f"  {os.path.basename(s)}: zpyr {pyr.shape}", flush=True)
    feats = {s: np.load(f"{s}/features.npy").astype(np.float32) for s in seqs}
    labels = {s: np.load(f"{s}/labels.npy").astype(np.float32) for s in seqs}
    metas = {s: json.load(open(f"{s}/events.json")) for s in seqs}
    masks = {}
    for s in seqs:
        info = cache[s]["adaptive"]
        n = min(info["n_frames"], len(feats[s]), len(labels[s]))
        masks[s] = clips_to_frames([tuple(c) for c in info["clips"]],
                                   info["fps"], n)

    def readout_vec(s: str, n: int, ro: str) -> np.ndarray:
        if ro == "zamp":
            return zfeat[s]["zamp"][:n]
        if ro == "zpyr":
            return zfeat[s]["zpyr"][:n]
        if ro == "feat":
            return feats[s][:n]
        return np.concatenate([feats[s][:n], zfeat[s]["zpyr"][:n]], axis=1)

    methods = [("cerebellum", "cb"), ("uniform(1x)", "uniform"),
               ("oracle", "oracle")]
    runs = {(m, ro): [] for m, _ in methods for ro in READOUTS}
    stored = {m: [] for m, _ in methods}
    for i, val_seq in enumerate(seqs):
        train_seqs = [s for s in seqs if s != val_seq]
        for seed in args.seeds:
            for name, kind in methods:
                Xs = {ro: [] for ro in READOUTS}
                ys, fr = [], []
                for s in train_seqs:
                    n = min(len(feats[s]), len(labels[s]))
                    if kind == "cb":
                        mask = masks[s]
                    elif kind == "oracle":
                        mask = oracle_frames(masks[s].mean(), labels[s][:n])
                    else:
                        mask = uniform_frames(masks[s].mean(),
                                              float(metas[s]["fps"]), n)
                    for ro in READOUTS:
                        Xs[ro].append(readout_vec(s, n, ro)[mask])
                    ys.append(labels[s][:n][mask])
                    fr.append(mask.mean())
                y = np.concatenate(ys)
                for ro in READOUTS:
                    X = np.vstack(Xs[ro])
                    head = LogisticHead(X.shape[1], seed=seed).fit(X, y)
                    ap_v, rec = eval_head(
                        head, [readout_vec(val_seq,
                                           min(len(feats[val_seq]),
                                               len(labels[val_seq])), ro)],
                        [labels[val_seq]])
                    runs[(name, ro)].append((ap_v, rec))
                stored[name].append(float(np.mean(fr)))
        print(f"fold {i + 1}/{len(seqs)} done (val={os.path.basename(val_seq)})",
              flush=True)

    print(f"\n{'method':<14}{'readout':<10}{'stored':>8}{'val AP':>18}"
          f"{'evRec@10%':>14}")
    out = {"protocol": f"LOO x {len(args.seeds)} seeds", "n_seg": len(seqs)}
    for name, _ in methods:
        for ro in READOUTS:
            a = np.array(runs[(name, ro)])
            print(f"{name:<14}{ro:<10}{np.mean(stored[name]):>8.3f}"
                  f"{a[:, 0].mean():>10.3f}±{a[:, 0].std():<6.3f}"
                  f"{a[:, 1].mean():>9.2f}±{a[:, 1].std():<4.2f}")
            out[f"{name}|{ro}"] = {
                "stored_frac": float(np.mean(stored[name])),
                "val_ap_mean": float(a[:, 0].mean()),
                "val_ap_std": float(a[:, 0].std()),
                "evrec_mean": float(a[:, 1].mean()),
                "evrec_std": float(a[:, 1].std()),
                "runs": a.tolist()}
    out_path = "runs/downstream_event_zreadout.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()

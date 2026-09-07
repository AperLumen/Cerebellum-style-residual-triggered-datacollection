"""Real-data downstream value experiment: on Open-AoE egocentric video with
atomic-action annotations, do cerebellum-triggered clips train a downstream
frame-level verb classifier better per stored byte than uniform sampling?

This is the real-data rerun of the synthetic headline experiment
(scripts/train_downstream.py, README "Downstream training comparison"),
with one task-level change forced by the data: the binary
"active-manipulation" mask is degenerate here (8 of 9 segments are
annotated active for ~100% of frames), so the task is frame-level verb
classification over the K target verbs from scripts/make_verb_labels.py
(7 verbs, each >= 300 annotated frames in both split halves), one-vs-rest
logistic heads on frozen 576-d pooled features, mAP + interval recall
at a top-10% frame budget.

  - clip source: feature-map cancellation with ROBUST stats (the gated EMA
    freezes on real video — measured; robust stats restore triggering)
  - split: the 9 prepared desk segments are all under val/; the first
    --n-train segments train, the next --n-val evaluate (segments were
    selected for scene diversity, so both halves mix scenes)
  - baselines: uniform 1s clips at k x the cerebellum budget; oracle spends
    the same budget only on verb-annotated frames

Usage: PYTHONPATH=. .venv/bin/python scripts/train_downstream_aoe.py data/aoe
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
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               maxcell_score, rollout)
from cerebellum_cancel.io import load_imu_csv
from train_downstream import (clips_for_score, clips_to_frames,
                              uniform_frames)

CACHE = "runs/downstream_aoe_clips.json"


def seq_dirs(dataset):
    return sorted(glob.glob(f"{dataset}/val/seq_*"))


def score_seq(seq):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    cfg = CancelConfig(fps=fps, robust=True)
    if "f_map" in meta:
        cfg.f_map = float(meta["f_map"])
    canc = FeatureMapCancel(cfg)
    z, _ = rollout(canc, fm[:n], gyro[:n])
    return maxcell_score(z), fps, n


def load_or_compute_clips(dataset):
    seqs = seq_dirs(dataset)
    if os.path.exists(CACHE):
        with open(CACHE) as fp:
            cache = json.load(fp)
        if all("adaptive" in cache.get(s, {}) for s in seqs):
            return cache
    cache = {}
    for seq in seqs:
        score, fps, n = score_seq(seq)
        clips = clips_for_score(score, fps, n, "adaptive")
        cache[seq] = {"adaptive": {"clips": clips, "fps": fps,
                                   "n_frames": n}}
        stored = sum(b - a for a, b in clips) / (n / fps)
        print(f"  {seq}: {len(clips)} clips stored={stored:.3f}", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fp:
        json.dump(cache, fp)
    return cache


def oracle_frames(budget_frac, any_verb):
    """Budget spent only on verb-annotated frames, evenly spread."""
    n = len(any_verb)
    k = int(budget_frac * n)
    if k <= 0:
        return np.zeros(n, dtype=bool)
    pos = np.where(any_verb > 0)[0]
    if len(pos) <= k:
        keep = np.zeros(n, dtype=bool)
        keep[pos] = True
        return keep
    step = len(pos) / k
    keep = np.zeros(n, dtype=bool)
    keep[pos[np.floor(np.arange(k) * step).astype(int)]] = True
    return keep


def interval_recall_at_budget(y, scores, frac=0.1):
    """Fraction of contiguous y-active intervals containing at least one of
    the top-`frac` scored frames."""
    n = len(y)
    k = max(int(frac * n), 1)
    keep = np.argsort(-scores)[:k]
    idx = np.where(y > 0)[0]
    groups = (np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
              if len(idx) else [])
    return sum(1 for g in groups if len(np.intersect1d(g, keep)) > 0) \
        / max(len(groups), 1)


def eval_heads(heads, val_feats, val_labels):
    """Per-class AP + interval recall@10%, macro-averaged over classes."""
    aps, recs = [], []
    for c in range(len(heads)):
        ap_c, rec_c = [], []
        for feats, labels in zip(val_feats, val_labels):
            y = labels[:, c]
            if y.sum() == 0:
                continue
            s = heads[c].scores(feats)
            ap_c.append(average_precision(y, s))
            rec_c.append(interval_recall_at_budget(y, s))
        aps.append(float(np.mean(ap_c)))
        recs.append(float(np.mean(rec_c)))
    return float(np.mean(aps)), float(np.mean(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe")
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--uniform-mults", default="0.5,1,2,4")
    args = ap.parse_args()
    mults = [float(m) for m in args.uniform_mults.split(",")]
    verbs = json.load(open(f"{args.dataset}/verbs.json"))["verbs"]
    K = len(verbs)

    seqs = seq_dirs(args.dataset)
    train_seqs = seqs[:args.n_train]
    val_seqs = seqs[args.n_train:args.n_train + args.n_val]
    print(f"verbs: {verbs}")
    print(f"train: {[os.path.basename(s) for s in train_seqs]}")
    print(f"val:   {[os.path.basename(s) for s in val_seqs]}")

    print("computing cerebellum clips (robust stats, cached) ...")
    cache = load_or_compute_clips(args.dataset)

    train = []
    for seq in train_seqs:
        feats = np.load(f"{seq}/features.npy").astype(np.float32)
        labels = np.load(f"{seq}/verb_labels.npy").astype(np.float32)
        info = cache[seq]["adaptive"]
        n = min(info["n_frames"], len(feats), len(labels))
        mask = clips_to_frames([tuple(c) for c in info["clips"]],
                               info["fps"], n)
        train.append((seq, feats, labels, info["fps"], mask))
    val = [(np.load(f"{s}/features.npy").astype(np.float32),
            np.load(f"{s}/verb_labels.npy").astype(np.float32))
           for s in val_seqs]
    val_feats = [v[0] for v in val]
    val_labels = [v[1] for v in val]

    specs = [("cerebellum", "cb", 1.0)]
    specs += [(f"uniform({m:g}x)", "uniform", m) for m in mults]
    specs += [("oracle", "oracle", 1.0)]
    print(f"\n{'method':<20}{'stored':>8}{'frames':>8}{'val mAP':>9}"
          f"{'ivRec@10%':>11}")
    results = {}
    for name, kind, mult in specs:
        Xs, Ys, fracs = [], [], []
        for seq, feats, labels, fps, cb_mask in train:
            n = min(len(feats), len(labels))
            if kind == "cb":
                mask = cb_mask
            elif kind == "oracle":
                mask = oracle_frames(cb_mask.mean(), labels[:n].sum(1))
            else:
                mask = uniform_frames(mult * cb_mask.mean(), fps, n)
            Xs.append(feats[:n][mask])
            Ys.append(labels[:n][mask])
            fracs.append(mask.mean())
        X = np.vstack(Xs)
        Y = np.concatenate(Ys)
        heads = [LogisticHead(X.shape[1], seed=c + 1).fit(X, Y[:, c])
                 for c in range(K)]
        map_mean, rec = eval_heads(heads, val_feats, val_labels)
        results[name] = {"stored_frac": float(np.mean(fracs)),
                         "n_frames": int(len(X)),
                         "val_map": map_mean, "iv_rec_at_10": rec}
        print(f"{name:<20}{np.mean(fracs):>8.3f}{len(X):>8}"
              f"{map_mean:>9.3f}{rec:>11.2f}")

    out_path = "runs/downstream_aoe_results.json"
    with open(out_path, "w") as fp:
        json.dump({"verbs": verbs, **results}, fp, indent=2)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()

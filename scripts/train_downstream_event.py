"""Sparse physical-event downstream test: the regime the trigger was
designed for. Open-AoE segments rich in event verbs (drop/throw/toss/break,
prepped with scripts/prep_aoe.py --event-verbs) give a binary frame-level
event task that is NOT degenerate (events are seconds-long and sparse).

Design mirrors §2.3.2 of GUIDE.md: do cerebellum-triggered clips train a
better event detector per stored byte than uniform / oracle? Small pool
(5 segments), so the resample protocol is leave-one-out x seeds: each run
trains on 4 segments, evaluates on the held-out one; 5 seeds per fold.
Clips: cached robust-stats cerebellum clips (selection is label-free).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train_downstream_event.py data/aoe_event
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.adapter import LogisticHead
from train_downstream import clips_to_frames, eval_head, oracle_frames, uniform_frames
from train_downstream_aoe import score_seq, seq_dirs

CACHE = "runs/downstream_event_clips.json"


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
        from train_downstream import clips_for_score
        clips = clips_for_score(score, fps, n, "adaptive")
        cache[seq] = {"adaptive": {"clips": clips, "fps": fps,
                                   "n_frames": n}}
        stored = sum(b - a for a, b in clips) / (n / fps)
        print(f"  {seq}: {len(clips)} clips stored={stored:.3f}", flush=True)
    with open(CACHE, "w") as fp:
        json.dump(cache, fp)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe_event")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--uniform-mults", default="0.5,1,2,4")
    # 只取前 N 段——用于复现**历史子集**（如 2026-09-07 的 5 段版表）。
    # 加这个旗标是为了让"老表中也补全量基线"可复现，而不是手工改代码
    # 或让老表带着已知缺口留着。默认 0 = 全段。
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N segments (0 = all)")
    args = ap.parse_args()
    mults = [float(m) for m in args.uniform_mults.split(",")]

    seqs = seq_dirs(args.dataset)
    if args.limit:
        seqs = seqs[:args.limit]
    print(f"leave-one-out x {len(args.seeds)} seeds over {len(seqs)} "
          f"event segments")
    print("cerebellum clips (cached) ...")
    cache = load_or_compute_clips(args.dataset)
    feats = {s: np.load(f"{s}/features.npy").astype(np.float32) for s in seqs}
    labels = {s: np.load(f"{s}/labels.npy").astype(np.float32) for s in seqs}
    metas = {s: json.load(open(f"{s}/events.json")) for s in seqs}
    masks = {}
    for s in seqs:
        info = cache[s]["adaptive"]
        n = min(info["n_frames"], len(feats[s]), len(labels[s]))
        masks[s] = clips_to_frames([tuple(c) for c in info["clips"]],
                                   info["fps"], n)

    specs = [("cerebellum", "cb", 1.0)]
    specs += [(f"uniform({m:g}x)", "uniform", m) for m in mults]
    specs += [("oracle", "oracle", 1.0)]
    # **全量基线**（2026-09-13 加，与 train_downstream.py / run_downstream_aoe_cv.py
    # 一致）：判据是"筛选有没有价值"，那要求这张表有分辨力。若"存 100%"不比
    # 最低预算好，表里的 1σ 重合就不能当证据。§2.3.6 漏了这行、结论因此被误读。
    specs += [("all(100%)", "all", 1.0)]

    runs = {name: [] for name, _, _ in specs}
    stored = {name: [] for name, _, _ in specs}
    for i, val_seq in enumerate(seqs):
        train_seqs = [s for s in seqs if s != val_seq]
        for seed in args.seeds:
            for name, kind, mult in specs:
                Xs, ys, fr = [], [], []
                for s in train_seqs:
                    n = min(len(feats[s]), len(labels[s]))
                    if kind == "cb":
                        mask = masks[s]
                    elif kind == "all":
                        mask = np.ones(n, dtype=bool)
                    elif kind == "oracle":
                        mask = oracle_frames(masks[s].mean(), labels[s][:n])
                    else:
                        mask = uniform_frames(mult * masks[s].mean(),
                                              float(metas[s]["fps"]), n)
                    Xs.append(feats[s][:n][mask])
                    ys.append(labels[s][:n][mask])
                    fr.append(mask.mean())
                X = np.vstack(Xs)
                y = np.concatenate(ys)
                head = LogisticHead(X.shape[1], seed=seed).fit(X, y)
                ap_v, rec = eval_head(head, [feats[val_seq]],
                                      [labels[val_seq]])
                runs[name].append((ap_v, rec))
                stored[name].append(float(np.mean(fr)))
        ev = float(labels[val_seq].mean())
        print(f"fold {i + 1}/{len(seqs)} done (val={os.path.basename(val_seq)}, "
              f"event frames {ev:.1%})", flush=True)

    print(f"\n{'method':<18}{'stored':>8}{'val AP':>18}{'evRec@10%':>14}")
    out = {"protocol": f"LOO x {len(args.seeds)} seeds", "n_seg": len(seqs)}
    for name, _, _ in specs:
        a = np.array(runs[name])
        print(f"{name:<18}{np.mean(stored[name]):>8.3f}"
              f"{a[:, 0].mean():>10.3f}±{a[:, 0].std():<6.3f}"
              f"{a[:, 1].mean():>9.2f}±{a[:, 1].std():<4.2f}")
        out[name] = {"stored_frac": float(np.mean(stored[name])),
                     "val_ap_mean": float(a[:, 0].mean()),
                     "val_ap_std": float(a[:, 0].std()),
                     "evrec_mean": float(a[:, 1].mean()),
                     "evrec_std": float(a[:, 1].std()),
                     "runs": a.tolist()}
    out_path = "runs/downstream_event_results.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()

"""Error-bar version of the real-data downstream value experiment
(scripts/train_downstream_aoe.py): 3-fold cross-segment x multi-seed.

The single-split table in §2.3.6 of GUIDE.md has noise-scale mAP spread;
this rerun turns "the synthetic advantage does not transfer" from a
single-point observation into a mean +/- std conclusion. Folds rotate the
val triple over the 9 prepared desk segments; each fold trains 7 one-vs-rest
logistic heads per seed (init-seed only; full-batch GD is deterministic
given data and init). Clips are the cached robust-stats cerebellum clips
(same trigger for every fold/seed — selection does not depend on the
downstream labels).

Usage: PYTHONPATH=. .venv/bin/python scripts/run_downstream_aoe_cv.py data/aoe
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.adapter import LogisticHead, average_precision
from train_downstream import clips_to_frames, uniform_frames
from train_downstream_aoe import (interval_recall_at_budget,
                                  load_or_compute_clips, oracle_frames,
                                  seq_dirs)

CACHE = "runs/downstream_aoe_clips.json"


def eval_fold(heads, val_feats, val_labels):
    """Macro mAP / interval-recall@10% over the classes PRESENT in this
    fold's val split (a class with no annotated val frames is undefined,
    not zero — record which classes were evaluated)."""
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
        if ap_c:
            aps.append(float(np.mean(ap_c)))
            recs.append(float(np.mean(rec_c)))
    return float(np.mean(aps)), float(np.mean(recs)), len(aps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5])
    ap.add_argument("--uniform-mults", default="0.5,1,2,4")
    args = ap.parse_args()
    mults = [float(m) for m in args.uniform_mults.split(",")]
    verbs = json.load(open(f"{args.dataset}/verbs.json"))["verbs"]
    K = len(verbs)

    seqs = seq_dirs(args.dataset)
    assert len(seqs) % args.folds == 0, "folds must divide n_segments"
    k = len(seqs) // args.folds
    print(f"{args.folds} folds x {len(args.seeds)} seeds over "
          f"{len(seqs)} segments, {K} verbs")

    print("cerebellum clips (cached) ...")
    cache = load_or_compute_clips(args.dataset)

    feats = {s: np.load(f"{s}/features.npy").astype(np.float32)
             for s in seqs}
    labels = {s: np.load(f"{s}/verb_labels.npy").astype(np.float32)
              for s in seqs}
    masks = {}
    for s in seqs:
        info = cache[s]["adaptive"]
        n = min(info["n_frames"], len(feats[s]), len(labels[s]))
        masks[s] = clips_to_frames([tuple(c) for c in info["clips"]],
                                   info["fps"], n)

    specs = [("cerebellum", "cb", 1.0)]
    specs += [(f"uniform({m:g}x)", "uniform", m) for m in mults]
    specs += [("oracle", "oracle", 1.0)]

    runs = {name: [] for name, _, _ in specs}
    stored = {name: [] for name, _, _ in specs}
    for fold in range(args.folds):
        val_seqs = seqs[fold * k:(fold + 1) * k]
        train_seqs = [s for s in seqs if s not in val_seqs]
        val_feats = [feats[s] for s in val_seqs]
        val_labels = [labels[s] for s in val_seqs]
        n_cls = 0
        for seed in args.seeds:
            for name, kind, mult in specs:
                Xs, Ys, fr = [], [], []
                for s in train_seqs:
                    n = min(len(feats[s]), len(labels[s]))
                    if kind == "cb":
                        mask = masks[s]
                    elif kind == "oracle":
                        mask = oracle_frames(masks[s].mean(), labels[s][:n].sum(1))
                    else:
                        mask = uniform_frames(mult * masks[s].mean(),
                                              float(json.load(open(
                                                  f"{s}/events.json"))["fps"]),
                                              n)
                    Xs.append(feats[s][:n][mask])
                    Ys.append(labels[s][:n][mask])
                    fr.append(mask.mean())
                X = np.vstack(Xs)
                Y = np.concatenate(Ys)
                heads = [LogisticHead(X.shape[1], seed=seed * 100 + c + 1)
                         .fit(X, Y[:, c]) for c in range(K)]
                map_mean, rec, n_cls = eval_fold(heads, val_feats, val_labels)
                runs[name].append((map_mean, rec))
                stored[name].append(float(np.mean(fr)))
        print(f"fold {fold + 1}/{args.folds} done (val = "
              f"{[os.path.basename(s) for s in val_seqs]}, "
              f"{n_cls}/{K} classes present)", flush=True)

    print(f"\n{'method':<18}{'stored':>8}{'val mAP':>18}{'ivRec@10%':>14}")
    out = {"verbs": verbs, "folds": args.folds, "seeds": args.seeds}
    for name, _, _ in specs:
        a = np.array(runs[name])
        print(f"{name:<18}{np.mean(stored[name]):>8.3f}"
              f"{a[:, 0].mean():>10.3f}±{a[:, 0].std():<6.3f}"
              f"{a[:, 1].mean():>9.2f}±{a[:, 1].std():<4.2f}")
        out[name] = {"stored_frac": float(np.mean(stored[name])),
                     "val_map_mean": float(a[:, 0].mean()),
                     "val_map_std": float(a[:, 0].std()),
                     "ivrec_mean": float(a[:, 1].mean()),
                     "ivrec_std": float(a[:, 1].std()),
                     "runs": a.tolist()}
    out_path = "runs/downstream_aoe_cv.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()

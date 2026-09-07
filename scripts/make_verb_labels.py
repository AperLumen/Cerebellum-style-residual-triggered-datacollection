"""Build per-frame multi-hot verb labels for the prepared AoE sequences.

The binary "active manipulation" mask is degenerate on this corpus (8 of 9
segments are annotated active for ~100% of frames), so the downstream value
experiment uses frame-level verb classification instead: K target verbs,
chosen automatically with >= --min-frames in BOTH split halves (train =
first --n-train seqs, val = the rest), everything else background.

Writes verb_labels.npy (n, K) float32 per seq and verbs.json (class list +
selection stats) at the dataset root.

Usage: PYTHONPATH=. .venv/bin/python scripts/make_verb_labels.py data/aoe
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def seq_dirs(dataset):
    return sorted(glob.glob(f"{dataset}/val/seq_*"))


def verb_frame_counts(seq, source_root):
    """verb -> frame count over this seq's annotation intervals."""
    meta = json.load(open(f"{seq}/events.json"))
    ann = json.load(open(os.path.join(source_root, meta["source"],
                                      "ego_action_annotation.json")))
    c = Counter()
    for seg in ann:
        n_fr = int(float(seg["end_frame"])) - int(float(seg["start_frame"]))
        for a in (seg.get("atomic_action") or []):
            if a.get("verb"):
                c[a["verb"]] += max(n_fr, 0)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe")
    ap.add_argument("--source-root", default="data/aoe_src")
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--min-frames", type=int, default=300)
    args = ap.parse_args()

    seqs = seq_dirs(args.dataset)
    counts = [verb_frame_counts(s, args.source_root) for s in seqs]
    vocab = sorted(set().union(*counts))

    chosen = []
    for v in vocab:
        tr = sum(c.get(v, 0) for c in counts[:args.n_train])
        va = sum(c.get(v, 0) for c in counts[args.n_train:])
        if tr >= args.min_frames and va >= args.min_frames:
            chosen.append(v)
    print(f"verbs ({len(chosen)}): "
          + ", ".join(f"{v}(tr={sum(c.get(v,0) for c in counts[:args.n_train])}"
                      f"/val={sum(c.get(v,0) for c in counts[args.n_train:])})"
                      for v in chosen))

    for s in seqs:
        meta = json.load(open(f"{s}/events.json"))
        ann = json.load(open(os.path.join(args.source_root, meta["source"],
                                          "ego_action_annotation.json")))
        n = len(np.load(f"{s}/featmaps.npy", mmap_mode="r"))
        y = np.zeros((n, len(chosen)), dtype=np.float32)
        for seg in ann:
            lo = min(max(int(float(seg["start_frame"])), 0), n - 1)
            hi = min(max(int(float(seg["end_frame"])), 0), n)
            for a in (seg.get("atomic_action") or []):
                if a.get("verb") in chosen:
                    y[lo:hi, chosen.index(a["verb"])] = 1.0
        np.save(f"{s}/verb_labels.npy", y)
        print(f"{s}: {s and int(y.sum())} verb frames / {n}", flush=True)

    with open(f"{args.dataset}/verbs.json", "w") as f:
        json.dump({"verbs": chosen,
                   "split": {"train": args.n_train,
                             "val": len(seqs) - args.n_train},
                   "min_frames": args.min_frames}, f, indent=2)
    print(f"-> {args.dataset}/verbs.json")


if __name__ == "__main__":
    main()

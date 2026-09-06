"""Train and evaluate the cerebellum adapter against baselines on cached
backbone features.

Variants:
  A static      head on raw features f_t            (no prediction)
  B slow-only   head on residual from W_slow        (learned predictor)
  C cerebellum  head on residual from W_slow+W_fast (online gated fast weights)

Metrics: mean per-sequence average precision (frame-level event detection),
and event recall when only the top-10% highest-scoring frames are kept
(the data-collection setting).

Usage: .venv/bin/python scripts/train_adapter.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings

import numpy as np

# macOS Accelerate BLAS emits spurious FP warnings from matmul worker
# threads (numpy issue); the computed values are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.adapter import (AdapterConfig, CerebellumAdapter,
                                       GranuleExpander, LogisticHead,
                                       action_features, average_precision,
                                       fit_slow_weights, rollout)
from cerebellum_cancel.io import load_imu_csv


def load_split(dataset, split):
    seqs = sorted(glob.glob(f"{dataset}/{split}/seq_*"))
    out = []
    for s in seqs:
        feats = np.load(f"{s}/features.npy").astype(np.float32)
        labels = np.load(f"{s}/labels.npy")
        gyros = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                         dtype=np.float32)
        n = min(len(feats), len(gyros))
        out.append((feats[:n], labels[:n], gyros[:n]))
    return out


def collect_slow_training(split_data, granule):
    """Design matrix of (phi([f_{t-1}, a_t]) -> f_t) over all sequences."""
    phis, targets = [], []
    for feats, _, gyros in split_data:
        for t in range(1, len(feats)):
            phis.append(granule(feats[t - 1], action_features(gyros[t])))
            targets.append(feats[t])
    return np.stack(phis), np.stack(targets)


def residuals_for(split_data, make_adapter):
    """Roll out a fresh adapter per sequence; return list of (res, labels, surprises)."""
    out = []
    for feats, labels, gyros in split_data:
        adapter = make_adapter()
        res, sur = rollout(adapter, feats, gyros)
        out.append((res, labels, sur))
    return out


def evaluate(residuals, head):
    aps, recalls = [], []
    for res, labels, _ in residuals:
        scores = np.abs(head.scores(res))
        aps.append(average_precision(labels, scores))
        k = max(int(0.1 * len(labels)), 1)      # top-10% budget
        keep = np.argsort(-scores)[:k]
        # event-level recall: each contiguous event hit if any kept frame in it
        idx = np.where(labels > 0)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []
        ev_hit = sum(1 for g in groups if len(np.intersect1d(g, keep)) > 0)
        recalls.append(ev_hit / max(len(groups), 1))
    return float(np.mean(aps)), float(np.mean(recalls))


class EnergyHead:
    """Untrained surprise readout: pooled standardized-residual energy."""

    def scores(self, res):
        return (res ** 2).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    args = ap.parse_args()

    train = load_split(args.dataset, "train")
    val = load_split(args.dataset, "val")
    cfg = AdapterConfig()
    granule = GranuleExpander(576, 4, cfg)

    print("fitting slow weights (ridge over train set)...")
    phis, targets = collect_slow_training(train, granule)
    w_slow = fit_slow_weights(phis, targets, cfg.ridge)
    pred_err = np.mean((phis @ w_slow - targets) ** 2)
    var_tgt = np.var(targets)
    print(f"slow predictor R^2 = {1 - pred_err / var_tgt:.3f}")

    def make(use_fast):
        return lambda: CerebellumAdapter(cfg, granule, w_slow, use_fast=use_fast)

    variants = {
        "A_static":     (None,          False),
        "B_slow_only":  (make(False),   False),
        "C_cerebellum": (make(True),    False),
        "C_energy":     (make(True),    True),   # untrained readout
    }
    ks = [1, 4, 16, 40]
    print(f"\n{'variant':<15}" + "".join(f"{'AP@' + str(k):>8}" for k in ks)
          + f"{'evRec@10%':>12}")
    for name, (factory, untrained) in variants.items():
        if factory is None:
            train_res = [(f, l, None) for f, l, _ in train]
            val_res = [(f, l, None) for f, l, _ in val]
        else:
            train_res = residuals_for(train, factory)
            val_res = residuals_for(val, factory)
        if untrained:
            head = EnergyHead()
            ap_mean, rec = evaluate(val_res, head)
            row = f"{'--':>8}" * (len(ks) - 1)
            print(f"{name:<15}{row}{ap_mean:>8.3f}{rec:>12.2f}")
            continue
        cells = []
        for k in ks:
            head = LogisticHead(train_res[0][0].shape[1]).fit(
                np.vstack([r for r, _, _ in train_res[:k]]),
                np.concatenate([l for _, l, _ in train_res[:k]]))
            ap_mean, _ = evaluate(val_res, head)
            cells.append(ap_mean)
        _, rec = evaluate(val_res, head)        # k=40 head
        print(f"{name:<15}" + "".join(f"{a:>8.3f}" for a in cells)
              + f"{rec:>12.2f}")


if __name__ == "__main__":
    main()

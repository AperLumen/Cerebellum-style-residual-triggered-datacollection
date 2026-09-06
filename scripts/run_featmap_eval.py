"""Evaluate gyro-aligned feature-map cancellation on cached featmaps.

Everything is untrained: the only quantities that adapt are per-cell running
statistics (gated to calm frames). Compares the gyro-warped residual against
the naive frame-diff ablation (same stats machinery, no warp) and reports the
trained pooled-feature head from the adapter experiment as reference.

Usage: .venv/bin/python scripts/run_featmap_eval.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import warnings

import numpy as np

from cerebellum_cancel.adapter import average_precision
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                              energy_score, maxcell_score,
                                              rollout)
from cerebellum_cancel.io import load_imu_csv

warnings.filterwarnings("ignore", message=".*encountered in matmul")


def load_split(dataset, split):
    out = []
    for s in sorted(glob.glob(f"{dataset}/{split}/seq_*")):
        fmaps = np.load(f"{s}/featmaps.npy")
        labels = np.load(f"{s}/labels.npy")
        gyros = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                         dtype=np.float32)
        n = min(len(fmaps), len(gyros))
        out.append((fmaps[:n], labels[:n], gyros[:n]))
    return out


def ap_recall(scores_per_seq, labels_per_seq):
    aps, recalls = [], []
    for sc, lab in zip(scores_per_seq, labels_per_seq):
        aps.append(average_precision(lab, sc))
        k = max(int(0.1 * len(lab)), 1)
        keep = np.argsort(-sc)[:k]
        idx = np.where(lab > 0)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []
        recalls.append(sum(1 for g in groups
                           if len(np.intersect1d(g, keep)) > 0) / max(len(groups), 1))
    return float(np.mean(aps)), float(np.mean(recalls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    args = ap.parse_args()
    val = load_split(args.dataset, "val")

    # sanity: does the gyro warp actually cancel calm-frame motion?
    cfg = CancelConfig()
    F, lab, gy = val[0]
    _, sur_warp = rollout(FeatureMapCancel(cfg), F, gy)
    _, sur_naive = rollout(FeatureMapCancel(CancelConfig(use_warp=False)), F, gy)
    calm = lab == 0
    print(f"calm-frame surprise (val seq_000, 3x motion): "
          f"warp={np.median(sur_warp[calm]):.3f}  "
          f"naive={np.median(sur_naive[calm]):.3f}")

    print(f"\n{'method':<22}{'val AP':>9}{'evRec@10%':>12}")
    for name, use_warp in [("gyro_warp", True), ("naive_diff", False)]:
        cfg = CancelConfig(use_warp=use_warp)
        escores, mscores, labs = [], [], []
        for F, lab, gy in val:
            z, _ = rollout(FeatureMapCancel(cfg), F, gy)
            escores.append(energy_score(z))
            mscores.append(maxcell_score(z))
            labs.append(lab)
        for sc_name, sc in [("energy", escores), ("maxcell", mscores)]:
            ap_mean, rec = ap_recall(sc, labs)
            print(f"{name + '+' + sc_name:<22}{ap_mean:>9.3f}{rec:>12.2f}")


if __name__ == "__main__":
    main()

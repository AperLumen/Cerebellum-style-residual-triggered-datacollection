"""Reproduce the reported AUC 0.47 on the Open-AoE event segments with the
shipped pipeline configuration (robust=True), so the review's number and the
follow-up analysis are on the same footing.

Usage: PYTHONPATH=. .venv/bin/python scripts/check_auc_repro.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                              energy_score, maxcell_score, rollout)
from cerebellum_cancel.io import load_imu_csv
from scripts.analyze_negatives import auc


def main():
    print(f"{'seq':<18}{'ev%':>7}{'AUC_energy':>12}{'AUC_maxcell':>13}"
          f"{'AUC_sur':>10}{'AUC_gyro':>10}")
    acc = {"e": [], "m": [], "s": [], "g": []}
    for s in sorted(glob.glob("data/aoe_event/val/seq_*")):
        lab_path = f"{s}/labels.npy"
        if not os.path.exists(lab_path) or not os.path.exists(f"{s}/featmaps.npy"):
            continue
        lab = np.load(lab_path).astype(np.int8)
        F = np.load(f"{s}/featmaps.npy")
        gy = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                      dtype=np.float32)
        f_map = float(json.load(open(f"{s}/events.json"))["f_map"])
        n = min(len(F), len(gy), len(lab))
        F, gy, lab = F[:n], gy[:n], lab[:n]
        cfg = CancelConfig(f_map=f_map, robust=True)
        z, sur = rollout(FeatureMapCancel(cfg), F, gy)
        e = energy_score(z)
        m = maxcell_score(z)
        gm = np.linalg.norm(gy, axis=1)
        # skip the first 100 frames (statistics converging)
        sl = slice(100, n)
        a_e, a_m, a_s, a_g = (auc(lab[sl], e[sl]), auc(lab[sl], m[sl]),
                              auc(lab[sl], sur[sl]), auc(lab[sl], gm[sl]))
        if not np.isnan(a_e):
            acc["e"].append(a_e)
            acc["m"].append(a_m)
            acc["s"].append(a_s)
            acc["g"].append(a_g)
        print(f"{os.path.basename(s):<18}{lab[sl].mean()*100:>6.1f}%"
              f"{a_e:>12.3f}{a_m:>13.3f}{a_s:>10.3f}{a_g:>10.3f}")
    print(f"\n{'mean':<18}{'':>7}{np.mean(acc['e']):>12.3f}"
          f"{np.mean(acc['m']):>13.3f}{np.mean(acc['s']):>10.3f}"
          f"{np.mean(acc['g']):>10.3f}")
    print(f"\nsegments used: {len(acc['e'])}")


if __name__ == "__main__":
    main()

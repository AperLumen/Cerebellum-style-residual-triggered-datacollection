"""Translation-term evaluation: does the constant-depth translation warp
close the gap the 448px cross-check attributed to unmodeled translation?

Predictors on the same featmaps (robust stats):
  naive       — no warp
  gyro        — rotation-only warp (trans_scale = 0)
  gyro_trans  — rotation + constant-depth translation warp, with the
                per-segment 1/Z scale calibrated in prep_aoe.py against DIS
                flow (same status as the gyro sign calibration / f_map)
  dis_flow    — dense-flow ceiling (needs frames on disk)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/run_trans_eval.py data/aoe_motion \
      --split val --n-seq 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from run_flow_eval import (cancel_cfg, load_seq, raw_calm_residual,  # noqa: E402
                           seq_dirs)
from cerebellum_cancel.featmap_cancel import FeatureMapCancel, rollout  # noqa: E402
from cerebellum_cancel.flow_cancel import FlowWarpCancel, FlowWarpConfig, rollout_flow  # noqa: E402


def load_translation(seq: str) -> np.ndarray | None:
    """Camera-frame translation velocity (m/s) from the prep_aoe.py extra
    columns; None when the CSV predates translation export."""
    with open(f"{seq}/imu.csv", newline="") as f:
        rdr = csv.DictReader(f)
        if not {"translation.x", "translation.y"} <= set(rdr.fieldnames or []):
            return None
        return np.array([[float(r["translation.x"]), float(r["translation.y"]),
                          float(r["translation.z"])] for r in rdr],
                        dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset root, e.g. data/aoe_motion")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--robust", action="store_true", default=True)
    args = ap.parse_args()

    rows = {}
    for seq in seq_dirs(args.dataset, args.split)[:args.n_seq]:
        fm, frames, gyro, meta, fps, labels, n = load_seq(seq)
        trans = load_translation(seq)
        if trans is None:
            print(f"{seq}: no translation columns — re-run prep_aoe.py")
            continue
        out = {}
        cfg_naive = cancel_cfg(meta, args.robust)
        cfg_naive.use_warp = False
        canc = FeatureMapCancel(cfg_naive)
        z, _ = rollout(canc, fm, gyro)
        out["naive"] = raw_calm_residual(canc, z)

        cfg_rot = cancel_cfg(meta, args.robust)
        cfg_rot.trans_scale = 0.0
        canc = FeatureMapCancel(cfg_rot)
        z, _ = rollout(canc, fm, gyro, trans)
        out["gyro"] = raw_calm_residual(canc, z)

        cfg_tr = cancel_cfg(meta, args.robust)
        cfg_tr.trans_scale = float(meta.get("trans_scale", 0.0))
        canc = FeatureMapCancel(cfg_tr)
        z, _ = rollout(canc, fm, gyro, trans)
        out["gyro_trans"] = raw_calm_residual(canc, z)

        canc = FlowWarpCancel(FlowWarpConfig())
        z, _ = rollout_flow(canc, fm, frames)
        out["dis_flow"] = raw_calm_residual(canc, z)
        rows[seq] = out
        print(f"{seq} (1/Z={cfg_tr.trans_scale:.3f}): "
              + "  ".join(f"{k}={v:.4f}" for k, v in out.items()), flush=True)

    name = args.dataset.rstrip("/").replace("/", "_") + f"_{args.split}"
    with open(f"runs/trans_eval_{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    agg = {k: float(np.mean([r[k] for r in rows.values()]))
           for k in next(iter(rows.values()))}
    print("mean: " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    print(f"-> runs/trans_eval_{name}.json")


if __name__ == "__main__":
    main()

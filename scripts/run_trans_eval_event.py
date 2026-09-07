"""Event-segment translation evaluation: is the event-regime residual noise
egomotion (translation-fixable) or world motion (translation-agnostic)?

Predictors on the same featmaps (robust stats, no frames needed):
  naive       — no warp
  gyro        — rotation-only warp (trans_scale = 0)
  gyro_trans  — rotation + constant-depth translation warp with the
                per-segment 1/Z scale stored in events.json (calibrated in
                prep_aoe.py against DIS flow)

For each predictor we report, alongside the global calm residual, the mean
frame residual and surprise INSIDE vs OUTSIDE the annotated event intervals.
If gyro_trans shrinks the outside-event residual more than the inside one,
the residual noise is camera egomotion and translation modeling is the right
fix (roadmap item 1). If it shrinks both equally, the "noise" is world
motion (hands) and the §2.3.7 failure is not translation-fixable.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/run_trans_eval_event.py data/aoe_event
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any

import numpy as np

sys.path.insert(0, "scripts")
from run_flow_eval import cancel_cfg, raw_calm_residual, seq_dirs  # noqa: E402
from cerebellum_cancel.featmap_cancel import FeatureMapCancel, rollout  # noqa: E402


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


def event_mask(meta: dict, n: int, fps: float) -> np.ndarray:
    """True on frames inside an annotated event interval."""
    m = np.zeros(n, dtype=bool)
    for a, b in meta.get("events", []):
        m[int(float(a) * fps):min(int(float(b) * fps) + 1, n)] = True
    return m


def frame_residual(canc, z) -> np.ndarray:
    """Per-frame mean |err|, reconstructed from z and running sd."""
    sd = np.sqrt(canc.var) + 1e-3
    return np.abs(z * sd).mean(axis=(1, 2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset root, e.g. data/aoe_event")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    rows = {}
    for seq in seq_dirs(args.dataset, args.split):
        meta = json.load(open(f"{seq}/events.json"))
        fps = float(meta.get("fps", 30.0))
        fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
        gyro = np.array([[float(r["gyroscope.x"]), float(r["gyroscope.y"]),
                          float(r["gyroscope.z"])]
                         for r in csv.DictReader(open(f"{seq}/imu.csv"))],
                        dtype=np.float32)
        trans = load_translation(seq)
        if trans is None:
            print(f"{seq}: no translation columns — re-run prep_aoe.py")
            continue
        n = min(len(fm), len(gyro), len(trans))
        fm, gyro, trans = fm[:n], gyro[:n], trans[:n]
        inside = event_mask(meta, n, fps)

        specs = [("naive", False, 0.0),
                 ("gyro", True, 0.0),
                 ("gyro_trans", True, float(meta.get("trans_scale", 0.0)))]
        out: dict[str, Any] = {"trans_scale": specs[2][2],
               "event_frac": float(inside.mean()),
               "trans_speed_p95": float(np.linalg.norm(trans, axis=1)
                                        .max() if n == 0 else
                                        np.percentile(np.linalg.norm(trans, axis=1), 95))}
        for name, warp, ts in specs:
            cfg = cancel_cfg(meta, True)
            cfg.use_warp = warp
            cfg.trans_scale = ts
            canc = FeatureMapCancel(cfg)
            z, sur = rollout(canc, fm, gyro, trans if ts > 0 else None)
            res = frame_residual(canc, z)
            out[name] = {"calm": raw_calm_residual(canc, z),
                         "res_in": float(res[inside].mean()),
                         "res_out": float(res[~inside].mean()),
                         "sur_in": float(sur[inside].mean()),
                         "sur_out": float(sur[~inside].mean())}
        rows[seq] = out
        g = out["gyro"]; gt = out["gyro_trans"]
        print(f"{json.load(open(f'{seq}/events.json'))['source']} "
              f"(ev={out['event_frac']:.2f}, 1/Z={out['trans_scale']:.2f}, "
              f"|v|_p95={out['trans_speed_p95']:.3f} m/s): "
              f"gyro in/out {g['res_in']:.4f}/{g['res_out']:.4f} -> "
              f"gyro_trans in/out {gt['res_in']:.4f}/{gt['res_out']:.4f}",
              flush=True)

    name = args.dataset.rstrip("/").replace("/", "_") + f"_{args.split}"
    with open(f"runs/trans_eval_{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    keys = ["naive", "gyro", "gyro_trans"]
    for k in keys:
        agg = {m: float(np.mean([r[k][m] for r in rows.values()]))
               for m in ("calm", "res_in", "res_out", "sur_in", "sur_out")}
        print(f"mean {k:11s}: calm {agg['calm']:.4f}  "
              f"res in/out {agg['res_in']:.4f}/{agg['res_out']:.4f}  "
              f"sur in/out {agg['sur_in']:.2f}/{agg['sur_out']:.2f}")
    print(f"-> runs/trans_eval_{name}.json")


if __name__ == "__main__":
    main()

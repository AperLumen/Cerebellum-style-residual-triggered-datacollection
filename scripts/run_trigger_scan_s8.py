"""Full train-split trigger scan: DINOv1 ViT-S/8 @224 vs MobileNetV3.

Answers the roadmap question on ALL 40 train sequences (not just the
3-seq probe): does the native-28x28 ViT-S/8 backbone fire the adaptive
trigger at the published operating point at MobileNetV3's rate
(published: 11/40 sequences)?

For every train sequence, both backbones go through the IDENTICAL
instrumented battery used in scripts/diag_dino_trigger_s8.py:
  - S8:   DINOv1 ViT-S/8 @224 features extracted ONLINE per sequence
          (28x28 native grid, 8px/cell, f_map=24.25); NOTHING is written
          to disk — features are discarded after the rollout
  - MNV3: the published featmaps.npy (24,28,28), read from disk
Per-seq numbers: clips fired at the CURRENT gate (unchanged protocol),
stored fraction, event-vs-calm contrast (fraction of event frames above
calm-p99), z>6 frame count / consecutive-run structure / event
coincidence, max z_ema after the 30-frame warmup (onset bar is 6), and
cancel-level surprise mean / calm-gate block rate.

Output: one JSON (--out, default runs/trigger_scan_s8.json), rewritten
after EVERY sequence so an interrupted run keeps partial progress;
re-running overwrites only the entries it processes (resume-friendly).
stdout: one progress line per sequence.

Self-check (2 seqs only, do not continue past this):
  PYTHONPATH=. .venv/bin/python scripts/run_trigger_scan_s8.py \
      --seqs seq_000,seq_001 --out runs/trigger_scan_s8_selfcheck.json

Full 40-seq scan (single process, no parallel workers):
  PYTHONPATH=. .venv/bin/python scripts/run_trigger_scan_s8.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from cerebellum_cancel.io import load_imu_csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_dino_trigger_s8 import (extract_s8, load_s8, rollout_score,  # noqa: E402
                                  trig_z)
from train_downstream import seq_dirs  # noqa: E402

WARMUP = 30                   # trigger stat warmup (frames), same as probe
GRID = 28                     # native patch grid for both backbones


def battery(fm: np.ndarray, gyro: np.ndarray, labels: np.ndarray,
            fps: float) -> dict:
    """The probe's full diagnostic battery for one sequence's featmaps."""
    n = min(len(fm), len(gyro))
    fm, gyro = fm[:n], gyro[:n]
    score, sur, blocked = rollout_score(fm, gyro)
    ev = labels[:n] > 0
    calm_p99 = float(np.percentile(score[~ev], 99))
    z, clips = trig_z(score, fps, n)
    hi = np.where(z > 6)[0]
    runs = []
    for i in hi:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    zema, max_ema = 0.0, 0.0
    for t in range(1, n):
        zema += 0.5 * (z[t] - zema)
        if t > WARMUP:
            max_ema = max(max_ema, zema)
    zev = z[ev]
    stored = sum(b - a for a, b in clips) / (n / fps)
    return {
        "n_frames": int(n), "n_event_frames": int(ev.sum()),
        "clips": [[float(a), float(b)] for a, b in clips],
        "n_clips": len(clips), "stored_frac": float(stored),
        "score_mean": float(score.mean()), "score_sd": float(score.std()),
        "calm_p99": calm_p99,
        "ev_above_calm_p99": float((score[ev] > calm_p99).mean()),
        "z_gt6": int(len(hi)), "z_runs": [len(r) for r in runs],
        "z_event_hits": int(sum(1 for i in hi if ev[i])),
        "max_z_ema_postwarmup": float(max_ema),
        "event_z_median": float(np.median(zev)),
        "event_z_p90": float(np.percentile(zev, 90)),
        "event_z_max": float(zev.max()),
        "surprise_mean": float(sur.mean()),
        "gate_blocked_frac": float(blocked / n),
    }


def load_inputs(seq: str):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    labels = np.load(f"{seq}/labels.npy")
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    return fps, labels, gyro


def summarize(scan: dict) -> dict:
    out = {}
    for bb in ("s8", "mnv3"):
        rows = {s: v[bb] for s, v in scan["sequences"].items() if bb in v}
        fired = sorted(s for s, r in rows.items() if r["n_clips"] > 0)
        out[bb] = {
            "n_seqs": len(rows),
            "n_fire": len(fired),
            "fire_seqs": [s.split("/")[-1] for s in fired],
            "mean_stored_frac": float(np.mean([r["stored_frac"]
                                               for r in rows.values()])),
            "mean_ev_above_calm_p99": float(
                np.mean([r["ev_above_calm_p99"] for r in rows.values()])),
            "mean_surprise": float(np.mean([r["surprise_mean"]
                                            for r in rows.values()])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset", nargs="?")
    ap.add_argument("--seqs", default=None,
                    help="comma-separated seq names to process "
                         "(default: all train seqs)")
    ap.add_argument("--out", default="runs/trigger_scan_s8.json")
    args = ap.parse_args()

    all_seqs = seq_dirs(args.dataset, "train")
    if args.seqs:
        wanted = set(args.seqs.split(","))
        seqs = [s for s in all_seqs if s.split("/")[-1] in wanted]
    else:
        seqs = all_seqs
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tf = load_s8(device)
    print(f"S/8 online extraction @224 ({device}); MNV3 from featmaps.npy; "
          f"{len(seqs)} sequences -> {args.out}", flush=True)

    if os.path.exists(args.out):
        with open(args.out) as fp:
            scan = json.load(fp)
        scan.setdefault("sequences", {})
    else:
        scan = {"config": {
                    "backbones": ["s8", "mnv3"],
                    "gate": "adaptive 6/2.5 (protocol unchanged)",
                    "f_map": 24.25, "grid": GRID,
                    "note": "s8 = DINOv1 ViT-S/8 @224 substitute for "
                            "unobtainable dinov2_vits8 (see "
                            "diag_dino_trigger_s8.py)"},
                "sequences": {}}

    for seq in seqs:
        t0 = time.time()
        name = seq.split("/")[-1]
        fps, labels, gyro = load_inputs(seq)
        entry = {}
        fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
        entry["mnv3"] = battery(fm, gyro, labels, fps)
        frames = sorted(glob.glob(f"{seq}/frames/*.png"))
        fm = extract_s8(model, tf, frames, device)
        entry["s8"] = battery(fm, gyro, labels, fps)
        scan["sequences"][seq] = entry
        scan["summary"] = summarize(scan)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump(scan, fp, indent=1)
        m, s = entry["mnv3"], entry["s8"]
        print(f"{name} ({time.time() - t0:.0f}s) | "
              f"MNV3: clips={m['n_clips']} stored={m['stored_frac']:.3f} "
              f"ev99={m['ev_above_calm_p99']:.2f} z6={m['z_gt6']} | "
              f"S8: clips={s['n_clips']} stored={s['stored_frac']:.3f} "
              f"ev99={s['ev_above_calm_p99']:.2f} z6={s['z_gt6']} "
              f"ema={s['max_z_ema_postwarmup']:.1f}", flush=True)

    print("\n== verdict ==")
    for bb in ("mnv3", "s8"):
        sm = scan["summary"][bb]
        print(f"{bb:<6} fired {sm['n_fire']}/{sm['n_seqs']} seqs "
              f"{sm['fire_seqs']} mean_stored={sm['mean_stored_frac']:.3f} "
              f"mean_ev99={sm['mean_ev_above_calm_p99']:.2f}")
    mn, s8 = scan["summary"]["mnv3"], scan["summary"]["s8"]
    print(f"S/8 passes if n_fire(s8)={s8['n_fire']} >= "
          f"n_fire(mnv3)={mn['n_fire']} (published-cache reference: 11/40)")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()

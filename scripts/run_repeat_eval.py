"""Repeat-measurement protocol for the resolution-axis numbers.

The quality metrics are deterministic (frozen backbone, no sampling in
the cancellation path), so repeating them 10x would only re-print the
same numbers: they are re-run ONCE per input size and checked for exact
reproduction against the recorded sweep (runs/flow_eval_res{224,336,448}.json).
The latency bench is the noisy quantity (thermals, MPS scheduling), so it
is repeated --reps times per size and reported as mean +/- std and range.

Writes runs/resolution_bench10x.json (machine-readable) and
runs/resolution_report.md (human-readable log report).

Usage: PYTHONPATH=. .venv/bin/python scripts/run_repeat_eval.py --reps 10
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from run_flow_eval import (bench, cancel_cfg, load_seq, raw_calm_residual,
                           seq_dirs)  # noqa: E402
from cerebellum_cancel.featmap_cancel import FeatureMapCancel, rollout  # noqa: E402
from cerebellum_cancel.flow_cancel import FlowWarpCancel, FlowWarpConfig, rollout_flow  # noqa: E402

SIZES = [(224, "data/dataset"), (336, "data/dataset_336"),
         (448, "data/dataset_448")]
STAGE_KEYS = ["dis_flow_ms", "gyro_shift_ms", "pixel_core_ms",
              "backbone_fwd_ms", "flow_step_ms"]


def quality_once(dataset: str):
    """naive/gyro/flow calm residual over all seqs (same recipe as
    run_flow_eval's quality stage)."""
    agg = {}
    for split, n_seq in [("train", 40), ("val", 25)]:
        for seq in seq_dirs(dataset, split)[:n_seq]:
            fm, frames, gyro, meta, fps, labels, n = load_seq(seq)
            cfg_naive = cancel_cfg(meta)
            cfg_naive.use_warp = False
            for name, canc in [("naive", FeatureMapCancel(cfg_naive)),
                               ("gyro", FeatureMapCancel(cancel_cfg(meta))),
                               ("flow", FlowWarpCancel(FlowWarpConfig()))]:
                if name == "flow":
                    z, _ = rollout_flow(canc, fm, frames)
                else:
                    z, _ = rollout(canc, fm, gyro)
                agg.setdefault(split, {}).setdefault(name, []).append(
                    raw_calm_residual(canc, z))
    return {sp: {k: float(np.nanmean(v))
                 for k, v in names.items()}
            for sp, names in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    out = {"reps": args.reps, "sizes": {}}

    # --- latency: 10 reps of the full bench stage per size ---
    for size, dataset in SIZES:
        seq0 = (seq_dirs(dataset, "train") or seq_dirs(dataset, "val"))[0]
        rows = []
        for r in range(args.reps):
            rows.append(bench(seq0, size=size))
            print(f"[{size}px] bench rep {r + 1}/{args.reps} done", flush=True)
        stats = {}
        for k in STAGE_KEYS:
            v = [row[k] for row in rows]
            stats[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                        "min": float(np.min(v)), "max": float(np.max(v)),
                        "runs": v}
        out["sizes"][str(size)] = {"latency": stats}
        print(f"[{size}px] latency: " + "  ".join(
            f"{k}={stats[k]['mean']:.3f}+-{stats[k]['std']:.3f}"
            for k in STAGE_KEYS), flush=True)

    # --- quality: deterministic, re-run once and diff vs the sweep ---
    for size, dataset in SIZES:
        q = quality_once(dataset)
        ref_p = f"runs/flow_eval_res{size}.json"
        ref = json.load(open(ref_p))["quality"]
        diffs = {}
        ok = True
        for sp in ("train", "val"):
            for name in ("naive", "gyro", "flow"):
                d = abs(q[sp][name] - ref[sp][name]["calm_resid"])
                diffs[f"{sp}/{name}"] = d
                if d > 1e-9:
                    ok = False
        out["sizes"][str(size)]["quality_repro"] = {
            "values": q, "max_abs_diff_vs_sweep": diffs, "exact": ok}
        print(f"[{size}px] quality re-run: exact reproduction = {ok} "
              f"(max |diff| = {max(diffs.values()):.2e})", flush=True)

    with open("runs/resolution_bench10x.json", "w") as f:
        json.dump(out, f, indent=2)
    write_report(out, args.reps)
    print("-> runs/resolution_bench10x.json, runs/resolution_report.md")


def write_report(out: dict, reps: int) -> None:
    L = []
    L.append("# Resolution-axis repeat-measurement report\n")
    L.append(f"Protocol: quality metrics are deterministic (frozen backbone, "
             f"no sampling in the cancellation path) and were re-run once per "
             f"input size and checked against the recorded sweep; the latency "
             f"bench is the noisy quantity and was repeated {reps}x per size. "
             f"Machine: Apple Silicon MPS, MobileNetV3-small features[:3], "
             f"40 train + 25 val synthetic sequences per size.\n")
    L.append("## Quality (calm residual, mean |err|) — determinism check\n")
    L.append("| input | split | naive | gyro | flow | exact vs sweep | max abs diff |")
    L.append("|---|---|---|---|---|---|---|")
    for size, _ in SIZES:
        q = out["sizes"][str(size)]["quality_repro"]
        for sp in ("train", "val"):
            L.append(f"| {size} | {sp} | {q['values'][sp]['naive']:.4f} | "
                     f"{q['values'][sp]['gyro']:.4f} | "
                     f"{q['values'][sp]['flow']:.4f} | "
                     f"{'yes' if q['exact'] else 'NO'} | "
                     f"{max(q['max_abs_diff_vs_sweep'].values()):.1e} |")
    L.append("\n## Latency (ms/frame, per-stage median) — "
             f"{reps} repetitions\n")
    L.append("| input | stage | mean | std | min | max |")
    L.append("|---|---|---|---|---|---|")
    for size, _ in SIZES:
        lat = out["sizes"][str(size)]["latency"]
        for k in STAGE_KEYS:
            s = lat[k]
            L.append(f"| {size} | {k} | {s['mean']:.3f} | {s['std']:.3f} | "
                     f"{s['min']:.3f} | {s['max']:.3f} |")
    with open("runs/resolution_report.md", "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

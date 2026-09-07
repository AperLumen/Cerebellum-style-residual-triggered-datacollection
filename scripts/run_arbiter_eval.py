"""Gyro-triggered, flow-arbitrated cascade: cheap gyro compensation runs
always; DIS flow is consulted ONLY on trigger candidates, to veto the ones
it can explain (ego-motion misprediction: tan approximation, f_map
calibration error, scene drift). What flow cannot explain is confirmed
surprise. Amortized cost = call_rate x ~4 ms instead of 4 ms/frame.

Per sequence:
  1. gyro-warp rollout -> z trace + running sd (existing FeatureMapCancel)
  2. candidate runs: contiguous frames with z > onset threshold
  3. at each run's peak: err_flow = F - dis_warp(F_prev); veto if
     explained_ratio(err_gyro, err_flow) < explain_ratio
  4. vetoed runs are zeroed from the trace; AdaptiveTrigger clips are then
     compared against the un-arbitrated gyro clips

Usage: PYTHONPATH=. .venv/bin/python scripts/run_arbiter_eval.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_flow_eval import cancel_cfg, load_seq, clip_stats  # noqa: E402

from cerebellum_cancel.featmap_cancel import (FeatureMapCancel,
                                               expected_shift, maxcell_score,
                                               shift_map)
from cerebellum_cancel.flow_cancel import (FlowWarpConfig, dis_flow_warp,
                                           explained_ratio)

DIS_MS = 3.813      # measured in run_flow_eval.py bench (this machine)
ONSET_Z = 6.0       # match AdaptiveTrigger's default onset


def gyro_trace(fm, gyro, meta, fmap_scale=1.0, robust=False):
    """Gyro rollout; returns (z trace, sd trace after each step).
    fmap_scale != 1 simulates focal-length miscalibration (the structured
    ego-motion modeling error the arbiter exists to catch)."""
    cfg = cancel_cfg(meta, robust)
    cfg.f_map *= fmap_scale
    canc = FeatureMapCancel(cfg)
    zs, sds = [], []
    for t in range(len(fm)):
        out = canc.step(fm[t], gyro[t])
        zs.append(out["z"])
        sds.append(np.sqrt(canc.var) + 1e-3)
    return np.asarray(zs), np.asarray(sds)


def candidate_runs(z, onset=ONSET_Z, gap=2):
    """Contiguous runs of z > onset (gaps <= `gap` frames merged)."""
    hot = np.where(z > onset)[0]
    if len(hot) == 0:
        return []
    runs, start, prev = [], hot[0], hot[0]
    for i in hot[1:]:
        if i - prev > gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def arbitrate(fm, grays, gyro, meta, z, runs, explain_ratio):
    """Veto candidate runs whose gyro residual flow can explain.

    Composition per candidate peak frame t: the gyro shift warps BOTH the
    gray pair (pre-warp, so DIS only searches the small residual the gyro
    model mispredicted — without it DIS loses track at large rotations,
    exactly where triggers fire) and F_{t-1} (coarse prediction); DIS then
    refines the prediction along the residual flow. rho = interior energy
    ratio ||F - pred_flow||^2 / ||F - pred_gyro||^2; rho < explain_ratio
    means flow explained the residual (ego-motion misprediction) -> veto.
    Roll wrap-around from shift_map is excluded via a shift-aware border.
    Returns (modified z trace, n_flow_calls, vetoed list (a, b, rho))."""
    z2 = z.copy()
    fz = np.abs(z).max(axis=(1, 2, 3))
    n_calls, vetoed = 0, []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fcfg = FlowWarpConfig()
    cfg = cancel_cfg(meta)
    for a, b in runs:
        t = a + int(np.argmax(fz[a:b + 1]))
        if t < 1:
            continue
        g = cv2.cvtColor(grays[t - 1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (fcfg.flow_size[1], fcfg.flow_size[0]))
        g2 = cv2.cvtColor(grays[t], cv2.COLOR_BGR2GRAY)
        g2 = cv2.resize(g2, (fcfg.flow_size[1], fcfg.flow_size[0]))
        dx, dy = expected_shift(gyro[t], cfg)      # map px
        sc = g.shape[1] / fm.shape[2]              # gray px per map px
        # compose: gyro warp first (coarse, always-on), DIS refines only the
        # residual displacement on the pre-warped gray pair — applying the
        # residual flow to the RAW F_prev would drop the gyro shift entirely
        g_pre = shift_map(g[None].astype(np.float32), -dx * sc, -dy * sc)[0]
        g_pre = np.ascontiguousarray(g_pre, dtype=np.uint8)
        F_gyro = shift_map(fm[t - 1], -dx, -dy)
        pred = dis_flow_warp(dis, F_gyro, g_pre, g2)
        err_gyro = fm[t] - F_gyro
        err_flow = fm[t] - pred
        border = max(fcfg.border, 3 + int(np.ceil(max(abs(dx), abs(dy)))))
        rho = explained_ratio(err_gyro, err_flow, border=border)
        n_calls += 1
        if rho < explain_ratio:
            z2[a:b + 1] = 0.0
            vetoed.append((a, b, rho))
    return z2, n_calls, vetoed


def run_stats(runs, fps, n, events):
    """Fixed-threshold readout (monotone in the trace): stored fraction,
    per-event recall, and run precision. Complements AdaptiveTrigger's
    clip stats, whose running normalization is NOT monotone — zeroing a
    vetoed run lowers the running stats and can re-sensitize the trigger
    to later moderate scores."""
    keep = np.zeros(n, dtype=bool)
    for a, b in runs:
        keep[a:b + 1] = True
    hit_ev = 0
    for a, b in events:
        if keep[int(a * fps):min(int(b * fps) + 1, n)].any():
            hit_ev += 1
    hits = 0
    for a, b in runs:
        for e0, e1 in events:
            lo, hi = int(e0 * fps), min(int(e1 * fps) + 1, n)
            if b >= lo and a < hi:
                hits += 1
                break
    return {"runs": len(runs), "stored": float(keep.mean()),
            "event_recall": hit_ev / max(len(events), 1),
            "precision": hits / max(len(runs), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--explain-ratio", type=float, default=0.5)
    ap.add_argument("--fmap-scale", type=float, default=1.0,
                    help="simulate focal-length miscalibration in the gyro "
                         "warp (1.0 = calibrated)")
    ap.add_argument("--robust", action="store_true",
                    help="median/MAD always-update stats (for real video)")
    args = ap.parse_args()

    results = {}
    for split, n_seq in [("train", args.n_train), ("val", args.n_val)]:
        seqs = sorted(glob.glob(f"{args.dataset}/{split}/seq_*"))[:n_seq]
        if not seqs:
            continue
        agg = {"gyro_runs": [], "cascade_runs": [],
               "gyro_clips": [], "cascade_clips": []}
        n_calls_total = 0
        for seq in seqs:
            fm, frames, gyro, meta, fps, labels, n = load_seq(seq)
            events = [tuple(e) for e in meta["events"]]
            z, _ = gyro_trace(fm, gyro, meta, args.fmap_scale, args.robust)
            score = maxcell_score(z)          # published readout: sum_c z^2,
                                              # max over interior cells
            runs = candidate_runs(np.abs(z).max(axis=(1, 2, 3)))
            z2, n_calls, vetoed = arbitrate(fm, frames, gyro, meta, z,
                                            runs, args.explain_ratio)
            n_calls_total += n_calls
            vset = {(a, b) for a, b, _ in vetoed}
            surv = [(a, b) for a, b in runs if (a, b) not in vset]
            rs_gyro = run_stats(runs, fps, n, events)
            rs_casc = run_stats(surv, fps, n, events)
            st_gyro = clip_stats(score, fps, n, events)
            st_casc = clip_stats(maxcell_score(z2), fps, n, events)
            agg["gyro_runs"].append(rs_gyro)
            agg["cascade_runs"].append(rs_casc)
            agg["gyro_clips"].append(st_gyro)
            agg["cascade_clips"].append(st_casc)
            print(f"  {seq}: runs={len(runs)} vetoed={len(vetoed)} "
                  f"gyro_runs(stored={rs_gyro['stored']:.3f},"
                  f"evRec={rs_gyro['event_recall']:.2f},"
                  f"prec={rs_gyro['precision']:.2f}) "
                  f"cascade_runs(stored={rs_casc['stored']:.3f},"
                  f"evRec={rs_casc['event_recall']:.2f},"
                  f"prec={rs_casc['precision']:.2f})")
        for k in agg:
            rows = agg[k]
            results[f"{split}/{k}"] = {
                "stored": float(np.mean([r["stored"] for r in rows])),
                "event_recall": float(np.mean([r["event_recall"]
                                               for r in rows])),
            }
            if "precision" in rows[0]:
                results[f"{split}/{k}"]["precision"] = float(
                    np.mean([r["precision"] for r in rows]))
            if "runs" in rows[0]:
                results[f"{split}/{k}"]["runs"] = float(
                    np.mean([r["runs"] for r in rows]))
        call_rate = n_calls_total / max(len(seqs), 1)
        results[f"{split}/flow_calls_per_seq"] = float(call_rate)
        results[f"{split}/amortized_flow_ms"] = call_rate * DIS_MS / 600.0
        print(f"[{split}] gyro_runs: {results[f'{split}/gyro_runs']}  "
              f"cascade_runs: {results[f'{split}/cascade_runs']}  "
              f"gyro_clips: {results[f'{split}/gyro_clips']}  "
              f"cascade_clips: {results[f'{split}/cascade_clips']}  "
              f"flow calls/seq: {call_rate:.1f} "
              f"(amortized {results[f'{split}/amortized_flow_ms']:.3f} ms/frame)")

    out = "runs/arbiter_eval.json"
    with open(out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()

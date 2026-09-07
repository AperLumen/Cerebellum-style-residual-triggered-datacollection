"""Optical-flow baseline + cost profile: what does the CV-native way to do
gyro-warp cancellation cost, and what residual quality does it buy?

Quality: per train/val sequence, three predictors on the same featmaps —
naive (no warp), gyro warp (FeatureMapCancel), DIS dense flow warp
(FlowWarpCancel) — compared on calm-frame raw residual, and on
AdaptiveTrigger clip stats (stored fraction, event recall) for the two
warped variants.

Cost: median per-frame latency of every pipeline stage on this machine
(DIS flow, remap, gyro shift, pixel-pipeline core, frozen backbone forward,
full pixel pipeline), for the architecture "what does the optimization cost"
table in the README.

Usage: PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import cast

import cv2
import numpy as np

from cerebellum_cancel.core import AdaptiveTrigger
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               maxcell_score, rollout,
                                               shift_map)
from cerebellum_cancel.flow_cancel import (FlowWarpCancel, FlowWarpConfig,
                                           dis_calc, rollout_flow)
from cerebellum_cancel.io import load_imu_csv


def seq_dirs(dataset, split):
    return sorted(glob.glob(f"{dataset}/{split}/seq_*"))


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def load_seq(seq):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    n = min(len(fm), int(meta.get("seconds", 20) * fps))
    paths = sorted(glob.glob(f"{seq}/frames/*.png"))[:n]
    if not paths:
        raise IOError(f"{seq}/frames is empty — the prepared AoE frames are "
                      "not kept on disk (disk budget); re-run "
                      f"scripts/prep_aoe.py on the source segment first")
    frames = [read_bgr(p) for p in paths]
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(n, len(frames), len(gyro))
    labels = np.load(f"{seq}/labels.npy")[:n] \
        if os.path.exists(f"{seq}/labels.npy") else None
    return fm[:n], frames[:n], gyro[:n], meta, fps, labels, n


def cancel_cfg(meta: dict, robust: bool = False) -> "CancelConfig":
    """CancelConfig honoring a per-sequence f_map (real cameras: fx differs
    from the synthetic rig's default) and the robust-stats variant."""
    cfg = CancelConfig(fps=float(meta.get("fps", 30.0)),
                       robust=robust)
    if "f_map" in meta:
        cfg.f_map = float(meta["f_map"])
    return cfg


def raw_calm_residual(canc, z, thr=4.0):
    """Mean |err| over calm frames, reconstructed from z and running sd."""
    calm = []
    sd = np.sqrt(canc.var) + 1e-3
    for t in range(len(z)):
        s = float(np.abs(z[t]).max())
        if s < thr:
            calm.append(float(np.abs(z[t] * sd).mean()))
    return float(np.mean(calm)) if calm else float("nan")


def clip_stats(score, fps, n, events):
    trig = AdaptiveTrigger()
    for t in range(1, n):
        trig.update(t / fps, score[t], fps)
    clips = trig.finalize(n / fps)
    keep = np.zeros(n, dtype=bool)
    for a, b in clips:
        keep[int(a * fps):min(int(b * fps) + 1, n)] = True
    ev_time = np.zeros(n, dtype=bool)
    for a, b in events:
        ev_time[int(a * fps):min(int(b * fps) + 1, n)] = True
    hit = sum(1 for i in range(n) if ev_time[i] and keep[i])
    hit_ev = 0
    for a, b in events:
        if keep[int(a * fps):min(int(b * fps) + 1, n)].any():
            hit_ev += 1
    return {"stored": float(keep.mean()), "clips": len(clips),
            "event_recall": hit_ev / max(len(events), 1),
            "event_frame_cover": hit / max(int(ev_time.sum()), 1)}


def bench(seq, size: int = 224):
    """Median per-frame latency of each stage (ms), measured on one seq."""
    fm, frames, gyro, meta, fps, labels, n = load_seq(seq)
    rep = 3
    n_bench = min(300, n)

    # 1. DIS flow alone
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:n_bench]]
    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ts = []
    for _ in range(rep):
        for t in range(1, n_bench):
            t0 = time.perf_counter()
            dis_calc(dis, g[t - 1], g[t])
            ts.append((time.perf_counter() - t0) * 1e3)
    dis_ms = float(np.median(ts))

    # 2. gyro shift_map alone (feature-map size)
    F = fm[0]
    ts = []
    for _ in range(rep):
        for _ in range(n_bench):
            t0 = time.perf_counter()
            shift_map(F, 0.37, -0.21)
            ts.append((time.perf_counter() - t0) * 1e3)
    shift_ms = float(np.median(ts))

    # 3. pixel-pipeline core per frame (read+gyro warp+cancel.process)
    from cerebellum_cancel.core import CerebellumCancel
    from cerebellum_cancel.io import cells_of, compensate_rotation
    canc = CerebellumCancel()
    ts = []
    for _ in range(rep):
        prev = None
        for t in range(n_bench):
            gray = cv2.resize(cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY),
                              (320, 240))
            t0 = time.perf_counter()
            resid = compensate_rotation(prev, gray, gyro[t], 1 / fps)
            canc.process(cells_of(resid, canc.grid), gyro[t], 1 / fps)
            ts.append((time.perf_counter() - t0) * 1e3)
            prev = gray
    pixel_ms = float(np.median(ts))

    # 4. frozen backbone forward (features[:3], `size`x`size`, batch 1)
    import torch
    import torchvision
    import torchvision.transforms as T
    from PIL import Image
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.mobilenet_v3_small(
        weights="DEFAULT").features[:3].to(device).eval()
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    with torch.no_grad():
        t_img = cast(torch.Tensor, tf(Image.fromarray(
            cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))))
        x = torch.stack([t_img]).to(device)
        model(x)  # warmup
        ts = []
        for _ in range(rep):
            for t in range(n_bench):
                t_img = cast(torch.Tensor, tf(Image.fromarray(
                    cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB))))
                x = torch.stack([t_img]).to(device)
                t0 = time.perf_counter()
                model(x)
                if device == "mps":
                    torch.mps.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
    backbone_ms = float(np.median(ts))

    # 5. full flow-cancel step (flow + remap + stats)
    fc = FlowWarpCancel(FlowWarpConfig())
    ts = []
    for _ in range(rep):
        fc.reset_online()
        for t in range(1, n_bench):
            t0 = time.perf_counter()
            fc.step(fm[t], frames[t])
            ts.append((time.perf_counter() - t0) * 1e3)
    flow_step_ms = float(np.median(ts))

    return {"dis_flow_ms": dis_ms, "gyro_shift_ms": shift_ms,
            "pixel_core_ms": pixel_ms, "backbone_fwd_ms": backbone_ms,
            "flow_step_ms": flow_step_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--robust", action="store_true",
                    help="median/MAD always-update stats (for real video)")
    ap.add_argument("--size", type=int, default=224,
                    help="backbone input edge the featmaps were extracted "
                         "with (bench stage only; dataset default is 224)")
    args = ap.parse_args()

    results = {"quality": {"train": {}, "val": {}}, "clips": {}}
    for split, n_seq in [("train", args.n_train), ("val", args.n_val)]:
        split_seqs = seq_dirs(args.dataset, split)[:n_seq]
        if not split_seqs:
            continue
        agg = {}
        for seq in split_seqs:
            fm, frames, gyro, meta, fps, labels, n = load_seq(seq)
            events = [tuple(e) for e in meta["events"]]
            rows = {}
            cfg_naive = cancel_cfg(meta, args.robust)
            cfg_naive.use_warp = False
            for name, canc in [("naive", FeatureMapCancel(cfg_naive)),
                               ("gyro", FeatureMapCancel(
                                   cancel_cfg(meta, args.robust))),
                               ("flow", FlowWarpCancel(FlowWarpConfig()))]:
                if name == "flow":
                    z, _ = rollout_flow(canc, fm, frames)
                else:
                    z, _ = rollout(canc, fm, gyro)
                rows[name] = {"calm_resid": raw_calm_residual(canc, z)}
                if name in ("gyro", "flow"):
                    st = clip_stats(maxcell_score(z), fps, n, events)
                    rows[name].update(st)
                    key = f"{split}/{name}"
                    results["clips"].setdefault(key, []).append(st)
            for k, v in rows.items():
                agg.setdefault(k, {}).setdefault("calm_resid", []).append(
                    v["calm_resid"])
            print(f"  {seq}: " + "  ".join(
                f"{k}: calm={v['calm_resid']:.4f}"
                + (f" stored={v['stored']:.3f} evRec={v['event_recall']:.2f}"
                   if "stored" in v else "")
                for k, v in rows.items()))
        results["quality"][split] = {
            k: {"calm_resid": float(np.nanmean(v["calm_resid"]))}
            for k, v in agg.items()}

    print("\nbenchmarking (median per-frame latency) ...")
    bench_seqs = seq_dirs(args.dataset, "train") or seq_dirs(args.dataset, "val")
    results["bench"] = bench(bench_seqs[0], size=args.size)
    for k, v in results["bench"].items():
        print(f"  {k:<18} {v:8.3f} ms/frame")

    out = "runs/flow_eval.json"
    with open(out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()

"""RAFT baseline: the learned optical-flow ceiling to compare DIS and the
gyro warp against, answering "is flow really the upper bound?"

RAFT (Teed & Deng, ECCV 2020, princeton-vl/RAFT) runs from third_party/RAFT
(code: BSD-3 per its LICENSE; weights raft-things.pth mirrored from HF
ddrfan/RAFT — both gitignored, fetch with:
  curl -L https://codeload.github.com/princeton-vl/RAFT/tar.gz/refs/heads/master | tar xz -C third_party
  curl -L -o third_party/RAFT/raft-things.pth https://hf-mirror.com/ddrfan/RAFT/resolve/main/raft-things.pth
). RAFT is a quality ceiling, not a latency contender: it runs GRU
recurrent updates on full-resolution features.

The flow is plugged into the SAME dis_flow_warp prediction as DIS (via a
calc-compatible shim) so the only difference is the flow estimator.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/run_raft_eval.py data/dataset --split val --n-seq 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "scripts")
from run_flow_eval import (cancel_cfg, load_seq, raw_calm_residual,  # noqa: E402
                           seq_dirs)
from cerebellum_cancel.featmap_cancel import FeatureMapCancel, rollout  # noqa: E402
from cerebellum_cancel.flow_cancel import FlowWarpCancel, FlowWarpConfig, dis_flow_warp  # noqa: E402


class RaftFlow:
    """calc()-compatible shim around RAFT so dis_flow_warp can consume it.
    calc(prev, cur) returns the dense flow field prev -> cur, like cv2.DIS."""

    def __init__(self, model, device):
        import torch
        self.torch = torch
        self.model = model
        self.device = device

    def calc(self, gray_prev, gray, _flow):
        torch = self.torch
        # RAFT expects 3-channel [0,255] float images, dims divisible by 8
        im0 = np.repeat(gray_prev[..., None], 3, axis=-1)
        im1 = np.repeat(gray[..., None], 3, axis=-1)
        t0 = torch.from_numpy(im0).float()[None].permute(0, 3, 1, 2).to(self.device)
        t1 = torch.from_numpy(im1).float()[None].permute(0, 3, 1, 2).to(self.device)
        h, w = gray.shape
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            t0 = torch.nn.functional.pad(t0, (0, pw, 0, ph))
            t1 = torch.nn.functional.pad(t1, (0, pw, 0, ph))
        with torch.no_grad():
            _, flow = self.model(t0, t1, iters=12, test_mode=True)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()[:h, :w]
        return flow.astype(np.float32)


def rollout_flow_with(canc: FlowWarpCancel, flow_op, fmaps, grays):
    """rollout_flow with an arbitrary calc()-compatible estimator; also
    returns the live cancel object for raw_calm_residual."""
    T_ = len(fmaps)
    z = np.zeros_like(fmaps, dtype=np.float32)
    sur = np.zeros(T_, dtype=np.float32)
    for t in range(T_):
        F = fmaps[t].astype(np.float32)
        gray = cv2.resize(grays[t], (canc.cfg.flow_size[1], canc.cfg.flow_size[0]),
                          interpolation=cv2.INTER_AREA)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if canc.F_prev is None or canc.gray_prev is None:
            pred = F
        else:
            pred = dis_flow_warp(flow_op, canc.F_prev, canc.gray_prev, gray)
        err = F - pred
        if canc.mu is None:
            canc.mu = np.zeros_like(err)
            canc.var = np.ones_like(err)
        assert canc.mu is not None and canc.var is not None
        sd = np.sqrt(canc.var) + 1e-3
        z[t] = err / sd
        b = canc.cfg.border
        sur[t] = float(np.abs(z[t][:, b:-b, b:-b]).max())
        canc.n += 1
        calm = sur[t] < canc.cfg.calm_tau or canc.n <= canc.cfg.warmup
        if calm:
            d = err - canc.mu
            canc.mu += canc.cfg.std_ema * d
            canc.var += canc.cfg.std_ema * (d * d - canc.var)
        canc.F_prev = F
        canc.gray_prev = gray
    return z, sur, canc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset root, e.g. data/dataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-seq", type=int, default=3)
    ap.add_argument("--robust", action="store_true")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, "third_party/RAFT/core")  # raft.py uses flat imports
    sys.path.insert(0, "third_party/RAFT")
    from core.raft import RAFT

    class Args(dict):
        __getattr__ = dict.get  # RAFT mixes `in` tests with attribute access
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = RAFT(Args(small=False, dropout=0, alternate_corr=False,
                      mixed_precision=False))
    sd = torch.load("third_party/RAFT/raft-things.pth", map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    model.to(device).eval()
    raft = RaftFlow(model, device)

    seqs = seq_dirs(args.dataset, args.split)
    rows = {}
    for seq in seqs[:args.n_seq]:
        fm, frames, gyro, meta, fps, labels, n = load_seq(seq)
        out = {}
        cfg_naive = cancel_cfg(meta, args.robust)
        cfg_naive.use_warp = False
        canc = FeatureMapCancel(cfg_naive)
        z, _ = rollout(canc, fm, gyro)
        out["naive"] = raw_calm_residual(canc, z)
        canc = FeatureMapCancel(cancel_cfg(meta, args.robust))
        z, _ = rollout(canc, fm, gyro)
        out["gyro"] = raw_calm_residual(canc, z)
        dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        ts = []
        for rep in range(2):    # 2nd rep is warm; median of the timed one
            t1 = time.perf_counter()
            z, _, canc = rollout_flow_with(FlowWarpCancel(FlowWarpConfig()),
                                           dis, fm, frames)
            dt = (time.perf_counter() - t1) * 1e3 / max(n - 1, 1)
            if rep == 1:
                ts.append(dt)
        out["dis_flow"] = raw_calm_residual(canc, z)
        out["dis_ms"] = float(np.median(ts))
        t1 = time.perf_counter()
        z, _, canc = rollout_flow_with(FlowWarpCancel(FlowWarpConfig()),
                                       raft, fm, frames)
        out["raft_ms"] = (time.perf_counter() - t1) * 1e3 / max(n - 1, 1)
        out["raft_flow"] = raw_calm_residual(canc, z)
        rows[seq] = out
        print(f"{seq}: " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()),
              flush=True)
    name = args.dataset.rstrip("/").replace("/", "_") + f"_{args.split}"
    with open(f"runs/raft_eval_{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    agg = {}
    for r in rows.values():
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
    print("mean: " + "  ".join(f"{k}={np.mean(v):.4f}" for k, v in agg.items()))


if __name__ == "__main__":
    main()

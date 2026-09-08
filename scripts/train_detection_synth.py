"""Detection-level downstream comparison — the §2.3.2 headline, upgraded.

GUIDE §4.3 (roadmap item 1): the synthetic rig's task goes from frame-level
0/1 classification ("is this frame surprising?") to detection level ("WHERE
is the surprise?"). Same storage-selection question as §2.3.2 — do
cerebellum-triggered clips train a better downstream model per stored byte
than uniform sampling (and the full-budget oracle)? — but the downstream
model is now a CenterNet-style heatmap regression head on the frozen
featmaps, evaluated by detection AP of the predicted event centers against
the exact bbox ground truth dumped by make_dataset.py (bboxes.json).

Design choices (kept deliberately small):
  - Head: 2x Conv(3x3) + 1x Conv(1x1), <=32 channels, on the (24,28,28)
    featmaps, single-channel 28x28 heatmap out. The task is single-class
    small-object center localization; no size/offset branches needed.
  - Loss: CenterNet's penalty-reduced focal (alpha=2, beta=4), not plain
    MSE: measured on this rig, MSE collapses to the all-zero heatmap (event
    frames are a small fraction of stored frames, and a near-zero prediction
    already achieves low MSE), giving detection AP = 0. The focal variant
    keeps positive cells dominant and is the standard fix; it is implemented
    inline (20 lines), no external detection dependency.
  - GT heatmap: one 2D Gaussian per bbox, centered at the bbox center in
    heatmap cells (frame 320x240 -> 224x224 resize -> /8 = 28x28 cells),
    sigma = max(box_w, box_h)/2 in cells (clamped >= 0.8), elementwise max;
    the center cell is pinned to exactly 1.0 so positives are well-defined.
  - AP: self-contained VOC-style detection AP. All local maxima of the
    sigmoid heatmap are candidate detections; greedy score-order matching to
    gt centers within a 1.5-cell radius, then all-point interpolated AP.
    One matching pass serves the whole precision-recall curve.
  - Storage selection reuses scripts/train_downstream.py verbatim (clips
    cache runs/downstream_clips.json, AdaptiveTrigger cerebellum clips,
    uniform kx budgets, oracle at the cerebellum budget spent on gt event
    frames) so the comparison stays anchored to the §2.3.2 protocol. Data
    loading and the trigger rollout are computed ONCE and shared across
    seeds; each seed only re-initializes and re-trains the head.

Usage: PYTHONPATH=. .venv/bin/python scripts/train_detection_synth.py data/dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from train_downstream import (clips_to_frames, load_or_compute_clips,  # noqa: E402
                              oracle_frames, seq_dirs, uniform_frames)

OUT_PATH = "runs/detection_synth_results.json"
FMAP_OUT = 28                      # featmap spatial edge (224px input / stride 8)
FRAME_W, FRAME_H = 320, 240        # rendered frame size (make_dataset.py)
MATCH_RADIUS_CELLS = 1.5           # det-to-gt center match tolerance (cells)
PEAK_MIN_SCORE = 0.005             # local maxima below this are not detections;
                                   # val-event confidence is ~1e-3..1e-2, so a
                                   # 0.05 floor would exclude nearly all TPs
                                   # (AP ranks by score, the floor only caps
                                   # the candidate pool at 28*28/frame)


# ---------------------------------------------------------------- heatmaps


def load_bboxes(seq: str) -> dict[int, list[list[float]]]:
    """bboxes.json -> {frame_idx: [[x1,y1,x2,y2], ...]} (pixel coords)."""
    with open(f"{seq}/bboxes.json") as fp:
        raw = json.load(fp)["bboxes"]
    return {int(k): [list(map(float, b)) for b in v] for k, v in raw.items()}


def boxes_to_heatmap(boxes: list[list[float]], out: int = FMAP_OUT,
                     frame_size: tuple[int, int] = (FRAME_W, FRAME_H),
                     sigma_min: float = 0.8) -> np.ndarray:
    """Render xyxy boxes (pixel coords) as one Gaussian per box on an
    `out`x`out` heatmap, elementwise max. Sigma follows the box size. The
    center cell is pinned to exactly 1.0 so positives are well-defined for
    the focal loss (centers land sub-cell, so a raw Gaussian can peak < 1)."""
    w, h = frame_size
    hm = np.zeros((out, out), dtype=np.float32)
    if not boxes:
        return hm
    yy, xx = np.mgrid[0:out, 0:out].astype(np.float32)
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0 / w * out
        cy = (y1 + y2) / 2.0 / h * out
        sigma = max(max(x2 - x1, y2 - y1) / max(w, h) * out / 2.0, sigma_min)
        hm = np.maximum(hm, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                   / (2.0 * sigma ** 2)))
        hm[int(round(cy)), int(round(cx))] = 1.0
    return hm


# ------------------------------------------------------------------- model


class HeatmapHead(nn.Module):
    """CenterNet-style tiny heatmap head on frozen (24,28,28) featmaps."""

    def __init__(self, cin: int = 24, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)          # logits (B, 28, 28)


def heatmap_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss (self-contained).

    positives (target == 1): -(1-p)^a log p ; negatives down-weighted by
    (1-t)^b so cells near a blob are not punished for partial overlap."""
    p = torch.sigmoid(logits)
    pos = targets == 1.0
    n = pos.sum().clamp(min=1)
    p_pos = p[pos].clamp(min=1e-6)
    pos_loss = -(((1 - p_pos) ** alpha) * torch.log(p_pos)).sum()
    p_neg = p[~pos]
    t_neg = targets[~pos]
    neg_loss = -(((1 - t_neg) ** beta) * (p_neg ** alpha)
                 * torch.log((1 - p_neg).clamp(min=1e-6))).sum()
    return (pos_loss + neg_loss) / n


def train_head(x: np.ndarray, y: np.ndarray, seed: int, device: str,
               epochs: int = 40, batch: int = 128,
               lr: float = 1e-3) -> HeatmapHead:
    """Focal-loss heatmap regression; init/shuffling derive from `seed` only."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = HeatmapHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x).to(device)        # (N,24,28,28) float16
    yt = torch.from_numpy(y).to(device)        # (N,28,28) float32
    n = len(xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            mb = perm[i:i + batch]
            loss = heatmap_focal_loss(model(xt[mb].float()), yt[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


# ------------------------------------------------------- detection eval AP


def heatmap_peaks(prob: torch.Tensor, min_score: float = PEAK_MIN_SCORE) -> torch.Tensor:
    """Local 3x3 maxima above min_score -> (P, 3) float [score, y, x]."""
    p = prob.unsqueeze(0).unsqueeze(0)
    mp = torch.nn.functional.max_pool2d(p, 3, stride=1, padding=1)
    mask = (p == mp) & (p >= min_score)
    ys, xs = mask[0, 0].nonzero(as_tuple=True)
    return torch.stack([prob[ys, xs], ys.float(), xs.float()], dim=1)


def detection_ap(dets: np.ndarray, gts: dict[int, list[tuple[float, float]]],
                 radius: float = MATCH_RADIUS_CELLS) -> float:
    """VOC-style interpolated AP for frame-tagged center detections.

    dets: (P, 4) [score, frame_idx, y, x]; gts: {frame_idx: [(cy, cx), ...]}.
    Greedy score-order matching, each gt matched at most once within
    `radius` cells; all-point interpolated precision-recall area.
    Sequences with no gt return NaN (excluded from the mean)."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0 or len(dets) == 0:
        return float("nan")
    order = np.argsort(-dets[:, 0])
    matched = {f: np.zeros(len(c), dtype=bool) for f, c in gts.items()}
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    for rank, di in enumerate(order):
        _, y, x, f = dets[di]
        centers = gts.get(int(f))
        hit = -1
        if centers is not None:
            d2 = [((cy - y) ** 2 + (cx - x) ** 2, gi)
                  for gi, (cy, cx) in enumerate(centers) if not matched[f][gi]]
            if d2 and min(d2)[0] <= radius ** 2:
                _, hit = min(d2)
        if hit >= 0:
            matched[f][hit] = True
            tp[rank] = 1
        else:
            fp[rank] = 1
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):       # 101-point interpolation
        mask = recall >= r
        ap += (prec[mask].max() if mask.any() else 0.0) / 101.0
    return float(ap)


@torch.no_grad()
def eval_detection(model: nn.Module, seqs: list[tuple[np.ndarray, dict]],
                   device: str, batch: int = 256) -> float:
    """Mean per-sequence detection AP over the given (featmaps, bboxes) seqs."""
    aps = []
    for feats, bboxes in seqs:
        gts = {i: [(((b[1] + b[3]) / 2) / FRAME_H * FMAP_OUT,
                     ((b[0] + b[2]) / 2) / FRAME_W * FMAP_OUT) for b in boxes]
               for i, boxes in bboxes.items()}
        dets = []
        xt = torch.from_numpy(feats).to(device)
        for i in range(0, len(xt), batch):
            prob = torch.sigmoid(model(xt[i:i + batch].float()))
            for j in range(len(prob)):
                pk = heatmap_peaks(prob[j]).cpu().numpy()
                if len(pk):
                    fr = np.full((len(pk), 1), float(i + j))
                    dets.append(np.hstack([pk, fr]))
        ap = detection_ap(np.vstack(dets) if dets else np.zeros((0, 4)), gts)
        aps.append(ap)
    return float(np.nanmean(aps))


# --------------------------------------------------------------------- main


def load_split(dataset: str, split: str):
    """-> list of (seq_dir, featmaps float16 (n,24,28,28),
    heatmaps float32 (n,28,28), event-frame indicator bool (n,), bboxes)."""
    out = []
    for seq in seq_dirs(dataset, split):
        fm = np.load(f"{seq}/featmaps.npy")
        bboxes = load_bboxes(seq)
        heat = np.stack([boxes_to_heatmap(bboxes.get(i, [])) for i in range(len(fm))])
        ev = np.array([i in bboxes for i in range(len(fm))])
        out.append((seq, fm, heat, ev, bboxes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--modes", default="adaptive",
                    help="clip triggering modes (train_downstream cache)")
    ap.add_argument("--uniform-mults", default="1,2,4")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    modes = args.modes.split(",")
    mults = [float(m) for m in args.uniform_mults.split(",")]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("loading clips (cached cerebellum trigger rollout) ...")
    cache = load_or_compute_clips(args.dataset, modes)
    mode = modes[0]

    print("loading featmaps + bbox heatmaps ...")
    train = load_split(args.dataset, "train")
    val = [(fm, bboxes) for _, fm, _, _, bboxes in load_split(args.dataset, "val")]

    # storage-selection masks, computed once, shared across seeds
    specs = []                     # (name, kind, mult)
    for m in modes:
        specs.append(("cerebellum" if m == "adaptive" else f"cerebellum({m})",
                      "cb", 1.0))
    specs += [(f"uniform({m:g}x)", "uniform", m) for m in mults]
    specs += [(f"oracle({mode})", "oracle", 1.0)]

    all_fm, all_heat = [], []
    seq_slices = []                # (name-independent) per-seq frame ranges
    off = 0
    for seq, fm, heat, ev, _ in train:
        all_fm.append(fm)
        all_heat.append(heat)
        seq_slices.append((seq, slice(off, off + len(fm)), ev))
        off += len(fm)
    big_fm = np.concatenate(all_fm)          # (N,24,28,28) float16
    big_heat = np.concatenate(all_heat)      # (N,28,28) float32

    datasets = {}
    print(f"{'method':<20}{'stored':>8}{'frames':>8}")
    for name, kind, mult in specs:
        mask = np.zeros(len(big_fm), dtype=bool)
        fracs = []
        for seq, sl, ev in seq_slices:
            info = cache[seq][mode]
            fps, n = info["fps"], min(info["n_frames"], sl.stop - sl.start)
            cb_mask = clips_to_frames([tuple(c) for c in info["clips"]], fps, n)
            frac = cb_mask.mean()
            if kind == "cb":
                m = cb_mask
            elif kind == "oracle":
                m = oracle_frames(frac, ev[:n].astype(np.float32))
            else:
                m = uniform_frames(mult * frac, fps, n)
            mask[sl.start:sl.start + n] = m
            fracs.append(m.mean())
        datasets[name] = (mask, float(np.mean(fracs)))
        print(f"{name:<20}{np.mean(fracs):>8.3f}{int(mask.sum()):>8}")

    results = {"config": {"seeds": args.seeds, "modes": modes,
                          "uniform_mults": mults, "epochs": args.epochs,
                          "device": device,
                          "match_radius_cells": MATCH_RADIUS_CELLS},
               "methods": {}}
    print(f"\nval: {len(val)} sequences; training detection heads "
          f"({args.seeds} seeds/method) on {device}\n")
    for name, _, _ in specs:
        mask, frac = datasets[name]
        aps = []
        t0 = time.time()
        for seed in range(args.seeds):
            model = train_head(big_fm[mask], big_heat[mask], seed, device,
                               epochs=args.epochs)
            aps.append(eval_detection(model, val, device))
            print(f"  {name:<18} seed {seed:>2}: AP {aps[-1]:.3f}")
        results["methods"][name] = {
            "stored_frac": frac, "n_frames": int(mask.sum()),
            "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps)),
            "runs": aps}
        print(f"{name:<20}{frac:>8.3f}{int(mask.sum()):>8}"
              f"{np.mean(aps):>7.3f}±{np.std(aps):<5.3f}  ({time.time() - t0:.0f}s)\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"{'method':<20}{'stored':>8}{'frames':>8}{'val detAP':>14}")
    for name, _, _ in specs:
        r = results["methods"][name]
        print(f"{name:<20}{r['stored_frac']:>8.3f}{r['n_frames']:>8}"
              f"{r['ap_mean']:>7.3f}±{r['ap_std']:<6.3f}")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()

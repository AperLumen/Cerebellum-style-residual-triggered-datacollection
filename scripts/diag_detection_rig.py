"""Diagnostic rig checks for the AP~0 mystery (not part of the protocol).

1. bbox stats + heatmap center mapping check
2. perfect-prediction AP: feed boxes_to_heatmap output through the eval path
3. is the val bar visible in the frozen featmaps? (energy near gt center)
"""

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_detection_synth import (FRAME_H, FRAME_W, FMAP_OUT,
                                   boxes_to_heatmap, detection_ap,
                                   eval_detection, heatmap_peaks, load_bboxes)

torch.manual_seed(0)


def check_bbox_stats(dataset, split, n_seq=3):
    print(f"== bbox stats {split} ==")
    seqs = sorted(glob.glob(f"{dataset}/{split}/seq_*"))[:n_seq]
    total_frames, ev_frames, widths, heights = 0, 0, [], []
    for s in seqs:
        bb = load_bboxes(s)
        n = len(np.load(f"{s}/featmaps.npy", mmap_mode="r"))
        total_frames += n
        ev_frames += len(bb)
        for v in bb.values():
            assert len(v) == 1
            x1, y1, x2, y2 = v[0]
            widths.append(x2 - x1)
            heights.append(y2 - y1)
    print(f"  {len(seqs)} seqs: {total_frames} frames, {ev_frames} event frames "
          f"({ev_frames / total_frames:.1%})")
    print(f"  box w px: mean {np.mean(widths):.1f}  h px: mean {np.mean(heights):.1f}"
          f"  -> cells {np.mean(widths) / FRAME_W * FMAP_OUT:.2f} x "
          f"{np.mean(heights) / FRAME_H * FMAP_OUT:.2f}")


def check_gt_mapping():
    """boxes_to_heatmap center vs eval-time gt center (same convention?),
    and vs a strict pixel-center -> cell mapping."""
    print("== GT heatmap mapping ==")
    # val-like bar: 56x12 at (200,120) px center
    box = [200.0 - 28, 120.0 - 6, 200.0 + 28, 120.0 + 6]
    hm = boxes_to_heatmap([box])
    py, px = np.unravel_index(np.argmax(hm), hm.shape)
    cx_px, cy_px = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    cx_cell = cx_px / FRAME_W * FMAP_OUT
    cy_cell = cy_px / FRAME_H * FMAP_OUT
    print(f"  bbox center px ({cx_px},{cy_px}) -> continuous cell "
          f"({cx_cell:.3f},{cy_cell:.3f}); pinned peak at cell ({px},{py}); "
          f"round -> ({int(round(cx_cell))},{int(round(cy_cell))})")
    # distance from pinned peak to eval gt center (what matching sees)
    d = ((py - cy_cell) ** 2 + (px - cx_cell) ** 2) ** 0.5
    print(f"  peak-to-gt-center distance: {d:.3f} cells (match radius 1.5)")
    # strict half-cell mapping: pixel center (x+0.5) maps into cell ((x+0.5)*s - 0.5)
    s = FMAP_OUT / FRAME_W
    strict = (cx_px + 0.5) * s - 0.5
    print(f"  strict cell-center mapping would give x={strict:.3f} "
          f"(script uses {cx_cell:.3f}, off by half a cell)")


def check_perfect_prediction_ap(dataset):
    """Perfect predictor: prob = GT heatmap -> peaks -> detection_ap."""
    print("== perfect-prediction AP (eval path with GT heatmap as prob) ==")
    seqs = sorted(glob.glob(f"{dataset}/val/seq_*"))[:3]
    for s in seqs:
        bb = load_bboxes(s)
        gts = {i: [(((b[1] + b[3]) / 2) / FRAME_H * FMAP_OUT,
                     ((b[0] + b[2]) / 2) / FRAME_W * FMAP_OUT) for b in boxes]
               for i, boxes in bb.items()}
        dets = []
        for i, boxes in bb.items():
            prob = torch.from_numpy(boxes_to_heatmap(boxes))
            pk = heatmap_peaks(prob).numpy()
            if len(pk):
                dets.append(np.hstack([pk, np.full((len(pk), 1), float(i))]))
        dets = np.vstack(dets)
        ap = detection_ap(dets, gts)
        n_pk = len(dets) / max(len(bb), 1)
        print(f"  {os.path.basename(s)}: AP={ap:.4f}, "
              f"{len(bb)} gt frames, {n_pk:.1f} peaks/gt-frame")


def check_bar_visible(dataset):
    """Is the val bar detectable in the frozen featmaps at all?
    Compare feature energy in the gt cell vs elsewhere on event frames,
    and the same cell on non-event frames."""
    print("== bar visibility in frozen featmaps ==")
    s = sorted(glob.glob(f"{dataset}/val/seq_*"))[0]
    fm = np.load(f"{s}/featmaps.npy").astype(np.float32)
    bb = load_bboxes(s)
    ev_idx = sorted(bb.keys())
    gyc = [int(round(((b[0][1] + b[0][3]) / 2) / FRAME_H * FMAP_OUT)) for b in
           (bb[i] for i in ev_idx)]
    gxc = [int(round(((b[0][0] + b[0][2]) / 2) / FRAME_W * FMAP_OUT)) for b in
           (bb[i] for i in ev_idx)]
    mag = np.abs(fm).mean(axis=1)                     # (n,28,28)
    at_gt = np.array([mag[i, y, x] for i, y, x in zip(ev_idx, gyc, gxc)])
    off_gt = np.array([mag[i, (y + 7) % 28, (x + 7) % 28]
                       for i, y, x in zip(ev_idx, gyc, gxc)])
    bg_idx = [i for i in range(0, len(fm), 7) if i not in bb]
    bg = mag[bg_idx].mean(axis=0)
    bg_at = np.array([bg[y, x] for y, x in zip(gyc, gxc)])
    print(f"  |fmap| at gt cell on event frames: {at_gt.mean():.3f}±{at_gt.std():.3f}")
    print(f"  |fmap| 7 cells away on same frames: {off_gt.mean():.3f}±{off_gt.std():.3f}")
    print(f"  |fmap| at same cells on bg frames : {bg_at.mean():.3f}±{bg_at.std():.3f}")


if __name__ == "__main__":
    ds = "data/dataset"
    check_bbox_stats(ds, "train")
    check_bbox_stats(ds, "val")
    check_gt_mapping()
    check_perfect_prediction_ap(ds)
    check_bar_visible(ds)

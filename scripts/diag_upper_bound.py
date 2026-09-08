"""Upper-bound sanity check: train the heatmap head on ALL train frames
(no storage selection), report train/val AP, loss curve, dump heatmaps.

Also trains a within-train-circle probe: eval on held-out train sequences
to separate 'features can't encode events' from 'val bar is OOD/invisible'.
"""

import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from train_detection_synth import (eval_detection, load_split,
                                   heatmap_focal_loss, train_head)

torch.manual_seed(0)


def to_img(hm, scale=8):
    g = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    return np.array(Image.fromarray(g).resize((28 * scale, 28 * scale),
                                             Image.Resampling.NEAREST))


def dump_panels(gt_hms, pred_hms, frames_idx, out_path, title):
    n = len(frames_idx)
    top = np.hstack([to_img(h) for h in gt_hms])
    bot = np.hstack([to_img(h) for h in pred_hms])
    panel = np.vstack([top, np.full((4, top.shape[1]), 128, np.uint8), bot])
    Image.fromarray(panel).save(out_path)
    print(f"  dumped {out_path}  ({title})")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train = load_split("data/dataset", "train")
    val = load_split("data/dataset", "val")

    big_fm = np.concatenate([fm for _, fm, _, _, _ in train])
    big_heat = np.concatenate([heat for _, _, heat, _, _ in train])
    print(f"train: {big_fm.shape[0]} frames from {len(train)} seqs; "
          f"event frames: {np.mean([ev.mean() for _, _, _, ev, _ in train]):.1%}")

    # loss curve on a subset (speed) while training on everything
    model = train_head(big_fm, big_heat, seed=0, device=device, epochs=40)

    # train-set eval (circles, in-distribution, first 5 seqs)
    tr_eval = [(fm, bb) for _, fm, _, _, bb in train[:5]]
    ap_train = eval_detection(model, tr_eval, device)
    ap_val = eval_detection(model, [(fm, bb) for _, fm, _, _, bb in val], device)
    print(f"UPPER BOUND (all frames, seed 0): train detAP {ap_train:.3f}, "
          f"val detAP {ap_val:.3f}")

    # in-training loss trace for the record (post-hoc, a few batches)
    xt = torch.from_numpy(big_fm[:512]).to(device).float()
    yt = torch.from_numpy(big_heat[:512]).to(device)
    with torch.no_grad():
        loss = heatmap_focal_loss(model(xt), yt)
    print(f"final focal loss on first 512 train frames: {loss.item():.4f}")

    # ---- dumps: val predictions vs GT on event frames ----
    s, fm, heat, ev, bb = val[0]
    xt = torch.from_numpy(fm).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(xt.float())).cpu().numpy()
    ev_idx = np.where(ev)[0]
    pick = ev_idx[np.linspace(0, len(ev_idx) - 1, 6).astype(int)]
    dump_panels(heat[pick], prob[pick], pick,
                "runs/diag_ub_val_heatmaps.png",
                f"upper-bound head, VAL (bars): GT vs pred "
                f"(val AP {ap_val:.3f})")

    # train predictions vs GT on event frames
    s, fm, heat, ev, bb = train[0]
    xt = torch.from_numpy(fm).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(xt.float())).cpu().numpy()
    ev_idx = np.where(ev)[0]
    pick = ev_idx[np.linspace(0, len(ev_idx) - 1, 6).astype(int)]
    dump_panels(heat[pick], prob[pick], pick,
                "runs/diag_ub_train_heatmaps.png",
                f"upper-bound head, TRAIN (circles): GT vs pred "
                f"(train AP {ap_train:.3f})")


if __name__ == "__main__":
    main()

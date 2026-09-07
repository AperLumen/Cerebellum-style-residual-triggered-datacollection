"""Extract 576-d pooled frozen-backbone features for the prepared AoE
sequences, streaming the source mp4 (frames are not kept on disk — disk
budget). Writes features.npy + labels.npy next to featmaps.npy in each
seq dir, frame-aligned with the existing featmaps (same mp4 frame order,
n = len(featmaps)).

Usage: PYTHONPATH=. .venv/bin/python scripts/extract_features_aoe.py data/aoe
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from typing import cast

SRC_ROOT = "data/aoe_src"


def event_mask(events, n, fps):
    y = np.zeros(n, dtype=np.float32)
    for a, b in events:
        y[int(a * fps):int(b * fps) + 1] = 1.0
    return y


def find_video(src_dir: str) -> str:
    for cand in ("raw_video_undistorted.mp4", "raw_video.mp4"):
        p = os.path.join(src_dir, cand)
        if os.path.exists(p):
            return p
    raise IOError(f"no mp4 found in {src_dir}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.mobilenet_v3_small(
        weights="DEFAULT").features.to(device).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])

    seqs = sorted(glob.glob(f"{args.dataset}/val/seq_*"))
    for s in seqs:
        if os.path.exists(f"{s}/features.npy"):
            continue
        meta = json.load(open(f"{s}/events.json"))
        fps = float(meta["fps"])
        n = len(np.load(f"{s}/featmaps.npy", mmap_mode="r"))

        cap = cv2.VideoCapture(find_video(os.path.join(SRC_ROOT,
                                                       meta["source"])))
        if not cap.isOpened():
            raise IOError(f"cannot open video for {s}")
        feats = np.zeros((n, 576), dtype=np.float16)
        batch = []
        t = 0
        while t < n:
            ok, fr = cap.read()
            if not ok:
                break
            batch.append(fr)
            if len(batch) == args.batch:
                feats[t - len(batch) + 1:t + 1] = forward(model, tf, batch,
                                                          device)
                batch = []
            t += 1
        if batch:
            feats[t - len(batch):t] = forward(model, tf, batch, device)
        cap.release()
        np.save(f"{s}/features.npy", feats)
        np.save(f"{s}/labels.npy", event_mask(meta["events"], n, fps))
        print(f"{s}: feats {feats.shape}, "
              f"event frames {int(event_mask(meta['events'], n, fps).sum())}",
              flush=True)
    print("done")


def forward(model, tf, bgr_frames, device) -> np.ndarray:
    imgs = [cast(torch.Tensor, tf(Image.fromarray(
        cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))) for f in bgr_frames]
    x = torch.stack(imgs).to(device)
    f = model(x).mean(dim=(2, 3))
    return f.cpu().numpy().astype(np.float16)


if __name__ == "__main__":
    main()

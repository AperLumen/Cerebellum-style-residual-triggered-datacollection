"""Extract frozen-backbone features for all sequences in the dataset.

MobileNetV3-small (ImageNet-pretrained, frozen) -> 576-d pooled features per
frame, saved alongside the event mask and gyro stream so adapter experiments
never re-run the backbone.

Usage: .venv/bin/python scripts/extract_features.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from typing import cast


def event_mask(events, n, fps):
    y = np.zeros(n, dtype=np.float32)
    for a, b in events:
        y[int(a * fps):int(b * fps) + 1] = 1.0
    return y


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.mobilenet_v3_small(
        weights="DEFAULT").features.to(device).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])

    seqs = sorted(glob.glob(f"{args.dataset}/*/seq_*"))
    for s in seqs:
        feat_path = f"{s}/features.npy"
        if os.path.exists(feat_path):
            continue
        meta = json.load(open(f"{s}/events.json"))
        fps, seconds = meta["fps"], meta["seconds"]
        n = int(seconds * fps)
        frames = sorted(glob.glob(f"{s}/frames/*.png"))
        feats = np.zeros((n, 576), dtype=np.float16)
        for i in range(0, n, args.batch):
            batch = frames[i:i + args.batch]
            imgs = [cast(torch.Tensor, tf(Image.open(p).convert("RGB")))
                    for p in batch]
            x = torch.stack(imgs).to(device)
            f = model(x).mean(dim=(2, 3))          # global pool -> (B, 576)
            feats[i:i + len(batch)] = f.cpu().numpy().astype(np.float16)
        np.save(feat_path, feats)
        np.save(f"{s}/labels.npy", event_mask(meta["events"], n, fps))
        if s == seqs[0] or s.endswith("seq_000"):
            print(f"example {s}: feats {feats.shape}, "
                  f"event frames {int(event_mask(meta['events'], n, fps).sum())}")
    print(f"features extracted for {len(seqs)} sequences -> "
          f"{args.dataset}/<split>/seq_XXX/features.npy")


if __name__ == "__main__":
    main()

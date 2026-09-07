"""Extract frozen-backbone FEATURE MAPS (spatial structure kept) for all
sequences — the input for geometric (gyro-aligned) cancellation, the
feature-space analogue of the ELL negative image.

Layer: MobileNetV3-small `features[:3]` -> (24, 28, 28). Chosen over the
pooled output because cancellation needs spatial alignment: a blob ~16 px
radius at 224 px input covers 2-4 cells here, vs <1 cell at 7x7.

Usage: .venv/bin/python scripts/extract_featmaps.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from typing import cast

warnings.filterwarnings("ignore", message=".*encountered in matmul")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=224,
                    help="backbone input edge; fmap edge = size/8 (stride-8 "
                         "features[:3]). Larger inputs raise f_map (finer "
                         "gyro-warp resolution) — the resolution-axis eval.")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.mobilenet_v3_small(
        weights="DEFAULT").features[:3].to(device).eval()
    tf = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    probe = model(torch.zeros(1, 3, args.size, args.size, device=device))
    _, c, h, w = probe.shape

    seqs = sorted(glob.glob(f"{args.dataset}/*/seq_*"))
    for s in seqs:
        out_path = f"{s}/featmaps.npy"
        if os.path.exists(out_path):
            continue
        frames = sorted(glob.glob(f"{s}/frames/*.png"))
        # the frame count on disk is authoritative: float seconds*fps can
        # truncate by 1 (e.g. 980/30*30 -> 979.99) and slice batches to
        # frames[n:] would overflow fmaps
        n = len(frames)
        fmaps = np.zeros((n, c, h, w), dtype=np.float16)
        for i in range(0, n, args.batch):
            batch = frames[i:i + args.batch]
            imgs = [cast(torch.Tensor, tf(Image.open(p).convert("RGB")))
                    for p in batch]
            x = torch.stack(imgs).to(device)
            f = model(x)                        # (B, C, H, W)
            fmaps[i:i + len(batch)] = f.cpu().numpy().astype(np.float16)
        np.save(out_path, fmaps)
        if s.endswith("seq_000"):
            print(f"example {s}: featmaps {fmaps.shape}")
    print(f"feature maps extracted for {len(seqs)} sequences -> "
          f"{args.dataset}/<split>/seq_XXX/featmaps.npy")


if __name__ == "__main__":
    main()

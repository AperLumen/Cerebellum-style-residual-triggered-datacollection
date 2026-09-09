"""Extract frozen-backbone DINOv2 ViT-S/14 feature maps for all sequences —
the roadmap-item-2 drop-in replacement for MobileNetV3 `features[:3]`.

Backbone: DINOv2 ViT-S/14 (timm `vit_small_patch14_dinov2.lvd142m`,
ImageNet-LVD-142M pretrained weights, frozen). Input 392x392 -> 28x28 patch
grid: the rig's events are blobs of radius ~11 px at the 224 px working
size, which cover only ~1 cell of a 16x16 grid (14 px/cell at 224 input) —
measured to erase event contrast in the cancellation residual (calm p99 ==
event max on 3x val). At 392 input the patch grid is 28x28, i.e. the same
8-px-per-cell granularity and f_map (24.25) as the MobileNetV3 rig, so the
backbone swap isolates feature quality from spatial sampling. (Frame-level
cost note: this is ~3x the 224 px forward — irrelevant offline, but the
always-on cost story of the pipeline stays with the cheap backbone.)

Two outputs per sequence, both derived from the SAME forward pass:
  - featmaps_dino.npy  (n, 384, 28, 28) float16 — patch tokens reshaped to a
    2D grid, the spatial input for gyro-warp cancellation (featmap_cancel)
    and the detection heatmap head
  - features_dino.npy  (n, 384) float16 — mean-pooled patch tokens, the
    frame-level input for the LogisticHead downstream comparison
Neither overwrites the published MobileNetV3 featmaps.npy / features.npy.

Usage: .venv/bin/python scripts/extract_featmaps_dino.py data/dataset
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, cast

# hf-mirror first (the fbaipublicfiles/github route is unreliable here);
# huggingface_hub reads HF_ENDPOINT at import time.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
FMAPS_NAME = "featmaps_dino.npy"
FEATS_NAME = "features_dino.npy"
SIZE = 392                          # 28x28 patch grid == rig's f_map/granularity


class _Dinov2(torch.nn.Module):
    """The slice of the timm DINOv2 interface this script uses."""

    pretrained_cfg: dict[str, Any]

    def forward_features(self, x: torch.Tensor) -> Any:
        ...                                    # post-norm tokens (B, 1+gh*gw, C)


def load_model(device: str, size: int = SIZE) -> tuple[_Dinov2, T.Compose]:
    """Frozen DINOv2 ViT-S/14 @`size` + its ImageNet normalization."""
    import timm

    model = cast(_Dinov2, timm.create_model(
        MODEL_NAME, pretrained=True, num_classes=0, img_size=size))
    model = model.to(device).eval()
    cfg = model.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                    T.Normalize(mean, std)])
    return model, tf


@torch.no_grad()
def extract_seq(model: _Dinov2, tf: T.Compose,
                frames: list[str], batch: int = 64,
                device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Frames (paths) -> (fmaps (n,384,gh,gw) float16, pooled (n,384) float16).

    Row i of both outputs is the backbone output of frame i (batch order
    preserved), so the featmaps stay aligned with frames/*.png, imu.csv and
    bboxes.json by index.
    """
    n = len(frames)
    fmaps: np.ndarray | None = None
    feats = np.zeros((n, 0), dtype=np.float16)     # sized on first batch
    for i in range(0, n, batch):
        paths = frames[i:i + batch]
        imgs = [cast(torch.Tensor, tf(Image.open(p).convert("RGB")))
                for p in paths]
        x = torch.stack(imgs).to(device)
        out = model.forward_features(x)            # (B, 1+gh*gw, 384) post-norm
        tokens = (out["x_norm_patchtokens"]
                  if isinstance(out, dict) else out)
        patch = tokens[:, 1:]                      # drop CLS token
        gh = gw = int(round(patch.shape[1] ** 0.5))
        fm = patch.reshape(len(paths), gh, gw, -1).permute(0, 3, 1, 2)
        if fmaps is None:
            fmaps = np.zeros((n, fm.shape[1], gh, gw), dtype=np.float16)
        fmaps[i:i + len(paths)] = fm.cpu().numpy().astype(np.float16)
        pooled = patch.mean(dim=1)
        if feats.shape[1] == 0:
            feats = np.zeros((n, pooled.shape[1]), dtype=np.float16)
        feats[i:i + len(paths)] = pooled.cpu().numpy().astype(np.float16)
    assert fmaps is not None
    return fmaps, feats


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/dataset")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=SIZE,
                    help="backbone input edge; patch grid = size/14. The "
                         "default 392 reproduces the rig's 28x28 grid (8 "
                         "px/cell, f_map 24.25) at matched FOV")
    ap.add_argument("--force", action="store_true",
                    help="re-extract sequences whose outputs already exist")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tf = load_model(device, args.size)

    seqs = sorted(glob.glob(f"{args.dataset}/*/seq_*"))
    for s in seqs:
        if not args.force and os.path.exists(f"{s}/{FMAPS_NAME}") and \
                os.path.exists(f"{s}/{FEATS_NAME}"):
            continue
        # the frame count on disk is authoritative: float seconds*fps can
        # truncate by 1 (same rationale as extract_featmaps.py)
        frames = sorted(glob.glob(f"{s}/frames/*.png"))
        fmaps, feats = extract_seq(model, tf, frames, args.batch, device)
        np.save(f"{s}/{FMAPS_NAME}", fmaps)
        np.save(f"{s}/{FEATS_NAME}", feats)
        if s.endswith("seq_000"):
            print(f"example {s}: featmaps {fmaps.shape}, feats {feats.shape}")
    print(f"DINOv2 feature maps extracted for {len(seqs)} sequences -> "
          f"{args.dataset}/<split>/seq_XXX/{FMAPS_NAME} + {FEATS_NAME}")


if __name__ == "__main__":
    main()

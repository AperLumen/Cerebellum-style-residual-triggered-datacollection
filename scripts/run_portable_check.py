"""Cross-platform portability check: CPU vs Apple Silicon MPS.

Answers review point 1 (only MPS was benchmarked). The cancellation path
itself is pure NumPy, so the only device-dependent piece is the frozen
backbone forward. This script therefore checks two things on the same
frames:

  1. numerical portability — featmaps extracted on CPU vs MPS must agree
     to float32 tolerance, and the downstream gyro-cancellation residual
     computed from them must agree too;
  2. latency — median backbone forward per device (N_BENCH reps).

Writes runs/portable_check.json. Run while nothing else hogs the GPU:
  PYTHONPATH=. .venv/bin/python scripts/run_portable_check.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch

sys.path.insert(0, "scripts")
from run_flow_eval import cancel_cfg, load_seq, raw_calm_residual, seq_dirs  # noqa: E402
from cerebellum_cancel.featmap_cancel import FeatureMapCancel, rollout  # noqa: E402

N_FRAMES = 32      # enough frames to bound the feature diff
N_BENCH = 100      # reps for the latency median


def to_tensor(tf, frame) -> torch.Tensor:
    return tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def main():
    import torch
    import torchvision
    import torchvision.transforms as T

    seq = seq_dirs("data/dataset", "val")[0]
    _, frames, gyro, meta, fps, labels, n = load_seq(seq)
    frames = frames[:N_FRAMES]
    gyro = gyro[:N_FRAMES]

    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])

    feats, lat = {}, {}
    for device in ["cpu", "mps"]:
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        model = torchvision.models.mobilenet_v3_small(
            weights="DEFAULT").features[:3].to(device).eval()
        with torch.no_grad():
            feats[device] = np.stack([
                model(torch.stack([to_tensor(tf, f)]).to(device))[0]
                .cpu().numpy() for f in frames])
            x = torch.stack([to_tensor(tf, frames[0])]).to(device)
            model(x)  # warmup
            if device == "mps":
                torch.mps.synchronize()
            ts = []
            for _ in range(N_BENCH):
                t0 = time.perf_counter()
                model(x)
                if device == "mps":
                    torch.mps.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
        lat[device] = float(np.median(ts))
        del model

    diff = float(np.max(np.abs(feats["cpu"] - feats["mps"])))
    rel = diff / max(float(np.max(np.abs(feats["cpu"]))), 1e-12)

    # downstream agreement: gyro rollout from each device's featmaps
    resid = {}
    for device, fmaps in feats.items():
        canc = FeatureMapCancel(cancel_cfg(meta))
        z, _ = rollout(canc, fmaps, gyro)
        resid[device] = raw_calm_residual(canc, z)

    out = {
        "seq": seq, "n_frames": N_FRAMES,
        "feature_max_abs_diff": diff, "feature_max_rel_diff": rel,
        "gyro_calm_resid": resid,
        "gyro_resid_abs_diff": abs(resid["cpu"] - resid["mps"]),
        "backbone_fwd_ms": lat, "machine": platform.platform(),
    }
    with open("runs/portable_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

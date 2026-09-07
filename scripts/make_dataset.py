"""Generate a multi-sequence synthetic dataset for the adapter experiments.
Each sequence: textured scene + smooth self-rotation (+IMU) + injected
surprise events with varied count/duration/salience. Ground truth event
masks are exact.

Usage: .venv/bin/python scripts/make_dataset.py --n-train 40 --n-val 10 \
          --seconds 20 --out data/dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_synthetic import make_background, rotation_traj  # noqa: E402

FPS = 30


def render_sequence(out_dir: str, seed: int, seconds: float,
                    motion_scale: float = 1.0, event_alpha: float = 1.0,
                    event_shape: str = "circle"):
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    W, H = 320, 240
    bg = make_background(H, W, seed=seed)
    rate, gyro = rotation_traj(n, FPS, seed=seed + 1, scale=motion_scale)
    # vary illumination drift per sequence
    illum_amp = rng.uniform(2.0, 6.0)
    illum_f = rng.uniform(0.03, 0.08)

    n_events = int(rng.integers(2, 6))
    # spacing must be achievable: n_events points in [3, seconds-3] need
    # (n_events-1) gaps; a fixed 4 s gap deadlocks when 4*(n-1) > seconds-6
    min_gap = min(4.0, (seconds - 6.0) / max(n_events - 1, 1))
    events = []
    tries = 0
    while len(events) < n_events and tries < 1000:
        tries += 1
        t0 = rng.uniform(3, seconds - 3)
        if any(abs(t0 - e[0]) < min_gap for e in events):
            continue
        dur = rng.uniform(0.7, 1.5)
        events.append([round(t0, 2), round(t0 + dur, 2)])
    events.sort()
    radius = int(rng.integers(18, 27))
    drift = rng.uniform(2.0, 6.0)      # px/s
    sensor_noise = rng.uniform(0.5, 1.0)

    f = W / (2.0 * np.tan(np.radians(60.0) / 2.0))
    dxs = f * np.tan(np.radians(rate[:, 1]) / FPS)
    dys = f * np.tan(np.radians(rate[:, 0]) / FPS)
    cum = np.stack([np.cumsum(dxs), np.cumsum(dys)], axis=1).astype(np.float32)

    frames_dir = f"{out_dir}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    imu_lines = ["time,gyroscope.x,gyroscope.y,gyroscope.z"]
    for i in range(n):
        M = np.array([[1, 0, -cum[i, 0]], [0, 1, -cum[i, 1]]], dtype=np.float32)
        frame = cv2.warpAffine(bg, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        frame = cv2.convertScaleAbs(
            frame, alpha=1.0, beta=illum_amp * np.sin(2 * np.pi * illum_f * i / FPS))
        frame = np.clip(frame.astype(np.int16)
                        + rng.normal(0, sensor_noise, frame.shape),
                        0, 255).astype(np.uint8)
        for ea, eb in events:
            if ea <= i / FPS <= eb:
                k = i / FPS - ea
                cx = int(60 + k * drift)
                cy = int(H / 2 + 12 * np.sin(k * 3.0))
                if event_shape == "bar":   # open-set variant: different shape
                    cx = int(200 + k * drift)
                    m = np.zeros_like(frame)
                    cv2.rectangle(m, (cx - 28, cy - 6), (cx + 28, cy + 6),
                                  (255, 255, 255), -1)
                elif event_alpha >= 1.0:
                    m = None
                    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), -1)
                else:
                    m = np.zeros_like(frame)
                    cv2.circle(m, (cx, cy), radius, (255, 255, 255), -1)
                if m is not None:
                    m = m.astype(np.float32) / 255.0
                    frame = (frame.astype(np.float32) * (1 - event_alpha * m)
                             + 255.0 * event_alpha * m)
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{frames_dir}/{i:05d}.png", frame)
        imu_lines.append(f"{i / FPS:.4f},{gyro[i,0]:.4f},{gyro[i,1]:.4f},{gyro[i,2]:.4f}")

    with open(f"{out_dir}/imu.csv", "w") as fp:
        fp.write("\n".join(imu_lines))
    with open(f"{out_dir}/events.json", "w") as fp:
        json.dump({"events": events, "fps": FPS, "seconds": seconds},
                  fp, indent=2)
    return len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--val-motion-scale", type=float, default=3.0,
                    help="val split motion amplitude multiplier — a regime "
                         "shift the fast weights must absorb online")
    ap.add_argument("--event-alpha", type=float, default=1.0,
                    help="event salience: 1.0 = solid white blob, lower = "
                         "faint blend that motion can mask")
    ap.add_argument("--val-event-shape", default="bar", choices=["circle", "bar"],
                    help="open-set test: val events use a shape the trained "
                         "heads have never seen")
    ap.add_argument("--out", default="data/dataset")
    args = ap.parse_args()

    total = 0
    for split, n_seq in [("train", args.n_train), ("val", args.n_val)]:
        scale = args.val_motion_scale if split == "val" else 1.0
        shape = args.val_event_shape if split == "val" else "circle"
        for k in range(n_seq):
            d = f"{args.out}/{split}/seq_{k:03d}"
            if os.path.exists(f"{d}/events.json"):
                continue                      # resume: skip finished sequences
            total += render_sequence(
                d, seed=zlib.crc32(f"{split}:{k}".encode()) % (2**31),
                seconds=args.seconds, motion_scale=scale,
                event_alpha=args.event_alpha, event_shape=shape)
        done = len(glob.glob(f"{args.out}/{split}/seq_*/events.json"))
        print(f"{split}: {done}/{n_seq} sequences")
    n_frames = len(glob.glob(f"{args.out}/train/*/frames/*.png"))
    print(f"total events {total}, train frames {n_frames}")


if __name__ == "__main__":
    main()

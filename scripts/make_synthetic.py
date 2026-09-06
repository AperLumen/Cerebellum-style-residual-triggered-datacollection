"""Synthetic phone-rig recording: textured scene + smooth self-rotation
(+ IMU) + injected surprise events. Produces video.mp4, imu.csv, events.json.

The rig's rotation is a smooth random walk; the IMU is the analytic derivative
plus noise. Events are objects that appear/move while the background motion is
fully explained by the gyro — so a perfect corollary-discharge system sees zero
residual except at events.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

W, H, FPS = 320, 240, 30


def make_background(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Structured, high-contrast texture — faint textures get flattened by the
    display->record chain (codec + compositor scaling), real scenes don't."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8)
    bg = cv2.resize(base, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    # low-frequency structure so compensation errors have spatial gradients
    low = cv2.resize(rng.integers(0, 255, (h // 32, w // 32), dtype=np.uint8),
                     (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    bg = cv2.addWeighted(bg, 0.7, cv2.GaussianBlur(low, (9, 9), 0), 0.3, 0)
    return bg


def rotation_traj(n: int, fps: float, seed: int = 1, scale: float = 1.0):
    """Smooth rotation (deg) via band-limited noise; returns (yaw, pitch, gyro).

    `scale` multiplies the motion amplitude — used to give the val split a
    different motion regime from train (the adapter's fast weights should
    absorb the shift; a static head should not).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    yaw = np.zeros(n)
    pitch = np.zeros(n)
    for f, amp in [(0.15, 25.0), (0.4, 10.0), (0.9, 4.0)]:
        yaw += scale * amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        pitch += scale * amp * 0.7 * np.sin(
            2 * np.pi * f * 1.3 * t + rng.uniform(0, 6.28))
    rate = np.stack([np.gradient(pitch, 1 / fps),   # clean true rate (deg/s)
                     np.gradient(yaw, 1 / fps),
                     np.zeros(n)], axis=1)
    gyro = rate + rng.normal(0, 0.3, rate.shape)     # what the IMU reports
    # Physically consistent rig: the rendered per-frame shift must be exactly
    # what the IMU reports, i.e. shift = f * tan(radians(gyro_rate) / fps).
    # Integrate the clean rate for rendering; the pipeline inverts it up to
    # sensor noise, which is what a real rig looks like.
    return rate, gyro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--n-events", type=int, default=6)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    n = int(args.seconds * FPS)
    bg = make_background(H, W)
    rate, gyro = rotation_traj(n, FPS)   # rate: clean true rate; gyro: as logged

    rng = np.random.default_rng(2)
    events = []
    taken = set()
    while len(events) < args.n_events:
        t0 = rng.uniform(5, args.seconds - 5)
        if any(abs(t0 - e[0]) < 6 for e in events):
            continue
        events.append([round(t0, 2), round(t0 + 1.0, 2)])
    events.sort()

    f = W / (2.0 * np.tan(np.radians(60.0) / 2.0))
    frames_dir = f"{args.out}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    imu_lines = ["time,gyroscope.x,gyroscope.y,gyroscope.z"]
    # per-frame shift from the CLEAN rate — exactly what a perfect IMU would
    # report; the logged gyro differs only by sensor noise. Render by warping
    # the PRISTINE source with the cumulative shift: incremental warping would
    # accumulate interpolation blur and wash the texture out over minutes.
    dxs = f * np.tan(np.radians(rate[:, 1]) / FPS)
    dys = f * np.tan(np.radians(rate[:, 0]) / FPS)
    cum = np.stack([np.cumsum(dxs), np.cumsum(dys)], axis=1).astype(np.float32)
    for i in range(n):
        M = np.float32([[1, 0, -cum[i, 0]], [0, 1, -cum[i, 1]]])
        frame = cv2.warpAffine(bg, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        # illumination drift (slow, global — should be absorbed by neg image)
        frame = cv2.convertScaleAbs(frame, alpha=1.0,
                                    beta=3.0 * np.sin(2 * np.pi * 0.05 * i / FPS))
        # sensor noise
        frame = np.clip(frame.astype(np.int16)
                        + rng.normal(0, 0.7, frame.shape), 0, 255).astype(np.uint8)
        # event: object appears and drifts slowly (like something new entering
        # the scene during collection — appearance gives a full-object residual)
        for ea, eb in events:
            if ea <= i / FPS <= eb:
                k = (i / FPS - ea)
                cx = int(60 + k * 4)
                cy = int(H / 2 + 12 * np.sin(k * 3.0))
                cv2.circle(frame, (cx, cy), 22, (255, 255, 255), -1)
        cv2.imwrite(f"{frames_dir}/{i:05d}.png", frame)
        imu_lines.append(f"{i / FPS:.4f},{gyro[i,0]:.4f},{gyro[i,1]:.4f},{gyro[i,2]:.4f}")
    with open(f"{args.out}/imu.csv", "w") as fp:
        fp.write("\n".join(imu_lines))
    with open(f"{args.out}/events.json", "w") as fp:
        json.dump({"events": events, "fps": FPS, "seconds": args.seconds,
                   "video": "frames/%05d.png"}, fp, indent=2)
    print(f"wrote {n} frames to {frames_dir}/, "
          f"{len(events)} events: {[tuple(e) for e in events]}")


if __name__ == "__main__":
    main()

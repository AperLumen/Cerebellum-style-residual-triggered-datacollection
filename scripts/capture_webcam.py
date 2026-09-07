"""Record the Mac webcam while the scene plays on the screen, then detect the
sync flash and emit a recording directory compatible with run_experiment.py.

Usage (two terminals):
  T1: .venv/bin/python scripts/play_scene.py data/synthetic/frames
  T2: .venv/bin/python scripts/capture_webcam.py data/webcam_rec --seconds 62

macOS will ask for camera access the first time — allow it for your terminal.
Position the laptop so the webcam faces the screen squarely, ~30-50 cm away,
with normal room lighting. Start the player first, then the capture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=62.0)
    ap.add_argument("--imu", default="data/synthetic/imu.csv",
                    help="motion stream the scene was rendered from")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("webcam unavailable — check camera permission")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    vw = None
    brightness, t_start = [], time.time()
    n = 0
    try:
        while time.time() - t_start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if vw is None:
                h, w = gray.shape
                vw = cv2.VideoWriter(f"{args.out}/video.mp4",
                                     cv2.VideoWriter.fourcc(*"mp4v"), 30, (w, h))
                print(f"recording at {w}x{h}")
            vw.write(frame)
            brightness.append(float(gray.mean()))
            n += 1
    finally:
        cap.release()
        if vw:
            vw.release()

    elapsed = time.time() - t_start
    fps = n / elapsed
    b = np.array(brightness)
    flash_idx = int(np.argmax(b))          # first flash = scene t 0
    print(f"captured {n} frames, {elapsed:.1f}s -> {fps:.1f} fps, "
          f"flash at frame {flash_idx} (t={flash_idx / fps:.2f}s, "
          f"brightness spike {b.max():.0f})")

    # events.json in *video clock*: scene-time events shifted by flash offset
    with open("data/synthetic/events.json") as fp:
        meta = json.load(fp)
    offset = flash_idx / fps
    events = [[round(a + offset, 3), round(b_ + offset, 3)]
              for a, b_ in meta["events"]]
    with open(f"{args.out}/events.json", "w") as fp:
        json.dump({"events": events, "fps": 30.0,
                   "video": "video.mp4", "imu_offset": round(offset, 3),
                   "note": f"scene clock offset {offset:.3f}s"}, fp, indent=2)
    shutil.copy(args.imu, f"{args.out}/imu.csv")
    print(f"wrote {args.out}/ (video.mp4, imu.csv, events.json)")
    print("next: PYTHONPATH=. .venv/bin/python scripts/run_experiment.py",
          args.out)


if __name__ == "__main__":
    main()

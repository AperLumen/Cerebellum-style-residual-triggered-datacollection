"""Estimate video-clock offset between a recording and the scene it shows,
by template-matching scene frames into the recording. More robust than
flash detection: works through exposure ramps, and doubles as a check that
the camera was actually pointing at the scene.

Writes ``imu_offset`` (seconds) and ``mirror`` into events.json.
run_pipeline applies both.

Usage: .venv/bin/python scripts/sync_offset.py data/webcam_rec [--max-shift 8]
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np


def load_recording(path: str, width: int = 320) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h = int(g.shape[0] * width / g.shape[1])
        frames.append(cv2.resize(g, (width, h)))
    cap.release()
    return np.array(frames).astype(np.float64), fps


def highpass(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()).ravel()


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, 0)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def match_scene_frame(rec: np.ndarray, scene_frame: np.ndarray,
                      lo: int, hi: int, mirror: bool) -> tuple[float, int]:
    t = scene_frame[:, ::-1] if mirror else scene_frame
    t = highpass(t)
    tn = np.linalg.norm(t)
    best_c, best_f = -2.0, lo
    for f in range(lo, min(hi, len(rec))):
        r = highpass(rec[f])
        c = float(np.dot(r, t) / (np.linalg.norm(r) * tn + 1e-9))
        if c > best_c:
            best_c, best_f = c, f
    return best_c, best_f


def detect_crop(rec: np.ndarray, frame_idx: int,
                scene_aspect: float) -> tuple[float, float, float, float] | None:
    """Find the bounding box of the scene window within the recording frame
    (desktop margins are near-white; menu bar is black). Restricts to the
    largest axis-aligned block that matches the scene aspect ratio."""
    g = rec[frame_idx]
    h, w = g.shape
    mask = ((g > 40) & (g < 250)).astype(np.float64)
    row_frac = mask.mean(axis=1)
    col_frac = mask.mean(axis=0)

    def runs(frac, thr=0.6, min_len=40):
        idx = np.where(frac > thr)[0]
        if len(idx) < min_len:
            return None
        return idx.min(), idx.max()

    r = runs(row_frac)
    c = runs(col_frac)
    if not r or not c:
        return None
    y0, y1 = r
    x0, x1 = c
    aspect = (x1 - x0) / max(y1 - y0, 1)
    if abs(aspect - scene_aspect) / scene_aspect > 0.15:
        return None
    pad = 6
    return (max(0, x0 + pad) / w, max(0, y0 + pad) / h,
            min(w, x1 - pad) / w, min(h, y1 - pad) / h)


def estimate_lag(rec_dir: str, rec_fps: float, mirror: bool,
                 search_pad: int = 40) -> tuple[float, float, tuple | None]:
    """Returns (lag_frames, mean_corr, crop).

    Sync uses the countdown digit '1' shown for 1 s right before scene frame
    0 (play_scene.sync_digit) — far more distinctive than noise texture.
    Scene frame 0 starts one second after the '1' first appears.
    """
    import sys
    sys.path.insert(0, "scripts")
    from play_scene import sync_digit

    meta = json.load(open(f"{rec_dir}/events.json"))
    video = meta.get("video", "video.mp4")
    rec, _ = load_recording(f"{rec_dir}/{video}")
    scene0 = read_gray("data/synthetic/frames/00000.png")
    aspect = scene0.shape[1] / scene0.shape[0]

    t = highpass(cv2.resize(sync_digit("1"), (rec.shape[2], rec.shape[1])))
    tn = np.linalg.norm(t)
    cors = []
    for f in range(min(len(rec), int(15 * rec_fps))):
        r = highpass(rec[f])
        cors.append(float(np.dot(r, t) / (np.linalg.norm(r) * tn + 1e-9)))
    cors = np.array(cors)
    if cors.max() < 0.2:
        raise RuntimeError("countdown digit not found in first 15 s — was the "
                           "recording pointed at the screen during playback?")
    hits = np.where(cors > 0.6 * cors.max())[0]
    p = float(np.median(hits))
    lag = p + rec_fps  # scene frame 0 = one second after '1' appears
    mean_corr = float(cors[int(hits[0])])

    crop = detect_crop(rec, int(min(lag + 10 * rec_fps, len(rec) - 1)), aspect)
    return lag, mean_corr, crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--scene-frames", default="data/synthetic/frames")
    args = ap.parse_args()

    with open(f"{args.dir}/events.json") as fp:
        meta = json.load(fp)
    fps = meta.get("fps")
    if not fps:
        cap = cv2.VideoCapture(f"{args.dir}/{meta.get('video', 'video.mp4')}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    # the digit '1' is near-mirror-symmetric: use it for lag only, then
    # decide mirroring from an actual scene frame
    lag, corr, crop = estimate_lag(args.dir, fps, mirror=False)
    meta_tmp = json.load(open(f"{args.dir}/events.json"))
    video_tmp = meta_tmp.get("video", "video.mp4")
    rec, _ = load_recording(f"{args.dir}/{video_tmp}")
    probe = int(min(lag + 3 * fps, len(rec) - 1))
    scene0 = read_gray(f"{args.scene_frames}/00090.png")
    sz = (rec.shape[2], rec.shape[1])
    t_n = highpass(cv2.resize(scene0, sz))
    t_m = highpass(cv2.resize(scene0[:, ::-1], sz))
    r = highpass(rec[probe])
    rn = np.linalg.norm(r)
    c_n = float(np.dot(r, t_n) / (rn * np.linalg.norm(t_n) + 1e-9))
    c_m = float(np.dot(r, t_m) / (rn * np.linalg.norm(t_m) + 1e-9))
    mirror = c_m > c_n
    offset = lag / fps

    meta["imu_offset"] = round(offset, 4)
    meta["mirror"] = bool(mirror)
    meta["sync_corr"] = round(corr, 3)
    if crop:
        meta["crop"] = [round(v, 4) for v in crop]
    # rewrite ground truth: shift scene-clock events into the video clock
    scene_meta = json.load(open("data/synthetic/events.json"))
    meta["events"] = [[round(a + offset, 3), round(b + offset, 3)]
                      for a, b in scene_meta["events"]]
    with open(f"{args.dir}/events.json", "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"lag {lag:.1f} frames -> offset {offset:.3f}s, mirror={mirror} "
          f"(scene corr normal {c_n:.3f} / mirrored {c_m:.3f}), digit corr "
          f"{corr:.3f}, crop {crop}")
    if corr < 0.3:
        print("WARNING: low digit correlation — is the recording actually "
              "showing the scene?")


if __name__ == "__main__":
    main()

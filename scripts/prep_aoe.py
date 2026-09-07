"""Prepare an Open-AoE segment (ant-research egocentric phone dataset) for
the cerebellum-cancel pipeline, in the same layout as data/dataset:
frames/*.png + imu.csv + events.json (+ featmaps.npy via extract_featmaps.py).

The IMU substitute: Open-AoE ships no gyro stream, only a visual-SLAM
camera trajectory (camera_traj.npz, cam_c2w (T,4,4)). Differentiating it
gives the physical camera rotation rate in camera axes [x=pitch, y=yaw,
z=roll] deg/s — the corollary-discharge signal the pipeline expects, at
SLAM accuracy (dataset RPE-Rot ~5 deg, measured median here ~0.1 deg/frame).
Per-axis signs are auto-calibrated against DIS flow: the predicted content
shift must positively correlate with the measured one.

events.json carries the segment's real f_map (fx * map_w / frame_w) so the
eval scripts build the right CancelConfig; events are the annotation's
active-manipulation intervals (verb != "hold", hand != "none").

Usage:
  PYTHONPATH=. .venv/bin/python scripts/prep_aoe.py data/aoe_src/<seg> \
      --out data/aoe/val/seq_000
  PYTHONPATH=. .venv/bin/python scripts/extract_featmaps.py data/aoe
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

MAP_W = 28          # featmap width (MobileNetV3 features[:3])

PASSIVE_VERBS = {"hold"}


def trajectory_gyro(cam_c2w: np.ndarray, fps: float) -> np.ndarray:
    """Physical camera rotation rate (deg/s) in camera axes, per frame.
    R_rel = R_w2c[t] @ R_c2w[t-1] maps cam(t-1) coords into cam(t) coords;
    the camera's physical rotation in its own frame is R_rel.T."""
    T = len(cam_c2w)
    gyro = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T):
        R_rel = cam_c2w[t][:3, :3].T @ cam_c2w[t - 1][:3, :3]
        rv, _ = cv2.Rodrigues(R_rel.T)          # physical rotation, cam axes
        gyro[t] = np.degrees(rv.ravel()) * fps
    return gyro


def calibrate_signs(gyro: np.ndarray, frames: list, fps: float,
                    fx_full: float) -> tuple[np.ndarray, list[float]]:
    """Flip per-axis signs so the predicted content shift correlates
    positively with DIS-measured flow. Returns (gyro, per-axis r)."""
    n = len(frames)
    f_flow = fx_full * 320.0 / frames[0].shape[1]   # focal at 320x240
    idx = np.linspace(1, n - 1, 40).astype(int)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    pred, meas = [], []
    for t in idx:
        g1 = cv2.resize(cv2.cvtColor(frames[t - 1], cv2.COLOR_BGR2GRAY),
                        (320, 240))
        g2 = cv2.resize(cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY),
                        (320, 240))
        flow = dis.calc(g1, g2, None)
        m = flow.reshape(-1, 2).mean(axis=0)        # content shift, flow px
        p = [-f_flow * np.tan(np.radians(gyro[t][1]) / fps),   # warp amount
             -f_flow * np.tan(np.radians(gyro[t][0]) / fps)]
        pred.append(p)
        meas.append(m)
    pred, meas = np.array(pred), np.array(meas)
    rs = []
    for ax in range(2):
        r = np.corrcoef(pred[:, ax], meas[:, ax])[0, 1]
        rs.append(float(r))
        if r < 0:
            gyro[:, ax] *= -1
    return gyro, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Open-AoE segment dir (with camera_traj.npz, "
                                "ego_action_annotation.json, mp4)")
    ap.add_argument("--out", required=True, help="output seq dir")
    ap.add_argument("--video", default=None,
                    help="mp4 path (default: raw_video_undistorted.mp4 "
                         "in src, else raw_video.mp4)")
    ap.add_argument("--frame-size", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = all frames shared by video and trajectory")
    args = ap.parse_args()

    video = args.video
    if video is None:
        for cand in ("raw_video_undistorted.mp4", "raw_video.mp4"):
            p = os.path.join(args.src, cand)
            if os.path.exists(p):
                video = p
                break
    info_p = os.path.join(args.src, "undistorted_video_info.json")
    info = json.load(open(info_p)) if os.path.exists(info_p) else {}
    fps = float(info.get("fps", 30.0))
    fx = float(info.get("cameraParams", {}).get("fx_pixels", 0)) \
        or float(np.load(os.path.join(args.src, "camera_traj.npz"))
                 ["intrinsic"][0, 0])

    traj = np.load(os.path.join(args.src, "camera_traj.npz"))["cam_c2w"]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, tuple(args.frame_size),
                                 interpolation=cv2.INTER_AREA))
    cap.release()
    n = min(len(frames), len(traj))
    if args.max_frames:
        n = min(n, args.max_frames)
    frames, traj = frames[:n], traj[:n]
    print(f"video frames={len(frames)} traj T={len(traj)} -> n={n} @ {fps}fps")

    gyro = trajectory_gyro(traj, fps)
    gyro, rs = calibrate_signs(gyro, frames, fps, fx)
    print(f"sign calibration r (shift vs flow): x={rs[0]:.3f} y={rs[1]:.3f}")
    print(f"gyro deg/s: mean|.|={np.abs(gyro).mean(axis=0).round(2)} "
          f"p95|yaw|={np.percentile(np.abs(gyro[:,1]),95):.1f}")

    os.makedirs(f"{args.out}/frames", exist_ok=True)
    for t, fr in enumerate(frames):
        cv2.imwrite(f"{args.out}/frames/frame_{t:06d}.png", fr)

    with open(f"{args.out}/imu.csv", "w") as f:
        f.write("time,gyroscope.x,gyroscope.y,gyroscope.z\n")
        for t in range(n):
            f.write(f"{t / fps:.6f},{gyro[t][0]:.6f},{gyro[t][1]:.6f},"
                    f"{gyro[t][2]:.6f}\n")

    ann = json.load(open(os.path.join(args.src, "ego_action_annotation.json")))
    events = []
    for seg in ann:
        acts = seg.get("atomic_action") or []
        if any(a.get("verb") not in PASSIVE_VERBS
               and a.get("hand") != "none" for a in acts):
            events.append([float(seg["start_ts"]), float(seg["end_ts"])])
    # featmap sees the backbone input (224 = full sensor width scaled),
    # not the stored frame: f_map = fx * map_w / sensor_w
    f_map = fx * MAP_W / 1920.0     # 224-input px / 8 stride -> map px
    meta = {"fps": fps, "seconds": n / fps, "events": events,
            "f_map": f_map, "source": os.path.basename(args.src.rstrip("/")),
            "note": "gyro derived from SLAM camera trajectory"}
    with open(f"{args.out}/events.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"events: {len(events)} active intervals; f_map={f_map:.2f} "
          f"(fx={fx:.1f}); frames -> {args.out}")


if __name__ == "__main__":
    main()

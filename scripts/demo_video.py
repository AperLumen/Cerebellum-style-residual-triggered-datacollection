"""PPT 演示视频渲染：合成 rig 上逐帧跑 AcquisitionLayer，可视化采集决策。

把"小脑式残差触发采集层"在一个 1280x720 画布上演出来，给 PPT/视频用：

  左半：原始帧（相机自运动 + 注入的 blob/bar 事件）
  右半：corollary-discharge 取消后的残差 z-map 热力图叠加
        （可预测的自运动已被抵消，剩下的亮斑 = 不可预测的世界运动）
  HUD ：场景/时间、● KEEPING / ○ SKIPPING 决策徽标、surprise  sparkline、
        存储占比进度条（cerebellum vs naive 全存）、GT 事件区间标记
  结尾：总结卡——帧级存储占比 vs GT 事件覆盖率（成本账的一句话）

数据源二选一：
  --src data/webcam_rec（默认）：webcam 实拍 + iPhone IMU + GT 事件标注
    （demo_acquisition.py 的已验证口径：robust 统计，14.4% 帧级存储）。
    真实 footage 有平静期，"跳过平静帧"这个故事讲得出来。
  --scenes data/dataset/val/seq_000 ...：合成 rig 单段（每帧都有注入事件，
    残差常年偏高——只适合做 z-map 可视化，存储对比数字不好看）。

Usage:
  PYTHONPATH=. .venv/bin/python scripts/demo_video.py --out runs/demo_video.mp4
  PYTHONPATH=. .venv/bin/python scripts/demo_video.py --out runs/demo_video.mp4 \
      --max-frames 900          # 只渲前 30 秒
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import deque

import cv2
import numpy as np

from cerebellum_cancel.acquisition import AcquisitionLayer
from cerebellum_cancel.featmap_cancel import CancelConfig
from cerebellum_cancel.io import gyro_at, load_imu_csv

W, H = 1280, 720            # canvas
FW, FH = 640, 480           # each panel (frame is 320x240, x2 nearest)
HUD = H - FH                # bottom HUD bar height
FPS = 30
SPARK_N = 160               # surprise sparkline window (frames)


# --------------------------------------------------------------------- data

def load_scene(src_dir: str):
    """-> (frames bgr list, imu samples, events [(t0,t1)], fps, robust).

    两种布局：webcam 录制目录（video.mp4 + imu_offset + robust 统计）
    或合成 rig 段目录（frames/*.png + 30fps 直连）。
    """
    with open(f"{src_dir}/events.json") as fp:
        meta = json.load(fp)
    fps = float(meta.get("fps") or FPS)
    events = [tuple(e) for e in meta["events"]]
    imu = load_imu_csv(f"{src_dir}/imu.csv")
    video = meta.get("video")
    if video:                                   # webcam 录制目录
        for s in imu:                           # IMU 时钟 -> 视频时钟
            s.t += float(meta.get("imu_offset", 0.0))
        cap = cv2.VideoCapture(os.path.join(src_dir, video))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames, imu, events, fps, True
    paths = sorted(glob.glob(f"{src_dir}/frames/*.png"))
    frames: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise IOError(f"broken frame: {p}")
        frames.append(img)
    return frames, imu, events, fps, False


def rollout(frames, imu, fps, robust):
    """逐帧跑采集层 -> (keeps, surprises, z_maps, clips)。z_maps 未归一化。"""
    layer = AcquisitionLayer(
        cancel_cfg=CancelConfig(fps=fps, robust=robust))
    keeps, surprises, z_maps = [], [], []
    hint = 0
    for idx, frame in enumerate(frames):
        t = idx / fps
        gyro, hint = gyro_at(imu, hint, t)
        out = layer.step(frame[:, :, ::-1].copy(), gyro, t)   # BGR -> RGB
        keeps.append(bool(out["keep"]))
        surprises.append(float(out["surprise"]))
        z_maps.append(out["z_map"])
    return keeps, surprises, z_maps, layer.finalize()


# ------------------------------------------------------------------ drawing

def heat_overlay(frame_bgr: np.ndarray, z_map: np.ndarray) -> np.ndarray:
    """z_map 热力图叠加原帧（与 runs/acquisition_demo_peak.png 同口径）。"""
    zm = cv2.resize(z_map, (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    zm = zm / (zm.max() + 1e-6)
    heat = cv2.applyColorMap((zm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.5, heat, 0.5, 0)


def draw_hud(canvas, scene_i, n_scenes, t, keep, surprise, spark, kept, total,
             ev_now, ev_kept, ev_total):
    y0 = FH
    cv2.rectangle(canvas, (0, y0), (W, H), (24, 24, 24), -1)
    # --- left: scene / time / decision badge ---
    cv2.putText(canvas, f"SCENE {scene_i + 1}/{n_scenes}   t={t:5.2f}s",
                (16, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    badge = ("KEEPING", (80, 220, 80)) if keep else ("SKIPPING", (140, 140, 140))
    cv2.circle(canvas, (30, y0 + 62), 9,
               badge[1] if keep else (140, 140, 140), -1)
    cv2.putText(canvas, badge[0], (48, y0 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, badge[1], 2)
    if ev_now:
        cv2.putText(canvas, "GT EVENT", (240, y0 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2)
    # --- middle: surprise sparkline ---
    sx, sw = 420, 380
    cv2.putText(canvas, "surprise", (sx, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    base, amp = y0 + 80, 48.0
    cv2.line(canvas, (sx, base), (sx + sw, base), (80, 80, 80), 1)
    pts = [(int(sx + i * sw / max(len(spark) - 1, 1)),
            int(base - min(v / amp, 1.0) * amp)) for i, v in enumerate(spark)]
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (0, 200, 255), 2)
    # --- right: storage bars (cerebellum vs naive) ---
    bx, bw = 860, 380
    frac = kept / max(total, 1)
    cv2.putText(canvas, f"STORAGE  {frac * 100:5.1f}% of frames kept",
                (bx, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    cv2.rectangle(canvas, (bx, y0 + 40), (bx + bw, y0 + 58), (60, 60, 60), -1)
    cv2.rectangle(canvas, (bx, y0 + 40), (bx + int(bw * frac), y0 + 58),
                  (80, 220, 80), -1)
    cv2.putText(canvas, "naive (store all)", (bx, y0 + 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    cv2.rectangle(canvas, (bx, y0 + 94), (bx + bw, y0 + 104), (60, 60, 60), -1)
    cv2.rectangle(canvas, (bx, y0 + 94), (bx + bw, y0 + 104), (120, 120, 200), -1)
    ev = f"event coverage {ev_kept}/{ev_total}" if ev_total else "no GT events"
    cv2.putText(canvas, ev, (bx, y0 + 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 120), 2)


def title_card(scene_i, n_scenes, seq_name, n_events) -> np.ndarray:
    card = np.zeros((H, W, 3), np.uint8)
    cv2.putText(card, f"SCENE {scene_i + 1} / {n_scenes}", (W // 2 - 260, H // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 220, 220), 3)
    cv2.putText(card, f"{seq_name}   ({n_events} ground-truth event(s))",
                (W // 2 - 330, H // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (160, 160, 160), 2)
    return card


def summary_card(total, kept, ev_kept, ev_total, clip_dur_s, total_s) -> np.ndarray:
    card = np.zeros((H, W, 3), np.uint8)
    frac = kept / max(total, 1)
    clip_frac = clip_dur_s / max(total_s, 1e-9)
    cov = ev_kept / max(ev_total, 1)
    cv2.putText(card, "SUMMARY", (W // 2 - 170, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (220, 220, 220), 3)
    lines = [
        f"frames stored:  {frac * 100:5.1f}%   ({kept} / {total})",
        f"clip-level storage:  {clip_frac * 100:5.1f}% of stream duration",
        f"GT event frames covered:  {cov * 100:5.1f}%   ({ev_kept} / {ev_total})",
        "",
        "always-on layer ~1 ms/frame: corollary-discharge cancellation",
        "+ residual triggering decides what deserves storage.",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(card, ln, (W // 2 - 480, 250 + i * 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200, 200, 200), 2)
    return card


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo_video.mp4")
    ap.add_argument("--src", default="data/webcam_rec",
                    help="webcam 录制目录（默认）")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="合成 rig 段目录列表（给了就走合成模式）")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 只渲前 N 帧")
    args = ap.parse_args()

    if args.scenes:                             # 合成 rig 模式
        scene_dirs = args.scenes
    else:                                       # webcam 单源模式
        scene_dirs = [args.src]

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter.fourcc(*"mp4v"),
                             FPS, (W, H))
    if not writer.isOpened():
        raise IOError(f"cannot open video writer: {args.out}")

    total = kept = ev_total = ev_kept = 0
    clip_dur_s = total_s = 0.0
    spark = deque(maxlen=SPARK_N)
    for si, d in enumerate(scene_dirs):
        frames, imu, events, fps, robust = load_scene(d)
        if args.max_frames:
            frames = frames[:args.max_frames]
        keeps, surprises, z_maps, clips = rollout(frames, imu, fps, robust)
        clip_dur_s += sum(b - a for a, b in clips)
        total_s += len(frames) / fps
        for _ in range(FPS):                       # 1s title card
            writer.write(title_card(si, len(scene_dirs), os.path.basename(d),
                                    len(events)))
        for idx, frame in enumerate(frames):
            t = idx / fps
            keep = keeps[idx]
            canvas = np.zeros((H, W, 3), np.uint8)
            left = cv2.resize(frame, (FW, FH), interpolation=cv2.INTER_NEAREST)
            right = cv2.resize(heat_overlay(frame, z_maps[idx]),
                               (FW, FH), interpolation=cv2.INTER_NEAREST)
            canvas[:FH, :FW] = left
            canvas[:FH, FW:] = right
            spark.append(surprises[idx])
            ev_now = any(a <= t <= b for a, b in events)
            is_ev = ev_now
            total += 1
            kept += keep
            ev_total += is_ev
            ev_kept += is_ev and keep
            draw_hud(canvas, si, len(scene_dirs), t, keep, surprises[idx],
                     spark, kept, total, ev_now, ev_kept, ev_total)
            writer.write(canvas)
        print(f"scene {os.path.basename(d)} rendered "
              f"(kept {sum(keeps)}/{len(keeps)})")
    for _ in range(2 * FPS):                       # 2s summary card
        writer.write(summary_card(total, kept, ev_kept, ev_total,
                                  clip_dur_s, total_s))
    writer.release()
    print(f"summary: stored {kept}/{total} ({kept / total * 100:.1f}%), "
          f"clips {clip_dur_s:.1f}s/{total_s:.1f}s, "
          f"event coverage {ev_kept}/{ev_total}")
    print(f"video -> {args.out}")


if __name__ == "__main__":
    main()

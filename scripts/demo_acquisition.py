"""采集协议层最小集成 demo：接管一个录制目录的采集决策。

对 data/webcam_rec/（webcam 实拍 + iPhone IMU，imu_offset 已在
events.json 里标定）逐帧跑 AcquisitionLayer，输出采集决策报告：

  - 逐帧 keep 决策压缩成的片段列表 + 两种口径的存储占比
    （帧级 keep 占比 / 片段级时长占比）
  - surprise 时间序列 -> runs/acquisition_demo.json
  - 残差峰值帧的 z_map 热力图叠加原帧 -> runs/acquisition_demo_peak.png
  - 成本估算（帧数 x 每帧成本，GUIDE §1.2 实测账）+ 本脚本实测墙钟

展示口径（GUIDE §2.3.7）：动能不可预测 ≠ 语义重要性。本 demo 演示的是
运动去重 / 带宽预筛 / 片段筛选——常开层花 ~1 ms/帧，把存储集中在
运动上不可预测的片段上；语义重要性仲裁在视觉模型侧，不在这里。

Usage:
  PYTHONPATH=. .venv/bin/python scripts/demo_acquisition.py [src_dir]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from cerebellum_cancel.acquisition import AcquisitionLayer
from cerebellum_cancel.core import TriggerConfig
from cerebellum_cancel.eval import evaluate
from cerebellum_cancel.featmap_cancel import CancelConfig
from cerebellum_cancel.io import gyro_at, load_imu_csv

# GUIDE §1.2 成本账（本机 MPS 实测）：冻结 backbone 前 3 段 0.827 ms/帧，
# 像素管线核心（warp+统计+触发）0.171 ms/帧
MS_BACKBONE = 0.827
MS_PIPELINE = 0.171


def compress_keeps(keeps: list[bool], fps: float) -> list[tuple[float, float]]:
    """逐帧 keep 决策压缩成 (start, end) 秒片段列表（决策报告用）。"""
    segs = []
    start = None
    for i, k in enumerate(keeps + [False]):     # 尾部哨兵封闭末段
        if k and start is None:
            start = i
        elif not k and start is not None:
            segs.append((round(start / fps, 3), round((i - 1) / fps, 3)))
            start = None
    return segs


def peak_overlay(frame_bgr: np.ndarray, z_map: np.ndarray, path: str):
    """z_map 热力图叠加原帧（0.5 alpha），存到 path。"""
    zm = cv2.resize(z_map, (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    zm = zm / (zm.max() + 1e-6)
    heat = cv2.applyColorMap((zm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(frame_bgr, 0.5, heat, 0.5, 0)
    cv2.imwrite(path, over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="data/webcam_rec",
                    help="录制目录（video.mp4 + imu.csv + events.json）")
    ap.add_argument("--out", default="runs/acquisition_demo.json")
    ap.add_argument("--peak-img", default="runs/acquisition_demo_peak.png")
    ap.add_argument("--max-frames", type=int, default=0,
                    help=">0 时只跑前 N 帧（调试用）")
    args = ap.parse_args()

    with open(f"{args.src}/events.json") as fp:
        meta = json.load(fp)
    video_path = os.path.join(args.src, meta.get("video", "video.mp4"))
    fps = float(meta.get("fps") or 30.0)
    events = [tuple(e) for e in meta["events"]]
    imu = load_imu_csv(f"{args.src}/imu.csv")
    for s in imu:                                   # IMU 时钟 -> 视频时钟
        s.t += float(meta.get("imu_offset", 0.0))

    # 真实数据用 robust 统计（winsorized always-update）：实测真实视频
    # 96% 的帧被 calm 门挡住，门控 EMA 会冻结，robust 模式才工作正常
    # （scripts/train_downstream_aoe.py 等真实数据配置同为 robust=True）
    layer = AcquisitionLayer(cancel_cfg=CancelConfig(fps=fps, robust=True),
                             trigger_cfg=TriggerConfig())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")

    keeps, surprises = [], []
    peak = {"surprise": -1.0, "t": 0.0, "z_map": None, "frame": None}
    hint, idx, t0 = 0, 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        gyro, hint = gyro_at(imu, hint, t)
        out = layer.step(frame[:, :, ::-1].copy(), gyro, t)  # BGR -> RGB
        keeps.append(out["keep"])
        surprises.append(out["surprise"])
        if out["surprise"] > peak["surprise"]:
            peak = {"surprise": out["surprise"], "t": t,
                    "z_map": out["z_map"], "frame": frame}
        idx += 1
        if args.max_frames and idx >= args.max_frames:
            break
    cap.release()
    wall_s = time.perf_counter() - t0

    clips = layer.finalize()
    duration = idx / fps
    keep_segs = compress_keeps(keeps, fps)
    stored_frames = float(np.mean(keeps)) if keeps else 0.0
    stored_clips = sum(b - a for a, b in clips) / max(duration, 1e-9)

    # 残差峰值帧热力图叠加
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if peak["frame"] is not None:
        peak_overlay(peak["frame"], peak["z_map"], args.peak_img)

    report = {
        "source": args.src,
        "fps": fps,
        "n_frames": idx,
        "duration_s": round(duration, 3),
        "keep_segments": keep_segs,
        "clips": [list(c) for c in clips],
        "stored_fraction_frames": round(stored_frames, 4),
        "stored_fraction_clips": round(stored_clips, 4),
        "surprise": [round(float(s), 4) for s in surprises],
        "peak": {"t": round(peak["t"], 3), "surprise": peak["surprise"]},
        "ground_truth_events": [list(e) for e in events],
    }
    with open(args.out, "w") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=1)

    # 信息性对照：与人工标注事件的重合度（运动去重口径，非语义重要性）
    ev = evaluate("acquisition", clips, events, duration, fps)

    print(f"\n源: {args.src}  ({idx} 帧 @ {fps:.0f} fps, {duration:.1f}s)")
    print(f"keep 决策片段: {len(keep_segs)} 段, "
          f"帧级存储占比 {stored_frames:.3f}")
    print(f"触发片段: {len(clips)} 段 {['(%.2f-%.2f)s' % c for c in clips]}")
    print(f"片段级存储占比 {stored_clips:.3f}   "
          f"(事件重合: recall {ev.recall:.2f}, precision {ev.precision:.2f})")
    print(f"surprise 峰值 {peak['surprise']:.1f} @ t={peak['t']:.2f}s "
          f"-> {args.peak_img}")
    est_ms = idx * (MS_BACKBONE + MS_PIPELINE)
    print(f"\n成本估算: {idx} 帧 x {MS_BACKBONE + MS_PIPELINE:.3f} ms/帧 "
          f"(backbone {MS_BACKBONE} + 管线 {MS_PIPELINE}, GUIDE §1.2)"
          f" = {est_ms / 1000:.1f}s CPU 时间")
    print(f"实测墙钟 {wall_s:.1f}s ({wall_s / max(idx, 1) * 1000:.2f} ms/帧,"
          f" 含视频解码)")
    print(f"报告 -> {args.out}")


if __name__ == "__main__":
    main()

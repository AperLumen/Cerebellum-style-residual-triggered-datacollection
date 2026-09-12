"""真机 IMU 残差对照：**第一个非替身陀螺仪**的数据点。

背景（GUIDE §4.2 边界 5）。此前全部真实数据（Open-AoE）的 gyro 都是**替身**：
对视觉 SLAM 轨迹差分。那条链路把 SLAM 的失效模式（低纹理、快速运动）继承给
了"副本"，而行走级 probe（§0.4 表 13）把"gyro warp 在真实数据上不赚"的归因
之一正指向**副本质量**。手机陀螺仪没有那些失效模式，所以这个实验把
"副本不行"与"warp 模型不行"分开。

数据：simorautiainen/moving-in-messy-apartment-video-with-imu（Galaxy S21 手持，
1088px 方帧 @ ~28 fps 不等间隔，真 gyro 经 prep_real_imu.py 标定为 rad/s、
FOV≈55°；f_map 在 events.json 里）。

对照三个预测器（与 §2.3.1 / run_flow_eval.py 同口径）：
  naive   不补偿
  gyro    FeatureMapCancel（真 IMU 副本）
  flow    FlowWarpCancel（DIS 稠密光流 warp，质量上限参照）

指标：平静帧原始残差、每帧延迟，以及触发器的存储率/片段数。**真机段没有人工
事件标注，所以不报 recall**——这是与 AoE 评测的口径差异，不是遗漏。

用法::

  PYTHONPATH=. .venv/bin/python scripts/run_real_imu_eval.py \
      data/real_imu/val/seq_000 --n 400
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.core import AdaptiveTrigger
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               maxcell_score, rollout)
from cerebellum_cancel.flow_cancel import (FlowWarpCancel, FlowWarpConfig,
                                           rollout_flow)
from cerebellum_cancel.io import load_imu_csv


def load(seq: str, n: int):
    """读帧 -> 特征图（与 extract_featmaps.py 同链）+ 320x240 灰度对。"""
    import torch
    import torchvision
    import torchvision.transforms as T
    from PIL import Image
    from typing import cast

    meta = json.load(open(f"{seq}/events.json"))
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    paths = sorted(glob.glob(f"{seq}/frames/*.png"))[:n]
    n = min(n, len(paths), len(gyro))

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.mobilenet_v3_small(
        weights="DEFAULT").features[:3].to(dev).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    fmaps, grays = [], []
    with torch.no_grad():
        for p in paths[:n]:
            img = Image.open(p).convert("RGB")
            # 与 io.VideoStream.read_gray 同口径：缩到 320x240 再轻 blur，
            # 抑制编码/插值噪声、保留事件结构
            g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (320, 240), interpolation=cv2.INTER_AREA)
            grays.append(cv2.GaussianBlur(g, (3, 3), 0))
            x = cast(torch.Tensor, tf(img))
            fmaps.append(model(x.unsqueeze(0).to(dev))[0].cpu().numpy()
                         .astype(np.float32))
    return np.stack(fmaps), grays, gyro[:n], meta, n


def calm_residual(z: np.ndarray, var: np.ndarray, thr: float = 4.0) -> float:
    """已发表口径（run_flow_eval.raw_calm_residual）：用运行 sd 把 z 还原成
    原始残差，只在"不惊讶"的帧上平均。

    **注意这是方法相关口径**：每个方法用自己的 surprise 选 calm 帧，选出的帧
    集不同（实测 naive 7.1%、rotation 8.4%），所以跨方法比较时会混入选择偏差。
    本脚本因此同时报方法无关的 ``full_frame`` 与 ``frame_diff`` 两个口径，
    以它们为准（见 docstring 的"度量"一节）。保留此函数只为与已发布数字对齐。
    """
    sd = np.sqrt(var) + 1e-3
    calm = []
    for t in range(len(z)):
        if float(np.abs(z[t]).max()) < thr:
            calm.append(float(np.abs(z[t] * sd).mean()))
    return float(np.mean(calm)) if calm else float("nan")


def full_frame_residual(errs: np.ndarray) -> float:
    """方法无关口径 A（主口径）：全帧原始残差均值。

    没有选择、没有阈值，是"预测器把可预测成分减掉了多少"的无偏读数。
    事件帧对所有方法贡献同量级误差，因此不构成对某一方法的偏袒。
    """
    return float(np.mean(errs))


def frame_diff_residual(errs: np.ndarray, imgs: np.ndarray) -> float:
    """方法无关口径 B：用**原始图像帧差**定义平静帧（与任何预测器无关），
    在这些帧上平均原始残差。

    帧差掩码取最小的一半帧——它们最可能是"无事件、纯自运动"的帧。
    这样"平静"的定义不依赖被评测的方法本身，避免了循环选择。
    """
    ref = np.asarray([float(np.abs(imgs[t] - imgs[t - 1]).mean())
                      for t in range(1, len(imgs))])
    med = float(np.median(ref))
    calm = np.zeros(len(errs), dtype=bool)
    calm[1:] = ref < med
    return float(np.mean(errs[calm])) if calm.any() else float("nan")


def trigger_behaviour(z: np.ndarray, fps: float, n: int) -> dict:
    sc = maxcell_score(z)
    trig = AdaptiveTrigger()
    for t in range(1, n):
        trig.update(t / fps, float(sc[t]), fps)
    clips = trig.finalize(n / fps)
    keep = np.zeros(n, dtype=bool)
    for a, b in clips:
        keep[int(a * fps):min(int(b * fps) + 1, n)] = True
    return {"stored": float(keep.mean()), "n_clips": len(clips)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="runs/real_imu_eval.json")
    ap.add_argument("--force", action="store_true",
                    help="允许小规模运行覆盖更大的已有结果（默认拒绝）")
    args = ap.parse_args()

    fm, grays, gyro, meta, n = load(args.seq, args.n)
    fps, f_map = float(meta["fps"]), float(meta["f_map"])
    cfg = CancelConfig(fps=fps, f_map=f_map, robust=True, source=args.seq)
    calib = meta.get("calib") or {}
    print(f"{args.seq}")
    print(f"  {n} 帧 @ {fps:.2f} fps | f_map {f_map:.2f} | IMU: "
          f"{meta.get('imu')}")
    print(f"  标定: 单位 {calib.get('unit_tag')}  FOV≈{calib.get('fov_deg')}°  "
          f"R²={calib.get('r2', float('nan')):.3f}")
    print(f"  特征图 {fm.shape} | 帧间隔 {meta['frame_dt_median'] * 1e3:.1f} ms"
          f"（抖动 p95 {meta['frame_dt_jitter_p95'] * 1e3:.1f} ms）")
    print(f"  gyro p95 [pitch,yaw,roll] = "
          f"{np.percentile(np.abs(gyro), 95, axis=0).round(1)} deg/s\n")

    report: dict = {
        "seq": args.seq, "n": n, "fps": fps, "f_map": f_map,
        "source": meta.get("source"), "imu": meta.get("imu"),
        "calib": calib,
        "note": "真机段无人工事件标注 -> 不报 recall；这是残差质量/成本对照",
        "methods": {},
    }

    runs: list[tuple[str, object, str]] = [
        ("naive", FeatureMapCancel(
            CancelConfig(**{**cfg.__dict__, "use_warp": False})), ""),
        ("gyro_shift", FeatureMapCancel(cfg), ""),
        ("gyro_rotation", FeatureMapCancel(
            CancelConfig(**{**cfg.__dict__, "pred": "rotation"})), ""),
        ("flow_dis", FlowWarpCancel(FlowWarpConfig()), "flow"),
    ]

    for name, c, kind in runs:
        t0 = time.perf_counter()
        if kind == "flow":
            assert isinstance(c, FlowWarpCancel)
            z, _ = rollout_flow(c, fm, grays)
        else:
            assert isinstance(c, FeatureMapCancel)
            z, _ = rollout(c, fm, gyro)
        ms = (time.perf_counter() - t0) / n * 1e3
        var = c.var
        assert var is not None
        # 逐帧原始残差（方法无关口径的基础）
        sd = np.sqrt(var) + 1e-3
        errs = np.abs(z * sd).mean(axis=(1, 2, 3))
        rec = {
            "full_frame": full_frame_residual(errs),
            "frame_diff_calm": frame_diff_residual(errs, fm),
            "published_calm": calm_residual(z, var),
            "ms_per_frame": ms,
        }
        rec.update(trigger_behaviour(z, fps, n))
        report["methods"][name] = rec
        print(f"  {name:14} full_frame={rec['full_frame']:.4f} "
              f"frame_diff_calm={rec['frame_diff_calm']:.4f}  "
              f"[已发表 calm={rec['published_calm']:.4f}]  "
              f"{ms:6.3f} ms/帧")

    m = report["methods"]
    nv = m["naive"]
    rel = {}
    for k in ("gyro_shift", "gyro_rotation", "flow_dis"):
        rel[k] = m[k]["full_frame"] / nv["full_frame"]

    # 机制量：每帧网格位移 p95，单位是**特征图格**（28 网格）。与
    # run_rotwarp_aoe.py 报同一个量，好让两个数据集的对比可被直接核验
    # （真机 2.790 格 vs AoE 0.596 格，4.7 倍——这是"同实现两个数据集收益
    # 差很多"的定量解释）。
    dt_f = 1.0 / fps
    _dx = f_map * np.tan(np.radians(gyro[:, 1]) * dt_f)
    _dy = f_map * np.tan(np.radians(gyro[:, 0]) * dt_f)
    _r = np.sqrt(2.0) * fm.shape[-1] / 2.0
    _dr = _r * np.radians(gyro[:, 2]) * dt_f
    grid_p95 = float(np.percentile(np.hypot(np.hypot(_dx, _dy), _dr), 95))

    report["summary"] = {
        "metric": "full_frame = 全帧原始残差均值（方法无关，主口径）",
        "ratios_full_frame": rel,
        "gyro_shift_vs_naive": rel["gyro_shift"],
        "gyro_rotation_vs_naive": rel["gyro_rotation"],
        "flow_vs_naive": rel["flow_dis"],
        "gyro_cost_x_flow": (m["flow_dis"]["ms_per_frame"]
                             / m["gyro_rotation"]["ms_per_frame"]),
        "rotation_wins": bool(rel["gyro_rotation"] < 1.0),
        # 与 run_rotwarp_aoe.py 同口径的机制量（见那里的判据说明）：
        # 只有每帧位移量级足够时，旋转模型的可表达性才兑现得出来。
        "grid_shift_p95": grid_p95,
        "verdict": (
            "完整旋转 warp 在真机数据上降低了残差；管线原有的 tan 近似常数"
            "平移则反而变差——即真 IMU 的负结果来自 warp 模型而非副本质量"
            if rel["gyro_rotation"] < 1.0 < rel["gyro_shift"] else
            "两种 gyro warp 都未降低残差"),
    }
    print("\n=== 结论（方法无关口径 full_frame）===")
    for k in ("gyro_shift", "gyro_rotation", "flow_dis"):
        print(f"  {k:14} /naive = {rel[k]:.3f}")
    print(f"  （flow 比 rotation 贵 {report['summary']['gyro_cost_x_flow']:.0f}x）")
    print(f"  每帧网格位移 p95 = {grid_p95:.3f} 格"
          f"（对照 AoE 0.596 格，比值 {grid_p95 / 0.596:.1f}x）")
    print(report["summary"]["verdict"])

    # 覆盖保护：小规模运行不许静默盖掉更大的正式结果。
    if os.path.exists(args.out) and not args.force:
        try:
            old = json.load(open(args.out))
            old_n = int(old.get("n", 0))
        except (OSError, ValueError):
            old_n = 0
        if n < old_n:
            raise SystemExit(
                f"拒绝覆盖 {args.out}：现有结果更大（{old_n} 帧 > 本次 {n} 帧）。"
                "确要覆盖请加 --force，或换个 --out 。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

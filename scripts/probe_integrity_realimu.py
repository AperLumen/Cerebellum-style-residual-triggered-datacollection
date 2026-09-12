"""在"IMU 真正有价值"的数据上验证故障检出——决定性交叉验证。

## 推导链

`scripts/diag_dropout_absorption.py` 查清了 IMU 丢包为何检不出，而且答案不是
"统计被吸收"：

* AoE 上，把 IMU 归零 = 预测器退化成朴素帧差；
* 而 AoE 上旋转 warp 相对 naive 只值 **0.977--0.994x**（`runs/rotwarp_aoe.json`）；
* 因此"丢掉 IMU"造成的残差上升被**上界锁死在 3--5%**，落在噪声里。

=> **故障检出能力有一个理论上界：IMU 自身在该数据上的边际贡献。** 这不是
工程调参问题，而是信息问题——如果一段数据里的 IMU 本来就没什么用，那么
"它坏了"和"它没坏"在这段数据上不可分辨。

## 可验证预测

真机陀螺仪段（`data/real_imu`，Galaxy S21 硬件 gyro）上，旋转 warp 相对 naive
值 **0.84x**（`runs/real_imu_eval.json`）——即 IMU 在那里有 ~16% 的边际贡献。
预测：**同一套注入，在真机段上应当可检出，而在 AoE 上不可检出。**

这是对本项目整体叙事的一个交叉验证：残差的价值与 IMU 的价值同源。

## 做法

同一注入协议（IMU 归零 / 时移 / 尖峰 / 帧重复），跑在真机段上。真机段没有
预计算的特征图，用 `run_real_imu_eval.load()`（与已发布数字同一前端）。
真机段无人工事件标注，因此只报"故障段 vs 干净同段"的比值，不报标注相关指标。

输出：runs/integrity_fault_realimu.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel)
from cerebellum_cancel.io import load_imu_csv


def load_real(seq: str, n: int):
    """读真机段的帧 -> 特征图（与 run_real_imu_eval.py 同一前端）。"""
    import glob
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
    fmaps = []
    with torch.no_grad():
        for p in paths[:n]:
            img = Image.open(p).convert("RGB")
            x = cast(torch.Tensor, tf(img))
            fmaps.append(model(x.unsqueeze(0).to(dev))[0].cpu().numpy()
                         .astype(np.float32))
    return np.stack(fmaps), gyro[:n], meta, n


def surprise_series(F, gy: np.ndarray, n: int, fps: float, f_map: float,
                    seq: str) -> np.ndarray:
    canc = FeatureMapCancel(CancelConfig(fps=fps, f_map=f_map, robust=True,
                                         pred="rotation", source=seq))
    s = np.zeros(n, dtype=np.float64)
    for t in range(n):
        s[t] = canc.step(np.asarray(F[t], dtype=np.float32), gy[t])["surprise"]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/real_imu/val/seq_000")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/integrity_fault_realimu.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} 已存在（--force 覆盖）")

    rng = np.random.default_rng(args.seed)
    F, gy, meta, n = load_real(args.seq, args.n)
    fps = float(meta["fps"])
    f_map = float(meta["f_map"])
    lo = 40
    evals = np.arange(lo, n)

    print(f"{args.seq}  n={n}  fps={fps:.2f}  f_map={f_map:.2f}")
    print("（真机段：旋转 warp 相对 naive 值 0.84x —— 即 IMU 有 ~16% 边际贡献）\n")

    s_clean = surprise_series(F, gy, n, fps, f_map, args.seq)
    res_clean = float(np.mean(s_clean[evals]))

    # 基线：完全不用 IMU（naive）——用来计算"IMU 的边际贡献"
    from cerebellum_cancel.featmap_cancel import shift_map  # noqa: F401
    naive = np.zeros(n, dtype=np.float64)
    for t in range(n):
        naive[t] = float(np.abs(np.asarray(F[t], dtype=np.float32)
                                - np.asarray(F[t - 1], dtype=np.float32)).mean())
    print(f"  干净 surprise 均值          = {res_clean:.3f}")
    print(f"  （naive 帧差残差均值        = "
          f"{float(naive[evals].mean()):.4f}，供参照）\n")

    out: dict = {"seq": args.seq, "n": n, "fps": fps, "f_map": f_map,
                 "faults": {}}

    # --- 段级：IMU 归零（丢包）与时移 ---
    for kind in ("imu_zero", "imu_desync"):
        ratios = []
        for _ in range(args.reps):
            g = gy.copy()
            if kind == "imu_zero":
                g[:] = 0.0                        # 整段 IMU 失效
            else:
                g = np.roll(g, 3, axis=0)         # 整段错位 3 帧
            s = surprise_series(F, g, n, fps, f_map, args.seq)
            ratios.append(float(np.mean(s[evals]) / max(res_clean, 1e-9)))
        out["faults"][kind] = {"ratio_mean": float(np.mean(ratios)),
                               "ratio_max": float(np.max(ratios)),
                               "n_reps": len(ratios)}
        print(f"  {kind:14s} 故障/干净 = {np.mean(ratios):5.2f}x")

    # --- 帧级：局部故障 ---
    from probe_integrity_faults import lift
    for kind in ("imu_local_zero", "frame_dup", "imu_spike"):
        vals = []
        for _ in range(args.reps):
            span = int(rng.integers(20, 50))
            a = int(rng.integers(lo, max(n - span, lo + 1)))
            b = min(a + span, n)
            g = gy.copy()
            Fx = F.copy()
            if kind == "imu_local_zero":
                g[a:b] = 0.0
            elif kind == "frame_dup":
                Fx[a:b] = Fx[a - 1]
            else:
                g[a:b] += rng.normal(0, 300.0, size=(b - a, 3)).astype(np.float32)
            s = surprise_series(Fx, g, n, fps, f_map, args.seq)
            vals.append(lift(s, np.arange(a, b), evals))
        out["faults"][kind] = {"lift_mean": float(np.nanmean(vals)),
                              "lift_max": float(np.nanmax(vals))}
        print(f"  {kind:14s} lift = {np.nanmean(vals):6.2f}x  "
              f"最大 {np.nanmax(vals):6.2f}x")

    out["note"] = ("验证'故障检出上界 = IMU 边际贡献'这个推导：真机段上 IMU 值 "
                   "0.84x，故应当可检出；AoE 上只值 0.98x，故不可检出。")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

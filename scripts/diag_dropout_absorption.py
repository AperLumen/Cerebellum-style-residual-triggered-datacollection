"""诊断：为什么 IMU 归零（丢包）没被残差检出？

## 背景

`scripts/probe_integrity_faults.py` 实测：

    imu_spike    lift = 10.92x   (10/10 段)   <- 检出
    imu_dropout  lift =  0.98x   (0/10 段)    <- 未检出
    frame_dup    lift =  0.09x   (0/10 段)    <- 未检出

`imu_dropout` 未检出是**反直觉**的：gyro 归零 -> `warp_by_rotation` 退化为
恒等变换 -> pred = F_{t-1} -> 残差变成朴素帧差，理应变大。而朴素帧差在旋转
场景下的残差**明显大于**旋转补偿后的残差（项目早期就实测过：合成 rig 上
naive 0.221 vs gyro 0.130）。

所以要么 (a) 注入方式有误，要么 (b) 在线统计把故障**吸收**了。后者若成立，
是一个必须记录的**结构性限制**：`robust=True` 的 winsorized always-update
会主动适配偶发的高残差，因此**持续数十帧的故障会被自适应掉**——这正好是
"不明显的错误"这一类目标的主场。

## 本脚本

直接打印逐帧 trace：residual（原始）、sd（运行尺度）、surprise（z 分数）。
三者对照即可区分 (a) 与 (b)：

* 若 residual 确实升高、但 sd 也跟着升高 -> (b) 自适应吸收。
* 若 residual 根本没升高 -> (a) 注入或 warp 逻辑有误。

输出：runs/diag_dropout_absorption.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")
sys.path.insert(0, ".")

from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel)
from cerebellum_cancel.io import load_imu_csv


def trace(F, gy: np.ndarray, n: int, fps: float, f_map: float, seq: str,
          robust: bool, span: tuple[int, int]) -> dict:
    """返回 residual / sd / surprise 三条逐帧序列。"""
    cfg = CancelConfig(fps=fps, f_map=f_map, robust=robust, pred="rotation",
                       source=seq)
    canc = FeatureMapCancel(cfg)
    res = np.zeros(n, dtype=np.float64)
    sd = np.zeros(n, dtype=np.float64)
    sur = np.zeros(n, dtype=np.float64)
    a, b = span
    for t in range(n):
        Fp = np.asarray(F[t], dtype=np.float32)
        out = canc.step(Fp, gy[t])
        sd_t = np.sqrt(np.asarray(canc.var)) + 1e-3
        res[t] = float(np.abs(out["z"] * sd_t).mean())
        sd[t] = float(sd_t.mean())
        sur[t] = out["surprise"]
    return {"residual": res, "sd": sd, "surprise": sur, "span": [a, b]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/aoe_event/val/seq_000")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--span", type=int, default=40)
    ap.add_argument("--out", default="runs/diag_dropout_absorption.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} 已存在（加 --force 覆盖）")

    F = np.load(f"{args.seq}/featmaps.npy", mmap_mode="r")
    gy = np.array([g.gyro for g in load_imu_csv(f"{args.seq}/imu.csv")],
                  dtype=np.float32)
    meta = json.load(open(f"{args.seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    f_map = float(meta["f_map"])
    n = min(len(F), len(gy), args.n)
    a, b = args.start, min(args.start + args.span, n)

    gy_f = gy[:n].copy()
    gy_f[a:b] = 0.0            # 注入：IMU 丢包

    print(f"{args.seq}  n={n}  注入区间 [{a},{b})（{b - a} 帧，"
          f"{(b - a) / fps:.2f}s）\n")
    out: dict = {"seq": args.seq, "n": n, "span": [a, b], "fps": fps,
                 "runs": {}}
    for tag, robust in (("robust=True（项目当前配置）", True),
                        ("robust=False（门控 EMA）", False)):
        clean = trace(F, gy[:n], n, fps, f_map, args.seq, robust, (a, b))
        fault = trace(F, gy_f, n, fps, f_map, args.seq, robust, (a, b))
        sl = slice(max(a - 5, 0), min(b + 25, n))
        seg = slice(a, b)
        r_in = float(fault["residual"][seg].mean())
        r_before = float(fault["residual"][max(a - 50, 0):a].mean())
        r_clean = float(clean["residual"][seg].mean())
        sd_in = float(fault["sd"][seg].mean())
        sd_before = float(fault["sd"][max(a - 50, 0):a].mean())
        su_in = float(fault["surprise"][seg].mean())
        su_clean = float(clean["surprise"][seg].mean())
        ratio = su_in / max(su_clean, 1e-9)
        print(f"--- {tag} ---")
        print(f"  故障段 residual: {r_in:.4f}   （故障前 {r_before:.4f}，"
              f"干净同段 {r_clean:.4f}）")
        print(f"  故障段 sd      : {sd_in:.4f}   （故障前 {sd_before:.4f}）"
              f"   -> sd 上升 {sd_in / max(sd_before, 1e-9):.2f}x")
        print(f"  故障段 surprise: {su_in:.3f}  vs 干净 {su_clean:.3f}"
              f"   -> 比值 {ratio:.2f}x")
        verdict = ("residual 升高但被 sd 吞掉 -> 在线统计吸收"
                   if r_in > 1.3 * r_clean and ratio < 1.5
                   else ("检出" if ratio >= 1.5 else "未检出"))
        print(f"  判定: {verdict}\n")
        out["runs"][tag] = {
            "residual_in": r_in, "residual_before": r_before,
            "residual_clean_same_span": r_clean,
            "sd_in": sd_in, "sd_before": sd_before,
            "sd_growth": sd_in / max(sd_before, 1e-9),
            "surprise_in": su_in, "surprise_clean": su_clean,
            "surprise_ratio": ratio, "verdict": verdict,
            "trace_around": {
                "t": list(range(max(a - 5, 0), min(b + 25, n))),
                "residual_faulted": [float(x) for x in fault["residual"][sl]],
                "residual_clean": [float(x) for x in clean["residual"][sl]],
                "sd_faulted": [float(x) for x in fault["sd"][sl]],
                "surprise_faulted": [float(x) for x in fault["surprise"][sl]],
                "surprise_clean": [float(x) for x in clean["surprise"][sl]],
            },
        }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

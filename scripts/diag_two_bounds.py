"""分解故障检出的两个上界：信号上界 vs 自适应上界。

## 现状

两轮实验都没检出 IMU 归零：

    AoE       (IMU 值 0.98x)  ->   0.98x
    真机段    (IMU 值 0.84x)  ->   1.04x   <- 预测应可检出，被证伪

第二个结果是关键：真机段上 IMU 有 ~16% 边际贡献，**把 IMU 归零本该让原始
残差上升约 20%**（naive 0.3945 vs rotation 0.3295）。但 surprise 只动 4%。
这说明原始信号确实变了，是 **z 分数化把它吃掉了**。

所以"能不能检出"至少受两个独立上界约束：

1. **信号上界**：raw 残差的变化 = IMU 在该数据上的边际贡献。IMU 无用 ->
   归零无害 -> 信息上不可分辨。这是数据决定的，改算法无解。
2. **自适应上界**：`surprise = |err| / sd`，而 sd 是**在线更新**的。一个持续
   几十帧的故障会被 winsorized always-update 学进 sd，z 分数被拉回 ~1。这是
   **算法**决定的，可以改（例如用冻结参考 sd，或改检测统计量）。

## 本脚本

对同一份注入，同时报：
* `raw_ratio`      —— 原始残差变化（信号上界）
* `surprise_ratio` —— 当前在线管线的读数（两个上界叠加）
* `frozen_z`       —— 用**前 200 帧冻结**的 sd 重算 z 分数（去掉自适应上界）

三者对照即可定位瓶颈。若 frozen_z ≫ surprise_ratio，则瓶颈是自适应，且
"检测完整性故障"这件事在改统计量之后是可行的。

输出：runs/diag_two_bounds.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")


def series(F, gy: np.ndarray, n: int, fps: float, f_map: float, seq: str,
           freeze_at: int) -> dict:
    """返回 raw 残差、在线 surprise、以及冻结基准 z 分数三条序列。

    通道数/网格尺寸从数据推断（AoE 是 24x28x28，真机段也是 24 通道，
    但不要把尺寸写死——写死会在换前端时静默算错）。
    """
    from cerebellum_cancel.featmap_cancel import (CancelConfig,
                                                  FeatureMapCancel)
    canc = FeatureMapCancel(CancelConfig(fps=fps, f_map=f_map, robust=True,
                                         pred="rotation", source=seq))
    raw = np.zeros(n, dtype=np.float64)
    sur = np.zeros(n, dtype=np.float64)
    first = np.asarray(F[0], dtype=np.float32)
    errs = np.zeros((n,) + first.shape, dtype=np.float32)
    for t in range(n):
        Fp = np.asarray(F[t], dtype=np.float32)
        out = canc.step(Fp, gy[t])
        sd = np.sqrt(np.asarray(canc.var)) + 1e-3
        raw[t] = float(np.abs(out["z"] * sd).mean())
        sur[t] = out["surprise"]
        errs[t] = np.asarray(out["z"], dtype=np.float32) * sd
    # 冻结基准：用前 freeze_at 帧的 err 估一个固定 mu/sd
    fz = np.zeros(n, dtype=np.float64)
    ref = errs[:freeze_at]
    mu_f = ref.mean(axis=0)
    sd_f = ref.std(axis=0) + 1e-3
    bnd = canc.cfg.border
    for t in range(n):
        z = (errs[t] - mu_f) / sd_f
        fz[t] = float(np.abs(z[:, bnd:-bnd, bnd:-bnd]).max())
    return {"raw": raw, "surprise": sur, "frozen_z": fz}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoe", default="data/aoe_event/val/seq_000")
    ap.add_argument("--realimu", default="data/real_imu/val/seq_000")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--freeze-at", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/diag_two_bounds.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out: dict = {"freeze_at": args.freeze_at, "datasets": {}}

    for tag, seqpath, loader in (
            ("aoe", args.aoe, "featmaps"),
            ("real_imu", args.realimu, "frames")):
        if not os.path.exists(seqpath):
            continue
        if loader == "featmaps":
            F = np.load(f"{seqpath}/featmaps.npy", mmap_mode="r")
            from cerebellum_cancel.io import load_imu_csv
            gy = np.array([g.gyro for g in load_imu_csv(f"{seqpath}/imu.csv")],
                          dtype=np.float32)
            meta = json.load(open(f"{seqpath}/events.json"))
            n = min(len(F), len(gy), args.n)
            F = np.asarray(F[:n], dtype=np.float32)
        else:
            from probe_integrity_realimu import load_real
            F, gy, meta, n = load_real(seqpath, args.n)
        fps = float(meta.get("fps", 30.0))
        f_map = float(meta["f_map"])

        base = series(F, gy, n, fps, f_map, seqpath, args.freeze_at)
        a, b = args.freeze_at + 40, min(args.freeze_at + 100, n)
        span = slice(a, b)

        gz = gy.copy()
        gz[a:b] = 0.0
        fz = series(F, gz, n, fps, f_map, seqpath, args.freeze_at)

        def rat(x, y, sl):
            return float(np.mean(y[sl]) / max(np.mean(x[sl]), 1e-9))

        res = {
            "n": n, "span": [a, b],
            "raw_ratio": rat(base["raw"], fz["raw"], span),
            "surprise_ratio": rat(base["surprise"], fz["surprise"], span),
            "frozen_z_ratio": rat(base["frozen_z"], fz["frozen_z"], span),
        }
        out["datasets"][tag] = res
        print(f"\n=== {tag} ({seqpath})  注入 [{a},{b}) n={b - a} 帧 ===")
        print(f"  原始残差变化      raw_ratio        = {res['raw_ratio']:5.2f}x"
              f"   <- 信号上界（数据决定）")
        print(f"  在线 surprise     surprise_ratio   = {res['surprise_ratio']:5.2f}x"
              f"   <- 当前管线实际读数")
        print(f"  冻结基准 z 分数   frozen_z_ratio   = {res['frozen_z_ratio']:5.2f}x"
              f"   <- 去掉自适应上界后")

    print("\n判读：")
    print("  raw_ratio ≈ 1        -> 信息上不可分辨（IMU 在该数据上没用）")
    print("  raw ≫1 但 surprise≈1 -> 自适应统计把信号吃掉了（算法可修）")
    print("  frozen_z ≫ surprise  -> 瓶颈是自适应，改用冻结基准即可检出")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

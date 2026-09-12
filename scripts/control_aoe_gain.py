"""AoE 上 rotation 补偿的"收益"到底来自哪里——决定性对照实验。

## 为什么需要这个脚本

`scripts/analyze_negatives.py`（N2 消融）报 all-three 旋转补偿 **+4.65%**
（12/12 段更好）。但 `scripts/diag_aoe_window.py` 发现两件事：

1. AoE 的每帧内容位移只有 **0.05-0.4 px**（特征图 28×28 上）。真机段是
   2.818 格（≈ 同量级 px），差 5.8 倍。亚像素位移下，任何运动补偿模型都
   无从施展——它连"要补多少"都比噪声小。
2. gyro 预测位移与**相位相关实测位移**的相关系数几乎全是零或负值
   （−0.27 … +0.06）。也就是说在这份数据上 **gyro 对帧间位移没有解释力**。

若 gyro 无解释力、位移又在亚像素，那么 +4.65% 必然来自别的机制。最可能的
嫌疑是**插值平滑**：双线性 warp 在亚像素偏移下等价于一个轻微低通滤波，而
低通能压低像素噪声，从而降低 |F_t − pred| —— 这与"预测了运动"无关。

## 对照设计

对同一批帧（同窗口、同内区口径），并排测：

* ``naive``             —— 不补偿，基线
* ``rotation_gyro``     —— 被检验的模型（gyro 驱动，正确符号）
* ``rotation_shuffled`` —— **把 gyro 在时间轴上打乱**：位移幅度分布完全保留，
                            时间对应被破坏。若收益仍在，说明收益不来自时序
                            预测。
* ``rotation_signflip`` —— 把所有轴符号取反：方向反了。若收益仍在，说明收益
                            不来自方向正确的补偿。
* ``blur{σ}``           —— **纯高斯模糊**，完全不用 IMU。若某个 σ 下的模糊能
                            复现同等收益，则"收益"是平滑效应。

判据：如果 shuffled / signflip / blur 都能复现 rotation_gyro 的收益，那么这个
收益**不是运动补偿**，AoE 上的"正结果"不成立。

输出：runs/aoe_gain_control.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from cerebellum_cancel.io import load_imu_csv                  # noqa: E402
from cerebellum_cancel.rotwarp import rotation_homography      # noqa: E402
from analyze_negatives import mean_abs_err, FPS                # noqa: E402

# gy 的通道顺序是 [pitch, yaw, roll]；AoE 的标定结果是 (pitch+1, yaw-1, roll+1)
SIGNS = np.array([1.0, -1.0, 1.0])
BLUR_SIGMAS = (0.25, 0.5, 0.75, 1.0)
MAX_FRAMES = 1500   # 与 analyze_negatives.MAX_FRAMES_ABLATION 对齐


def warp_h(Fp: np.ndarray, g: np.ndarray, f_map: float,
           dt: float) -> np.ndarray:
    """按旋转单应 warp。**必须带 WARP_INVERSE_MAP**——见 rotwarp 的说明。"""
    ch, h, w = Fp.shape
    H = rotation_homography(float(g[0]), float(g[1]), float(g[2]), dt, f_map,
                            w / 2.0, h / 2.0)
    return np.stack([cv2.warpPerspective(Fp[c], H, (w, h),
                                         flags=cv2.INTER_LINEAR
                                         | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)
                     for c in range(ch)])


def eval_seq(F, gy: np.ndarray, f_map: float, dt: float,
             ev: np.ndarray, sigmas, rng) -> dict[str, float]:
    """返回 {模型名: 内区平均残差}。"""
    acc: dict[str, list[float]] = {}

    def add(k: str, v: float) -> None:
        acc.setdefault(k, []).append(v)

    perm = rng.permutation(len(gy))
    sgnflip = -SIGNS
    for t in ev:
        Ft = np.asarray(F[t], dtype=np.float32)
        Fp = np.asarray(F[t - 1], dtype=np.float32)
        add("naive", mean_abs_err(Fp, Ft))
        add("rotation_gyro",
            mean_abs_err(warp_h(Fp, gy[t] * SIGNS, f_map, dt), Ft))
        add("rotation_shuffled",
            mean_abs_err(warp_h(Fp, gy[perm[t]] * SIGNS, f_map, dt), Ft))
        add("rotation_signflip",
            mean_abs_err(warp_h(Fp, gy[t] * sgnflip, f_map, dt), Ft))
        for s in sigmas:
            add(f"blur{s:g}",
                mean_abs_err(cv2.GaussianBlur(Fp, (0, 0), s), Ft))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def dedup_key(seq: str) -> tuple:
    """判重：同一段数据被挂在不同目录下（实测 aoe_event/seq_007 与
    aoe_motion/seq_001 逐位相同），重复计入会虚增样本量。"""
    fm = f"{seq}/featmaps.npy"
    h = hashlib.md5(open(f"{seq}/imu.csv", "rb").read()).hexdigest()
    return (os.path.getsize(fm), h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_motion,data/aoe_event")
    ap.add_argument("--n", type=int, default=MAX_FRAMES)
    ap.add_argument("--out", default="runs/aoe_gain_control.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = []
    for st in args.sets.split(","):
        paths += sorted(glob.glob(f"{st.strip()}/val/seq_*"))

    seqs, seen = [], set()
    for s in paths:
        k = dedup_key(s)
        if k in seen:
            print(f"  [跳过重复段] {s}")
            continue
        seen.add(k)
        seqs.append(s)

    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out))
        if len(old.get("seqs", [])) > len(seqs):
            raise SystemExit(f"拒绝覆盖 {args.out}：现有结果段数更多。"
                             "确要覆盖请加 --force。")

    rng = np.random.default_rng(0)
    rows = []
    for s in seqs:
        if not os.path.exists(f"{s}/featmaps.npy"):
            continue
        F = np.load(f"{s}/featmaps.npy", mmap_mode="r")
        gy = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                      dtype=np.float32)
        meta = json.load(open(f"{s}/events.json"))
        f_map = float(meta["f_map"])
        n = min(len(F), len(gy), args.n)
        # 后半评估（与 analyze_negatives 同窗口）
        sp = n // 2
        ev = np.linspace(sp, n - 1, n - sp).astype(int)
        err = eval_seq(F, gy, f_map, 1.0 / FPS, ev, BLUR_SIGMAS, rng)
        base = err["naive"]
        gains = {k: 1.0 - v / base for k, v in err.items() if k != "naive"}
        rows.append({"seq": s, "n": int(n), "f_map": f_map,
                     "gains": gains, "errs": err})
        print(f"  {s.split('/')[-1]:9s}[{s.split('/')[1][4:]:6s}] "
              f"gyro {gains['rotation_gyro'] * 100:+6.2f}%  "
              f"shuffled {gains['rotation_shuffled'] * 100:+6.2f}%  "
              f"signflip {gains['rotation_signflip'] * 100:+6.2f}%  "
              + "  ".join(f"blur{s_:g} {gains[f'blur{s_:g}'] * 100:+5.2f}%"
                          for s_ in BLUR_SIGMAS))

    keys = list(rows[0]["gains"].keys()) if rows else []
    summary = {}
    for k in keys:
        v = np.array([r["gains"][k] for r in rows])
        summary[k] = {"mean_gain": float(v.mean()), "n": len(v),
                      "better": int((v > 0).sum())}

    print(f"\n=== 汇总（{len(rows)} 段，后半窗口，内区口径）===")
    print(f"{'模型':<20}{'均值增益':>10}{'更好段数':>12}")
    for k in keys:
        d = summary[k]
        print(f"{k:<20}{d['mean_gain'] * 100:>9.2f}%"
              f"{d['better']:>8}/{d['n']}")
    print("\n判读：若 shuffled / signflip / blur 与 gyro 同量级，则收益不是运动补偿。")

    report = {
        "sets": args.sets, "n_frames_per_seq": args.n,
        "window": "每段后一半（与 analyze_negatives 对齐）",
        "metric": "内区 mean|err|（BORDER=2）",
        "signs": SIGNS.tolist(),
        "blur_sigmas": list(BLUR_SIGMAS),
        "note": "rotation_shuffled 把 gyro 在时间轴上打乱（幅度分布保留、时序"
                "破坏）；blur 完全不用 IMU。两者若与 rotation_gyro 同量级，"
                "说明 AoE 上的收益来自插值平滑而非运动补偿。",
        "summary": summary,
        "seqs": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

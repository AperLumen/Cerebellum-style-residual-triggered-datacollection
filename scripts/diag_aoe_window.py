"""AoE 上 rotation 补偿的收益为何依赖评估窗口——机制诊断。

背景（见 MEMORY_FORMAT §8.6）：`scripts/analyze_negatives.py` 报 all-three
+4.65%（12/12 段更好），而 `scripts/run_rotwarp_aoe.py` 在**同一份数据**上
只报 +2.06%（9/12）。两者实现已逐位对齐（复现偏差 <0.35 个百分点），差异
唯一来源是**评估窗口**：

    全序列前 600 帧   +2.06%   9/12
    前 1500 帧的后半  +4.55%  11/12

本脚本查清"为什么"——把收益按序列内时间分块，并同时测**该块的 gyro 是否
可靠**（用相位相关测出的真实帧间位移作为参照，算 gyro 预测位移与它的相
关）。若 gyro 在序列开头不可靠，则开头收益低就是"输入坏了"而非"模型没用"。

输出：runs/aoe_window_profile.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from cerebellum_cancel.io import load_imu_csv          # noqa: E402
from cerebellum_cancel.rotwarp import rotation_homography  # noqa: E402
from analyze_negatives import mean_abs_err, FPS        # noqa: E402

# gy 的顺序是 [pitch, yaw, roll]；AoE 的标定结果（见 rotwarp.DATASET_CONVENTIONS）
SIGNS = np.array([1.0, -1.0, 1.0])

BG = 2      # 内区边界（与管线读数一致）
BLOCK = 250  # 分块长度（帧）


def warp_h(Fp: np.ndarray, g: np.ndarray, f_map: float) -> np.ndarray:
    """按旋转单应 warp。必须带 WARP_INVERSE_MAP —— 见 rotwarp 的说明。"""
    _, h, w = Fp.shape
    H = rotation_homography(float(g[0]), float(g[1]), float(g[2]), 1.0 / FPS,
                            f_map, w / 2.0, h / 2.0)
    return np.stack([cv2.warpPerspective(Fp[c], H, (w, h),
                                         flags=cv2.INTER_LINEAR
                                         | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)
                     for c in range(Fp.shape[0])])


def measured_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """用相位相关测帧间内容位移（特征图通道平均后的 2D）。

    这是 gyro 的**独立参照**：它只看像素，不看 IMU。返回 (dx, dy) 像素。
    """
    a = prev.mean(0).astype(np.float32)
    b = cur.mean(0).astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    return float(dx), float(dy)


def profile_seq(seq: str, n_max: int, block: int) -> dict | None:
    fm_path = f"{seq}/featmaps.npy"
    if not os.path.exists(fm_path):
        return None
    F = np.load(fm_path, mmap_mode="r")
    gy = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                  dtype=np.float32)
    f_map = float(json.load(open(f"{seq}/events.json"))["f_map"])
    n = min(len(F), len(gy), n_max)

    blocks = []
    for a in range(1, n, block):
        b = min(a + block, n)
        if b - a < 30:
            continue
        gain_n, gain_r = [], []
        gx_meas, gx_pred = [], []
        for t in range(a, b):
            Ft = np.asarray(F[t], dtype=np.float32)
            Fp = np.asarray(F[t - 1], dtype=np.float32)
            # 朴素帧差 = A；rotation 补偿 = B
            gain_n.append(mean_abs_err(Fp, Ft))
            g = gy[t] * SIGNS
            gain_r.append(mean_abs_err(warp_h(Fp, g, f_map), Ft))
            # gyro 可靠性：预测位移 vs 实测位移
            r = np.radians(g) / FPS
            gx_pred.append((f_map * np.tan(r[1]), f_map * np.tan(r[0])))
            gx_meas.append(measured_shift(Fp, Ft))
        A = float(np.mean(gain_n))
        B = float(np.mean(gain_r))
        pm = np.array(gx_pred)                      # 预测 (dx, dy)
        mm = np.array(gx_meas)                      # 实测 (dx, dy)
        # 相关性：gyro 预测的位移方向/大小是否与实测一致
        corrs = []
        for k in (0, 1):
            if pm[:, k].std() > 1e-9 and mm[:, k].std() > 1e-9:
                corrs.append(float(np.corrcoef(pm[:, k], mm[:, k])[0, 1]))
        # 实测位移幅度（px），去掉 DC 后的 std：反映"这一块动得厉害吗"
        blocks.append({
            "t0": a, "t1": b - 1, "n": b - a,
            "gain": 1.0 - B / A,
            "naive": A, "rotation": B,
            "pred_shift_mean": float(np.hypot(pm[:, 0], pm[:, 1]).mean()),
            "meas_shift_mean": float(np.hypot(mm[:, 0], mm[:, 1]).mean()),
            "meas_shift_std": float(np.hypot(mm[:, 0], mm[:, 1]).std()),
            "gyro_corr_mean": float(np.mean(corrs)) if corrs else None,
            "gyro_p95": float(np.percentile(np.linalg.norm(gy[a:b], axis=1),
                                            95)),
        })
    return {"seq": seq, "n": int(n), "f_map": f_map, "blocks": blocks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_motion,data/aoe_event")
    ap.add_argument("--n", type=int, default=1500,
                    help="每段最多用多少帧（与 analyze_negatives 的 "
                         "MAX_FRAMES_ABLATION 对齐）")
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--out", default="runs/aoe_window_profile.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seqs = []
    for st in args.sets.split(","):
        seqs += sorted(glob.glob(f"{st.strip()}/val/seq_*"))

    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out))
        if len(old.get("seqs", [])) > len(seqs):
            raise SystemExit(f"拒绝覆盖 {args.out}：现有结果段数更多。"
                             "确要覆盖请加 --force。")

    out: dict = {
        "sets": args.sets, "n_frames_per_seq": args.n, "block": args.block,
        "metric": "内区 mean|err|（与管线读数/analyze_negatives 同口径）",
        "note": "gain = 1 - rotation/naive；gyro_corr 是 gyro 预测位移与相位"
                "相关实测位移的逐轴相关系数（gyro 可靠性的独立参照）。",
        "seqs": [],
    }
    for s in seqs:
        r = profile_seq(s, args.n, args.block)
        if r is None:
            continue
        out["seqs"].append(r)
        print(f"\n{s}")
        for bl in r["blocks"]:
            print(f"  t {bl['t0']:5d}-{bl['t1']:5d}  gain {bl['gain'] * 100:+6.2f}%"
                  f"  预测位移 {bl['pred_shift_mean']:5.2f}px"
                  f"  实测 {bl['meas_shift_mean']:5.2f}px"
                  f"  gyro_corr {bl['gyro_corr_mean']:+.3f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

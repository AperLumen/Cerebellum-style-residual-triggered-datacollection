"""AoE 上 gyro 的"解释力"终极裁判：穷举最优逐帧位移（oracle）对照。

## 为什么需要它

`scripts/control_aoe_gain.py` 的对照已经证明 AoE 的 rotation 收益**不是**
插值平滑：时序打乱（−6.26%）和方向取反（−6.53%）都无法复现 +4.46%，而正
向能——**正反不对称本身就说明方向系统性正确**（若 warp 与内容无关，正反两
向必然同分）。

但 `scripts/diag_aoe_window.py` 用相位相关测得的 gyro 相关系数却≈0 或为负。
两者矛盾。相位相关在 28×28 特征图、0.2 px 位移下**信噪比极低**（估计器噪声
底就约 0.2 px），不足以当裁判。

## 本脚本的做法

不依赖任何估计器。对每一帧**穷举**内容平移 (sx, sy)，找出真正让内区残差最
小的那个位移——这是"纯平移模型能达到的上界"（oracle）。然后直接问三件事：

1. **oracle 上界有多高？** —— 若 oracle 也只有 +5%，说明这份数据能给的就是
   这么多，gyro 不是瓶颈。
2. **gyro 预测的位移与 oracle 位移一致吗？**（逐轴相关 + 幅度比）——
   这才是 gyro 解释力的无偏测量。
3. **gyro 的实际增益离 oracle 多远？** —— 差距就是模型（纯平移 vs 旋转单应）
   和读数噪声吃掉的部分。

同时按序列内时间分窗报告——上一轮的窗口依赖（前半 +2% / 后半 +5%）在这里
应该能直接看到根因（是 gyro 在开头不准，还是内容在开头就不动）。

输出：runs/aoe_oracle_shift.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from cerebellum_cancel.featmap_cancel import shift_map           # noqa: E402
from cerebellum_cancel.io import load_imu_csv                    # noqa: E402
from cerebellum_cancel.rotwarp import rotation_homography        # noqa: E402
from analyze_negatives import FPS                                # noqa: E402

import cv2                                                       # noqa: E402

# gy 顺序 [pitch, yaw, roll]；AoE 标定值（见 rotwarp.DATASET_CONVENTIONS）
SIGNS = np.array([1.0, -1.0, 1.0])
WINDOWS = ((1, 500), (500, 1000), (1000, 1500))
SUB = 8          # 每窗抽样：每 SUB 帧取一帧（穷举很贵）
SEARCH = 2.0     # 平移搜索半径（特征图像素）
STEP = 0.25      # 平移搜索步长


def interior(a: np.ndarray, m: int = 5) -> np.ndarray:
    """内区切片——留出比搜索半径更大的边距，避免 np.roll 的环绕污染。"""
    return a[:, m:-m, m:-m]


def oracle_shift(Fp: np.ndarray, Ft: np.ndarray, margin: int = 5
                 ) -> tuple[float, float, float]:
    """穷举 (sx, sy) 求内区残差最小的纯平移，返回 (sx, sy, 残差)。

    残差口径与管线一致（内区 mean|err|），边距取 margin 以覆盖搜索范围。
    """
    tgt = interior(Ft, margin)
    lo, hi = -SEARCH, SEARCH
    grid = np.arange(lo, hi + 1e-9, STEP)
    best = (0.0, 0.0, float("inf"))
    for sx in grid:
        for sy in grid:
            r = float(np.abs(tgt - interior(shift_map(Fp, sx, sy), margin)).mean())
            if r < best[2]:
                best = (float(sx), float(sy), r)
    return best


def warp_rot(Fp: np.ndarray, g: np.ndarray, f_map: float) -> np.ndarray:
    """旋转单应 warp（带 WARP_INVERSE_MAP，见 rotwarp 的说明）。"""
    ch, h, w = Fp.shape
    H = rotation_homography(float(g[0]), float(g[1]), float(g[2]), 1.0 / FPS,
                            f_map, w / 2.0, h / 2.0)
    return np.stack([cv2.warpPerspective(Fp[c], H, (w, h),
                                         flags=cv2.INTER_LINEAR
                                         | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)
                     for c in range(ch)])


def dedup_key(seq: str) -> tuple:
    """判重：实测 aoe_event/seq_007 与 aoe_motion/seq_001 逐位相同。"""
    fm = f"{seq}/featmaps.npy"
    return (os.path.getsize(fm),
            hashlib.md5(open(f"{seq}/imu.csv", "rb").read()).hexdigest())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_motion,data/aoe_event")
    ap.add_argument("--out", default="runs/aoe_oracle_shift.json")
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
            raise SystemExit(f"拒绝覆盖 {args.out}：现有段数更多。加 --force 覆盖。")

    out: dict = {
        "sets": args.sets, "windows": WINDOWS, "subsample": SUB,
        "search_radius_px": SEARCH, "search_step_px": STEP,
        "metric": "内区 mean|err|（margin=5，覆盖搜索半径）",
        "note": "oracle_shift 是逐帧穷举出的最优纯平移；gyro_applied 是 gyro 经"
                "本项目的 tan 近似给出的内容位移（即 expected_shift 取负后真正"
                "施加的量）。两者相关=gyro 解释力；两者比值=尺度是否对。",
        "seqs": [],
    }

    agg: dict[tuple, list] = {}
    for s in seqs:
        if not os.path.exists(f"{s}/featmaps.npy"):
            continue
        F = np.load(f"{s}/featmaps.npy", mmap_mode="r")
        gy = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                      dtype=np.float32)
        f_map = float(json.load(open(f"{s}/events.json"))["f_map"])
        n = min(len(F), len(gy), WINDOWS[-1][1])
        rec = {"seq": s, "n": int(n), "f_map": f_map, "windows": []}

        for (a, b) in WINDOWS:
            if b > n:
                continue
            idx = np.arange(max(a, 1), b, SUB)
            e_naive, e_gyro, e_oracle = [], [], []
            ox, oy, gdx, gdy = [], [], [], []
            for t in idx:
                Ft = np.asarray(F[t], dtype=np.float32)
                Fp = np.asarray(F[t - 1], dtype=np.float32)
                g = gy[t] * SIGNS
                # naive / gyro 单应 / oracle 平移，三者同口径（margin=5 内区）
                e_naive.append(float(np.abs(interior(Ft, 5)
                                            - interior(Fp, 5)).mean()))
                e_gyro.append(float(np.abs(
                    interior(Ft, 5) - interior(warp_rot(Fp, g, f_map), 5)
                ).mean()))
                sx, sy, r = oracle_shift(Fp, Ft, 5)
                e_oracle.append(r)
                ox.append(sx)
                oy.append(sy)
                # gyro 经本项目的 tan 近似真正施加的内容位移
                # （expected_shift 给出 (dx,dy)，shift_map 实际用 (-dx,-dy)）
                yaw_r = float(np.radians(g[1])) / FPS
                pitch_r = float(np.radians(g[0])) / FPS
                gdx.append(-f_map * float(np.tan(yaw_r)))
                gdy.append(-f_map * float(np.tan(pitch_r)))
            ox, oy = np.array(ox), np.array(oy)
            gdx, gdy = np.array(gdx), np.array(gdy)

            def corr(u: np.ndarray, v: np.ndarray) -> float | None:
                if u.std() > 1e-9 and v.std() > 1e-9:
                    return float(np.corrcoef(u, v)[0, 1])
                return None

            win = {
                "t0": a, "t1": b, "n_eval": int(len(idx)),
                "naive": float(np.mean(e_naive)),
                "gyro_rot": float(np.mean(e_gyro)),
                "oracle": float(np.mean(e_oracle)),
                "oracle_shift_absmean": [float(np.abs(ox).mean()),
                                         float(np.abs(oy).mean())],
                "gyro_applied_absmean": [float(np.abs(gdx).mean()),
                                         float(np.abs(gdy).mean())],
                "corr_dx": corr(ox, gdx),
                "corr_dy": corr(oy, gdy),
            }
            base = win["naive"]
            win["gain_gyro"] = 1.0 - win["gyro_rot"] / base
            win["gain_oracle"] = 1.0 - win["oracle"] / base
            rec["windows"].append(win)
            agg.setdefault((a, b), []).append(win)

        out["seqs"].append(rec)
        for win in rec["windows"]:
            print(f"  {s.split('/')[-1]:9s}[{s.split('/')[1][4:]:6s}] "
                  f"t{win['t0']:5d}-{win['t1']:5d}  "
                  f"gyro {win['gain_gyro'] * 100:+6.2f}%  "
                  f"oracle {win['gain_oracle'] * 100:+6.2f}%  "
                  f"|shift| gyro [{win['gyro_applied_absmean'][0]:.2f} "
                  f"{win['gyro_applied_absmean'][1]:.2f}] "
                  f"oracle [{win['oracle_shift_absmean'][0]:.2f} "
                  f"{win['oracle_shift_absmean'][1]:.2f}]  "
                  f"corr [{win['corr_dx'] if win['corr_dx'] is None else round(win['corr_dx'], 2)} "
                  f"{win['corr_dy'] if win['corr_dy'] is None else round(win['corr_dy'], 2)}]")

    print("\n=== 按窗口汇总 ===")
    summary = {}
    for k in sorted(agg):
        ws = agg[k]
        gy_ = np.array([w["gain_gyro"] for w in ws])
        orc = np.array([w["gain_oracle"] for w in ws])
        cx = np.array([w["corr_dx"] for w in ws if w["corr_dx"] is not None])
        cy = np.array([w["corr_dy"] for w in ws if w["corr_dy"] is not None])
        summary[f"{k[0]}-{k[1]}"] = {
            "n_seqs": len(ws),
            "gain_gyro_mean": float(gy_.mean()),
            "gain_oracle_mean": float(orc.mean()),
            "corr_dx_mean": float(cx.mean()) if len(cx) else None,
            "corr_dy_mean": float(cy.mean()) if len(cy) else None,
        }
        print(f"  t{k[0]:5d}-{k[1]:5d}  n={len(ws):2d}  "
              f"gyro {gy_.mean() * 100:+6.2f}%  oracle {orc.mean() * 100:+6.2f}%  "
              f"corr_dx {cx.mean():+.3f}  corr_dy {cy.mean():+.3f}")

    out["summary"] = summary
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

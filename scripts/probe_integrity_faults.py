"""残差的主场：数据完整性故障检测（而非标注错误）。

## 为什么要换问题

前三轮实验（`probe_label_audit.py` / `audit_planted_labels.py` /
`compare_residual_vs_loss.py`）测的都是**标注错误**，结论偏负。原因是类别
不匹配：

* 标注错误是**语义/约定**问题——"这段该不该算一次操作"。残差不知道语义，
  所以定位不到（实测 barely_shift 0.00x、extend_far 0.46x）。
* 残差测的是**运动学可预测性**：|实际帧间变化 − IMU 预测的变化|。它天然
  能看见的是**这一因果链本身被破坏**。

这正好对应多模态数据集里另一类、同样昂贵且更难被肉眼发现的问题：
**数据完整性 / 采集链路故障**——时间戳错位、IMU 丢包、帧重复/丢帧、
传感器尖峰。这类问题与标注错误的关键区别是：它对**语义完全无知**，
因此标准工具（loss / 置信度）对它**结构性失明**——标注完全没变，loss
不会动。而残差会直接炸。

这一点是机制上**保证**的，不是猜测：预测项由 IMU 给出，一旦 IMU 与视频
的关系被破坏，残差必然升高。

## 故障类型与判据

段级故障（整段受影响 -> 报"能否把这一段挑出来"，用 AUC）：
1. `imu_desync`     IMU 整体时移 ±3 帧（多模态数据集最常见、最难查）
2. `imu_scale`      IMU 增益错 15%（标定表错/单位混用）

帧级故障（局部受影响 -> 报 lift，即能否定位到具体帧）：
3. `imu_dropout`    IMU 在某段变成零（丢包/接口断开）
4. `frame_dup`      帧重复（丢帧被上层复制补齐）
5. `imu_spike`      IMU 尖峰（单帧异常值）

输出：runs/integrity_fault_probe.json
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


def surprise_series(F, gy: np.ndarray, n: int, fps: float, f_map: float,
                    seq: str) -> np.ndarray:
    canc = FeatureMapCancel(CancelConfig(fps=fps, f_map=f_map, robust=True,
                                         pred="rotation", source=seq))
    s = np.zeros(n, dtype=np.float64)
    for t in range(n):
        s[t] = canc.step(np.asarray(F[t], dtype=np.float32), gy[t])["surprise"]
    return s


def auc(pos: np.ndarray, score: np.ndarray) -> float:
    """AUC（Mann-Whitney），pos 是布尔正类掩码。"""
    p = score[pos]
    q = score[~pos]
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    # 秩和公式，处理并列
    all_v = np.concatenate([p, q])
    r = np.argsort(np.argsort(all_v)).astype(np.float64) + 1.0
    rp = r[:len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))


def lift(score: np.ndarray, pos_idx: np.ndarray, evals: np.ndarray) -> float:
    if len(pos_idx) == 0:
        return float("nan")
    k = max(len(pos_idx), 1)
    posset = set(pos_idx.tolist())
    ranked = evals[np.argsort(-score[evals])][:k]
    hit = sum(int(f in posset) for f in ranked)
    return (hit / k) / (k / len(evals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_event")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--skip-first", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/integrity_fault_probe.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seqs = []
    for st in args.sets.split(","):
        seqs += sorted(glob.glob(f"{st.strip()}/val/seq_*"))
    if not seqs:
        raise SystemExit("没有找到序列")

    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out))
        if len(old.get("seqs", [])) > len(seqs):
            raise SystemExit(f"拒绝覆盖 {args.out}。加 --force。")

    rng = np.random.default_rng(args.seed)
    seg_rows, fr_rows = [], []

    for seq in seqs:
        if not os.path.exists(f"{seq}/featmaps.npy"):
            continue
        F = np.load(f"{seq}/featmaps.npy", mmap_mode="r")
        gy = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                      dtype=np.float32)
        meta = json.load(open(f"{seq}/events.json"))
        fps = float(meta.get("fps", 30.0))
        f_map = float(meta["f_map"])
        n = min(len(F), len(gy), args.n)
        lo = min(args.skip_first, n // 4)
        evals = np.arange(lo, n)

        s_clean = surprise_series(F, gy, n, fps, f_map, seq)
        base_med = float(np.median(s_clean[evals]))

        # ---------- 段级：整段 IMU 故障 -> 能否把它们挑出来 ----------
        for kind in ("imu_desync", "imu_scale"):
            scores = []
            for _ in range(6):
                g = gy.copy()
                if kind == "imu_desync":
                    k = int(rng.integers(2, 6)) * (1 if rng.random() < 0.5 else -1)
                    g = np.roll(g, k, axis=0)
                else:
                    g = g * float(rng.uniform(1.12, 1.20))
                s = surprise_series(F, g, n, fps, f_map, seq)
                scores.append(float(np.median(s[evals])))
            seg_rows.append({"seq": seq, "kind": kind,
                             "clean_median": base_med,
                             "faulted_median_mean": float(np.mean(scores)),
                             "ratio_mean": float(np.mean(scores) / max(base_med, 1e-9))})

        # ---------- 帧级：局部故障 -> 能否定位到帧 ----------
        # 注意：注入故障后必须重算整条 surprise（在线统计会被影响，这正是
        # 真实场景的样子——我们不能"假装干净地"扫一遍再检测）
        fault_kinds = ("imu_dropout", "frame_dup", "imu_spike")
        for kind in fault_kinds:
            per_rep = []
            for _ in range(8):
                span = int(rng.integers(20, 60))
                a = int(rng.integers(lo, max(n - span, lo + 1)))
                b = min(a + span, n)
                g = gy.copy()
                Fx = np.asarray(F[:n], dtype=np.float32).copy()
                if kind == "imu_dropout":
                    g[a:b] = 0.0
                elif kind == "frame_dup":
                    Fx[a:b] = Fx[a - 1]           # 复制同一帧 = 丢帧
                else:
                    idx = np.arange(a, b)
                    g[idx] += rng.normal(0, 300.0, size=(len(idx), 3))
                s = surprise_series(Fx, g, n, fps, f_map, seq)
                pos = np.arange(a, b)
                pos = pos[(pos >= lo)]
                if len(pos) == 0:
                    continue
                per_rep.append(lift(s, pos, evals))
            if per_rep:
                fr_rows.append({"seq": seq, "kind": kind,
                                "lift_mean": float(np.nanmean(per_rep)),
                                "lift_max": float(np.nanmax(per_rep))})
        print(f"  {seq.split('/')[-1]:9s} n={n:5d} 干净中位 surprise={base_med:6.3f}  "
              f"已完成")

    print(f"\n=== 段级故障（能否挑出被污染的整段）===")
    seg_sum = {}
    for kind in ("imu_desync", "imu_scale"):
        v = np.array([r["ratio_mean"] for r in seg_rows if r["kind"] == kind])
        if len(v):
            seg_sum[kind] = {"median_ratio": float(np.median(v)),
                             "min_ratio": float(v.min()), "n": len(v)}
            print(f"  {kind:14s} faulted/clean 中位倍数 = {np.median(v):6.2f}x  "
                  f"最小 {v.min():5.2f}x  (n={len(v)})")
    print("  判读：倍数 ≫1 -> 残差能把这些段挑出来（清洗时可直接按段隔离）。")

    print(f"\n=== 帧级故障（能否定位到具体帧）===")
    fr_sum = {}
    for kind in ("imu_dropout", "frame_dup", "imu_spike"):
        v = np.array([r["lift_mean"] for r in fr_rows if r["kind"] == kind])
        if len(v):
            fr_sum[kind] = {"mean_lift": float(v.mean()),
                            "median_lift": float(np.median(v)),
                            "seqs_above_3x": int((v > 3).sum()), "n": len(v)}
            print(f"  {kind:14s} lift = {v.mean():7.2f}x  中位 {np.median(v):7.2f}x  "
                  f">3x 的段数 {int((v > 3).sum())}/{len(v)}")
    print("  判读：lift ≫1 -> 残差指向了故障帧，可输出候选清单人工复核。")

    report = {
        "sets": args.sets, "n_frames_per_seq": args.n, "seed": args.seed,
        "predictor": "FeatureMapCancel(pred='rotation', robust=True).surprise",
        "why_this_question": "残差测的是运动学可预测性，因此它的主场是"
                             "'IMU-视频因果链被破坏'这类**数据完整性故障**，"
                             "而不是语义/约定层面的**标注错误**（后者已实测"
                             "偏负，见 runs/label_audit_planted.json）。"
                             "关键：loss 类工具对完整性故障结构性失明——"
                             "标注没变，loss 不会动。",
        "segment_level": seg_sum,
        "frame_level": fr_sum,
        "seg_rows": seg_rows, "frame_rows": fr_rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

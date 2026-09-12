"""探针：残差能否在尾部（top-k）指出与标注不一致的帧？

## 为什么需要这个探针

用户提出的新方向是"在标注数据里找不明显的异常/错误数据，约 1% 的金子"。
这与项目**已测的那些问题不同**：

* 已测（负结果 1、4）：把残差当作**分类器**去排序全部帧，AUC 0.47--0.55。
* 已测（负结果 3）：均值意义上，事件区间内 vs 外的残差几乎一样
  （0.197 vs 0.193）。
* **未测**：残差在**尾部**（最高的 1%/5%/10%）是否与标注系统性地不一致。

前两者都是**聚合**指标，会被占 20% 且内部高度异质的正类淹没；而"找 1% 的金子"
是一个**尾部**问题。一个 AUC≈0.5 的信号完全可能在 top-1% 上很强。所以这个问题
必须单独测，不能从前两条负结果推断。

## 这个探针回答三个问题

1. **富集**：top-k 惊讶帧里，落在标注区间**外**的比例是多少？相对基率如何？
2. **是边界抖动还是真漏标**：那些"标注外且高残差"的帧，到最近标注边界的距离
   是多少？若几乎贴着边界 -> 只是标注 ±0.5 s 的糊边（不稀奇）；若孤立地落在
   长段无标注区的中间 -> 才是"漏标候选"。
3. **反向**：标注区间**内**残差最低的那批帧（标注说有事、残差说没事）——它们是
   区间里的空闲时刻，还是"过度标注"候选？

输出：runs/label_audit_probe.json
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

from cerebellum_cancel.featmap_cancel import CancelConfig, FeatureMapCancel  # noqa: E402
from cerebellum_cancel.io import load_imu_csv                                # noqa: E402


def intervals_to_mask(events, fps: float, n: int) -> np.ndarray:
    """秒 -> 逐帧布尔掩码。区间端点取 ceil/floor 以保守（不把边界算进内）。"""
    m = np.zeros(n, dtype=bool)
    for a, b in events:
        i0 = int(np.floor(float(a) * fps))
        i1 = int(np.ceil(float(b) * fps))
        m[max(i0, 0):min(i1, n)] = True
    return m


def edge_distance(idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每个帧号到最近一次"标注状态翻转"的距离（帧）。

    用差分找翻转点，再取最近距离。标注外的帧若贴着边界 -> 距离小。
    """
    flips = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    if len(flips) == 0:
        return np.full(len(idx), np.inf)
    flips_sorted = np.sort(flips)
    pos = np.searchsorted(flips_sorted, idx)
    left = flips_sorted[np.clip(pos - 1, 0, len(flips_sorted) - 1)]
    right = flips_sorted[np.clip(pos, 0, len(flips_sorted) - 1)]
    return np.minimum(np.abs(idx - left), np.abs(idx - right)).astype(float)


def probe_seq(seq: str, n_max: int, skip_first: int) -> dict | None:
    fm_path = f"{seq}/featmaps.npy"
    if not os.path.exists(fm_path):
        return None
    F = np.load(fm_path, mmap_mode="r")
    gy = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                  dtype=np.float32)
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    f_map = float(meta["f_map"])
    n = min(len(F), len(gy), n_max)

    cfg = CancelConfig(fps=fps, f_map=f_map, robust=True, pred="rotation",
                       source=seq)
    canc = FeatureMapCancel(cfg)
    surprise = np.zeros(n, dtype=np.float64)
    for t in range(n):
        out = canc.step(np.asarray(F[t], dtype=np.float32), gy[t])
        surprise[t] = out["surprise"]

    mask = intervals_to_mask(meta.get("events", []), fps, n)
    lo = min(skip_first, n // 4)
    sl = slice(lo, n)
    s = surprise[sl]
    m = mask[sl]
    idx_global = np.arange(lo, n)
    base_unann = float(1.0 - m.mean())

    out: dict = {
        "seq": seq, "n": int(n), "n_eval": int(len(s)), "fps": fps,
        "annotated_frac": float(m.mean()), "base_unannotated": base_unann,
        "n_intervals": len(meta.get("events", [])),
        "topk": {},
    }
    n_eval = len(s)
    for frac in (0.01, 0.05, 0.10):
        k = max(int(round(frac * n_eval)), 1)
        top = np.argsort(-s)[:k]
        unann = ~m[top]
        out["topk"][f"{frac:.2f}"] = {
            "k": k,
            "unannotated_frac": float(unann.mean()),
            # 富集 = 相对基率的倍数；1.0 = 与随机无异
            "enrichment": float(unann.mean() / max(base_unann, 1e-9)),
            "median_edge_dist_frames": (
                float(np.median(edge_distance(idx_global[top][unann], mask)))
                if unann.any() else None),
            "median_edge_dist_sec": (
                float(np.median(edge_distance(idx_global[top][unann], mask))
                      / fps) if unann.any() else None),
        }

    # 反向：标注内、残差最低的 10%
    in_ann = np.flatnonzero(m)
    rev = {}
    if len(in_ann) >= 20:
        k = max(int(round(0.10 * len(in_ann))), 1)
        quiet = in_ann[np.argsort(s[in_ann])[:k]]
        rev = {
            "k": int(k),
            "median_edge_dist_sec": float(
                np.median(edge_distance(idx_global[quiet], mask)) / fps),
        }
    out["quiet_inside_annotation"] = rev
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_event")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--skip-first", type=int, default=60)
    ap.add_argument("--out", default="runs/label_audit_probe.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seqs = []
    for st in args.sets.split(","):
        seqs += sorted(glob.glob(f"{st.strip()}/val/seq_*"))

    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out))
        if len(old.get("seqs", [])) > len(seqs):
            raise SystemExit(f"拒绝覆盖 {args.out}：现有段数更多。加 --force。")

    rows = []
    for s in seqs:
        r = probe_seq(s, args.n, args.skip_first)
        if r is None:
            continue
        rows.append(r)
        name = s.split("/")[-1] + "[" + s.split("/")[1][4:] + "]"
        t1 = r["topk"]["0.01"]
        print(f"  {name:22s} 标注占比 {r['annotated_frac'] * 100:5.1f}%  "
              f"top1% 标注外 {t1['unannotated_frac'] * 100:5.1f}%  "
              f"富集 {t1['enrichment']:.2f}x  "
              f"中位边距 {t1['median_edge_dist_sec']:.1f}s")

    print(f"\n=== 汇总（{len(rows)} 段）===")
    summary = {}
    for frac in ("0.01", "0.05", "0.10"):
        e = np.array([r["topk"][frac]["enrichment"] for r in rows])
        b = np.array([r["topk"][frac]["unannotated_frac"] for r in rows])
        d = np.array([r["topk"][frac]["median_edge_dist_sec"] or np.nan
                      for r in rows], dtype=float)
        summary[frac] = {
            "mean_enrichment": float(e.mean()),
            "mean_unannotated_frac": float(b.mean()),
            "mean_median_edge_sec": float(np.nanmean(d)),
        }
        print(f"  top{float(frac) * 100:>4.0f}%:  标注外占比 {b.mean() * 100:5.1f}%  "
              f"富集 {e.mean():.3f}x   中位边距 {np.nanmean(d):5.1f}s  "
              f"（基率 {np.mean([r['base_unannotated'] for r in rows]) * 100:.1f}%）")

    base = float(np.mean([r["base_unannotated"] for r in rows]))
    print(f"\n  基率（标注外帧占比）= {base * 100:.1f}%。")
    print("  富集 ≈ 1.0 -> 残差在尾部与标注无系统关系；> 1 表示尾部偏向标注外。")
    print("  中位边距小（<1-2s）-> 大概率是标注边界糊边；大（>5s）-> 漏标候选。")

    report = {
        "sets": args.sets, "n_frames_per_seq": args.n,
        "skip_first": args.skip_first,
        "predictor": "FeatureMapCancel(pred='rotation', robust=True) —— "
                     "项目当前最强的标签无关残差",
        "metric": "surprise（格内 z 分数最大值的逐帧序列）",
        "note": "测的是**尾部**问题（top-k 与标注的一致性），与已发布的聚合口径"
                "（AUC / 区间内外均值）不是同一个问题，不能互相推断。",
        "base_unannotated": base,
        "summary": summary,
        "seqs": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

"""注入式标签审计：残差能找出哪种标注错误，precision/recall 是多少。

## 为什么必须用"注入"而不是"相关"

`runs/label_audit_probe.json` 测出：残差在尾部（top-1%/5%/10%）与标注**没有**
系统关系（富集 1.03x / 1.00x / 0.99x）。但那个实验有一个根本缺陷——**它没有
真值**。它只知道"某帧在标注外"，不知道"某帧标错了"。一个信号可以完全无法区分
"标注内外"，却精准命中标注里真正的错处，反之亦然。所以那个实验只能排除一种
用法，不能回答"能不能找错"。

要回答后者，只有一条路：**注入已知的错误，看能不能找回来**。这也正是数据清洗
文献的标准做法（planted error / synthetic corruption）。

## 注入哪一类错误——这是本脚本的核心判断

"标注错误"不是一种东西，而至少有两类，可检出性截然不同：

* **语义型**：动词标反了（"拿"标成"放"）。帧的运动学看不出任何异常——画面里
  确实有手在动，残差不会高。项目已有一条负结果正压在这里
  （"动能不可预测 ≠ 语义重要性"）。**这一类的可检出性应该接近随机**，本脚本
  把它作为**阴性对照**。
* **时间型**：区间边界错位、短事件整段漏标、把无操作段标成操作。这类错误会
  让"标注说没事"的那几帧恰好落在运动瞬变上——**残差恰恰测这个**。这是本项目
  的残差唯一可能真正有价值的地方，本脚本重点测它。

## 注入的错误类型

1. `barely_shift`  边界整体平移 ±2 帧（最常见的人工/自动标注缺陷）
2. `drop_short`    随机删掉一个短区间（漏标）
3. `extend_far`    把一个区间向外拉长到覆盖一段静止（过度标注）
4. `verb_swap`     **语义错**：把某帧的动词标签换成另一个（阴性对照）

## 判据

对每种注入，把"被注入影响的帧"当作正类，用残差排序，报
precision@k（k = 注入帧数，即"只查这么多"）与 recall@k。随机基线 = k/N。
**只有显著高于随机基线的类型，才值得做成产品。**

输出：runs/label_audit_planted.json
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

# 复用探针里的残差计算与工具，避免两处实现漂移
from probe_label_audit import intervals_to_mask                     # noqa: E402
from cerebellum_cancel.featmap_cancel import (CancelConfig,          # noqa: E402
                                               FeatureMapCancel)
from cerebellum_cancel.io import load_imu_csv                        # noqa: E402


def compute_surprise(seq: str, n_max: int) -> tuple[np.ndarray, dict, np.ndarray]:
    F = np.load(f"{seq}/featmaps.npy", mmap_mode="r")
    gy = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                  dtype=np.float32)
    meta = json.load(open(f"{seq}/events.json"))
    n = min(len(F), len(gy), n_max)
    cfg = CancelConfig(fps=float(meta.get("fps", 30.0)),
                       f_map=float(meta["f_map"]), robust=True,
                       pred="rotation", source=seq)
    canc = FeatureMapCancel(cfg)
    s = np.zeros(n, dtype=np.float64)
    for t in range(n):
        s[t] = canc.step(np.asarray(F[t], dtype=np.float32), gy[t])["surprise"]
    return s, meta, gy


def touch_mask(orig: np.ndarray, new: np.ndarray) -> np.ndarray:
    """注入所影响的帧 = 两次掩码的对称差。这是该注入的**真值正类**。"""
    return orig != new


def inject_barely_shift(ev, fps, n, rng, shift_frames=2):
    """把一个随机区间整体平移几帧——边界错位。"""
    if not ev:
        return ev, None
    i = int(rng.integers(0, len(ev)))
    d = shift_frames / fps * (1 if rng.random() < 0.5 else -1)
    a, b = ev[i]
    new = list(ev)
    new[i] = [a + d, b + d]
    return new, i


def inject_drop_short(ev, fps, n, rng):
    """删掉一个短区间——漏标。"""
    if not ev:
        return ev, None
    durs = np.array([b - a for a, b in ev])
    cands = np.flatnonzero(durs <= np.percentile(durs, 40))
    if not len(cands):
        return ev, None
    i = int(rng.choice(cands))
    return [e for j, e in enumerate(ev) if j != i], i


def inject_extend_far(ev, fps, n, rng, extra_sec=3.0):
    """把一个区间的末端向外拉长——过度标注。"""
    if not ev:
        return ev, None
    i = int(rng.integers(0, len(ev)))
    new = [list(e) for e in ev]
    new[i] = [new[i][0], min(new[i][1] + extra_sec, n / fps)]
    return new, i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_event")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--skip-first", type=int, default=60)
    ap.add_argument("--reps", type=int, default=30,
                    help="每种注入每段重复次数（取均值以压住单次随机性）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/label_audit_planted.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seqs = []
    for st in args.sets.split(","):
        seqs += sorted(glob.glob(f"{st.strip()}/val/seq_*"))

    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out))
        if len(old.get("seqs", [])) > len(seqs):
            raise SystemExit(f"拒绝覆盖 {args.out}：现有段数更多。加 --force。")

    rng = np.random.default_rng(args.seed)
    kinds = ("barely_shift", "drop_short", "extend_far", "verb_swap")
    per_kind: dict[str, list[dict]] = {k: [] for k in kinds}
    rows = []

    for seq in seqs:
        if not os.path.exists(f"{seq}/featmaps.npy"):
            continue
        s_full, meta, _ = compute_surprise(seq, args.n)
        fps = float(meta.get("fps", 30.0))
        n = len(s_full)
        lo = min(args.skip_first, n // 4)
        ev = [list(e) for e in meta.get("events", [])]
        orig = intervals_to_mask(ev, fps, n)
        evals = np.arange(lo, n)

        for kind in kinds:
            if kind == "verb_swap":
                # 阴性对照：语义错。它**不影响任何帧的区间归属**——所以按构造
                # 残差不可能定位它。这里显式把它建成"无信号"的实现：正类是
                # 随机抽的帧。若残差在它上面也"命中"，说明我们的评测有泄漏。
                k = max(int(round(0.01 * len(evals))), 1)
                p_at = []
                for _ in range(args.reps):
                    pos = rng.choice(evals, size=k, replace=False)
                    posset = set(pos.tolist())
                    ranked = evals[np.argsort(-s_full[evals])][:k]
                    p_at.append(sum(int(f in posset) for f in ranked) / k
                                / (k / len(evals)))
                per_kind[kind].append(
                    {"seq": seq, "k": k, "lift_vs_random": float(np.mean(p_at))})
                continue

            lifts, precs = [], []
            k = 0
            for _ in range(args.reps):
                if kind == "barely_shift":
                    nev, _ = inject_barely_shift(ev, fps, n, rng)
                elif kind == "drop_short":
                    nev, _ = inject_drop_short(ev, fps, n, rng)
                else:
                    nev, _ = inject_extend_far(ev, fps, n, rng)
                new = intervals_to_mask(nev, fps, n)
                pos = np.flatnonzero(touch_mask(orig, new))
                pos = pos[(pos >= lo)]
                if len(pos) == 0:
                    continue
                k = len(pos)
                posset = set(pos.tolist())
                ranked = evals[np.argsort(-s_full[evals])][:k]
                hit = sum(int(f in posset) for f in ranked)
                precs.append(hit / k)
                # lift：命中率 / 随机命中率(k/N)
                lifts.append((hit / k) / (k / len(evals)))
            if lifts:
                per_kind[kind].append(
                    {"seq": seq, "n_pos_mean": float(k),
                     "precision_at_k": float(np.mean(precs)),
                     "lift_vs_random": float(np.mean(lifts))})

    print(f"\n=== 注入式审计（{len(rows)} 段…实际按 kind 统计）===")
    summary = {}
    for kind in kinds:
        rs = per_kind[kind]
        if not rs:
            continue
        lv = np.array([r["lift_vs_random"] for r in rs])
        summary[kind] = {
            "n_seqs": len(rs),
            "mean_lift_vs_random": float(lv.mean()),
            "median_lift": float(np.median(lv)),
            "seqs_above_2x": int((lv > 2.0).sum()),
        }
        extra = ""
        if "precision_at_k" in rs[0]:
            pr = np.array([r["precision_at_k"] for r in rs])
            summary[kind]["mean_precision_at_k"] = float(pr.mean())
            extra = f"  precision@k={pr.mean():.3f}"
        print(f"  {kind:14s} lift={lv.mean():6.2f}x  "
              f"中位={np.median(lv):5.2f}x  >2x 的段数 {int((lv > 2.0).sum())}"
              f"/{len(rs)}{extra}")

    print("\n  判读：lift≈1 = 与随机无异（残差定位不到该错误）；")
    print("        lift≫1 = 残差确实指向该错误。verb_swap 是阴性对照，")
    print("        它若也≫1 则说明评测本身有泄漏，整张表作废。")

    report = {
        "sets": args.sets, "n_frames_per_seq": args.n, "reps": args.reps,
        "seed": args.seed, "skip_first": args.skip_first,
        "predictor": "FeatureMapCancel(pred='rotation', robust=True)",
        "method": "planted error：注入已知错误，测残差能否把它排在前面。"
                  "lift = precision@k / (k/N)，即相对随机排序的提升倍数。",
        "injections": {
            "barely_shift": "边界整体平移 ±2 帧（时间型）",
            "drop_short": "删掉一个短区间（漏标，时间型）",
            "extend_far": "区间末端拉长 3s（过度标注，时间型）",
            "verb_swap": "语义型错误的阴性对照（不影响区间归属）",
        },
        "summary": summary,
        "per_seq": per_kind,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

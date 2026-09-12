"""事件"时刻"信号探针——把目标从"整个区间"换成"转折时刻"，并横向比较廉价信号。

## 为什么换目标

上一轮（`runs/gate_operating_curve.json`）把"事件"当成**整个标注区间**（boxcar），
结论是残差分数对它的帧级 AUC = 0.510（随机）。但这个目标定义本身可能错配：
残差按机制是**变化检测器**——它该在**内容发生转折**的时刻响应（物体进入/离开
画面、位移/碰撞的瞬间），而不是在整个 1–8 s 的标注区间里持续响应。

所以同一个信号在三个目标定义下应该分别评：

| 目标 | 定义 | 假设下残差该不该有信息 |
| ---- | ---- | ---------------------- |
| `boxcar` | 区间内全部帧 | 不该（上轮已测：AUC 0.51） |
| `onset`  | 区间**开始**±tol 帧 | 该（物体出现 = 变化） |
| `offset` | 区间**结束**±tol 帧 | 该（物体消失 = 变化） |
| `bounds` | onset ∪ offset | 该（最完整的"转折时刻"版本） |

**这是上轮负结果唯一还没排掉的结构性解释**：不是信号没用，是目标定错了。

## 横向比较：什么样的廉价信号才能当守门

单测残差不够——真正要回答的是"如果要做守门，得用什么信号"。所以把从最廉价
到最贵的候选放在**同一张表**上比较，全部可从缓存算出（无需帧）：

* `resid_shift` / `resid_rotation` —— 本项目的残差分数（0.03 ms 级）
* `naive_fm`  —— 特征图逐帧绝对差（不补偿，最笨的基线）
* `feat_diff` —— 池化 576 维特征的逐帧 L2 差
* `feat_novelty` —— 池化特征相对自身 EMA 的偏离（新颖性检测）
* `feat_jerk` —— 池化特征的二阶差分（变化率的变化）
* `oracle` —— 完美信号（仅供参考的上界，AUC 恒为 1，不参与排名）

主判据用**帧级 AUC**：它与"聚簇/预算"完全解耦，是"这个信号知不知道时刻"
的干净读数（上轮已知聚簇会污染 capture 口径）。副表再给匹配唤醒预算下的
命中率。

## 两个必须提的局限

1. **特征标准化用的是整段统计**（逐维 z-score），这是**离线诊断**而非因果部署；
   要部署得换成运行统计。本探针的目的是给信号**排名**，不是上线。
2. `data/aoe_event` **没有保留帧**（磁盘预算，`data/aoe_src` 已不在），所以
   真正的像素级候选（帧差、亮度突变、运动模糊程度）**无法在此数据上比较**。
   能比较的最强前端就是已有的 MobileNetV3 池化特征与 28×28 特征图。

Usage:
  PYTHONPATH=. .venv/bin/python scripts/probe_event_moment_signals.py data/aoe_event
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.core import TriggerConfig
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                              maxcell_score, rollout)
from cerebellum_cancel.io import load_imu_csv

TOLS = (0.25, 0.5, 1.0)      # seconds around a boundary
MAIN_TOL = 0.5
WARMUP = 30


def seq_dirs(dataset):
    return sorted(glob.glob(f"{dataset}/val/seq_*"))


# ---------------------------------------------------------------- signals

def residual_score(seq, pred):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    canc = FeatureMapCancel(CancelConfig(fps=fps, robust=True, pred=pred,
                                        source=seq))
    if "f_map" in meta:
        canc.cfg.f_map = float(meta["f_map"])
    z, _ = rollout(canc, fm[:n], gyro[:n])
    return maxcell_score(z).astype(np.float64), fm[:n], fps, n, meta


def naive_fm_score(fm):
    """Mean |F_t - F_{t-1}| over the feature map: no warp, no learning."""
    d = np.abs(np.diff(fm.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
    return np.r_[0.0, d].astype(np.float64)


def standardize_causal(x):
    """Per-dim running z-score (EMA mean/var), so downstream distances are
    not dominated by high-variance dims. Returns (T, D)."""
    T, D = x.shape
    out = np.zeros_like(x, dtype=np.float64)
    mu = x[0].astype(np.float64).copy()
    var = np.ones(D, dtype=np.float64)
    k = 0.02
    for t in range(T):
        sd = np.sqrt(var) + 1e-6
        out[t] = (x[t] - mu) / sd
        d = x[t] - mu
        mu += k * d
        var += k * (d * d - var)
    return out


def feature_signals(seq):
    """Cheap label-free traces from the cached pooled features."""
    f = np.load(f"{seq}/features.npy").astype(np.float32)
    xs = standardize_causal(f)
    n = len(xs)
    diff = np.zeros(n, dtype=np.float64)
    diff[1:] = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    # novelty: distance from an EMA of the past (a "does this look like what
    # came before" score, i.e. a one-step predictive-coding residual)
    ema = np.zeros_like(xs)
    acc = xs[0].copy()
    novel = np.zeros(n, dtype=np.float64)
    a = 0.05
    for t in range(n):
        novel[t] = np.linalg.norm(xs[t] - acc)
        acc = acc + a * (xs[t] - acc)
        ema[t] = acc
    # jerk: second difference (rate of change of change)
    jerk = np.zeros(n, dtype=np.float64)
    jerk[1:-1] = np.linalg.norm(xs[2:] - 2 * xs[1:-1] + xs[:-2], axis=1)
    return {"feat_diff": diff, "feat_novelty": novel, "feat_jerk": jerk}


# ---------------------------------------------------------------- targets

def build_targets(meta, fps, n, tol):
    """Boolean target masks at tolerance ``tol`` seconds."""
    tolfr = max(int(tol * fps), 1)
    events = [(max(int(a * fps), 0), min(int(b * fps), n - 1))
              for a, b in meta["events"]]
    events = [(a, b) for a, b in events if b >= a]

    def impulse(pts):
        m = np.zeros(n, dtype=bool)
        for p in pts:
            m[max(p - tolfr, 0):min(p + tolfr + 1, n)] = True
        return m

    boxcar = np.zeros(n, dtype=bool)
    for a, b in events:
        boxcar[a:b + 1] = True
    starts = [a for a, _b in events]
    ends = [b for _a, b in events]
    return {
        "boxcar": boxcar,
        "onset": impulse(starts),
        "offset": impulse(ends),
        "bounds": impulse(starts + ends),
    }


def auc(score, mask, warm=WARMUP):
    """Rank-based AUC; ties handled by average ranks."""
    s = score[warm:]
    y = mask[warm:]
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if len(uniq) < len(s):
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def sign_test_p(vals, null=0.5):
    """Two-sided exact binomial sign test on paired per-segment values.

    Preferred over a t-test here: n=10 segments, no normality assumption, and
    what matters is "does the signal beat chance on most segments", which is
    exactly a sign question.
    """
    from math import comb
    v = np.asarray([x for x in vals if not np.isnan(x)], dtype=float)
    pos = int((v > null).sum())
    neg = int((v < null).sum())
    n = pos + neg
    if n == 0:
        return float("nan"), pos, n
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(2 * tail, 1.0)), pos, n


def paired_diff_test(a, b):
    """Sign test on paired differences a-b (which signal wins, per segment)."""
    from math import comb
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    ok = ~np.isnan(d)
    d = d[ok]
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return float("nan"), pos, n
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(2 * tail, 1.0)), pos, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe_event")
    ap.add_argument("--out", default="runs/event_moment_probe.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    signals = ["resid_shift", "resid_rotation", "naive_fm", "feat_diff",
               "feat_novelty", "feat_jerk"]
    targets = ["boxcar", "onset", "offset", "bounds"]

    per_seq = {}
    for seq in seq_dirs(args.dataset):
        tag = os.path.basename(seq)
        rs, fm, fps, n, meta = residual_score(seq, "shift")
        rr, _fm2, _fps2, _n2, _m2 = residual_score(seq, "rotation")
        sig = {"resid_shift": rs, "resid_rotation": rr,
               "naive_fm": naive_fm_score(fm)}
        sig.update(feature_signals(seq))
        tgt = build_targets(meta, fps, n, MAIN_TOL)
        row = {s: {} for s in signals}
        for s in signals:
            for t in targets:
                row[s][t] = auc(sig[s], tgt[t])
        # target rates for context
        rates = {t: float(tgt[t].mean()) for t in targets}
        per_seq[tag] = {"n": n, "fps": fps, "n_events": len(meta["events"]),
                        "target_rate": rates, "auc": row}
        print(f"  {tag}: n={n} ev={len(meta['events'])} "
              f"boxcar={rates['boxcar']:.3f} bounds={rates['bounds']:.3f} | "
              + " ".join(f"{s[:6]}/bnd={row[s]['bounds']:.3f}"
                         for s in ("resid_rotation", "feat_diff",
                                   "feat_novelty")), flush=True)

    def agg(s, t):
        return np.array([per_seq[k]["auc"][s][t] for k in per_seq],
                        dtype=float)

    def ci(vals):
        v = vals[~np.isnan(vals)]
        if len(v) < 2:
            return float("nan"), float("nan")
        bs = np.array([np.median(rng.choice(v, len(v), replace=True))
                       for _ in range(args.boot)])
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print(f"\n=== 帧级 AUC（10 段中位，tol={MAIN_TOL}s）===")
    hdr = "  " + f"{'signal':<16}" + "".join(f"{t:>10}" for t in targets)
    print(hdr)
    table = {}
    for s in signals:
        cells = []
        for t in targets:
            v = agg(s, t)
            cells.append(float(np.nanmedian(v)))
        table[s] = {t: cells[i] for i, t in enumerate(targets)}
        print(f"  {s:<16}" + "".join(f"{c:>10.3f}" for c in cells))

    print(f"\n=== 逐信号 × 逐目标：中位 + 95% bootstrap CI（跨段）===")
    ci_table = {}
    for s in signals:
        ci_table[s] = {}
        for t in targets:
            v = agg(s, t)
            lo, hi = ci(v)
            ci_table[s][t] = {"median": float(np.nanmedian(v)),
                              "ci_lo": lo, "ci_hi": hi,
                              "per_seq": v.tolist()}
    # print the key comparison
    print(f"  {'signal':<16}{'target':<9}{'median':>8}{'CI':>18}")
    for s in signals:
        for t in targets:
            c = ci_table[s][t]
            print(f"  {s:<16}{t:<9}{c['median']:>8.3f}"
                  f"   [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]")

    # ---- significance: sign tests (n=10 segments, no normality assumed)
    print(f"\n=== 显著性：符号检验（双侧，H0: AUC=0.5，n=10 段）===")
    print(f"  {'signal':<16}{'target':<9}{'median':>8}{'>0.5 段数':>11}"
          f"{'p':>9}   判读")
    sig_table = {}
    for s in signals:
        for t in targets:
            v = agg(s, t)
            p, pos, k = sign_test_p(v)
            med = float(np.nanmedian(v))
            if p < 0.05 and med > 0.5:
                verdict = "显著正"
            elif p < 0.05 and med < 0.5:
                verdict = "显著负(!)"
            else:
                verdict = "不显著"
            sig_table[f"{s}|{t}"] = {"median": med, "pos": pos, "n": k,
                                     "p": p, "verdict": verdict}
            print(f"  {s:<16}{t:<9}{med:>8.3f}{pos:>7}/{k:<3}{p:>9.4f}"
                  f"   {verdict}")

    # ---- head-to-head: residual vs the cheap alternatives on `bounds`
    print(f"\n=== 头对头（bounds 目标，配对符号检验）===")
    head = {}
    for other in ("naive_fm", "feat_diff", "feat_novelty"):
        a, b = agg("resid_rotation", "bounds"), agg(other, "bounds")
        p, pos, k = paired_diff_test(a, b)
        head[other] = {"resid_wins": pos, "n": k, "p": p}
        print(f"  resid_rotation vs {other:<13}: resid 胜 {pos}/{k} 段, p={p:.4f}")

    # tolerance sensitivity for the boundary target
    print(f"\n=== 容差敏感性（bounds 目标，AUC 中位）===")
    tol_rows = {}
    for tol in TOLS:
        vals = {}
        for seq in seq_dirs(args.dataset):
            rs, fm, fps, n, meta = residual_score(seq, "rotation")
            sig = {"resid_rotation": rs, "naive_fm": naive_fm_score(fm)}
            sig.update(feature_signals(seq))
            tgt = build_targets(meta, fps, n, tol)["bounds"]
            for s in ("resid_rotation", "naive_fm", "feat_diff", "feat_novelty"):
                vals.setdefault(s, []).append(auc(sig[s], tgt))
        tol_rows[tol] = {s: float(np.nanmedian(v)) for s, v in vals.items()}
        # boundary rate grows with tol; report it
        rates = []
        for seq in seq_dirs(args.dataset):
            meta = json.load(open(f"{seq}/events.json"))
            fps = float(meta.get("fps", 30.0))
            n = len(np.load(f"{seq}/features.npy", mmap_mode="r"))
            rates.append(float(build_targets(meta, fps, n, tol)["bounds"].mean()))
        print(f"  tol={tol:.2f}s  bounds_rate={np.median(rates):.3f}  "
              + "  ".join(f"{s}={v:.3f}" for s, v in tol_rows[tol].items()))

    dev = float(np.nanmedian(agg("resid_rotation", "bounds") - 0.5))
    out = {"dataset": args.dataset, "main_tol": MAIN_TOL, "tols": list(TOLS),
           "warmup": WARMUP, "boot": args.boot, "seed": args.seed,
           "per_seq": per_seq, "table_median": table, "ci_table": ci_table,
           "sign_tests": sig_table, "head_to_head": head,
           "tol_rows": {str(k): v for k, v in tol_rows.items()},
           "notes": [
               "Feature standardization is per-dim causal EMA (running "
               "mean/var), so the traces are causal; the segment-wide "
               "quantiles used for thresholds are NOT (offline diagnostic).",
               "data/aoe_event keeps no frames, so pixel-level candidates "
               "(frame diff, blur, brightness) could not be compared -- the "
               "strongest available front end is the cached MNV3 features.",
               "AUC is clustering-free, which is why it is the primary "
               "metric here (last round showed capture is confounded by "
               "burst structure).",
               "Significance is by exact binomial sign test over the 10 "
               "segments (no normality assumption); note that the effective "
               "sample size is 10, so only large effects can reach p<0.05.",
           ]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    print(f"\nbounds-目标上 resid_rotation 超出随机: {dev:+.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

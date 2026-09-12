"""免标注守门能得到多少？——学习式上界，量化"不许用标注"这个约束的代价。

## 位置

已知（全部实测）：

1. 训练-free 残差**不携带事件判别信息**（§2.3.7 追测④，250 轮/格）。
2. 训练-free 残差也**不携带事件时刻信息**（`runs/gate_operating_curve.json`：
   帧级 AUC 0.510、输给随机撒帧、onset 捕获 0.000）。
3. **换目标定义也救不回来**（`runs/event_moment_probe.json`：把目标从整个标注
   区间换成 onset/offset/bounds 转折时刻，6 种信号 × 4 种目标共 24 格
   **没有一格通过符号检验**，且残差打不赢最笨的"特征图逐帧差"）。

于是剩下唯一尚未量化的量：**如果允许用标注训练，能到多少？**
这个数字就是"免标注"这个约束的代价，也是决定这条线是否值得继续投入的依据。

## 判据（leave-one-segment-out，10 折；阈值只在 9 个训练段上标定）

* **帧级 AUC**（目标 = 标注区间）与 **AUC(bounds)**（目标 = 区间转折时刻）
  —— 与上一轮训练-free 的读数**直接可比**。
* **区间捕获 / onset 捕获 @5%**：阈值取**训练段**分数的 95 分位，在留出段上评。
  这是泛化口径（不看留出段自己的答案），比每段自定分位更严格。

## 两个特征集分开报（关键对照）

* **raw**：576 维冻结特征做 K 近邻一致性（每帧取 k 个最近邻的事件帧占比）
  —— 只用"帧**长什么样**"，**不含任何时间信息**。
* **raw+time**：raw 再拼接 [相对运行 EMA 的偏离范数, 逐帧变化范数]
  —— 正是训练-free 探针里唯一反复露头的两种信号形态。

**两者之差 = 时序结构的增量**。若 raw 已经很高，说明信号在静态外观里
（那训练-free 方案只是没读对）；若 raw ≈ 0.5 而 raw+time 明显更高，
说明信息在**时间结构**里（那才是免标注方案该去对齐的目标）。

## 计算

环境无 sklearn，故 kNN 与距离全部用 numpy（Gram 形式，分块，避免
(chunk, pool, dim) 广播）。训练池按固定步长下采样到 POOL_MAX 帧以内，
保证可复现的运行时间。

Usage:
  PYTHONPATH=. .venv/bin/python scripts/train_labeled_gate_upper_bound.py data/aoe_event
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from math import comb

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

TAIL_PAD = 0.5
ONSET_WIN = 0.5
POOL_MAX = 24000        # frames in the kNN reference pool
WARMUP = 30


def seq_dirs(dataset):
    return sorted(glob.glob(f"{dataset}/val/seq_*"))


def load_seg(seq):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    f = np.load(f"{seq}/features.npy").astype(np.float32)
    n = len(f)
    y = np.zeros(n, dtype=np.float32)
    bounds = np.zeros(n, dtype=np.float32)
    tol = max(int(ONSET_WIN * fps), 1)
    for a, b in meta["events"]:
        i0, i1 = max(int(a * fps), 0), min(int(b * fps), n - 1)
        if i1 >= i0:
            y[i0:i1 + 1] = 1.0
        for p in (i0, i1):
            bounds[max(p - tol, 0):min(p + tol + 1, n)] = 1.0
    return f, y, bounds, fps, n, meta


def temporal_feats(f):
    """Two causal temporal channels: deviation from a running EMA, and
    frame-to-frame change -- the shapes the training-free probe surfaced."""
    n = len(f)
    novel = np.zeros(n, dtype=np.float32)
    acc = f[0].copy()
    a = 0.05
    for t in range(n):
        novel[t] = np.linalg.norm(f[t] - acc)
        acc = acc + a * (f[t] - acc)
    diff = np.zeros(n, dtype=np.float32)
    diff[1:] = np.linalg.norm(np.diff(f, axis=0), axis=1)
    return np.stack([novel, diff], axis=1)


def knn_scores(Xpool, ypool, Xq, k=15, chunk=1024):
    """Mean label over each query frame's k nearest pool frames.

    Distances via the Gram form (|a|^2 + |b|^2 - 2ab) so nothing is broadcast
    into a (chunk, pool, dim) tensor.
    """
    P = np.ascontiguousarray(Xpool, dtype=np.float32)
    Q = np.ascontiguousarray(Xq, dtype=np.float32)
    p2 = np.einsum("ij,ij->i", P, P)
    out = np.zeros(len(Q), dtype=np.float32)
    kk = min(k, len(P))
    for i in range(0, len(Q), chunk):
        blk = Q[i:i + chunk]
        b2 = np.einsum("ij,ij->i", blk, blk)
        d = b2[:, None] + p2[None, :] - 2.0 * (blk @ P.T)
        np.maximum(d, 0.0, out=d)
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        out[i:i + len(blk)] = ypool[idx].mean(axis=1)
    return out


def auc(score, y, warm=WARMUP):
    s, yy = np.asarray(score, np.float64)[warm:], np.asarray(y, np.float64)[warm:]
    pos = yy > 0
    if pos.all() or not pos.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if len(uniq) < len(s):
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def capture(score, thr, meta, fps, n, pad=TAIL_PAD, win=ONSET_WIN):
    keep = np.zeros(n, dtype=bool)
    keep[WARMUP:] = score[WARMUP:] > thr
    ow = max(int(win * fps), 1)
    cap, oncap = [], []
    for a, b in meta["events"]:
        i0 = max(int(a * fps), 0)
        i1 = min(int(b * fps) + int(pad * fps), n - 1)
        if i1 < i0:
            continue
        hit = np.flatnonzero(keep[i0:i1 + 1])
        cap.append(1.0 if len(hit) else 0.0)
        oncap.append(1.0 if len(hit) and hit[0] < ow else 0.0)
    return (float(np.mean(cap)) if cap else float("nan"),
            float(np.mean(oncap)) if oncap else float("nan"))


def capture_two_ways(score, s_ref, meta, fps, n, rate):
    """Capture under BOTH calibration protocols.

    ``q_seg``: threshold = the held-out segment's own score quantile. This is
    an ORACLE-style calibration (it looks at the segment being evaluated) but
    it is the protocol the training-free curve used (runs/
    gate_operating_curve.json), so it is the only number directly comparable
    to it.

    ``q_tr``: threshold = the training segments' score quantile. Honest
    generalization, but it transfers badly whenever the training and
    held-out segments differ in event base rate (3.3%-26.3% here), so a low
    value here confounds calibration with signal.
    """
    q_seg = float(np.quantile(score[WARMUP:], 1.0 - rate))
    c1, o1 = capture(score, q_seg, meta, fps, n)
    q_tr = float(np.quantile(s_ref[WARMUP:], 1.0 - rate))
    c2, o2 = capture(score, q_tr, meta, fps, n)
    return {"cap_segq": c1, "on_segq": o1,
            "cap_trainq": c2, "on_trainq": o2,
            "thr_seg": q_seg, "thr_train": q_tr}


def sign_test(vals, null=0.5):
    v = np.asarray([x for x in vals if not np.isnan(x)], dtype=float)
    pos, neg = int((v > null).sum()), int((v < null).sum())
    n = pos + neg
    if n == 0:
        return float("nan"), pos, n
    k = min(pos, neg)
    return float(min(2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)), pos, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe_event")
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--out", default="runs/labeled_gate_upper_bound.json")
    args = ap.parse_args()

    seqs = seq_dirs(args.dataset)
    segs = {s: load_seg(s) for s in seqs}
    rows = []
    print(f"{'held-out':<10}{'variant':<10}{'AUC':>7}{'AUCbnd':>8}"
          f"{'capSeg':>8}{'onSeg':>7}{'capTr':>7}{'onTr':>7}")
    for val in seqs:
        tag = os.path.basename(val)
        tr = [s for s in seqs if s != val]
        f_val, y_val, b_val, fps, n, meta = segs[val]

        for variant in ("raw", "raw+time"):
            Xtr = np.vstack([segs[s][0] for s in tr])
            ytr = np.concatenate([segs[s][1] for s in tr])
            mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
            Xs = (Xtr - mu) / sd
            Xv = (f_val - mu) / sd
            if variant == "raw+time":
                ttr = np.vstack([temporal_feats(segs[s][0]) for s in tr])
                tmu, tsd = ttr.mean(axis=0), ttr.std(axis=0) + 1e-6
                Xs = np.hstack([Xs, (ttr - tmu) / tsd])
                tv = temporal_feats(f_val)
                Xv = np.hstack([Xv, (tv - tmu) / tsd])

            # deterministic reference-pool subsample
            if len(Xs) > POOL_MAX:
                step = len(Xs) / POOL_MAX
                idx = (np.arange(POOL_MAX) * step).astype(int)
                Xp, yp = Xs[idx], ytr[idx]
            else:
                Xp, yp = Xs, ytr

            score = knn_scores(Xp, yp, Xv, k=args.knn)
            a = auc(score, y_val)
            ab = auc(score, b_val)
            rec = {"seq": tag, "variant": variant, "auc": a,
                   "auc_bounds": ab, "n_events": len(meta["events"])}
            s_ref = None
            if variant == "raw":
                s_ref = knn_scores(Xp, yp, Xs[:6000], k=args.knn)
                rec.update(capture_two_ways(score, s_ref, meta, fps, n,
                                            args.rate))
            rows.append(rec)
            if variant == "raw":
                print(f"{tag:<10}{variant:<10}{a:>7.3f}{ab:>8.3f}"
                      f"{rec['cap_segq']:>8.3f}{rec['on_segq']:>7.3f}"
                      f"{rec['cap_trainq']:>7.3f}{rec['on_trainq']:>7.3f}",
                      flush=True)
            else:
                print(f"{tag:<10}{variant:<10}{a:>7.3f}{ab:>8.3f}"
                      f"{'—':>8}{'—':>7}{'—':>7}{'—':>7}", flush=True)

    print("\n=== 汇总（10 段中位 + 符号检验）===")
    summary = {}
    for variant in ("raw", "raw+time"):
        v = [r for r in rows if r["variant"] == variant]
        aucs = np.array([r["auc"] for r in v], dtype=float)
        abnd = np.array([r["auc_bounds"] for r in v], dtype=float)
        p1, pos1, n1 = sign_test(aucs)
        p2, pos2, n2 = sign_test(abnd)
        summary[variant] = {
            "auc_median": float(np.nanmedian(aucs)), "auc_p": p1,
            "auc_pos": pos1, "auc_n": n1,
            "auc_bounds_median": float(np.nanmedian(abnd)),
            "auc_bounds_p": p2, "auc_bounds_pos": pos2, "auc_bounds_n": n2,
            "per_seq_auc": aucs.tolist(),
            "per_seq_bounds": abnd.tolist(),
        }
        print(f"  {variant:<10} AUC={np.nanmedian(aucs):.3f} "
              f"({pos1}/{n1} 段>0.5, p={p1:.4f})   "
              f"AUC(bounds)={np.nanmedian(abnd):.3f} "
              f"({pos2}/{n2} 段>0.5, p={p2:.4f})")

    v = [r for r in rows if r["variant"] == "raw"]
    for key, label in (("cap_segq", "本段分位"), ("cap_trainq", "训练段分位")):
        caps = np.array([r[key] for r in v], dtype=float)
        onk = "on_segq" if key == "cap_segq" else "on_trainq"
        oncaps = np.array([r[onk] for r in v], dtype=float)
        print(f"  {label}: 区间捕获@{args.rate:.0%}={np.nanmedian(caps):.3f}  "
              f"onset 捕获@{args.rate:.0%}={np.nanmedian(oncaps):.3f}")
    print(f"  （对照：训练-free 残差 onset 捕获 @2-5% = 0.000–0.054；"
          f"随机撒帧 @2% = 0.782）")

    d = (np.array(summary["raw+time"]["per_seq_bounds"], dtype=float)
         - np.array(summary["raw"]["per_seq_bounds"], dtype=float))
    p_d, pos_d, n_d = sign_test(d, null=0.0)
    print(f"\n  raw+time − raw（bounds 目标，时序结构增量）："
          f"中位 {np.nanmedian(d):+.3f}，{pos_d}/{n_d} 段为正，p={p_d:.4f}")

    sd = same_dist_variant(args, seqs, segs)

    out = {"dataset": args.dataset, "knn": args.knn, "rate": args.rate,
           "pool_max": POOL_MAX, "warmup": WARMUP,
           "protocol": "leave-one-segment-out (10 folds); capture reported "
                       "under BOTH calibrations: per-segment quantile "
                       "(comparable to the training-free curve) and "
                       "training-segment quantile (honest generalization)",
           "rows": rows, "summary": summary, "same_dist": sd,
           "temporal_delta": {"median": float(np.nanmedian(d)),
                              "pos": pos_d, "n": n_d, "p": p_d},
           "notes": [
               "raw = 576-d frozen pooled features -> kNN label consistency. "
               "STATIC appearance only; no temporal context at all.",
               "raw+time = raw + [running-EMA novelty, frame-to-frame "
               "change] == the two shapes the training-free probe surfaced.",
               "Directly comparable to the training-free numbers: frame AUC "
               "0.510, onset capture 0.000 (runs/gate_operating_curve.json).",
               "No sklearn available: kNN via Gram-form distances with "
               "deterministic pool subsampling.",
               "IMPORTANT scope note: leave-one-SEGMENT-out is cross-scene "
               "transfer. A deployed always-on device sees its OWN scene, so "
               "this protocol measures the harder problem; see the same-"
               "distribution variant in runs/labeled_gate_same_dist.json.",
           ]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


def same_dist_variant(args, seqs, segs, n_blocks=10):
    """Within-scene version: interleaved contiguous blocks, even blocks as the
    reference pool and odd blocks as queries. This answers "if you had labels
    for THIS scene, would the signal be usable?" -- the optimistic bound.

    A plain first-half/second-half split was tried first and is WRONG for this
    data: events are bunched, so 6 of 10 segments have zero event frames in the
    second half (measured), leaving the pool with no positives and degenerating
    the score to a constant (AUC exactly 0.500, no frames above threshold).

    CAVEAT, stated because it flatters the number: adjacent frames are nearly
    identical, so blocks cut through an event let the query side see
    near-duplicates of pool frames. This is therefore an OPTIMISTIC bound, not
    a deployment number; the cross-scene LOO table above is the honest one.
    """
    print(f"\n=== 同分布变体（交错分块：偶数块为池、奇数块为查询）===")
    print("  （乐观上界：块边界会切开事件，相邻帧近似重复；部署口径看上面的跨段表）")
    print(f"{'segment':<10}{'AUC':>7}{'AUCbnd':>8}{'cap':>7}{'onCap':>7}")
    rows = []
    for seq in seqs:
        tag = os.path.basename(seq)
        f, y, b, fps, n, meta = segs[seq]
        if len(meta["events"]) < 2 or n < n_blocks * 60:
            continue
        edges = (np.arange(n_blocks + 1) * n / n_blocks).astype(int)
        pool_idx, q_idx = [], []
        for k in range(n_blocks):
            (pool_idx if k % 2 == 0 else q_idx).append(
                np.arange(edges[k], edges[k + 1]))
        pi = np.concatenate(pool_idx)
        qi = np.concatenate(q_idx)
        Xp_raw, yp = f[pi], y[pi]
        if yp.sum() == 0 or yp.sum() == len(yp):
            continue
        mu, sd = Xp_raw.mean(axis=0), Xp_raw.std(axis=0) + 1e-6
        Xp = (Xp_raw - mu) / sd
        Xq = (f[qi] - mu) / sd
        score = knn_scores(Xp, yp, Xq, k=args.knn)
        a = auc(score, y[qi], warm=0)
        ab = auc(score, b[qi], warm=0)
        # capture: events whose interval falls entirely inside a query block
        ev = []
        for s0, s1 in meta["events"]:
            i0, i1 = int(s0 * fps), int(s1 * fps)
            if i1 <= i0:
                continue
            in_q = np.isin(np.arange(i0, min(i1, n)), qi).mean()
            if in_q > 0.9:
                ev.append([s0, s1])
        cap = {"cap_segq": float("nan"), "on_segq": float("nan")}
        if ev:
            full = np.zeros(n, dtype=np.float32)
            full[qi] = score
            ranked = np.zeros(n, dtype=bool)
            for j, qq in enumerate(qi):
                ranked[qq] = score[j] > np.quantile(score, 1 - args.rate)
            ow = max(int(ONSET_WIN * fps), 1)
            mc, mo = [], []
            for s0, s1 in ev:
                i0, i1 = int(s0 * fps), min(int(s1 * fps), n - 1)
                hit = np.flatnonzero(ranked[i0:i1 + 1])
                mc.append(1.0 if len(hit) else 0.0)
                mo.append(1.0 if len(hit) and hit[0] < ow else 0.0)
            cap = {"cap_segq": float(np.mean(mc)), "on_segq": float(np.mean(mo))}
        rows.append({"seq": tag, "auc": a, "auc_bounds": ab, "n_events_eval": len(ev),
                     **cap})
        print(f"{tag:<10}{a:>7.3f}{ab:>8.3f}{cap['cap_segq']:>7.3f}"
              f"{cap['on_segq']:>7.3f}", flush=True)
    if not rows:
        return {}
    aucs = np.array([r["auc"] for r in rows], dtype=float)
    abnd = np.array([r["auc_bounds"] for r in rows], dtype=float)
    p1, pos1, n1 = sign_test(aucs)
    p2, pos2, n2 = sign_test(abnd)
    caps = np.array([r["cap_segq"] for r in rows], dtype=float)
    onc = np.array([r["on_segq"] for r in rows], dtype=float)
    print(f"  中位: AUC={np.nanmedian(aucs):.3f} ({pos1}/{n1} 段>0.5, p={p1:.4f})  "
          f"AUC(bounds)={np.nanmedian(abnd):.3f} ({pos2}/{n2}, p={p2:.4f})")
    print(f"  区间捕获={np.nanmedian(caps):.3f}  onset 捕获={np.nanmedian(onc):.3f}")
    return {"rows": rows, "n_segments": len(rows),
            "auc_median": float(np.nanmedian(aucs)), "auc_p": p1,
            "auc_pos": pos1, "auc_n": n1,
            "auc_bounds_median": float(np.nanmedian(abnd)),
            "auc_bounds_p": p2, "auc_bounds_pos": pos2, "auc_bounds_n": n2,
            "capture_median": float(np.nanmedian(caps)),
            "onset_capture_median": float(np.nanmedian(onc)),
            "blocks": n_blocks}


if __name__ == "__main__":
    main()

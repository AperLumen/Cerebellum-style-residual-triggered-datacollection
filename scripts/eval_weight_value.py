#!/usr/bin/env python3
"""判据：`weight` 字段值不值？（把"未验证主张"变成有数字的结论）

## 背景

数据集格式（GUIDE §4.2）给每帧存了一个 `weight`：

    w_t = clip(surprise_t / median(surprise), 1e-3, 10)

它的**主张**是"残余不可压缩性 = 每字节信息密度"，用途是**均衡难度采样**。
但这是 CLAIM, NOT ESTABLISHED——本脚本就是那个判据。在此之前，manifest 与
自动生成的 README 都必须把它写成假设（它们也确实这么写了）。

## 判据设计（三条纪律）

1. **同等字节预算**比，不是同等帧数。`weight` 的卖点是"每字节信息量"，
   用帧数对齐等于把要检验的东西假设掉了。
2. 指标是**下游 mAP / 每字节信息增益**，**不是每帧命中率**——后者已经被
   §2.3.8 证伪（残差不携带事件时刻信息）。
3. **必须给上界（真值难度采样）**，否则"weigh 打赢均匀"说明不了它接近最优。
   同时必须给**随机基线多次采样**的分布，否则单次抽样的差异会被误读成信号。

## 三种采样

| 臂 | 采样概率 ∝ |
| --- | --- |
| `weight`  | `weight_t`（本格式的主张） |
| `uniform` | 常数（基线） |
| `oracle`  | 真值难度（上界）——用 labels 的"难"程度代理 |
| `random`  | 每次独立随机（用于得到 sampling noise 的尺度） |

## 为什么 oracle 用这个定义

真值难度没有唯一答案。这里用 **label 变化率**（事件边界帧）作代理：它代表
"这一帧在语义上发生了转折"，是下游分类最可能需要的帧。这是**代理量**不是
真值，所以在报告里明确标注，不能被当成绝对上界。

## 用法

    PYTHONPATH=. .venv/bin/python scripts/eval_weight_value.py
    PYTHONPATH=. .venv/bin/python scripts/eval_weight_value.py --n-seeds 25
    PYTHONPATH=. .venv/bin/python scripts/eval_weight_value.py --budget 0.05

输出写到 `runs/eval_weight_value.json`（gitignore），并把结论打到 stdout。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cerebellum_cancel.dataset import load_dataset, load_section  # noqa: E402

DEFAULT_DS = os.path.join(ROOT, "data", "cmci_ds_aoe_event")
DEFAULT_OUT = os.path.join(ROOT, "runs", "eval_weight_value.json")

# **macOS Accelerate BLAS 在 float64 matmul 上会发假警报**：
# 对完全有限的 A/w 报 "divide by zero / overflow / invalid value encountered
# in matmul"，每个元素一条（实测一次调用 1803 条）。已核实是假警报——
# 用手写参考实现（不经过任何本项目代码）同样触发，且 w 保持有限
# （max|w| ≈ 1.9）、AP 可复现（同一 seed 两次结果逐位相同）。
#
# 这里只屏蔽这三条 matmul 警告，**不**全局过滤——真正的数值问题（比如真的
# 发散出 inf）仍会报警，并由 fit_eval 里的有限性检查兜住。
warnings.filterwarnings(
    "ignore", message=r".*(divide by zero|overflow|invalid value).*matmul",
    category=RuntimeWarning)


# ------------------------------------------------------------------ 数据装载

def load_feats(section_id: str, n_frames: int, dim: int | None) -> np.ndarray | None:
    """取该段的 backbone 特征（下游读出的输入）。

    数据集本身**不存特征**（只存残差侧的量与可选帧），所以这里从各数据源的
    缓存里找。找不到就返回 None，由调用方降级为"只用残差侧特征"。
    """
    cands = []
    for base in ("data/aoe_event", "data/real_imu", "data/aoe"):
        for split in ("val", "train"):
            cands.append(os.path.join(ROOT, base, split, section_id, "features.npy"))
            cands.append(os.path.join(ROOT, base, split, section_id,
                                      "featmaps.npy"))
    for p in cands:
        if os.path.exists(p):
            try:
                a = np.load(p, mmap_mode="r")
            except Exception:
                continue
            if a.ndim == 1:
                continue
            flat = a.reshape(len(a), -1)
            if len(flat) >= n_frames and flat.shape[1] == (dim or flat.shape[1]):
                return np.asarray(flat[:n_frames], dtype=np.float32)
            # 数量对得上、维度不同也接受（下游自己适应）
            if len(flat) >= n_frames:
                return np.asarray(flat[:n_frames], dtype=np.float32)
    return None


def build_pool(ds_dir: str, max_side: int = 32):
    """把所有段拼成一个池，并按段切分以便做 leave-one-section-out。

    返回 list[dict]，每项是一段：X(帧特征) / y(标签) / w(weight) /
    difficulty(真值难度代理) / section_id。
    """
    man = load_dataset(ds_dir)
    segs = []
    for meta in man["sections"]:
        sid = meta["section_id"]
        sec = load_section(ds_dir, sid)
        n = sec.n_frames
        p = sec.payload
        if "labels" not in p:
            print(f"  [跳过] {sid}: 没有 labels，无法定义上游任务")
            continue
        dim = meta.get("features_dim")
        X = load_feats(sid, n, dim)
        if X is None:
            # 降级：用数据集内的残差侧量当特征（信息量低，但可跑）
            # 目的只是让脚本在无特征缓存时也能给出数字，而不是静默跳过。
            X = np.stack([p["surprise"], np.abs(p["gyro"]).mean(axis=1),
                          p["weight"]], axis=1).astype(np.float32)
            degraded = True
        else:
            degraded = False
        y = np.asarray(p["labels"], dtype=np.float64).ravel()
        # 连续标签 → 二值事件（>0 视为事件），与 §2.3.7 的口径一致
        yb = (y > 0).astype(np.int64)

        w = np.asarray(p["weight"], dtype=np.float64).ravel()

        # 真值难度代理 = label 变化率（事件边界处最大）
        # 用滑动窗内的变化次数，边界帧得高分。
        ch = np.zeros(n, dtype=np.float64)
        d = np.abs(np.diff(yb, prepend=yb[:1]))
        for k in (-2, -1, 0, 1, 2):
            ch += np.roll(d, k)
        # 叠一个"事件帧"的微弱偏好：单看边界会导致全是 0 的段退化为常数
        ch = ch + 0.1 * yb
        ch = np.clip(ch, 1e-6, None)

        segs.append({"section_id": sid, "X": X, "y": yb, "w": w,
                     "difficulty": ch, "degraded_features": degraded})
    return segs


# ------------------------------------------------------------------ 采样

def sample_idx(p: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """按概率 p 无放回地抽 k 个（k >= 非零项数时直接返回非零项）。"""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        return rng.choice(len(p), size=min(k, len(p)), replace=False)
    p = p / s
    k = min(k, int((p > 0).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    return rng.choice(len(p), size=k, replace=False, p=p)


def fit_eval(Xtr: np.ndarray, ytr: np.ndarray,
             Xte: np.ndarray, yte: np.ndarray, seed: int) -> float:
    """逻辑回归（numpy 实现，项目不装 sklearn）→ AP。

    与 §2.3.7 的读出协议一致：标准化 + 梯度下降 + one-vs-rest。

    **数值稳定性（实测踩过）**：学习率必须随维度收缩。576 维上 `lr=0.5`
    会发散（`max|z|` 从 2.7 涨到 25 再溢出成 inf），而 inf 一旦进入
    `average_precision` 会**静默产出看似正常的 AP**——所以本函数在最后
    检查有限性，不有限就返回 nan 让调用方丢弃该折。

    收缩策略：`lr ∝ 1/sqrt(d)`（标准做法），并对梯度做范数裁剪。
    """
    if len(np.unique(ytr)) < 2:
        return float("nan")
    d = Xtr.shape[1]
    # **必须升到 float64**：特征缓存是 float32，576 维点积在 float32 下会
    # 溢出（实测 `divide by zero encountered in matmul`）。这不是"精度差一点"，
    # 是直接产出 inf → AP 变假值。
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    A = np.concatenate([A, np.ones((len(A), 1), dtype=A.dtype)], axis=1)
    B = np.concatenate([B, np.ones((len(B), 1), dtype=B.dtype)], axis=1)

    # 自适应学习率：维度越高步子越小（否则发散）
    lr = 0.5 / np.sqrt(d / 4.0)
    lr = float(min(lr, 0.5))
    l2 = 1e-3
    w = np.zeros(A.shape[1], dtype=np.float64)
    ytr_f = ytr.astype(np.float64)

    for _ in range(300):
        z = np.clip(A @ w, -30.0, 30.0)
        pr = 1.0 / (1.0 + np.exp(-z))
        g = A.T @ (pr - ytr_f) / len(A) + l2 * w
        gn = float(np.linalg.norm(g))
        if gn > 10.0:                       # 梯度裁剪，兜住偶发的大步
            g = g * (10.0 / gn)
        w -= lr * g
        if not np.all(np.isfinite(w)):
            return float("nan")             # 发散必须显式失败，不产出假 AP

    sco = B @ w
    if not np.all(np.isfinite(sco)):
        return float("nan")
    return average_precision(yte, sco)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """AP（不依赖 sklearn）：按分数降序累积 precision 对 recall 积分。

    **必须先检查有限性**：实测见过 `score` 含 inf 时 AP 仍返回一个
    "看起来正常"的小数（因为 argsort 把 inf 排在两端、积分照样算得出来）。
    这种静默假值比报错危险得多，所以这里直接返回 nan。
    """
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    if not np.all(np.isfinite(score)):
        return float("nan")
    order = np.argsort(-score)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / y.sum()
    # 阶梯积分
    ap = float(np.sum(prec * np.diff(np.concatenate([[0.0], rec]))))
    return ap


# ------------------------------------------------------------------ 主流程

def main() -> int:
    ap_ = argparse.ArgumentParser(description="weight 字段价值判据")
    ap_.add_argument("--dataset", default=DEFAULT_DS)
    ap_.add_argument("--out", default=DEFAULT_OUT)
    ap_.add_argument("--n-seeds", type=int, default=15)
    ap_.add_argument("--budget", type=float, default=0.10,
                     help="每折用于训练的**帧比例**（字节预算的代理）")
    ap_.add_argument("--max-frames", type=int, default=1500,
                     help="每段最多取多少帧（控制运行时间）")
    args = ap_.parse_args()

    if not os.path.exists(args.dataset):
        print(f"找不到数据集 {args.dataset}", file=sys.stderr)
        print("先构建：PYTHONPATH=. .venv/bin/python scripts/build_dataset.py "
              "--source data/aoe_event --out data/cmci_ds_aoe_event "
              "--frames none --max-frames 6000", file=sys.stderr)
        return 2

    print("=" * 62)
    print("weight 字段价值判据（leave-one-section-out）")
    print("=" * 62)
    segs = build_pool(args.dataset)
    if len(segs) < 3:
        print(f"段数不足（{len(segs)}），无法做 LOO", file=sys.stderr)
        return 2

    for s in segs:
        s["X"] = s["X"][:args.max_frames]
        s["y"] = s["y"][:args.max_frames]
        s["w"] = s["w"][:args.max_frames]
        s["difficulty"] = s["difficulty"][:args.max_frames]

    n_deg = sum(s["degraded_features"] for s in segs)
    print(f"段数 {len(segs)}, 每段 ≤{args.max_frames} 帧, "
          f"预算 {args.budget:.0%} 帧/折, seeds {args.n_seeds}")
    if n_deg:
        print(f"  ⚠ {n_deg}/{len(segs)} 段没有特征缓存，已降级用残差侧量当特征"
              "（结论的信息量会降低，应视为机制跑通而非终值）")
    print()

    arms = ["uniform", "weight", "oracle", "random"]
    scores: dict[str, list[float]] = {a: [] for a in arms}

    for hi in range(len(segs)):
        te = segs[hi]
        tr = [s for i, s in enumerate(segs) if i != hi]
        # 训练池 = 其余段拼接
        Xp = np.concatenate([s["X"] for s in tr])
        yp = np.concatenate([s["y"] for s in tr])
        wp = np.concatenate([s["w"] for s in tr])
        dp = np.concatenate([s["difficulty"] for s in tr])

        n_pool = len(Xp)
        k = max(20, int(round(args.budget * n_pool)))

        # oracle 的"难度"要归一成概率
        dnorm = dp / dp.sum()
        wnorm = wp / wp.sum()

        for seed in range(args.n_seeds):
            rng = np.random.default_rng(1000 * hi + seed)
            picks = {
                "uniform": sample_idx(np.ones(n_pool), k, rng),
                "weight": sample_idx(wnorm, k, rng),
                "oracle": sample_idx(dnorm, k, rng),
                "random": sample_idx(np.ones(n_pool), k, rng),
            }
            for arm, idx in picks.items():
                if len(idx) < 10:
                    continue
                # 若抽样后只剩单一类别，该折该臂无信息
                if len(np.unique(yp[idx])) < 2:
                    continue
                a = fit_eval(Xp[idx], yp[idx], te["X"], te["y"], seed)
                if np.isfinite(a):
                    scores[arm].append(float(a))

    def stat(v: list[float]) -> tuple[float, float, int]:
        if not v:
            return float("nan"), float("nan"), 0
        return float(np.mean(v)), float(np.std(v)), len(v)

    results = {}
    print(f"{'臂':<10} {'AP (mean±sd)':<22} {'n':>5}")
    print("-" * 42)
    for a in arms:
        m, s, n = stat(scores[a])
        results[a] = {"mean_ap": m, "std_ap": s, "n": n}
        print(f"{a:<10} {m:.4f} ± {s:.4f}       {n:>5}")
    print()

    # ---- 结论（按预设判据）
    m_u, s_u, _ = stat(scores["uniform"])
    m_w, s_w, _ = stat(scores["weight"])
    m_o, s_o, _ = stat(scores["oracle"])
    m_r, s_r, _ = stat(scores["random"])

    verdict = "INCONCLUSIVE"
    note = ""
    if np.isfinite(m_w) and np.isfinite(m_u) and np.isfinite(s_u):
        # 用 random 臂估 sampling noise 的尺度（它才是纯噪声）
        noise = max(s_r, s_u, 1e-6)
        diff = m_w - m_u
        results["delta_weight_minus_uniform"] = float(diff)
        results["noise_scale_from_random"] = float(noise)
        if diff > 1.0 * noise:
            verdict = "WEIGHT_BETTER"
            note = (f"weight 比 uniform 高 {diff:+.4f}（> 1× 采样噪声 {noise:.4f}）"
                    "——主张获得初步支持")
        elif diff < -1.0 * noise:
            verdict = "WEIGHT_WORSE"
            note = (f"weight 比 uniform 低 {diff:+.4f}（> 1× 采样噪声 {noise:.4f}）"
                    "——主张被否，`weight` 应降级为诊断用")
        else:
            note = (f"weight 与 uniform 差 {diff:+.4f}，在采样噪声 "
                    f"({noise:.4f}) 以内——**无法区分**")

    print("=" * 62)
    print(f"判据结论: {verdict}")
    print(f"  {note}")
    if np.isfinite(m_o):
        print(f"  参照（真值难度代理上界）: {m_o:.4f} ± {s_o:.4f}")
        gap = m_o - max(m_w, m_u)
        print(f"  与上界的差距: {gap:+.4f}"
              + ("（两种采样都没接近上界）" if gap > 1.0 * max(s_o, 1e-6)
                 else "（已接近上界，提升空间有限）"))
        if m_o < m_w:
            print("  ⚠ **oracle 低于 weight**：说明代理量不是有效上界。"
                  "下面的一致性诊断才是主证据。")
    print()

    # ---- 一致性诊断（**比聚合 AP 更锋利**）
    #
    # 为什么需要它：apex AP 被"特征本身对任务的判别力"主导——三种采样用的
    # 是同一批特征、同一个读出，AP 的差异被这个共同成分压缩。实测就是
    # 这样：weight 与 uniform 差 +0.02 而采样噪声 0.09，看不出区别。
    #
    # 但"被选中的帧里正例占多少"是**不受读出能力影响**的直接度量：
    # 如果 weight 有效，它的 top-k 应该富集正例；如果它选的帧正例率与基率
    # 一样，那这个权重对"选到有用的帧"毫无帮助。
    print("一致性诊断（正例率富集，免疫读出能力）")
    print("-" * 62)
    print(f"{'段':<12} {'基率':>7} {'weight':>8} {'difficulty':>11} {'随机p95':>9}")
    enrich_rows = []
    for s in segs:
        y = s["y"]
        n = len(y)
        k = max(10, int(0.10 * n))
        base = float(y.mean())
        rng = np.random.default_rng(7)
        rand_hi = []
        for _ in range(200):
            ridx = rng.choice(n, size=k, replace=False)
            rand_hi.append(float(y[ridx].mean()))
        p95 = float(np.percentile(rand_hi, 95))
        w_top = float(y[np.argsort(-s["w"])[:k]].mean())
        d_top = float(y[np.argsort(-s["difficulty"])[:k]].mean())
        enrich_rows.append({"section": s["section_id"], "base_rate": base,
                            "weight_top10": w_top, "difficulty_top10": d_top,
                            "random_p95": p95})
        print(f"{s['section_id']:<12} {base:>7.3f} {w_top:>8.3f} "
              f"{d_top:>11.3f} {p95:>9.3f}")

    w_vals = np.array([r["weight_top10"] for r in enrich_rows])
    b_vals = np.array([r["base_rate"] for r in enrich_rows])
    p_vals = np.array([r["random_p95"] for r in enrich_rows])
    d_vals = np.array([r["difficulty_top10"] for r in enrich_rows])
    n_w_beat = int((w_vals > p_vals).sum())
    print()
    print(f"  weight top10% 超过随机 p95 的段数: {n_w_beat}/{len(enrich_rows)}")
    print(f"  difficulty top10% 超过随机 p95 的段数: "
          f"{int((d_vals > p_vals).sum())}/{len(enrich_rows)}")
    ratio = float(np.median(w_vals / np.maximum(b_vals, 1e-9)))
    print(f"  weight top10% 正例率 / 基率 的中位比: {ratio:.3f}"
          "（≈1 表示与随机无异）")
    results["enrichment"] = {
        "per_section": enrich_rows,
        "weight_sections_beating_random_p95": n_w_beat,
        "difficulty_sections_beating_random_p95": int((d_vals > p_vals).sum()),
        "weight_top10_over_base_rate_median_ratio": ratio,
    }

    # 如果一致性诊断也不支持，把结论从 INCONCLUSIVE 收紧为否定——
    # "统计上无法区分"与"实测与随机无异"是两件事，后者证据更强。
    if verdict in ("INCONCLUSIVE", "WEIGHT_WORSE") and n_w_beat <= len(enrich_rows) // 2:
        verdict = "WEIGHT_NOT_SUPPORTED"
        results["verdict"] = verdict
        print()
        print("  ⇒ 一致性诊断不支持该权重：它选中的帧里正例比例与随机抽样"
              "无异。")
        print("     结论收紧为 WEIGHT_NOT_SUPPORTED（比'无法区分'更强的否定）。")

    print()
    print("  重要限定：oracle 用的是**代理量**（label 变化率），不是真值难度，"
          "所以它是参照而非绝对上界。")
    print("  负结果照记——若 weight 不优于 uniform，应把该字段降级为诊断用，"
          "并同步更新 GUIDE §4.2 与 manifest。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "verdict": verdict,
        "note": note,
        "protocol": {
            "scheme": "leave-one-section-out",
            "n_seeds": args.n_seeds,
            "budget_frames_per_fold": args.budget,
            "max_frames_per_section": args.max_frames,
            "metric": "frame-level AP (event = labels > 0)",
            "oracle_definition": "label change rate (PROXY, not ground truth)",
            "n_sections": len(segs),
            "sections_with_degraded_features": n_deg,
        },
        "arms": results,
        "caveats": [
            "oracle 是代理量（label 变化率），非真值难度上界",
            "数据集不存特征；特征来自各数据源缓存，缺失时降级为残差侧量",
            "本判据测的是**采样价值**，与此前证伪的**定位价值**（§2.3.8）不同",
        ],
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=1, ensure_ascii=False)
    print(f"\n结果写入: {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

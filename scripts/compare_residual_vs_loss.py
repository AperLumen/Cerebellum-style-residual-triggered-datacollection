"""残差 vs 模型自身 loss：谁能找到标注错误？

## 为什么必须做这个对照

用户的目标（"在垃圾山找 <1% 的金子"）在数据清洗领域有成熟基线：**用模型自己
的训练 loss / 预测置信度排序**（confident learning、small-loss trick）。
任何"用残差找错"的主张都必须先跟它比，否则可能只是把同一件事做得更贵。

本项目已具备全部要素，`data/aoe_event` 的 10 段同时含 `featmaps.npy` /
`features.npy` / `labels.npy`，三者帧数严格对齐：

* **残差**：`FeatureMapCancel(pred='rotation')` 的 surprise——**标签无关**。
* **模型 loss**：在冻结 576 维 `features.npy` 上训练逐帧二分类器，取交叉熵。
  这是**标签相关**的经典清洗基线。

（本项目环境无 sklearn，逻辑回归用 numpy 手写——见 `_fit_logreg`。这样也避免
"基线用了更好的优化器"这类混淆。）

## 检验的核心不是"谁更强"，而是"残差是否有独立价值"

loss 是标签的函数——标注错得越厉害 loss 越高，所以它天然能找到"与标签冲突剧
烈的帧"。但它因此对一类错误**结构性失明**：当某个错误在数据里足够普遍、模型
把它学成了"正确"，loss 就降下来了。残差不知道标签，不受此影响。

所以本脚本报四件事：

1. 两个信号各自的 lift（对每种注入）
2. 二者的 **Spearman 相关**（高度相关 -> 残差无独立价值）
3. **互补性**：两个信号各取 k 名的**并集**覆盖率（对照各取 k 名）
4. **加权混合**：标准化后等权相加再排序

输出：runs/label_audit_vs_loss.json
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

from audit_planted_labels import (inject_barely_shift, inject_drop_short,
                                 inject_extend_far, touch_mask)
from probe_label_audit import intervals_to_mask
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel)
from cerebellum_cancel.io import load_imu_csv


def _fit_logreg(X: np.ndarray, y: np.ndarray, iters: int = 300,
                l2: float = 1.0, lr: float = 0.5) -> np.ndarray:
    """带 L2 的梯度下降逻辑回归（避免引入 sklearn 依赖）。

    X 已标准化；返回 [w, b] 拼成的向量。300 次全批量迭代在这个规模
    （<=1500 帧 x 576 维）上足够收敛，且比任何外部实现都更容易复现。
    """
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        gw = X.T @ g / n + l2 * w / n
        gb = float(g.mean())
        w -= lr * gw
        b -= lr * gb
    return np.concatenate([w, [b]])


def _predict_logreg(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    z = X @ params[:-1] + params[-1]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def residual_signal(seq: str, n: int) -> np.ndarray:
    F = np.load(f"{seq}/featmaps.npy", mmap_mode="r")
    gy = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                  dtype=np.float32)
    meta = json.load(open(f"{seq}/events.json"))
    cfg = CancelConfig(fps=float(meta.get("fps", 30.0)),
                       f_map=float(meta["f_map"]), robust=True,
                       pred="rotation", source=seq)
    canc = FeatureMapCancel(cfg)
    s = np.zeros(n, dtype=np.float64)
    for t in range(n):
        s[t] = canc.step(np.asarray(F[t], dtype=np.float32), gy[t])["surprise"]
    return s


def loss_signal(feats: np.ndarray, labels: np.ndarray,
                idx_train: np.ndarray) -> np.ndarray:
    """逐帧交叉熵——数据清洗的经典基线（small-loss / confident learning）。"""
    y = (labels > 0.5).astype(np.float64)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    X = (feats - mu) / sd
    if len(np.unique(y[idx_train])) < 2:
        return np.zeros(len(labels))
    params = _fit_logreg(X[idx_train], y[idx_train])
    p = _predict_logreg(X, params)
    eps = 1e-7
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def loss_signal_retrained(feats: np.ndarray, noisy: np.ndarray,
                          idx_train: np.ndarray) -> np.ndarray:
    """**真实清洗场景**的 loss：在（已被污染的）标签上训练后再取交叉熵。

    这是与上面那个函数的关键区别，也是本脚本最重要的方法学判断：

    * 上一版用**原始干净标签**训练模型 -> 模型知道"正确边界在哪"，于是
      边界错位会立刻表现为高 loss（barely_shift 86x）。但真实场景里没有
      干净标签可训练——你手里的标签就是脏的。
    * 本节在**污染后**的标签上训练 -> 模型把错误也学进去一部分。这是
      confident learning / small-loss 真正面对的设定，也是唯一诚实的设定。
      若这个设定下 loss 的优势消失，那"残差标签无关"就不再是学术趣味，
      而是**真实的结构性优势**。
    """
    y = (noisy > 0.5).astype(np.float64)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    X = (feats - mu) / sd
    if len(np.unique(y[idx_train])) < 2:
        return np.zeros(len(noisy))
    params = _fit_logreg(X[idx_train], y[idx_train])
    p = _predict_logreg(X, params)
    eps = 1e-7
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def lift(score: np.ndarray, pos: np.ndarray, evals: np.ndarray) -> float:
    """lift = precision@k / (k/N)，k = 真值正类帧数。"""
    if len(pos) == 0:
        return float("nan")
    k = max(int(len(pos)), 1)
    posset = set(pos.tolist())
    ranked = evals[np.argsort(-score[evals])][:k]
    hit = sum(int(f in posset) for f in ranked)
    return (hit / k) / (k / len(evals))


def union_lift(a: np.ndarray, b: np.ndarray, pos: np.ndarray,
               evals: np.ndarray) -> float:
    """两个信号各取 k 名的并集覆盖率 / (|union|/N)——互补性指标。"""
    if len(pos) == 0:
        return float("nan")
    k = max(int(len(pos)), 1)
    posset = set(pos.tolist())
    ra = set(evals[np.argsort(-a[evals])][:k].tolist())
    rb = set(evals[np.argsort(-b[evals])][:k].tolist())
    u = ra | rb
    if not u:
        return float("nan")
    return (len(posset & u) / len(u)) / (len(u) / len(evals))


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-9)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman = 秩上的 Pearson。用 numpy 实现以避免 scipy 版本差异。"""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_event")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--skip-first", type=int, default=60)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/label_audit_vs_loss.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seqs = []
    for st in args.sets.split(","):
        seqs += sorted(glob.glob(f"{st.strip()}/val/seq_*"))

    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out))
        if len(old.get("seqs", [])) > len(seqs):
            raise SystemExit(f"拒绝覆盖 {args.out}。加 --force。")

    rng = np.random.default_rng(args.seed)
    kinds = ("drop_short", "extend_far", "barely_shift")
    rows = []

    for seq in seqs:
        if not all(os.path.exists(f"{seq}/{k}.npy")
                   for k in ("featmaps", "features", "labels")):
            continue
        feats = np.load(f"{seq}/features.npy").astype(np.float32)
        labels = np.load(f"{seq}/labels.npy").astype(np.float32)
        meta = json.load(open(f"{seq}/events.json"))
        fps = float(meta.get("fps", 30.0))
        n = min(len(feats), len(labels), args.n)

        s_res = residual_signal(seq, n)[:n]
        lo = min(args.skip_first, n // 4)
        evals = np.arange(lo, n)
        # 参照上限：在**干净**标签上训练（真实清洗场景拿不到，仅为定标）
        tr = evals[labels[evals] >= 0]
        tr = tr if len(tr) > 50 else evals
        s_loss_clean = loss_signal(feats[:n], labels[:n], tr)

        rho = spearman(s_res[evals], s_loss_clean[evals])
        row: dict = {"seq": seq, "n": n, "spearman_res_vs_loss": rho,
                     "kinds": {}}
        for kind in kinds:
            ev = [list(e) for e in meta.get("events", [])]
            orig = intervals_to_mask(ev, fps, n)
            lr, lc, le, lu = [], [], [], []
            npos = 0
            for _ in range(args.reps):
                if kind == "drop_short":
                    nev, _ = inject_drop_short(ev, fps, n, rng)
                elif kind == "extend_far":
                    nev, _ = inject_extend_far(ev, fps, n, rng)
                else:
                    nev, _ = inject_barely_shift(ev, fps, n, rng)
                new = intervals_to_mask(nev, fps, n)
                pos = np.flatnonzero(touch_mask(orig, new))
                pos = pos[pos >= lo]
                if len(pos) == 0:
                    continue
                npos = len(pos)
                # 真实设定：手里只有**被污染后**的标签，只能在它上面训练
                noisy = new.astype(np.float32)
                s_loss_ret = loss_signal_retrained(feats[:n], noisy, tr)
                lr.append(lift(s_res, pos, evals))
                lc.append(lift(s_loss_clean, pos, evals))
                le.append(lift(s_loss_ret, pos, evals))
                lu.append(union_lift(s_res, s_loss_ret, pos, evals))
            if lr:
                row["kinds"][kind] = {
                    "n_pos_mean": int(npos),
                    "lift_residual": float(np.nanmean(lr)),
                    "lift_loss_clean_upper": float(np.nanmean(lc)),
                    "lift_loss_retrained": float(np.nanmean(le)),
                    "lift_union": float(np.nanmean(lu)),
                }
        rows.append(row)
        d = row["kinds"].get("drop_short", {})
        print(f"  {seq.split('/')[-1]:9s} n={n:5d} rho={rho:+.2f}  "
              f"drop_short: res {d.get('lift_residual', float('nan')):6.2f}x  "
              f"loss(clean) {d.get('lift_loss_clean_upper', float('nan')):6.2f}x  "
              f"loss(retrain) {d.get('lift_loss_retrained', float('nan')):6.2f}x")

    print(f"\n=== 汇总（{len(rows)} 段）===")
    print(f"{'注入类型':<14}{'残差':>9}{'loss(干净)':>12}{'loss(重训)':>12}{'并集':>9}")
    summary = {}
    for kind in kinds:
        got = [r["kinds"][kind] for r in rows if kind in r["kinds"]]
        if not got:
            continue
        summary[kind] = {
            k: float(np.nanmean([g[k] for g in got]))
            for k in ("lift_residual", "lift_loss_clean_upper",
                      "lift_loss_retrained", "lift_union")
        }
        v = summary[kind]
        print(f"{kind:<14}{v['lift_residual']:>8.2f}x"
              f"{v['lift_loss_clean_upper']:>11.2f}x"
              f"{v['lift_loss_retrained']:>11.2f}x{v['lift_union']:>8.2f}x")
    rho = np.array([r["spearman_res_vs_loss"] for r in rows])
    print(f"\n  残差 vs loss 的 Spearman 相关（{len(rho)} 段）: "
          f"均值 {rho.mean():+.3f}  中位 {np.median(rho):+.3f}")
    print("  判读：loss(干净) 是参照上限（真实清洗拿不到）；")
    print("        loss(重训) 才是可比基线。若它掉到 ≈ 残差，")
    print("        则'标签无关'是真实的结构性优势。")

    report = {
        "sets": args.sets, "n_frames_per_seq": args.n, "reps": args.reps,
        "seed": args.seed,
        "predictor_residual": "FeatureMapCancel(pred='rotation', robust=True)"
                              ".surprise —— 标签无关",
        "predictor_loss_clean_upper": "在**干净**标签上训练的逐帧交叉熵"
                                      "（清洗场景拿不到，仅作参照上限）",
        "predictor_loss_retrained": "在**注入后污染**的标签上重新训练的逐帧"
                                    "交叉熵 —— 真实清洗场景的可比基线",
        "metric": "lift = precision@k / (k/N)，k = 注入影响的帧数",
        "note": "关键方法学判断：现实中只能在脏标签上训练，所以 loss 的真实"
                "基线是 retrained 那一列，不是 clean 那一列。",
        "spearman_res_vs_loss_mean": float(rho.mean()),
        "spearman_res_vs_loss_median": float(np.median(rho)),
        "summary": summary,
        "seqs": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

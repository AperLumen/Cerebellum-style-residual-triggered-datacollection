"""为什么守门在等唤醒预算下**输给随机撒帧**？分解成两个独立效应。

`runs/gate_operating_curve.json` 的核心读数（pred=rotation，10 段中位数）：

    budget   gate capture   random capture
    0.5%     0.136          0.301
    2%       0.464          0.790
    5%       0.775          0.972

同样的**帧数**，随机撒点的覆盖反而更好。两种解释必须先分开：

* **A 不可解读**：残差高分帧在事件区间内的占比 ≈ 基率（信号对"事件何时发生"
  没有信息）。此时"覆盖"差异只能来自选择**结构**，不是来自信息。
* **B 结构惩罚**：闸门输出的是**聚簇**（连续帧成 burst），而随机撒点均匀铺开。
  在"某区间是否有任一帧被保留"这个口径下，铺开优于聚簇——同样 100 帧，
  分 100 处比挤在 3 处能覆盖更多区间。

A 用**帧级富集**测（top-k 帧落在事件内的比例 / 基率），B 用**聚簇度**测
（保留帧的连通段数与平均段长）。两者都能独立证伪，故必须都报。

Usage: PYTHONPATH=. .venv/bin/python scripts/diag_gate_vs_random.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gate_operating_curve import (KEEP_FRACS, event_windows, score_seq)

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
N_RAND = 100
SEED = 0


def runs_of(mask):
    """(n_runs, mean_run_len) of a boolean mask."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return 0, 0.0
    splits = np.where(np.diff(idx) > 1)[0] + 1
    lens = [len(r) for r in np.split(idx, splits)]
    return len(lens), float(np.mean(lens))


def main():
    rng = np.random.default_rng(SEED)
    seqs = sorted(glob.glob("data/aoe_event/val/seq_*"))
    rows, enrich, cluster = {}, {}, {}
    for seq in seqs:
        tag = os.path.basename(seq)
        score, fps, n = score_seq(seq, "rotation")
        meta = json.load(open(f"{seq}/events.json"))
        wins = event_windows(meta["events"], fps, n)
        warm = 30
        ev = np.zeros(n, dtype=bool)
        for i0, _i1, i1_true in wins:
            ev[i0:i1_true + 1] = True
        base = float(ev[warm:].mean())
        s = score[warm:]
        order = np.argsort(-s)                    # descending score
        rows[tag] = {"n": n, "n_events": len(wins), "base_rate": base}
        enrich[tag], cluster[tag] = {}, {}
        for f in BUDGETS:
            k = max(int(f * n), 1)
            top = np.zeros(n, dtype=bool)
            top[warm:][order[:k]] = True
            e = float(ev[top].mean() / max(base, 1e-9))
            nr, ml = runs_of(top)
            r_nr, r_ml = [], []
            for _ in range(N_RAND):
                rk = np.zeros(n, dtype=bool)
                rk[warm:][rng.choice(len(s), size=min(k, len(s)),
                                     replace=False)] = True
                a, b = runs_of(rk)
                r_nr.append(a)
                r_ml.append(b)
            enrich[tag][f] = e
            cluster[tag][f] = {"gate_runs": nr, "gate_mean_run": ml,
                               "rand_runs": float(np.mean(r_nr)),
                               "rand_mean_run": float(np.mean(r_ml))}
        print(f"  {tag}: base={base:.3f} n_ev={len(wins)} "
              f"enrich@2%={enrich[tag][0.02]:.2f} "
              f"runs@g2%={cluster[tag][0.02]['gate_runs']}"
              f"(rand {cluster[tag][0.02]['rand_runs']:.0f})", flush=True)

    print("\n=== A 帧级富集（top-k 帧落在事件内的比例 / 基率）===")
    print("   ≈1.0 = 无信息；>1 = 有信息；<1 = 反而避开事件")
    hdr = "  seg      " + "".join(f"{f * 100:>8.1f}%" for f in BUDGETS)
    print(hdr)
    for tag in enrich:
        print(f"  {tag:<9}" + "".join(
            f"{enrich[tag][f]:>9.3f}" for f in BUDGETS))
    med = {f: float(np.median([enrich[t][f] for t in enrich])) for f in BUDGETS}
    print(f"  {'MEDIAN':<9}" + "".join(f"{med[f]:>9.3f}" for f in BUDGETS))

    print("\n=== B 聚簇度（保留帧的连通段数；随机撒点对照）===")
    print(f"  {'seg':<9}" + "".join(f"{f * 100:>9.1f}%" for f in BUDGETS))
    for tag in cluster:
        print(f"  {tag:<9}" + "".join(
            f"{cluster[tag][f]['gate_runs']:>9d}" for f in BUDGETS))
    print("  随机撒点段数（同预算）:")
    for tag in cluster:
        print(f"  {tag:<9}" + "".join(
            f"{cluster[tag][f]['rand_runs']:>9.0f}" for f in BUDGETS))
    medg = {f: int(np.median([cluster[t][f]["gate_runs"] for t in cluster]))
            for f in BUDGETS}
    medr = {f: float(np.median([cluster[t][f]["rand_runs"] for t in cluster]))
            for f in BUDGETS}
    print(f"  {'MEDIAN':<9}" + "".join(f"{medg[f]:>9d}" for f in BUDGETS))
    print(f"  {'rand md':<9}" + "".join(f"{medr[f]:>9.0f}" for f in BUDGETS))
    print("\n  平均段长（帧）:")
    print(f"  {'seg':<9}" + "".join(f"{f * 100:>9.1f}%" for f in BUDGETS))
    for tag in cluster:
        print(f"  {tag:<9}" + "".join(
            f"{cluster[tag][f]['gate_mean_run']:>9.1f}" for f in BUDGETS))

    out = {"budgets": list(BUDGETS), "n_rand": N_RAND,
           "per_seq": {t: {"n": rows[t]["n"], "n_events": rows[t]["n_events"],
                           "base_rate": rows[t]["base_rate"],
                           "enrichment": {str(f): enrich[t][f]
                                          for f in BUDGETS},
                           "cluster": {str(f): cluster[t][f]
                                       for f in BUDGETS}}
                       for t in rows},
           "enrichment_median": {str(f): med[f] for f in BUDGETS},
           "gate_runs_median": {str(f): medg[f] for f in BUDGETS},
           "rand_runs_median": {str(f): medr[f] for f in BUDGETS}}
    with open("runs/diag_gate_vs_random.json", "w") as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    print("\n-> runs/diag_gate_vs_random.json")


if __name__ == "__main__":
    main()

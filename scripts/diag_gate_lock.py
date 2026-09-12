"""诊断守门锁死：AdaptiveTrigger 自己的 mu/var 归一化是否冻结。

## 假设

§2.3.3 发现的"门控统计冻结"在 `FeatureMapCancel` 层面已用 robust 修好；
但 `AdaptiveTrigger` **自己**还有一层 mu/var 归一化，那层没有 robust 化：

    z = (score - mu) / sd
    只在 `not on and z < threshold_k` 时更新 mu/var

如果 score 是重尾的（实测 max/median 可达数百倍），那么过门失败会让 mu 与 var
同时失真，归一化尺度错位后 z 长期 > 6 → 一旦触发就永不释放 → 闸门锁死开。

`runs/downstream_event_clips.json` 里 seq_000 存 99.2%、seq_002 存 95.8%，
正是这个形态。

## 判据

逐段打印 z 轨迹的分布、`on` 的占空比、被挡更新的帧比例、sd/mu 尺度比。
若锁定段的"被挡帧比例"接近 1 且 sd 尺度不匹配，则假设成立。

Usage: PYTHONPATH=. .venv/bin/python scripts/diag_gate_lock.py data/aoe_event
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cerebellum_cancel.core import AdaptiveTrigger, TriggerConfig


def trace(score, fps, n, onset_k=6.0, off_k=2.5):
    """Drive the real AdaptiveTrigger, recording each step's z / mu / sd.

    The z that gates the stat update is computed *inside* ``update()`` from
    the pre-update ``mu``/``var``, so we mirror it from those same values
    before calling; nothing is re-implemented, only observed.
    """
    trig = AdaptiveTrigger(TriggerConfig(threshold_k=onset_k, off_k=off_k))
    zs = np.full(n, np.nan)
    mus = np.full(n, np.nan)
    sds = np.full(n, np.nan)
    ons = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    for t in range(1, n):
        s = float(score[t])
        # mirror update()'s pre-state readout (n==0 seeds mu <- score, z = 0)
        if trig.n == 0:
            mu, sd = s, 1.0
        else:
            mu = trig.mu
            sd = float(np.sqrt(trig.var)) + 1e-3 * (abs(mu) + 1.0)
        z = (s - mu) / sd
        in_warmup = (trig.n + 1) <= trig.warmup_frames
        on = trig.update(t / fps, s, fps)
        zs[t], mus[t], sds[t], ons[t] = z, mu, sd, on
        # the stat update is skipped exactly when not warmup and (on or z>=k)
        blocked[t] = (not in_warmup) and (on or z >= onset_k)
    return {"z": zs, "mu": mus, "sd": sds, "on": ons, "blocked": blocked,
            "clips": trig.finalize(n / fps), "final_mu": trig.mu,
            "final_var": trig.var}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe_event")
    ap.add_argument("--warmup-skip", type=int, default=60)
    args = ap.parse_args()

    from train_downstream_aoe import score_seq
    seqs = sorted(glob.glob(f"{args.dataset}/val/seq_*"))
    out = {}
    print(f"{'segment':<10}{'n':>6}{'stored':>8}{'blocked%':>9}"
          f"{'z med':>7}{'z p95':>8}{'z max':>9}{'sd/mu%':>8}{'score med':>10}"
          f"{'max/med':>8}")
    for seq in seqs:
        tag = os.path.basename(seq)
        score, fps, n = score_seq(seq)
        r = trace(score, fps, n)
        k = args.warmup_skip
        on = r["on"][k:]
        z = r["z"][k:]
        sc = score[k:]
        stored = float(on.mean())
        blk = float(np.mean(r["blocked"][k:]))
        sd_med = float(np.nanmedian(r["sd"][k:]))
        mu_med = float(np.nanmedian(r["mu"][k:]))
        out[tag] = {
            "n": int(n), "stored": stored, "blocked_frac": blk,
            "z_median": float(np.nanmedian(z)),
            "z_p95": float(np.nanpercentile(z, 95)),
            "z_max": float(np.nanmax(z)),
            "sd_median": sd_med, "mu_median": mu_med,
            "sd_over_mu": sd_med / max(abs(mu_med), 1e-9),
            "score_median": float(np.median(sc)),
            "score_p99": float(np.percentile(sc, 99)),
            "score_max_over_median": float(sc.max()
                                           / max(np.median(sc), 1e-9)),
            "n_clips": len(r["clips"]),
        }
        o = out[tag]
        print(f"{tag:<10}{n:>6}{stored:>8.3f}{blk:>9.2f}"
              f"{o['z_median']:>7.2f}{o['z_p95']:>8.2f}{o['z_max']:>9.1f}"
              f"{o['sd_over_mu'] * 100:>8.2f}{o['score_median']:>10.1f}"
              f"{o['score_max_over_median']:>8.1f}")

    locked = [k for k, v in out.items() if v["stored"] > 0.5]
    dead = [k for k, v in out.items() if v["stored"] < 0.005]
    good = [k for k, v in out.items() if 0.005 <= v["stored"] <= 0.5]
    print(f"\n锁死开 ({len(locked)}/10): {locked}")
    print(f"空转   ({len(dead)}/10): {dead}")
    print(f"正常   ({len(good)}/10): {good}")
    if locked:
        print("\n锁定段对照（假设：blocked 近 1、sd 相对尺度失配）")
        for k in locked:
            v = out[k]
            print(f"  {k}: blocked={v['blocked_frac']:.2f} "
                  f"sd={v['sd_median']:.1f} mu={v['mu_median']:.1f} "
                  f"z_med={v['z_median']:.2f} "
                  f"max/med={v['score_max_over_median']:.0f}")

    with open("runs/diag_gate_lock.json", "w") as fp:
        json.dump({"per_seq": out, "locked": locked, "dead": dead,
                   "normal": good}, fp, indent=1, ensure_ascii=False)
    print("\n-> runs/diag_gate_lock.json")


if __name__ == "__main__":
    main()

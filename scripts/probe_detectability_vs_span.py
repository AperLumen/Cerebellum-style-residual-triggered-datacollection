"""检出能力 vs 故障覆盖范围——"能找到多小的一片金子"。

## 为什么必须测这个

前面两轮留下一个未解矛盾，而且它正好卡在用户问题的核心上：

    probe_integrity_realimu.py  整段 IMU 归零   -> 1.04x   检不到
    diag_two_bounds.py          局部 60 帧归零   -> 1.49x   能检到

同样的数据、同样的故障、同样的统计量，只差**故障覆盖了多少帧**。
"在垃圾山里找 1% 的金子"这个说法本身就意味着**局部**异常，所以这条
区间关系就是可操作性的核心：**多小的异常还能被找到？**

## 假设与判读

如果差异来自"有没有健康参照"，那么检出能力应当随故障覆盖比例上升而**下降**，
并在故障覆盖整段（没有"之外"可比）时**归零**——因为在线/冻结统计都失去基准。

本脚本扫一条**检出率曲线**：

* 故障覆盖比例 f ∈ {1.2%, 2.5%, 5%, 10%, 20%, 40%, 80%, 100%}
* 每个 f 随机位置重复若干次
* 判据（每段自适应的、真实可用的判据）：
  用**故障段之外**的 surprise 分布定阈值（P99），看故障段内的最大值是否越界。
  这正是实际巡检的做法：一段录制内部自己定基线。

同时报"冻结基准 z 分数"版本，以对照 §自适应上界。

输出：runs/detectability_vs_span.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

FRACTIONS = (0.012, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.00)


def series(F, gy: np.ndarray, n: int, fps: float, f_map: float, seq: str,
           freeze_at: int) -> dict:
    """在线 surprise + 冻结基准 z 分数（同 diag_two_bounds 的算法）。"""
    from cerebellum_cancel.featmap_cancel import (CancelConfig,
                                                  FeatureMapCancel)
    canc = FeatureMapCancel(CancelConfig(fps=fps, f_map=f_map, robust=True,
                                        pred="rotation", source=seq))
    sur = np.zeros(n, dtype=np.float64)
    first = np.asarray(F[0], dtype=np.float32)
    errs = np.zeros((n,) + first.shape, dtype=np.float32)
    for t in range(n):
        Fp = np.asarray(F[t], dtype=np.float32)
        out = canc.step(Fp, gy[t])
        sd = np.sqrt(np.asarray(canc.var)) + 1e-3
        sur[t] = out["surprise"]
        errs[t] = np.asarray(out["z"], dtype=np.float32) * sd

    # 冻结基准：前 freeze_at 帧的逐格 mu/sd
    ref = errs[:freeze_at]
    mu_f = ref.mean(axis=0)
    sd_f = ref.std(axis=0) + 1e-3
    bnd = canc.cfg.border
    fz = np.zeros(n, dtype=np.float64)
    for t in range(n):
        z = (errs[t] - mu_f) / sd_f
        fz[t] = float(np.abs(z[:, bnd:-bnd, bnd:-bnd]).max())
    return {"surprise": sur, "frozen_z": fz}


def detect(sig: np.ndarray, span: slice, lo: int) -> dict:
    """段内自适应判据：故障段之外定 P99 阈值，看段内最大值是否越界。"""
    idx_out = np.r_[np.arange(lo, span.start), np.arange(span.stop, len(sig))]
    if len(idx_out) < 30:
        # 故障覆盖几乎整段 -> 没有健康参照
        return {"detected": False, "no_reference": True, "peak": float("nan"),
                "thr": float("nan"), "margin": float("nan")}
    thr = float(np.percentile(sig[idx_out], 99))
    peak = float(np.max(sig[span]))
    return {"detected": bool(peak > thr), "no_reference": False,
            "peak": peak, "thr": thr, "margin": float(peak / max(thr, 1e-9))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="data/real_imu/val/seq_000,"
                                      "data/aoe_event/val/seq_000")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--freeze-at", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/detectability_vs_span.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out: dict = {"fractions": list(FRACTIONS), "datasets": {}}

    for seqpath in args.seqs.split(","):
        seqpath = seqpath.strip()
        if not os.path.exists(seqpath):
            print(f"  [跳过] {seqpath}")
            continue
        if "real_imu" in seqpath:
            from probe_integrity_realimu import load_real
            F, gy, meta, n = load_real(seqpath, args.n)
        else:
            from cerebellum_cancel.io import load_imu_csv
            Fm = np.load(f"{seqpath}/featmaps.npy", mmap_mode="r")
            gy = np.array([g.gyro for g in load_imu_csv(f"{seqpath}/imu.csv")],
                          dtype=np.float32)
            meta = json.load(open(f"{seqpath}/events.json"))
            n = min(len(Fm), len(gy), args.n)
            F = np.asarray(Fm[:n], dtype=np.float32)
        fps = float(meta.get("fps", 30.0))
        f_map = float(meta["f_map"])
        lo = min(args.freeze_at, n // 4)

        base = series(F, gy, n, fps, f_map, seqpath, args.freeze_at)
        clean_out_p99 = float(np.percentile(base["surprise"][lo:], 99))
        tag = seqpath.split("/")[1]
        print(f"\n=== {tag}  n={n}  （干净 surprise 的 P99 = {clean_out_p99:.2f}）===")
        print(f"{'覆盖比例':>8}{'帧数':>7}{'在线检出率':>12}{'冻结z检出率':>13}"
              f"{'在线边界中位':>14}{'冻结z边界中位':>15}")

        rows = []
        for f in FRACTIONS:
            span_len = max(int(round(f * n)), 3)
            det_on, det_fz, mg_on, mg_fz = [], [], [], []
            for _ in range(args.reps):
                if span_len >= n:
                    a, b = lo, n
                else:
                    a = int(rng.integers(lo, max(n - span_len, lo + 1)))
                    b = min(a + span_len, n)
                g = gy.copy()
                g[a:b] = 0.0                     # 规范完整性故障：IMU 归零
                s = series(F, g, n, fps, f_map, seqpath, args.freeze_at)
                d_on = detect(s["surprise"], slice(a, b), lo)
                d_fz = detect(s["frozen_z"], slice(a, b), lo)
                det_on.append(d_on["detected"])
                det_fz.append(d_fz["detected"])
                if not d_on["no_reference"]:
                    mg_on.append(d_on["margin"])
                    mg_fz.append(d_fz["margin"])
            r = {
                "fraction": f, "span_len": int(span_len),
                "detect_rate_online": float(np.mean(det_on)),
                "detect_rate_frozen": float(np.mean(det_fz)),
                "margin_online_median": float(np.median(mg_on)) if mg_on else None,
                "margin_frozen_median": float(np.median(mg_fz)) if mg_fz else None,
                "had_reference": bool(mg_on),
            }
            rows.append(r)
            mo = f"{r['margin_online_median']:.2f}" if r["margin_online_median"] else "—"
            mf = f"{r['margin_frozen_median']:.2f}" if r["margin_frozen_median"] else "—"
            print(f"{f * 100:>7.1f}%{span_len:>7}{r['detect_rate_online']:>12.2f}"
                  f"{r['detect_rate_frozen']:>13.2f}{mo:>14}{mf:>15}")

        out["datasets"][tag] = {
            "seq": seqpath, "n": n, "fps": fps, "f_map": f_map,
            "clean_surprise_p99": clean_out_p99, "rows": rows}

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    print("\n判读：")
    print("  检出率随覆盖比例上升而下降、并在覆盖整段时归零 -> 门槛是'有没有")
    print("  健康参照'。这意味着可检出的最小异常由**录制长度 / 异常占比**决定，")
    print("  而不是由算法灵敏度决定——这直接回答了'能找到多小的一片金子'。")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

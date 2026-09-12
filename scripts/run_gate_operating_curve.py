"""守门工作特性曲线——CMCI 当常开门（唤醒重型模型）时，代价与捕获怎么换。

## 为什么测这个

前几轮的判据是"下游训练价值"（§2.3.6/2.3.7），结论是负的：残差选出的帧
训练不出更好的分类器。但那不是守门（gating）的判据。守门的判据是：

    **它以多低的唤醒率，换来多高的事件捕获率？事件到来后多快唤醒？**

两者可以解耦：一个信号可以"训练价值为零"（不含类别判别信息）却"守门价值
很高"（以极低代价圈出事件发生的时刻）。此前从没测过这个口径。

## 两种部署语义必须分开测（关键）

* **唤醒语义**（用户的用法：重型模型只在 triggered 帧上跑）——代价 = 唤醒的
  帧数，帧可以**离散**。此时正确的零假设是**随机撒帧**：撒 5% 的帧进 1 s
  区间，期望覆盖率 1-(1-0.05)^30 ≈ 79%。**不做这个对照就会把"撒点撒得够多"
  误读成"守门有效"**——本轮实测正是如此。
* **存储语义**（本项目原有用法）——存的是**片段**（pre/post roll + 合并 +
  最短长度）。同帧数下连续片段只能覆盖少数位置，离散帧能撒满更多区间，
  故两者不可直接比较；用同语义的**随机突发**做零假设。

## 四个读数

1. **在线自适应触发器**（可部署，无需标定）；
2. **阈值扫描**（标定 oracle）——曲线形状 = 信号的守门信息量；
3. **随机零假设**（随机撒帧 / 随机突发，各 n_rand 次）——含 p95；
4. **事件内 oracle** 与 **均匀突发**——同预算参照。

## 唤醒延迟

抓取失败是一个**瞬间**。守门即使圈中事件，若唤醒发生在事件**结束之后**，
对下游毫无价值。故报 ``capture_raw``（事件真实进行中有帧被保留，严格口径；
宽松口径含 0.5 s 尾垫）、``onset_capture``（开始后 0.5 s 内唤醒）与 lat。

## 报告口径修正

此前报 10 段**均值** stored = 0.233。逐段查下来它是双峰分布的产物：
seq_000 存 99.2%、seq_002 存 95.8%（闸门锁死开），seq_006 存 0.0%（全漏），
其余 8 段 1.3%–10.7%，中位数仅 0.053。**均值在此误导**，故逐段报 + 中位数。

Usage:
  PYTHONPATH=. .venv/bin/python scripts/run_gate_operating_curve.py data/aoe_event
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.core import AdaptiveTrigger, TriggerConfig
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                              maxcell_score, rollout)
from cerebellum_cancel.io import load_imu_csv

KEEP_FRACS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
TAIL_PAD = 0.5          # matches cerebellum_cancel.eval.evaluate
ONSET_WIN = 0.5         # seconds; "woken at the failure moment"
CLIP_LEN = 1.0          # seconds per burst (matches uniform_baseline)
PRE_ROLL = 0.3          # matches TriggerConfig
POST_ROLL = 0.8         # matches TriggerConfig
MIN_CLIP = 0.4          # matches TriggerConfig
MERGE_GAP = 0.4         # matches TriggerConfig.min_clip_len


def seq_dirs(dataset):
    return sorted(glob.glob(f"{dataset}/val/seq_*"))


def score_seq(seq, pred):
    """Roll the cancel layer over a prepared segment and return the trace.

    ``pred`` selects the warp model: "shift" is the released default
    (§2.3.7 used it), "rotation" is the measured-better variant
    (runs/rotwarp_aoe.json). Both are reported because the released
    negatives used "shift".
    """
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
    return maxcell_score(z), fps, n


def event_windows(events, fps, n, tail=TAIL_PAD):
    """Frame spans per event as (i_start, i_padded_end, i_true_end)."""
    out = []
    for a, b in events:
        i0 = max(int(a * fps), 0)
        i1_true = min(int(b * fps), n - 1)
        i1 = min(i1_true + int(tail * fps), n - 1)
        if i1 >= i0:
            out.append((i0, i1, max(i1_true, i0)))
    return out


# ---------------------------------------------------------------- keep masks

def clips_from_mask(mask, fps, pre=PRE_ROLL, post=POST_ROLL,
                    min_len=MIN_CLIP, gap=MERGE_GAP):
    """Per-frame mask -> stored clips (deployed storage semantics):
    merge close runs, drop runs shorter than ``min_len``, add pre/post roll."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = [(int(r[0]), int(r[-1])) for r in np.split(idx, splits)]
    merge_fr = max(int(gap * fps), 1)
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= merge_fr:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    pre_fr, post_fr = int(pre * fps), int(post * fps)
    min_fr = max(int(min_len * fps), 1)
    out = []
    for a, b in merged:
        if b - a + 1 < min_fr:
            continue
        out.append((max(a - pre_fr, 0), b + post_fr))
    return out


def mask_from_clips(clips, n):
    keep = np.zeros(n, dtype=bool)
    for a, b in clips:
        keep[max(a, 0):min(b + 1, n)] = True
    return keep


def random_scatter_mask(n, budget, rng):
    """Null hypothesis for WAKE semantics: the same number of frames, at
    random positions. Without this control, "5% of frames lands inside 83% of
    1 s intervals" is the birthday-problem expectation, not evidence the gate
    finds anything."""
    k = int(budget * n)
    keep = np.zeros(n, dtype=bool)
    if k <= 0:
        return keep
    keep[rng.choice(n, size=min(k, n), replace=False)] = True
    return keep


def random_burst_clips(fps, n, budget, rng):
    """Null hypothesis for STORAGE semantics: same number of 1 s bursts,
    random positions."""
    clip_fr = max(int(CLIP_LEN * fps), 1)
    n_clips = int(budget * n) // clip_fr
    if n_clips < 1:
        return []
    starts = rng.integers(0, max(n - clip_fr, 1), size=n_clips)
    return [(int(s), int(s) + clip_fr) for s in starts]


def uniform_burst_clips(fps, n, budget):
    """Evenly spread 1 s bursts (the project's uniform_baseline convention)."""
    clip_fr = max(int(CLIP_LEN * fps), 1)
    n_clips = int(budget * n) // clip_fr
    if n_clips < 1:
        return []
    spacing = n / n_clips
    return [(int(j * spacing), int(j * spacing) + clip_fr)
            for j in range(n_clips)]


def oracle_burst_clips(events, fps, n, budget, tail=TAIL_PAD):
    """Same 1 s bursts, placed inside annotated windows — the "knows where
    the events are" upper bound at equal cost."""
    wins = event_windows(events, fps, n, tail)
    clip_fr = max(int(CLIP_LEN * fps), 1)
    n_clips = int(budget * n) // clip_fr
    if not wins or n_clips < 1:
        return []
    per = np.full(len(wins), n_clips // len(wins), dtype=int)
    per[:n_clips % len(wins)] += 1
    out = []
    for (i0, i1, _), c in zip(wins, per):
        span = i1 - i0 + 1
        if c <= 0 or span <= 0:
            continue
        for j in range(c):
            s = i0 + int(j * span / c)
            out.append((s, min(s + clip_fr, n)))
    return out


def readout(keep, wins, fps, onset_win=ONSET_WIN):
    """(capture, capture_raw, onset_capture, median latency) over windows."""
    cap, cap_raw, oncap, lat = [], [], [], []
    ow = max(int(onset_win * fps), 1)
    for i0, i1, i1_true in wins:
        hit = np.flatnonzero(keep[i0:i1 + 1])
        raw_hit = np.flatnonzero(keep[i0:i1_true + 1])
        cap_raw.append(1.0 if len(raw_hit) else 0.0)
        if len(hit) == 0:
            cap.append(0.0)
            oncap.append(0.0)
            continue
        cap.append(1.0)
        lat.append(float(hit[0]) / fps)
        oncap.append(1.0 if hit[0] < ow else 0.0)
    return (float(np.mean(cap)) if cap else float("nan"),
            float(np.mean(cap_raw)) if cap_raw else float("nan"),
            float(np.mean(oncap)) if oncap else float("nan"),
            float(np.median(lat)) if lat else float("nan"))


def score_auc(score, wins, warm):
    """Frame-level AUC of the score against the event windows — sanity check
    against §2.3.3's 0.47 (human action labels); this uses event intervals."""
    n = len(score)
    y = np.zeros(n, dtype=bool)
    for i0, _i1, i1_true in wins:
        y[i0:i1_true + 1] = True
    m = np.zeros(n, dtype=bool)
    m[warm:] = True
    s, yy = score[m], y[m]
    if yy.all() or not yy.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = int(yy.sum()), int((~yy).sum())
    return float((ranks[yy].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def sweep_thresholds(base, n_thr=80):
    """Quantiles of the post-warmup score distribution (scale-free); with
    +-inf endpoints so the sweep spans the full stored range."""
    qs = np.quantile(base, np.linspace(0.0, 0.9995, n_thr))
    lo, hi = float(base.min()), float(base.max())
    eps = (hi - lo) * 1e-9 + 1e-12
    return np.unique(np.r_[lo - eps, qs, hi + eps])


def sweep_curve(base, thr, fps, n, warm, wins, mode, n_rand, rng):
    """Gate curve at one threshold under WAKE or CLIP semantics, plus the
    matched random null at the same stored fraction."""
    raw = np.zeros(n, dtype=bool)
    raw[warm:] = base > thr
    if mode == "wake":
        keep = raw
    else:
        keep = mask_from_clips(clips_from_mask(raw, fps), n)
    stored = float(keep.mean())
    cap, cap_raw, oncap, lat = readout(keep, wins, fps)
    r_cap, r_raw, r_on = [], [], []
    for _ in range(n_rand):
        rk = (random_scatter_mask(n, stored, rng) if mode == "wake"
              else mask_from_clips(random_burst_clips(fps, n, stored, rng), n))
        a, b, c, _d = readout(rk, wins, fps)
        r_cap.append(a)
        r_raw.append(b)
        r_on.append(c)
    return {
        "thr": float(thr), "stored": stored,
        "n_clips": len(clips_from_mask(raw, fps)),
        "capture": cap, "capture_raw": cap_raw, "onset_capture": oncap,
        "latency_med": lat,
        "rand_capture_mean": float(np.nanmean(r_cap)),
        "rand_capture_p95": float(np.nanpercentile(r_cap, 95)),
        "rand_capture_raw_mean": float(np.nanmean(r_raw)),
        "rand_onset_mean": float(np.nanmean(r_on)),
    }


def interp_at(curve, field, target):
    """Linear interpolation of ``field`` at a target stored fraction."""
    xs = np.array([r["stored"] for r in curve], dtype=float)
    ys = np.array([r[field] for r in curve], dtype=float)
    ok = ~np.isnan(ys)
    if ok.sum() < 2:
        return float("nan")
    xs, ys = xs[ok], ys[ok]
    if target <= xs[0]:
        return float(ys[0])
    if target >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(target, xs, ys))


def curve_for_pred(dataset, pred, n_rand=200, seed=0):
    seqs = seq_dirs(dataset)
    per_seq = {}
    for seq in seqs:
        tag = os.path.basename(seq)
        rng = np.random.default_rng(seed)
        score, fps, n = score_seq(seq, pred)
        meta = json.load(open(f"{seq}/events.json"))
        wins = event_windows(meta["events"], fps, n)
        warm = 30                                    # match AdaptiveTrigger
        base = score[warm:]

        # 1. online adaptive triggers: released stats vs winsorized fix
        trigs = {}
        for label, rob in (("online_frozen_stats", False),
                           ("online_robust_stats", True)):
            trig = AdaptiveTrigger(TriggerConfig(robust=rob))
            for t in range(1, n):
                trig.update(t / fps, float(score[t]), fps)
            clips = trig.finalize(n / fps)
            keep = mask_from_clips(
                [(int(a * fps), min(int(b * fps), n - 1))
                 for a, b in clips], n)
            cap, cap_raw, oncap, lat = readout(keep, wins, fps)
            trigs[label] = {"stored": float(keep.mean()),
                            "n_clips": len(clips), "capture": cap,
                            "capture_raw": cap_raw, "onset_capture": oncap,
                            "latency_med": lat,
                            "clips": [[float(a), float(b)] for a, b in clips]}

        # 2. threshold sweeps under both semantics. `thr_all` ascends, so the
        #    resulting stored fractions DESCEND; sort ascending afterwards
        #    because interp_at requires a monotonically increasing x-axis.
        thr_all = sweep_thresholds(base)
        sweep = {"wake": [], "clip": []}
        for t in thr_all:
            for mode in ("wake", "clip"):
                row = sweep_curve(base, t, fps, n, warm, wins, mode,
                                  n_rand, rng)
                if sweep[mode] and round(row["stored"], 8) == round(
                        sweep[mode][-1]["stored"], 8):
                    continue
                sweep[mode].append(row)
        for mode in sweep:
            sweep[mode].sort(key=lambda r: r["stored"])

        # 3. matched targets
        targets = []
        for f in KEEP_FRACS:
            g_wake = {k: interp_at(sweep["wake"], k, f)
                      for k in ("capture", "capture_raw", "onset_capture",
                                "latency_med", "rand_capture_mean",
                                "rand_capture_p95", "rand_capture_raw_mean")}
            g_clip = {k: interp_at(sweep["clip"], k, f)
                      for k in ("capture", "capture_raw", "onset_capture",
                                "latency_med", "rand_capture_mean",
                                "rand_capture_p95", "rand_capture_raw_mean")}
            uk = mask_from_clips(uniform_burst_clips(fps, n, f), n)
            u_cap, u_raw, u_on, u_lat = readout(uk, wins, fps)
            ok = mask_from_clips(oracle_burst_clips(meta["events"], fps, n, f),
                                 n)
            o_cap, o_raw, o_on, _o_lat = readout(ok, wins, fps)
            targets.append({
                "budget": f,
                "wake_gate_capture": g_wake["capture"],
                "wake_gate_capture_raw": g_wake["capture_raw"],
                "wake_gate_onset_capture": g_wake["onset_capture"],
                "wake_gate_latency_med": g_wake["latency_med"],
                "wake_rand_capture_mean": g_wake["rand_capture_mean"],
                "wake_rand_capture_p95": g_wake["rand_capture_p95"],
                "wake_rand_capture_raw_mean": g_wake["rand_capture_raw_mean"],
                "clip_gate_capture": g_clip["capture"],
                "clip_gate_capture_raw": g_clip["capture_raw"],
                "clip_gate_onset_capture": g_clip["onset_capture"],
                "clip_gate_latency_med": g_clip["latency_med"],
                "clip_rand_capture_mean": g_clip["rand_capture_mean"],
                "clip_rand_capture_p95": g_clip["rand_capture_p95"],
                "uniform_capture": u_cap, "uniform_capture_raw": u_raw,
                "uniform_onset_capture": u_on, "uniform_latency_med": u_lat,
                "oracle_capture": o_cap, "oracle_capture_raw": o_raw,
                "oracle_onset_capture": o_on,
                "oracle_stored": float(ok.mean()),
            })

        per_seq[tag] = {
            "fps": fps, "n": n, "n_events": len(wins),
            "event_frame_frac": float(np.mean([(i1_true - i0 + 1) / n
                                               for i0, _i1, i1_true in wins])),
            "score_auc": score_auc(score, wins, warm),
            "triggers": trigs, "sweep": sweep, "targets": targets,
        }
        fz, rb = trigs["online_frozen_stats"], trigs["online_robust_stats"]
        print(f"  {tag}: n={n} ev={len(wins)} "
              f"auc={per_seq[tag]['score_auc']:.3f} | "
              f"frozen {fz['stored']:.3f}c{fz['capture_raw']:.2f} | "
              f"robust {rb['stored']:.3f}c{rb['capture_raw']:.2f}", flush=True)
    return per_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="data/aoe_event")
    ap.add_argument("--preds", default="shift,rotation")
    ap.add_argument("--out", default="runs/gate_operating_curve.json")
    ap.add_argument("--n-rand", type=int, default=200)
    args = ap.parse_args()
    preds = [p.strip() for p in args.preds.split(",") if p.strip()]

    out = {"dataset": args.dataset, "keep_fracs": list(KEEP_FRACS),
           "tail_pad": TAIL_PAD, "onset_win": ONSET_WIN,
           "clip_len": CLIP_LEN, "pre_roll": PRE_ROLL,
           "post_roll": POST_ROLL, "n_rand": args.n_rand, "preds": {}}

    for pred in preds:
        print(f"\n=== pred={pred} ===")
        per_seq = curve_for_pred(args.dataset, pred, n_rand=args.n_rand)
        V = list(per_seq.values())

        st = np.array([v["triggers"]["online_frozen_stats"]["stored"]
                       for v in V])
        sr = np.array([v["triggers"]["online_robust_stats"]["stored"]
                       for v in V])
        auc = np.array([v["score_auc"] for v in V], dtype=float)
        frz = [v["triggers"]["online_frozen_stats"] for v in V]
        rob = [v["triggers"]["online_robust_stats"] for v in V]
        locked, dead = int((st > 0.5).sum()), int((st < 0.005).sum())

        print(f"\n  帧级 AUC(score vs 事件区间): median={np.nanmedian(auc):.3f}")
        print(f"  在线触发器 stored: frozen median={np.median(st):.3f} "
              f"mean={st.mean():.3f} | robust median={np.median(sr):.3f} "
              f"mean={sr.mean():.3f}")
        print(f"    锁死开(>0.5)={locked}/10  空转(<0.005)={dead}/10")
        print(f"    frozen per-seg: {np.round(st, 3).tolist()}")
        print(f"    robust per-seg: {np.round(sr, 3).tolist()}")
        for nm, rr in (("frozen", frz), ("robust", rob)):
            print(f"    {nm} 捕获中位 宽松/严格/onset/延迟s: "
                  f"{np.nanmedian([r['capture'] for r in rr]):.3f}/"
                  f"{np.nanmedian([r['capture_raw'] for r in rr]):.3f}/"
                  f"{np.nanmedian([r['onset_capture'] for r in rr]):.3f}/"
                  f"{np.nanmedian([r['latency_med'] for r in rr]):.2f}")

        header = ("budget", "wakeCap", "wakeRaw", "wakeRand", "rP95",
                  "clipCap", "clipRaw", "clipRand", "uCap", "oCap")
        print("\n  " + "".join(f"{h:>9}" for h in header))
        table = []
        for i, f in enumerate(KEEP_FRACS):
            def md(name):
                return float(np.nanmedian([v["targets"][i][name]
                                           for v in per_seq.values()]))
            row = {
                "budget": f,
                "wake_gate_capture": md("wake_gate_capture"),
                "wake_gate_capture_raw": md("wake_gate_capture_raw"),
                "wake_gate_onset_capture": md("wake_gate_onset_capture"),
                "wake_gate_latency_med": md("wake_gate_latency_med"),
                "wake_rand_capture_mean": md("wake_rand_capture_mean"),
                "wake_rand_capture_p95": md("wake_rand_capture_p95"),
                "clip_gate_capture": md("clip_gate_capture"),
                "clip_gate_capture_raw": md("clip_gate_capture_raw"),
                "clip_gate_onset_capture": md("clip_gate_onset_capture"),
                "clip_rand_capture_mean": md("clip_rand_capture_mean"),
                "clip_rand_capture_p95": md("clip_rand_capture_p95"),
                "uniform_capture": md("uniform_capture"),
                "uniform_capture_raw": md("uniform_capture_raw"),
                "oracle_capture": md("oracle_capture"),
                "oracle_onset_capture": md("oracle_onset_capture"),
            }
            table.append(row)
            print(f"  {f:>7.3f} {row['wake_gate_capture']:>9.3f} "
                  f"{row['wake_gate_capture_raw']:>9.3f} "
                  f"{row['wake_rand_capture_mean']:>9.3f} "
                  f"{row['wake_rand_capture_p95']:>9.3f} "
                  f"{row['clip_gate_capture']:>9.3f} "
                  f"{row['clip_gate_capture_raw']:>9.3f} "
                  f"{row['clip_rand_capture_mean']:>9.3f} "
                  f"{row['uniform_capture']:>9.3f} "
                  f"{row['oracle_capture']:>9.3f}")

        out["preds"][pred] = {
            "per_seq": per_seq, "table": table,
            "score_auc_median": float(np.nanmedian(auc)),
            "frozen_stored_median": float(np.median(st)),
            "frozen_stored_mean": float(st.mean()),
            "robust_stored_median": float(np.median(sr)),
            "robust_stored_mean": float(sr.mean()),
            "frozen_stored_per_seq": st.tolist(),
            "robust_stored_per_seq": sr.tolist(),
            "frozen_capture_median": float(np.nanmedian(
                [r["capture"] for r in frz])),
            "frozen_capture_raw_median": float(np.nanmedian(
                [r["capture_raw"] for r in frz])),
            "robust_capture_median": float(np.nanmedian(
                [r["capture"] for r in rob])),
            "robust_capture_raw_median": float(np.nanmedian(
                [r["capture_raw"] for r in rob])),
            "n_locked_open": locked, "n_dead": dead,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

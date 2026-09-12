"""层数扫描：多加一层残差预测器，值不值？

问题（"层数"线）。Petrini et al.（Nature 657, 377, 2026）在 CIFAR-10 上观察到：
光层是线性的，但**增加层数仍提升性能**。翻译到本项目：现有预测器是一层线性
warp（gyro shift）。在它的残差上再叠预测层，残差能否继续降？代价多少？净收益
正负？

两种"层"的实现（``CascadeConfig.layer_kind``）：

- ``temporal``：第 2 层起用**慢速时序水平**预测该格下一帧的残余（帧差消不掉的
  漂移/呼吸，对应论文的 granule-cell temporal basis / 负像）；
- ``spatial``：第 2 层起做**中心-周边去相关**（减掉局部空间均值），对应论文的
  ON/OFF 对抗通道与侧抑制——事件是局部尖峰，会被凸显出来。

**方法学要点**（两条都是实测踩出来的，写在这里防空转）：

1. 度量用**原始残差**，不用 z 分数。z 是归一化的：成功的预测器把残差和尺度
   一起压小，z 保持 O(1)，用它衡量层数会得出相反结论。
2. 聚合时必须**只在所有层都产出有限值的序列上比较**，且事件/平静划分用真值
   事件区间（与层无关）。否则每层各用一套样本、再加上丢弃 NaN，会把某层均值
   系统性美化——初版就出现过这个假阳性（L=4 显示 0.832，实为剔除了最差段）。

判读标准：**平静残差降 + 事件对比度升** = 真预测掉了可预测成分；
平静残差降但对比度持平 = 只是缩小了增益（无信息增益）。

用法::

  PYTHONPATH=. .venv/bin/python scripts/run_cascade_scan.py --n 1500 --seqs 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.cascade import CascadeCancel, CascadeConfig
from cerebellum_cancel.core import AdaptiveTrigger
from cerebellum_cancel.featmap_cancel import CancelConfig, maxcell_score
from cerebellum_cancel.io import load_imu_csv

LAYERS = (1, 2, 3, 4)
WARMUP_FRAMES = 60


def load_seq(seq: str, limit: int | None):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    if limit:
        n = min(n, limit)
    f_map = float(meta.get("f_map", 24.25 * fm.shape[-1] / 28.0))
    events = [tuple(e) for e in meta.get("events", [])]
    return fm[:n], gyro[:n], fps, f_map, events, n


def event_mask(events, fps: float, n: int) -> np.ndarray:
    """真值事件帧掩码——与层无关，所有层共用（公平比较的前提）。"""
    m = np.zeros(n, dtype=bool)
    for a, b in events:
        m[int(a * fps):min(int(b * fps) + 1, n)] = True
    return m


def trigger_stats(scores: np.ndarray, fps: float, n: int,
                  events: list[tuple[float, float]]) -> dict:
    trig = AdaptiveTrigger()
    for t in range(1, n):
        trig.update(t / fps, float(scores[t]), fps)
    clips = trig.finalize(n / fps)
    keep = np.zeros(n, dtype=bool)
    for a, b in clips:
        keep[int(a * fps):min(int(b * fps) + 1, n)] = True
    ev = event_mask(events, fps, n)
    hit_ev = sum(1 for a, b in events
                 if keep[int(a * fps):min(int(b * fps) + 1, n)].any())
    return {
        "stored": float(keep.mean()),
        "clips": len(clips),
        "event_recall": hit_ev / max(len(events), 1),
        "event_frame_cover": float((keep & ev).sum()) / max(int(ev.sum()), 1),
    }


def run_layer(fm, gyro, fps, f_map, events, n, L, kind, slow_ema, gate_k,
              spatial_leak) -> dict:
    cfg = CascadeConfig(
        cancel=CancelConfig(fps=fps, f_map=f_map, robust=True),
        n_layers=L, layer_kind=kind, slow_ema=slow_ema, gate_k=gate_k,
        spatial_leak=spatial_leak)
    canc = CascadeCancel(cfg)
    b = cfg.cancel.border
    raw = np.zeros(n, dtype=np.float64)
    sur = np.zeros(n, dtype=np.float64)
    # 触发读数用已发表口径：maxcell_score 作用于**末层** z 图（§2.3.2）
    score = np.zeros(n, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        o = canc.step(fm[i], gyro[i])
        raw[i] = float(np.abs(o["r"]).mean())
        sur[i] = float(o["surprise"])
        score[i] = float(maxcell_score(o["z_final"][None], b)[0])
    ms = (time.perf_counter() - t0) / n * 1e3

    warm = np.arange(n) >= WARMUP_FRAMES
    ev = event_mask(events, fps, n)
    quiet = warm & ~ev                     # 与层无关的样本划分
    event = warm & ev
    calm = warm & (sur < 4.0)              # 已发表的 calm 口径（供对照）

    def m(mask):
        return float(np.mean(raw[mask])) if mask.any() else float("nan")

    q = m(quiet)
    rec = {
        "quiet_residual": q,
        "event_residual": m(event),
        "event_contrast": (m(event) / q if np.isfinite(q) and q > 0
                           else float("nan")),
        "calm_residual": m(calm),
        "quiet_frames": int(quiet.sum()),
        "event_frames": int(event.sum()),
        "surprise_median": float(np.median(sur[warm])),
        "ms_per_frame": ms,
    }
    rec.update(trigger_stats(score, fps, n, events))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/aoe_event/val")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seqs", type=int, default=10)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--kinds", default="temporal,spatial")
    ap.add_argument("--slow-ema", type=float, default=0.02)
    ap.add_argument("--gate-k", type=float, default=4.0)
    ap.add_argument("--spatial-leak", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true",
                    help="额外跑 temporal 的 slow_ema × gate_k 网格"
                         "（含 gate_k=0 的'关掉稳态门'对照）")
    ap.add_argument("--sweep-seqs", type=int, default=4)
    ap.add_argument("--sweep-emas", default="0.02,0.1,0.3,0.6")
    ap.add_argument("--sweep-gates", default="4.0,0.0")
    ap.add_argument("--out", default="runs/cascade_scan.json")
    ap.add_argument("--force", action="store_true",
                    help="允许小规模运行覆盖更大的已有结果（默认拒绝，"
                         "防止冒烟测试盖掉正式 json）")
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    seqs = sorted(glob.glob(f"{args.src}/seq_*"))[:args.seqs]
    if not seqs:
        raise SystemExit(f"找不到序列: {args.src}/seq_*")

    t0 = time.time()
    report: dict = {
        "src": args.src, "n_frames_per_seq": args.n,
        "warmup_frames": WARMUP_FRAMES, "slow_ema": args.slow_ema,
        "gate_k": args.gate_k, "spatial_leak": args.spatial_leak,
        "method_note":
            "事件/平静划分用真值事件区间（与层无关）；残差为原始尺度（非 z）；"
            "score = maxcell_score 作用于末层 z 图（已发表读出）；聚合只在"
            "所有层都有有限值的序列上做",
        "seqs": [],
    }
    for s in seqs:
        fm, gyro, fps, f_map, events, n = load_seq(s, args.n)
        entry: dict = {"seq": os.path.basename(os.path.dirname(s)), "n": n,
                       "n_events": len(events), "f_map": f_map, "runs": {}}
        for kind in kinds:
            for L in layers:
                rec = run_layer(fm, gyro, fps, f_map, events, n, L, kind,
                                args.slow_ema, args.gate_k, args.spatial_leak)
                entry["runs"][f"{kind}_L{L}"] = rec
                print(f"  [{entry['seq']}] {kind} L={L} "
                      f"quiet={rec['quiet_residual']:.4f} "
                      f"contrast={rec['event_contrast']:.2f} "
                      f"stored={rec['stored']:.3f} "
                      f"evRec={rec['event_recall']:.2f} "
                      f"{rec['ms_per_frame']:.3f}ms")
        report["seqs"].append(entry)

    summary: dict = {"runs": {}}
    keys = ("quiet_residual", "event_residual", "event_contrast",
            "calm_residual", "surprise_median", "stored", "event_recall",
            "event_frame_cover", "ms_per_frame")
    for kind in kinds:
        tags = [f"{kind}_L{L}" for L in layers]
        finite = [q for q in report["seqs"]
                  if all(np.isfinite(q["runs"][t]["quiet_residual"])
                         and np.isfinite(q["runs"][t]["event_contrast"])
                         for t in tags)]
        base_c = float(np.mean([q["runs"][tags[0]]["event_contrast"]
                                for q in finite])) if finite else float("nan")
        base_q = float(np.mean([q["runs"][tags[0]]["quiet_residual"]
                                for q in finite])) if finite else float("nan")
        summary["runs"][kind] = {
            "n_seqs_used": len(finite),
            "n_seqs_total": len(report["seqs"]),
            "baseline_quiet": base_q, "baseline_contrast": base_c,
            "layers": {},
        }
        for L, tag in zip(layers, tags):
            e = {}
            for k in keys:
                vals = [q["runs"][tag][k] for q in finite
                        if np.isfinite(q["runs"][tag][k])]
                e[k] = float(np.mean(vals)) if vals else float("nan")
                e[k + "_std"] = float(np.std(vals)) if vals else float("nan")
            e["contrast_vs_L1"] = (e["event_contrast"] / base_c
                                   if np.isfinite(base_c) and base_c > 0
                                   else float("nan"))
            e["quiet_vs_L1"] = (e["quiet_residual"] / base_q
                                if np.isfinite(base_q) and base_q > 0
                                else float("nan"))
            wins = sum(1 for q in finite
                       if q["runs"][tag]["event_contrast"]
                       > q["runs"][tags[0]]["event_contrast"] + 1e-9)
            e["contrast_wins_vs_L1"] = f"{wins}/{len(finite)}"
            summary["runs"][kind]["layers"][str(L)] = e
    if args.sweep:
        # 超参网格：给"层数"假设一个公平的机会——如果默认参数只是没调好，
        # 这里应该能找到有效的组合。对照含 gate_k=0（关掉稳态门，允许把
        # 事件也学进预测里——理论上是"作弊"路径，但能验证门是否真的必要）。
        emas = [float(x) for x in args.sweep_emas.split(",") if x.strip()]
        gates = [float(x) for x in args.sweep_gates.split(",") if x.strip()]
        sweep_seqs = seqs[:args.sweep_seqs]
        grid: dict = {"n_seqs": len(sweep_seqs), "cells": {}}
        for ema in emas:
            for gk in gates:
                recs: list[dict] = []
                wins = 0
                for s in sweep_seqs:
                    fm, gyro, fps, f_map, events, n = load_seq(s, args.n)
                    r2 = run_layer(fm, gyro, fps, f_map, events, n, L=2,
                                   kind="temporal", slow_ema=ema, gate_k=gk,
                                   spatial_leak=args.spatial_leak)
                    r1 = run_layer(fm, gyro, fps, f_map, events, n, L=1,
                                   kind="temporal", slow_ema=ema, gate_k=gk,
                                   spatial_leak=args.spatial_leak)
                    recs.append(r2)
                    if (np.isfinite(r2["event_contrast"])
                            and np.isfinite(r1["event_contrast"])
                            and r2["event_contrast"] > r1["event_contrast"]):
                        wins += 1
                q = float(np.mean([r["quiet_residual"] for r in recs]))
                cts = [r["event_contrast"] for r in recs
                       if np.isfinite(r["event_contrast"])]
                ct = float(np.mean(cts)) if cts else float("nan")
                grid["cells"][f"ema{ema}_gate{gk}"] = {
                    "quiet_residual_L2": q, "event_contrast_L2": ct,
                    "contrast_wins_vs_L1": f"{wins}/{len(sweep_seqs)}",
                    "stored": float(np.mean([r["stored"] for r in recs])),
                    "event_recall": float(np.mean([r["event_recall"]
                                                   for r in recs])),
                }
                print(f"  [sweep] ema={ema} gate={gk}: quiet={q:.4f} "
                      f"contrast={ct:.2f} wins={wins}/{len(sweep_seqs)}")
        report["sweep"] = grid

    summary["wall_seconds"] = time.time() - t0
    report["summary"] = summary

    # 覆盖保护：小规模运行（冒烟测试/调试）不许静默盖掉更大的正式结果。
    # 实测踩到过：一次 --seqs 2 的冒烟把 10 段的正式 json 覆盖了。
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out))
            old_seqs = len(old.get("seqs", []))
            old_n = int(old.get("n_frames_per_seq", 0))
        except (OSError, ValueError):
            old_seqs = old_n = 0
        if not args.force and (len(seqs) < old_seqs or args.n < old_n):
            raise SystemExit(
                f"拒绝覆盖 {args.out}：现有结果更大"
                f"（{old_seqs} 段 x {old_n} 帧 > 本次 {len(seqs)} 段 x "
                f"{args.n} 帧）。确要覆盖请加 --force，或换个 --out 。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2)

    for kind in kinds:
        s = summary["runs"][kind]
        print(f"\n=== {kind}（{s['n_seqs_used']}/{s['n_seqs_total']} 段可比）===")
        print(f"{'层数':<5}{'平静残差':>10}{'相对L1':>9}{'事件对比度':>11}"
              f"{'对比vsL1':>10}{'胜负':>7}{'stored':>9}{'evRec':>7}{'ms/帧':>9}")
        for L in layers:
            e = s["layers"][str(L)]
            print(f"{L:<5}{e['quiet_residual']:>10.4f}{e['quiet_vs_L1']:>9.3f}"
                  f"{e['event_contrast']:>11.2f}{e['contrast_vs_L1']:>10.3f}"
                  f"{e['contrast_wins_vs_L1']:>7}{e['stored']:>9.3f}"
                  f"{e['event_recall']:>7.2f}{e['ms_per_frame']:>9.3f}")
    print("\n判读：平静残差降 + 对比度**升** = 真预测；残差降但对比度持平 = "
          "只是缩小增益（无信息增益）。")
    print(f"-> {args.out}（{summary['wall_seconds']:.1f}s）")


if __name__ == "__main__":
    main()

"""事件驱动架构仿真：廉价监控通道门控全量处理，测成本账与保真度。

背景（runs/event_profile.json 的刻画结论）：稀疏物理事件（drop/throw/
toss/break）在残差 z 里不可见（逐帧 AUC≈0.51），在 gyro 里弱（≈0.62），
在 featmap 差分里弱（≈0.55）。所以本仿真**不问"监控能否找到事件标注"**
（那是已结案的定义域问题），问的是一个工程问题：

  全量取消层只在监控窗内运行，能否保住密集路径的触发输出，
  以及平均成本降到多少？

设计：
  监控通道 M(t) = pooled |F_t − F_{t-1}|（无 warp、无统计的最便宜视觉读数；
  部署时对应 raw-pixel diff——本数据的 featmaps 是预计算的，注意这是
  监控质量的上界）。M 走与已发布片段相同的生产触发协议
  （scripts/train_downstream.py 的 clips_for_score "adaptive"）得到监控窗。
  门控两种重入模式：
    stale   — F_prev 直接用上次处理帧（欠账 K 帧 → 重入帧残差含全部欠账）
    bridged — 重入帧用缺口积分 gyro 桥接（expected_shift 的 dt=K/fps：
              step() 的 dt 恒为 1/fps，故临时把 cfg.fps 设为 fps/K 并喂
              缺口平均角速度，dx = f·tan(radians(mean)·K/fps) 即积分旋转）

指标（逐段 + 聚合）：
  duty      监控窗覆盖的帧占比（= 全量处理器的运行占空比）
  recall    事件区间被监控窗覆盖的比例（≥1 帧落在区间内）；附平均延迟
  cov/pre   监控窗对密集触发片段的覆盖 / 窗内落在密集片段里的占比
  z 保真    处理帧上 z_gated 与 z_dense 的相关与比值（stats 只更新在窗内，
            这是事件驱动架构真正的设计风险）
  成本      实测 cancel 步耗时 + 实测监控耗时 + 已发布 backbone 数字
            （README：采集层端到端 ~1.0 ms/帧，其中 cancel 在预计算
            featmaps 上实测 ~0.1 ms → backbone+IO ≈ 0.9 ms/帧）

用法：PYTHONPATH=. .venv/bin/python scripts/sim_event_driven.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from train_downstream import clips_for_score  # noqa: E402

from cerebellum_cancel.core import AdaptiveTrigger  # noqa: E402
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,  # noqa: E402
                                               maxcell_score, rollout)

BACKBONE_MS = 0.9     # 已发布口径：端到端 1.0 ms/帧 − cancel 实测 ≈0.1


def load_seq(seq: str):
    meta = json.load(open(f"{seq}/events.json"))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    import csv
    with open(f"{seq}/imu.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    gyro = np.array([[float(r[f"gyroscope.{a}"]) for a in ("x", "y", "z")]
                     for r in rows], dtype=np.float32)
    labels = np.load(f"{seq}/labels.npy") > 0
    n = min(len(fm), len(gyro))
    cfg = CancelConfig(fps=float(meta["fps"]), robust=True, source=seq)
    if "f_map" in meta:
        cfg.f_map = float(meta["f_map"])
    return meta, fm[:n], gyro[:n], labels[:n], cfg


def monitor_windows(fm: np.ndarray, fps: float, onset_k: float):
    """pooled diff 喂生产触发协议（门限扫参）→ 监控窗（秒）。

    门限必须扫参而不是沿用 maxcell z² 流的 k=6：那是为已发布分数的
    尾形标定的，diff 信号尾形不同，直接套 k=6 在任何一段都永不触发
    （实测 duty 全 0）。方法学沿用 §2.3.8 的守门工作点曲线。
    """
    from cerebellum_cancel.core import TriggerConfig
    p = fm.reshape(len(fm), -1)
    d = np.zeros(len(fm), dtype=np.float64)
    d[1:] = np.abs(p[1:] - p[:-1]).mean(axis=1)
    trig = AdaptiveTrigger(TriggerConfig(threshold_k=onset_k,
                                         off_k=onset_k / 2))
    for t in range(1, len(fm)):
        trig.update(t / fps, float(d[t]), fps)
    return trig.finalize(len(fm) / fps), d


def window_mask(windows, n: int, fps: float) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in windows:
        m[int(a * fps):int(b * fps)] = True
    return m


def gated_rollout(fm: np.ndarray, gyro: np.ndarray, cfg: CancelConfig,
                  mask: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """只在 mask 帧上运行取消层。返回 (z_surprise 逐处理帧, 处理帧索引)。

    bridged 模式：缺口首帧临时把 cfg.fps 降为 fps/K 并喂缺口平均 gyro，
    让 expected_shift 积分整个缺口的旋转（见模块 docstring）。
    """
    fps = cfg.fps
    canc = FeatureMapCancel(cfg)
    idx = np.flatnonzero(mask)
    z_out = np.zeros(len(idx), dtype=np.float64)
    for i, t in enumerate(idx):
        if mode == "bridged" and i > 0:
            gap = t - idx[i - 1]
            if gap > 1:
                g = gyro[t - gap:t].mean(axis=0) * gap
                canc.cfg.fps = fps / gap
                out = canc.step(fm[t], g)
                canc.cfg.fps = fps
            else:
                out = canc.step(fm[t], gyro[t])
        else:
            out = canc.step(fm[t], gyro[t])
        z_out[i] = out["surprise"]
    return z_out, idx


def clips_mask(clips, n: int, fps: float) -> np.ndarray:
    return window_mask(clips, n, fps)


def main():
    seqs = sorted(glob.glob("data/aoe_event/val/seq_*"))
    rows = []
    # 成本实测（seq_000 上，float32 featmaps）
    meta0, fm0, gyro0, _, cfg0 = load_seq(seqs[0])
    canc = FeatureMapCancel(cfg0)
    t0 = time.time()
    rollout(canc, fm0, gyro0)
    cancel_ms = (time.time() - t0) / len(fm0) * 1000
    p = fm0.reshape(len(fm0), -1)
    t0 = time.time()
    for t in range(1, len(fm0)):
        float(np.abs(p[t] - p[t - 1]).mean())
    mon_ms = (time.time() - t0) / len(fm0) * 1000
    print(f"实测: cancel={cancel_ms:.3f} ms/帧  monitor(diff)={mon_ms:.3f} ms/帧"
          f"  (backbone 用已发布 {BACKBONE_MS} ms/帧)")

    sweep = [2.0, 3.0, 4.0, 6.0, 9.0, 14.0]
    for seq in seqs:
        meta, fm, gyro, labels, cfg = load_seq(seq)
        fps = cfg.fps
        n = len(fm)
        # 密集基线（生产路径）：片段提取必须用 maxcell_score(z)（score_seq
        # 的同一读数，已发布 stored 数字来自它）；surprise 标量流只用于
        # 门控保真对比。
        canc = FeatureMapCancel(cfg)
        z_maps, sur = rollout(canc, fm, gyro)
        s_dense = maxcell_score(z_maps)
        clips_dense = clips_for_score(s_dense, fps, n, "adaptive")
        stored_dense = sum(b - a for a, b in clips_dense) / (n / fps)
        ev_frames = [(int(round(a * fps)), int(round(b * fps)))
                     for a, b in meta["events"]]
        dm = clips_mask(clips_dense, n, fps)
        for k in sweep:
            wins, _ = monitor_windows(fm, fps, k)
            m = window_mask(wins, n, fps)
            duty = float(m.mean())
            rec = {"seq": os.path.basename(seq), "onset_k": k, "duty": duty,
                   "stored_dense": stored_dense,
                   "n_events": len(ev_frames)}
            hit, lats = 0, []
            for a, b in ev_frames:
                b = min(b, n)
                if a >= n or b <= a:
                    continue
                inside = np.flatnonzero(m[a:b])
                if inside.size:
                    hit += 1
                    lats.append(inside[0] / fps - a / fps)
            rec["event_recall"] = hit / len(ev_frames)
            rec["event_latency_s"] = (float(np.median(lats))
                                      if lats else None)
            rec["dense_cov"] = float(dm[m].mean()) if m.any() else 0.0
            rec["dense_prec"] = float(m[dm].mean()) if dm.any() else 0.0
            for gmode in ("stale", "bridged"):
                if not m.any():
                    rec[f"z_corr_{gmode}"] = float("nan")
                    rec[f"z_ratio_{gmode}"] = float("nan")
                    continue
                z_g, idx = gated_rollout(fm, gyro, cfg, m, gmode)
                z_d = sur[idx]
                ok = z_d > 0
                if ok.sum() > 2:
                    rec[f"z_corr_{gmode}"] = float(
                        np.corrcoef(z_g[ok], z_d[ok])[0, 1])
                    rec[f"z_ratio_{gmode}"] = float(
                        np.median(z_g[ok] / z_d[ok]))
                else:
                    rec[f"z_corr_{gmode}"] = float("nan")
                    rec[f"z_ratio_{gmode}"] = float("nan")
            full_ms = BACKBONE_MS + cancel_ms
            rec["cost_dense_ms"] = full_ms
            rec["cost_gated_ms"] = duty * full_ms + (1 - duty) * mon_ms
            rows.append(rec)
            print(f"{rec['seq']} k={k:4.1f}: duty={duty:.3f} "
                  f"event_recall={rec['event_recall']:.2f} "
                  f"dense_cov={rec['dense_cov']:.2f} "
                  f"dense_prec={rec['dense_prec']:.2f} "
                  f"z_corr stale/bridged={rec['z_corr_stale']:.2f}/"
                  f"{rec['z_corr_bridged']:.2f} "
                  f"z_ratio stale/bridged={rec['z_ratio_stale']:.2f}/"
                  f"{rec['z_ratio_bridged']:.2f} "
                  f"cost={rec['cost_gated_ms']:.3f} ms", flush=True)

    def agg(key, subset):
        v = [r[key] for r in subset
             if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        return {"median": float(np.median(v)), "mean": float(np.mean(v))} \
            if v else {"median": float("nan"), "mean": float("nan")}

    by_k = []
    for k in sweep:
        sub = [r for r in rows if r["onset_k"] == k]
        by_k.append({
            "onset_k": k,
            "n_segments": len(sub),
            "duty": agg("duty", sub),
            "event_recall": agg("event_recall", sub),
            "dense_cov": agg("dense_cov", sub),
            "dense_prec": agg("dense_prec", sub),
            "z_corr_stale": agg("z_corr_stale", sub),
            "z_corr_bridged": agg("z_corr_bridged", sub),
            "z_ratio_stale": agg("z_ratio_stale", sub),
            "z_ratio_bridged": agg("z_ratio_bridged", sub),
            "cost_gated_ms": agg("cost_gated_ms", sub),
        })
    summary = {
        "n_segments": len({r["seq"] for r in rows}),
        "measured_ms": {"cancel": cancel_ms, "monitor_diff": mon_ms,
                        "backbone_published": BACKBONE_MS},
        "stored_dense_median": float(np.median(
            [r["stored_dense"] for r in rows if r["onset_k"] == sweep[0]])),
        "cost_dense_ms": BACKBONE_MS + cancel_ms,
        "by_onset_k": by_k,
    }
    print("\n== 聚合（按监控门限） ==")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open("runs/event_driven_sim.json", "w") as fp:
        json.dump({"summary": summary, "per_seq": rows}, fp, indent=1,
                  ensure_ascii=False)
    print("wrote runs/event_driven_sim.json")


if __name__ == "__main__":
    main()

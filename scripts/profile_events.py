"""稀疏物理事件刻画：drop/throw/toss/break 事件在残差域与 IMU 域长什么样。

回答的问题（全部用实测数字）：
  1. 每个事件区间里，残差 z（maxcell）升到多少？峰在 onset 前还是后？
  2. 同一区间里，|gyro| 升到多少？峰对齐吗？（事件在廉价 IMU 通道里可见吗？）
  3. 跨 89 个事件，gyro 峰与 z 峰相关吗？事件与安静期的可分性（p50/p95）？
  4. labels.npy 与 events.json 是否仍逐帧等价（IoU）？

产出 runs/event_profile.json；本脚本只读数据，不改任何生产路径。

用法：PYTHONPATH=. .venv/bin/python scripts/profile_events.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               maxcell_score, rollout)

WINDOW_BEFORE = 2.0   # 事件 onset 前纳入剖面的秒数
WINDOW_AFTER = 2.0    # 事件 offset 后纳入剖面的秒数


def load_seq(seq: str):
    meta = json.load(open(f"{seq}/events.json"))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    # load_imu_csv 只保留 gyro（ImuSample 无 trans 字段）；刻画需要平移列，
    # 这里直接按表头取 gyroscope.* 与 translation.*。
    import csv
    with open(f"{seq}/imu.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    gyro = np.array([[float(r[f"gyroscope.{a}"]) for a in ("x", "y", "z")]
                     for r in rows], dtype=np.float32)
    trans = np.array([[float(r[f"translation.{a}"]) for a in ("x", "y", "z")]
                      for r in rows], dtype=np.float32)
    labels = np.load(f"{seq}/labels.npy").astype(np.float32)
    n = min(len(fm), len(gyro))
    cfg = CancelConfig(fps=float(meta["fps"]), robust=True, source=seq)
    if "f_map" in meta:
        cfg.f_map = float(meta["f_map"])
    canc = FeatureMapCancel(cfg)
    z, _ = rollout(canc, fm[:n], gyro[:n])
    score = maxcell_score(z)                       # (n,) 逐帧残差读数
    gnorm = np.linalg.norm(gyro[:n], axis=1)       # deg/s
    tnorm = np.linalg.norm(trans[:n], axis=1)
    return meta, score, gnorm, tnorm, labels[:n]


def event_profile(score: np.ndarray, gnorm: np.ndarray, tnorm: np.ndarray,
                  a: int, b: int, fps: float) -> dict:
    """单个事件区间 [a,b)（帧索引）的两侧剖面统计。"""
    w0 = max(0, int(a - WINDOW_BEFORE * fps))
    w1 = min(len(score), int(b + WINDOW_AFTER * fps))
    win = slice(w0, w1)
    s, g, tn = score[win], gnorm[win], tnorm[win]
    # 剖面内的全局峰位置（相对 onset，秒；负 = 峰在 onset 之前）
    rel = (np.arange(w0, w1) - a) / fps
    return {
        "dur_s": (b - a) / fps,
        "z_peak": float(s.max()),
        "z_peak_lag_s": float(rel[int(s.argmax())]),
        "z_in_med": float(np.median(score[a:b])),
        "z_out_med": float(np.median(np.concatenate(
            [score[w0:a], score[b:w1]])) if a > w0 or b < w1 else np.nan),
        "g_peak": float(g.max()),
        "g_peak_lag_s": float(rel[int(g.argmax())]),
        "g_in_med": float(np.median(gnorm[a:b])),
        "g_out_med": float(np.median(np.concatenate(
            [gnorm[w0:a], gnorm[b:w1]])) if a > w0 or b < w1 else np.nan),
        "t_peak": float(tn.max()),
        "t_in_med": float(np.median(tnorm[a:b])),
    }


def main():
    seqs = sorted(glob.glob("data/aoe_event/val/seq_*"))
    per_event: list[dict] = []
    per_seq: dict[str, dict] = {}
    for seq in seqs:
        meta, score, gnorm, tnorm, labels = load_seq(seq)
        fps = float(meta["fps"])
        n = len(score)
        ev_frames = [(int(round(a * fps)), int(round(b * fps)))
                     for a, b in meta["events"]]
        ev_mask = np.zeros(n, dtype=bool)
        for a, b in ev_frames:
            ev_mask[a:min(b, n)] = True
        lab_mask = labels > 0
        inter = float((ev_mask & lab_mask).sum())
        union = float((ev_mask | lab_mask).sum())
        profs = []
        for a, b in ev_frames:
            b = min(b, n)
            if a >= n or b <= a:
                continue
            p = event_profile(score, gnorm, tnorm, a, b, fps)
            p["seq"] = os.path.basename(seq)
            profs.append(p)
            per_event.append(p)
        per_seq[seq] = {
            "n_frames": n,
            "fps": fps,
            "n_events": len(profs),
            "label_iou": inter / union if union else 1.0,
            "event_coverage": float(ev_mask.mean()),
            "z_calm_p50": float(np.median(score[~ev_mask])),
            "z_calm_p95": float(np.percentile(score[~ev_mask], 95)),
            "z_event_p50": float(np.median(score[ev_mask])),
            "z_event_p95": float(np.percentile(score[ev_mask], 95)),
            "g_calm_p95": float(np.percentile(gnorm[~ev_mask], 95)),
            "g_event_p95": float(np.percentile(gnorm[ev_mask], 95)),
        }
        print(f"{os.path.basename(seq)}: events={len(profs)} "
              f"cov={per_seq[seq]['event_coverage']:.3f} "
              f"IoU={per_seq[seq]['label_iou']:.4f} "
              f"z calm p50/p95={per_seq[seq]['z_calm_p50']:.0f}/"
              f"{per_seq[seq]['z_calm_p95']:.0f} "
              f"event p50/p95={per_seq[seq]['z_event_p50']:.0f}/"
              f"{per_seq[seq]['z_event_p95']:.0f}")

    zpk = np.array([p["z_peak"] for p in per_event])
    gpk = np.array([p["g_peak"] for p in per_event])
    zlg = np.array([p["z_peak_lag_s"] for p in per_event])
    glg = np.array([p["g_peak_lag_s"] for p in per_event])
    dur = np.array([p["dur_s"] for p in per_event])
    # 可分性：事件内 p50 / 安静期 p95（同段内聚合用全体事件的分组中位）
    z_in = np.array([p["z_in_med"] for p in per_event])
    z_out = np.array([p["z_out_med"] for p in per_event])
    g_in = np.array([p["g_in_med"] for p in per_event])
    g_out = np.array([p["g_out_med"] for p in per_event])
    summary = {
        "n_events": len(per_event),
        "n_segments": len(seqs),
        "duration_s": {
            "median": float(np.median(dur)), "p25": float(np.percentile(dur, 25)),
            "p75": float(np.percentile(dur, 75)),
        },
        "z_peak": {
            "median": float(np.median(zpk)),
            "p25": float(np.percentile(zpk, 25)),
            "p75": float(np.percentile(zpk, 75)),
        },
        "z_peak_lag_s": {
            "median": float(np.median(zlg)),
            "p25": float(np.percentile(zlg, 25)),
            "p75": float(np.percentile(zlg, 75)),
            "frac_peak_inside": float(((zlg >= 0) & (zlg <= dur)).mean()),
        },
        "g_peak_lag_s": {
            "median": float(np.median(glg)),
            "frac_peak_inside": float(((glg >= 0) & (glg <= dur)).mean()),
        },
        "corr_z_g_peak": float(np.corrcoef(np.log(zpk), np.log(gpk))[0, 1]),
        "z_in_over_out": float(np.median(z_in / np.maximum(z_out, 1e-9))),
        "g_in_over_out": float(np.median(g_in / np.maximum(g_out, 1e-9))),
        "g_peak_deg_s": {
            "median": float(np.median(gpk)),
            "p95": float(np.percentile(gpk, 95)),
        },
    }
    print("\n== 聚合（89 事件口径以实际计数为准） ==")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = {"summary": summary, "per_seq": per_seq, "per_event": per_event}
    os.makedirs("runs", exist_ok=True)
    with open("runs/event_profile.json", "w") as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False)
    print("\nwrote runs/event_profile.json")


if __name__ == "__main__":
    main()

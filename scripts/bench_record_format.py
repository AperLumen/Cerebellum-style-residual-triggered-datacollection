"""记录格式 v2 的实测基准：字节账、决策保真、warm start。

回答三个问题（GUIDE 的口径：只报实测数字，负结果照记）：

1. **字节账**——记录格式 vs 基线每帧多少字节？
   基线：float32 z 图（内存里的表示）、float16 z 图（featmaps 的落盘惯例）、
   float16 逐格 z（跨通道 RMS，demo 的可视化读出）。
   三种 z_policy：``all``（每帧留 z 图）、``keep_only``（默认，只对触发
   片段留 z 图，其余帧只留标量）、``none``（纯决策日志）。
2. **决策保真**——量化 + zlib 之后重放，触发片段是否与 float32 原始 z
   一致？两个口径分开报：
   - 无损 ``score`` 重放（格式主契约）：触发器只吃 score，它无损则决策无损。
   - 从反量化 z 图反推 score：**实测会漂**，且这是格式设计上刻意避免的
     路径（z 图是有损的观测载荷，不该被用来恢复决策）。
3. **warm start**——"中断后从侧车续采"是否等于"不中断地跑"？副产物是
   GUIDE §2.3.5 已知局限（统计未收敛、最早事件漏检）的工程解法。

用法::

  PYTHONPATH=. .venv/bin/python scripts/bench_record_format.py --n 2000 --seqs 6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.core import AdaptiveTrigger
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               maxcell_score, rollout)
from cerebellum_cancel.io import load_imu_csv
from cerebellum_cancel.record import (RecordWriter, dequantize, quantize,
                                      read_record)

SRC = "data/aoe_event/val"


def load_seq(seq: str, limit: int | None):
    """读一条真实序列的 featmaps + gyro，跑出 z 轨迹（与既有脚本同口径）。"""
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    if limit:
        n = min(n, limit)
    f_map = 24.25 * fm.shape[-1] / 28.0
    canc = FeatureMapCancel(CancelConfig(f_map=f_map, robust=True))
    z, surprise = rollout(canc, fm[:n], gyro[:n])
    score = maxcell_score(z)
    return z, surprise, score, fps, f_map, canc


def trigger_clips(score: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """与 train_downstream.py 的 "adaptive" 口径完全一致的触发片段。"""
    trig = AdaptiveTrigger()
    for t in range(1, len(score)):
        trig.update(t / fps, float(score[t]), fps)
    return trig.finalize(len(score) / fps)


def clips_equal(a, b, tol=1e-6) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(x1 - x2) <= tol and abs(y1 - y2) <= tol
               for (x1, y1), (x2, y2) in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--n", type=int, default=1200,
                    help="每条序列最多用多少帧（控制运行时间）")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--out", default="runs/record_format_bench.json")
    args = ap.parse_args()

    seqs = sorted(glob.glob(f"{args.src}/seq_*"))[:args.seqs]
    if not seqs:
        raise SystemExit(f"找不到序列: {args.src}/seq_*")

    report: dict = {"src": args.src, "n_frames_per_seq": args.n, "seqs": []}
    t0 = time.time()

    for seq in seqs:
        z, surprise, score, fps, f_map, canc = load_seq(seq, args.n)
        n, C, H, W = z.shape
        zc = np.sqrt((z ** 2).mean(axis=1))          # 逐格 z（跨通道 RMS）
        ref_clips = trigger_clips(score, fps)

        tmpd = tempfile.mkdtemp(prefix="cmci_bench_")
        try:
            rows: dict = {}

            # ---- 基线：未压缩的内存/落盘表示 ----
            rows["baseline_f32_zmap"] = z.astype(np.float32).nbytes / n
            rows["baseline_f16_zmap"] = z.astype(np.float16).nbytes / n
            rows["baseline_f16_cellz"] = zc.astype(np.float16).nbytes / n

            # ---- 本格式：z_policy × 量化档位 ----
            # 决策保真用到两个读数：
            #  1) 无损 score（记录里 float64 存下）-> 决策可精确重放。这是
            #     格式的**主契约**：触发器只吃 score，它无损则决策无损。
            #  2) 从反量化 z 图重算 score -> 观测级保真。z 图是有损的，
            #     它反推出的决策会漂——这正是不拿 z 反推决策的理由。
            keep_mask = np.zeros(n, dtype=bool)
            for a, b in ref_clips:
                lo, hi = int(a * fps), min(int(b * fps) + 1, n)
                keep_mask[lo:hi] = True

            mode_cases: tuple[tuple[str, str, float], ...] = (
                ("all_db0.5", "all", 0.5),
                ("all_db1.0", "all", 1.0),
                ("keep_db0.5", "keep_only", 0.5),
                ("none", "none", 0.5),
            )
            for tag, policy, deadband in mode_cases:
                path = os.path.join(tmpd, f"{tag}.cmcir")
                t_w = time.time()
                with RecordWriter(path, fps=fps, f_map=f_map, shape=z.shape[1:],
                                  mode="fixed", deadband=deadband,
                                  z_policy=policy) as w:
                    for i in range(n):
                        w.append(z[i], t=i / fps, surprise=float(surprise[i]),
                                 score=float(score[i]), keep=bool(keep_mask[i]))
                    s = w.close(stats=canc.dump_stats())
                w_ms = (time.time() - t_w) / n * 1e3
                with read_record(path) as rec:
                    assert not rec.truncated
                    clips_lossless = trigger_clips(rec.score, fps)
                rows[f"fmt_{tag}_bytes_per_frame"] = s["bytes_per_frame"]
                rows[f"fmt_{tag}_payload_per_frame"] = s["payload_bytes_per_frame"]
                rows[f"fmt_{tag}_write_ms_per_frame"] = w_ms
                rows[f"fmt_{tag}_clips_match_score"] = clips_equal(
                    clips_lossless, ref_clips)
                if tag == "all_db0.5":
                    with read_record(path) as rec:
                        qs = [maxcell_score(rec.frame(i)[None],
                                            canc.cfg.border)[0]
                              for i in range(n)]
                        err = np.abs(np.asarray([rec.frame(i)
                                                 for i in range(n)]) - z)
                        clipped = np.abs(z) > rec.header["quant"]["zmax"]
                        err_in = (float(err[~clipped].max())
                                  if (~clipped).any() else 0.0)
                    rows["fmt_all_db0.5_err_within_range"] = err_in
                    rows["fmt_all_db0.5_clipped_frac"] = float(clipped.mean())
                    rows["fmt_all_db0.5_clips_match_zrefit"] = clips_equal(
                        trigger_clips(np.asarray(qs), fps), ref_clips)
                if tag == "keep_db0.5":
                    rows["keep_only_effective_ratio"] = (
                        s["bytes_per_frame"] / rows["baseline_f16_zmap"])
                    rows["keep_only_total_bytes"] = s["bytes"]

            rows["stored_frac_trigger"] = float(keep_mask.mean())
            # keep_only 的真实成本账：只有触发片段留 z 图，其余帧 27 字节标量。
            # 用 payload 口径与"全量帧都存 z 图"比（侧车不计入逐帧成本）：
            rows["keep_vs_all_payload_ratio"] = (
                rows["fmt_keep_db0.5_payload_per_frame"]
                / max(rows["fmt_all_db0.5_payload_per_frame"], 1e-9))
            rows["keep_vs_f16z_payload_ratio"] = (
                rows["fmt_keep_db0.5_payload_per_frame"]
                / rows["baseline_f16_zmap"])

            # ---- 原始 int8（关 zlib）：分离"量化"与"熵编码"各自的贡献 ----
            path = os.path.join(tmpd, "raw.cmcir")
            with RecordWriter(path, fps=fps, f_map=f_map, shape=z.shape[1:],
                              codec="none", deadband=0.5,
                              z_policy="all") as w:
                for i in range(n):
                    w.append(z[i])
                s = w.close()
            rows["fmt_int8_only_bytes_per_frame"] = s["bytes_per_frame"]

            # ---- float16 + zlib（对照：不做量化，只做熵编码）----
            import zlib
            f16 = z.astype(np.float16).tobytes()
            rows["fmt_f16_zlib_bytes_per_frame"] = (
                len(zlib.compress(f16, 6)) / n)

            # ---- warm start：跨录制复用统计 ----
            # 正确口径："中断后从侧车续采" 必须等于 "不中断地跑"。
            # 侧车要带 mu/var **和** F_prev，否则续采首帧残差恒为 0。
            s_idx = n // 2
            K = min(60, n - s_idx)
            cont = FeatureMapCancel(CancelConfig(f_map=f_map, robust=True))
            sur_ref = []
            for i in range(n):
                out = cont.step(z[i], np.zeros(3))
                if s_idx <= i < s_idx + K:
                    sur_ref.append(out["surprise"])
            half = FeatureMapCancel(CancelConfig(f_map=f_map, robust=True))
            for i in range(s_idx):
                half.step(z[i], np.zeros(3))
            stats_half = half.dump_stats()

            path = os.path.join(tmpd, "warm.cmcir")
            with RecordWriter(path, fps=fps, f_map=f_map, shape=z.shape[1:]) as w:
                for i in range(s_idx, s_idx + K):
                    w.append(z[i])
                w.close(stats=stats_half)
            with read_record(path) as rec:
                resumed = FeatureMapCancel(CancelConfig(f_map=f_map, robust=True))
                resumed.load_stats(dict(rec.stats))
                sur_resumed = [resumed.step(z[i], np.zeros(3))["surprise"]
                               for i in range(s_idx, s_idx + K)]
                # 侧车容量（mu/var/F_prev）
                rows["stats_sidecar_bytes"] = int(
                    sum(v.nbytes for v in rec.stats.values()))
            rows["warm_start_matches_continuation"] = bool(
                np.allclose(sur_resumed, sur_ref, rtol=1e-5, atol=1e-5))
            rows["warm_start_max_rel_diff"] = float(
                np.max(np.abs(np.asarray(sur_resumed) - np.asarray(sur_ref))
                       / (np.abs(sur_ref) + 1e-6)))

            # 冷启动 vs warm start 的首帧可见性（GUIDE §2.3.5 的已知局限）
            cold = FeatureMapCancel(CancelConfig(f_map=f_map, robust=True))
            rows["cold_start_first_frame_surprise"] = float(
                cold.step(z[0], np.zeros(3))["surprise"])
            warm0 = FeatureMapCancel(CancelConfig(f_map=f_map, robust=True))
            warm0.load_stats(stats_half)
            rows["warm_start_first_frame_surprise"] = float(
                warm0.step(z[s_idx], np.zeros(3))["surprise"])

            # ---- zlib 档位：压缩比 vs 成本（固定 z_policy=all 才有可比性）----
            for lvl in (1, 6):
                path = os.path.join(tmpd, f"lvl{lvl}.cmcir")
                t_w = time.time()
                with RecordWriter(path, fps=fps, f_map=f_map, shape=z.shape[1:],
                                  level=lvl, z_policy="all") as w:
                    for i in range(n):
                        w.append(z[i])
                    s = w.close()
                rows[f"zlib_lvl{lvl}_B"] = s["bytes_per_frame"]
                rows[f"zlib_lvl{lvl}_ms"] = (time.time() - t_w) / n * 1e3

            report["seqs"].append({
                "seq": os.path.basename(os.path.dirname(seq)),
                "n_frames": n, "frames_per_seq_total": len(z),
                "n_events": len(json.load(open(f"{seq}/events.json"))
                                .get("events", [])),
                "ref_clips": ref_clips,
                **rows,
            })
            print(f"[{os.path.basename(os.path.dirname(seq))}] n={n} "
                  f"f16z={rows['baseline_f16_zmap']:.0f}B "
                  f"keep={rows['fmt_keep_db0.5_bytes_per_frame']:.0f}B "
                  f"all={rows['fmt_all_db0.5_bytes_per_frame']:.0f}B "
                  f"none={rows['fmt_none_bytes_per_frame']:.1f}B "
                  f"stored={rows['stored_frac_trigger']:.3f} "
                  f"clip_match={rows['fmt_all_db0.5_clips_match_score']}")
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)

    # ---- 汇总 ----
    def mean(key):
        vals = [s[key] for s in report["seqs"] if key in s]
        return float(np.mean(vals)) if vals else float("nan")

    report["summary"] = {
        "baseline_f32_zmap_B": mean("baseline_f32_zmap"),
        "baseline_f16_zmap_B": mean("baseline_f16_zmap"),
        "baseline_f16_cellz_B": mean("baseline_f16_cellz"),
        # 逐帧载荷口径（不含头/页脚/统计侧车这些一次性开销）
        "fmt_all_B": mean("fmt_all_db0.5_payload_per_frame"),
        "fmt_keep_B": mean("fmt_keep_db0.5_payload_per_frame"),
        "fmt_none_B": mean("fmt_none_payload_per_frame"),
        "fmt_all_db1.0_B": mean("fmt_all_db1.0_payload_per_frame"),
        # 含一次性开销的整文件口径（短视频上会被侧车抬高）
        "fmt_all_incl_fixed_B": mean("fmt_all_db0.5_bytes_per_frame"),
        "fmt_keep_incl_fixed_B": mean("fmt_keep_db0.5_bytes_per_frame"),
        "fmt_int8_only_B": mean("fmt_int8_only_bytes_per_frame"),
        "fmt_f16_zlib_B": mean("fmt_f16_zlib_bytes_per_frame"),
        "compression_vs_f16_all": mean("baseline_f16_zmap")
        / max(mean("fmt_all_db0.5_payload_per_frame"), 1e-9),
        "compression_vs_f32_all": mean("baseline_f32_zmap")
        / max(mean("fmt_all_db0.5_payload_per_frame"), 1e-9),
        "keep_vs_all_payload_ratio": mean("keep_vs_all_payload_ratio"),
        "keep_vs_f16z_payload_ratio": mean("keep_vs_f16z_payload_ratio"),
        "clips_match_score": all(s["fmt_all_db0.5_clips_match_score"]
                                 and s["fmt_keep_db0.5_clips_match_score"]
                                 and s["fmt_none_clips_match_score"]
                                 for s in report["seqs"]),
        "clips_match_zrefit": all(s["fmt_all_db0.5_clips_match_zrefit"]
                                  for s in report["seqs"]),
        "err_within_range_max": max(s["fmt_all_db0.5_err_within_range"]
                                    for s in report["seqs"]),
        "clipped_frac": mean("fmt_all_db0.5_clipped_frac"),
        "stored_frac_trigger": mean("stored_frac_trigger"),
        "write_ms_all": mean("fmt_all_db0.5_write_ms_per_frame"),
        "write_ms_keep": mean("fmt_keep_db0.5_write_ms_per_frame"),
        "stats_sidecar_B": mean("stats_sidecar_bytes"),
        "warm_start_fixed_first_frame": all(
            s["warm_start_first_frame_surprise"] > 0 for s in report["seqs"]),
        "cold_start_first_frame_zero": all(
            s["cold_start_first_frame_surprise"] == 0 for s in report["seqs"]),
        "warm_start_matches_continuation": all(
            s["warm_start_matches_continuation"] for s in report["seqs"]),
        "warm_start_max_rel_diff": max(s["warm_start_max_rel_diff"]
                                       for s in report["seqs"]),
        "wall_seconds": time.time() - t0,
    }
    report["sot"] = "format v2 = self-describing container + int8/deadband " \
                    "+ zlib + stats sidecar"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2)
    sm = report["summary"]
    print("\n=== 汇总 ===")
    print(f"每帧字节: f32 z图 {sm['baseline_f32_zmap_B']:.0f} | "
          f"f16 z图 {sm['baseline_f16_zmap_B']:.0f} | "
          f"f16 逐格 z {sm['baseline_f16_cellz_B']:.0f}")
    print(f"本格式（逐帧载荷口径，侧车不摊入）:")
    print(f"  z_policy=all      {sm['fmt_all_B']:6.0f} B/帧 "
          f"（{sm['compression_vs_f16_all']:.1f}x vs f16 z图，"
          f"{sm['compression_vs_f32_all']:.1f}x vs f32；deadband 1.0 -> "
          f"{sm['fmt_all_db1.0_B']:.0f}）")
    print(f"  z_policy=keep_only{sm['fmt_keep_B']:7.0f} B/帧 "
          f"（为 all 的 {sm['keep_vs_all_payload_ratio'] * 100:.1f}%，"
          f"为 f16 z图全量的 {sm['keep_vs_f16z_payload_ratio'] * 100:.1f}%）")
    print(f"  z_policy=none     {sm['fmt_none_B']:6.1f} B/帧（纯决策日志）")
    print(f"  对照: int8 无熵编码 {sm['fmt_int8_only_B']:.0f} | "
          f"f16+zlib {sm['fmt_f16_zlib_B']:.0f}")
    print(f"  含一次性开销（头/页脚/侧车，短录制会被抬高）: "
          f"all {sm['fmt_all_incl_fixed_B']:.0f} / "
          f"keep {sm['fmt_keep_incl_fixed_B']:.0f} B/帧")
    print(f"决策保真（片段逐位一致）: 无损score {sm['clips_match_score']} | "
          f"int8 反推 {sm['clips_match_zrefit']}")
    print(f"z图观测保真: 范围内最大误差 {sm['err_within_range_max']:.4f}σ，"
          f"截顶比例 {sm['clipped_frac'] * 100:.3f}%")
    print(f"写入成本 {sm['write_ms_keep']:.3f} ms/帧（keep_only）/ "
          f"{sm['write_ms_all']:.3f} ms/帧（all）；"
          f"统计侧车 {sm['stats_sidecar_B'] / 1024:.1f} KiB/序列")
    print(f"warm start: 首帧非零 {sm['warm_start_fixed_first_frame']} "
          f"(冷启动首帧全零 {sm['cold_start_first_frame_zero']})，"
          f"续采==不中断 {sm['warm_start_matches_continuation']} "
          f"(最大相对差 {sm['warm_start_max_rel_diff']:.2e})")
    print(f"-> {args.out}（{sm['wall_seconds']:.1f}s）")


if __name__ == "__main__":
    main()

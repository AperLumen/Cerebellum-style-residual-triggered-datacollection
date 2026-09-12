"""把现有序列目录摄入成 CMCI 多模态数据集。

格式与契约见 ``GUIDE.md §4.2``。

## 输入：两种来源（自动识别）

1. **prepared-real**（`data/aoe*` 系列）：`events.json` + `imu.csv`
   (+ `featmaps.npy` / `features.npy` / `labels.npy`)。
   残差分数由 ``FeatureMapCancel`` 在缓存的特征图上重放得到——这与已发表的
   全部实验同一条链路，所以数据集的 score 与论文数字可逐位对齐。
   帧通常**不在磁盘**（磁盘预算），此时用 ``--frames none`` 或 ``--frames symlink``。
2. **raw-capture**（`data/real_imu` 系列）：有真实帧 + `imu.csv` + `accel.npy`。
   用 ``--frontend`` 在线跑前端得到特征图与残差（需要 frames）。

## 关于 weight 的纪律

本脚本写入的 ``weight`` 是**待验证主张**（残余不可压缩性），不是重要性。
manifest 与自动生成的 README 都会把状态标成 ``CLAIM, NOT ESTABLISHED``。
验证脚本：``scripts/eval_weight_value.py``（尚未执行）。

Usage:
  PYTHONPATH=. .venv/bin/python scripts/build_dataset.py \
      --source data/aoe_event --out data/cmci_ds_aoe_event --frames none
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul")

from cerebellum_cancel.dataset import (DatasetWriter, build_section,
                                       compute_weight)
from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                              maxcell_score, rollout)
from cerebellum_cancel.io import load_imu_csv

MAX_FRAMES_DEFAULT = 6000       # 每段上限，控制构建时间与体积


def sha1_of(path: str, limit: int = 1 << 20) -> str:
    """文件内容指纹（默认只哈希前 1 MiB，够做去重且快）。"""
    h = hashlib.sha1()
    with open(path, "rb") as fp:
        h.update(fp.read(limit))
    return h.hexdigest()[:16]


def load_prepared(seq: str, max_frames: int, pred: str):
    """读取 prepared-real 段：从缓存特征图重放残差（与论文同链路）。"""
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro_full = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                         dtype=np.float32)
    # n 是权威长度：featmaps / imu / 上限三者取最小，且**所有**载荷都必须
    # 截到这个长度（逐帧对齐是硬契约，校验器会拒绝任何不符的载荷）
    n = min(len(fm), len(gyro_full), max_frames)
    gyro = gyro_full[:n]
    cfg = CancelConfig(fps=fps, robust=True, pred=pred, source=seq)
    if "f_map" in meta:
        cfg.f_map = float(meta["f_map"])
    canc = FeatureMapCancel(cfg)
    z, sur = rollout(canc, fm[:n], gyro[:n])
    score = maxcell_score(z).astype(np.float64)
    t = np.arange(n, dtype=np.float64) / fps
    # 未归一化的残差幅度：完整性巡检的正确输入（surprise 已被在线统计
    # 归一化，会把故障吸收掉，LABEL_AUDIT §5）
    sd = np.sqrt(np.asarray(canc.var)) + 1e-3
    raw_amp = np.abs(z * sd).reshape(n, -1).mean(axis=1).astype(np.float32)

    labels = None
    lp = f"{seq}/labels.npy"
    if os.path.exists(lp):
        labels = np.load(lp).astype(np.float32)[:n]

    extra = {"fps": fps, "f_map": float(cfg.f_map), "pred": pred,
             "n_events": len(meta.get("events", [])),
             "event_verbs": meta.get("event_verbs"),
             "featmap_sha1": sha1_of(f"{seq}/featmaps.npy"),
             "imu_sha1": sha1_of(f"{seq}/imu.csv")}
    if os.path.exists(f"{seq}/features.npy"):
        extra["features_sha1"] = sha1_of(f"{seq}/features.npy")
        extra["features_dim"] = int(
            np.load(f"{seq}/features.npy", mmap_mode="r").shape[1])
    return t, gyro, score, sur, raw_amp, labels, extra


def load_frames_jpeg(seq: str, n: int, quality: int, max_side: int):
    """把 frames/ 下的 PNG 读进来转成 JPEG 字节串（可选模态）。"""
    import cv2
    paths = sorted(glob.glob(f"{seq}/frames/*.png"))[:n]
    if not paths:
        return None
    out: list[bytes] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            out.append(b"")
            continue
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)))
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        out.append(buf.tobytes() if ok else b"")
    if len(out) < n:
        out += [b""] * (n - len(out))
    return out


def load_from_frames(seq: str, max_frames: int, pred: str, frontend_spec: str,
                     jpeg_quality: int, max_side: int):
    """从 **frames/ 目录**摄入（raw-capture 路径，无缓存特征图）。

    这条路径把帧送进 ``AcquisitionLayer`` 的可插拔前端得到特征图，再跑
    `FeatureMapCancel` 得到残差——与在线部署同一条链路。

    与 ``load_prepared`` 的区别：那条走缓存特征图（与已发表实验逐位对齐），
    这条必须**真的有帧**（因此可以同时带上视频模态）。
    """
    import cv2
    from cerebellum_cancel.acquisition import _build_frontend
    from cerebellum_cancel.featmap_cancel import (CancelConfig,
                                                  FeatureMapCancel,
                                                  maxcell_score, rollout)

    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    gyro_full = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                         dtype=np.float32)
    paths = sorted(glob.glob(f"{seq}/frames/*.png"))
    if not paths:
        raise IOError(f"{seq}/frames 为空——raw-capture 路径需要真实帧")
    n = min(len(paths), len(gyro_full), max_frames)

    frontend = _build_frontend(frontend_spec, None)
    fmaps, jpegs = [], []
    for p in paths[:n]:
        bgr = cv2.imread(p)
        if bgr is None:
            raise IOError(f"无法读取 {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fmaps.append(frontend(rgb))
        img = bgr
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)))
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        jpegs.append(buf.tobytes() if ok else b"")

    fm = np.stack(fmaps).astype(np.float32)
    gyro = gyro_full[:n]
    cfg = CancelConfig(fps=fps, robust=True, pred=pred, source=seq)
    if "f_map" in meta:
        cfg.f_map = float(meta["f_map"])
    canc = FeatureMapCancel(cfg)
    z, sur = rollout(canc, fm, gyro)
    score = maxcell_score(z).astype(np.float64)
    t = np.arange(n, dtype=np.float64) / fps
    sd = np.sqrt(np.asarray(canc.var)) + 1e-3
    raw_amp = np.abs(z * sd).reshape(n, -1).mean(axis=1).astype(np.float32)

    accel = None
    if os.path.exists(f"{seq}/accel.npy"):
        a = np.load(f"{seq}/accel.npy").astype(np.float32)
        if len(a) >= n:
            accel = a[:n]
    labels = None
    if os.path.exists(f"{seq}/labels.npy"):
        lb = np.load(f"{seq}/labels.npy").astype(np.float32)
        if len(lb) >= n:
            labels = lb[:n]

    extra = {"fps": fps, "f_map": float(cfg.f_map), "pred": pred,
             "source_kind": "frames(online frontend)",
             "frontend": frontend_spec,
             "n_events": len(meta.get("events", [])),
             "gyro_sha1": sha1_of(f"{seq}/imu.csv")}
    return t, gyro, score, sur, raw_amp, labels, extra, jpegs


def infer_modalities(seq: str, frames_mode: str) -> dict:
    return {
        "has_featmaps": os.path.exists(f"{seq}/featmaps.npy"),
        "has_features": os.path.exists(f"{seq}/features.npy"),
        "has_labels": os.path.exists(f"{seq}/labels.npy"),
        "has_accel": os.path.exists(f"{seq}/accel.npy"),
        "has_frames_dir": os.path.isdir(f"{seq}/frames"),
        "frames_mode": frames_mode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/aoe_event",
                    help="包含 val/seq_* 的目录（或单个 seq 目录）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--frames", default="none",
                    choices=["jpeg", "none", "symlink"],
                    help="帧的存储方式（jpeg=可发布；none=只要 IMU+残差；"
                         "symlink=只记路径）")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--pred", default="rotation",
                    help="残差预测器；rotation 为实测较好者（rotwarp_aoe）")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT)
    ap.add_argument("--frontend", default="mobilenet_v3_small[:3]",
                    help="前端规格（仅 frames 路径需要；与 "
                         "AcquisitionLayer/已发布实验同一条链路）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 段（调试）")
    args = ap.parse_args()

    seqs = sorted(glob.glob(f"{args.source}/val/seq_*"))
    if not seqs:
        seqs = sorted(glob.glob(f"{args.source}/seq_*"))
    if not seqs and os.path.exists(f"{args.source}/events.json"):
        seqs = [args.source]
    if not seqs:
        print(f"在 {args.source} 下找不到 seq_* 目录", file=sys.stderr)
        return 1
    if args.limit:
        seqs = seqs[:args.limit]

    name = args.name or os.path.basename(args.out.rstrip("/"))
    w = DatasetWriter(args.out, name=name, source=args.source,
                      licenses="see upstream dataset (Open-AoE / recorded)",
                      description=(f"由 {args.source} 摄入的多模态数据集"
                                   f"（prepared-real，残差用 pred={args.pred} "
                                   f"在缓存特征图上重放）"))
    print(f"源: {args.source}  段数: {len(seqs)}  帧模式: {args.frames}")
    # 关于完整性巡检：**不跨段借参照**。
    #
    # 曾尝试用"前面各段的帧索引"当后续段的 healthy_ref，那是错的：
    # healthy_ref 是**本段帧的索引**，跨段索引没有任何意义（实测两者阈值
    # 几乎相同纯属巧合、并非机制正确）。参照只能来自同一段录制内部。
    #
    # 因此本脚本对每段只用其**自身**的均匀参照集做筛查。其已知代价是
    # 检不出稀疏故障（参照集必然包含故障帧）——这在 integrity.py 里已
    # 写明。要可靠检出，需要同一场景、同一设备、同一设置的**独立健康
    # 录制**，那属于采集侧的事，不是本脚本能凭空造出来的。
    for seq in seqs:
        sid = os.path.basename(seq)
        mods = infer_modalities(seq, args.frames)
        use_frames = (not mods["has_featmaps"]) and mods["has_frames_dir"]
        if not mods["has_featmaps"] and not use_frames:
            print(f"  跳过 {sid}（既无 featmaps.npy 也无 frames/）")
            continue

        frames = None
        if use_frames:
            # raw-capture 路径：真的有帧，走在线前端
            t, gyro, score, sur, raw_amp, labels, extra, frames = \
                load_from_frames(seq, args.max_frames, args.pred,
                                 args.frontend, args.jpeg_quality, args.max_side)
            accel = None
            if os.path.exists(f"{seq}/accel.npy"):
                a = np.load(f"{seq}/accel.npy").astype(np.float32)
                if len(a) >= len(t):
                    accel = a[:len(t)]
        else:
            t, gyro, score, sur, raw_amp, labels, extra = load_prepared(
                seq, args.max_frames, args.pred)
            accel = None
            if mods["has_accel"]:
                a = np.load(f"{seq}/accel.npy").astype(np.float32)
                accel = a[:len(t)]
                if len(accel) < len(t):
                    accel = None   # 长度不符按不可用处理（对齐契约优先）
            if args.frames == "jpeg" and mods["has_frames_dir"]:
                frames = load_frames_jpeg(seq, len(t), args.jpeg_quality,
                                          args.max_side)
            elif args.frames == "symlink":
                paths = sorted(glob.glob(f"{seq}/frames/*.png"))[:len(t)]
                if paths and len(paths) == len(t):
                    frames = np.array(paths, dtype=object)

        if frames is not None and args.frames == "jpeg" \
                and any(len(b) == 0 for b in frames):
            print(f"  警告 {sid}: 部分帧读取失败（已置空）", file=sys.stderr)

        sec = build_section(sid, t=t, gyro=gyro, score=score, surprise=sur,
                            keep=None, fps=float(extra["fps"]),
                            raw_residual=raw_amp, accel=accel, residual=None,
                            labels=labels, healthy_ref=None,
                            meta={**extra, "modalities_source": mods},
                            frames_jpg=(list(frames) if frames is not None
                                        and args.frames != "symlink" else None))
        w.add(sec)
        kf = float(np.mean(sec.keep > 0))
        print(f"  {sid}: n={sec.n_frames} keep={kf:.3f} "
              f"mods={','.join(k for k, v in sec.modalities().items() if v)} "
              f"verdict={sec.meta['integrity']['verdict']}")

    man = w.finalize()
    print(f"\n数据集: {args.out}")
    print(f"  段数={man['n_sections']} 帧数={man['n_frames']:,} "
          f"体积={man['total_bytes'] / 1e6:.1f} MB "
          f"({man['bytes_per_frame']:.0f} B/帧)")
    print(f"  模态覆盖: {man['modality_coverage']}")
    print(f"  完整性结论: {man['integrity_verdicts']}")
    print(f"  weight 状态: {man['weight']['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

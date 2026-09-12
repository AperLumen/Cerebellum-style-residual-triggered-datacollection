"""轴消融的**窗口稳定性**检验：排序会不会随评估窗口变。

## 动机

`runs/negative_followups.json`（既有 `scripts/analyze_negatives.py`）报 all-three
旋转单应在后半窗口 +4.65%。`scripts/oracle_aoe_shift.py` 又发现同一实现在全序列
与后半窗口的收益差一倍以上。这两个事实合起来提出一个必须回答的问题：

> Table~\\ref{tab:axis} 的**轴序**（roll > yaw+pitch > ...）是数据集的属性，
> 还是评估窗口的属性？

如果是窗口的属性，那论文里"the failure is the shadow of roll"这条结论就只能
限定在"后半窗口"下成立——必须显式声明。

## 做法

同一份数据、同一实现、同一符号标定流程（前半拟合 / 后半评估），只换**评估
窗口**，跑完整的 7 模型消融：

* 窗口 H（held-out）：前半标定符号 -> 只评**后半**。这是既有脚本的口径。
* 窗口 W（whole）：同一符号 -> 评**全序列**。这是 `run_rotwarp_aoe.py` 的
  默认口径（-60 帧 warmup）。

两组都报：每模型的均值增益、逐段更好的段数、以及**轴序是否一致**。

输出：runs/axis_window_stability.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from cerebellum_cancel.io import load_imu_csv                  # noqa: E402
from analyze_negatives import (mean_abs_err, rot_homography,    # noqa: E402
                               shipped_shift, FPS)

# gy 通道序 [pitch, yaw, roll]；与 analyze_negatives 的轴索引一致
AXIS_MODELS: dict[str, tuple[int, ...] | None | str] = {
    "naive (no warp)": None,
    "shipped 2D shift": "shift",
    "yaw only": (1,),
    "pitch only": (0,),
    "roll only": (2,),
    "yaw+pitch": (0, 1),
    "all three": (0, 1, 2),
}
WARMUP = 60


def warp_channels(Fp: np.ndarray, H: np.ndarray) -> np.ndarray:
    """与 analyze_negatives.warp_channels 同实现（含 WARP_INVERSE_MAP）。

    这里的 flag 与符号是**联合**约定（见 MEMORY_FORMAT 8.7）：漏掉 flag 会
    把预测反向 warp，效果从降残差变增残差，且只看残差数值发现不了。
    """
    ch, h, w = Fp.shape
    return np.stack([cv2.warpPerspective(Fp[c], H, (w, h),
                                         flags=cv2.INTER_LINEAR
                                         | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REFLECT)
                     for c in range(ch)])


def dedup_key(seq: str) -> tuple:
    """判重：实测 aoe_event/seq_007 与 aoe_motion/seq_001 逐位相同。"""
    fm = f"{seq}/featmaps.npy"
    return (os.path.getsize(fm),
            hashlib.md5(open(f"{seq}/imu.csv", "rb").read()).hexdigest())


def calibrate(F, gy, f_map, idx: np.ndarray, cal_idx: np.ndarray,
              h: int, w: int) -> tuple[float, float, float]:
    """前半拟合符号（坐标下降 2^3），与既有脚本同流程。"""
    best, best_err = (1.0, 1.0, 1.0), np.inf
    for sg in itertools.product((1.0, -1.0), repeat=3):
        s = np.array(sg, dtype=np.float32)
        e = 0.0
        for t in cal_idx:
            Ft = np.asarray(F[idx[t]], dtype=np.float32)
            Fp = np.asarray(F[idx[t - 1]], dtype=np.float32)
            rv = np.radians(gy[idx[t]] * s) / FPS
            e += mean_abs_err(warp_channels(Fp, rot_homography(rv, f_map, h, w)),
                              Ft)
        if e < best_err:
            best, best_err = (float(sg[0]), float(sg[1]), float(sg[2])), e
    return best


def eval_models(F, gy, f_map, idx: np.ndarray, ev_idx: np.ndarray,
                signs: tuple, h: int, w: int) -> dict[str, dict[str, float]]:
    """在 ev_idx 上评 7 个模型，**同时**返回内区与全帧两个口径。

    为什么要两个口径：内区（``mean_abs_err``，BORDER=2）排除了边界格，而
    warp 在边界必须凭空生成内容——所以全帧口径系统性地对 warp 不利。实测
    同一份数据同一实现：内区 11/11 段更好，全帧只有 9/12。**报收益必须连
    口径一起给**，否则读者会以为两个数字矛盾。
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for t in ev_idx:
        Ft = np.asarray(F[idx[t]], dtype=np.float32)
        Fp = np.asarray(F[idx[t - 1]], dtype=np.float32)
        g = gy[idx[t]]
        for name, axes in AXIS_MODELS.items():
            if axes is None:
                pred = Fp
            elif axes == "shift":
                pred = shipped_shift(Fp, g, f_map)
            else:
                keep = np.zeros(3, dtype=np.float32)
                for a in axes:
                    keep[a] = g[a]
                rv = np.radians(keep * np.asarray(signs)) / FPS
                pred = warp_channels(Fp, rot_homography(rv, f_map, h, w))
            d = acc.setdefault(name, {"interior": [], "full": []})
            d["interior"].append(mean_abs_err(pred, Ft))
            d["full"].append(float(np.abs(Ft - pred).mean()))
    return {k: {m: float(np.mean(v)) for m, v in d.items()}
            for k, d in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_motion,data/aoe_event")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="runs/axis_window_stability.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = []
    for st in args.sets.split(","):
        paths += sorted(glob.glob(f"{st.strip()}/val/seq_*"))
    seqs, seen = [], set()
    for s in paths:
        k = dedup_key(s)
        if k in seen:
            print(f"  [跳过重复段] {s}")
            continue
        seen.add(k)
        seqs.append(s)
    print(f"唯一段数: {len(seqs)}")

    if os.path.exists(args.out) and not args.force:
        old = json.load(open(args.out))
        if len(old.get("seqs", [])) > len(seqs):
            raise SystemExit(f"拒绝覆盖 {args.out}：现有段数更多。加 --force。")

    rows = []
    for s in seqs:
        if not os.path.exists(f"{s}/featmaps.npy"):
            continue
        F = np.load(f"{s}/featmaps.npy", mmap_mode="r")
        gy = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                      dtype=np.float32)
        f_map = float(json.load(open(f"{s}/events.json"))["f_map"])
        n = min(len(F), len(gy), args.n)
        idx = np.linspace(0, n - 1, n).astype(int)
        _, h, w = np.asarray(F[0]).shape
        half = n // 2
        cal_idx = np.linspace(1, half - 1, min(half - 1, 60)).astype(int)
        ev_half = np.linspace(half, n - 1, n - half).astype(int)
        ev_whole = idx[WARMUP:]

        signs = calibrate(F, gy, f_map, idx, cal_idx, h, w)
        e_half = eval_models(F, gy, f_map, idx, ev_half, signs, h, w)
        e_whole = eval_models(F, gy, f_map, idx, ev_whole, signs, h, w)
        row: dict = {"seq": s, "n": int(n), "f_map": f_map,
                     "signs": list(signs)}
        for wname, e in (("half", e_half), ("whole", e_whole)):
            for mname in ("interior", "full"):
                base = e["naive (no warp)"][mname]
                row[f"{wname}_{mname}"] = {
                    k: 1.0 - v[mname] / base for k, v in e.items()}
        rows.append(row)
        print(f"  {s.split('/')[-1]:9s}[{s.split('/')[1][4:]:6s}] "
              f"n={n:5d}  all-three: 后半 {row['half_interior']['all three'] * 100:+6.2f}%"
              f"  全序列 {row['whole_interior']['all three'] * 100:+6.2f}%"
              f"  | 全帧 后半 {row['half_full']['all three'] * 100:+6.2f}%"
              f"  全序列 {row['whole_full']['all three'] * 100:+6.2f}%")

    print(f"\n=== 7 模型 × 2 窗口 × 2 口径（{len(rows)} 段）===")
    summary = {}
    for wname, wlab in (("half", "后半（held-out）"),
                        ("whole", "全序列")):
        for mname, mlab in (("interior", "内区 BORDER=2"),
                            ("full", "全帧")):
            key = f"{wname}_{mname}"
            print(f"\n  窗口 = {wlab}   口径 = {mlab}")
            print(f"    {'模型':<20}{'均值增益':>10}{'更好段数':>10}")
            order = []
            for k in AXIS_MODELS:
                v = np.array([r[key][k] for r in rows])
                summary.setdefault(key, {})[k] = {
                    "mean_gain": float(v.mean()),
                    "better": int((v > 0).sum()), "n": len(v)}
                order.append((k, float(v.mean())))
                print(f"    {k:<20}{v.mean() * 100:>9.2f}%"
                      f"{(v > 0).sum():>7}/{len(v)}")
            rot = [(k, g_) for k, g_ in order
                   if k not in ("naive (no warp)", "shipped 2D shift")]
            rot.sort(key=lambda kv: -kv[1])
            summary[key]["rotation_order"] = [k for k, _ in rot]
            print(f"    旋转模型轴序: {' > '.join(k for k, _ in rot)}")

    orders = {k: tuple(v["rotation_order"]) for k, v in summary.items()}
    same = len(set(orders.values())) == 1
    interior_same = orders["half_interior"] == orders["whole_interior"]
    print(f"\n  轴序在四个组合间完全一致? {'是' if same else '否'}")
    for k, o in orders.items():
        print(f"    {k:<18} {' > '.join(x.replace(' only', '') for x in o)}")

    # ---- 按运动档位分组：论文 Table tab:axis 的口径是分档的 ----
    # 论文报的是 walking level（gyro p95 >= 60 deg/s, n=7）与 desk/mid
    # (n=5) 两组。这里复现该分组，并**同时**给出两个度量口径——因为论文那句
    # "roll alone > yaw+pitch combined" 是否成立于口径无关，必须直接看。
    p95 = {}
    for s in seqs:
        g = np.array([x.gyro for x in load_imu_csv(f"{s}/imu.csv")],
                     dtype=np.float64)
        p95[s] = float(np.percentile(np.linalg.norm(g, axis=1), 95))
    groups = {
        "walking (p95>=60)": [r for r in rows if p95[r["seq"]] >= 60.0],
        "desk/mid (p95<60)": [r for r in rows if p95[r["seq"]] < 60.0],
    }
    motion_split: dict = {}
    print("\n=== 按运动档位分组（论文 Table tab:axis 的口径）===")
    for gname, grp in groups.items():
        if not grp:
            continue
        print(f"\n  {gname}  n={len(grp)}")
        motion_split[gname] = {"n": len(grp)}
        for key in ("half_interior", "half_full"):
            mlab = "内区" if key.endswith("interior") else "全帧"
            line = []
            for m in ("roll only", "yaw+pitch", "all three", "shipped 2D shift"):
                v = np.array([r[key][m] for r in grp])
                motion_split[gname].setdefault(key, {})[m] = {
                    "mean_gain": float(v.mean()),
                    "better": int((v > 0).sum()), "n": len(v)}
                line.append(f"{m} {v.mean() * 100:+.2f}%")
            roll_m = motion_split[gname][key]["roll only"]["mean_gain"]
            yp_m = motion_split[gname][key]["yaw+pitch"]["mean_gain"]
            verdict = "roll 更优" if roll_m > yp_m else "yaw+pitch 更优"
            motion_split[gname][key]["roll_beats_yawpitch"] = bool(roll_m > yp_m)
            print(f"    {mlab:<4} " + "  ".join(line) + f"   -> {verdict}")

    roll_claims = [motion_split[g][k]["roll_beats_yawpitch"]
                   for g in motion_split if g.startswith("walking")
                   for k in ("half_interior", "half_full")]
    print(f"\n  『rolling 单轴 > yaw+pitch』在 walking 档的两个口径下都成立? "
          f"{'是' if roll_claims and all(roll_claims) else '否'}")

    report = {
        "sets": args.sets, "n_frames_per_seq": args.n, "warmup": WARMUP,
        "metric": "同时报内区（BORDER=2，与管线读数一致）与全帧两个口径",
        "protocol": "符号在前半拟合（坐标下降 2^3），两个窗口都用同一组符号；"
                    "只变评估窗口与度量口径，不变实现与标定。",
        "note": "回答三件事：(1) 轴序是数据的属性还是窗口的属性；"
                "(2) 报收益时口径的影响有多大（内区 vs 全帧）；"
                "(3) 论文 tab:axis 那句『roll alone > yaw+pitch combined』"
                "是否口径无关。",
        "signs_identical_across_seqs": all(
            r["signs"] == rows[0]["signs"] for r in rows) if rows else None,
        "gyro_p95_per_seq": p95,
        "axis_order_stable": bool(same),
        "axis_order_stable_interior": bool(interior_same),
        "motion_split": motion_split,
        "summary": summary,
        "seqs": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

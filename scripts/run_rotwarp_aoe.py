"""rotwarp 交叉验证：在 Open-AoE 高运动/事件段上重测 `pred="rotation"`。

## 问题

真机 IMU 段（MEMORY_FORMAT §8）给出一个正结果：同一份真 gyro 下，tan 近似
常数平移 1.31x（帮倒忙）、完整旋转矩阵 0.88x（降残差）。这个脚本要在**与已
发布负结果相同的数据**（Open-AoE）上复核，判断是否该改默认值。

## 实测结论（2026-09-11，第二轮修正后）

### 与既有脚本的分歧已解决：根因是**评估窗口**，不是实现

`scripts/analyze_negatives.py`（既有）在同一份数据上报 all-three **+4.65%**、
12/12 段更好；本脚本最初报 **+2.06%**、9/12。逐位对齐实现后（逐段最大偏差
0.32 个百分点，按它的口径可复现 +4.63% vs 它的 +4.65%），把差异缩到一处：

| 口径（同一实现、同一符号） | 均值增益 | 更好段数 | p |
| -------------------------- | -------- | -------- | - |
| A 全序列前 600 帧 | +2.06% | 9/12 | 0.146 |
| B 前 1500 帧的**后半** | **+4.55%** | **11/12** | **0.006** |
| C B + 逐段标定符号 | +4.63% | 11/12 | 0.006 |

**A→B（换窗口）差 2.5pp 且把 p 从 0.146 拉到 0.006；B→C（标符号）只差
0.08pp。** 即**显著性完全由"在哪些帧上评"决定**。下表的读数就是 A 口径。

### 判据必须分两层

| 层次 | 读数（A 口径，n=600，12 段含一重复） | 符号检验 |
| ---- | ---- | -------- |
| 相对：rotation vs **shift** | **11/12 段更好**，均值 0.994 vs 1.049 | **p=0.006** ✓ |
| 绝对：rotation vs **naive** | 6/12 段更好，均值 0.994 | p=1.000 ✗ |

**含义**：rotation warp 相对旧模型（tan 近似常数平移）的优势在 AoE 上**成立
且显著**；但它的**绝对收益**取决于评估窗口——A 口径不显著（p=1.000），
B 口径显著（+4.55%，p=0.006）。**引用时连窗口一起说。**

### 收益是真的，但只兑现了一小部分

`scripts/control_aoe_gain.py` 排除了"只是插值平滑伪影"的假设：把 gyro 时间轴
打乱得 −6.26%、所有轴符号取反得 −6.53%，而正向 **+4.46%**——**正/反不对称
（差 11pp）本身就证明方向系统性正确**；纯高斯模糊（不用 IMU）最好只有
+0.06%。

`scripts/oracle_aoe_shift.py`（逐帧穷举最优平移，不用任何估计器）给出了上界：

| 时间窗 | 本实现 | oracle 上界 | corr(gyro, oracle) yaw |
| ------ | ------ | ----------- | ---------------------- |
| t 1–500 | +1.56% | +4.79% | +0.454 |
| t 500–1000 | +2.65% | +5.22% | +0.432 |
| t 1000–1500 | +1.43% | +6.20% | +0.447 |

**gyro 确有解释力**（yaw 轴相关 +0.43~+0.45，三窗一致）——所以下面那句"位移
太小所以没用"是**被推翻的旧说法**，瓶颈是**模型缺平移项**，不是数据。

> 旧说法（保留作记录，**已被 oracle 推翻**）："AoE 每帧网格位移 p95 只有
> 0.596 格（真机 2.790 格，差 4.7 倍）；位移小于一格时旋转模型的优势落在
> 噪声地板上。" —— `grid_shift_p95` 是两个模型的**差别有多大**的指标，不是
> **收益上限**的指标；拿它当唯一判据会把人领向"数据不行"的错结论。

## 一个必须记录的历史缺陷

三个真实 bug（详见 MEMORY_FORMAT §8.7）：

1. `warp_by_rotation` 漏了 `cv2.WARP_INVERSE_MAP`——预测被**反向** warp，
   效果从降残差变增残差，且**只看残差数值发现不了**（本轮被它误导两轮）。
2. **符号与 flag 是联合约定**（`(-s, 有 flag) ≡ (+s, 无 flag)`），补 flag 后
   两个数据源的符号都必须整体取反。
3. `prep_aoe.calibrate_signs` 强制 ``corr(-f·tan(g), dx) > 0``，而
   `prep_real_imu.fit_calibration` 只取 ``|corr|`` 最大、不约束符号——两者
   目标相反，导致存储的 gyro 方向不同。

配合同样**逐数据集不同**（真机 lag=1、AoE lag=0），不能全局取值。

教训：**用相关系数标定符号时，必须有一个绝对参照**；**符号、flag、配对
三者是联合约定，改一处必须同步其余**。

用法::

  PYTHONPATH=. .venv/bin/python scripts/run_rotwarp_aoe.py --n 600
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

from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               rollout)
from cerebellum_cancel.io import load_imu_csv
from cerebellum_cancel.rotwarp import signs_for, lag_for

MODE_NAMES = ("naive", "shift", "rotation", "rotation_no_roll")


def _grid_shift_p95(seq: str) -> float:
    """该段每帧位移的 p95，单位是**特征图格**（28 网格）。

    位移 = 旋转项 + roll 的切向项：
      |d| = hypot(f_map·tan(ω_yaw·dt), f_map·tan(ω_pitch·dt), r·ω_roll·dt)
    其中 r = √2·W/2 是网格角落到中心的距离（roll 的最大切向半径）。

    **它衡量的是"两个模型的预测差多少"，不是"收益上限有多高"**——曾经拿它
    当唯一判据，得出"AoE 位移太小（0.596 格 vs 真机 2.790 格）所以补不了"，
    后来被逐帧穷举的 oracle 推翻：AoE 上 oracle 能到 +4.8~6.2%，gyro 实现
    却只有 +1.4~2.7%。真正的原因是**模型缺平移项**，不是数据不行。
    见 `scripts/oracle_aoe_shift.py` 与模块 docstring。
    """
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    f_map = float(meta.get("f_map", 24.25))
    g = np.array([x.gyro for x in load_imu_csv(f"{seq}/imu.csv")],
                 dtype=np.float64)
    dt = 1.0 / fps
    dx = f_map * np.tan(np.radians(g[:, 1]) * dt)
    dy = f_map * np.tan(np.radians(g[:, 0]) * dt)
    r = np.sqrt(2.0) * 28.0 / 2.0
    dr = r * np.radians(g[:, 2]) * dt
    return float(np.percentile(np.hypot(np.hypot(dx, dy), dr), 95))


def _sign_test_p(wins: int, n: int) -> float:
    """配对符号检验的双侧 p 值（精确二项，p=0.5）。

    用来区分"真的更好"与"段数少、碰巧赢多"。n=12 时 11 胜给出 p≈0.006，
    7 胜给出 p≈0.77——所以下面两个判据必须分开报。
    """
    if n == 0:
        return 1.0
    from math import comb
    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _verdict(shift_ratio: float, rot_ratio: float, paired: int,
             rot_wins_naive: int, n: int) -> str:
    """分两层判据——**相对优势**与**绝对收益**必须分开报。

    这个区分是本轮实测逼出来的：在 AoE 上 rotation 赢 shift 赢得很干净
    （11/12，符号检验 p≈0.006），但它**并没有**稳定赢过不补偿——而且这个
    "赢不赢"**高度依赖评估窗口**（全序列前 600 帧 p=0.146；前 1500 帧的
    后半 p=0.006）。把两者混成一句"复现/未复现"会掩盖真正重要的那条：
    **引用绝对收益时，必须连评估窗口一起给。**

    第一层（相对）：rotation vs shift，配对符号检验。
    第二层（绝对）：rotation vs naive，同样检验；这层对窗口敏感，
      报告时把 A/B/C 三种口径的差别一并写出（见模块 docstring）。
    """
    if n == 0:
        return "无序列"
    p_rel = _sign_test_p(paired, n)
    p_abs = _sign_test_p(rot_wins_naive, n)
    rel = (f"rotation 优于 shift {paired}/{n} 段（符号检验 p={p_rel:.3f}）"
           + ("，相对优势成立" if p_rel < 0.05 else "，相对优势不显著"))
    if p_abs < 0.05 and rot_ratio < 1.0:
        ab = (f"且优于不补偿 {rot_wins_naive}/{n} 段（p={p_abs:.3f}），"
              f"均值 {rot_ratio:.3f}——该数据上值得开 rotation")
    elif rot_ratio < 1.0:
        ab = (f"但优于不补偿只有 {rot_wins_naive}/{n} 段（p={p_abs:.3f}，"
              f"均值 {rot_ratio:.3f}）——**本口径下绝对收益不显著**。注意这"
              "个显著性高度依赖**评估窗口**：全序列前 600 帧给 p=0.146，"
              "前 1500 帧的后半给 p=0.006（见模块 docstring 的 A/B/C 表）。"
              "引用时必须连窗口一起给。另外 oracle 上界（约 +5~6%）远高于"
              "本实现（约 +1.5~2.7%），瓶颈是**模型缺平移项**，不是数据。")
    else:
        ab = (f"且不补偿都赢不了（{rot_wins_naive}/{n} 段，均值 "
              f"{rot_ratio:.3f}）")
    return f"{rel}；{ab}"


def load_seq(seq: str, limit: int | None):
    meta = json.load(open(f"{seq}/events.json"))
    fps = float(meta.get("fps", 30.0))
    f_map = float(meta.get("f_map", 24.25))
    fm = np.load(f"{seq}/featmaps.npy").astype(np.float32)
    gyro = np.array([g.gyro for g in load_imu_csv(f"{seq}/imu.csv")],
                    dtype=np.float32)
    n = min(len(fm), len(gyro))
    if limit:
        n = min(n, limit)
    return fm[:n], gyro[:n], fps, f_map, meta, n


def run_mode(fm, gyro, fps, f_map, mode: str,
             yaw_sign: float = 1.0, pitch_sign: float = 1.0,
             roll_sign: float = 1.0, gyro_lag: int | None = None
             ) -> tuple[np.ndarray, float]:
    """返回 (逐帧全帧原始残差, ms/帧)。

    ``yaw_sign``/``pitch_sign``/``roll_sign``/``gyro_lag`` 只对 ``rotation``
    模式生效——它们把存储通道映射到旋转轴并决定取哪一帧读数。各数据源的
    约定相反（rotwarp.DATASET_CONVENTIONS）：AoE 为 (-1,+1,+1) + lag 0，
    真机 IMU 为 (+1,+1,+1) + lag 1。不指定时按调用方传入值原样使用。
    """
    if mode == "naive":
        cfg = CancelConfig(fps=fps, f_map=f_map, robust=True, use_warp=False)
    elif mode == "shift":
        cfg = CancelConfig(fps=fps, f_map=f_map, robust=True, pred="shift")
    else:
        cfg = CancelConfig(fps=fps, f_map=f_map, robust=True, pred="rotation",
                           yaw_sign=yaw_sign, pitch_sign=pitch_sign,
                           roll_sign=roll_sign,
                           gyro_lag=0 if gyro_lag is None else gyro_lag)
    canc = FeatureMapCancel(cfg)

    n = len(fm)
    errs = np.zeros(n, dtype=np.float64)
    t0 = time.perf_counter()
    if mode == "rotation_no_roll":
        # 消融：自己驱动 step 的预测，把 roll 置零
        from cerebellum_cancel.featmap_cancel import shift_map  # noqa: F401
        from cerebellum_cancel.rotwarp import warp_by_rotation
        dt = 1.0 / fps
        for t in range(n):
            F = fm[t]
            if canc.F_prev is None:
                pred = F
            else:
                lag = 0 if gyro_lag is None else gyro_lag
                if lag == 0:
                    g = gyro[t]
                else:
                    g = (canc.gyro_prev if canc.gyro_prev is not None
                         else gyro[t])
                pred = warp_by_rotation(canc.F_prev, g, dt, f_map,
                                        use_roll=False,
                                        yaw_sign=yaw_sign,
                                        pitch_sign=pitch_sign,
                                        roll_sign=roll_sign)
            # 复用 step 的统计/触发逻辑：喂一个"预测已算好"的等价输入
            out = _step_with_pred(canc, F, gyro[t], pred)
            errs[t] = _raw_err(canc, out["z"])
    else:
        for t in range(n):
            out = canc.step(fm[t], gyro[t])
            errs[t] = _raw_err(canc, out["z"])
    ms = (time.perf_counter() - t0) / n * 1e3
    return errs, ms


def _step_with_pred(canc: FeatureMapCancel, F: np.ndarray, gyro: np.ndarray,
                    pred: np.ndarray) -> dict:
    """与 FeatureMapCancel.step 同逻辑，但用外部给定的 pred（消融用）。"""
    F = F.astype(np.float32)
    err = F - pred
    if canc.mu is None:
        canc.mu = np.zeros_like(err)
        canc.var = np.ones_like(err)
    assert canc.mu is not None and canc.var is not None
    sd = np.sqrt(canc.var) + 1e-3
    z = err / sd
    b = canc.cfg.border
    surprise = float(np.abs(z[:, b:-b, b:-b]).max())
    canc.n += 1
    lr = canc.cfg.std_ema
    d = np.clip(err - canc.mu, -canc.cfg.winsor * sd, canc.cfg.winsor * sd)
    canc.mu += lr * d
    canc.var += lr * (d * d - canc.var)
    canc.F_prev = F
    canc.gyro_prev = np.asarray(gyro, dtype=np.float64).copy()
    return {"z": z, "surprise": surprise, "learned": True}


def _raw_err(canc: FeatureMapCancel, z: np.ndarray) -> float:
    assert canc.var is not None
    sd = np.sqrt(canc.var) + 1e-3
    return float(np.abs(z * sd).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="data/aoe_motion,data/aoe_event")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--skip-first", type=int, default=60,
                    help="跳过统计收敛期（warmup）")
    ap.add_argument("--out", default="runs/rotwarp_aoe.json")
    ap.add_argument("--force", action="store_true",
                    help="允许小规模运行覆盖更大的已有结果（默认拒绝，"
                         "防止冒烟测试盖掉正式 json）")
    ap.add_argument("--no-window-check", action="store_true",
                    help="跳过评估窗口敏感性复算（省掉每段两次额外 rollout）")
    args = ap.parse_args()

    seqs: list[str] = []
    for d in args.sets.split(","):
        seqs.extend(sorted(glob.glob(f"{d.strip()}/val/seq_*")))
    if not seqs:
        raise SystemExit("找不到序列")

    # 覆盖保护（放在计算之前，避免白跑一遍）：小规模运行不许静默盖掉更大
    # 的正式结果。实测踩到过：一次小规模冒烟把正式 json 覆盖了。
    if os.path.exists(args.out) and not args.force:
        try:
            old = json.load(open(args.out))
            old_n = len(old.get("seqs", []))
        except (OSError, ValueError):
            old_n = 0
        if len(seqs) < old_n:
            raise SystemExit(
                f"拒绝覆盖 {args.out}：现有结果更大"
                f"（{old_n} 段 > 本次 {len(seqs)} 段）。"
                "确要覆盖请加 --force，或换个 --out 。")

    t0 = time.time()
    report: dict = {
        "sets": args.sets, "n_frames_per_seq": args.n,
        "metric": "全帧原始残差均值（方法无关；跳过 warmup 帧）",
        "seqs": [], "modes": list(MODE_NAMES),
        "sign_note": "AoE 存储的 yaw 方向与旋转轴相反：rotation 模式按 "
                     "rotwarp.DATASET_CONVENTIONS 传入 signs=(-1,+1,+1) "
                     "且 lag=0（prep_aoe 存的就是 t-1 -> t 转角）。"
                     "warp 必须带 WARP_INVERSE_MAP，符号与它配套。"
                     "各段最优符号记在 per-seq 的 best_signs 里。",
    }
    rows: list[dict] = []
    for s in seqs:
        fm, gyro, fps, f_map, meta, n = load_seq(s, args.n)
        name = f"{s.split('/')[1]}/{s.split('/')[-1]}"
        ys, ps, rs = signs_for(s)
        lag = lag_for(s)
        entry: dict = {"seq": name, "n": n, "f_map": f_map, "fps": fps,
                       "yaw_sign": ys, "pitch_sign": ps, "roll_sign": rs,
                       "gyro_lag": lag,
                       "roll_p95": float(np.percentile(np.abs(gyro[:, 2]), 95)),
                       "yaw_p95": float(np.percentile(np.abs(gyro[:, 1]), 95)),
                       "runs": {}}
        sl = slice(min(args.skip_first, n // 4), None)
        for mode in MODE_NAMES:
            errs, ms = run_mode(fm, gyro, fps, f_map, mode,
                                yaw_sign=ys, pitch_sign=ps, roll_sign=rs,
                                gyro_lag=lag)
            entry["runs"][mode] = {"full_frame": float(errs[sl].mean()),
                                   "ms_per_frame": ms}
        base = entry["runs"]["naive"]["full_frame"]
        for mode in MODE_NAMES:
            e = entry["runs"][mode]
            e["vs_naive"] = e["full_frame"] / base if base > 0 else float("nan")

        # 为"评估窗口敏感性"检查保留逐帧残差（rotation 与 naive 各一份）。
        # 只存两个数组，成本可忽略；它让后面的后半窗口复算不必整段重跑。
        if not args.no_window_check:
            rot_e, _ = run_mode(fm, gyro, fps, f_map, "rotation",
                                yaw_sign=ys, pitch_sign=ps, roll_sign=rs,
                                gyro_lag=lag)
            nai_e, _ = run_mode(fm, gyro, fps, f_map, "naive")
            entry["resid"] = (rot_e.astype(np.float32).tolist(),
                              nai_e.astype(np.float32).tolist())

        # 符号扫描诊断：rotation 模式在 8 种 (yaw,pitch,roll) 组合下哪个最好。
        # 这是"存储约定是否被正确解读"的直接证据。
        sweep: dict = {}
        for sy in (1.0, -1.0):
            for sp in (1.0, -1.0):
                for sr in (1.0, -1.0):
                    e, _ = run_mode(fm, gyro, fps, f_map, "rotation",
                                    yaw_sign=sy, pitch_sign=sp, roll_sign=sr,
                                    gyro_lag=lag)
                    sweep[f"yaw{sy:+.0f}_pitch{sp:+.0f}_roll{sr:+.0f}"] = \
                        float(e[sl].mean()) / base
        entry["sign_sweep"] = sweep
        entry["best_signs"] = min(sweep, key=lambda k: sweep[k])
        report["seqs"].append(entry)
        rows.append(entry)
        print(f"  [{name}] roll_p95={entry['roll_p95']:5.1f} lag={lag} "
              f"naive={base:.4f} "
              f"shift={entry['runs']['shift']['vs_naive']:.3f} "
              f"rot={entry['runs']['rotation']['vs_naive']:.3f} "
              f"rot_noRoll={entry['runs']['rotation_no_roll']['vs_naive']:.3f} "
              f"| best={entry['best_signs']}")

    def agg(mode: str, key: str = "vs_naive") -> float:
        v = [q["runs"][mode][key] for q in rows]
        return float(np.mean(v))

    # 逐段胜负
    wins = {m: sum(1 for q in rows if q["runs"][m]["vs_naive"] < 1.0)
            for m in MODE_NAMES}
    paired = sum(1 for q in rows
                 if q["runs"]["rotation"]["vs_naive"]
                 < q["runs"]["shift"]["vs_naive"])
    # 符号扫描：最优组合分布（诊断存储约定是否与模型期望一致），
    # 以及"各自取最优符号"时 rotation 的上界——用来区分"模型不行"与
    # "模型没问题但输入符号被误读"。
    from collections import Counter
    best_cnt = Counter(q["best_signs"] for q in rows)
    best_rot = float(np.mean([min(q["sign_sweep"].values()) for q in rows])) \
        if rows else float("nan")

    # 机制量：每帧网格位移 p95。AoE（0.485 格）与真机段（2.790 格）差 5.8 倍，
    # 这是"为什么 AoE 上收益小得多"的定量解释，不是判据本身。
    grid_motion = float(np.mean([_grid_shift_p95(s) for s in seqs]))

    report["summary"] = {
        "n_seqs": len(rows),
        "mean_vs_naive": {m: agg(m) for m in MODE_NAMES},
        "wins": wins,
        "rotation_beats_shift_paired": f"{paired}/{len(rows)}",
        "best_signs_distribution": dict(best_cnt),
        "rotation_best_signs_mean": best_rot,
        "roll_p95_mean": float(np.mean([q["roll_p95"] for q in rows])),
        "grid_shift_p95_mean": grid_motion,
        # 两层判据的 p 值：相对（vs shift）与绝对（vs naive）。分开报，
        # 因为本轮实测里前者极显著、后者强烈依赖评估窗口。
        "p_relative_vs_shift": _sign_test_p(paired, len(rows)),
        "p_absolute_vs_naive": _sign_test_p(wins["rotation"], len(rows)),
        "wall_seconds": time.time() - t0,
    }
    sm = report["summary"]
    sm["verdict"] = _verdict(agg("shift"), agg("rotation"), paired,
                             wins["rotation"], len(rows))

    # 评估窗口敏感性（关键发现）：同一实现换窗口会让绝对收益的显著性从
    # 不显著变成显著（A: 全序列 -> p=0.146；B: 后半 -> p=0.006）。这里在
    # **本次实际跑的窗口**上再算一遍"后半"的子窗口，把差异直接量化出来，
    # 免得读者以为表格里的 p 值是数据集的固有属性。
    win_check = {}
    half_ratios: list[float] = []
    for q in rows:
        rr = q.get("resid")
        if rr is None:
            continue
        rot_e = np.asarray(rr[0], dtype=np.float64)
        nai_e = np.asarray(rr[1], dtype=np.float64)
        h = len(rot_e) // 2
        if h < 30:
            continue
        half_ratios.append(float(rot_e[h:].mean() / max(nai_e[h:].mean(),
                                                        1e-12)))
    if half_ratios:
        hr = np.array(half_ratios)
        win_check = {
            "note": "同一实现在**本次窗口的后半**上的 rotation/naive 比值；"
                    "与上面全窗口的读数对比即可看出窗口敏感性。",
            "second_half_mean_ratio": float(hr.mean()),
            "second_half_better": int((hr < 1).sum()),
            "second_half_n": len(hr),
            "second_half_p": _sign_test_p(int((hr < 1).sum()), len(hr)),
        }
        sm["window_sensitivity"] = win_check

    # 逐帧残差只用于上面的窗口复算，不入盘（会让 json 膨胀 12 段 × 数千帧）。
    for q in report["seqs"]:
        q.pop("resid", None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2)

    print(f"\n=== 汇总（{sm['n_seqs']} 段，roll p95 均值 "
          f"{sm['roll_p95_mean']:.1f} deg/s）===")
    print(f"{'方法':<18}{'相对naive均值':>14}{'逐段更好':>10}")
    for m in MODE_NAMES:
        print(f"{m:<18}{sm['mean_vs_naive'][m]:>14.3f}"
              f"{sm['wins'][m]:>7}/{sm['n_seqs']}")
    print(f"\nrotation 优于 shift 的段数: {sm['rotation_beats_shift_paired']}"
          f"  (p={sm['p_relative_vs_shift']:.3f})")
    print(f"rotation 优于 naive 的段数: {sm['wins']['rotation']}/{sm['n_seqs']}"
          f"  (p={sm['p_absolute_vs_naive']:.3f})")
    if win_check:
        print(f"  ↳ 同实现换到**后半窗口**后: "
              f"{win_check['second_half_better']}/{win_check['second_half_n']}"
              f" 段更好，均值 {win_check['second_half_mean_ratio']:.3f}，"
              f"p={win_check['second_half_p']:.3f}"
              f"  ← 显著性随窗口变化，引用请连窗口一起给")
    print(f"符号扫描最优组合分布: {sm['best_signs_distribution']}")
    print(f"每帧网格位移 p95: {sm['grid_shift_p95_mean']:.3f} 格"
          f"（真机段 2.790 格，比值 "
          f"{2.790 / max(sm['grid_shift_p95_mean'], 1e-9):.1f}x）")
    print(sm["verdict"])
    print(f"-> {args.out}（{sm['wall_seconds']:.1f}s）")


if __name__ == "__main__":
    main()

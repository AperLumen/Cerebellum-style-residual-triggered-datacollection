"""采集链路完整性巡检——把 GUIDE §2.3.10 的部分正结果产品化。

## 这是什么，不是什么

**是**：多模态数据集录制级的**完整性哨兵**。检出"采集链路本身被破坏"
（IMU 掉线、尖峰、时间戳错位），这类问题**不改变标注**，因此对 loss 类
数据清洗工具**结构性失明**——实测相关只有 +0.028，即它是**另一个传感器**，
不是"更好的 loss"。

**不是**：不是标注质检（语义错误原理上不可见），不是"找 1% 的金子"
（实测需要约 40% 覆盖才可靠检出，见下）。

## 两个必须保留的护栏

1. **`no_reference`**：故障覆盖 80–100% 时没有健康参照可供标定，
   必须主动报"无法得出结论"，而不是猜一个。这是**正确行为**。
2. **不可分性要如实说**："IMU 对这段无贡献"与"IMU 坏了"在观测上不可分。
   所以输出的是 `raw_ratio`（边际贡献）+ `verdict`，而不是一句"IMU 坏了"。

## 实测依据（GUIDE.md §2.3.10，本轮不重测）

| 项 | 读数 |
| --- | --- |
| 单帧尖峰 | lift **10.92×**（10/10 段）→ 可靠 |
| 整段 IMU 归零（信号有贡献时） | 1.49×（局部）→ 需覆盖约 40% |
| 覆盖 1.2% | 峰值仅健康 P99 的 **0.38×**（比最忙的正常帧更静） |
| 覆盖 80–100% | 无参照 → 正确报 `no_reference` |
| 与 loss 基线相关 | **+0.028**（正交） |

## 判据的两个上界（评估任何检出能力时必须一起给）

* **信号上界** = IMU 在该数据上的边际贡献。IMU 越没用，"IMU 坏了"越检不出。
* **参照上界** = 必须留健康段。没有健康段就无法标定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import (InsufficientDataError, NonFiniteError, check_finite_array,
                        check_min_frames)

# 检出阈值（沿用 GUIDE §2.3.10 的实测设定）
SPIKE_SIGMA = 6.0           # 单帧尖峰的 z 阈值（仅用于序列过短时的回退）
FROZEN_FRAMES = 120         # 冻结基准：标定前抽取的帧数
REF_MIN_FRACTION = 0.20     # 健康参照至少占总帧数这个比例，否则 no_reference
WARMUP_SKIP = 60            # 标定前跳过的起始帧数（统计收敛期）
REF_MAX_FRACTION = 0.10     # 参照窗最多占录制这么长
MIN_REF_FRAMES = 120        # 参照窗最少这么多帧，太少则分位不可靠

# 任何完整性结论所需的**最少帧数**。低于此值直接拒绝，理由：
#   * 参照集至少要能给出 99.9 分位（MIN_REF_FRAMES=120 时仍只有 ~1 个样本
#     落在该分位以上），帧数再少则"阈值"完全由噪声决定；
#   * "未检出异常"在样本不足时**没有统计意义**——报 `ok` 是误导，
#     所以必须显式失败而不是返回一个看似正常的结论。
# 这是工业化要求的一条：宁可拒绝出结论，不可给出无依据的结论。
MIN_ANALYSIS_FRAMES = 200

# **为什么参照窗不能取"开头的固定 120 帧"**（实测教训）：
# data/aoe_event/seq_000 的开头 120 帧恰好是一段安静期，其聚合 z 最大值
# 只有 2.68，而整段 p99 是 17.85、max 是 29.2。用这个不具代表性的窄窗
# 标定，阈值被压到 8.0，于是**干净段报出 725 个假尖峰**；而那些高 z 帧
# 落在标注区间内的比例（18.4%）与整段标注占比（20.9%）几乎相同——它们
# 既不是故障也不是事件，只是重尾。
# 结论：参照窗必须**覆盖录制的相当一部分**（REF_MAX_FRACTION），用**分位**
# 而不是最大值标定，且对非平稳录制要用**均匀抽取**而非单个连续块。

SPIKE_QUANTILE = 0.999      # 相对判据：超过**参照集**自身 99.9 分位
SPIKE_MIN_SEPARATION = 3.0  # 且要与阈值有量级分离，才算真故障

# **已实测的检出下限（必须如实告知，不可省略）**：参照集是录制的
# REF_MAX_FRACTION（10%）比例，所以当故障帧数远少于参照集长度时，故障帧
# 会**落在参照集内部**并被算作"正常"分布的一部分，从而把阈值抬到自己之上。
# 实测：4000 帧里注入 5 个 40× 尖峰，5 个全部被抽进参照集（394 帧），
# 阈值升到 53.2 而注入点的 z 只有 17.7 → **全部漏检**。
# 这与 GUIDE §2.3.10 §5b 的结论一致：检出能力由**故障覆盖率**
# 决定，不由算法灵敏度决定（那里测出需要约 40% 覆盖）。
# 因此本哨兵的定位是**录制级筛查**：它可靠地回答"这段录制整体是否健康"，
# **不**声称能逐帧揪出少量尖峰。要抓少数尖峰必须另配一个不含待查帧的
# 独立参照段（例如同一设备的上一段录制）。
SPIKE_MIN_COVERAGE_NOTE = (
    "检出下限：故障帧数需显著多于参照集长度（录制的 10%）；"
    "稀疏尖峰会被参照集吸收而漏检——这是覆盖率下限，不是调参问题")

# **为什么不用 σ 下限兜底**：实测这批真实数据逐帧聚合 z 的干净段 p99 就有
# 15–18σ（重尾）。此时"8σ 下限"不是严格条件而是**常见取值**——实测加了
# σ=8 下限后，四段干净数据分别报出 635/1089/391/569 个假尖峰，因为阈值
# 恒等于 8.0 而重尾高端普遍超过它。σ 这种"绝对尺度"在重尾数据上没有
# 意义，正确做法是**纯相对判据** + **量级分离**两道条件同时满足。


@dataclass
class IntegrityReport:
    """一段录制的完整性巡检结果。"""
    n_frames: int
    raw_ratio: float                  # IMU 边际贡献（信号上界）
    surprise_ratio: float             # 在线 surprise 的变化倍率
    frozen_z_ratio: float             # 冻结基准 z 的变化倍率（尖峰更灵敏）
    spike_frames: list[int] = field(default_factory=list)
    verdict: str = "ok"               # ok / suspect / no_reference
    note: str = ""
    threshold: float = float("nan")   # 实际使用的尖峰阈值（可复核）
    ref_frames: int = 0               # 参照集帧数

    def as_dict(self) -> dict:
        return {
            "n_frames": int(self.n_frames),
            "raw_ratio": float(self.raw_ratio),
            "surprise_ratio": float(self.surprise_ratio),
            "frozen_z_ratio": float(self.frozen_z_ratio),
            "spike_frames": [int(i) for i in self.spike_frames],
            "verdict": self.verdict,
            "note": self.note,
            "threshold": float(self.threshold),
            "ref_frames": int(self.ref_frames),
            "limits": SPIKE_MIN_COVERAGE_NOTE,
        }


def frozen_baseline_z(err: np.ndarray, n_frozen: int = FROZEN_FRAMES,
                      skip: int = WARMUP_SKIP, robust: bool = True,
                      agg: str = "mean") -> np.ndarray:
    """逐帧 z 分数，基准取 [skip, skip+n_frozen) 这些帧（冻结基准）。

    为什么用冻结基准而不是在线自适应 surprise：实测（LABEL_AUDIT §5）
    在线 sd 会把故障吸收掉——真机段信号有 1.49× 而在线只报 1.43×，
    冻结基准能到 **1.83×**。尖峰类瞬变要靠冻结基准才灵敏。

    Args:
        err: (T, ...) 逐帧残差（**未归一化**的幅度量）。不要传 surprise
            ——那已被在线统计归一化，故障会被吸收，且它本身重尾。
        n_frozen: 标定基准用的帧数。
        skip: 标定前跳过的帧数（默认 ``WARMUP_SKIP``）。**不是可调优化**：
            管线统计需要收敛，最前面的帧残差偏高且不稳定；实测用第 0 帧
            起标定会在干净数据上报出数百个假尖峰。
        robust: True（默认）用中位数/MAD 而非均值/标准差标定基准。
            **同样不是可选优化**：故障若从录制开头就在（线缆没插好、装配
            错误），均值/sd 会被故障本身污染——实测注入 33% 覆盖的尖峰后，
            均值/sd 标定把 z 全压到 1.4 以下，尖峰**完全检不出**。
        agg: 跨格聚合方式。"mean"（默认，推荐）对**故障覆盖一大片格子**的
            情形灵敏且不被单格重尾拖走；"max" 只适合格数很少（如逐帧已聚合成
            一列）的情形——实测在 24x28x28=18816 列上取 max，干净数据会报出
            近 6000 个假尖峰，因为"18816 个重尾变量取最大"必然饱和。
    Returns:
        (T,) 逐帧的聚合 z 分数。
    """
    a = check_finite_array(err, "err")
    flat = a.reshape(len(a), -1) if a.ndim > 1 else a.reshape(len(a), -1)
    t = len(flat)
    if t < MIN_ANALYSIS_FRAMES:
        raise InsufficientDataError(
            f"frozen_baseline_z: 需要至少 {MIN_ANALYSIS_FRAMES} 帧才能标定参照分位，"
            f"实得 {t}。帧数过少时的 z 分数没有统计意义。")
    lo = min(max(int(skip), 0), max(t - 2, 0))
    hi = min(lo + max(int(n_frozen), 1), t)
    if hi - lo < 2:                      # 太短：退回整段，但降级为不跳 warmup
        lo, hi = 0, max(t, 1)
    ref = flat[lo:hi]
    if robust:
        mu = np.median(ref, axis=0)
        mad = np.median(np.abs(ref - mu), axis=0)
        sd = 1.4826 * mad          # MAD -> sigma，正态下一致
    else:
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0)
    # MAD 可能在常数格上为 0；退回该格的全段尺度，再退回极小值
    sd = np.where(sd > 1e-9, sd, np.std(flat, axis=0))
    sd = np.where(sd > 1e-9, sd, 1e-6)
    z = (flat - mu) / sd
    z = np.abs(z)
    if z.shape[1] == 1 or agg == "max":
        return z.max(axis=1)
    # mean 聚合：对"一片格子同时异常"的故障灵敏，且不被单格重尾主导。
    # 打分的帧数越多，均值越稳定——这正是完整性巡检需要的性质。
    return z.mean(axis=1)


def reference_indices(n: int, warmup: int = WARMUP_SKIP,
                      max_fraction: float = REF_MAX_FRACTION,
                      min_frames: int = MIN_REF_FRAMES) -> np.ndarray:
    """选取标定用的**参照帧**（均匀抽取，覆盖录制的相当一部分）。

    设计依据（实测）：开头固定 120 帧可能整段落在安静期（seq_000 的参照窗
    聚合 z 最大值仅 2.68，而整段 p99 是 17.85），据此标定会把阈值压低、
    在干净数据上报出数百个假尖峰。故：

    * **均匀抽取**而非连续块——非平稳录制里任何单个连续块都可能不具代表性；
    * 参照长度取 ``min(n * max_fraction, n - warmup)``，至少 ``min_frames``
      （但不超过可用帧数），使分位数在任何长度的录制上都可靠。

    **已知代价（实测，不可隐瞒）**：均匀抽取使参照集**必然包含故障帧**
    （故障覆盖 f 时，参照集里也有约 f 比例是故障）。实测 1/3 覆盖的密集
    故障也会因此漏检。这是"参照集有代表性"与"参照集干净"之间的真实张力：
    没有独立健康段时两者不可兼得。要可靠检出故障，必须用
    ``reference_indices_from_healthy`` 提供一段已知健康的参照（LABEL_AUDIT
    §5b 的"必须留健康参照"上界）。
    """
    lo = min(max(int(warmup), 0), max(n - 1, 0))
    avail = n - lo
    if avail <= 0:
        return np.array([], dtype=int)
    k = int(min(max(avail * max_fraction, min(min_frames, avail)), avail))
    if k >= avail:
        return np.arange(lo, n)
    # 均匀抽取（确定性，不用随机数：同样输入必须给同样结论）
    idx = lo + (np.arange(k) * (avail / k)).astype(int)
    return np.unique(np.clip(idx, lo, n - 1))


def reference_indices_from_healthy(n: int, healthy: np.ndarray,
                                   max_frames: int = FROZEN_FRAMES
                                   ) -> np.ndarray:
    """从**已知健康帧**里取参照集（有外部参照段时的正确用法）。

    这是本模块唯一能可靠检出**稀疏**尖峰的方式：用另一段（或同一设备上一段）
    已知健康的录制来标定阈值，待查段整体当作未知。实测这是 LABEL_AUDIT §5b
    里"必须留健康参照"那条上界的直接实现。

    Args:
        n: 待查段帧数（仅用于裁剪返回值）。
        healthy: 布尔掩码或帧索引数组，标记**健康**的帧。
        max_frames: 参照集上限。
    """
    h = np.asarray(healthy)
    if h.dtype == bool:
        h = np.flatnonzero(h)
    h = np.asarray(h, dtype=int)
    h = h[(h >= 0) & (h < n)]
    if len(h) == 0:
        return h
    if len(h) <= max_frames:
        return h
    sel = (np.arange(max_frames) * (len(h) / max_frames)).astype(int)
    return h[np.clip(sel, 0, len(h) - 1)]


def spike_threshold(frozen_z: np.ndarray, warmup: int = WARMUP_SKIP,
                    n_frozen: int = FROZEN_FRAMES,
                    quantile: float = SPIKE_QUANTILE,
                    min_separation: float = SPIKE_MIN_SEPARATION,
                    ref_idx: np.ndarray | None = None) -> tuple[float, float]:
    """标定尖峰阈值，返回 ``(阈值, 参照集残余最大值)``。

    阈值 = ``参照集 quantile 分位 × min_separation``。

    * 参照集默认由 ``reference_indices`` 均匀抽取（不是开头连续块）；
    * 若调用方有**已知健康**的参照帧，应通过 ``ref_idx`` 传入——这是
      可靠检出稀疏故障的唯一途径（见 ``reference_indices`` 的已知代价）；
    * 用**乘性分离**而不是绝对 σ 下限——实测干净段 p99 就有 15–18σ，
      任何绝对尺度（"8σ"）都会变成常见取值，导致干净数据大量误报。
    """
    z = np.asarray(frozen_z, dtype=np.float64)
    if ref_idx is None:
        idx = reference_indices(len(z), warmup=max(warmup, 1),
                                min_frames=n_frozen)
    else:
        idx = np.asarray(ref_idx, dtype=int)
    if len(idx) < 8:
        return float("inf"), float("nan")
    ref = z[idx]
    q = float(np.quantile(ref, quantile))
    base = max(q, 1e-9)
    return base * float(min_separation), float(ref.max())


def detect_spikes(frozen_z: np.ndarray, sigma: float = SPIKE_SIGMA,
                  warmup: int = WARMUP_SKIP,
                  n_frozen: int = FROZEN_FRAMES,
                  quantile: float = SPIKE_QUANTILE,
                  min_separation: float = SPIKE_MIN_SEPARATION,
                  ref_idx: np.ndarray | None = None) -> list[int]:
    """单帧尖峰：**相对参照集**且**有量级分离**的离群帧。

    两道条件（由 ``spike_threshold`` 给出）：

    * 超过参照集 ``quantile`` 分位（默认 99.9%）；
    * 且超过该分位的 ``min_separation`` 倍（默认 3×）。

    第二道是必要的：只靠分位则任何重尾数据都会稳定报出 0.1% 的"尖峰"，
    那不是故障而是分布的形状。

    Args:
        frozen_z: ``frozen_baseline_z`` 的输出。
        sigma: 仅当序列过短无法标定时回退使用的固定阈值。
        warmup: 跳过的起始帧数（统计收敛期）。
        n_frozen: 参照帧数下限。
        quantile: 分位判据。
        min_separation: 相对分位的倍数下限。
        ref_idx: 已知健康帧的索引（有外部参照段时应传入）。
    """
    z = np.asarray(frozen_z, dtype=np.float64)
    if len(z) <= warmup + 8:
        thr = float(sigma)
    else:
        thr, _ = spike_threshold(z, warmup, n_frozen, quantile,
                                 min_separation, ref_idx)
    idx = np.flatnonzero(z[warmup:] > thr) + warmup
    return [int(i) for i in idx]


def raw_ratio_from_residual(mutated: np.ndarray, clean: np.ndarray) -> float:
    """IMU 边际贡献 = 注入故障后的残差 / 干净残差。

    这是**信号上界**：它的上限就是 IMU 在该数据上的价值。用来判断
    "这段数据上还值不值得做完整性巡检"。
    """
    c = float(np.mean(np.abs(np.asarray(clean, dtype=np.float64))))
    m = float(np.mean(np.abs(np.asarray(mutated, dtype=np.float64))))
    return m / max(c, 1e-12)


def report(clean_err: np.ndarray, imu_zeroed_err: np.ndarray | None = None,
           n_frozen: int = FROZEN_FRAMES, skip: int = WARMUP_SKIP,
           ref_fraction: float = REF_MIN_FRACTION,
           healthy_ref: np.ndarray | None = None) -> IntegrityReport:
    """对一段录制出完整性报告。

    Args:
        clean_err: (T,...) 干净录制的逐帧残差（幅度量）。
        imu_zeroed_err: (T,...) IMU 归零后的残差；None 表示不测边际贡献
            （此时 raw_ratio 为 nan，verdict 仍可基于尖峰给出）。
        n_frozen: 冻结基准帧数。
        skip: 标定基准前跳过的 warmup 帧数。
        ref_fraction: 健康参照比例下限，低于此报 no_reference。
        healthy_ref: **已知健康**帧的索引（例如同一设备上一段干净录制）。
            传入时用它标定阈值，这是可靠检出**稀疏**尖峰的唯一途径；
            不传则用待查段自身均匀抽取的参照集（见 ``reference_indices``
            的已知代价：会漏检稀疏故障，只能做录制级筛查）。

    Raises:
        NonFiniteError: 任一输入含 NaN/Inf。**这是一条重要的静默失败修复**：
            修复前全 NaN 输入会返回一份"0 个尖峰"的报告，即把数据损坏
            误报成"一切正常"。
        InsufficientDataError: 帧数少于 ``MIN_ANALYSIS_FRAMES``。
        ShapeError: ``imu_zeroed_err`` 与 ``clean_err`` 不等长。
    """
    clean = check_finite_array(clean_err, "clean_err")
    check_min_frames(len(clean), MIN_ANALYSIS_FRAMES, "integrity.report")
    if imu_zeroed_err is not None:
        check_finite_array(imu_zeroed_err, "imu_zeroed_err")
        if len(imu_zeroed_err) != len(clean):
            from .contracts import ShapeError
            raise ShapeError(
                f"imu_zeroed_err: 必须与 clean_err 等长（逐帧对齐），"
                f"实得 {len(imu_zeroed_err)} vs {len(clean)}")
    n = len(clean)
    fz = frozen_baseline_z(clean, n_frozen, skip)
    ref_idx = (reference_indices_from_healthy(n, healthy_ref, n_frozen)
               if healthy_ref is not None else None)
    thr, _ref_max = spike_threshold(fz, skip, n_frozen, ref_idx=ref_idx)
    spikes = detect_spikes(fz, warmup=skip, n_frozen=n_frozen,
                           ref_idx=ref_idx)
    n_ref = (len(ref_idx) if ref_idx is not None
             else len(reference_indices(n, max(skip, 1), min_frames=n_frozen)))

    if imu_zeroed_err is None:
        rr = float("nan")
        sr = float("nan")
        fzr = float("nan")
    else:
        mut = np.asarray(imu_zeroed_err, dtype=np.float64)
        rr = raw_ratio_from_residual(mut, clean)
        # 在线 surprise 的代理：帧级残差均值（避免依赖管线内部状态）
        sr = (float(np.mean(np.abs(mut))) / max(float(np.mean(np.abs(clean))), 1e-12))
        fzm = frozen_baseline_z(mut, n_frozen, skip)
        fzr = float(np.mean(fzm) / max(float(np.mean(fz)), 1e-12))

    # ---- verdict：两个上界都检查
    # 注意：不要复用参数名 healthy_ref（那是索引数组），这里算的是"未报警帧比例"
    healthy_frac = 1.0 - (len(spikes) / max(n, 1))
    if healthy_frac < ref_fraction:
        verdict = "no_reference"
        note = (f"健康参照仅 {healthy_frac:.0%}（< {ref_fraction:.0%}）：无法标定基线，"
                "拒绝出结论（这是正确行为，不是失败）")
    elif not np.isnan(rr) and rr < 1.05:
        verdict = "suspect"
        note = (f"raw_ratio={rr:.2f} 接近 1：该数据上 IMU 的边际贡献太低，"
                "'IMU 坏了'与'IMU 无贡献'在本观测上不可分")
    elif spikes:
        verdict = "suspect"
        note = (f"检出 {len(spikes)} 个尖峰帧（相对参照集阈值 {thr:.1f}，"
                f"参照 {n_ref} 帧）")
    else:
        verdict = "ok"
        note = (f"未检出完整性故障（阈值 {thr:.1f}，参照 {n_ref} 帧）。"
                f"注意：{SPIKE_MIN_COVERAGE_NOTE}")
    if healthy_ref is not None:
        note += "；本次使用了外部健康参照"

    return IntegrityReport(n_frames=n, raw_ratio=rr, surprise_ratio=sr,
                           frozen_z_ratio=fzr, spike_frames=spikes,
                           verdict=verdict, note=note,
                           threshold=float(thr), ref_frames=int(n_ref))

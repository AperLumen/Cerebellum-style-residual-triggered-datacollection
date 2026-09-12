"""多模态数据集模块——把常开多模态流变成可训练资产。

设计文档：``GUIDE.md §4.2``。与 ``record.py`` 的分工见该文档 §1：
``record.py`` 是在线单流记录（掉电安全、可续采），本模块是离线多模态数据集
（可训练、可发布、带权重与完整性标记）。

## 唯一的硬契约

**逐帧对齐**：单段 npz 里所有数组的第 0 维都是同一时间轴上的帧索引，
长度都等于 ``n_frames``。这条不满足，下游的模态融合就是错的。

## 三样交付

1. 可训练的模态对齐样本（帧 + IMU + 残差权重，同一时间轴）；
2. 每帧的**信息密度权重**（均衡难度采样用，不是重要性排序）；
3. 每段的**采集链路完整性标记**（来自 ``cerebellum_cancel.integrity``）。

## 关于 ``weight`` 的诚实说明

它是**待验证的主张**：残差是残余不可压缩性的估计量，用于按"每字节信息量"
做均衡采样，**不是**"选重要的帧"。此前四轮已否证"残差=语义重要性/事件时刻"。
判据（尚未执行）：``scripts/eval_weight_value.py``——同等字节预算下
weight 采样是否优于均匀采样。**在测出来之前，任何代码注释与文档都不得
把它写成已成立的结论。**
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "cmci-dataset-1"

# 必需载荷（逐帧对齐契约的成员）
REQUIRED_KEYS = ("t", "gyro", "score", "surprise", "keep", "weight")
# 可选载荷 -> 模态名（写进 section json 的 modalities 字段）
OPTIONAL_KEYS = {
    "accel": "accel",
    "frames_jpg": "video",
    "residual": "residual",
    "labels": "labels",
}

WEIGHT_MAX_DEFAULT = 10.0
WEIGHT_MIN = 1e-3           # 权重下限：保证最冷的帧仍有非零采样概率
                            # （不加下限时 surprise=0 的帧概率下溢到 1e-16，
                            #  等效于永不被采到——数据集不该有取不到的样本）


@dataclass
class Section:
    """一段录制（section = 一段连续录制的载荷 + 元信息）。"""
    section_id: str
    t: np.ndarray
    gyro: np.ndarray
    score: np.ndarray
    surprise: np.ndarray
    keep: np.ndarray
    weight: np.ndarray
    accel: np.ndarray | None = None
    frames_jpg: np.ndarray | None = None
    residual: np.ndarray | None = None
    labels: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(len(self.t))

    def modalities(self) -> dict[str, bool]:
        m = {"video": self.frames_jpg is not None,
             "gyro": True, "accel": self.accel is not None,
             "residual": self.residual is not None,
             "labels": self.labels is not None}
        return m

    def validate(self) -> None:
        """逐帧对齐是硬契约：所有载荷长度必须一致，且维度自洽。"""
        n = self.n_frames
        if n == 0:
            raise ValueError(f"{self.section_id}: 空段")
        lengths = {"t": len(self.t), "gyro": len(self.gyro),
                   "score": len(self.score), "surprise": len(self.surprise),
                   "keep": len(self.keep), "weight": len(self.weight)}
        for k, arr in (("accel", self.accel), ("frames_jpg", self.frames_jpg),
                       ("residual", self.residual), ("labels", self.labels)):
            if arr is not None:
                lengths[k] = len(arr)
        bad = {k: v for k, v in lengths.items() if v != n}
        if bad:
            raise ValueError(
                f"{self.section_id}: 逐帧对齐被破坏——n_frames={n}，"
                f"但 {bad} 长度不符。所有载荷第 0 维必须等于 n_frames。")
        if self.gyro.ndim != 2 or self.gyro.shape[1] != 3:
            raise ValueError(f"{self.section_id}: gyro 应为 (T,3)，实为 "
                             f"{self.gyro.shape}")
        if self.accel is not None and (self.accel.ndim != 2
                                       or self.accel.shape[1] != 3):
            raise ValueError(f"{self.section_id}: accel 应为 (T,3)，实为 "
                             f"{self.accel.shape}")
        # 决策契约：score 必须是 float64（无损），不得被量化
        if self.score.dtype != np.float64:
            raise ValueError(
                f"{self.section_id}: score 必须为 float64（决策无损契约），"
                f"实为 {self.score.dtype}。不要量化 score。")

    def to_npz(self) -> bytes:
        """序列化为 npz 字节串（帧以 JPEG 字节串存，故用 pickle 允许）。"""
        self.validate()
        payload: dict[str, Any] = {
            "t": np.asarray(self.t, dtype=np.float64),
            "gyro": np.asarray(self.gyro, dtype=np.float32),
            "score": np.asarray(self.score, dtype=np.float64),
            "surprise": np.asarray(self.surprise, dtype=np.float32),
            "keep": np.asarray(self.keep, dtype=np.uint8),
            "weight": np.asarray(self.weight, dtype=np.float32),
            "schema": np.array([SCHEMA_VERSION]),
        }
        if self.accel is not None:
            payload["accel"] = np.asarray(self.accel, dtype=np.float32)
        if self.residual is not None:
            payload["residual"] = np.asarray(self.residual, dtype=np.float16)
        if self.labels is not None:
            payload["labels"] = np.asarray(self.labels, dtype=np.float32)
        if self.frames_jpg is not None:
            payload["frames_jpg"] = np.asarray(self.frames_jpg, dtype=object)
        buf = io.BytesIO()
        np.savez_compressed(buf, **payload)
        return buf.getvalue()

    def meta_dict(self) -> dict[str, Any]:
        m = dict(self.meta)
        m.update({
            "section_id": self.section_id,
            "n_frames": self.n_frames,
            "duration_s": float(self.t[-1] - self.t[0]) if self.n_frames > 1
            else 0.0,
            "modalities": self.modalities(),
            "weight_stats": {
                "median": float(np.median(self.weight)),
                "p95": float(np.percentile(self.weight, 95)),
                "max": float(self.weight.max()),
                "frac_at_max": float((self.weight >= WEIGHT_MAX_DEFAULT - 1e-6)
                                     .mean()),
            },
            "keep_fraction": float(np.mean(self.keep > 0)),
        })
        return m


def compute_keep(score: np.ndarray, fps: float,
                 cfg: Any | None = None) -> np.ndarray:
    """用**和已发表实验同一个触发器**（``AdaptiveTrigger``）产出 keep 标记。

    为什么必须复用真触发器、不能用启发式阈值：``keep`` 是数据集里
    "这段为什么被存/被丢"的唯一记录，也是决策可复现契约的落点。
    早先版本用 ``surprise > median(surprise)`` 当占位符，结果每段都恰好
    keep 50%——那是中位数的定义，不是触发器的行为，会误导任何读这个字段的人。
    """
    from .core import AdaptiveTrigger, TriggerConfig

    c = cfg if cfg is not None else TriggerConfig(robust=True)
    trig = AdaptiveTrigger(c)
    n = len(score)
    keep = np.zeros(n, dtype=np.uint8)
    for i in range(1, n):
        on = trig.update(i / fps, float(score[i]), fps)
        keep[i] = 1 if on else 0
    clips = trig.finalize(n / fps)
    # finalize 可能会把尾部未关闭的 clip 补上，按 clips 重刷一次掩码
    for a, b in clips:
        keep[max(int(a * fps), 0):min(int(b * fps) + 1, n)] = 1
    return keep


def compute_weight(surprise: np.ndarray,
                   w_max: float = WEIGHT_MAX_DEFAULT,
                   w_min: float = WEIGHT_MIN) -> np.ndarray:
    """信息密度权重 = 残余不可压缩性（段内中位数归一化）。

    ``w_t = clip(surprise_t / median(surprise), w_min, w_max)``

    为什么用中位数归一化：使不同段（不同场景/运动量）的权重可放进同一
    采样池。**这是跨 regime 可比的前提**，也是这个量能被当作"信息密度"
    而不是"某段的绝对运动量"的关键。

    为什么有 ``w_min`` 下限：不加下限时，surprise 恰为 0 的单帧会得到权重
    0，采样概率下溢到 1e-16——等效于**永不出现**。数据集不应有"任何情况下
    都取不到"的样本，故设下限（默认 1e-3，使最冷帧仍有非零概率）。

    纪律：这是**权重**（用于均衡难度采样），不是**排序**（不筛掉任何帧）。
    """
    s = np.asarray(surprise, dtype=np.float64)
    med = float(np.median(s))
    if not np.isfinite(med) or med <= 0:
        return np.ones(len(s), dtype=np.float32)
    return np.clip(s / med, w_min, w_max).astype(np.float32)


def build_section(section_id: str, t: np.ndarray, gyro: np.ndarray,
                  score: np.ndarray, surprise: np.ndarray,
                  keep: np.ndarray | None = None, fps: float = 30.0,
                  raw_residual: np.ndarray | None = None,
                  accel: np.ndarray | None = None,
                  frames_jpg: list[bytes] | None = None,
                  residual: np.ndarray | None = None,
                  labels: np.ndarray | None = None,
                  meta: dict[str, Any] | None = None,
                  w_max: float = WEIGHT_MAX_DEFAULT,
                  trigger_cfg: Any | None = None,
                  healthy_ref: np.ndarray | None = None) -> Section:
    """组装一段（keep 用真触发器、weight 自动算、并跑完整性巡检）。

    Args:
        keep: None 时用 ``compute_keep`` 跑真 ``AdaptiveTrigger``（推荐）。
        raw_residual: (T,...) **未归一化**的残差幅度。完整性巡检必须用它，
            不能用 ``surprise``：surprise 已被在线统计归一化，会把故障
            "吸收"掉（LABEL_AUDIT §5：信号 1.49× 而在线上只剩 1.43×）。
        fps: 触发器需要帧率；与 ``t`` 不一致会得到错误的 clip 边界。
        healthy_ref: **已知健康**帧的索引（例如同一设备上一段干净录制）。
            传入时完整性巡检能可靠检出**密集**故障；不传则只能用本段
            自身的均匀参照集，只能做粗筛查（见 ``reference_indices``
            的已知代价）。
    """
    from .contracts import InsufficientDataError, NonFiniteError, ShapeError
    from .integrity import MIN_ANALYSIS_FRAMES, report

    # 时间轴是**这条硬契约的另一半**：`n_frames` 只保证长度一致，
    # 而 `t` 保证"同一时间轴"这个语义。此前只查了长度，没查 t 本身
    # ——于是 NaN 时间戳能静默进入数据集（协议层压力测试发现，2026-09-13）。
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ShapeError(
            f"{section_id}: t 必须是 1 维时间轴 (T,)，实得 shape={t_arr.shape}")
    if not np.all(np.isfinite(t_arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(t_arr)))
        raise NonFiniteError(
            f"{section_id}: t 含 {n_bad} 个 NaN/Inf。时间轴是「逐帧对齐」"
            "契约的另一半——长度对但时间戳是 NaN，下游按时间对齐就会出错。")
    score = np.asarray(score, dtype=np.float64)
    surprise = np.asarray(surprise, dtype=np.float32)
    n = len(score)
    if len(t_arr) != n:
        raise ShapeError(
            f"{section_id}: t 长度 {len(t_arr)} != score 长度 {n}")
    w = compute_weight(surprise, w_max)
    if keep is None:
        keep_arr = compute_keep(score, fps, trigger_cfg)
    else:
        keep_arr = np.asarray(keep, dtype=np.uint8)

    m = dict(meta or {})
    # 完整性巡检：幅度来源优先用未归一化的 raw_residual；
    # 没给就退回 surprise 并**在元信息里声明降级**，不能默默替换。
    if raw_residual is not None:
        amp = np.asarray(raw_residual, dtype=np.float64)
        m["integrity_input"] = "raw_residual"
    else:
        amp = np.asarray(surprise, dtype=np.float64)
        m["integrity_input"] = "surprise(normalized) — 降级口径，灵敏度偏低"
    # 帧数不足以做完整性判定时，**记录"无法判定"而不是中断写数据**。
    # 写数据与判定数据是两件事：短片段是合法的数据，只是它的完整性
    # 结论没有统计意义（见 integrity.MIN_ANALYSIS_FRAMES）。这里显式写
    # 一个 insufficient_data 结论，让下游看到的是"未判定"而非"通过"——
    # 把"没结论"伪装成"ok"才是真正的危险。
    try:
        rep = report(amp[:, None] if amp.ndim == 1 else amp,
                     healthy_ref=healthy_ref)
        m["integrity"] = rep.as_dict()
    except InsufficientDataError as exc:
        m["integrity"] = {
            "verdict": "insufficient_data",
            "n_frames": int(n),
            "min_analysis_frames": int(MIN_ANALYSIS_FRAMES),
            "reason": str(exc),
            "spike_frames": [],
        }

    sec = Section(section_id=section_id, t=np.asarray(t, dtype=np.float64),
                  gyro=np.asarray(gyro, dtype=np.float32), score=score,
                  surprise=surprise, keep=keep_arr, weight=w, accel=accel,
                  frames_jpg=(np.asarray(frames_jpg, dtype=object)
                              if frames_jpg is not None else None),
                  residual=residual, labels=labels, meta=m)
    # **在构造处就校验**，而不是拖到 `DatasetWriter.add()`（协议层压力测试
    # 发现，2026-09-13）：此前 gyro 传 1 维时 `build_section` 兴高采烈地返回
    # 一个非法 Section，错误直到写入时才冒出来——**失败出现在与犯错不同的
    # 调用上**，调用方很难把两者联系起来。契约校验应当在"造出这个东西"的
    # 那一刻发生。
    sec.validate()
    return sec


# ------------------------------------------------------------------ writer

class DatasetWriter:
    """把一个数据集写到目录。

    布局（GUIDE.md §4.2 §2）::

        <out>/manifest.json
        <out>/README.md
        <out>/sections/<id>.npz
        <out>/sections/<id>.json
    """

    def __init__(self, out_dir: str, name: str, description: str = "",
                 source: str = "", licenses: str = "",
                 weight_is_claim: bool = True):
        self.out = out_dir
        self.name = name
        self.description = description
        self.source = source
        self.licenses = licenses
        self.weight_is_claim = weight_is_claim
        self.sections: list[dict[str, Any]] = []
        os.makedirs(os.path.join(out_dir, "sections"), exist_ok=True)

    def add(self, sec: Section) -> None:
        sec.validate()
        npz_path = os.path.join(self.out, "sections", f"{sec.section_id}.npz")
        with open(npz_path, "wb") as fp:
            fp.write(sec.to_npz())
        meta = sec.meta_dict()
        meta["npz_bytes"] = os.path.getsize(npz_path)
        with open(os.path.join(self.out, "sections",
                               f"{sec.section_id}.json"), "w") as fp:
            json.dump(meta, fp, indent=1, ensure_ascii=False)
        self.sections.append(meta)

    def finalize(self) -> dict[str, Any]:
        n_frames = sum(s["n_frames"] for s in self.sections)
        n_bytes = sum(s["npz_bytes"] for s in self.sections)
        mod_counts: dict[str, int] = {}
        for s in self.sections:
            for k, v in s["modalities"].items():
                if v:
                    mod_counts[k] = mod_counts.get(k, 0) + 1
        verdicts: dict[str, int] = {}
        for s in self.sections:
            v = s.get("integrity", {}).get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1

        man = {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "licenses": self.licenses,
            "n_sections": len(self.sections),
            "n_frames": int(n_frames),
            "total_bytes": int(n_bytes),
            "bytes_per_frame": float(n_bytes / max(n_frames, 1)),
            "modality_coverage": mod_counts,
            "integrity_verdicts": verdicts,
            "hard_contract": ("所有载荷第 0 维 = 同一时间轴帧索引 = n_frames；"
                              "score 必须 float64（决策无损）"),
            "section_ids": [s["section_id"] for s in self.sections],
            "weight": {
                "definition": (f"clip(surprise / median(surprise), "
                               f"{WEIGHT_MIN:g}, {WEIGHT_MAX_DEFAULT:g})"),
                "meaning": "残余不可压缩性（信息密度权重的估计量）",
                "NOT": "语义重要性 / 事件时刻 —— 这两条已被四轮实测否证",
                "status": ("CLAIM, NOT ESTABLISHED" if self.weight_is_claim
                           else "established"),
                "min_floor_note": ("权重的下限保证最冷的帧仍有非零采样概率，"
                                   "数据集不应有取不到的样本"),
                "eval_script": "scripts/eval_weight_value.py",
            },
            "sections": self.sections,
        }
        with open(os.path.join(self.out, "manifest.json"), "w") as fp:
            json.dump(man, fp, indent=1, ensure_ascii=False)
        with open(os.path.join(self.out, "README.md"), "w") as fp:
            fp.write(render_readme(man))
        return man


def render_readme(man: dict[str, Any]) -> str:
    mods = ", ".join(f"{k}×{v}" for k, v in sorted(man["modality_coverage"].items()))
    verds = ", ".join(f"{k}×{v}" for k, v in sorted(man["integrity_verdicts"].items()))
    lines = [
        f"# {man['name']}", "",
        man["description"], "",
        f"- schema: `{man['schema']}`",
        f"- source: `{man['source']}`",
        f"- licenses: {man['licenses'] or '(未声明)'}",
        f"- sections: **{man['n_sections']}**, frames: **{man['n_frames']:,}**",
        f"- size: **{man['total_bytes'] / 1e6:.1f} MB** "
        f"({man['bytes_per_frame']:.0f} B/frame)",
        f"- modality coverage (sections): {mods}",
        f"- integrity verdicts: {verds}", "",
        "## 硬契约", "",
        man["hard_contract"], "",
        "## `weight` 字段的意义（重要）", "",
        f"- 定义：`{man['weight']['definition']}`",
        f"- 含义：{man['weight']['meaning']}",
        f"- **不是**：{man['weight']['NOT']}",
        f"- 状态：**{man['weight']['status']}**——判据见 "
        f"`{man['weight']['eval_script']}`",
        f"- 下限：{man['weight']['min_floor_note']}", "",
        "## 逐段", "",
        "| section | frames | dur(s) | keep% | 模态 | integrity |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in man["sections"]:
        m = ",".join(k for k, v in s["modalities"].items() if v)
        iv = s.get("integrity", {}).get("verdict", "—")
        lines.append(f"| `{s['section_id']}` | {s['n_frames']} | "
                     f"{s['duration_s']:.1f} | {s['keep_fraction'] * 100:.1f} | "
                     f"{m} | {iv} |")
    lines += ["", "---", "",
              "*由 `cerebellum_cancel/dataset.py` 自动生成；格式说明见 "
              "`GUIDE.md §4.2`。*", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ reader

@dataclass
class LoadedSection:
    """读回的一段：载荷 + 元信息。"""
    payload: dict[str, Any]
    meta: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    def has(self, modality: str) -> bool:
        """按**声明的模态**查询，而不是猜哪个键存在。"""
        return bool(self.meta["modalities"].get(modality, False))


def load_dataset(path: str) -> dict[str, Any]:
    with open(os.path.join(path, "manifest.json")) as fp:
        return json.load(fp)


def load_section(path: str, section_id: str,
                 mmap: bool = True) -> LoadedSection:
    """读回一段。验证逐帧对齐契约（读也要查，不能只靠写入端自觉）。"""
    npz = os.path.join(path, "sections", f"{section_id}.npz")
    with open(os.path.join(path, "sections", f"{section_id}.json")) as fp:
        meta = json.load(fp)
    with np.load(npz, allow_pickle=True) as z:
        payload = {k: (z[k] if k == "frames_jpg" else z[k]) for k in z.files}
    lens = {k: len(v) for k, v in payload.items()
            if isinstance(v, np.ndarray) and v.ndim >= 1
            and k not in ("schema",)}
    bad = {k: v for k, v in lens.items() if v != meta["n_frames"]}
    if bad:
        raise ValueError(f"{section_id}: 读回的逐帧对齐被破坏——"
                         f"n_frames={meta['n_frames']}，不符 {bad}")
    if payload["score"].dtype != np.float64:
        raise ValueError(f"{section_id}: score 不是 float64，决策无损契约被破坏")
    return LoadedSection(payload=payload, meta=meta)


def sampling_weights(sec: "LoadedSection | Section", temperature: float = 1.0
                     ) -> np.ndarray:
    """把 ``weight`` 转成采样概率（用于均衡难度采样）。

    ``p ∝ weight ** temperature``（temperature=0 → 均匀）。

    **注意**：这是"按信息密度加权"，不是"选重要的帧"。所有帧都在池子里，
    权重只改变被抽中的概率。其价值尚未被验证——见 GUIDE.md §4.2 §7。

    Args:
        sec: ``LoadedSection``（读回的段）或 ``Section``（构建中的段）都可。
    """
    w = (sec.payload["weight"] if isinstance(sec, LoadedSection)
         else sec.weight)
    w = np.asarray(w, dtype=np.float64)
    if temperature == 0.0:
        return np.full(len(w), 1.0 / max(len(w), 1))
    p = np.power(np.maximum(w, 1e-12), temperature)
    return (p / p.sum()).astype(np.float64)

"""CMCI 引擎：把一段录制变成可交付的产物。**A（CLI）与 B（前端）共用这一层。**

## 为什么要单独一个引擎

工程化落地时最容易犯的错，是把同一套逻辑在 CLI 里写一遍、在界面里再写一遍——
于是有了两个真值源，改一处忘一处。本项目已经在 `public_edition/` 上踩过同类问题
（副本悄悄漂移，而它自己的测试照样全过）。

所以这里的分层是**硬边界**：

    输入适配        引擎（本文件）              导出器
    ----------      -------------------        ----------------
    mp4 + csv  ->   run() -> CMCIResult  ->    dataset / html / parquet / jsonl
                                               （各自只读中间态，不碰引擎）
    frame dir  ->   （同一个 run()）
    未来: 摄像头 ->   （同一个 run()）

`run()` 是**唯一**做计算的地方；导出器是**纯函数**（中间态 → 文件）。
CLI 和前端都只调 `run()` + 选导出器。

## 两层过滤（这是产品定义的核心，必须说清）

用户要的是"过滤出有用的片段"，但要小心：**残差触发器只能回答"哪里动了"，
回答不了"什么重要"**——后者在本项目上被四轮实验否证（GUIDE §2.3.6–2.3.8，
含 250 轮真实数据复核）。所以引擎把过滤明确拆成两层：

  **第 1 层（本引擎负责）：运动去重。** 剪掉"设备自己的运动就能解释"的冗余帧。
      这一层实测成立，且成本极低（副本 0.033 ms/帧，整层 0.17 ms/帧，而光流
      要 8.5 ms——差 250 倍）。

  **第 2 层（调用方负责，引擎只留钩子）：语义/业务判断。** 哪些片段真的重要，
      取决于任务——抓取失败？异常动作？标注？引擎不猜，通过 `rule` 参数把
      候选片段交给调用方筛。

这个划分不是洁癖：把第 2 层塞进引擎，就会把已被证伪的"残差=重要性"当成
产品能力卖出去。**输出里每个片段都带 `selection_reason`，如实标明它是被
哪一层选中的。**

## 用法

    from cerebellum_cancel.engine import run, EngineConfig

    res = run("clip.mp4", "imu.csv")           # 塞进去
    res.save("out/")                            # 出数据集
    # 或分开：res.to_dataset(...) / res.to_html(...) / res.to_parquet(...)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from .acquisition import AcquisitionLayer
from .contracts import (ConfigError, InsufficientDataError, NonFiniteError,
                        ShapeError, check_min_frames, check_positive)

# 一段录制至少要这么多帧才值得跑。低于此值的结论没有统计意义——
# 与 integrity.MIN_ANALYSIS_FRAMES 同一纪律：宁可拒绝出结论，不可给出
# 无依据的结论。（采集层做的是"存不存"的在线决策，比整段体检的阈值低。）
MIN_RUN_FRAMES = 30


class SourceProtocol(Protocol):
    """输入适配器契约。任何能逐帧吐 (frame, gyro, t) 的东西都能接进来。"""

    def __len__(self) -> int: ...
    def frames(self) -> Iterable[tuple[np.ndarray, np.ndarray, float]]: ...
    @property
    def meta(self) -> dict: ...


@dataclass
class Segment:
    """一个被选中的片段。"""

    start: float
    end: float
    n_frames: int
    mean_score: float
    peak_score: float
    # **诚实字段**：这个片段是被哪一层选中的。
    #   "motion_dedup" = 第 1 层（设备自己预测不了）；引擎能负责的全部
    #   "rule:<name>"   = 第 2 层业务规则进一步筛出来的
    selection_reason: str = "motion_dedup"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end,
                "duration": self.duration, "n_frames": self.n_frames,
                "mean_score": self.mean_score, "peak_score": self.peak_score,
                "selection_reason": self.selection_reason}


@dataclass
class CMCIResult:
    """统一中间态。**导出器只读这个，不接触引擎内部。**

    逐帧数组的轴 0 必须是同一帧索引——这是本项目唯一的硬契约
    （见 GUIDE §4.2）。导出器可以依赖它。
    """

    # --- 逐帧数据（全部等长，axis 0 = 帧索引）---
    t: np.ndarray                 # (N,) float64 秒
    gyro: np.ndarray              # (N,3) float32 deg/s
    score: np.ndarray             # (N,) float64 —— 决策依据，无损
    surprise: np.ndarray          # (N,) float32
    keep: np.ndarray              # (N,) uint8
    weight: np.ndarray            # (N,) float32 残余不可压缩性（诊断读数）

    # --- 片段级 ---
    segments: list[Segment] = field(default_factory=list)

    # --- 元信息 ---
    meta: dict[str, Any] = field(default_factory=dict)

    # --- 可选：z 图（有损诊断件，不参与决策）---
    z_maps: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.t)
        for name in ("gyro", "score", "surprise", "keep", "weight"):
            a = getattr(self, name)
            if len(a) != n:
                raise ShapeError(
                    f"{name}: 长度 {len(a)} != 帧数 {n}。"
                    "本格式的唯一硬契约是 axis 0 = 帧索引，长度必须一致。")

    @property
    def n_frames(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n_frames > 1 else 0.0

    @property
    def keep_fraction(self) -> float:
        return float(self.keep.mean()) if self.n_frames else 0.0

    @property
    def stored_seconds(self) -> float:
        """被选中片段的总时长（合并重叠后）。"""
        return sum(s.duration for s in _merge(self.segments))

    def summary(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "duration_s": self.duration,
            "n_segments": len(self.segments),
            "keep_fraction": self.keep_fraction,
            "stored_seconds": self.stored_seconds,
            "fps": self.meta.get("fps"),
            "source": self.meta.get("source"),
        }

    # ------------------------------------------------------------ 导出入口

    def save(self, out_dir: str, formats: Iterable[str] = ("dataset", "html"),
             **kw) -> dict[str, str]:
        """一次导出多种格式。返回 {格式: 实际路径}。"""
        os.makedirs(out_dir, exist_ok=True)
        written: dict[str, str] = {}
        for fmt in formats:
            fn = _EXPORTERS.get(fmt)
            if fn is None:
                raise ConfigError(
                    f"未知导出格式 {fmt!r}；可选 {sorted(_EXPORTERS)}")
            written[fmt] = fn(self, out_dir, **kw)
        return written

    def to_dataset(self, out_dir: str, **kw) -> str:
        return _export_dataset(self, out_dir, **kw)

    def to_html(self, out_dir: str, **kw) -> str:
        return _export_html(self, out_dir, **kw)

    def to_parquet(self, out_dir: str, **kw) -> str:
        return _export_parquet(self, out_dir, **kw)

    def to_jsonl(self, out_dir: str, **kw) -> str:
        return _export_jsonl(self, out_dir, **kw)


def _merge(segs: list[Segment]) -> list[Segment]:
    """合并重叠/相邻片段，避免时长重复计数。"""
    if not segs:
        return []
    out = [segs[0]]
    for s in segs[1:]:
        if s.start <= out[-1].end:
            out[-1].end = max(out[-1].end, s.end)
            out[-1].n_frames += s.n_frames
            out[-1].peak_score = max(out[-1].peak_score, s.peak_score)
        else:
            out.append(s)
    return out


# ------------------------------------------------------------------ 引擎

@dataclass
class EngineConfig:
    """引擎参数。默认值都来自实测/已发布口径，不要凭感觉改。"""

    frontend: str = "mobilenet_v3_small[:3]"
    fps: float | None = None          # None = 用源自身
    imu_offset: float = 0.0
    mirror: bool = False
    crop: tuple[float, ...] | None = None
    max_frames: int | None = None
    # 是否保留 z 图。z 图是**有损诊断件**（int8/float32 量化），
    # 不参与决策；决策用无损 float64 score（见 GUIDE §4.1）。
    keep_z_maps: bool = True


@dataclass
class FrameView:
    """喂给第 2 层规则的**只读逐帧视图**。

    为什么不让规则直接拿 `CMCIResult`：规则是调用方写的代码，给它完整对象
    就等于给了改坏引擎状态的机会（改 keep、改 segments……）。这里只暴露
    只读数组，规则能筛片段、能看分数，但改不动引擎的账。
    """

    t: np.ndarray                 # (N,) float64
    score: np.ndarray             # (N,) float64
    surprise: np.ndarray          # (N,) float32
    keep: np.ndarray              # (N,) uint8
    gyro: np.ndarray              # (N,3) float32

    @property
    def n_frames(self) -> int:
        return len(self.t)


Rule = Callable[[list[Segment], FrameView], list[Segment]]


def run(video_path: str, imu_path: str | None = None, *,
        cfg: EngineConfig | None = None,
        rule: Rule | None = None,
        source: SourceProtocol | None = None) -> CMCIResult:
    """**核心入口。** 塞进一段录制，出来一个可导出的中间态。

    Args:
        video_path / imu_path: 输入（`source` 给出时忽略这两个）。
        cfg: 引擎参数。
        rule: **第 2 层过滤钩子**。引擎先做运动去重（第 1 层），再把候选
            片段交给 `rule` 做业务筛选。签名 ``(segments, result) -> segments``。
            **不传就不做第二层**——引擎不会假装知道什么是"重要"。
        source: 自定义输入适配器（跳过文件读取，供前端/测试注入）。

    Raises:
        InsufficientDataError: 帧数少于 ``MIN_RUN_FRAMES``。
        ConfigError / NonFiniteError / ShapeError: 参数非法（见 contracts）。
        OSError: 输入读不出来。
    """
    cfg = cfg or EngineConfig()
    if not np.isfinite(cfg.imu_offset):
        raise NonFiniteError(f"imu_offset: 必须是有限标量，实得 {cfg.imu_offset!r}")

    src = source if source is not None else _FileSource(
        video_path, imu_path, fps=cfg.fps, crop=cfg.crop,
        imu_offset=cfg.imu_offset, mirror=cfg.mirror)

    layer = AcquisitionLayer(frontend=cfg.frontend)

    ts, gyros, scores, surprises, keeps, zs = [], [], [], [], [], []
    for i, (frame, gyro, t) in enumerate(src.frames()):
        if cfg.max_frames is not None and i >= cfg.max_frames:
            break
        out = layer.step(frame, gyro, t)
        ts.append(out["t"])
        gyros.append(np.asarray(gyro, dtype=np.float64))
        scores.append(out["score"])
        surprises.append(out["surprise"])
        keeps.append(1 if out["keep"] else 0)
        if cfg.keep_z_maps:
            zs.append(out["z_map"])

    n = len(ts)
    check_min_frames(n, MIN_RUN_FRAMES, "engine.run")

    t_arr = np.asarray(ts, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(score_arr)):
        raise NonFiniteError(
            f"score 含非有限值（{int((~np.isfinite(score_arr)).sum())} 处）。"
            "上游可能传入了 NaN——这正是 contracts 要拦的东西，不该走到这里。")

    clips = layer.finalize()
    segments = _clips_to_segments(clips, t_arr, score_arr, keeps)
    if rule is not None:
        view = FrameView(
            t=t_arr, score=score_arr,
            surprise=np.asarray(surprises, dtype=np.float32),
            keep=np.asarray(keeps, dtype=np.uint8),
            gyro=(np.asarray(gyros, dtype=np.float32)
                  if gyros else np.zeros((0, 3), np.float32)))
        segments = list(rule(segments, view))

    res = CMCIResult(
        t=t_arr,
        gyro=(np.asarray(gyros, dtype=np.float32)
              if gyros else np.zeros((0, 3), np.float32)),
        score=score_arr,
        surprise=np.asarray(surprises, dtype=np.float32),
        keep=np.asarray(keeps, dtype=np.uint8),
        weight=_weights(np.asarray(surprises, dtype=np.float64)),
        segments=segments,
        z_maps=(np.stack(zs) if zs else None),
        meta={
            "source": src.meta.get("source", str(video_path)),
            "fps": src.meta.get("fps", cfg.fps),
            "frontend": cfg.frontend,
            "imu": src.meta.get("imu"),
            "engine_version": 1,
            # 把已知边界写进产物，避免下游误读
            "selection_scope": (
                "layer1=motion_dedup only. Residual triggering answers WHERE "
                "something moved, not WHAT happened; semantic importance is "
                "falsified on real data (GUIDE 2.3.6-2.3.8). Use `rule=` for "
                "task-specific selection."),
        },
    )
    return res


def _clips_to_segments(clips, t: np.ndarray, score: np.ndarray,
                       keeps: list[int]) -> list[Segment]:
    """把触发器的片段转成 Segment，并标上「第 1 层选中」。"""
    segs: list[Segment] = []
    for a, b in clips:
        m = (t >= a) & (t <= b)
        s = score[m]
        segs.append(Segment(
            start=float(a), end=float(b), n_frames=int(m.sum()),
            mean_score=float(s.mean()) if s.size else 0.0,
            peak_score=float(s.max()) if s.size else 0.0,
            selection_reason="motion_dedup",
        ))
    return segs


def _weights(surprise: np.ndarray) -> np.ndarray:
    """残余不可压缩性（诊断读数）。

    **注意**：这个字段**不得**用于采样决策——实测它选出的帧正例率与随机
    无异（GUIDE §4.2 判据，2/10 段超随机 p95）。保留它是为了诊断。
    """
    med = float(np.median(surprise)) if surprise.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return np.clip(surprise / med, 1e-3, 10.0).astype(np.float32)


# ------------------------------------------------------------ 输入适配器

def replay(res: CMCIResult, *, trigger_cfg: Any = None,
           rule: Rule | None = None,
           keep_z_maps: bool | None = None) -> CMCIResult:
    """**只重跑触发层**，复用 `res` 里已算好的逐帧 score —— 用于"调参重跑"。

    ## 为什么需要它（实测决定的）

    分解成本后发现，改触发阈值时**真正贵的是前端前向，不是解码**：

        解码 + 补通道      0.68 ms/帧   （21%）
        前端 + 取消 + 触发  2.54 ms/帧   （79%）

    而 `score` 是前端+取消的产物、与触发阈值**无关**。所以调阈值时把前 79%
    省掉即可，触发本身是纯 EMA（微秒级），整段几乎瞬时。

    ## 一条不可越过的纪律

    重放**只能读引擎产出的 `score`**，不能拿 z 图反推、更不能自己重算。
    z 图是有损诊断件（GUIDE §4.1：从量化 z 反推决策会漂），而 score 是
    无损 float64 决策标量。这正是 `record`/`dataset` 两条线共用的契约。

    Args:
        res: 已跑过的结果（提供逐帧 score/t/gyro）。
        trigger_cfg: 新的 TriggerConfig（None = 用默认）。
        rule: 第 2 层钩子（同 `run`）。
        keep_z_maps: 是否保留原 z 图（None = 沿用 res）。

    Returns:
        新的 `CMCIResult`。t/gyro/score/surprise 与输入**逐位相同**
        （它们不依赖触发参数），只有 keep/segments 会变。
    """
    from .core import AdaptiveTrigger, TriggerConfig

    cfg = trigger_cfg or TriggerConfig()
    trig = AdaptiveTrigger(cfg=cfg)

    fps = res.meta.get("fps") or 30.0
    check_positive(fps, "fps")

    keeps: list[int] = []
    for i in range(res.n_frames):
        on = trig.update(float(res.t[i]), float(res.score[i]), fps)
        keeps.append(1 if on else 0)
    clips = trig.finalize(float(res.t[-1]) if res.n_frames else 0.0)

    keep_arr = np.asarray(keeps, dtype=np.uint8)
    segments = _clips_to_segments(clips, res.t, res.score, keeps)
    if rule is not None:
        view = FrameView(t=res.t, score=res.score, surprise=res.surprise,
                         keep=keep_arr, gyro=res.gyro)
        segments = list(rule(segments, view))

    meta = dict(res.meta)
    meta["replayed"] = True
    meta["trigger_cfg"] = {
        "threshold_k": cfg.threshold_k, "off_k": cfg.off_k,
        "pre_roll": cfg.pre_roll, "post_roll": cfg.post_roll,
        "min_clip_len": cfg.min_clip_len, "robust": cfg.robust,
    }

    keep_z = res.z_maps if keep_z_maps is None else (
        res.z_maps if keep_z_maps else None)
    return CMCIResult(
        t=res.t, gyro=res.gyro, score=res.score, surprise=res.surprise,
        keep=keep_arr, weight=res.weight, segments=segments,
        z_maps=keep_z, meta=meta)


class _FileSource:
    """文件输入：视频帧 + IMU csv。把 I/O 与引擎隔开，便于换源。"""

    def __init__(self, video_path: str, imu_path: str | None, *,
                 fps: float | None, crop, imu_offset: float, mirror: bool):
        from .io import VideoStream, gyro_at, load_imu_csv

        if fps is not None:
            check_positive(fps, "fps")
        if not np.isfinite(imu_offset):
            raise NonFiniteError(f"imu_offset 非有限: {imu_offset!r}")

        self._stream = VideoStream(video_path, fps=fps, crop=crop)
        imu = load_imu_csv(imu_path) if imu_path else []
        for s in imu:
            s.t += imu_offset
        self._imu = imu
        self._mirror = mirror
        self._video_path = video_path
        self._imu_path = imu_path

    def __len__(self) -> int:
        # 未知时给 0（引擎不依赖它做分配）
        return 0

    @property
    def meta(self) -> dict:
        return {"source": self._video_path, "fps": self._stream.fps,
                "imu": self._imu_path}

    def frames(self) -> Iterable[tuple[np.ndarray, np.ndarray, float]]:
        from .io import gyro_at
        t, hint = 0.0, 0
        while True:
            gray = self._stream.read_gray()
            if gray is None:
                break
            # **接口契约差异（实测踩过）**：`AcquisitionLayer.step` 要求
            # `(H,W,3)` RGB，而 `read_gray()` 给的是 `(H,W)`。直接传进去会在
            # torchvision 的 Normalize 处报 broadcast 错误
            # （[1,224,224] vs [3,224,224]），信息量还看不出来。
            # 这里显式补齐三通道：取消路径本来就只用亮度（warp/残差都是
            # 单通道量），复制三份不丢信息，但满足了前端契约。
            if gray.ndim == 2:
                frame = np.repeat(gray[:, :, None], 3, axis=2)
            else:
                frame = gray
            gyro, hint = gyro_at(self._imu, hint, t)
            if self._mirror:
                gyro = np.asarray(gyro, dtype=np.float64).copy()
                gyro[1] = -gyro[1]          # 镜像视图翻转 yaw
            yield frame, gyro, t
            t += self._stream.frame_dt


# ---------------------------------------------------------------- 导出器
#
# 每个导出器都是纯函数：只读 CMCIResult，写文件，返回路径。
# 它们**不**做计算——要改数值请改引擎，别在这里打补丁。

def _export_dataset(res: CMCIResult, out_dir: str, *,
                    section_id: str = "sec_000", **_) -> str:
    """导出为本项目的数据集格式（schema cmci-dataset-1）。"""
    from .dataset import DatasetWriter, build_section

    path = os.path.join(out_dir, "dataset")
    sec = build_section(
        section_id,
        t=res.t, gyro=res.gyro, score=res.score,
        surprise=res.surprise, keep=res.keep,
        fps=res.meta.get("fps") or 30.0,
        raw_residual=res.surprise,      # 完整性哨兵要未归一化残差
        meta={"engine": res.meta, "segments": [s.as_dict()
                                               for s in res.segments]},
    )
    w = DatasetWriter(path, name=os.path.basename(out_dir.rstrip("/")) or "cmci")
    w.add(sec)
    w.finalize()
    return path


def _export_jsonl(res: CMCIResult, out_dir: str, **_) -> str:
    """逐帧 JSONL —— 给外部系统消费，流式可读。"""
    path = os.path.join(out_dir, "frames.jsonl")
    with open(path, "w", encoding="utf-8") as fp:
        for i in range(res.n_frames):
            fp.write(json.dumps({
                "i": i, "t": float(res.t[i]),
                "gyro": [float(x) for x in res.gyro[i]],
                "score": float(res.score[i]),
                "surprise": float(res.surprise[i]),
                "keep": bool(res.keep[i]),
                "weight": float(res.weight[i]),
            }, ensure_ascii=False) + "\n")
    # 片段单独一份，便于按片段检索
    seg_path = os.path.join(out_dir, "segments.jsonl")
    with open(seg_path, "w", encoding="utf-8") as fp:
        for s in res.segments:
            fp.write(json.dumps(s.as_dict(), ensure_ascii=False) + "\n")
    return path


def _export_parquet(res: CMCIResult, out_dir: str, **_) -> str:
    """Parquet —— 列存，适合大规模下游分析。

    需要 pyarrow；未安装时给出可执行的提示而不是静默失败。

    注：pyarrow 是**可选依赖**（不在 requirements 里），所以 pyright 会报
    reportMissingImports。这是预期的——用 type: ignore 显式记录，而不是把它
    加进硬依赖让别人为了导出一个 jsonl 就装 100MB 的 arrow。
    """
    try:
        import pyarrow as pa                            # type: ignore
        import pyarrow.parquet as pq                    # type: ignore
    except ImportError as exc:               # pragma: no cover
        raise ConfigError(
            "导出 parquet 需要 pyarrow：pip install pyarrow。"
            "或者改用 --format jsonl（零依赖）。") from exc

    path = os.path.join(out_dir, "frames.parquet")
    table = pa.table({
        "i": np.arange(res.n_frames, dtype=np.int64),
        "t": res.t.astype(np.float64),
        "gyro_pitch": res.gyro[:, 0].astype(np.float32),
        "gyro_yaw": res.gyro[:, 1].astype(np.float32),
        "gyro_roll": res.gyro[:, 2].astype(np.float32),
        "score": res.score.astype(np.float64),
        "surprise": res.surprise.astype(np.float32),
        "keep": res.keep.astype(np.int8),
        "weight": res.weight.astype(np.float32),
    })
    pq.write_table(table, path)
    return path


def _export_html(res: CMCIResult, out_dir: str, *, max_thumb: int = 200,
                 **_) -> str:
    """自包含 HTML 索引 —— 给人看/检索。

    设计要点（避免"看起来很厉害但没法验证"）：
      * **不内嵌视频**（体积会爆）。生成帧缩略图 + 时间轴，指向源文件。
      * **如实标注选择依据**——每个片段写明是第 1 层还是第 2 层选的。
      * 残差曲线与片段区间叠在一起，能直接看出"为什么这里被选中"。
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.html")
    s = res.summary()
    rows = "\n".join(
        f"<tr><td>{i}</td><td>{seg.start:.2f}</td><td>{seg.end:.2f}</td>"
        f"<td>{seg.duration:.2f}</td><td>{seg.n_frames}</td>"
        f"<td>{seg.mean_score:.3f}</td><td>{seg.peak_score:.3f}</td>"
        f"<td><code>{seg.selection_reason}</code></td></tr>"
        for i, seg in enumerate(res.segments))

    # 残差曲线用内联 SVG（零依赖、自包含、可放大）
    svg = _sparkline_svg(res, width=900, height=120)

    # 片段为空时要**明确说出来**，不要留一张空表让人猜是"没跑"还是"没选到"。
    if res.segments:
        seg_block = (
            "<table>\n"
            "<tr><th>#</th><th>起点(s)</th><th>终点(s)</th><th>时长(s)</th>"
            "<th>帧数</th><th>均值 score</th><th>峰值 score</th>"
            "<th>选择依据</th></tr>\n"
            + rows + "\n</table>")
    else:
        seg_block = (
            "<p><em>本次没有选出任何片段。</em>这通常意味着整段录制里没有"
            "出现「设备自身运动解释不了」的变化（例如全程静止、或全程匀速"
            "运动）。它<strong>不</strong>表示数据有问题。</p>")

    # 如实报出实际出现的 selection_reason 取值（机器可校验，也便于人核对）
    reasons = sorted({s.selection_reason for s in res.segments})
    reasons_str = ", ".join(f"<code>{r}</code>" for r in reasons) or "（无）"

    html = f"""<!DOCTYPE html>
<html lang="zh"><meta charset="utf-8">
<title>CMCI 过滤结果 · {s['source']}</title>
<style>
 body{{font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       margin:2rem auto;max-width:960px;color:#222}}
 h1{{font-size:1.3rem}} table{{border-collapse:collapse;width:100%;margin:1rem 0}}
 th,td{{border-bottom:1px solid #e5e5e5;padding:.4rem .6rem;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px;font-size:.85em}}
 .kpi{{display:flex;gap:2rem;flex-wrap:wrap;margin:1rem 0}}
 .kpi div{{background:#fafafa;border:1px solid #eee;border-radius:6px;
          padding:.6rem 1rem;min-width:8rem}}
 .kpi b{{display:block;font-size:1.3rem}}
 .note{{background:#fffbe6;border-left:3px solid #f0c000;padding:.7rem 1rem;
        margin:1rem 0;font-size:.92em}}
</style>
<h1>CMCI 过滤结果</h1>
<div class="kpi">
 <div><b>{s['n_frames']}</b>帧</div>
 <div><b>{s['duration_s']:.1f}s</b>时长</div>
 <div><b>{s['n_segments']}</b>片段</div>
 <div><b>{s['keep_fraction']*100:.1f}%</b>帧级保留</div>
 <div><b>{s['stored_seconds']:.1f}s</b>片段总时长</div>
</div>
<div class="note">
 <b>选择依据说明。</b> 本结果由<strong>第 1 层「运动去重」</strong>产生：
 被选中的是<em>设备自身运动预测不了</em>的帧。这不等于"语义重要"——
 后者在本项目的真实数据上已被否证。如需按业务规则筛选，请在引擎里传
 <code>rule=</code> 钩子（片段会标成 <code>rule:&lt;name&gt;</code>）。
 <br>本页片段实际取值（<code>selection_reason</code>）: {reasons_str}
</div>
<h2>残差曲线与片段区间</h2>
{svg}
<h2>片段</h2>
{seg_block}
<p><small>源: <code>{s['source']}</code> · 前端: <code>{res.meta.get('frontend')}</code>
 · 生成自 cerebellum_cancel.engine</small></p>
</html>
"""
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(html)
    return path


def _sparkline_svg(res: CMCIResult, width: int, height: int) -> str:
    """把 surprise 曲线与 keep 区间画成内联 SVG（不引外部图表库）。"""
    n = res.n_frames
    if n < 2:
        return "<p>帧数不足，无法绘图。</p>"
    y = res.surprise.astype(np.float64)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    pts = " ".join(f"{i / (n - 1) * width:.1f},{height - v * height:.1f}"
                   for i, v in enumerate(y))

    # keep 区间画成底部色带，和曲线对齐同一时间轴
    bands = []
    i = 0
    while i < n:
        if res.keep[i]:
            j = i
            while j + 1 < n and res.keep[j + 1]:
                j += 1
            x0 = i / (n - 1) * width
            x1 = j / (n - 1) * width
            bands.append(f'<rect x="{x0:.1f}" y="{height - 8}" '
                         f'width="{max(x1 - x0, 1):.1f}" height="8" '
                         f'fill="#2a7" opacity="0.55"/>')
            i = j + 1
        else:
            i += 1

    return (f'<svg viewBox="0 0 {width} {height + 10}" width="100%" '
            f'style="background:#fcfcfc;border:1px solid #eee">'
            f'<polyline points="{pts}" fill="none" stroke="#c33" '
            f'stroke-width="1.2"/>'
            + "".join(bands) +
            f'<text x="4" y="12" font-size="11" fill="#888">surprise</text>'
            f'<text x="{width - 90}" y="{height + 8}" font-size="11" '
            f'fill="#2a7">keep 区间</text></svg>')


_EXPORTERS: dict[str, Callable[..., str]] = {
    "dataset": _export_dataset,
    "html": _export_html,
    "parquet": _export_parquet,
    "jsonl": _export_jsonl,
}

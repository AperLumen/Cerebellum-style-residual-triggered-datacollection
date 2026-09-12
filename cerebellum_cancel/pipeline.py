"""End-to-end pipeline: video+IMU -> surprise-triggered clips + traces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import ConfigError, NonFiniteError, ShapeError, check_positive
from .core import CerebellumCancel, SurpriseTrigger, TriggerConfig
from .io import VideoStream, cells_of, compensate_rotation, gyro_at, load_imu_csv


@dataclass
class PipelineResult:
    clips: list[tuple[float, float]] = field(default_factory=list)
    surprise: np.ndarray | None = None      # (N,) frame-level surprise
    surprise_std: np.ndarray | None = None  # (N,)
    mean_cell_residual: float = 0.0
    n_frames: int = 0


def run_pipeline(video_path: str, imu_path: str | None,
                 cancel: CerebellumCancel | None = None,
                 trigger_cfg: TriggerConfig | None = None,
                 imu_offset: float = 0.0,
                 fps: float | None = None,
                 mirror: bool = False,
                 crop: tuple[float, ...] | None = None) -> PipelineResult:
    """Run the cerebellum-style cancellation pipeline over a recording.

    imu_offset: seconds added to IMU timestamps to align with the video clock
    (determined once at capture time via shake-to-sync, see README).
    fps: override stream fps (image-sequence sources often misreport it).
    mirror: the view is horizontally flipped (e.g. mirror in front of the
    camera) — negate yaw before compensation.
    crop: fractional (x0, y0, x1, y1) region of the source frame to use.

    Raises:
        ConfigError: ``fps <= 0``; ``crop`` 不是 4 元组、不在 [0,1] 内、
            或 ``x1 <= x0`` / ``y1 <= y0``（反序会让 OpenCV 抛内部错误，
            信息不可用）。
        NonFiniteError: ``imu_offset`` 非有限值。**这是一条静默失败**：
            ``imu_offset=NaN`` 会把整条 IMU 时间轴变成 NaN，``gyro_at``
            于是匹配不到任何帧，补偿**静默退化为 naive**——读数和"这段没接
            IMU"完全一样，看不出是标定参数写错了。
        InsufficientDataError: 输入路径不含任何可读帧。

    注：``fps`` 缺省（None）时沿用流自身的 fps，不做校验——流可能误报，
    这正是本参数存在的原因。
    """
    # --- 参数校验（在任何 IO 之前，避免"读了半天才报错"）---
    if fps is not None:
        # 复用 contracts 的正数检查：fps=0 会让帧时间戳与 clip 边界全部退化
        # （实测静默通过且 n_frames 不变，clips 数却从 4 变 1）。
        check_positive(fps, "fps")
    if not np.isfinite(imu_offset):
        raise NonFiniteError(
            f"imu_offset: 必须是有限标量，实得 {imu_offset!r}。"
            "NaN 会把 IMU 时间轴整体变成 NaN，匹配不到任何帧，"
            "补偿会**静默退化为 naive**——读数与没接 IMU 无法区分。")
    if crop is not None:
        if len(crop) != 4:
            raise ShapeError(
                f"crop: 期望 4 元组 (x0, y0, x1, y1)，实得 {len(crop)} 个值")
        x0, y0, x1, y1 = (float(v) for v in crop)
        if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
            raise NonFiniteError(f"crop: 含 NaN/Inf，实得 {crop!r}")
        for name, lo, hi in (("x0", x0, x1), ("y0", y0, y1)):
            if hi <= lo:
                raise ConfigError(
                    f"crop: {name} / 上界反序或相等（{lo} -> {hi}）。"
                    "反序会让 OpenCV 抛内部错误，故在此提前拒绝。")
        if not (0.0 <= x0 < 1.0 and 0.0 <= y0 < 1.0
                and 0.0 < x1 <= 1.0 and 0.0 < y1 <= 1.0):
            raise ConfigError(
                f"crop: 必须是 [0,1] 内的分数区域，实得 {crop!r}。"
                "越界（如 (0,0,2,2)）此前被静默接受。")

    cancel = cancel or CerebellumCancel()
    stream = VideoStream(video_path, fps=fps, crop=crop)
    imu = load_imu_csv(imu_path) if imu_path else []
    for s in imu:
        s.t += imu_offset
    trig = SurpriseTrigger(trigger_cfg or TriggerConfig(), stream.fps)

    result = PipelineResult()
    surprise_trace, std_trace = [], []
    prev_gray, t, hint = None, 0.0, 0
    resid_sum, n_frames = 0.0, 0

    while True:
        gray = stream.read_gray()
        if gray is None:
            break
        gyro, hint = gyro_at(imu, hint, t)
        if mirror:
            gyro = np.array(gyro, dtype=np.float64)
            gyro[1] = -gyro[1]          # mirrored view flips yaw direction
        resid = compensate_rotation(prev_gray, gray, gyro, stream.frame_dt)
        out = cancel.process(cells_of(resid, cancel.grid), gyro, stream.frame_dt)
        if cancel.state.get("n", 0) > cancel.warmup_frames:
            trig.update(t, out["surprise"], out["surprise_std"])
        surprise_trace.append(out["surprise"])
        std_trace.append(out["surprise_std"])
        resid_sum += resid.mean()
        prev_gray = gray
        t += stream.frame_dt
        n_frames += 1

    result.clips = trig.finalize(t)
    result.surprise = np.array(surprise_trace)
    result.surprise_std = np.array(std_trace)
    result.mean_cell_residual = resid_sum / max(n_frames, 1)
    result.n_frames = n_frames
    return result

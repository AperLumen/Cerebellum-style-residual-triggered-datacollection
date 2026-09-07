"""End-to-end pipeline: video+IMU -> surprise-triggered clips + traces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
                 crop: tuple[float, float, float, float] | None = None) -> PipelineResult:
    """Run the cerebellum-style cancellation pipeline over a recording.

    imu_offset: seconds added to IMU timestamps to align with the video clock
    (determined once at capture time via shake-to-sync, see README).
    fps: override stream fps (image-sequence sources often misreport it).
    mirror: the view is horizontally flipped (e.g. mirror in front of the
    camera) — negate yaw before compensation.
    crop: fractional (x0, y0, x1, y1) region of the source frame to use.
    """
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

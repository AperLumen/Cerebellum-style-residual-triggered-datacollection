"""Baselines and evaluation metrics for residual-triggered collection.

Evaluation contract: the experimenter marks ground-truth "important events"
as (start, end) second-intervals in the video clock. We then measure, per
method, how well the *saved clips* cover the events under a storage budget.

Metrics:
- recall: fraction of events with at least one saved frame inside the window
- precision: fraction of saved clips that overlap at least one event
- stored_fraction: fraction of total stream duration actually saved
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .core import TriggerConfig
from .io import VideoStream


@dataclass
class EvalResult:
    method: str
    recall: float
    precision: float
    stored_fraction: float
    n_clips: int = 0

    def as_dict(self):
        return self.__dict__.copy()


def clip_frames(clips, fps):
    frames = set()
    for a, b in clips:
        frames.update(range(int(a * fps), int(b * fps) + 1))
    return frames


def evaluate(method: str, clips, events, duration_s: float, fps: float,
             tail_pad: float = 0.5) -> EvalResult:
    """Events get ``tail_pad`` seconds of grace after their end: an object's
    disappearance (which the two-sided residual catches) belongs to the event
    even if it lands just outside the annotated window."""
    padded = [(a, b + tail_pad) for a, b in events]
    frames = clip_frames(clips, fps)
    stored = sum(b - a for a, b in clips)
    ev_hit = sum(1 for a, b in padded
                 if any(a * fps <= f <= b * fps for f in frames))
    clips_hit = sum(1 for a, b in clips
                    if any(not (b < ea or a > eb) for ea, eb in padded))
    return EvalResult(
        method=method,
        recall=ev_hit / max(len(events), 1),
        precision=clips_hit / max(len(clips), 1),
        stored_fraction=stored / max(duration_s, 1e-9),
        n_clips=len(clips),
    )


def stream_duration_and_fps(video_path: str,
                            fps_override: float | None = None
                            ) -> tuple[float, float]:
    cap = cv2.VideoCapture(video_path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        # image-sequence sources often report no frame count: count files
        import glob
        frames = glob.glob(video_path.replace("%05d", "*"))
        n = len(frames)
    if n <= 0:
        raise RuntimeError(f"cannot determine frame count for {video_path}")
    return n / fps, fps


def uniform_baseline(duration_s: float, budget: float) -> list[tuple[float, float]]:
    """Save ``budget`` (fraction of duration) as evenly spread 1 s clips."""
    n_clips = max(int(budget * duration_s), 1)
    spacing = duration_s / n_clips
    return [(i * spacing, min(i * spacing + 1.0, duration_s))
            for i in range(n_clips)]


def naive_diff_baseline(video_path: str, cfg: TriggerConfig,
                        quantile: float = 0.995) -> list[tuple[float, float]]:
    """Naive frame-difference trigger: threshold = fixed dataset-level quantile
    of |frame - prev|. No gyro compensation, no plasticity — expected to drown
    in self-motion."""
    stream = VideoStream(video_path)
    prev, diffs = None, []
    while True:
        gray = stream.read_gray()
        if gray is None:
            break
        if prev is not None:
            diffs.append(float(np.abs(gray.astype(np.float64)
                                      - prev.astype(np.float64)).mean() / 255.0))
        prev = gray
    thr = float(np.quantile(diffs, quantile)) if diffs else 1.0
    clips, on, start, last_hit, t_end = [], False, 0.0, 0.0, 0.0
    for i, d in enumerate(diffs):
        t = (i + 1) * stream.frame_dt
        t_end = t
        if not on and d > thr:
            on, start, last_hit = True, t, t
        elif on:
            if d > thr:
                last_hit = t
            if t - last_hit > cfg.post_roll:
                on = False
                clips.append((start, last_hit))
    if on:
        clips.append((start, t_end))
    return [(max(0.0, a - cfg.pre_roll), min(t_end, b + cfg.post_roll))
            for a, b in clips]

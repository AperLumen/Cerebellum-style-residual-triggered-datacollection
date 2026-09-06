"""I/O front-end: synchronized video + IMU stream.

Supports two sources:

1. Real capture from an iPhone: record video (Camera app) and IMU with a
   logging app (e.g. Sensor Logger exports CSV with
   ``time,gyroscope.x,gyroscope.y,gyroscope.z``), then sync by aligning the
   first IMU timestamp after manual "shake to sync" (see README).
2. Synthetic rig (scripts/make_synthetic.py) which produces the same formats.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ImuSample:
    t: float            # seconds, stream-local clock
    gyro: np.ndarray    # deg/s, [x, y, z]


def load_imu_csv(path: str) -> list[ImuSample]:
    """Load Sensor-Logger-style CSV. Accepts header names among common variants."""
    out = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        fields = {f.lower(): f for f in (rdr.fieldnames or [])}
        t_key = next((fields[k] for k in fields if k in
                      ("time", "timestamp", "time_stamp", "seconds")), None)
        gkeys = [fields.get(f"gyroscope.{a}") or fields.get(f"gyro_{a}")
                 or fields.get(f"gyro{a}") for a in ("x", "y", "z")]
        if t_key is None or any(k is None for k in gkeys):
            raise ValueError(f"unrecognized IMU CSV columns: {rdr.fieldnames}")
        for row in rdr:
            try:
                out.append(ImuSample(float(row[t_key]),
                                     np.array([float(row[k]) for k in gkeys])))
            except (ValueError, KeyError):
                continue
    return out


class VideoStream:
    def __init__(self, path: str, downsize: tuple[int, int] = (320, 240),
                 fps: float | None = None,
                 crop: tuple[float, float, float, float] | None = None):
        """crop: fractional (x0, y0, x1, y1) of the source frame, applied
        before downsizing — e.g. to cut away desktop margins around a
        windowed scene."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.downsize = downsize
        self.crop = crop

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    def read_gray(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.crop:
            h, w = g.shape
            x0, y0, x1, y1 = self.crop
            g = g[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        g = cv2.resize(g, self.downsize, interpolation=cv2.INTER_AREA)
        # mild blur: suppresses codec/interpolation noise on high-frequency
        # texture while barely affecting the low-frequency structure that
        # real "surprise" events are made of
        return cv2.GaussianBlur(g, (3, 3), 0)


def gyro_at(imu: list[ImuSample], idx_hint: int, t: float) -> tuple[np.ndarray, int]:
    """Linear-interpolated gyro lookup at time t; returns (gyro, hint) with
    advancing hint. Interpolation matters when IMU and video clocks differ
    (e.g. 30 Hz IMU, 31 fps video): nearest-sample error is a frame-quantized
    shift that shows up as residual noise."""
    if not imu:
        return np.zeros(3), idx_hint
    i = min(max(idx_hint, 0), len(imu) - 1)
    while i + 1 < len(imu) and imu[i + 1].t <= t:
        i += 1
    while i > 0 and imu[i].t > t:
        i -= 1
    if i + 1 < len(imu):
        t0, t1 = imu[i].t, imu[i + 1].t
        if t1 > t0:
            a = (t - t0) / (t1 - t0)
            return imu[i].gyro * (1 - a) + imu[i + 1].gyro * a, i
    return imu[i].gyro, i


def compensate_rotation(prev_gray: np.ndarray | None, gray: np.ndarray,
                        gyro, dt: float, fov_deg: float = 60.0) -> np.ndarray:
    """Warp the previous frame by the gyro-measured rotation and return the
    per-pixel residual (|current - predicted|).

    Approximates rotation flow by a 2D shift:  shift = f * tan(omega * dt).
    """
    if prev_gray is None:
        return np.zeros_like(gray, dtype=np.float64)
    h, w = gray.shape
    f = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    dx = -f * np.tan(np.radians(gyro[1]) * dt)   # yaw   -> x shift
    dy = -f * np.tan(np.radians(gyro[0]) * dt)   # pitch -> y shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(prev_gray, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return np.abs(gray.astype(np.float64) - warped.astype(np.float64)) / 255.0


def cells_of(residual: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Mean residual per grid cell -> (gx*gy,) vector."""
    gx, gy = grid
    h, w = residual.shape
    cell_h, cell_w = h // gy, w // gx
    r = residual[: cell_h * gy, : cell_w * gx]
    return r.reshape(gy, cell_h, gx, cell_w).mean(axis=(1, 3)).ravel()

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
import sys
from dataclasses import dataclass

import cv2
import numpy as np

from .contracts import (ConfigError, InsufficientDataError, NonFiniteError,
                        ShapeError, check_1d, check_finite_array,
                        check_scalar)


@dataclass
class ImuSample:
    t: float            # seconds, stream-local clock
    gyro: np.ndarray    # deg/s, [x, y, z]


def load_imu_csv(path: str, *, on_bad_row: str = "warn") -> list[ImuSample]:
    """Load Sensor-Logger-style CSV. Accepts header names among common variants.

    Args:
        on_bad_row: 遇到解析不了的行怎么办。
            ``"warn"``（默认）跳过并**在 stderr 打出计数**；
            ``"raise"`` 直接报错；
            ``"skip"`` 静默跳过（**不推荐**——实测原来就是静默的，
                5 行数据里 2 行坏掉会被无声吞掉，事后无从察觉）。

    Raises:
        ValueError: 表头无法识别，或 ``on_bad_row="raise"`` 时遇到坏行。
        NonFiniteError: CSV 里有 NaN/Inf。**必须显式失败**：NaN 一路传下去会
            污染在线统计（与 core/rotwarp 同一失效模式），而它在 CSV 里
            看起来只是一串字符。
    """
    out = []
    bad = 0
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
                vals = np.array([float(row[k]) for k in gkeys], dtype=np.float64)
                ts = float(row[t_key])
            except (ValueError, KeyError, TypeError):
                bad += 1
                if on_bad_row == "raise":
                    raise ValueError(
                        f"IMU CSV 有无法解析的行（第 {rdr.line_num} 行）: "
                        f"{row}")
                continue
            if not (np.isfinite(ts) and np.all(np.isfinite(vals))):
                raise NonFiniteError(
                    f"IMU CSV 第 {rdr.line_num} 行含 NaN/Inf（t={ts}）。"
                    "它会一路传进在线统计并让后续所有帧失效——"
                    "请在源头清洗，或显式声明该段不可用。")
            out.append(ImuSample(ts, vals))
    if bad and on_bad_row != "skip":
        print(f"[io.load_imu_csv] 跳过 {bad} 行无法解析的 IMU 数据"
              f"（文件 {path}）。若不希望跳过，用 on_bad_row='raise'。",
              file=sys.stderr)
    if not out:
        # 表头对但一条数据都没有 —— 此前静默返回空列表，调用方只会看到
        # "IMU 是空的"，无从判断是文件坏了还是真的没数据。必须说出来。
        raise InsufficientDataError(
            f"IMU CSV 里没有任何可用数据行（表头已识别，但 0 条记录）：{path}。"
            "常见原因：导出时只写了表头、或所有行都解析失败。")
    if any(out[i + 1].t < out[i].t for i in range(len(out) - 1)):
        # 时间戳不单调会让 gyro_at 的线性插值给出无意义的读数（实测会外推出
        # 负值）。这不一定是错误数据，但**必须让人知道**。
        print(f"[io.load_imu_csv] 警告：{path} 的时间戳不是单调递增的，"
              "插值结果可能无意义。", file=sys.stderr)
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
    shift that shows up as residual noise.

    Raises:
        NonFiniteError: ``t`` 非有限值。**必须拒绝**：NaN 会让下面所有比较
            都为假，函数会静默走到最后返回一个样本，而调用方完全看不出
            这不是插值结果（实测原行为返回 ``[nan nan nan]``）。
        ConfigError: ``imu`` 为空。原行为返回全零 = **假装设备静止**，
            正是本项目要避免的失效模式（GUIDE §0.2）。空 IMU 应当由调用方
            决定怎么办（引擎里是显式的 no_imu 三选一），不该在这里被悄悄
            糊成一个读数。

    注：``t`` 落在样本区间**之外**时不做外推，直接夹到最近的样本——原实现
    会线性外推出负角速度（实测 t 早于全部样本时得到 ``[-14,-13,-12]``），
    那是凭空造数据。
    """
    if not imu:
        raise ConfigError(
            "gyro_at: IMU 列表为空。返回全零等于假设设备静止——"
            "请由调用方显式决定降级方式（见 engine 的 no_imu 选项）。")
    t = check_scalar(t, "t", "gyro_at")
    i = min(max(idx_hint, 0), len(imu) - 1)
    while i + 1 < len(imu) and imu[i + 1].t <= t:
        i += 1
    while i > 0 and imu[i].t > t:
        i -= 1
    # 只在 t 落于 [imu[i], imu[i+1]] 内部时插值；不要外推
    if i + 1 < len(imu) and imu[i].t <= t <= imu[i + 1].t:
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

    Raises:
        NonFiniteError: ``gyro`` 或 ``dt`` 含 NaN/Inf。**这是本函数最危险的
            静默失败**：NaN 会产生一个形状正确、数值全 NaN 的残差图，
            下游只看到"残差不对"，不知道根源在 I/O 层。
        ShapeError: 两帧尺寸不一致。原实现会静默返回**按 gray 尺寸**的结果，
            而内部 warp 用的是 prev_gray 的尺寸——等于悄悄错位。

    注：``gyro`` 只取前 3 轴；``prev_gray is None``（首帧）返回全零残差，
    这是**有意**的（首帧没有可比的前一帧）。
    """
    dt = check_scalar(dt, "dt", "compensate_rotation")
    g = check_1d(gyro, "gyro")
    if g.size < 3:
        raise ShapeError(
            f"compensate_rotation: gyro 至少需要 3 轴，实得 {g.shape}")
    if not np.all(np.isfinite(g[:3])):
        raise NonFiniteError(
            "compensate_rotation: gyro 含 NaN/Inf。会产出形状正常但全 NaN 的"
            "残差图，下游无法定位根源。")
    # **fuzz 发现（2026-09-12）**：原实现对 object-dtype / 字符串数组直接做
    # `prev_gray.shape` 判等后送进 cv2，抛的是原生 TypeError/ValueError。
    # 统一经契约函数归一化（顺带拒绝 NaN，cv2 遇 NaN 会给内部错误）。
    gray = check_finite_array(gray, "gray")
    if prev_gray is None:
        return np.zeros_like(gray, dtype=np.float64)
    prev_gray = check_finite_array(prev_gray, "prev_gray")
    if prev_gray.shape != gray.shape:
        raise ShapeError(
            f"compensate_rotation: 两帧尺寸必须一致，"
            f"实得 {prev_gray.shape} vs {gray.shape}")

    h, w = gray.shape
    f = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    dx = -f * np.tan(np.radians(g[1]) * dt)   # yaw   -> x shift
    dy = -f * np.tan(np.radians(g[0]) * dt)   # pitch -> y shift
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    warped = cv2.warpAffine(prev_gray, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return np.abs(gray.astype(np.float64) - warped.astype(np.float64)) / 255.0


def cells_of(residual: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Mean residual per grid cell -> (gx*gy,) vector.

    Raises:
        ShapeError: ``residual`` 不是 2 维，或某个方向比网格还小（此时
            ``cell_h`` 或 ``cell_w`` 为 0，会切出空数组并产出全 NaN——
            原实现正是如此，还伴随一条容易被忽略的 RuntimeWarning）。
        NonFiniteError: ``residual`` 含 NaN/Inf，**或分格求均值时溢出**。

    **fuzz 发现（2026-09-13，随机模式）**：输入 ``1e308`` 批量值能通过
    有限性检查（它确实是有限值），但格内 49 个 1e308 求和**溢出成 Inf**，
    于是静默返回含 Inf 的网格——与 ``rotation_homography`` 同一个教训：
    **光查输入有限不够，必须查结果**。溢出不报错，下游只会看到"残差不对"。
    """
    gx, gy = int(grid[0]), int(grid[1])
    if gx <= 0 or gy <= 0:
        raise ConfigError(f"cells_of: 网格必须为正，实得 {grid!r}")
    r = check_finite_array(residual, "residual")
    if r.ndim != 2:
        raise ShapeError(
            f"cells_of: 期望 2 维残差图 (H,W)，实得 shape={r.shape}")
    h, w = r.shape
    if h < gy or w < gx:
        raise ShapeError(
            f"cells_of: 残差图 {r.shape} 小于网格 {grid}（每个格至少需要 1 "
            "像素）。此前会切出空数组并静默产出全 NaN。")
    cell_h, cell_w = h // gy, w // gx
    rr = r[: cell_h * gy, : cell_w * gx]
    # 用 errstate 抑制 numpy 的 RuntimeWarning——我们**不忽略**溢出，
    # 而是显式检查结果并抛出契约异常（警告容易被无视，异常不会）。
    with np.errstate(over="ignore", invalid="ignore"):
        out = rr.reshape(gy, cell_h, gx, cell_w).mean(axis=(1, 3)).ravel()
    if not np.all(np.isfinite(out)):
        bad = int(np.count_nonzero(~np.isfinite(out)))
        mx = float(np.nanmax(np.abs(r))) if r.size else 0.0
        raise NonFiniteError(
            f"cells_of: 分格均值溢出（{bad}/{out.size} 个格非有限）。"
            f"输入最大绝对值 {mx:.3g} 是有限值，但格内求和在 float64 下"
            "溢出成 Inf。请先归一化残差，或改用更稳健的聚合（如分位数）。")
    return out

"""Gyro-aligned feature-map cancellation — the spatially structured version of
the ELL negative image, and the honest fix for the pooled-feature adapter's
failure mode (a linear map on pooled features cannot express "content shifted
by k px", so ego-motion leaked into the residual).

Pipeline per frame, no training anywhere:
  1. predict  F_hat_t(x) = F_{t-1}(x - dx_t)   with dx_t from the IMU gyro
     (corollary discharge), converted to feature-map pixels
  2. residual  r_t = F_t - F_hat_t
  3. running per-cell mean/var over CALM frames only (equilibrium-rate gate:
     a currently-surprising frame updates nothing), z-score
  4. surprise readout: pooled z-energy over interior cells

The only learned-looking quantities (running stats) are local, causal, and
calibrate themselves within the first frames of any sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CancelConfig:
    fps: float = 30.0
    f_map: float = 24.25     # focal length in map px: (W/2)/tan(fov/2) * map_w/W
    calm_tau: float = 4.0    # surprise above this blocks stat updates
    std_ema: float = 0.02    # running-statistics rate
    warmup: int = 10         # frames before any update
    border: int = 2          # cells excluded from readout (roll wrap-around)
    use_warp: bool = True    # False = naive frame-diff ablation
    robust: bool = False     # True = winsorized always-update stats
                             # (the gated EMA freezes when surprise is dense —
                             #  measured on real video: 96% of frames blocked —
                             #  winsorizing the stat update at `winsor` sigma
                             #  makes outliers unable to inflate or freeze it)
    winsor: float = 3.0      # clip (in sd units) applied to the stat update
    pred: str = "shift"      # "shift" = 管线原有：gyro -> (dx,dy) 的 tan 近似
                             # 常数平移；
                             # "rotation" = 完整旋转矩阵单应（含 roll）。
                             # 真机 IMU 实测（MEMORY_FORMAT §8）：同一份 gyro
                             # 下 shift 让残差 1.31x（帮倒忙）、rotation 降到
                             # 0.84x；差异来自 roll 未建模 + 常数平移近似在
                             # 视野边缘失效。默认保持 "shift" 以不改变已发布
                             # 数字，需要时显式开启。
    yaw_sign: float = 0.0    # rotation 模式的轴符号；0 = 按数据集默认
                             # （rotwarp.signs_for）。两数据源约定相反（真机
                             # IMU 为 +1/+1/+1、AoE 为 -1/+1/+1），显式给值
                             # 可覆盖；见 rotwarp.DATASET_CONVENTIONS 与
                             # MEMORY_FORMAT §8.7。**符号与
                             # warp_by_rotation 的 WARP_INVERSE_MAP 是联合
                             # 约定，改一边必须同步另一边。**
    pitch_sign: float = 0.0
    roll_sign: float = 0.0
    gyro_lag: int = -1      # rotation 模式预测 F_t 时取哪一帧的读数：
                            # 0 = gyro[t]（存储值描述 t-1 -> t 位移），
                            # 1 = gyro[t-1]（存储值描述 t -> t+1 位移）。
                            # -1 = 按数据集默认（rotwarp.lag_for）。
                            # 两个数据源确实相反，不能全局取值：真机 IMU 的
                            # prep 用当前帧标定 -> lag=1（实测 0.835 vs
                            # 0.906）；AoE 的 prep 存的就是 (t-1 -> t) 转角
                            # -> lag=0（12 段独立标定 12/12 一致）。
    trans_scale: float = 0.0 # effective 1/Z (1/m) for the translation term;
                             # 0 disables it. With a camera-frame translation
                             # velocity v (m/s), the content shift is
                             # f_map * trans_scale * v * dt, same sign
                             # convention as the rotation term (prep folds
                             # per-axis signs into v).
    source: str = ""         # 数据来源路径/标识（如 "data/aoe_event/val/seq_000"
                             # 或 "real_imu"）。仅用于在 yaw_sign/pitch_sign/
                             # roll_sign/gyro_lag 未显式指定时按数据集惯例
                             # 取默认（rotwarp.convention_for）。留空则用
                             # 存储原样 + lag=0。


def shift_map(F: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Subpixel translation G(x, y) -> G(x - dx, y - dy), bilinear via
    roll+blend, vectorized over the channel axis."""
    def axis_shift(G, d, ax):
        d_int = int(np.floor(d))
        f = d - d_int
        return ((1 - f) * np.roll(G, d_int, axis=ax)
                + f * np.roll(G, d_int + 1, axis=ax))
    return axis_shift(axis_shift(F, dx, -1), dy, -2)


def expected_shift(gyro: np.ndarray, cfg: CancelConfig,
                   trans: np.ndarray | None = None) -> tuple[float, float]:
    """Frame-to-frame content shift (map px): rotation term plus an optional
    constant-depth translation term."""
    dx = cfg.f_map * np.tan(np.radians(gyro[1]) / cfg.fps)   # yaw   -> x
    dy = cfg.f_map * np.tan(np.radians(gyro[0]) / cfg.fps)   # pitch -> y
    if trans is not None and cfg.trans_scale > 0:
        dx += cfg.trans_scale * cfg.f_map * trans[0] / cfg.fps
        dy += cfg.trans_scale * cfg.f_map * trans[1] / cfg.fps
    return dx, dy


class FeatureMapCancel:
    def __init__(self, cfg: CancelConfig):
        self.cfg = cfg
        # 数据源惯例（符号 + gyro 配对）只在调用方没显式给值时生效。
        from .rotwarp import convention_for
        conv = convention_for(cfg.source)
        self._default_signs: tuple[float, float, float] = conv["signs"]  # type: ignore[assignment]
        self._default_lag: int = conv["lag"]  # type: ignore[assignment]
        self.reset_online()

    def reset_online(self):
        self.F_prev: np.ndarray | None = None
        self.gyro_prev: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.n = 0

    def dump_stats(self) -> dict:
        """导出在线状态（mu/var/n + F_prev）供落盘或跨录制复用（warm start）。

        协议层的真实使用场景是"关掉再打开"：一次录制结束把状态存下来，
        下一次开机直接接着用。**必须连 F_prev 一起存**——只存 mu/var 的话，
        重启后的第一帧没有可 warp 的上一帧特征图（``step`` 里 pred = F），
        残差恒为 0、surprise 恒为 0，统计收敛问题解决了但第一帧仍盲。
        F_prev 是一次性开销（~C×H×W），相对逐帧载荷可忽略。

        未跑过任何帧时返回空 dict。
        """
        if self.mu is None or self.var is None:
            return {}
        out = {"mu": self.mu, "var": self.var, "n": self.n}
        if self.F_prev is not None:
            out["F_prev"] = self.F_prev
        if self.gyro_prev is not None:
            # "rotation" 模式续采需要它：否则重启后首帧退化为用当帧 gyro
            out["gyro_prev"] = self.gyro_prev
        return out

    def load_stats(self, stats: dict) -> None:
        """载入 ``dump_stats()`` 的产物（warm start）。

        校验形状：统计必须来自同一前端/同一特征图网格，否则触发读数会
        在两个不相容的尺度之间漂移（比不载入更糟），直接拒绝。
        """
        mu = stats.get("mu")
        var = stats.get("var")
        if mu is None or var is None:
            return
        mu = np.asarray(mu, dtype=np.float32)
        var = np.asarray(var, dtype=np.float32)
        if self.mu is not None and self.mu.shape != mu.shape:
            raise ValueError(
                f"统计形状不符: 当前网格 {self.mu.shape} vs 载入 {mu.shape}"
                "（前端或输入分辨率变了，不能续用旧统计）")
        f_prev = stats.get("F_prev")
        if f_prev is not None:
            f_prev = np.asarray(f_prev, dtype=np.float32)
            if f_prev.shape != mu.shape:
                raise ValueError(
                    f"F_prev 形状不符: {f_prev.shape} vs 统计 {mu.shape}")
        self.mu, self.var = mu.copy(), var.copy()
        self.F_prev = None if f_prev is None else f_prev.copy()
        g_prev = stats.get("gyro_prev")
        if g_prev is not None:
            g_prev = np.asarray(g_prev, dtype=np.float64).ravel()
            if g_prev.size != 3:
                raise ValueError(f"gyro_prev 应为 3 维，实得 {g_prev.size}")
            self.gyro_prev = g_prev.copy()
        self.n = int(stats.get("n", 0))

    def step(self, F: np.ndarray, gyro: np.ndarray,
             trans: np.ndarray | None = None) -> dict:
        F = F.astype(np.float32)
        if self.F_prev is None:
            pred = F
        elif self.cfg.use_warp and self.cfg.pred == "rotation":
            # 完整旋转矩阵单应（含 roll）——真机 IMU 实测比 shift 近似好得多
            # （MEMORY_FORMAT §8），但默认关闭以不改动已发布数字。
            #
            # 时序约定：预测 F_t 要用**描述 (F_{t-1} -> F_t) 这一步位移**的角
            # 速度。存储值描述的是哪一步、以及该取哪帧，两个数据源相反，由
            # cfg.gyro_lag 决定（-1 = 按路径默认，见 rotwarp.lag_for）：
            #   lag=0 -> gyro[t]     （AoE：prep 存的就是 t-1 -> t 转角）
            #   lag=1 -> gyro[t-1]   （真机 IMU：prep 用当前帧读数标定）
            # 这里用自己保存的上一帧 gyro，因此只支持 lag<=1；这个参数**不是**
            # 可全局统一的调优项，两个数据集实测结论相反（§8.7）。
            from .rotwarp import warp_by_rotation, lag_for
            lag = self.cfg.gyro_lag
            if lag < 0:
                lag = self._default_lag
            g = gyro if lag == 0 else (
                self.gyro_prev if self.gyro_prev is not None else gyro)
            ys = self.cfg.yaw_sign or self._default_signs[0]
            ps = self.cfg.pitch_sign or self._default_signs[1]
            rs = self.cfg.roll_sign or self._default_signs[2]
            pred = warp_by_rotation(self.F_prev, g, 1.0 / self.cfg.fps,
                                    self.cfg.f_map, yaw_sign=ys,
                                    pitch_sign=ps, roll_sign=rs)
        elif self.cfg.use_warp:
            dx, dy = expected_shift(gyro, self.cfg, trans)
            # sign verified empirically on the synthetic rig: the content
            # shift that best predicts F_t is the negated physical shift
            pred = shift_map(self.F_prev, -dx, -dy)
        else:
            pred = self.F_prev
        err = F - pred
        if self.mu is None:
            self.mu = np.zeros_like(err)
            self.var = np.ones_like(err)
        assert self.mu is not None and self.var is not None
        if self.mu.shape != err.shape:
            raise ValueError(
                f"统计形状 {self.mu.shape} 与当前特征图 {err.shape} 不符"
                "（warm start 的统计必须来自同一前端与输入分辨率）")
        sd = np.sqrt(self.var) + 1e-3
        z = err / sd
        b = self.cfg.border
        surprise = float(np.abs(z[:, b:-b, b:-b]).max())

        self.n += 1
        lr = self.cfg.std_ema
        if self.cfg.robust:
            # Winsorized always-update: outliers enter the running stats
            # clipped at `winsor` sigma, so they can neither inflate the
            # scale (raising the trigger bar) nor freeze it via a gate —
            # real video is surprising most of the time.
            d = np.clip(err - self.mu,
                        -self.cfg.winsor * sd, self.cfg.winsor * sd)
            self.mu += lr * d
            self.var += lr * (d * d - self.var)
            learned = self.n > self.cfg.warmup
        else:
            # During warmup, update unconditionally: the gate needs calibrated
            # stats to mean anything, and the first frames of a recording are
            # background by construction (events are injected later).
            calm = surprise < self.cfg.calm_tau or self.n <= self.cfg.warmup
            if calm:
                d = err - self.mu
                self.mu += lr * d
                self.var += lr * (d * d - self.var)
            learned = calm and self.n > self.cfg.warmup
        self.F_prev = F
        self.gyro_prev = np.asarray(gyro, dtype=np.float64).copy()
        return {"z": z, "surprise": surprise, "learned": learned}


def rollout(canc: FeatureMapCancel, fmaps: np.ndarray,
            gyros: np.ndarray,
            trans: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (z-maps (T,C,H,W), surprises (T,)). `trans` is an optional
    (T,3) camera-frame translation velocity for the constant-depth term."""
    T_ = len(fmaps)
    z = np.zeros_like(fmaps, dtype=np.float32)
    sur = np.zeros(T_, dtype=np.float32)
    for t in range(T_):
        tv = trans[t] if trans is not None else None
        out = canc.step(fmaps[t], gyros[t], tv)
        z[t] = out["z"]
        sur[t] = out["surprise"]
    return z, sur


def energy_score(z: np.ndarray, border: int = 2) -> np.ndarray:
    """Untrained readout: pooled squared-z over interior cells, per frame."""
    zi = z[:, :, border:-border, border:-border]
    return (zi ** 2).sum(axis=(1, 2, 3))


def maxcell_score(z: np.ndarray, border: int = 2) -> np.ndarray:
    """Untrained readout: strongest single interior cell (channel-summed z^2)."""
    s = (z ** 2).sum(axis=1)                    # (T, H, W)
    s = s[:, border:-border, border:-border]
    return s.reshape(len(s), -1).max(axis=1)

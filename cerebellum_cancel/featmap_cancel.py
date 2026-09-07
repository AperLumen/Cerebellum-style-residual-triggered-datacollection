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


def shift_map(F: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Subpixel translation G(x, y) -> G(x - dx, y - dy), bilinear via
    roll+blend, vectorized over the channel axis."""
    def axis_shift(G, d, ax):
        d_int = int(np.floor(d))
        f = d - d_int
        return ((1 - f) * np.roll(G, d_int, axis=ax)
                + f * np.roll(G, d_int + 1, axis=ax))
    return axis_shift(axis_shift(F, dx, -1), dy, -2)


def expected_shift(gyro: np.ndarray, cfg: CancelConfig) -> tuple[float, float]:
    """Frame-to-frame content shift (map px) implied by the gyro rate."""
    dx = cfg.f_map * np.tan(np.radians(gyro[1]) / cfg.fps)   # yaw   -> x
    dy = cfg.f_map * np.tan(np.radians(gyro[0]) / cfg.fps)   # pitch -> y
    return dx, dy


class FeatureMapCancel:
    def __init__(self, cfg: CancelConfig):
        self.cfg = cfg
        self.reset_online()

    def reset_online(self):
        self.F_prev: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.n = 0

    def step(self, F: np.ndarray, gyro: np.ndarray) -> dict:
        F = F.astype(np.float32)
        if self.F_prev is None:
            pred = F
        elif self.cfg.use_warp:
            dx, dy = expected_shift(gyro, self.cfg)
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
        return {"z": z, "surprise": surprise, "learned": learned}


def rollout(canc: FeatureMapCancel, fmaps: np.ndarray,
            gyros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (z-maps (T,C,H,W), surprises (T,))."""
    T_ = len(fmaps)
    z = np.zeros_like(fmaps, dtype=np.float32)
    sur = np.zeros(T_, dtype=np.float32)
    for t in range(T_):
        out = canc.step(fmaps[t], gyros[t])
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

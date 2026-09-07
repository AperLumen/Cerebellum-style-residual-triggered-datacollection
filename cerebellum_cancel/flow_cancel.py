"""Optical-flow compensation baseline: predict F_t by warping F_{t-1} along
dense DIS flow instead of the gyro-derived shift.

This is the CV-native baseline the gyro warp must justify itself against:
same cancellation contract (predict the predictable, keep the residual), but
the alignment signal comes from the image content itself — no IMU needed.
Trade-offs measured in scripts/run_flow_eval.py: per-frame cost (2-3 orders
of magnitude above a gyro shift), texture dependence, and flow corruption in
event regions (a moving blob has no stable texture to track).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FlowWarpConfig:
    flow_size: tuple[int, int] = (320, 240)   # resolution DIS runs at
    border: int = 2                            # excluded from readout
    calm_tau: float = 4.0
    std_ema: float = 0.02
    warmup: int = 10


def dis_flow_warp(dis, F_prev: np.ndarray, gray_prev: np.ndarray,
                  gray: np.ndarray) -> np.ndarray:
    """Predict F_t by warping F_{t-1} along dense DIS flow estimated from the
    raw frame pair. Module-level so both the always-on baseline
    (FlowWarpCancel) and the on-demand arbiter can share it."""
    flow = dis.calc(gray_prev, gray, None)                 # (H,W,2) px
    c, mh, mw = F_prev.shape
    fh, fw = gray.shape
    flow = cv2.resize(flow, (mw, mh),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    flow[..., 0] *= mw / fw
    flow[..., 1] *= mh / fh
    gx, gy = np.meshgrid(np.arange(mw, dtype=np.float32),
                         np.arange(mh, dtype=np.float32))
    map_x = gx - flow[..., 0]
    map_y = gy - flow[..., 1]
    src = np.ascontiguousarray(F_prev.transpose(1, 2, 0))
    warped = cv2.remap(src, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped.transpose(2, 0, 1).astype(np.float32)


def explained_ratio(err_gyro: np.ndarray, err_flow: np.ndarray,
                    border: int = 2) -> float:
    """How much of the gyro-compensated residual the flow-warped prediction
    explains: interior energy ratio ||r_flow||^2 / ||r_gyro||^2 in [0, ~1].
    Low (~<0.5) -> flow explains the residual (ego-motion misprediction);
    high (~1)   -> flow cannot explain it either (real surprise: an event
    has no correspondence in the previous frame)."""
    g = err_gyro[:, border:-border, border:-border]
    f = err_flow[:, border:-border, border:-border]
    return float((f ** 2).sum() / max(float((g ** 2).sum()), 1e-12))


class FlowWarpCancel:
    """Same step/rollout contract as FeatureMapCancel, but the prediction is
    F_prev warped along dense flow estimated from the raw frames."""

    def __init__(self, cfg: FlowWarpConfig):
        self.cfg = cfg
        self.dis = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.reset_online()

    def reset_online(self):
        self.F_prev = None
        self.gray_prev = None
        self.mu = None
        self.var = None
        self.n = 0

    def predict(self, F: np.ndarray, gray: np.ndarray) -> np.ndarray:
        if self.F_prev is None:
            return F
        return dis_flow_warp(self.dis, self.F_prev, self.gray_prev, gray)

    def step(self, F: np.ndarray, gray: np.ndarray) -> dict:
        F = F.astype(np.float32)
        gray = cv2.resize(gray, (self.cfg.flow_size[1], self.cfg.flow_size[0]),
                          interpolation=cv2.INTER_AREA)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        pred = self.predict(F, gray)
        err = F - pred
        if self.mu is None:
            self.mu = np.zeros_like(err)
            self.var = np.ones_like(err)
        sd = np.sqrt(self.var) + 1e-3
        z = err / sd
        b = self.cfg.border
        surprise = float(np.abs(z[:, b:-b, b:-b]).max())

        self.n += 1
        calm = surprise < self.cfg.calm_tau or self.n <= self.cfg.warmup
        if calm:
            d = err - self.mu
            self.mu += self.cfg.std_ema * d
            self.var += self.cfg.std_ema * (d * d - self.var)
        self.F_prev = F
        self.gray_prev = gray
        return {"z": z, "surprise": surprise,
                "learned": calm and self.n > self.cfg.warmup}


def rollout_flow(canc: FlowWarpCancel, fmaps: np.ndarray,
                 grays: list) -> tuple[np.ndarray, np.ndarray]:
    """Returns (z-maps (T,C,H,W), surprises (T,)) like featmap rollout."""
    T_ = len(fmaps)
    z = np.zeros_like(fmaps, dtype=np.float32)
    sur = np.zeros(T_, dtype=np.float32)
    for t in range(T_):
        out = canc.step(fmaps[t], grays[t])
        z[t] = out["z"]
        sur[t] = out["surprise"]
    return z, sur

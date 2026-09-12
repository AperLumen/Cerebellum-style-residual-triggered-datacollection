"""Core cancellation layer: grid-based negative image with paired fast/slow plasticity.

The frame is divided into a grid of cells. For each cell we maintain:

1. ``neg_slow`` — a slow-moving "negative image" (EMA) of the residual that is
   *not* explained by the action-conditioned predictor. It absorbs static
   scene structure, lighting gradients, and stable artifacts (slow plasticity,
   minutes-to-hours timescale).
2. ``w_fast`` / ``w_slow`` — per-cell linear weights mapping an action-feature
   vector (normed gyro rates + bias) to the expected residual. ``w_fast`` adapts
   quickly (seconds), ``w_slow`` slowly and only from ``w_fast``'s settled
   values plus a slow error signal. This mirrors the ELL circuit's pairing of
   fast and slow plasticity sites so that fast plasticity can compensate for
   slow plasticity and vice versa (Perks et al. 2026, Discussion).

Plasticity rules (all local, no backprop):

- anti-Hebbian associative:   err = residual - prediction - neg
  w_fast += lr_fast * err * feat,   w_fast decayed toward w_slow each step
  w_slow += lr_slow * err * feat
- non-associative (slow negative image): neg_slow += ema * err, updated only
  when the cell is *not* currently surprised (mirrors the ELL equilibrium
  firing-rate constraint: learn only when nothing unexpected is happening).

Surprise per cell: relu(err). Frame-level surprise = mean of per-cell surprise
normalized by the running estimate of its own std (self-calibrating threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import (ConfigError, NonFiniteError, check_1d, check_gyro,
                        check_min_frames, check_positive)


@dataclass
class TriggerConfig:
    """Hysteresis trigger on frame-level surprise."""
    threshold_k: float = 6.0        # onset: max per-cell z-score above this
    off_k: float = 2.5              # offset: z-score below this
    pre_roll: float = 0.3           # seconds saved before onset
    post_roll: float = 0.8          # seconds saved after off
    min_clip_len: float = 0.4       # seconds; shorter clips are merged
    robust: bool = False            # winsorized always-update score stats
                                    # (default keeps the released behaviour:
                                    # the z<k gate freezes the baseline when
                                    # surprise is dense, latching the gate
                                    # OPEN — measured 2/10 event segments at
                                    # stored 0.96-0.99, see
                                    # runs/diag_gate_lock.json)
    winsor: float = 3.0             # clip (in sd units) on the stat update


@dataclass
class AdaptiveTrigger:
    """Self-calibrating trigger: normalizes the raw surprise score by its own
    calm-frame running statistics before applying the (dimensionless)
    hysteresis thresholds, so the thresholds stay valid across datasets and
    motion regimes — the only calibrated quantities are the running stats
    upstream, not a hand-set score threshold.

    The score gate mirrors the per-cell equilibrium constraint: statistics
    update only on frames that are outside a clip AND below the onset
    threshold (a frame that looks like an event must not move the baseline).
    During warmup the trigger is suppressed and stats bootstrap
    unconditionally (the first frames of a recording are background by
    construction, same assumption as the featmap warmup); mu seeds from the
    first observed score so the bootstrap is scale-free.
    """

    cfg: TriggerConfig = field(default_factory=TriggerConfig)
    stat_ema: float = 0.02          # running mean/var rate of the score
    warmup_frames: int = 30         # trigger suppressed, unconditional stats
    mu: float = 0.0                 # seeded from the first observed score
    var: float = 1.0
    n: int = 0

    def __post_init__(self):
        self._inner_state: SurpriseTrigger | None = None  # created in update()

    def _stat_update(self, score: float):
        """Update the score baseline. With ``cfg.robust`` the residual enters
        winsorized at ``cfg.winsor`` sigma, so a dense burst of large scores
        cannot freeze the baseline (and therefore cannot latch the gate open);
        this mirrors ``featmap_cancel``'s robust stats one level down."""
        d = score - self.mu
        sd = float(np.sqrt(self.var)) + 1e-3 * (abs(self.mu) + 1.0)
        if self.cfg.robust:
            lim = self.cfg.winsor * sd
            d = float(np.clip(d, -lim, lim))
        self.mu += self.stat_ema * d
        self.var += self.stat_ema * (d * d - self.var)

    def update(self, t: float, score: float, fps: float) -> bool:
        """Feed one frame's raw score (e.g. maxcell z^2); returns True while
        inside a triggered clip.

        Raises:
            ConfigError: ``fps <= 0``（clip 边界以秒计，帧率为 0 会让所有
                时间换算退化）或 ``t`` 非有限值。
            NonFiniteError: ``score`` 为 NaN/Inf。这必须显式失败——一个 NaN
                会经 ``_stat_update`` 进入 ``mu``/``var``，此后**每一帧**的
                z 分数都是 NaN，触发静默失效且没有任何迹象。
        """
        if not np.isfinite(t):
            raise ConfigError(f"t: 必须是有限标量，实得 {t!r}")
        if not np.isfinite(score):
            raise NonFiniteError(
                f"score: 必须有限，实得 {score!r}。"
                "NaN/Inf 一旦进入运行统计（mu/var），此后所有帧的 z 分数都是 "
                "NaN，触发静默失效——必须在入口拦截。")
        if self._inner_state is None:
            self._inner_state = SurpriseTrigger(self.cfg,
                                                check_positive(fps, "fps"))
        self.n += 1
        if self.n == 1:
            self.mu = float(score)
        sd = float(np.sqrt(self.var)) + 1e-3 * (abs(self.mu) + 1.0)
        z = (score - self.mu) / sd
        if self.n <= self.warmup_frames:
            self._stat_update(score)
            return False
        on = self._inner_state.update(t, z, 1.0)
        if self.cfg.robust:
            # always update, winsorized: the gate can never freeze the
            # baseline, which is what latches it open under dense surprise
            self._stat_update(score)
        elif not on and z < self.cfg.threshold_k:
            self._stat_update(score)
        return on

    @property
    def clips(self):
        return self._inner_state.clips if self._inner_state else []

    def finalize(self, t_end: float):
        if self._inner_state is None:
            return []
        return self._inner_state.finalize(t_end)


@dataclass
class CerebellumCancel:
    grid: tuple[int, int] = (8, 6)          # (cols, rows) of residual cells
    fast_lr: float = 0.05                   # fast plasticity lr
    slow_lr: float = 2e-4                   # slow plasticity lr
    fast_decay: float = 0.01                # per-step pull of w_fast toward w_slow
    neg_ema: float = 5e-3                   # slow negative-image EMA rate
    std_ema: float = 0.02                   # running std of surprise EMA rate
    warmup_frames: int = 20                 # frames before plasticity engages

    state: dict = field(default_factory=dict, init=False)

    def _init_state(self):
        gx, gy = self.grid
        interior = np.zeros((gy, gx), dtype=bool)
        interior[1:-1, 1:-1] = True          # ignore border cells (warp artifacts)
        self.state = {
            "w_fast": np.zeros((gx * gy, 4), dtype=np.float64),
            "w_slow": np.zeros((gx * gy, 4), dtype=np.float64),
            "neg": np.zeros((gx * gy,), dtype=np.float64),
            "cell_mu": np.zeros((gx * gy,), dtype=np.float64),
            "cell_var": np.full((gx * gy,), 1e-2, dtype=np.float64),
            "interior": interior.ravel(),
            "n": 0,
        }

    def reset(self):
        self.state = {}

    @staticmethod
    def _features(gyro, dt: float) -> np.ndarray:
        """Action feature vector from gyro (deg/s) + bias term."""
        if dt <= 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        rate = np.abs(np.asarray(gyro, dtype=np.float64)) / 360.0  # rev/s
        return np.array([1.0, *np.clip(rate, 0.0, 2.0)])

    def process(self, residual_cells: np.ndarray, gyro, dt: float) -> dict:
        """One step of cancellation + plasticity.

        Args:
            residual_cells: (gx*gy,) non-negative residual per cell, already
                compensated for rotation by the caller (see pipeline).
            gyro: (3,) gyro rates deg/s at this frame.
            dt: seconds since previous frame. ``dt <= 0`` is **legal** and
                means "no elapsed time" — the action feature then keeps only
                its bias term (see ``_features``).

        Returns dict with per-cell and frame-level surprise.

        Raises:
            ShapeError: ``residual_cells`` 不是 1 维或长度不等于网格大小；
                ``gyro`` 不是 3 维。
            NonFiniteError: 输入含 NaN/Inf（会污染在线统计，必须显式失败）。
            ConfigError: ``dt`` 非有限值。

        契约说明：修复前 ``.ravel()`` 会把任意多维输入**静默展平**，得到
        "数值正常但语义错误"的结果；NaN 会静默污染 mu/var/neg，使后续所有帧
        的判断失效且不报错。两者都是实测到的静默失败模式。
        """
        gx, gy = self.grid
        r = check_1d(residual_cells, "residual_cells",
                     expected_len=gx * gy).astype(np.float64, copy=False)
        g = check_gyro(gyro)
        if not np.isfinite(dt):
            raise ConfigError(f"dt: 必须是有限标量，实得 {dt!r}")

        if not self.state:
            self._init_state()
        s = self.state
        feat = self._features(g, dt)

        pred = (s["w_fast"] + s["w_slow"]) @ feat          # action-conditioned expectation
        err = r - pred - s["neg"]                          # what's truly unexpected

        # per-cell self-calibrating z-score (two-sided: both appearances and
        # disappearances are surprises); frame surprise = strongest interior
        # cell. Sparse events hit only a few cells, so a grid-wide mean would
        # dilute them by ~N_cells.
        sd = np.sqrt(s["cell_var"]) + 1e-3
        z_cells = np.abs(err) / sd
        interior_z = z_cells[s["interior"]]
        surprise = float(interior_z.max()) if interior_z.size else 0.0
        s["n"] += 1

        if s["n"] <= self.warmup_frames:
            # bootstrap per-cell statistics quickly; no plasticity yet
            boot = 0.1
            d = err - s["cell_mu"]
            s["cell_mu"] += boot * d
            s["cell_var"] = (1 - boot) * s["cell_var"] + boot * d * d
        else:
            # plasticity is gated PER CELL by surprise: a cell that is
            # currently surprising must not learn, otherwise it absorbs the
            # event into its negative image and cancels it (the calm-before-
            # plasticity equilibrium constraint of the ELL).
            calm = z_cells < 2.0
            cm = calm[:, None]
            feat_row = feat[None, :]
            s["w_fast"] = np.where(cm, s["w_fast"] + self.fast_lr * feat_row * err[:, None],
                                   s["w_fast"])
            s["w_fast"] -= self.fast_decay * (s["w_fast"] - s["w_slow"])
            s["w_slow"] = np.where(cm, s["w_slow"] + self.slow_lr * feat_row * err[:, None],
                                   s["w_slow"])
            s["neg"] = np.where(calm, s["neg"] + self.neg_ema * err, s["neg"])
            d = err - s["cell_mu"]
            s["cell_mu"] = np.where(calm, s["cell_mu"] + self.neg_ema * d, s["cell_mu"])
            s["cell_var"] = np.where(calm, (1 - self.neg_ema) * s["cell_var"]
                                     + self.neg_ema * d * d, s["cell_var"])

        return {
            "surprise": surprise,                # max per-cell z-score
            "surprise_std": 1.0,                 # already normalized
            "surprise_cells": z_cells,
            "err": err,
        }


class SurpriseTrigger:
    """Hysteresis trigger turning per-frame surprise into saved clips."""

    def __init__(self, cfg: TriggerConfig, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.on = False
        self.start_t: float | None = None
        self.last_hit_t: float | None = None
        self.clips: list[tuple[float, float]] = []
        self.z_ema = 0.0
        self.cnt = 0
        self.prev_strong = False

    def update(self, t: float, surprise: float, std: float) -> bool:
        """Feed one frame; returns True while inside a triggered clip.

        Onset needs PERSISTENCE: EMA-smoothed z above threshold for 3
        consecutive frames, or a strong raw spike (2x threshold) on 2
        consecutive frames. Real events (object appears and stays) satisfy
        both; 1-2 frame glitches (codec, interpolation) satisfy neither.
        """
        self.z_ema += 0.5 * (surprise - self.z_ema)
        thr_on = self.cfg.threshold_k * max(std, 1e-4)
        thr_off = self.cfg.off_k * max(std, 1e-4)
        strong = surprise > 2.0 * thr_on
        if not self.on:
            if self.z_ema > thr_off:
                self.cnt += 1 if self.z_ema > thr_on else 0
            else:
                self.cnt = 0
            if (self.z_ema > thr_on and self.cnt >= 3) or (strong and self.prev_strong):
                self.on = True
                self.start_t = t
                self.last_hit_t = t
                self.cnt = 0
            self.prev_strong = strong
        else:
            # self.on implies onset already set start_t/last_hit_t
            assert self.start_t is not None and self.last_hit_t is not None
            if self.z_ema > thr_off or strong:
                self.last_hit_t = t
            if t - self.last_hit_t > self.cfg.post_roll:
                self.on = False
                self.clips.append((self.start_t, self.last_hit_t))
        return self.on

    def finalize(self, t_end: float):
        if self.on:
            # self.on implies onset already set start_t
            assert self.start_t is not None
            self.clips.append((self.start_t, t_end))
            self.on = False
        # merge clips separated by less than min_clip_len
        merged = []
        for c in self.clips:
            if merged and c[0] - merged[-1][1] < self.cfg.min_clip_len:
                merged[-1] = (merged[-1][0], c[1])
            else:
                merged.append(list(c))
        self.clips = [tuple(c) for c in merged]
        return self.clips

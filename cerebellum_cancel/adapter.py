"""Cerebellum adapter: a predictive add-on for a frozen CV backbone.

Maps the ELL circuit (Perks et al. 2026) onto a neural module:

- granule cells   -> fixed random expansion phi([f_{t-1}, a_t])  (not trained)
- negative image  -> linear predictor W_slow + W_fast on phi
- slow plasticity -> W_slow trained by regression on the train set (long-term)
- fast plasticity -> W_fast updated ONLINE per sequence by a gated local
                     (anti-Hebbian) rule; fast weights relax toward zero
- equilibrium gate-> frames that are currently surprising do NOT update
                     W_fast or the running statistics
- residual r_t    -> f_t - f_hat_t feeds the downstream head

The host backbone is untouched: the adapter wraps its features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdapterConfig:
    n_granule: int = 1024
    ridge: float = 1e-3        # slow-weight regularization
    fast_lr: float = 0.1       # online fast-weight lr (unit-norm granules:
                               # LMS-stable while < 2.0)
    fast_decay: float = 0.02   # per-step pull of W_fast toward 0
    calm_tau: float = 4.0      # surprise above this blocks learning
    std_ema: float = 0.02      # running per-dim residual-statistics rate
    seed: int = 0


class GranuleExpander:
    """Fixed random expansion of [feature, action] — the granule-cell basis."""

    def __init__(self, feat_dim: int, act_dim: int, cfg: AdapterConfig):
        rng = np.random.default_rng(cfg.seed)
        self.E = rng.normal(0, 1 / np.sqrt(feat_dim + act_dim),
                            (cfg.n_granule, feat_dim + act_dim)).astype(np.float32)
        self.b = rng.normal(0, 0.1, cfg.n_granule).astype(np.float32)

    def __call__(self, feat: np.ndarray, action: np.ndarray) -> np.ndarray:
        x = np.concatenate([feat, action]).astype(np.float32)
        phi = np.maximum(self.E @ x + self.b, 0.0)
        # unit norm: keeps the online LMS-style update well-conditioned
        # (stable for lr < 2) regardless of input scale
        return phi / (np.linalg.norm(phi) + 1e-6)


def action_features(gyro_row: np.ndarray) -> np.ndarray:
    """[1, |gx|, |gy|, |gz|] normalized — corollary-discharge embedding."""
    return np.concatenate([[1.0], np.abs(gyro_row[:3]) / 360.0]).astype(np.float32)


def fit_slow_weights(granules: np.ndarray, targets: np.ndarray,
                     ridge: float) -> np.ndarray:
    """Ridge regression for W_slow: granules (N, G) -> targets (N, D)."""
    G, D = granules.shape[1], targets.shape[1]
    A = granules.T @ granules + ridge * np.eye(G)
    return np.linalg.solve(A, granules.T @ targets).astype(np.float32)


class CerebellumAdapter:
    def __init__(self, cfg: AdapterConfig, granule: GranuleExpander,
                 w_slow: np.ndarray | None, use_fast: bool):
        self.cfg = cfg
        self.granule = granule
        self.w_slow = w_slow if w_slow is not None else 0.0
        self.use_fast = use_fast
        self.reset_online()

    def reset_online(self):
        G, D = self.granule.E.shape[0], self.granule.E.shape[1] - 4
        self.w_fast = np.zeros((G, D), dtype=np.float32)
        self.resid_mu = np.zeros(D, dtype=np.float32)
        self.resid_var = np.ones(D, dtype=np.float32)
        self.n = 0

    def predict(self, feat_prev, gyro) -> np.ndarray:
        phi = self.granule(feat_prev, action_features(gyro))
        w = self.w_slow + (self.w_fast if self.use_fast else 0.0)
        return phi @ w, phi

    def step(self, feat, feat_prev, gyro, warmup: int = 10) -> dict:
        """One frame: predict from history, residual, gated plasticity."""
        pred, phi = self.predict(feat_prev, gyro)
        err = feat.astype(np.float32) - pred
        sd = np.sqrt(self.resid_var) + 1e-3
        surprise = float(np.abs(err).max() / sd.max())

        self.n += 1
        calm = surprise < self.cfg.calm_tau
        if self.n > warmup and calm:
            if self.use_fast:
                self.w_fast *= (1 - self.cfg.fast_decay)
                self.w_fast += self.cfg.fast_lr * np.outer(phi, err)
            d = err - self.resid_mu
            self.resid_mu += self.cfg.std_ema * d
            self.resid_var += self.cfg.std_ema * (d * d - self.resid_var)
        return {"residual": err, "zresidual": err / sd,
                "surprise": surprise,
                "learned": calm and self.n > warmup}


def rollout(adapter: CerebellumAdapter, feats: np.ndarray,
            gyros: np.ndarray, warmup: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Run the adapter over one sequence; returns (z-scored residuals, surprises).

    Residuals are standardized by the adapter's own running (causal) calm-frame
    statistics, so the downstream head sees whitened surprise, not raw error.
    """
    adapter.reset_online()
    T, D = feats.shape
    res = np.zeros((T, D), dtype=np.float32)
    sur = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        out = adapter.step(feats[t], feats[t - 1], gyros[t], warmup)
        res[t] = out["zresidual"]
        sur[t] = out["surprise"]
    return res, sur


class LogisticHead:
    """Tiny logistic-regression event head trained with full-batch GD."""

    def __init__(self, dim: int, seed: int = 1, l2: float = 1e-3):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, dim).astype(np.float32)
        self.b = np.float32(0.0)
        self.l2 = l2

    def scores(self, X):
        return X @ self.w + self.b

    def fit(self, X, y, iters=300, lr=0.5):
        n = len(X)
        for _ in range(iters):
            z = np.clip(self.scores(X), -30, 30)
            p = 1 / (1 + np.exp(-z))
            g = X.T @ (p - y) / n + self.l2 * self.w
            gb = float((p - y).mean())
            self.w -= lr * g.astype(np.float32)
            self.b -= np.float32(lr * gb)
        return self


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / max(y.sum(), 1e-9))

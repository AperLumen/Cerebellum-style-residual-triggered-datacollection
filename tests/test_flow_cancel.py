"""Unit tests for cerebellum_cancel.flow_cancel (DIS flow baseline).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import cv2
import numpy as np

from cerebellum_cancel.flow_cancel import (FlowWarpCancel, FlowWarpConfig,
                                           explained_ratio)


def _textured(h=64, w=64, sigma=1.5, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 40, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def test_flow_warp_cancels_translation_better_than_naive():
    """On textured content with pure translation, flow-warped prediction
    must beat no prediction by a wide margin (the same property the gyro
    warp is judged on). Frame shift kept within DIS's search range and
    chosen so the map-space shift is an integer number of cells: bilinear
    resampling is then exact, matching the near-integer alignment regime
    of the real pipeline."""
    cfg = FlowWarpConfig(flow_size=(64, 64), warmup=3, calm_tau=1e9)
    base_img = _textured()
    fmap = _textured(h=32, w=32, sigma=2.0, seed=1)[None]  # (1,32,32)
    fmap = np.concatenate([fmap, fmap], axis=0)

    moved_img = np.roll(base_img, (4, 6), axis=(0, 1))    # -> map shift (2,3)
    moved_fmap = np.roll(fmap, (2, 3), axis=(-2, -1))

    canc = FlowWarpCancel(cfg)
    canc.step(fmap.astype(np.float32), base_img.astype(np.uint8))
    out = canc.step(moved_fmap.astype(np.float32),
                    moved_img.astype(np.uint8))
    sd = np.sqrt(canc.var) + 1e-3
    flow_resid = float(np.abs(out["z"] * sd).mean())

    naive = np.abs(moved_fmap - fmap).mean()
    # DIS per-pixel flow is noisy even when the mean is right (gradient-free
    # patches of blurred noise), so the win is significant but not the
    # near-exact cancellation a known gyro shift achieves — that gap is the
    # baseline's whole story, quantified in scripts/run_flow_eval.py
    assert flow_resid < 0.75 * naive


def test_flow_warp_step_contract():
    """step() must return the same keys as FeatureMapCancel."""
    cfg = FlowWarpConfig(flow_size=(64, 64))
    canc = FlowWarpCancel(cfg)
    img = _textured().astype(np.uint8)
    fmap = _textured(16, 16, seed=2)[None].astype(np.float32)
    out = canc.step(fmap, img)
    assert set(out) == {"z", "surprise", "learned"}
    assert out["z"].shape == fmap.shape


def test_explained_ratio_identities():
    """Same residual -> ratio 1; flow explains half the energy -> 0.5;
    flow explains everything -> ~0. Border cells are excluded."""
    rng = np.random.default_rng(0)
    err = rng.normal(size=(3, 16, 16)).astype(np.float32)
    assert abs(explained_ratio(err, err.copy(), border=2) - 1.0) < 1e-5
    # half the energy: scale residual by sqrt(0.5)
    assert abs(explained_ratio(err, err * np.sqrt(0.5), border=2)
               - 0.5) < 1e-5
    # zero flow residual (flow prediction perfect)
    assert explained_ratio(err, np.zeros_like(err), border=2) < 1e-6


def test_explained_ratio_ignores_border():
    """Energy concentrated in the excluded border must not count."""
    err = np.zeros((1, 12, 12), dtype=np.float32)
    err[:, :2, :2] = 10.0                      # border-only signal
    inner = np.zeros_like(err)
    inner[:, 4:8, 4:8] = 1.0                   # interior-only signal
    # err has huge border energy, tiny interior; flow residual vice versa
    assert explained_ratio(err, inner, border=3) > 1.0

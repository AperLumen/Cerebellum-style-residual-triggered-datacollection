"""Unit tests for cerebellum_cancel.featmap_cancel (warp + gating).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import numpy as np

from cerebellum_cancel.featmap_cancel import (CancelConfig, FeatureMapCancel,
                                               energy_score, expected_shift,
                                               maxcell_score, shift_map)


def test_shift_map_zero_is_identity():
    F = np.random.default_rng(0).normal(size=(3, 8, 8)).astype(np.float32)
    assert np.allclose(shift_map(F, 0.0, 0.0), F)


def test_shift_map_integer_shift_is_exact_roll():
    F = np.random.default_rng(1).normal(size=(2, 6, 6)).astype(np.float32)
    assert np.allclose(shift_map(F, 2.0, -1.0), np.roll(F, (2, -1), axis=(-1, -2)))


def test_expected_shift_math():
    cfg = CancelConfig(fps=30.0, f_map=24.25)
    dx, dy = expected_shift(np.array([0.0, 30.0, 0.0]), cfg)
    assert np.isclose(dx, 24.25 * np.tan(np.radians(30.0) / 30.0))
    assert dy == 0.0


def test_warp_roundtrip_error_small():
    """The gyro warp's round-trip error (shift by +d then back by -d) must be
    much smaller than the motion itself on spatially smooth content — the
    property that makes gyro-driven prediction possible at all. Uses blurred
    noise as a stand-in for the low-frequency content of real conv feature
    maps (on white noise, bilinear subpixel interpolation is lossy and this
    ratio does not hold)."""
    import cv2
    rng = np.random.default_rng(2)
    base = rng.normal(size=(2, 64, 64)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=2.0)
    cfg = CancelConfig(fps=30.0, f_map=24.25)
    gyro = np.array([300.0, 150.0, 0.0])
    dx, dy = expected_shift(gyro, cfg)
    moved = shift_map(base, dx, dy)
    back = shift_map(moved, -dx, -dy)
    inner = np.s_[:, 8:-8, 8:-8]  # exclude the roll wrap-around band
    roundtrip = np.abs(back[inner] - base[inner]).mean()
    motion = np.abs(moved[inner] - base[inner]).mean()
    assert roundtrip < 0.3 * motion


def test_calm_gate_blocks_updates():
    cfg = CancelConfig(warmup=2, calm_tau=2.0)
    canc = FeatureMapCancel(cfg)
    F = np.ones((2, 8, 8), dtype=np.float32)
    canc.step(F, np.zeros(3))
    canc.step(F, np.zeros(3))
    mu_before = canc.mu.copy()
    big = np.ones((2, 8, 8), dtype=np.float32) * 20.0
    out = canc.step(big, np.zeros(3))  # surprise >> calm_tau
    assert not out["learned"]
    assert np.allclose(canc.mu, mu_before)


def test_energy_and_maxcell_scores_agree_on_single_hot():
    z = np.zeros((3, 2, 8, 8), dtype=np.float32)
    z[1, 0, 4, 4] = 3.0
    e = energy_score(z)
    m = maxcell_score(z)
    assert e[1] == 9.0
    assert m[1] == 9.0
    assert e[0] == e[2] == 0.0

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


def test_robust_stats_immune_to_spikes():
    """Winsorized always-update stats must keep tracking the bulk of the
    distribution under repeated outliers: calm frames stay low-z, the
    stats never freeze, and a genuine step shift still produces a large
    z."""
    rng = np.random.default_rng(3)
    cfg = CancelConfig(warmup=5, robust=True, std_ema=0.1)
    canc = FeatureMapCancel(cfg)
    base = rng.normal(0, 0.1, (2, 8, 8)).astype(np.float32)
    zs = []
    for t in range(200):
        F = base + rng.normal(0, 0.05, base.shape).astype(np.float32)
        if t in (50, 120):                      # outlier frames
            F = F + 30.0
        zs.append(canc.step(F, np.zeros(3))["z"])
    zs = np.asarray(zs)
    mz = np.abs(zs).max(axis=(1, 2, 3))
    calm = [t for t in range(200) if t not in (50, 120)]
    assert np.median(mz[10:]) < 5.0             # bulk stays ordinary
    assert mz[50] > 20 and mz[120] > 20         # spikes still visible


def test_robust_detects_step_change():
    """A persistent step in the feature must produce a large z at the onset
    frame (the residual is a frame difference, so the step appears once) —
    the property the trigger reads out."""
    rng = np.random.default_rng(4)
    cfg = CancelConfig(warmup=5, robust=True, std_ema=0.1)
    canc = FeatureMapCancel(cfg)
    base = rng.normal(0, 0.1, (2, 8, 8)).astype(np.float32)
    surprises = []
    for t in range(100):
        F = base + (3.0 if t >= 60 else 0.0)
        F = F + rng.normal(0, 0.02, base.shape).astype(np.float32)
        surprises.append(canc.step(F, np.zeros(3))["surprise"])
    assert max(surprises[6:60]) < 4.0           # quiet before the event
    assert surprises[60] > 20.0                 # onset detected
    assert np.median(surprises[61:]) < 4.0      # absorbed right after


def test_robust_step_contract_and_always_updates():
    """Robust mode returns the same keys and never gates stat updates."""
    cfg = CancelConfig(warmup=1, robust=True)
    canc = FeatureMapCancel(cfg)
    F = np.ones((2, 8, 8), dtype=np.float32)
    canc.step(F, np.zeros(3))
    out = canc.step(F * 100.0, np.zeros(3))     # huge surprise
    assert set(out) == {"z", "surprise", "learned"}
    assert out["learned"]
    assert out["z"].shape == F.shape

"""Regression test for the adaptive-trigger stat freeze (gate latching open).

Measured 2026-09-12 on the 10 Open-AoE event segments: with the released
``TriggerConfig`` (``robust=False``) the score baseline mu/var is updated only
on frames that are outside a clip AND below the onset threshold. Under dense
surprise that update never runs, so the normaliser freezes, z stays above the
onset threshold, and the gate latches OPEN — 2 of 10 segments stored 96-99% of
the stream (``runs/diag_gate_lock.json``). ``robust=True`` replaces the gate
with a winsorized always-update rule.

Measured on the synthetic traces below (fps=30, calm=100, hot=1500):

    trace (n=3000)              released   robust
    dense burst to end (t=6.7s) 0.933      0.032
    settled burst [6.7s, 40s)   0.337      0.032
    1 s isolated excursion      0.014      0.012

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import numpy as np

from cerebellum_cancel.core import AdaptiveTrigger, TriggerConfig


def drive(score: np.ndarray, fps: float = 30.0, **cfg_kw):
    """Run an AdaptiveTrigger over a score trace; return (stored, n_clips)."""
    trig = AdaptiveTrigger(TriggerConfig(**cfg_kw))
    for i in range(1, len(score)):
        trig.update(i / fps, float(score[i]), fps)
    clips = trig.finalize(len(score) / fps)
    keep = np.zeros(len(score), dtype=bool)
    for a, b in clips:
        keep[max(int(a * fps), 0):min(int(b * fps) + 1, len(score))] = True
    return float(keep.mean()), len(clips)


def dense_tail(n: int = 3000, calm: float = 100.0, hot: float = 1500.0,
               start: int = 200) -> np.ndarray:
    """Calm warmup, then a dense burst that runs to the end of the trace.

    Mirrors the real latched segments, where annotated events (and the fast
    hand motion around them) are spread throughout the recording, so the gated
    update never sees a calm frame again after onset.
    """
    score = np.full(n, calm)
    score[start:] = hot
    return score


def test_default_config_preserves_released_behaviour():
    """``robust`` must default to False so every released number is unchanged."""
    assert TriggerConfig().robust is False


def test_gated_stats_latch_open_on_dense_burst():
    """The documented failure mode: once the burst starts and no calm frame
    follows, the released rule freezes its baseline and stores the whole tail
    (measured 0.933)."""
    stored, _ = drive(dense_tail(), robust=False)
    assert stored > 0.8, (
        f"expected the released rule to latch open, got {stored:.3f}")


def test_robust_stats_keep_the_gate_monotone():
    """The winsorized always-update rule must NOT latch open: the baseline
    keeps adapting during the burst, so z decays and the gate closes
    (measured 0.032)."""
    stored, _ = drive(dense_tail(), robust=True)
    assert stored < 0.2, f"robust rule latched open at {stored:.3f}"


def test_robust_stats_beat_released_when_calm_returns():
    """With a settled burst the released rule still over-stores (0.337) while
    the robust rule returns to its floor (0.032)."""
    n = 3000
    score = np.full(n, 100.0)
    score[200:1200] = 1500.0                  # 33 s burst, then calm again
    stored_r, _ = drive(score, robust=True)
    stored_f, _ = drive(score, robust=False)
    assert stored_r < stored_f, (
        f"robust ({stored_r:.3f}) should store less than released "
        f"({stored_f:.3f}) when calm frames return")


def test_robust_stats_do_not_suppress_a_real_event():
    """The fix must not trade the latch for blindness: an isolated excursion
    in an otherwise calm stream still triggers (measured 0.012)."""
    n = 3000
    score = np.full(n, 100.0)
    score[1500:1530] = 1500.0                 # 1 s excursion
    stored, n_clips = drive(score, robust=True)
    assert n_clips >= 1, "robust stats missed a clear 1 s excursion"
    assert 0.0 < stored < 0.5


def test_robust_stats_are_scale_free():
    """Two traces with the same temporal shape but different score scales must
    store similarly under the robust rule — the trigger is self-calibrating,
    and a frozen baseline is precisely what breaks that property."""
    loom = dense_tail(calm=100.0, hot=1500.0)
    quiet = dense_tail(calm=5.0, hot=75.0)    # same ratio, 20x smaller
    s_loom, _ = drive(loom, robust=True)
    s_quiet, _ = drive(quiet, robust=True)
    assert abs(s_loom - s_quiet) < 0.15, (
        f"robust rule is not scale-free: {s_loom:.3f} vs {s_quiet:.3f}")

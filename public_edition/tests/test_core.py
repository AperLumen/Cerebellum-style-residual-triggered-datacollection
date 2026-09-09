"""Unit tests for cerebellum_cancel.core (plasticity rules, trigger).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import numpy as np

from cerebellum_cancel.core import CerebellumCancel, SurpriseTrigger, TriggerConfig


def make_cancel(**kw):
    return CerebellumCancel(**kw)


def cells(nx, ny):
    """All-zero residual cells for an (nx, ny) grid."""
    return np.zeros(nx * ny)


def test_features_bias_only_when_dt_zero():
    f = CerebellumCancel._features([100.0, -50.0, 25.0], 0.0)
    assert np.allclose(f, [1.0, 0.0, 0.0, 0.0])


def test_features_abs_and_clip():
    f = CerebellumCancel._features([360.0, -720.0, 3600.0], 0.5)
    # |rate| = |gyro|/360 rev/s, clipped to [0, 2]
    assert np.allclose(f, [1.0, 1.0, 2.0, 2.0])


def test_surprise_spike_in_single_cell():
    """A residual confined to one interior cell must dominate frame surprise."""
    c = make_cancel(warmup_frames=20)
    for _ in range(30):
        out = c.process(cells(8, 6), [0, 0, 0], dt=1 / 30)
    r = cells(8, 6)
    r[8 + 1] = 10.0  # one interior cell
    out = c.process(r, [0, 0, 0], dt=1 / 30)
    assert out["surprise"] > 5.0
    assert out["surprise_cells"][8 + 1] == out["surprise"]


def test_surprising_cell_does_not_learn():
    """Equilibrium-rate gate: a cell that is surprising right now must not
    absorb the event into its negative image."""
    c = make_cancel(warmup_frames=5, neg_ema=0.1)
    for _ in range(10):
        c.process(cells(8, 6), [0, 0, 0], dt=1 / 30)
    r = cells(8, 6)
    r[8 + 1] = 50.0  # far above the gate (z > 2)
    before = c.state["neg"][8 + 1]
    c.process(r, [0, 0, 0], dt=1 / 30)
    assert c.state["neg"][8 + 1] == before


def test_calm_cell_negative_image_absorbs_offset():
    """A persistent calm offset must be absorbed by the negative image +
    action weights, driving the mean error back toward zero. The offset must
    stay below the surprise gate (z < 2), as in the real pipeline where sd is
    calibrated on live frames."""
    rng = np.random.default_rng(0)
    c = make_cancel(warmup_frames=5, neg_ema=0.01)
    noise = rng.normal(0.0, 1.0, size=(400, 48))
    for t in range(10):
        c.process(noise[t], [0, 0, 0], dt=1 / 30)
    errs = []
    for t in range(10, 400):
        out = c.process(noise[t] + 1.0, [0, 0, 0], dt=1 / 30)
        errs.append(out["err"][8 + 1])
    errs = np.array(errs)
    # per-frame err keeps the iid noise (unpredictable by construction); the
    # direct check is on the absorbed state, which must equal the offset
    assert np.mean(errs[:20]) > 0.35
    absorbed = (c.state["neg"][9] + c.state["w_fast"][9, 0]
                + c.state["w_slow"][9, 0])
    assert abs(absorbed - 1.0) < 0.3


def test_w_fast_decays_toward_w_slow():
    c = make_cancel(warmup_frames=0, fast_lr=0.0, fast_decay=0.1)
    for _ in range(5):
        c.process(cells(8, 6) + 1.0, [30, 0, 0], dt=1 / 30)
    c.state["w_fast"][:] = 1.0
    c.state["w_slow"][:] = 0.0
    for _ in range(200):
        c.process(cells(8, 6), [0, 0, 0], dt=1 / 30)
    assert np.all(np.abs(c.state["w_fast"]) < 1e-3)


def test_trigger_persistence_blocks_single_spike():
    """Onset requires persistence: a single spike must not trigger; a
    sustained above-threshold signal must. With EMA rate 0.5, three counted
    frames land ~4-5 frames in."""
    cfg = TriggerConfig(threshold_k=6.0, off_k=2.5)
    tr = SurpriseTrigger(cfg, fps=30.0)
    assert not tr.update(0.0, 12.0, 1.0)  # single spike
    for i in range(4):
        assert not tr.update((i + 1) / 30, 0.0, 1.0)  # decays back, still off
    tr2 = SurpriseTrigger(cfg, fps=30.0)
    on = False
    for i in range(8):
        on = tr2.update(i / 30, 8.0, 1.0)
    assert on


def test_trigger_merges_short_gaps():
    cfg = TriggerConfig(min_clip_len=0.4)
    tr = SurpriseTrigger(cfg, fps=30.0)
    for i in range(5):
        tr.update(i / 30, 8.0, 1.0)
    for i in range(5, 8):  # 0.1 s gap < min_clip_len
        tr.update(i / 30, 8.0, 1.0)
    clips = tr.finalize(8 / 30)
    assert len(clips) == 1


def test_trigger_finalize_opens_clip_at_end():
    cfg = TriggerConfig()
    tr = SurpriseTrigger(cfg, fps=30.0)
    for i in range(5):
        tr.update(i / 30, 8.0, 1.0)
    clips = tr.finalize(10.0)
    assert clips == [(4 / 30, 10.0)]  # onset lands on the 5th frame


def _run_adaptive(base, noise_sd, spike, n_calm=300, n_spike=10):
    from cerebellum_cancel.core import AdaptiveTrigger
    rng = np.random.default_rng(0)
    tr = AdaptiveTrigger()
    scores = np.concatenate([rng.normal(base, noise_sd, n_calm),
                             np.full(n_spike, spike),
                             rng.normal(base, noise_sd, 100)])
    for i, s in enumerate(scores):
        tr.update(i / 30, float(s), 30.0)
    tr.finalize(len(scores) / 30)
    return tr


def test_adaptive_trigger_fires_without_set_threshold():
    """A spike ~8 calm-sd above baseline must onset with NO hand-set score
    threshold — the same dimensionless 6/2.5 gate the pixel pipeline uses."""
    tr = _run_adaptive(base=100.0, noise_sd=10.0, spike=180.0)
    assert len(tr.clips) == 1
    a, b = tr.clips[0]
    assert 300 / 30 - 1 <= a <= 310 / 30 + 1   # clip covers the spike
    assert b > a


def test_adaptive_trigger_is_scale_invariant():
    """The same RELATIVE spike at a 10x different score scale must behave
    the same — the property a fixed raw threshold (T=300) does not have."""
    tr = _run_adaptive(base=1000.0, noise_sd=100.0, spike=1800.0)
    assert len(tr.clips) == 1


def test_adaptive_stats_not_polluted_by_event():
    """Event frames must not move the calm baseline (equilibrium gate)."""
    from cerebellum_cancel.core import AdaptiveTrigger
    rng = np.random.default_rng(1)
    tr = AdaptiveTrigger(warmup_frames=30)
    calm = rng.normal(50.0, 5.0, 400)
    scores = np.concatenate([calm[:200], np.full(10, 500.0), calm[200:]])
    for i, s in enumerate(scores):
        tr.update(i / 30, float(s), 30.0)
    tr.finalize(len(scores) / 30)
    assert abs(tr.mu - 50.0) < 5.0 * 0.5
    assert len(tr.clips) == 1

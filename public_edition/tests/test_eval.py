"""Unit tests for cerebellum_cancel.eval (metrics and baselines).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

from cerebellum_cancel.eval import evaluate, uniform_baseline


def test_evaluate_perfect_clips():
    events = [(1.0, 2.0), (5.0, 6.0)]
    clips = [(0.9, 2.1), (5.0, 6.0)]
    r = evaluate("m", clips, events, duration_s=10.0, fps=30.0)
    assert r.recall == 1.0
    assert r.precision == 1.0
    assert abs(r.stored_fraction - 2.2 / 10.0) < 1e-9
    assert r.n_clips == 2


def test_evaluate_tail_pad_counts_disappearance():
    """A clip just after the annotated window (disappearance) still counts."""
    events = [(1.0, 2.0)]
    clips = [(2.2, 2.4)]  # within tail_pad=0.5 of the event end
    r = evaluate("m", clips, events, duration_s=10.0, fps=30.0)
    assert r.recall == 1.0


def test_evaluate_no_clips():
    r = evaluate("m", [], [(1.0, 2.0)], duration_s=10.0, fps=30.0)
    assert r.recall == 0.0
    assert r.stored_fraction == 0.0
    assert r.n_clips == 0


def test_uniform_baseline_budget_and_coverage():
    clips = uniform_baseline(60.0, 0.1)
    assert len(clips) == 6
    stored = sum(b - a for a, b in clips)
    assert abs(stored - 6.0) < 1e-9  # 6 x 1 s clips
    assert clips[0][0] == 0.0
    assert clips[-1][1] <= 60.0

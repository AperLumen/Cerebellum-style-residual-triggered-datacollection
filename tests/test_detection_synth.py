"""Unit tests for scripts/train_detection_synth.py (heatmap GT + detection AP).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_detection_synth import (FRAME_H, FRAME_W, boxes_to_heatmap,
                                   detection_ap, eval_detection, heatmap_peaks)


class _PerfectHeatmapModel(torch.nn.Module):
    """Returns logits of the GT heatmap -> sigmoid(pred) == GT heatmap."""

    def __init__(self, heatmaps: np.ndarray):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.logit(torch.from_numpy(heatmaps).clamp(1e-4, 1 - 1e-4)),
            requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits[:len(x)]


def test_heatmap_empty_is_zeros():
    hm = boxes_to_heatmap([], out=28)
    assert hm.shape == (28, 28)
    assert hm.dtype == np.float32
    assert hm.max() == 0.0


def test_heatmap_peak_at_bbox_center():
    # square box centered at (160, 120) px = frame center -> heatmap center
    hm = boxes_to_heatmap([[140, 100, 180, 140]], out=28)
    py, px = np.unravel_index(np.argmax(hm), hm.shape)
    assert (px, py) == (14, 14)
    assert np.isclose(hm.max(), 1.0)


def test_heatmap_sigma_scales_with_box_size():
    small = boxes_to_heatmap([[150, 110, 170, 130]], out=28)
    big = boxes_to_heatmap([[120, 80, 200, 160]], out=28)
    above_half_small = (small > 0.5).sum()
    above_half_big = (big > 0.5).sum()
    assert above_half_big > above_half_small


def test_peaks_finds_gaussian_maximum():
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float32)
    prob = torch.from_numpy(np.exp(-((xx - 10) ** 2 + (yy - 8) ** 2) / 8.0))
    pk = heatmap_peaks(prob).numpy()
    assert len(pk) == 1
    assert np.allclose(pk[0], [1.0, 8.0, 10.0], atol=1e-4)


def test_detection_ap_perfect_is_one():
    gts = {0: [(8.0, 10.0)], 5: [(20.0, 3.0)]}
    dets = np.array([[0.9, 8.0, 10.0, 0.0],
                     [0.8, 20.0, 3.0, 5.0]])
    assert np.isclose(detection_ap(dets, gts), 1.0)


def test_detection_ap_misses_are_zero():
    gts = {0: [(8.0, 10.0)]}
    dets = np.array([[0.9, 20.0, 20.0, 0.0]])     # far away -> fp
    assert detection_ap(dets, gts) == 0.0
    assert np.isnan(detection_ap(np.zeros((0, 4)), gts))


def test_detection_ap_penalizes_high_scoring_false_positive():
    """A confident far-away detection outranks the true hit: precision at
    full recall drops to 1/2, and interpolated AP follows."""
    gts = {0: [(8.0, 10.0)]}
    dets = np.array([[0.95, 25.0, 25.0, 0.0],    # fp with the top score
                     [0.50, 8.0, 10.0, 0.0]])    # true hit, lower score
    assert np.isclose(detection_ap(dets, gts), 0.5)


def test_eval_detection_perfect_model_ap_is_one():
    """End-to-end eval-path guard: a model whose sigmoid output equals the
    GT heatmap must score AP ~1. If this regresses, AP~0 results are an
    eval bug, not a learning failure (see runs/diag_detection_rig.py)."""
    boxes_by_frame = {
        0: [[100.0, 80.0, 148.0, 92.0]],              # val-like bar
        3: [[60.0, 60.0, 116.0, 116.0]],              # circle
        4: [],
    }
    heat = np.stack([boxes_to_heatmap(boxes_by_frame.get(i, []))
                     for i in range(5)])
    feats = np.zeros((5, 24, 28, 28), dtype=np.float16)
    model = _PerfectHeatmapModel(heat)
    ap = eval_detection(model, [(feats, boxes_by_frame)], "cpu")
    assert ap > 0.99

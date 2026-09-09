"""Unit tests for scripts/extract_featmaps_dino.py (DINOv2 backbone swap).

Covers the properties the downstream comparisons rely on:
  - output shapes (patch-token grid + pooled features, float16)
  - determinism (repeat passes agree, batch-size independent)
  - alignment with the input frames (row i == backbone(frame i), in batch
    and one-by-one)

Weights load from the local HF cache (already downloaded by the extraction
run); the module-wide fixture builds the 392-input model once (~seconds).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import glob
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from extract_featmaps_dino import FEATS_NAME, FMAPS_NAME, extract_seq, load_model

FRAME_DIR = os.path.join(os.path.dirname(__file__), "..",
                         "data", "dataset", "train", "seq_000", "frames")


@pytest.fixture(scope="module")
def model_tf():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tf = load_model(device)          # default SIZE=392 -> 28x28 grid
    return model, tf, device


@pytest.fixture(scope="module")
def three_frames():
    frames = sorted(glob.glob(f"{FRAME_DIR}/*.png"))[:3]
    assert len(frames) == 3, "synthetic dataset frames missing"
    return frames


def test_output_shapes(model_tf, three_frames):
    model, tf, device = model_tf
    fmaps, feats = extract_seq(model, tf, three_frames, batch=2, device=device)
    assert fmaps.shape == (3, 384, 28, 28)     # 392/14 patch grid == rig grid
    assert feats.shape == (3, 384)
    assert fmaps.dtype == np.float16
    assert feats.dtype == np.float16
    assert np.isfinite(fmaps.astype(np.float32)).all()
    assert np.isfinite(feats.astype(np.float32)).all()


def test_deterministic_across_batch_sizes(model_tf, three_frames):
    model, tf, device = model_tf
    fm_a, ft_a = extract_seq(model, tf, three_frames, batch=1, device=device)
    fm_b, ft_b = extract_seq(model, tf, three_frames, batch=3, device=device)
    assert np.array_equal(fm_a, fm_b)
    assert np.array_equal(ft_a, ft_b)


def test_rows_aligned_with_input_frames(model_tf, three_frames):
    """Row i of the outputs must be exactly what a single-frame extraction
    of frame i returns — featmaps stay index-aligned with frames/*.png,
    imu.csv and bboxes.json, for both batched and one-by-one passes."""
    model, tf, device = model_tf
    fmaps, feats = extract_seq(model, tf, three_frames, batch=3, device=device)
    for i, p in enumerate(three_frames):
        fm_i, ft_i = extract_seq(model, tf, [p], batch=1, device=device)
        assert np.array_equal(fmaps[i], fm_i[0])
        assert np.array_equal(feats[i], ft_i[0])


def test_distinct_frames_give_distinct_rows(model_tf, three_frames):
    model, tf, device = model_tf
    fmaps, _ = extract_seq(model, tf, three_frames, batch=3, device=device)
    assert not np.array_equal(fmaps[0], fmaps[1])


def test_pooled_equals_patch_mean(model_tf, three_frames):
    """features_dino.npy must be the mean over the patch-token grid of
    featmaps_dino.npy — both come from one forward pass."""
    model, tf, device = model_tf
    fmaps, feats = extract_seq(model, tf, three_frames, batch=3, device=device)
    pooled = fmaps.astype(np.float32).mean(axis=(2, 3))
    assert np.allclose(pooled, feats.astype(np.float32), atol=1e-2)


def test_saved_files_do_not_clobber_mobilenet(model_tf, three_frames,
                                              tmp_path):
    """The DINOv2 outputs land in their own filenames next to the
    published MobileNetV3 featmaps.npy / features.npy."""
    model, tf, device = model_tf
    fmaps, feats = extract_seq(model, tf, three_frames, batch=3, device=device)
    np.save(tmp_path / FMAPS_NAME, fmaps)
    np.save(tmp_path / FEATS_NAME, feats)
    assert os.path.exists(tmp_path / "featmaps_dino.npy")
    assert os.path.exists(tmp_path / "features_dino.npy")
    assert not os.path.exists(tmp_path / "featmaps.npy")
    back = np.load(tmp_path / FMAPS_NAME)
    assert back.shape == (3, 384, 28, 28)

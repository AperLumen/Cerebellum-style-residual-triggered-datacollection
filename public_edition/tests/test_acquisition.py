"""Unit tests for cerebellum_cancel.acquisition (pluggable acquisition layer).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import numpy as np

from cerebellum_cancel.acquisition import AcquisitionLayer
from cerebellum_cancel.core import AdaptiveTrigger
from cerebellum_cancel.featmap_cancel import CancelConfig

FPS = 30.0
DT = 1.0 / FPS


def gray_frontend(frame: np.ndarray) -> np.ndarray:
    """最小可复现前端：4x 抽帧灰度，HWC uint8 -> (1,H,W) float32。"""
    g = frame[::4, ::4].mean(axis=2, dtype=np.float32) / 255.0
    return g[None]


def make_layer(frontend=gray_frontend, std_ema: float = 0.02,
               **trigger_kwargs) -> AcquisitionLayer:
    trig = AdaptiveTrigger(warmup_frames=trigger_kwargs.pop("warmup_frames", 2),
                           **trigger_kwargs)
    return AcquisitionLayer(frontend=frontend,
                            cancel_cfg=CancelConfig(fps=FPS, warmup=2,
                                                    std_ema=std_ema),
                            trigger_cfg=trig)


def test_step_output_shapes():
    """接口形状：逐帧 step 返回 keep/in_clip/surprise/score/t/z_map/clips，
    类型正确。"""
    rng = np.random.default_rng(0)
    layer = make_layer()
    out: dict | None = None
    for t in range(20):
        frame = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        out = layer.step(frame, np.zeros(3), t * DT)
    assert out is not None
    assert set(out) == {"keep", "in_clip", "surprise", "score", "t", "z_map",
                        "clips"}
    assert isinstance(out["keep"], bool) and isinstance(out["in_clip"], bool)
    assert isinstance(out["surprise"], float) and isinstance(out["score"], float)
    assert isinstance(out["t"], float)           # 原样回传的时间戳
    assert out["z_map"].shape == (8, 8) and out["z_map"].dtype == np.float32
    assert isinstance(out["clips"], list)
    for a, b in layer.finalize():                # finalize 返回 [(start, end)]
        assert isinstance(a, float) and isinstance(b, float)


def test_static_sequence_never_triggers():
    """零运动（恒定帧 + 零 gyro）下不触发：keep 全 False，finalize 为空。"""
    layer = make_layer()
    frame = np.full((32, 32, 3), 120, dtype=np.uint8)
    keeps = []
    for t in range(20):
        keeps.append(layer.step(frame, np.zeros(3), t * DT)["keep"])
    assert not any(keeps)
    assert layer.finalize() == []


def test_persistent_residual_triggers_and_boundaries():
    """大残差事件（t>=30 帧内容阶跃）触发：finalize 1 个片段，右端点
    收在末帧。残差是帧差，阶跃只在 onset 帧出现一次尖峰，靠触发器的
    z-EMA 衰减跨帧持续满足 onset；事件持续到流结束，off 回滞等不到，
    finalize 把开放片段闭到最后一帧 t=39。"""
    rng = np.random.default_rng(3)
    layer = make_layer(std_ema=0.3)   # 短序列内让运行方差收敛到噪声底
    base = rng.integers(40, 160, (32, 32, 3)).astype(np.int16)  # 固定纹理底
    keeps = []
    for t in range(40):
        frame = base + rng.integers(-15, 16, base.shape)
        if t >= 30:
            frame = frame + 250                    # 内容阶跃：大残差事件
        keeps.append(layer.step(frame.clip(0, 255).astype(np.uint8),
                                np.zeros(3), t * DT)["keep"])
    clips = layer.finalize()
    assert any(keeps)
    assert len(clips) == 1
    start, end = clips[0]
    # onset 因 z-EMA 平滑与强尖峰判据落在事件后 1~2 帧（t=1.0667，帧 32），
    # 再被 pre_roll 前移 0.3s。所以起点不允许晚于 onset 上界，
    # 也不允许晚到 onset 减 pre_roll 之外——即 pre_roll 确实生效了。
    pre = layer.trigger.cfg.pre_roll
    assert start >= 0.0
    assert start <= 36 * DT, f"onset 太晚: {start}"
    assert start <= 32 * DT - pre + 1e-6, (
        f"pre_roll 未生效: start={start:.4f}, "
        f"期望 <= {32 * DT - pre:.4f}")
    assert np.isclose(end, 39 * DT)          # 开放片段闭到末帧


def test_frontend_is_replaceable():
    """前端是可插拔 callable：不同输出形状的前端都能驱动同一层。"""
    def rgb_frontend(frame: np.ndarray) -> np.ndarray:
        g = frame[::2, ::2].astype(np.float32).transpose(2, 0, 1) / 255.0
        return g

    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (24, 24, 3), dtype=np.uint8) for _ in range(6)]
    out_a: dict | None = None
    out_b: dict | None = None
    layer = make_layer()
    for t, f in enumerate(frames):
        out_a = layer.step(f, np.zeros(3), t * DT)
    layer = make_layer(frontend=rgb_frontend)   # 3 通道、不同的图尺寸
    for t, f in enumerate(frames):
        out_b = layer.step(f, np.zeros(3), t * DT)
    assert out_a is not None and out_b is not None
    assert out_a["z_map"].shape == (6, 6)       # (1,6,6) 的 H,W
    assert out_b["z_map"].shape == (12, 12)     # (3,12,12) 的 H,W


def test_builtin_frontend_default_spec():
    """字符串默认规格 mobilenet_v3_small[:3]：随机帧 -> (24,28,28) 特征图，
    z 图尺寸随之确定。权重为 torchvision 缓存时离线可用。"""
    rng = np.random.default_rng(2)
    layer = AcquisitionLayer(cancel_cfg=CancelConfig(fps=FPS, warmup=2),
                             trigger_cfg=AdaptiveTrigger(warmup_frames=2))
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    out = layer.step(frame, np.zeros(3), 0.0)
    assert out["z_map"].shape == (28, 28)
    assert np.isfinite(out["surprise"]) and np.isfinite(out["score"])

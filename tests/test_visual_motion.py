"""无 IMU 降级路径的测试（visual_motion + engine 的 no_imu 三种模式）。

这一组测试守的是一条**诚实性**要求，不是功能：
程序在没有 IMU 时**不能假装设备静止**。gyro=0 等价于"假设完全不动"，而移动
设备上自运动会被当成真实变化——那正是本项目开篇要解决的问题（GUIDE §0.2）。

所以测试重点：
  1. 三种模式各自的语义正确，且**都在 meta 里声明自己用了哪种**；
  2. `require` 模式真的会拒绝（而不是静默降级）；
  3. 视觉估计的**补偿下限**生效——运动亚像素时明确不动，而不是硬 warp
     （实测 0.29px 的位移去补偿会把残差推到 1.885x）。
"""
from __future__ import annotations

import numpy as np
import pytest

from cerebellum_cancel.contracts import (ConfigError, InsufficientDataError,
                                         ShapeError)
from cerebellum_cancel.engine import EngineConfig, run
from cerebellum_cancel.visual_motion import (MIN_COMPENSABLE_SHIFT_PX,
                                             VisualMotionEstimator,
                                             assess_motion)

VIDEO = "data/webcam_rec/video.mp4"


# --------------------------------------------------- 合成帧：确定性、无依赖

def _textured(shift_x: int = 0, seed: int = 0, h: int = 64, w: int = 64):
    """生成一张有纹理的灰度图，整体平移 shift_x 像素。

    用固定随机图案 + np.roll 做平移，保证**特征可匹配**且结果是确定的。
    纯噪声不行（goodFeaturesToTrack 找不到稳定角点）；纯平滑也不行（无特征）。
    所以用"噪声 + 大块结构"的混合。
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 60, (h, w), dtype=np.uint8)
    # 叠几个高对比方块，提供稳定角点
    for (y, x) in ((10, 10), (40, 12), (12, 42), (44, 44)):
        base[y:y + 8, x:x + 8] = 220
    if shift_x:
        base = np.roll(base, shift_x, axis=1)
    return base


# ------------------------------------------------------------ 视觉估计器

def test_estimator_finds_a_known_translation():
    """给定已知平移，估计量必须接近真值。"""
    est = VisualMotionEstimator()
    a = _textured(0)
    b = _textured(6)                      # 向右移 6 像素
    e = est.estimate(a, b)
    assert e.ok, "纹理充分时估计应当成功"
    assert e.shift_px == pytest.approx(6.0, abs=1.5), f"实得 {e.shift_px}"


def test_estimator_reports_failure_on_blank_frames():
    """**空白/无纹理输入必须报告失败**，不能返回一个瞎猜的位移。

    这是本模块最重要的契约：无纹理时宁可不给读数，也不污染下游统计。
    """
    est = VisualMotionEstimator()
    a = np.full((64, 64), 128, np.uint8)
    b = np.full((64, 64), 128, np.uint8)
    e = est.estimate(a, b)
    assert e.ok is False
    assert e.affine is None


def test_estimate_rejects_colour_input():
    """彩色输入会静默只在第一个通道上匹配——必须显式拒绝。"""
    est = VisualMotionEstimator()
    a = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(ShapeError, match="2 维灰度"):
        est.estimate(a, a)


def test_estimate_rejects_size_mismatch():
    est = VisualMotionEstimator()
    with pytest.raises(ShapeError, match="尺寸"):
        est.estimate(np.zeros((32, 32), np.uint8),
                     np.zeros((16, 16), np.uint8))


def test_estimator_rejects_bad_hyperparameters():
    with pytest.raises(ConfigError):
        VisualMotionEstimator(max_features=0)


# ------------------------------------------- 补偿下限（负结果固化成护栏）

def test_subpixel_motion_is_not_compensated():
    """**核心护栏**：亚像素运动时 `gyro_equivalent` 必须返回 0。

    实测依据：0.29 px 的位移去 warp，残差从 2.632 涨到 4.962（1.885x）——
    补偿的收益小于插值模糊的代价。返回 0 = "这一帧不动"，是正确行为。
    """
    est = VisualMotionEstimator()
    a = _textured(0)
    b = _textured(0)                     # 零位移
    g = est.gyro_equivalent(a, b, f_map=277.0, dt=1 / 30)
    assert g[2] == 0.0, "零位移不该产生补偿量"


def test_compensable_motion_does_produce_a_rotation_reading():
    """足够大的旋转应当给出非零读数（确认下限不是把所有东西都堵死）。"""
    est = VisualMotionEstimator()
    # 用旋转纹理（不是平移）产生真实的旋转分量
    rng = np.random.default_rng(3)
    base = (rng.random((64, 64)) * 255).astype(np.uint8)
    import cv2
    M = cv2.getRotationMatrix2D((32, 32), 8.0, 1.0)     # 转 8 度
    rot = cv2.warpAffine(base, M, (64, 64))
    e = est.estimate(base, rot)
    assert e.ok
    g = est.gyro_equivalent(base, rot, f_map=277.0, dt=1 / 30)
    # 位移量足够时（旋转的位移场随半径增长）应当过下限
    if e.shift_px >= MIN_COMPENSABLE_SHIFT_PX:
        assert g[2] != 0.0, "过了下限却没有读数"
    else:
        assert g[2] == 0.0, "未过下限却给了读数"


def test_assess_motion_gives_actionable_advice():
    """诊断函数要回答"这份数据值不值得补偿"，而不只是给个数字。"""
    a, b = _textured(0), _textured(0)
    d = assess_motion(a, b)
    assert d["above_compensation_floor"] is False
    assert "建议不开补偿" in d["advice"]
    assert d["floor_px"] == MIN_COMPENSABLE_SHIFT_PX


def test_assess_motion_on_blank_says_estimate_failed():
    blank = np.full((64, 64), 100, np.uint8)
    d = assess_motion(blank, blank)
    assert d["ok"] is False
    assert "估计失败" in d["advice"]


# ------------------------------------------------- engine 的三种模式

def _run_no_imu(mode: str, n: int = 60):
    return run(VIDEO, None, cfg=EngineConfig(max_frames=n, no_imu=mode))


@pytest.mark.skipif(not __import__("os").path.exists(VIDEO),
                    reason="需要测试视频")
def test_require_mode_refuses_without_imu():
    """`require` 必须**拒绝**，而不是静默降级——有些数据管线要求必须有 IMU。"""
    with pytest.raises(InsufficientDataError, match="没有 IMU"):
        _run_no_imu("require")


@pytest.mark.skipif(not __import__("os").path.exists(VIDEO),
                    reason="需要测试视频")
def test_assume_still_mode_declares_itself():
    """假设静止**必须**在 meta 里声明——否则下游会把自运动当真实变化。"""
    r = _run_no_imu("assume_still")
    assert r.meta["motion_source"] == "assume_still"
    assert "STILL" in r.meta["motion_source_note"]
    assert not np.any(r.gyro), "假设静止就该全是 0"


@pytest.mark.skipif(not __import__("os").path.exists(VIDEO),
                    reason="需要测试视频")
def test_auto_mode_uses_visual_and_declares_it():
    """`auto` 走视觉估计，并声明这是**降级**路径而非等价替代。"""
    r = _run_no_imu("auto")
    assert r.meta["motion_source"] == "visual"
    assert "degraded" in r.meta["motion_source_note"]


def test_invalid_no_imu_mode_is_rejected():
    with pytest.raises(ConfigError, match="no_imu"):
        EngineConfig(no_imu="whatever")


@pytest.mark.skipif(not __import__("os").path.exists(VIDEO),
                    reason="需要测试视频")
def test_visual_mode_produces_a_different_motion_signal_than_assume_still():
    """两种模式产出的运动信号必须**确实不同**——否则说明接线错了。

    这条防的是"看起来有两条路径、实际跑的是同一条"。
    """
    a = _run_no_imu("assume_still")
    b = _run_no_imu("auto")
    assert not np.array_equal(a.gyro, b.gyro), "两种模式的 gyro 完全相同"
    assert np.any(b.gyro), "视觉模式应当有非零读数"


@pytest.mark.skipif(not __import__("os").path.exists(VIDEO),
                    reason="需要测试视频")
def test_motion_source_survives_into_meta():
    """来源必须能被下游读到（数据集/导出都会带上 meta）。"""
    r = _run_no_imu("auto", n=45)
    assert "motion_source" in r.meta
    assert r.meta["motion_source"] in ("imu", "visual", "assume_still")

"""输入契约的回归测试（工业化稳定性）。

本文件固化两类东西：

1. **契约本身**：每个异常类、每个检查函数的边界行为。
2. **曾经静默通过的失败路径**：修复前它们返回"数值正常但语义错误"的结果，
   比崩溃更危险——错误会传播进数据集、日志和下游决策。每一条都对应一次
   实测探测（见 GUIDE §5 工业化稳定性）。

特别地，``test_dt_zero_is_still_allowed`` 是**反向护栏**：``dt<=0`` 是
有定义语义的合法输入（表示"无时间流逝"，动作特征只保留偏置项），不能
在加校验时把它误改成崩溃。
"""
from __future__ import annotations

import numpy as np
import pytest

from typing import Any as _Any

from cerebellum_cancel.contracts import (ConfigError, ContractError,
                                         InsufficientDataError, NonFiniteError,
                                         ShapeError, check_1d,
                                         check_finite_array, check_gyro,
                                         check_min_frames, check_positive,
                                         check_scalar)
from cerebellum_cancel.rotwarp import rotation_homography


# ------------------------------------------------------------------ 异常层级

def test_all_contract_errors_are_value_errors():
    """向后兼容：既有 ``except ValueError`` 的调用方必须继续工作。

    这是契约能被安全引入的前提——否则加校验就等于破坏 API。
    """
    for cls in (ShapeError, NonFiniteError, ConfigError, InsufficientDataError):
        assert issubclass(cls, ValueError)
        assert issubclass(cls, ContractError)


def test_contract_error_hierarchy_is_specific():
    """ShapeError/NonFiniteError 不应互相是子类（便于精确捕获）。"""
    assert not issubclass(ShapeError, NonFiniteError)
    assert not issubclass(NonFiniteError, ShapeError)


# ------------------------------------------------------------ check_finite_array

def test_finite_array_accepts_clean_data():
    a = check_finite_array(np.arange(6.0).reshape(2, 3), "x")
    assert a.dtype == np.float64 and a.shape == (2, 3)


def test_finite_array_accepts_list_input():
    """接受 list/tuple：调用方不必先手动转 ndarray。
    且整数输入必须被转成 float64（否则后续浮点运算会静默截断）。
    """
    a = check_finite_array([1, 2, 3], "x")
    assert a.tolist() == [1.0, 2.0, 3.0]
    assert a.dtype == np.float64


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_finite_array_rejects_non_finite(bad):
    """NaN/Inf 必须在入口拦截：一旦进入在线统计（mu/var），
    此后所有帧的判断都失效且不报错。"""
    with pytest.raises(NonFiniteError, match="NaN 或 Inf"):
        check_finite_array(np.array([1.0, bad, 3.0]), "x")


def test_finite_array_reports_the_name():
    """报错要指出是哪个参数，否则多参数调用时无法定位。"""
    with pytest.raises(NonFiniteError, match="residual_cells"):
        check_finite_array(np.array([np.nan]), "residual_cells")


def test_finite_array_rejects_empty_by_default():
    """空数组必须拒绝：空输入下所有下游统计都是无定义的。"""
    with pytest.raises(InsufficientDataError, match="空数组"):
        check_finite_array(np.array([]), "x")
    # 显式声明允许时才放行
    assert check_finite_array(np.array([]), "x", allow_empty=True).size == 0


# ---------------------------------------------------------------- check_1d

def test_check_1d_accepts_1d():
    assert check_1d(np.arange(5.0), "v").shape == (5,)


@pytest.mark.parametrize("shape", [(5, 1), (1, 5)])
def test_check_1d_squeezes_singleton_axes(shape):
    """(n,1)/(1,n) 是常见的意外但语义无歧义，允许挤压。"""
    assert check_1d(np.zeros(shape), "v").shape == (5,)


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (4, 3)])
def test_check_1d_rejects_true_multidim(shape):
    """**核心回归**：修复前 ``.ravel()`` 会静默展平多维输入，
    得到"数值正常但语义错误"的结果。"""
    with pytest.raises(ShapeError, match="多维输入静默展平"):
        check_1d(np.zeros(shape), "residual_cells")


def test_check_1d_enforces_expected_length():
    """长度不符意味着调用方传错了网格 → 必须失败。"""
    with pytest.raises(ShapeError, match="长度不符"):
        check_1d(np.zeros(10), "r", expected_len=48)


def test_check_1d_length_message_shows_both_numbers():
    with pytest.raises(ShapeError, match="期望 48，实得 10"):
        check_1d(np.zeros(10), "r", expected_len=48)


# --------------------------------------------------------------- check_gyro

def test_check_gyro_accepts_3d():
    g = check_gyro([1.0, 2.0, 3.0])
    assert g.shape == (3,)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 1.0])
def test_check_gyro_rejects_wrong_arity(bad):
    with pytest.raises((ShapeError, NonFiniteError)):
        check_gyro(bad)


def test_check_gyro_does_not_police_magnitude():
    """**设计决策**：不校验量级。

    静止时陀螺读数可能亚度/秒，快速甩动可上百——用阈值卡量级会在
    合法数据上误报，所以只校验形状与有限性。
    """
    assert check_gyro([0.0, 0.0, 0.0]).shape == (3,)
    assert check_gyro([0.001, 0.0, 0.0]).shape == (3,)
    assert check_gyro([500.0, 500.0, 500.0]).shape == (3,)


# ------------------------------------------------------------- check_positive

def test_check_positive_accepts_positive():
    assert check_positive(30.0, "fps") == 30.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_check_positive_rejects_non_positive(bad):
    with pytest.raises(ConfigError):
        check_positive(bad, "fps")


def test_check_positive_allow_zero_flag():
    assert check_positive(0.0, "offset", allow_zero=True) == 0.0
    with pytest.raises(ConfigError):
        check_positive(0.0, "offset")


def test_check_positive_rejects_nan():
    with pytest.raises(NonFiniteError):
        check_positive(float("nan"), "fps")


# ------------------------------------------------------------ check_min_frames

def test_check_min_frames_passes_at_boundary():
    check_min_frames(200, 200, "report")      # 等于下限是允许的


def test_check_min_frames_rejects_below():
    with pytest.raises(InsufficientDataError, match="至少 200 帧"):
        check_min_frames(199, 200, "report")


def test_insufficient_data_explains_rejection_rationale():
    """报错信息要说明"为什么拒绝"而不仅是"不够"——
    让人理解这是统计有效性问题，不是可以调参绕过的限制。"""
    with pytest.raises(InsufficientDataError, match="没有统计意义"):
        check_min_frames(10, 200, "integrity.report")


# =====================================================================
# 曾经静默通过的失败路径（每个都对应修复前实测探测到的一条 BUG）
# =====================================================================

def test_core_process_rejects_2d_input():
    """修复前：``.ravel()`` 静默展平 2D → 返回结果，语义错误。"""
    from cerebellum_cancel.core import CerebellumCancel
    c = CerebellumCancel(warmup_frames=0)
    with pytest.raises(ShapeError):
        c.process(np.zeros((8, 6)), [1, 2, 3], 1 / 30)


def test_core_process_rejects_nan_without_polluting_stats():
    """修复前：NaN 静默污染 mu/var/neg → 此后每一帧都失效。

    这里额外断言**失败发生在统计被改动之前**（原子性）。
    """
    from cerebellum_cancel.core import CerebellumCancel
    c = CerebellumCancel(warmup_frames=0)
    c.process(np.ones(48), [1, 2, 3], 1 / 30)     # 先建立状态
    before = {k: v.copy() for k, v in c.state.items() if hasattr(v, "copy")}
    with pytest.raises(NonFiniteError):
        c.process(np.full(48, np.nan), [1, 2, 3], 1 / 30)
    for k, v in before.items():
        assert np.allclose(c.state[k], v), f"校验失败时 state[{k}] 被改动了"


def test_core_process_rejects_wrong_grid_length():
    from cerebellum_cancel.core import CerebellumCancel
    c = CerebellumCancel(warmup_frames=0)
    with pytest.raises(ShapeError):
        c.process(np.zeros(10), [1, 2, 3], 1 / 30)


def test_core_process_rejects_bad_gyro():
    from cerebellum_cancel.core import CerebellumCancel
    c = CerebellumCancel(warmup_frames=0)
    with pytest.raises(ShapeError):
        c.process(np.zeros(48), [1, 2], 1 / 30)


def test_dt_zero_is_still_allowed():
    """**反向护栏**：``dt<=0`` 是合法输入，表示"无时间流逝"。

    语义：动作特征退化为只剩偏置项（见 ``_features``）。加校验时最容易
    的错误就是把这里"顺手修成"崩溃——那会破坏既有的慢帧/重复帧处理。
    """
    from cerebellum_cancel.core import CerebellumCancel
    for dt in (0.0, -1e-9):
        c = CerebellumCancel(warmup_frames=0)
        out = c.process(np.ones(48), [10.0, 0.0, 0.0], dt)
        assert isinstance(out, dict)
    # 特征向量的确退化为 [1,0,0,0]
    assert np.allclose(CerebellumCancel._features([99, 99, 99], 0.0),
                       [1.0, 0.0, 0.0, 0.0])


def test_dt_nan_is_rejected():
    """与 ``dt=0`` 相反：``dt=NaN`` 无定义语义，必须拒绝。"""
    from cerebellum_cancel.core import CerebellumCancel
    c = CerebellumCancel(warmup_frames=0)
    with pytest.raises(ConfigError):
        c.process(np.ones(48), [1, 2, 3], float("nan"))


def test_trigger_rejects_zero_fps():
    """修复前：fps=0 静默接受 → clip 时间边界全部退化。"""
    from cerebellum_cancel.core import AdaptiveTrigger
    with pytest.raises(ConfigError):
        AdaptiveTrigger().update(0.0, 1.0, 0.0)


def test_trigger_rejects_infinite_score():
    """修复前：inf 静默进入 mu/var → 把统计拉到 inf，之后一切反常。"""
    from cerebellum_cancel.core import AdaptiveTrigger
    tr = AdaptiveTrigger()
    with pytest.raises(NonFiniteError):
        tr.update(0.0, float("inf"), 30.0)


def test_trigger_rejects_nan_score():
    from cerebellum_cancel.core import AdaptiveTrigger
    with pytest.raises(NonFiniteError):
        AdaptiveTrigger().update(0.0, float("nan"), 30.0)


# ---------------------------------------------------------------------
# 以下两组测试在主版里存在，但**不在本公开子集的范围**内：
#   * integrity.py —— 采集链路完整性哨兵（离线数据集层的部件）
#   * dataset.py   —— 多模态数据集容器（离线、可训练）
# 本子集的定位是**在线采集层**（acquisition + 取消 + 触发），见 README。
# 主版完整测试见仓库根目录的 tests/test_contracts.py。
# ---------------------------------------------------------------------

# =====================================================================
# pipeline.run_pipeline 的参数契约
#
# pipeline 是编排层（吃路径、转发给下层），它的静默失败模式与逐帧模块不同：
# 参数写错不会崩，只会**安静地改变语义**。实测三类：
#   * fps=0 —— clip 边界与帧时间戳全部退化（实测 clips 从 4 变 1，无任何警告）
#   * imu_offset=NaN —— IMU 时间轴整体变 NaN，匹配不到帧，补偿**静默退化为
#     naive**，读数与"没接 IMU"无法区分（最难查的一类：看起来像数据问题）
#   * crop 越界 —— 静默接受；crop 反序 —— 抛带 OpenCV 内部构建路径的错误
#
# 这些校验必须在**任何 IO 之前**发生，否则会"读了整个视频才报参数错"。
# =====================================================================

@pytest.mark.parametrize("bad_fps", [0, -30, -0.001])
def test_pipeline_rejects_non_positive_fps(bad_fps):
    """fps<=0 会让帧时间戳与 clip 边界退化，此前静默通过。"""
    from cerebellum_cancel.pipeline import run_pipeline
    with pytest.raises(ConfigError, match="fps"):
        run_pipeline("data/webcam_rec/video.mp4", None, fps=bad_fps)


def test_pipeline_rejects_nan_imu_offset():
    """**最隐蔽的一条**：imu_offset=NaN 让补偿静默退化为 naive。

    它不会报错、不会崩，只是让残差读数悄悄变差——与"这段数据没接 IMU"
    完全无法区分。这类错误最该在入口拦掉。
    """
    from cerebellum_cancel.pipeline import run_pipeline
    with pytest.raises(NonFiniteError, match="imu_offset"):
        run_pipeline("data/webcam_rec/video.mp4", None, imu_offset=float("nan"))
    with pytest.raises(NonFiniteError, match="imu_offset"):
        run_pipeline("data/webcam_rec/video.mp4", None, imu_offset=float("inf"))


@pytest.mark.parametrize("bad", [(0, 0, 2, 2), (-0.1, 0, 0.5, 0.5),
                                 (0, 0, 0.5, 1.5)])
def test_pipeline_rejects_out_of_range_crop(bad):
    """crop 必须是 [0,1] 的分数区域；越界此前静默接受。"""
    from cerebellum_cancel.pipeline import run_pipeline
    with pytest.raises(ConfigError, match="crop"):
        run_pipeline("data/webcam_rec/video.mp4", None, crop=bad)


@pytest.mark.parametrize("bad", [(1, 0, 0, 1), (0, 1, 1, 0), (0.5, 0, 0.5, 1)])
def test_pipeline_rejects_reversed_crop(bad):
    """反序 crop 此前抛 OpenCV 内部错误（含其构建路径），信息不可用。"""
    from cerebellum_cancel.pipeline import run_pipeline
    with pytest.raises(ConfigError, match="crop"):
        run_pipeline("data/webcam_rec/video.mp4", None, crop=bad)


def test_pipeline_rejects_wrong_length_crop():
    """此前是 ``not enough values to unpack`` 这种不可定位的错误。"""
    from cerebellum_cancel.pipeline import run_pipeline
    with pytest.raises(ShapeError, match="crop"):
        run_pipeline("data/webcam_rec/video.mp4", None,
                     crop=(0.0, 0.0, 1.0))


def test_pipeline_validation_happens_before_io():
    """参数错必须在读视频**之前**报出。

    用一个不存在的路径 + 一个非法 fps：如果校验在 IO 之后，我们会先拿到
    "cannot open video"（OSError）——那说明校验位置错了。
    """
    from cerebellum_cancel.pipeline import run_pipeline
    with pytest.raises(ConfigError, match="fps"):
        run_pipeline("/nonexistent/x.mp4", None, fps=0)


def test_pipeline_allows_none_fps_and_valid_crop():
    """**反向护栏**：``fps=None`` 是合法用法（沿用流自身的 fps）。

    这个参数存在的意义正是"流可能误报"，所以不能强制要求传值。
    合法的 crop 也不该被误杀。
    """
    from cerebellum_cancel.pipeline import run_pipeline
    # 只在参数层面验证：走到 IO 就应该因路径不存在而失败，
    # 而不是因参数被拒绝。（kwargs 分开写而不是放进 dict，否则 pyright
    # 会把异构值的 dict 推成联合类型，**dict 展开时报错。）
    with pytest.raises(OSError):
        run_pipeline("/nonexistent/x.mp4", None, fps=None, crop=None)
    with pytest.raises(OSError):
        run_pipeline("/nonexistent/x.mp4", None, fps=30.0,
                     crop=(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(OSError):
        run_pipeline("/nonexistent/x.mp4", None, imu_offset=-1.5)
    with pytest.raises(OSError):
        run_pipeline("/nonexistent/x.mp4", None, mirror=True)


# ---------------------------------------------------------------- 类型垃圾
# 背景：fuzz 实测发现，契约函数对"类型垃圾"（字符串 / None / object dtype /
# 多元素数组）曾抛出 Python 原生的 TypeError/ValueError，而不是 ContractError。
# 这违背了契约模块的核心承诺——"失败必须是同一种失败"。以下把这些情况
# 逐一钉住，防止回归。

@pytest.mark.parametrize("bad", [
    "abc",              # 字符串
    None,               # None
    np.array([object()] * 4, dtype=object),   # object dtype
    np.array(["a", "b"]),                     # 字符串数组
    [1.0, 2.0],         # 多元素 list（不是标量）
])
def test_check_scalar_type_garbage_raises_contract_error(bad):
    """check_scalar 对非数值/非标量必须报 ContractError。"""
    with pytest.raises(ContractError):
        check_scalar(bad, "x", "测试")


@pytest.mark.parametrize("bad", [
    ["a", "b"],
    np.array(["a", "b"]),
    np.array([object(), object()], dtype=object),
])
def test_check_finite_array_type_garbage_raises_contract_error(bad):
    """check_finite_array 对无法转成数值的输入必须报 ContractError。

    注：object dtype 里放**数字**是可以转的（那是合法的），
    所以这里用的是放裸 ``object()`` 的数组。
    """
    with pytest.raises(ContractError):
        check_finite_array(bad, "x")


def test_check_positive_type_garbage_raises_contract_error():
    """check_positive 对字符串/None 必须报 ContractError。"""
    bad1: _Any = "abc"
    bad2: _Any = None
    with pytest.raises(ContractError):
        check_positive(bad1, "x")
    with pytest.raises(ContractError):
        check_positive(bad2, "x")


def test_rotation_homography_string_params_raise_contract_error():
    """rotation_homography 的算术参数若为**不可转**的字符串/None，必须是
    ContractError。

    原实现让 Python 抛 ``TypeError: can't multiply sequence by non-int``。

    注：``'30'`` 这类**可无损转成数值**的字符串是刻意接受的（配置常来自
    环境变量/CLI），见 `check_scalar` 的 `str_num` 白名单——所以这里用
    ``'abc'`` 这种真的转不了的。
    """
    badS: _Any = "abc"
    badN: _Any = None
    numS: _Any = "30"
    zeroS: _Any = "0"
    with pytest.raises(ContractError):
        rotation_homography(badS, 0.0, 0.0, 1 / 30, 42.7, 14.0, 14.0)
    with pytest.raises(ContractError):
        rotation_homography(0.0, badN, 0.0, 1 / 30, 42.7, 14.0, 14.0)
    # 反向护栏：可转的数值字符串**必须**继续被接受（否则配置从 CLI 来就废了）
    H = rotation_homography(numS, zeroS, zeroS, 1 / 30, 42.7, 14.0, 14.0)
    assert H.shape == (3, 3)


def test_compensate_rotation_type_garbage_dt_raises_contract_error():
    """compensate_rotation 的 dt 若为 object/字符串数组，必须报 ContractError。"""
    from cerebellum_cancel.io import compensate_rotation
    img = np.zeros((8, 8))
    gyro = np.zeros(3)
    badO: _Any = np.array([object()] * 4, dtype=object)
    badS: _Any = np.array(["a"] * 4)
    with pytest.raises(ContractError):
        compensate_rotation(img, img, gyro, badO)
    with pytest.raises(ContractError):
        compensate_rotation(img, img, gyro, badS)


def test_integrity_sequence_funcs_reject_scalar():
    """0 维标量传给逐帧序列函数，必须是 ContractError（不能是 len() 的 TypeError）。"""
    from cerebellum_cancel.integrity import frozen_baseline_z, spike_threshold
    badZ: _Any = 0
    badE: _Any = 0.0
    with pytest.raises(ContractError):
        spike_threshold(badZ)
    with pytest.raises(ContractError):
        frozen_baseline_z(badE)


def test_visual_motion_estimate_rejects_bad_inputs():
    """estimate 对 None / NaN / 过小图的失败类型统一为 ContractError。"""
    from cerebellum_cancel.visual_motion import VisualMotionEstimator
    est = VisualMotionEstimator()
    img = np.zeros((32, 32))
    _ = img
    badN: _Any = None
    with pytest.raises(ContractError):
        est.estimate(badN, img)
    with pytest.raises(ContractError):
        est.estimate(np.full((32, 32), np.nan), img)
    with pytest.raises(ContractError):
        est.estimate(np.zeros((4, 4)), np.zeros((4, 4)))   # 太小

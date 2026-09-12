"""`io.py` 的契约测试 —— 这是最后一个没有测试覆盖的模块。

它是 **I/O 边界**，也就是最容易被脏数据穿透的地方：文件里的一个 NaN、
一个坏行、一个错列名，都会从这里进入整条链路。而它此前：
  * 坏行被 `except: continue` **静默丢弃**（实测 5 行丢 2 行，无任何提示）
  * NaN 被 `float()` 正常解析后**一路传下去**污染在线统计
  * `gyro_at` 在 t 超出样本范围时**线性外推**出负角速度（凭空造数据）
  * 空 IMU 列表返回全零 = **假装设备静止**（正是本项目要避免的失效模式）
  * `cells_of` 在网格比图还大时切出空数组 → 静默全 NaN + 一条易忽略的警告

这些都不是假想：每一条都对应下面一个测试。
"""
from __future__ import annotations

import numpy as np
import pytest

from cerebellum_cancel.contracts import (ConfigError, InsufficientDataError,
                                         NonFiniteError, ShapeError)
from cerebellum_cancel.io import (ImuSample, cells_of, compensate_rotation,
                                  gyro_at, load_imu_csv)

HEADER = "time,gyroscope.x,gyroscope.y,gyroscope.z\n"


def _csv(tmp_path, body: str, name: str = "imu.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return str(p)


# ----------------------------------------------------------- load_imu_csv

def test_load_reads_plain_rows(tmp_path):
    imu = load_imu_csv(_csv(tmp_path, "0.0,1,2,3\n0.1,4,5,6\n"))
    assert len(imu) == 2
    assert imu[0].t == 0.0
    assert np.allclose(imu[0].gyro, [1, 2, 3])
    assert np.allclose(imu[1].gyro, [4, 5, 6])


def test_load_accepts_header_variants(tmp_path):
    """Sensor Logger / 各家导出的列名不一样，要能认出来。"""
    p = tmp_path / "v.csv"
    p.write_text("timestamp,gyro_x,gyro_y,gyro_z\n1.0,1,2,3\n", encoding="utf-8")
    imu = load_imu_csv(str(p))
    assert len(imu) == 1 and imu[0].t == 1.0


def test_load_rejects_unrecognized_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized"):
        load_imu_csv(str(p))


def test_load_rejects_nan_instead_of_passing_it_through(tmp_path):
    """**核心回归**：NaN 此前被 float() 正常解析后一路传进在线统计。

    CSV 里的 NaN 看起来只是一串字符，但它会让此后所有帧的 z 分数失效，
    且调用方完全看不出根源在 I/O 层。
    """
    with pytest.raises(NonFiniteError, match="NaN/Inf"):
        load_imu_csv(_csv(tmp_path, "0.0,nan,0,0\n"))


def test_load_rejects_inf(tmp_path):
    with pytest.raises(NonFiniteError, match="NaN/Inf"):
        load_imu_csv(_csv(tmp_path, "0.0,inf,0,0\n"))


def test_load_rejects_nan_timestamp(tmp_path):
    with pytest.raises(NonFiniteError, match="NaN/Inf"):
        load_imu_csv(_csv(tmp_path, "nan,1,2,3\n"))


def test_load_header_only_file_is_an_error(tmp_path):
    """表头对但零数据 —— 此前静默返回空列表，调用方只会看到"IMU 是空的"。"""
    p = tmp_path / "empty.csv"
    p.write_text(HEADER, encoding="utf-8")
    with pytest.raises(InsufficientDataError, match="没有任何可用数据行"):
        load_imu_csv(str(p))


def test_load_warns_about_dropped_rows_by_default(tmp_path, capsys):
    """**核心回归**：坏行此前被静默丢弃（实测 5 行丢 2 行，零提示）。"""
    p = _csv(tmp_path, "0.0,1,2,3\nBAD,1,2,3\n0.2,1,2,3\n")
    imu = load_imu_csv(p)
    assert len(imu) == 2, "坏行应被跳过"
    err = capsys.readouterr().err
    assert "跳过 1 行" in err, "必须让人知道丢了数据"


def test_load_can_raise_on_bad_rows(tmp_path):
    """需要严格模式时能直接报错。"""
    p = _csv(tmp_path, "0.0,1,2,3\nBAD,1,2,3\n")
    with pytest.raises(ValueError, match="无法解析"):
        load_imu_csv(p, on_bad_row="raise")


def test_load_can_silently_skip_when_asked(tmp_path, capsys):
    """显式要求静默时就不吵——但这是**要写出来**的选择，不是默认。"""
    p = _csv(tmp_path, "0.0,1,2,3\nBAD,1,2,3\n")
    imu = load_imu_csv(p, on_bad_row="skip")
    assert len(imu) == 1
    assert capsys.readouterr().err == ""


def test_load_warns_about_non_monotonic_timestamps(tmp_path, capsys):
    """时间戳不单调会让插值给出无意义读数（实测能外推出负值）。"""
    p = _csv(tmp_path, "1.0,1,2,3\n0.5,1,2,3\n")
    load_imu_csv(p)
    assert "单调" in capsys.readouterr().err


# ---------------------------------------------------------------- gyro_at

def _imu3():
    return [ImuSample(0.0, np.array([0.0, 0.0, 0.0])),
            ImuSample(1.0, np.array([10.0, 10.0, 10.0]))]


def test_gyro_at_interpolates_midway():
    g, _ = gyro_at(_imu3(), 0, 0.5)
    assert np.allclose(g, [5.0, 5.0, 5.0])


def test_gyro_at_clamps_instead_of_extrapolating():
    """**核心回归**：t 超出范围时此前会线性**外推**。

    实测 t=-5.0 得到 [-14,-13,-12] —— 凭空造出负角速度。正确行为是夹到
    最近的样本（"t 比第一帧还早"≠"设备在反向旋转"）。
    """
    g, _ = gyro_at(_imu3(), 0, -5.0)
    assert np.allclose(g, [0.0, 0.0, 0.0]), "不应外推出负值"
    g2, _ = gyro_at(_imu3(), 0, 99.0)
    assert np.allclose(g2, [10.0, 10.0, 10.0]), "不应外推出更大值"


def test_gyro_at_rejects_empty_imu():
    """**核心回归**：空 IMU 此前返回全零 = 假装设备静止。

    这是本项目开篇定义的失效模式（固定监控能用、移动设备崩溃）。库不该
    替调用方做这个假设——降级方式应由 engine 的 no_imu 显式决定。
    """
    with pytest.raises(ConfigError, match="为空"):
        gyro_at([], 0, 0.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_gyro_at_rejects_non_finite_t(bad):
    """t=NaN 会让所有比较为假，静默走到最后返回一个非插值的结果。"""
    with pytest.raises(NonFiniteError, match="有限"):
        gyro_at(_imu3(), 0, bad)


def test_gyro_at_tolerates_out_of_range_hint():
    """hint 越界应被夹住（它是加速用的缓存，不是真值）。"""
    g, h = gyro_at(_imu3(), 999, 0.5)
    assert np.allclose(g, [5.0, 5.0, 5.0])
    assert 0 <= h < 2


# ------------------------------------------------------- compensate_rotation

def test_compensate_returns_zero_residual_for_identical_frames():
    a = np.zeros((16, 16), np.uint8)
    r = compensate_rotation(a, a, np.zeros(3), 1 / 30)
    assert r.shape == (16, 16)
    assert np.allclose(r, 0.0)


def test_compensate_first_frame_returns_zeros():
    """首帧没有可比的前一帧 —— 返回全零是**有意**的，不是缺陷。"""
    r = compensate_rotation(None, np.zeros((16, 16), np.uint8),
                            np.zeros(3), 1 / 30)
    assert np.allclose(r, 0.0)


def test_compensate_rejects_nan_gyro():
    """**核心回归**：NaN 会产生形状正确、数值全 NaN 的残差图。"""
    with pytest.raises(NonFiniteError, match="gyro"):
        compensate_rotation(np.zeros((16, 16), np.uint8),
                            np.zeros((16, 16), np.uint8),
                            np.array([np.nan, 0, 0]), 1 / 30)


def test_compensate_rejects_non_finite_dt():
    with pytest.raises(NonFiniteError, match="dt"):
        compensate_rotation(np.zeros((16, 16), np.uint8),
                            np.zeros((16, 16), np.uint8),
                            np.zeros(3), float("nan"))


def test_compensate_rejects_size_mismatch():
    """**核心回归**：此前静默返回**按 gray 尺寸**的结果，而内部 warp 用的是
    prev_gray 的尺寸 —— 等于悄悄错位。"""
    with pytest.raises(ShapeError, match="尺寸"):
        compensate_rotation(np.zeros((16, 16), np.uint8),
                            np.zeros((8, 8), np.uint8), np.zeros(3), 1 / 30)


def test_compensate_rejects_short_gyro():
    with pytest.raises(ShapeError, match="3 轴"):
        compensate_rotation(np.zeros((16, 16), np.uint8),
                            np.zeros((16, 16), np.uint8),
                            np.array([1.0, 2.0]), 1 / 30)


# ---------------------------------------------------------------- cells_of

def test_cells_of_averages_per_cell():
    r = np.zeros((8, 8), np.float64)
    r[:4, :4] = 1.0
    out = cells_of(r, (2, 2))
    assert out.shape == (4,)
    # 网格 (gx=2, gy=2) → reshape(gy,ch,gx,cw)，顺序是 [左上, 右上, 左下, 右下]
    assert np.allclose(sorted(out), [0.0, 0.0, 0.0, 1.0])


def test_cells_of_rejects_non_2d():
    with pytest.raises(ShapeError, match="2 维"):
        cells_of(np.zeros((4, 4, 3)), (2, 2))


def test_cells_of_rejects_grid_larger_than_image():
    """**核心回归**：此前切出空数组 → 静默全 NaN + 一条易忽略的 RuntimeWarning。"""
    with pytest.raises(ShapeError, match="小于网格"):
        cells_of(np.zeros((2, 2)), (4, 4))


def test_cells_of_rejects_empty():
    with pytest.raises(InsufficientDataError, match="空数组"):
        cells_of(np.zeros((0, 0)), (4, 4))


def test_cells_of_rejects_nan():
    with pytest.raises(NonFiniteError, match="NaN"):
        cells_of(np.full((8, 8), np.nan), (2, 2))


def test_cells_of_rejects_bad_grid():
    with pytest.raises(ConfigError, match="网格"):
        cells_of(np.zeros((8, 8)), (0, 2))


def test_cells_of_truncates_non_divisible_sizes():
    """不能整除时应截断到最大整数倍（既有行为，固化成契约）。"""
    out = cells_of(np.ones((7, 7)), (2, 2))
    assert out.shape == (4,) and np.allclose(out, 1.0)

"""Unit tests for cerebellum_cancel.rotwarp（完整旋转 warp）。

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import numpy as np
import pytest

from cerebellum_cancel.featmap_cancel import CancelConfig, FeatureMapCancel
from cerebellum_cancel.rotwarp import (rotation_homography,
                                       rotation_matrix_zyx,
                                       warp_by_rotation)


def test_zero_rotation_is_identity():
    assert np.allclose(rotation_matrix_zyx(0.0, 0.0, 0.0, 1 / 30), np.eye(3))
    F = np.random.default_rng(0).normal(size=(2, 8, 8)).astype(np.float32)
    out = warp_by_rotation(F, np.zeros(3), 1 / 30, f_map=10.0)
    assert np.allclose(out, F, atol=1e-5)


def test_rotation_matrix_is_orthonormal_and_right_handed():
    """无论角度多大，矩阵必须仍是旋转（正交、det=+1）——用 Rodrigues
    而不是小角近似，甩动时不失效。"""
    for g in ((30.0, 0.0, 0.0), (0.0, 400.0, 0.0), (50.0, 60.0, 70.0),
              (1000.0, -500.0, 300.0)):
        R = rotation_matrix_zyx(*g, dt=1 / 30)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_pure_roll_rotates_about_image_centre():
    """纯 roll（绕光轴）应产生**旋转**场：画布中心不动，边缘切向移动、半径
    不变。这正是一阶常数平移近似无法表达的部分（也是它输掉的原因）。

    用质心而不是 argmax 定位亮点：4°/帧的位移在本例是亚像素量级，argmax
    只会返回最近的整数格。
    """
    F = np.zeros((1, 24, 24), dtype=np.float32)
    F[0, 12, 18] = 1.0                       # 中心右侧 6 px 一个亮点
    g = np.array([0.0, 0.0, 120.0])          # 纯 roll，4°/帧
    out = warp_by_rotation(F, g, 1 / 30, f_map=10.0)
    assert out[0].max() > 0.5
    tot = float(out[0].sum())
    yy, xx = np.mgrid[0:24, 0:24]
    cy = float((out[0] * yy).sum() / tot)
    cx = float((out[0] * xx).sum() / tot)
    assert abs(cy - 12.0) > 0.05, "纯 roll 未把亮点移出水平线（没有旋转场）"
    r_in = 6.0
    r_out = float(np.hypot(cx - 12.0, cy - 12.0))
    assert abs(r_out - r_in) < 0.15, f"旋转场半径应不变: {r_in} -> {r_out:.3f}"


def test_homography_reduces_content_shift_better_than_constant_translation():
    """机制测试：在一个**已知旋转生成**的合成特征图上，完整旋转 warp 的
    预测误差必须小于"常数平移"近似——这是真机数据结论的可复现版本。"""
    f, dt = 12.0, 1 / 30
    g = np.array([40.0, 120.0, 90.0])        # 明显带 roll 的角速度
    # 用真旋转生成 F_t 与 F_{t-1}
    rng = np.random.default_rng(3)
    import cv2
    base = cv2.GaussianBlur(rng.normal(size=(2, 24, 24)).astype(np.float32),
                            (0, 0), 2.0)
    H = rotation_homography(g[0], g[1], g[2], dt, f, 12.0, 12.0)
    F_prev = base
    F_t = np.stack([cv2.warpPerspective(base[c], H, (24, 24),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                    for c in range(2)])

    pred_rot = warp_by_rotation(F_prev, g, dt, f)
    # 常数平移近似（管线原方法）
    from cerebellum_cancel.featmap_cancel import shift_map
    dx = f * np.tan(np.radians(g[1]) * dt)
    dy = f * np.tan(np.radians(g[0]) * dt)
    pred_shift = shift_map(F_prev, -dx, -dy)

    err_rot = float(np.abs(F_t - pred_rot).mean())
    err_shift = float(np.abs(F_t - pred_shift).mean())
    err_naive = float(np.abs(F_t - F_prev).mean())
    assert err_rot < err_shift, (
        f"旋转 warp 未优于常数平移: {err_rot:.4f} vs {err_shift:.4f}")
    assert err_rot < err_naive, "旋转 warp 未优于不补偿"


def test_use_roll_false_drops_the_roll_term():
    """use_roll=False 的消融开关必须真的丢掉 roll（真机实测其贡献约 4%）。"""
    F = np.random.default_rng(5).normal(size=(1, 12, 12)).astype(np.float32)
    g = np.array([0.0, 0.0, 90.0])            # 只有 roll
    with_roll = warp_by_rotation(F, g, 1 / 30, f_map=10.0, use_roll=True)
    without = warp_by_rotation(F, g, 1 / 30, f_map=10.0, use_roll=False)
    assert not np.allclose(with_roll, F, atol=1e-4)   # roll 有作用
    assert np.allclose(without, F, atol=1e-5)          # 关掉后为恒等


def test_feature_map_cancel_rotation_mode_runs_and_differs():
    """管线接入：cfg.pred="rotation" 可用，且在同一输入上给出与 shift 不同的
    预测（说明真的走了另一条分支）。"""
    rng = np.random.default_rng(7)
    import cv2
    base = cv2.GaussianBlur(rng.normal(size=(2, 16, 16)).astype(np.float32),
                            (0, 0), 2.0)
    fmaps = [base + rng.normal(0, 0.01, base.shape).astype(np.float32)
             for _ in range(20)]
    gyro = np.tile(np.array([30.0, 60.0, 45.0], dtype=np.float32), (20, 1))
    zs = []
    for pred in ("shift", "rotation"):
        c = FeatureMapCancel(CancelConfig(fps=30.0, f_map=12.0, robust=True,
                                          pred=pred))
        zz = np.stack([c.step(f, gyro[0])["z"] for f in fmaps])
        zs.append(zz)
    assert not np.allclose(zs[0], zs[1], atol=1e-4)
    assert np.all(np.isfinite(zs[1]))


def test_unknown_pred_mode_is_rejected_by_config_documented_values():
    """配置常量集合是文档化契约：仅 'shift' / 'rotation' 有意义。
    （FeatureMapCancel 不显式校验以保持向后兼容，这里锁定常量集合。）"""
    cfg = CancelConfig()
    assert cfg.pred in ("shift", "rotation")


def test_rotation_mode_gyro_lag_is_configurable_and_dataset_defaulted():
    """时序约定（实测得出，两个数据源相反，**不能全局取值**）：

    * ``gyro_lag=0`` -> 预测 F_t 用 ``gyro[t]``，即存储值描述的就是
      ``t-1 -> t`` 这一步（AoE：``prep_aoe.trajectory_gyro`` 直接存的就是它）。
    * ``gyro_lag=1`` -> 用上一帧读数（真机 IMU：``prep_real_imu`` 用当前帧
      读数标定，实测 0.835 vs 0.906）。

    这里锁定两点：(a) 两种 lag 给出不同结果（参数不是死参数）；(b) 未指定时
    按数据集路径取默认（AoE -> 0，real_imu -> 1）。
    """
    rng = np.random.default_rng(11)
    import cv2
    base = cv2.GaussianBlur(rng.normal(size=(1, 12, 12)).astype(np.float32),
                            (0, 0), 2.0)

    def run(g_third: np.ndarray, **kw) -> np.ndarray:
        c = FeatureMapCancel(CancelConfig(fps=30.0, f_map=10.0, robust=True,
                                          pred="rotation", **kw))
        c.step(base, np.array([0.0, 0.0, 0.0]))          # 帧 1
        c.step(base, np.array([30.0, 60.0, 45.0]))        # 帧 2
        return c.step(base, np.array(g_third))["z"]      # 帧 3

    third = np.array([999.0, -999.0, 500.0])
    # lag=1（旧默认）：帧 3 用的是帧 2 保存的 gyro，与当帧无关
    a = run(np.array([0.0, 0.0, 0.0]), gyro_lag=1)
    b = run(third, gyro_lag=1)
    assert np.allclose(a, b), "lag=1 时似乎用了当帧 gyro"
    # lag=0：帧 3 用当帧 gyro，故改变它必然改变结果
    c0 = run(np.array([0.0, 0.0, 0.0]), gyro_lag=0)
    c1 = run(third, gyro_lag=0)
    assert not np.allclose(c0, c1), "lag=0 时似乎没用到当帧 gyro"
    # -1 = 按数据集默认：AoE 为 0，真机 IMU 为 1
    for src, expect_lag in (("data/aoe_event/val/seq_000", 0),
                            ("data/real_imu/val/seq_000", 1)):
        d0 = run(np.array([0.0, 0.0, 0.0]), source=src)
        d1 = run(third, source=src)
        same = bool(np.allclose(d0, d1))
        assert same == (expect_lag == 1), f"{src} 默认 lag 与预期不符"


def test_warp_direction_matches_the_calibrated_shift_model():
    """方向约定（本线开发中被连续误导两轮的坑）：

    ``cv2.warpPerspective`` 的默认语义是 ``dst(x) = src(H⁻¹ x)``，而我们要的
    是 ``dst(x) = src(H x)``——即 H 把上一帧的点映射到这一帧的位置。因此
    **必须带 ``WARP_INVERSE_MAP``**。

    不加这个 flag 会把预测反向 warp：效果从降残差变成增残差，且只看残差
    数值发现不了（只会看到"变差了"）。此处用已标定自洽的 shift 模型作
    绝对参照——对同一份 gyro，两个模型把同一个点移到同一处才算方向正确。
    """
    from cerebellum_cancel.rotwarp import warp_by_rotation
    from cerebellum_cancel.featmap_cancel import shift_map, expected_shift
    f, dt, S = 18.86, 1 / 30.0, 28
    cfg = CancelConfig(fps=30.0, f_map=f)
    F = np.zeros((1, S, S), np.float32)
    F[0, 14, 20] = 1.0
    for g in (np.array([0.0, 60.0, 0.0]), np.array([0.0, -60.0, 0.0]),
              np.array([45.0, 0.0, 0.0])):
        dx, dy = expected_shift(g, cfg)
        p = shift_map(F, -dx, -dy)
        ys_s, xs_s = np.unravel_index(np.argmax(p[0]), p[0].shape)
        q = warp_by_rotation(F, g, dt, f)
        ys_r, xs_r = np.unravel_index(np.argmax(q[0]), q[0].shape)
        assert (xs_r, ys_r) == (xs_s, ys_s), (
            f"gyro={g} 时 rotation 把点移到 ({xs_r},{ys_r})，"
            f"而 shift 参照给出 ({xs_s},{ys_s})——方向约定反了")


def test_gyro_prev_round_trips_through_stats():
    """续采一致性：lag=1 下重启后必须复用上一帧 gyro，否则首帧预测退化为
    用当帧读数。lag=0 不需要它（用当帧），但一并落盘以便切换 lag。"""
    rng = np.random.default_rng(13)
    import cv2
    base = cv2.GaussianBlur(rng.normal(size=(2, 12, 12)).astype(np.float32),
                            (0, 0), 2.0)
    cfg = CancelConfig(fps=30.0, f_map=10.0, robust=True, pred="rotation",
                       gyro_lag=1)

    full = FeatureMapCancel(cfg)
    full.step(base, np.array([0.0, 0.0, 0.0]))
    full.step(base, np.array([20.0, 40.0, 30.0]))
    ref = full.step(base, np.array([10.0, 10.0, 10.0]))["z"]

    half = FeatureMapCancel(cfg)
    half.step(base, np.array([0.0, 0.0, 0.0]))
    half.step(base, np.array([20.0, 40.0, 30.0]))
    stats = half.dump_stats()
    assert "gyro_prev" in stats
    resumed = FeatureMapCancel(cfg)
    resumed.load_stats(stats)
    got = resumed.step(base, np.array([10.0, 10.0, 10.0]))["z"]
    assert np.allclose(got, ref, atol=1e-5)


def test_dataset_conventions_are_recorded():
    """锁定两个数据源的存储约定（符号 + gyro 配对）——这是实测标定值，
    改动会让历史结论漂移（见 MEMORY_FORMAT §8.7）。

    **符号与 ``WARP_INVERSE_MAP`` 是联合约定**：``(-s, 有 flag)`` 与
    ``(+s, 无 flag)`` 数值完全等价（实测 0.9655 == 0.9655）。本表记录的是
    **带 flag 时**该用的值。这一点极易踩坑：`rotwarp` 早期漏了 flag，当时
    标出的 `real_imu=(+,+,+)` 是"无 flag 时代"的值；补上 flag 后必须整体
    取反为 `(-,-,-)` 才能保持同一行为。若不取反，真机段残差会从 0.835 变成
    1.289（比不补偿还差）。

    真机 IMU：(-,-,-) + lag 1（实测 0.835，优于 lag 0 的 0.906）。
    AoE：(-,+,+) + lag 0（前半标定 / 后半评估，12 段 12/12 一致；与既有
    ``scripts/analyze_negatives.py`` 的 pitch+1 / yaw-1 / roll+1 吻合——那个
    脚本自带 flag，故可直接对照，无需再取反）。
    根因：``prep_aoe.trajectory_gyro`` 存的是 ``t-1 -> t`` 转角（故 lag=0），
    且其 ``calibrate_signs`` 强制 ``corr(-f·tan(g), dx) > 0``，而
    ``prep_real_imu.fit_calibration`` 只取 |corr| 最大、不约束符号。
    """
    from cerebellum_cancel.rotwarp import (DATASET_CONVENTIONS, signs_for,
                                          lag_for, convention_for)
    assert DATASET_CONVENTIONS["real_imu"] == {"signs": (-1.0, -1.0, -1.0),
                                               "lag": 1}
    assert DATASET_CONVENTIONS["aoe"] == {"signs": (-1.0, 1.0, 1.0),
                                          "lag": 0}
    # 路径解析
    assert signs_for("data/real_imu/val/seq_000") == (-1.0, -1.0, -1.0)
    assert lag_for("data/real_imu/val/seq_000") == 1
    assert signs_for("data/aoe_event/val/seq_003") == (-1.0, 1.0, 1.0)
    assert lag_for("data/aoe_event/val/seq_003") == 0
    # 未知来源：存储原样 + lag 0（不猜）
    assert convention_for("data/unknown/seq_000") == {
        "signs": (1.0, 1.0, 1.0), "lag": 0}


def test_roll_sign_flips_the_rotation_direction():
    """roll_sign 必须真的反转 roll 的作用方向（不是死参数）。"""
    rng = np.random.default_rng(17)
    F = np.zeros((1, 32, 32), dtype=np.float32)
    F[0, 8, 24] = 1.0                     # 中心右上一点
    g = np.array([0.0, 0.0, 120.0])       # 纯 roll
    a = warp_by_rotation(F, g, 1 / 30, f_map=20.0, roll_sign=1.0)
    b = warp_by_rotation(F, g, 1 / 30, f_map=20.0, roll_sign=-1.0)

    def com(img):
        t = float(img.sum())
        yy, xx = np.mgrid[0:32, 0:32]
        return float((img[0] * xx).sum() / t), float((img[0] * yy).sum() / t)

    ax, ay = com(a)
    bx, by = com(b)
    assert not np.isclose(ax, bx, atol=1e-3), "roll_sign 未反转水平方向"
    # 两者应绕中心对称（同半径、角度相反）
    r1 = np.hypot(ax - 16, ay - 16)
    r2 = np.hypot(bx - 16, by - 16)
    assert abs(r1 - r2) < 0.05, f"两侧半径不一致: {r1:.3f} vs {r2:.3f}"


# =====================================================================
# 输入契约（工业化稳定性，见 GUIDE §4.4）
#
# 这些用例固化的是**曾经静默通过**的路径。rotwarp 是本项目里"约定最密"的
# 模块（时序/符号/方向三个约定，且 GUIDE 明确记着"残差数值上看不出方向错"
# 的坑），却一度 0 校验——探针实测 9 条无效输入里 6 条静默产出结果。
# =====================================================================

def _F(seed: int = 0, shape=(4, 28, 28)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def test_warp_rejects_nan_gyro_instead_of_returning_nan_plane():
    """**最危险的一条**：gyro 含 NaN 时 warp 会把 NaN 铺满输出。

    调用方只看到"残差全是 NaN"，不会想到根源在 rotwarp——而 GUIDE 里已经
    记过"这个模块的错在残差数值上看不出来"。
    """
    from cerebellum_cancel.contracts import NonFiniteError
    with pytest.raises(NonFiniteError, match="gyro"):
        warp_by_rotation(_F(), np.array([np.nan, 0.0, 0.0]), 1 / 30, 42.7)


def test_warp_rejects_inf_gyro():
    from cerebellum_cancel.contracts import NonFiniteError
    with pytest.raises(NonFiniteError, match="gyro"):
        warp_by_rotation(_F(), np.array([np.inf, 0.0, 0.0]), 1 / 30, 42.7)


def test_warp_rejects_nan_feature_map():
    """F_prev 含 NaN 同样会铺满输出，且会污染后续所有统计。"""
    from cerebellum_cancel.contracts import NonFiniteError
    with pytest.raises(NonFiniteError, match="F_prev"):
        warp_by_rotation(np.full((4, 28, 28), np.nan),
                         np.array([1.0, 0.0, 0.0]), 1 / 30, 42.7)


@pytest.mark.parametrize("bad_dt", [float("nan"), float("inf"), -float("inf")])
def test_warp_rejects_non_finite_dt(bad_dt):
    from cerebellum_cancel.contracts import NonFiniteError
    with pytest.raises(NonFiniteError, match="dt"):
        warp_by_rotation(_F(), np.array([10.0, 0.0, 0.0]), bad_dt, 42.7)


def test_warp_rejects_wrong_gyro_arity():
    """从 IndexError（偶然）升级为 ShapeError（明确）。"""
    from cerebellum_cancel.contracts import ShapeError
    with pytest.raises(ShapeError, match="gyro"):
        warp_by_rotation(_F(), np.array([1.0, 2.0]), 1 / 30, 42.7)
    with pytest.raises(ShapeError, match="gyro"):
        warp_by_rotation(_F(), np.array([1.0, 2.0, 3.0, 4.0]), 1 / 30, 42.7)


@pytest.mark.parametrize("shape", [(28, 28), (2, 4, 28, 28), (28,)])
def test_warp_rejects_non_3d_feature_map(shape):
    """必须是 (C,H,W)：OpenCV 把多通道数组第一维当空间维，
    维度错了会静默 warp 错轴。"""
    from cerebellum_cancel.contracts import ShapeError
    with pytest.raises(ShapeError, match="F_prev|3 维"):
        warp_by_rotation(np.zeros(shape, dtype=np.float32),
                         np.array([1.0, 0.0, 0.0]), 1 / 30, 42.7)


def test_warp_rejects_zero_f_map():
    """f_map=0 让 K 不可逆（数学退化）。此前是 numpy 的 LinAlgError，
    现在给出说明性的 ConfigError。"""
    from cerebellum_cancel.contracts import ConfigError
    with pytest.raises(ConfigError, match="f_map"):
        warp_by_rotation(_F(), np.array([1.0, 0.0, 0.0]), 1 / 30, 0.0)


def test_warp_accepts_gyro_as_plain_list():
    """契约检查不该要求调用方先自己 np.asarray——但形状错误仍要报错。

    这条同时是"不误杀"的护栏：列表入参是合法用法（早期调用点就这么传）。
    """
    out = warp_by_rotation(_F(), [1.0, 2.0, 3.0], 1 / 30, 42.7)
    assert out.shape == (4, 28, 28)
    assert np.all(np.isfinite(out))


def test_dt_zero_and_negative_are_still_allowed():
    """**反向护栏**：``dt<=0`` 是合法输入。

    语义：无时间流逝（首帧、或时间戳重合）→ 旋转矩阵退化为单位阵。
    ``core.CerebellumCancel`` 对 dt 也有同样的允许（见 test_contracts）。
    加校验时最容易把这里"顺手修成"报错。
    """
    base = _F()
    for dt in (0.0, -1e-9):
        out = warp_by_rotation(base, np.array([50.0, 0.0, 0.0]), dt, 42.7)
        assert np.allclose(out, base, atol=1e-5), f"dt={dt} 应等价于不 warp"


def test_negative_f_map_is_deliberately_not_validated():
    """**记录一个刻意的设计留白**：f_map 为负不做校验。

    本项目尚未给它裁定语义（可能用来表达方向翻转，也可能是笔误），在没有
    明确约定的情况下拒绝它会禁掉未知的合法用法。此测试把"我们**知道**这里
    没校验、且是有意为之"固化下来——将来若定了语义，应改这里而不是悄悄加。

    注：负 f_map 产生的正是 GUIDE 里警告过的"方向错但数值看不出"那一类
    （同 WARP_INVERSE_MAP 缺失），所以若真要拒绝，理由很充分。
    """
    out = warp_by_rotation(_F(), np.array([1.0, 0.0, 0.0]), 1 / 30, -42.7)
    assert out.shape == (4, 28, 28)
    assert np.all(np.isfinite(out))


def test_validation_does_not_change_numerical_behaviour():
    """**回归护栏**：加校验只增加拒绝路径，不得改动任何数值。

    与手工重算的参考实现逐位比对——若将来有人在校验里顺手做了
    dtype 转换或 copy，这条会立刻失败。
    """
    import cv2
    F = _F(seed=7)
    g = np.array([12.3, -45.6, 7.8])
    dt, f = 1 / 30, 42.674
    out = warp_by_rotation(F, g, dt, f)

    H = rotation_homography(float(g[0]), float(g[1]), float(g[2]), dt, f,
                            14.0, 14.0)
    ref = np.empty_like(F)
    for ch in range(F.shape[0]):
        ref[ch] = cv2.warpPerspective(
            F[ch], H, (28, 28),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE)
    assert np.array_equal(out, ref), "校验改动影响了数值结果"
    # 确定性：同样输入必须同样输出
    assert np.array_equal(out, warp_by_rotation(F, g, dt, f))

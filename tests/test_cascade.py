"""Unit tests for cerebellum_cancel.cascade（多层残差预测）。

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import numpy as np
import pytest

from cerebellum_cancel.cascade import (CascadeCancel, CascadeConfig,
                                       rollout_cascade)
from cerebellum_cancel.featmap_cancel import CancelConfig


def _smooth_field(shape=(2, 12, 12), seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32),
                            (0, 0), 2.0)
    return base


def test_single_layer_matches_base_pipeline():
    """n_layers=1 必须与权威的 FeatureMapCancel 逐位一致——级联不能改变
    已发表的单层行为。"""
    from cerebellum_cancel.featmap_cancel import FeatureMapCancel, rollout
    shape = (2, 12, 12)
    rng = np.random.default_rng(1)
    base = _smooth_field(shape, 2)
    fmaps = np.stack([base + rng.normal(0, 0.01, shape).astype(np.float32)
                      for _ in range(30)])
    gyros = np.zeros((30, 3), dtype=np.float32)
    gyros[:, 1] = 20.0
    cfg = CancelConfig(fps=30.0, f_map=2.0, robust=True)

    z, sur_ref = rollout(FeatureMapCancel(cfg), fmaps, gyros)
    sur_casc, ls, _ = rollout_cascade(
        CascadeCancel(CascadeConfig(cancel=cfg, n_layers=1)), fmaps, gyros)
    assert np.allclose(sur_casc, sur_ref, atol=1e-5)
    assert ls.shape == (30, 1)


def test_rejects_zero_layers():
    with pytest.raises(ValueError):
        CascadeCancel(CascadeConfig(n_layers=0))


def test_layers_reduce_raw_residual_on_predictable_drift():
    """级联的机制测试。两个要点（都是实测踩出来的）：

    1. **度量必须是原始残差，不是 z 峰值**。z 是归一化的：一个成功的预测器
       会把残差和尺度同时压小，z 保持 O(1)——用 z 衡量"层数有没有用"会得出
       相反的结论。项目既有口径（§2.3.1 的 calm residual）正是原始残差。
    2. **什么结构对第 2 层可预测**：静态偏置会被一阶帧差自动消掉；可预测的是
       **帧差后仍留下的恒定残余**（慢漂移的导数），对应真实数据里的自动对焦
       呼吸、曝光适应这类长期残余。
    """
    shape = (2, 12, 12)
    rng = np.random.default_rng(3)
    base = _smooth_field(shape, 3)
    drift = np.zeros(shape, dtype=np.float32)
    drift[:, 3:6, 3:6] = 0.02                      # 局部慢漂移（每帧增量）
    n = 120
    fmaps = np.stack([
        base + drift * i + rng.normal(0, 0.002, shape).astype(np.float32)
        for i in range(n)])
    gyros = np.zeros((n, 3), dtype=np.float32)

    cfg_kw = CancelConfig(fps=30.0, f_map=2.0, robust=True)
    canc1 = CascadeCancel(CascadeConfig(cancel=cfg_kw, n_layers=1,
                                        slow_ema=0.05))
    canc3 = CascadeCancel(CascadeConfig(cancel=cfg_kw, n_layers=3,
                                        slow_ema=0.05))
    # 逐帧收集原始残差（跨通道均值绝对误差）
    raw1, raw3 = [], []
    for i in range(n):
        raw1.append(float(np.abs(canc1.step(fmaps[i], gyros[i])["r"]).mean()))
        raw3.append(float(np.abs(canc3.step(fmaps[i], gyros[i])["r"]).mean()))
    tail = slice(60, None)                          # 跳过收敛期
    m1, m3 = np.median(raw1[tail]), np.median(raw3[tail])
    assert m3 < m1, f"级联未降原始残差: 1 层 {m1:.4f} vs 3 层 {m3:.4f}"
    # 收益幅度：这个场景里可预测成分只占少数格（9/144），故留出宽松下界
    assert m3 < 0.8 * m1, f"级联收益过小: 1 层 {m1:.4f} vs 3 层 {m3:.4f}"


def test_gate_protects_events_from_being_learned_away():
    """稳态门的作用：一个突发的、孤立的尖峰**不能**被第 2 层学成预测
    （否则就是把要检测的东西取消了——论文里负像学习的核心约束）。

    度量用原始残差：事件帧的残差必须仍然显著高于平静帧。
    """
    shape = (2, 12, 12)
    rng = np.random.default_rng(5)
    base = _smooth_field(shape, 5)
    drift = np.zeros(shape, dtype=np.float32)
    drift[:, 1:4, 9:12] = 0.02                     # 背景有可预测漂移
    fmaps = [base + drift * i
             + rng.normal(0, 0.002, shape).astype(np.float32)
             for i in range(80)]
    # 第 60 帧注入一次强事件
    fmaps[60] = fmaps[60].copy()
    fmaps[60][:, 4:8, 4:8] += 3.0
    fmaps = np.stack(fmaps)
    gyros = np.zeros((80, 3), dtype=np.float32)

    for gate_k in (4.0, 0.0):
        canc = CascadeCancel(CascadeConfig(
            cancel=CancelConfig(fps=30.0, f_map=2.0, robust=True),
            n_layers=2, slow_ema=0.05, gate_k=gate_k))
        raw = [float(np.abs(canc.step(fmaps[i], gyros[i])["r"]).mean())
               for i in range(80)]
        calm_med = float(np.median(raw[40:60]))
        assert raw[60] > 2.0 * calm_med, (
            f"gate_k={gate_k} 下事件被级联吸收掉了: "
            f"event {raw[60]:.4f} vs calm {calm_med:.4f}")


def test_cascade_does_not_help_on_pure_noise():
    """负结果护栏：逐帧独立的噪声**没有**可预测结构时，多加一层不应降低
    原始残差（否则说明末层只是把自己拟合进了噪声——虚假提升）。

    注意度量是原始残差；z 分数在这里也会下降（尺度被压低），用它判定会
    得到错误结论。
    """
    shape = (2, 12, 12)
    rng = np.random.default_rng(7)
    fmaps = rng.normal(size=(60, *shape)).astype(np.float32)   # 逐帧独立噪声
    gyros = np.zeros((60, 3), dtype=np.float32)
    cfg_kw = CancelConfig(fps=30.0, f_map=2.0, robust=True)
    c1 = CascadeCancel(CascadeConfig(cancel=cfg_kw, n_layers=1))
    c2 = CascadeCancel(CascadeConfig(cancel=cfg_kw, n_layers=2))
    r1, r2 = [], []
    for i in range(60):
        r1.append(float(np.abs(c1.step(fmaps[i], gyros[i])["r"]).mean()))
        r2.append(float(np.abs(c2.step(fmaps[i], gyros[i])["r"]).mean()))
    tail = slice(20, None)
    m1, m2 = np.median(r1[tail]), np.median(r2[tail])
    # 噪声上没有可预测结构：第 2 层不该取得实质收益
    assert m2 > 0.95 * m1, f"纯噪声上出现虚假收益: 1 层 {m1:.4f} vs 2 层 {m2:.4f}"


def test_stats_roundtrip_enables_continuation():
    """级联状态必须可落盘续采：中断后从侧车续跑 == 不中断地跑。"""
    shape = (2, 10, 10)
    rng = np.random.default_rng(11)
    base = _smooth_field(shape, 11)
    bias = np.zeros(shape, dtype=np.float32)
    bias[:, 2:5, 2:5] = 0.4
    fmaps = np.stack([base + bias
                      + rng.normal(0, 0.01, shape).astype(np.float32)
                      for _ in range(60)])
    gyros = np.zeros((60, 3), dtype=np.float32)
    cfg = CascadeConfig(cancel=CancelConfig(fps=30.0, f_map=2.0, robust=True),
                        n_layers=2)

    # 参照：跑到底
    full = CascadeCancel(cfg)
    ref = []
    for i in range(60):
        out = full.step(fmaps[i], gyros[i])
        if i >= 30:
            ref.append(out["surprise"])

    # 中断：跑前 30 帧 -> 落盘 -> 载入 -> 续跑
    half = CascadeCancel(cfg)
    for i in range(30):
        half.step(fmaps[i], gyros[i])
    stats = half.dump_stats()
    assert "cascade_pred_0" in stats and "cascade_psd_0" in stats

    resumed = CascadeCancel(cfg)
    resumed.load_stats(stats)
    got = [resumed.step(fmaps[i], gyros[i])["surprise"] for i in range(30, 60)]
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_load_stats_rejects_layer_shape_mismatch():
    cfg = CascadeConfig(cancel=CancelConfig(fps=30.0, f_map=2.0), n_layers=2)
    canc = CascadeCancel(cfg)
    bad = {"mu": np.zeros((2, 5, 5), dtype=np.float32),
           "var": np.ones((2, 5, 5), dtype=np.float32),
           "cascade_pred_0": np.zeros((2, 9, 9), dtype=np.float32),
           "cascade_psd_0": np.ones((2, 9, 9), dtype=np.float32), "n": 3}
    with pytest.raises(ValueError):
        canc.load_stats(bad)


def test_layer_surprise_has_one_column_per_layer():
    shape = (1, 8, 8)
    fmaps = np.random.default_rng(13).normal(
        size=(20, *shape)).astype(np.float32)
    gyros = np.zeros((20, 3), dtype=np.float32)
    for n_layers in (1, 2, 4):
        cfg = CascadeConfig(cancel=CancelConfig(fps=30.0, f_map=1.0),
                            n_layers=n_layers)
        sur, ls, r = rollout_cascade(CascadeCancel(cfg), fmaps, gyros)
        assert sur.shape == (20,)
        assert ls.shape == (20, n_layers)
        assert r.shape == (20, *shape)


def test_unknown_layer_kind_is_rejected():
    with pytest.raises(ValueError):
        CascadeCancel(CascadeConfig(n_layers=2, layer_kind="bogus"))


def test_spatial_layer_kind_removes_local_background():
    """spatial 层（中心-周边去相关）：当**残差背景是空间平滑的**（帧差消不掉
    的平滑漂移）而**事件是局部尖峰**时，该层把平滑背景减掉、保住尖峰。这是
    ON/OFF 对抗通道的工程对应物。

    场景要点：漂移必须是**空间不均匀**的。全局均匀漂移会被一阶帧差完全抵消
    （残差为 0），那就测不出这一层的作用——这个坑在初版测试里踩到过。

    度量用**局部对比度**（尖峰处残差 / 周边残差），比值形式不受整体增益缩小
    影响——这正是区分"真预测"与"只是缩小增益"的判据。
    """
    shape = (1, 24, 24)
    rng = np.random.default_rng(17)
    # 空间平滑但不均匀的漂移场：帧差后残余 = 该场的空间梯度（非零）
    yy, xx = np.mgrid[0:24, 0:24].astype(np.float32)
    drift = 0.05 * (xx / 24.0) + 0.03 * np.sin(yy / 3.0)
    drift = drift.astype(np.float32)[None]
    n = 25
    fmaps = []
    for i in range(n):
        fmaps.append(drift * i + rng.normal(0, 0.001, shape).astype(np.float32))
    spike = fmaps[-1].copy()
    spike[:, 10:14, 10:14] += 1.0               # 局部事件

    def run(L: int) -> tuple[float, float]:
        cfg = CascadeConfig(cancel=CancelConfig(fps=30.0, f_map=1.0,
                                                robust=True),
                            n_layers=L, layer_kind="spatial")
        c = CascadeCancel(cfg)
        for f in fmaps[:-1]:
            c.step(f, np.zeros(3))              # 建立基线
        r = np.abs(c.step(spike, np.zeros(3))["r"])
        centre = float(r[:, 10:14, 10:14].mean())
        surround = float(np.concatenate([
            r[:, 4:9, 4:20].ravel(), r[:, 15:20, 4:20].ravel()]).mean())
        return centre, surround

    c1, s1 = run(1)
    c2, s2 = run(2)
    assert s1 > 0 and s2 > 0
    ratio1, ratio2 = c1 / s1, c2 / s2
    # 平滑背景在 spatial 层被抵消：周边残差下降
    assert s2 < s1, f"spatial 层未压低平滑背景: L1 {s1:.4f} vs L2 {s2:.4f}"
    # 尖峰保住，故局部对比度上升
    assert ratio2 > ratio1, (
        f"spatial 层未提升局部对比度: L1 {ratio1:.2f} vs L2 {ratio2:.2f}")

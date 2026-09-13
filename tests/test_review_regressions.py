"""回归测试：2026-09-13 代码审查发现的隐藏 bug。

这些 bug 的共同特征都是**静默**的——不报错、不让 `make check` 变红，
只让产物里的数字或区间悄悄错掉。每一个都先写出"注入即失败"的判据。
"""

from __future__ import annotations

import numpy as np
import pytest

from cerebellum_cancel.engine import (CMCIResult, EngineConfig, Segment,
                                      _merge)
from cerebellum_cancel.featmap_cancel import (CancelConfig, maxcell_score,
                                              energy_score)

# --------------------------------------------------------------- 1. fps 贯通


def test_engine_passes_source_fps_into_cancel_config(monkeypatch):
    """``engine.run`` 必须把源的 fps 传进 ``CancelConfig``。

    修复前：``AcquisitionLayer(frontend=...)`` 没给 ``cancel_cfg``，
    于是 ``CancelConfig.fps`` 恒为默认 30.0。``FeatureMapCancel`` 的 warp
    帧间隔取 ``1/cfg.fps``，所以 60 fps 视频会按 1/30 秒预测位移——
    **预测位移翻倍、残差被人为放大**，而产物 meta 里写的是 fps=60。

    判据锚定独立真值：检查图层**实际使用**的 fps（``layer.fps``，
    它来自 ``cancel.cfg.fps``）是否等于源的 fps。
    """
    import cerebellum_cancel.engine as E
    from cerebellum_cancel.acquisition import AcquisitionLayer

    captured: dict = {}
    real_init = AcquisitionLayer.__init__

    class _Layer(AcquisitionLayer):
        def __init__(self, *a, **kw):
            real_init(self, *a, **kw)
            captured["fps"] = self.cancel.cfg.fps

    # 用真前端 callable 绕开 torch：形状固定 (1? ,4,4) 足够
    monkeypatch.setattr(E, "AcquisitionLayer",
                        lambda frontend="x", cancel_cfg=None, **kw:
                        _Layer(frontend=lambda f: np.zeros((2, 8, 8),
                                                           np.float32),
                               cancel_cfg=cancel_cfg, **kw))

    class Src:
        def __init__(self, fps, n):
            self._fps, self._n = fps, n

        def __len__(self):
            return self._n

        def frames(self):
            for i in range(self._n):
                yield (np.zeros((32, 32, 3), np.uint8),
                       np.zeros(3, np.float64), i / self._fps)

        @property
        def meta(self):
            return {"source": "fake", "fps": self._fps, "imu": None,
                    "motion_source": "imu"}

    res = E.run("ignored.mp4", None, cfg=EngineConfig(frontend="noop"),
                source=Src(60.0, 40))
    assert captured["fps"] == pytest.approx(60.0), (
        f"CancelConfig.fps 未跟随源的 fps：实得 {captured['fps']}。"
        "warp 的 dt 会是 1/30，60fps 数据上预测位移翻倍。")
    assert res.meta["fps"] == pytest.approx(60.0)


def test_engine_fps_defaults_to_30_when_source_silent(monkeypatch):
    """源不报 fps 时仍要给出一个可用值（不能把 None 传进 CancelConfig）。"""
    import cerebellum_cancel.engine as E
    from cerebellum_cancel.acquisition import AcquisitionLayer

    captured: dict = {}
    real_init = AcquisitionLayer.__init__

    class _Layer(AcquisitionLayer):
        def __init__(self, *a, **kw):
            real_init(self, *a, **kw)
            captured["fps"] = self.cancel.cfg.fps

    monkeypatch.setattr(E, "AcquisitionLayer",
                        lambda frontend="x", cancel_cfg=None, **kw:
                        _Layer(frontend=lambda f: np.zeros((2, 8, 8),
                                                           np.float32),
                               cancel_cfg=cancel_cfg, **kw))

    class Src:
        def __len__(self):
            return 40

        def frames(self):
            for i in range(40):
                yield (np.zeros((32, 32, 3), np.uint8), np.zeros(3), i / 30.0)

        @property
        def meta(self):
            return {"source": "fake", "fps": None, "imu": None,
                    "motion_source": "imu"}

    E.run("x.mp4", None, cfg=EngineConfig(frontend="noop"), source=Src())
    assert captured["fps"] == pytest.approx(30.0)


# --------------------------------------------------- 2. _merge 不得有副作用


def test_merge_does_not_mutate_caller_segments():
    """``_merge`` 必须是纯函数——修复前它会原地改写调用方的 Segment。

    修复前：``stored_seconds`` / ``summary()`` 都调 ``_merge``，
    第一次读就把 ``segments[0].end`` 从 1.0 改成 2.0、n_frames 从 10 改成 20，
    而该对象仍留在 ``res.segments`` 里——导出器随后看到的是被改写的片段。
    """
    a = Segment(0.0, 1.0, 10, 1.0, 2.0, "motion_dedup")
    b = Segment(0.5, 2.0, 45, 1.0, 3.0, "motion_dedup")
    before = a.as_dict()
    out = _merge([a, b])
    assert a.as_dict() == before, (
        f"_merge 改写了入参: {before} -> {a.as_dict()}")
    assert out[0] is not a, "_merge 返回了入参对象本身"
    assert out[0].end == pytest.approx(2.0)
    assert out[0].n_frames == 55
    assert out[0].peak_score == pytest.approx(3.0)
    # b 未被合并进 a 之外的副本，原对象也不该被动
    assert b.end == pytest.approx(2.0)


def test_stored_seconds_is_idempotent_and_segments_unchanged():
    """反复读 ``stored_seconds`` 不得改变 ``segments``。"""
    n = 90
    segs = [Segment(0.0, 1.0, 30, 1.0, 2.0, "motion_dedup"),
            Segment(0.5, 2.0, 45, 1.0, 3.0, "motion_dedup")]
    res = CMCIResult(t=np.arange(n, dtype=np.float64) / 30.0,
                     gyro=np.zeros((n, 3), np.float32), score=np.zeros(n),
                     surprise=np.zeros(n, np.float32),
                     keep=np.zeros(n, np.uint8),
                     weight=np.ones(n, np.float32), segments=segs)
    snap = [s.as_dict() for s in res.segments]
    s1 = res.stored_seconds
    s2 = res.stored_seconds
    assert s1 == pytest.approx(s2)
    assert [s.as_dict() for s in res.segments] == snap, (
        "stored_seconds 改写了 res.segments")


# ----------------------------------------- 3. CMCIResult 时间轴/keep 校验


def _result(*, n: int = 5, t=None, keep=None) -> CMCIResult:
    """构造一个最小合法的 CMCIResult，供校验类测试注入非法字段。

    显式列关键字（不用 ``**dict`` 展开）：pyright 对 ``**dict`` 展开成
    dataclass 参数会报错（见仓库约定）。
    """
    return CMCIResult(
        t=np.arange(n, dtype=np.float64) / 30.0 if t is None else t,
        gyro=np.zeros((n, 3), np.float32),
        score=np.zeros(n),
        surprise=np.zeros(n, np.float32),
        keep=np.zeros(n, np.uint8) if keep is None else keep,
        weight=np.ones(n, np.float32),
    )


def test_result_rejects_nan_time_axis():
    """NaN 时间轴必须拒绝——修复前它静默进入产物。

    修复前：``t`` 含 NaN 时 ``duration`` 仍返回有限值（差值被后来的有限值
    掩盖），记录格式里的 ``t`` 却带着 NaN 落盘。``dataset.build_section``
    早就查这一条，引擎入口漏了——同一个非法输入两条路径行为不一致。
    """
    from cerebellum_cancel.contracts import NonFiniteError

    t = np.array([0.0, 0.033, np.nan, 0.1, 0.133])
    with pytest.raises(NonFiniteError):
        _result(t=t)


def test_result_rejects_inf_time_axis():
    from cerebellum_cancel.contracts import NonFiniteError

    t = np.array([0.0, 1.0, np.inf, 3.0, 4.0])
    with pytest.raises(NonFiniteError):
        _result(t=t)


def test_result_rejects_non_binary_keep():
    """``keep`` 必须是 0/1——修复前 keep=200 会让 keep_fraction=200.0。"""
    from cerebellum_cancel.contracts import ShapeError

    with pytest.raises(ShapeError):
        _result(keep=np.full(5, 200, np.uint8))


def test_result_accepts_binary_keep_as_bool_and_uint8():
    for k in (np.zeros(5, bool), np.ones(5, np.uint8)):
        r = _result(keep=k)
        assert r.keep_fraction in (0.0, 1.0)


# -------------------------------------------- 4. border 过大的可读错误


def test_maxcell_score_rejects_border_larger_than_grid():
    """border 过大要给出**可读**的 ValueError，而不是 numpy 的 zero-size 原生错误。"""
    z = np.zeros((1, 2, 5, 5), np.float32)
    with pytest.raises(ValueError) as ei:
        maxcell_score(z, border=3)
    msg = str(ei.value)
    assert "border" in msg and "过大" in msg, (
        f"错误信息未说明是 border 问题: {msg}")


def test_energy_score_rejects_border_larger_than_grid():
    z = np.zeros((1, 2, 5, 5), np.float32)
    with pytest.raises(ValueError) as ei:
        energy_score(z, border=3)
    assert "border" in str(ei.value)


def test_scores_agree_on_single_hot_cell():
    """守卫不得改变正常路径的数值（对照既有语义）。"""
    z = np.zeros((3, 2, 8, 8), dtype=np.float32)
    z[1, 0, 4, 4] = 3.0
    assert energy_score(z)[1] == pytest.approx(9.0)
    assert maxcell_score(z)[1] == pytest.approx(9.0)


def test_cancel_step_rejects_border_larger_than_grid():
    """``FeatureMapCancel.step`` 在 border 过大时也要给可读错误。"""
    from cerebellum_cancel.featmap_cancel import FeatureMapCancel

    cfg = CancelConfig(fps=30.0, f_map=1.0, border=2)
    c = FeatureMapCancel(cfg)
    with pytest.raises(ValueError) as ei:
        c.step(np.zeros((1, 4, 4), np.float32), np.zeros(3))
    assert "border" in str(ei.value)


# --------------------------------- 5. record read_record 只解析一次页脚


def test_read_record_parses_footer_exactly_once(tmp_path, monkeypatch):
    """``read_record`` 修复前把页脚（含整个 npz 侧车）解析了两遍。

    判据用调用计数：第一次 try 里解一次，紧接着又解一次。改成计数后
    "多解一次"必然被抓住（不是比较两个投影）。
    """
    import cerebellum_cancel.record as rec

    calls = []
    orig = rec._load_footer

    def counting(fh, off):
        calls.append(off)
        return orig(fh, off)

    monkeypatch.setattr(rec, "_load_footer", counting)
    p = tmp_path / "t.cmcir"
    with rec.RecordWriter(str(p), fps=30.0, f_map=1.0, shape=(1, 4, 4)) as w:
        for i in range(4):
            w.append(np.zeros((1, 4, 4), np.float32), t=i / 30.0,
                     score=float(i), keep=True)
    with rec.read_record(str(p)) as r:
        assert r.n_frames == 4
    assert len(calls) == 1, f"_load_footer 被调用了 {len(calls)} 次（期望 1）"


# --------------------------------------- 6. compute_keep 与逐帧决策一致


def test_compute_keep_matches_per_frame_trigger_output():
    """``compute_keep`` 必须**逐位等于**逐帧 ``update`` 的输出。

    修复前：末尾有一段"按 clips 重刷掩码"，把 clips 整段置 1。实测多标
    9 帧（那些帧在 update 里明确是 False）——因为 ``clips`` 的起点含
    ``pre_roll`` 前移，而逐帧决策没有。``keep`` 是"触发器此刻是否在片段内"
    （GUIDE §4.8 的不变量），含前缀的区间归 ``clips``/``Segment`` 管。
    两种语义混用会让同一数据集里出现互相矛盾的"存/不存"记录。

    判据锚定**独立真值**：真值来自直接驱动 AdaptiveTrigger 的逐帧输出，
    不是 compute_keep 自身的另一个投影。
    """
    from cerebellum_cancel.core import AdaptiveTrigger, TriggerConfig
    from cerebellum_cancel.dataset import compute_keep

    rng = np.random.default_rng(0)
    n = 400
    score = np.concatenate([rng.normal(1, 0.2, 200), rng.normal(30, 3, 40),
                            rng.normal(1, 0.2, 160)]).astype(np.float64)
    fps = 30.0
    cfg = TriggerConfig(robust=True)

    # 真值：直接驱动 AdaptiveTrigger 的逐帧输出
    tr = AdaptiveTrigger(cfg)
    ref = np.zeros(n, np.uint8)
    onset_t = None
    for i in range(1, n):
        ref[i] = 1 if tr.update(i / fps, float(score[i]), fps) else 0
        if onset_t is None and ref[i]:
            onset_t = i / fps
    clips = tr.finalize(n / fps)

    # **前置条件**：本测试的判别力完全来自"clip 起点含 pre_roll 前缀"。
    # 若 pre_roll 失效（那是另一个 bug，由 test_trigger_finalize_applies_
    # pre_roll 负责），clips 与逐帧决策恰好逐位一致，**重刷掩码退化为无操作**，
    # 本测试就会空转通过——实测确认过这个耦合。
    # 所以先把前置条件钉住，避免"pre_roll 坏掉"静默地让本测试也失效。
    assert onset_t is not None, "序列未触发任何片段，本测试无判别力"
    assert clips, "finalize 未给出片段，本测试无判别力"
    assert clips[0][0] < onset_t - 1e-9, (
        f"clip 起点 ({clips[0][0]:.4f}) 未早于 onset ({onset_t:.4f})——"
        "pre_roll 前缀没生效，本测试将失去判别力（它测的是"
        "'前缀不该被写进逐帧 keep'）。请先修 pre_roll。")

    got = compute_keep(score, fps, cfg)

    diff = int(np.count_nonzero(got != ref))
    assert diff == 0, (
        f"compute_keep 与逐帧决策差 {diff} 帧"
        f"（keep 数 {int(got.sum())} vs {int(ref.sum())}）。"
        "clips 的 pre_roll 前缀不应被写进逐帧 keep。")


def test_compute_keep_is_consistent_across_calls():
    """同样的输入必须给同样的 keep（compute_keep 不得有隐藏状态）。"""
    from cerebellum_cancel.dataset import compute_keep

    rng = np.random.default_rng(1)
    score = np.abs(rng.normal(size=300)).astype(np.float64)
    a = compute_keep(score, 30.0)
    b = compute_keep(score, 30.0)
    assert np.array_equal(a, b)


# ------------------------- 7. 视觉降级路径必须真的影响计算（pred=rotation）


def test_visual_fallback_uses_rotation_prediction(monkeypatch):
    """``motion_source='visual'`` 时必须用 ``pred='rotation'``。

    修复前：``CancelConfig`` 恒为默认 ``pred="shift"``，而
    ``gyro_equivalent`` **只填 roll**、``shift`` 模式**只读 pitch/yaw**——
    两者叠加使整条视觉降级路径**对结果零影响**：照样付 0.96 ms/帧做特征
    匹配（比 IMU 贵 29 倍），算出 roll 后丢进一个不看 roll 的函数，
    而 meta 里还写着 "self-motion estimated VISUALLY (degraded path)"。

    判据锚定独立真值：检查取消层**实际配置**的 pred，而不是比较输出
    （输出在两种 pred 下可能碰巧接近，那是弱判据）。
    """
    import cerebellum_cancel.engine as E
    from cerebellum_cancel.acquisition import AcquisitionLayer

    captured: dict = {}
    real_init = AcquisitionLayer.__init__

    class _Layer(AcquisitionLayer):
        def __init__(self, *a, **kw):
            real_init(self, *a, **kw)
            captured["pred"] = self.cancel.cfg.pred

    monkeypatch.setattr(E, "AcquisitionLayer",
                        lambda frontend="x", cancel_cfg=None, **kw:
                        _Layer(frontend=lambda f: np.zeros((2, 8, 8),
                                                           np.float32),
                               cancel_cfg=cancel_cfg, **kw))

    class Src:
        def __len__(self):
            return 40

        def frames(self):
            for i in range(40):
                yield (np.zeros((32, 32, 3), np.uint8), np.zeros(3), i / 30.0)

        @property
        def meta(self):
            return {"source": "fake", "fps": 30.0, "imu": None,
                    "motion_source": "visual"}

    E.run("x.mp4", None, cfg=EngineConfig(frontend="noop"), source=Src())
    assert captured["pred"] == "rotation", (
        f"视觉降级用了 pred={captured['pred']!r}；gyro_equivalent 只填 roll，"
        "shift 模式只读 pitch/yaw，故视觉估计对计算零影响。")


def test_imu_path_keeps_shift_prediction(monkeypatch):
    """有 IMU 的主路径保持默认 ``pred='shift'``——不得改变已发布数字。

    这是"修复不能改坏别处"的护栏：真 IMU 的 gyro 三个轴都有值，
    ``shift`` 是已发布口径。
    """
    import cerebellum_cancel.engine as E
    from cerebellum_cancel.acquisition import AcquisitionLayer

    captured: dict = {}
    real_init = AcquisitionLayer.__init__

    class _Layer(AcquisitionLayer):
        def __init__(self, *a, **kw):
            real_init(self, *a, **kw)
            captured["pred"] = self.cancel.cfg.pred

    monkeypatch.setattr(E, "AcquisitionLayer",
                        lambda frontend="x", cancel_cfg=None, **kw:
                        _Layer(frontend=lambda f: np.zeros((2, 8, 8),
                                                           np.float32),
                               cancel_cfg=cancel_cfg, **kw))

    class Src:
        def __len__(self):
            return 40

        def frames(self):
            for i in range(40):
                yield (np.zeros((32, 32, 3), np.uint8),
                       np.array([10.0, 20.0, 5.0]), i / 30.0)

        @property
        def meta(self):
            return {"source": "fake", "fps": 30.0, "imu": "imu.csv",
                    "motion_source": "imu"}

    E.run("x.mp4", None, cfg=EngineConfig(frontend="noop"), source=Src())
    assert captured["pred"] == "shift"


def test_shift_prediction_ignores_roll_only_signal():
    """这个测试把 bug 的**机制**钉住：shift 模式对只含 roll 的 gyro 不敏感。

    它是上面两个测试的"为什么"——如果哪天 shift 改成会用 roll，
    这个测试会失败，提示我们去重新审视 pred 的选择。
    """
    from cerebellum_cancel.featmap_cancel import (CancelConfig,
                                                  expected_shift)

    cfg = CancelConfig(fps=30.0, f_map=24.25, pred="shift")
    zero = expected_shift(np.zeros(3), cfg)
    roll_only = expected_shift(np.array([0.0, 0.0, 45.0]), cfg)
    assert np.allclose(zero, roll_only), (
        "shift 模式开始对 roll 敏感了——请重新审视 engine 的 pred 选择逻辑")


def test_rotation_prediction_does_use_roll():
    """对照面：rotation 模式确实使用 roll（所以它才是正确的降级选择）。"""
    from cerebellum_cancel.rotwarp import warp_by_rotation

    F = np.random.default_rng(0).normal(size=(2, 8, 8)).astype(np.float32)
    dt = 1 / 30
    w0 = warp_by_rotation(F, np.zeros(3), dt, 24.25)
    w1 = warp_by_rotation(F, np.array([0.0, 0.0, 45.0]), dt, 24.25)
    assert not np.allclose(w0, w1), "rotation 模式未使用 roll"


def test_gyro_equivalent_only_fills_roll():
    """钉住 ``gyro_equivalent`` 的输出契约：只填 roll。

    这是决定"必须配 rotation 模式"的上游事实。若它将来开始填 pitch/yaw，
    本测试失败 -> 提示重新评估 engine 的 pred 选择。
    """
    import cv2

    from cerebellum_cancel.visual_motion import VisualMotionEstimator

    rng = np.random.default_rng(2)
    # 造一个有纹理、且旋转了的图对，保证估计成功
    base = rng.integers(0, 255, (96, 96), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    M = cv2.getRotationMatrix2D((48, 48), 3.0, 1.0)
    rotated = cv2.warpAffine(base, M, (96, 96))

    est = VisualMotionEstimator()
    g = est.gyro_equivalent(base.astype(np.float64),
                            rotated.astype(np.float64),
                            f_map=277.0, dt=1 / 30)
    assert g.shape == (3,)
    assert g[0] == 0.0 and g[1] == 0.0, (
        f"gyro_equivalent 不应填 pitch/yaw（实得 {g}）——"
        "若契约变了，engine 的 pred 选择需同步重新评估")

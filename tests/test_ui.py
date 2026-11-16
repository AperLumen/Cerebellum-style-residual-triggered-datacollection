"""前端（cmci_ui）与其 API 的测试。

前端是**薄壳**，所以测试重点不是"界面好不好看"，而是三条纪律：

1. **前端不算东西。** 所有数值都来自引擎。测试方式是：UI 的 `run` 结果
   必须与直接调 `engine.run` 的摘要一致。
2. **调参走 replay，且不改变不该改的。** `replay` 后的 score/surprise/t
   必须与原来逐位相同——否则"调参"就成了"换了一套数据"。
3. **错误要给人话。** 路径写错、参数非法，返回的是可读句子，不是堆栈。

API 层直接调 act_* 函数（不起 HTTP 服务），因为 HTTP 层只是 JSON 转发，
没有逻辑；起真服务会让测试变慢且易受端口占用影响。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from cerebellum_cancel.core import TriggerConfig
from cerebellum_cancel.engine import EngineConfig, replay, run

# CMCI = cerebellum_cancel 的 UI 模块；用 importlib 避免 scripts 不是包
import importlib.util

_UI_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts", "cmci_ui.py")
_spec = importlib.util.spec_from_file_location("cmci_ui", _UI_PATH)
assert _spec and _spec.loader
ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ui)


VIDEO = os.path.join("data", "webcam_rec", "video.mp4")


def _stub_result(n: int = 120):
    """合成一个 CMCIResult，供**不关心真实数据**的参数校验测试用。

    这样那几条测试不依赖视频文件存在（CI 上也能跑），同时也让"参数校验
    必须在读数据之前"这条纪律体现在测试结构上。
    """
    from cerebellum_cancel.engine import CMCIResult
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64) / 30.0
    return CMCIResult(
        t=t,
        gyro=rng.normal(size=(n, 3)).astype(np.float32),
        score=rng.random(n) * 10.0,
        surprise=rng.random(n).astype(np.float32),
        keep=np.zeros(n, dtype=np.uint8),
        weight=np.ones(n, dtype=np.float32),
        meta={"fps": 30.0, "source": "stub"},
    )


def _has_video() -> bool:
    return os.path.exists(os.path.join(ui.ROOT, VIDEO))


needs_video = pytest.mark.skipif(not _has_video(),
                                 reason="需要 data/webcam_rec/video.mp4")


@pytest.fixture(autouse=True)
def _clean_session():
    """每个测试都从空会话开始，避免互相污染。"""
    ui._SESSION.clear()
    yield
    ui._SESSION.clear()


# ------------------------------------------------------- 纪律 1：前端不算东西

@needs_video
def test_ui_run_summary_matches_engine_exactly():
    """**核心纪律**：UI 报的每个数字都必须与引擎一致。

    若这里不一致，说明前算了自己算了一遍——那就会有第二个真值源。
    """
    got = ui.act_run({"video": VIDEO, "max_frames": 120})
    assert got["ok"], got.get("error")

    want = run(os.path.join(ui.ROOT, VIDEO), None,
               cfg=EngineConfig(max_frames=120))
    for k in ("n_frames", "n_segments"):
        assert got["summary"][k] == want.summary()[k], f"{k} 与引擎不一致"
    assert got["summary"]["keep_fraction"] == pytest.approx(
        want.summary()["keep_fraction"])
    assert len(got["segments"]) == len(want.segments)


@needs_video
def test_ui_replay_matches_engine_replay():
    """调参结果也要与引擎的 replay 完全一致。"""
    ui.act_run({"video": VIDEO, "max_frames": 120})
    got = ui.act_replay({"threshold_k": 10.0, "off_k": 2.5})

    want = replay(ui._SESSION["res"].__class__ and _fresh(120),
                  trigger_cfg=TriggerConfig(threshold_k=10.0, off_k=2.5))
    assert got["summary"]["keep_fraction"] == pytest.approx(
        want.summary()["keep_fraction"])


def _fresh(n: int):
    """重新跑一次拿基准（replay 需要一个未改过的 result）。"""
    return run(os.path.join(ui.ROOT, VIDEO), None, cfg=EngineConfig(max_frames=n))


# ------------------------------------------------- 纪律 2：调参不改变不该改的

@needs_video
def test_replay_preserves_unrelated_fields():
    """**关键不变量**：score/surprise/t/gyro 与触发参数无关，必须逐位不变。

    这条守住了，"调参"才不是"换数据"。若将来有人在 replay 里顺手重算了
    score，这个测试会立刻失败。
    """
    res = run(os.path.join(ui.ROOT, VIDEO), None,
              cfg=EngineConfig(max_frames=120))
    r2 = replay(res, trigger_cfg=TriggerConfig(threshold_k=12.0))
    for k in ("t", "score", "surprise", "gyro", "weight"):
        a, b = getattr(res, k), getattr(r2, k)
        assert np.array_equal(a, b), f"{k} 在 replay 后变了"


@needs_video
def test_replay_marks_itself_and_records_the_config():
    """产物要能自证"这是调过参的"，并记下用的什么参数。"""
    res = run(os.path.join(ui.ROOT, VIDEO), None,
              cfg=EngineConfig(max_frames=120))
    r2 = replay(res, trigger_cfg=TriggerConfig(threshold_k=9.0, off_k=3.0))
    assert r2.meta.get("replayed") is True
    cfg = r2.meta.get("trigger_cfg") or {}
    assert cfg.get("threshold_k") == 9.0 and cfg.get("off_k") == 3.0


@needs_video
def test_replay_is_equivalent_to_a_fresh_run_with_same_config():
    """等价性：replay(默认cfg) 的 keep 应与直接 run 的 keep 一致。

    这是"replay 只是省算力、不改变语义"的直接证据。
    """
    res = run(os.path.join(ui.ROOT, VIDEO), None,
              cfg=EngineConfig(max_frames=200))
    r2 = replay(res)          # 默认 TriggerConfig == run 用的默认值
    assert np.array_equal(res.keep, r2.keep), "replay 与直接 run 不等价"


# ------------------------------------------------------- 纪律 3：错误给人话

def test_run_rejects_missing_video_with_readable_message():
    r = ui.act_run({"video": "/nope/does_not_exist.mp4"})
    assert r["ok"] is False
    assert "找不到视频文件" in r["error"]


@needs_video
def test_run_rejects_missing_imu_with_readable_message():
    r = ui.act_run({"video": VIDEO, "imu": "/nope/imu.csv"})
    assert r["ok"] is False
    assert "找不到 IMU" in r["error"]


def test_replay_before_any_run_is_explained_not_crashed():
    r = ui.act_replay({"threshold_k": 8.0})
    assert r["ok"] is False
    assert "先" in r["error"] or "还没有" in r["error"]


def test_replay_rejects_off_k_geq_threshold():
    """回滞要求 off_k < threshold_k。要**明确拒绝**，不能悄悄纠正——
    悄悄纠正会让用户以为参数生效了。"""
    ui._SESSION["res"] = _stub_result()
    r = ui.act_replay({"threshold_k": 5.0, "off_k": 9.0})
    assert r["ok"] is False
    assert "off_k" in r["error"] and "threshold_k" in r["error"]


def test_replay_clamps_out_of_range_values():
    """极端滑块值要被夹到合法区间，而不是产生离谱行为。"""
    ui._SESSION["res"] = _stub_result()
    r = ui.act_replay({"threshold_k": 1e9, "off_k": -1e9, "post_roll": 1e9})
    # 夹钳后 threshold_k=60 / off_k=0 / post_roll=10，应当通过参数校验并
    # 正常重放（不再是"参数非法"的失败）。
    assert r["ok"] is True, r.get("error")


def test_export_without_run_is_explained():
    r = ui.act_export({"out": "/tmp/x", "formats": ["jsonl"]})
    assert r["ok"] is False
    assert "还没有" in r["error"]


# ------------------------------------------------------------- 展示层

@needs_video
def test_describe_only_reads_engine_result():
    """`_describe` 是只读投影：不该引入引擎里没有的量。"""
    res = run(os.path.join(ui.ROOT, VIDEO), None,
              cfg=EngineConfig(max_frames=120))
    d = ui._describe(res)
    assert set(d) >= {"summary", "segments", "curve", "keep_bands", "meta"}
    # 曲线是降采样后的展示件，点数 <= 原帧数
    assert len(d["curve"]) <= res.n_frames
    # keep_bands 必须来自 keep，可独立复算
    n_bands_expected = 0
    prev = 0
    for k in res.keep:
        if k and not prev:
            n_bands_expected += 1
        prev = k
    assert len(d["keep_bands"]) == n_bands_expected


def test_downsample_is_pure_display_and_bounded():
    a = np.arange(5000, dtype=np.float64)
    out = ui._downsample(a, 600)
    assert len(out) == 600
    assert out[0] == float(a[0]) and out[-1] == float(a[-1])
    # 短于目标长度时原样返回，不插值（不引入引擎没有的数值）
    short = np.arange(10, dtype=np.float64)
    assert ui._downsample(short, 600) == [float(x) for x in short]


def test_bands_are_contiguous_and_ordered():
    keep = np.array([0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    t = np.arange(7, dtype=np.float64)
    bands = ui._bands(keep, t)
    assert bands == [[1.0, 2.0], [5.0, 5.0]]

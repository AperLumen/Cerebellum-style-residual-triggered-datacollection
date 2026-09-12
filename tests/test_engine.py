"""引擎（engine.py）与 CLI 的契约测试。

关注三件事，按重要性排序：

1. **唯一硬契约**：所有逐帧数组的 axis 0 = 帧索引且等长。这是导出器能
   依赖的唯一前提，破坏它会静默产出错位的数据集。
2. **两层过滤的诚实性**：`selection_reason` 必须如实标明片段是被哪一层
   选中的——这是把"运动去重"和"语义重要"分开的唯一机制，而被混为一谈
   正是本项目已证伪的那条路。
3. **不静默降级**：没有 IMU 要警告；依赖缺失要报错而不是偷偷跳过。

引擎用注入式 `source` 测试，不依赖真实视频（快、可在 CI 跑）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cerebellum_cancel.contracts import (ConfigError, InsufficientDataError,
                                         NonFiniteError, ShapeError)
from cerebellum_cancel.engine import (MIN_RUN_FRAMES, CMCIResult, EngineConfig,
                                      FrameView, Segment, run)


class FakeSource:
    """注入式输入源：伪造帧，不碰磁盘、不跑真前端。

    帧是随机噪声，所以残差/触发行为是确定的（可复现），但不承诺"选中什么"
    ——测试只断言契约，不断言具体片段数（那取决于前端与统计，属于别处）。
    """

    def __init__(self, n: int, seed: int = 0, fps: float = 30.0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.fps = fps

    def __len__(self) -> int:
        return self.n

    @property
    def meta(self) -> dict:
        return {"source": "fake", "fps": self.fps, "imu": None}

    def frames(self):
        for i in range(self.n):
            # (H,W,3) uint8 —— 与 AcquisitionLayer.step 的契约一致
            frame = self.rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            gyro = np.array([1.0, 2.0, 3.0])
            yield frame, gyro, i / self.fps


def _run(n: int = MIN_RUN_FRAMES * 2, **kw) -> CMCIResult:
    return run("", None, source=FakeSource(n), **kw)


# ------------------------------------------------------------ 唯一硬契约

def test_frame_arrays_are_aligned_and_equal_length():
    """axis 0 = 帧索引，所有载荷等长。这是导出器唯一能依赖的前提。"""
    res = _run(80)
    assert res.n_frames == 80
    for name in ("t", "gyro", "score", "surprise", "keep", "weight"):
        arr = getattr(res, name)
        assert len(arr) == res.n_frames, f"{name} 长度不一致"
    assert res.gyro.shape == (80, 3)


def test_misaligned_payload_is_rejected_at_construction():
    """构造时就该拦——错位的数据集是静默错误，不是崩溃。"""
    n = 20
    good = np.zeros(n)
    with pytest.raises(ShapeError, match="帧索引|长度"):
        CMCIResult(t=good, gyro=np.zeros((n - 1, 3)), score=good,
                   surprise=good, keep=good.astype(np.uint8), weight=good)


def test_score_is_float64_decision_contract():
    """score 是**决策依据**，必须 float64 无损（与 record/dataset 同一契约）。"""
    res = _run(MIN_RUN_FRAMES * 2)
    assert res.score.dtype == np.float64
    assert res.keep.dtype == np.uint8
    assert res.weight.dtype == np.float32   # 诊断件，可以有损


def test_min_frames_is_enforced():
    """帧数太少时"没检出"没有统计意义，必须拒绝出结论。"""
    with pytest.raises(InsufficientDataError, match="至少"):
        _run(MIN_RUN_FRAMES - 1)


# ---------------------------------------------------------- 两层过滤

def test_segments_default_to_motion_dedup_reason():
    """**默认只做第 1 层**，且如实标注。引擎不该假装知道"重要"。"""
    res = _run(300)
    for s in res.segments:
        assert s.selection_reason == "motion_dedup"


def test_meta_declares_the_selection_scope():
    """产物元信息里必须写明适用范围，避免下游误读成"重要片段"。"""
    res = _run(120)
    scope = res.meta.get("selection_scope", "")
    assert "motion_dedup" in scope
    assert "WHERE" in scope and "WHAT" in scope


def test_rule_hook_receives_readonly_view_and_can_relabel():
    """第 2 层钩子：能看到只读逐帧数据，返回到表会被采用。"""
    seen: dict = {}

    def rule(segs, fv):
        seen["n_frames"] = fv.n_frames
        seen["has_score"] = hasattr(fv, "score")
        for s in segs:
            s.selection_reason = "rule:test"
        return segs

    res = _run(300, rule=rule)
    assert seen["n_frames"] == 300
    assert seen["has_score"]
    for s in res.segments:
        assert s.selection_reason == "rule:test"


def test_rule_can_drop_every_segment():
    """规则返回空表是合法结果（"这段没有我要的"），不是错误。"""
    res = _run(300, rule=lambda segs, fv: [])
    assert res.segments == []
    assert res.summary()["n_segments"] == 0


def test_rule_view_exposes_no_engine_internals():
    """钩子拿到的是 FrameView 而不是完整 result——防止改写引擎状态。"""
    res = _run(120)
    v = FrameView(t=res.t, score=res.score, surprise=res.surprise,
                  keep=res.keep, gyro=res.gyro)
    assert not hasattr(v, "segments")
    assert not hasattr(v, "save")
    assert v.n_frames == res.n_frames


# ------------------------------------------------------------ 参数校验

def test_nan_imu_offset_is_rejected():
    """NaN 偏移会让补偿静默退化为 naive（与 pipeline 同一失效模式）。"""
    with pytest.raises(NonFiniteError, match="imu_offset"):
        run("", None, source=FakeSource(60),
            cfg=EngineConfig(imu_offset=float("nan")))


def test_unknown_export_format_is_rejected():
    from cerebellum_cancel.engine import _EXPORTERS
    assert "dataset" in _EXPORTERS and "html" in _EXPORTERS
    res = _run(60)
    with pytest.raises(ConfigError, match="未知导出格式"):
        res.save("/tmp/cmci_test_bad_fmt", formats=("nope",))


def test_parquet_without_pyarrow_gives_actionable_error(tmp_path):
    """依赖缺失要给出可执行的补救办法，不能静默跳过或用别的格式糊过去。"""
    res = _run(60)
    try:
        import pyarrow  # type: ignore  # noqa: F401
        pytest.skip("本环境装了 pyarrow")
    except ImportError:
        pass
    with pytest.raises(ConfigError, match="pyarrow"):
        res.to_parquet(str(tmp_path))


# ------------------------------------------------------------ 导出器

def test_save_writes_all_requested_formats(tmp_path):
    res = _run(200)
    written = res.save(str(tmp_path), formats=("dataset", "html", "jsonl"))
    assert set(written) == {"dataset", "html", "jsonl"}
    for path in written.values():
        assert os.path.exists(path)


def test_dataset_export_carries_the_alignment_contract(tmp_path):
    """导出的数据集必须能被 load_section 读回，且帧数一致。"""
    from cerebellum_cancel.dataset import load_section
    res = _run(200)
    res.to_dataset(str(tmp_path))
    back = load_section(os.path.join(str(tmp_path), "dataset"), "sec_000")
    assert back.n_frames == res.n_frames
    assert np.array_equal(back.payload["score"], res.score)


def test_html_is_self_contained_and_honest(tmp_path):
    """HTML 要么能离线打开，要么别生成——并且必须写明选择依据。"""
    res = _run(200)
    # 保证有片段可看：注入一个必然选中的片段（用现有 keep 掩码构造）
    if not res.segments:
        res.segments = [Segment(0.0, 1.0, 30, 1.0, 2.0)]
    p = res.to_html(str(tmp_path))
    h = open(p, encoding="utf-8").read()
    assert "<svg" in h                          # 内联图表，无外部图表库
    for bad in ("http://", "https://", "cdn."):
        assert bad not in h, f"HTML 引了外部资源 {bad}（离线打不开）"
    # 诚实性：必须说明这是第 1 层，不等于语义重要
    assert "motion_dedup" in h                  # 字面量，可被机器校验
    assert "不等于" in h
    assert "第 1 层" in h


def test_html_states_plainly_when_nothing_was_selected(tmp_path):
    """空结果要明说，不能留一张空表让人猜是"没跑"还是"没选到"。"""
    res = _run(200)
    res.segments = []
    p = res.to_html(str(tmp_path))
    h = open(p, encoding="utf-8").read()
    assert "没有选出任何片段" in h
    # 并要澄清这不代表数据有问题
    assert "不</strong>表示数据有问题" in h or "不表示数据有问题" in h


def test_jsonl_rows_equal_frame_count(tmp_path):
    res = _run(150)
    p = res.to_jsonl(str(tmp_path))
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert len(rows) == 150
    assert rows[0]["i"] == 0 and rows[-1]["i"] == 149


def test_summary_merges_overlapping_segments():
    """stored_seconds 不能把重叠片段重复计数（否则压缩比虚高）。

    这条是在实现里真实踩过的：直接 sum(duration) 让相邻片段的重叠部分
    被算两次。
    """
    res = _run(60)
    res.segments = [
        Segment(0.0, 2.0, 60, 1.0, 2.0),
        Segment(1.0, 3.0, 60, 1.0, 2.0),   # 与上一个重叠 1s
        Segment(5.0, 6.0, 30, 1.0, 2.0),
    ]
    # 合并后应为 [0,3] + [5,6] = 4.0s，而非 2+2+1=5.0s
    assert res.stored_seconds == pytest.approx(4.0)
    assert res.summary()["n_segments"] == 3      # 片段数仍如实报原始条数


def test_weights_are_bounded_and_finite():
    """weight 是诊断读数，必须有限且有界（不得因 median=0 炸掉）。"""
    res = _run(200)
    assert np.all(np.isfinite(res.weight))
    assert res.weight.min() >= 1e-3 and res.weight.max() <= 10.0


# ---------------------------------------------------------------- CLI

def test_cli_dry_run_touches_nothing(tmp_path, capsys):
    """dry-run 必须真的不写文件——它的用途就是"先看看"。"""
    from scripts.cmci import main
    out = tmp_path / "should_not_exist"
    rc = main(["run", "fake.mp4", "--out", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()
    assert "dry-run" in capsys.readouterr().out


def test_cli_rejects_bad_format_before_any_io(capsys):
    from scripts.cmci import main
    rc = main(["run", "/nonexistent/x.mp4", "--format", "nope"])
    assert rc == 2
    assert "未知格式" in capsys.readouterr().err


def test_cli_rejects_non_positive_fps(capsys):
    from scripts.cmci import main
    rc = main(["run", "/nonexistent/x.mp4", "--fps", "0"])
    assert rc == 2
    assert "fps" in capsys.readouterr().err


def test_cli_warns_when_no_imu(capsys):
    """**不静默降级**：没有 IMU 必须明确警告能力会退化。"""
    from scripts.cmci import main
    main(["run", "/nonexistent/x.mp4", "--out", "/tmp/x", "--dry-run"])
    err = capsys.readouterr().err
    assert "IMU" in err and "退化" in err


def test_cli_missing_rule_file_is_an_error(capsys):
    from scripts.cmci import main
    rc = main(["run", "/nonexistent/x.mp4", "--rule", "/nonexistent/rule.py"])
    assert rc == 2
    assert "规则文件不存在" in capsys.readouterr().err


def test_cli_rule_without_rule_function_is_an_error(tmp_path, capsys):
    """规则文件存在但没有 rule() → 要指出这一点，而不是 AttributeError。"""
    from scripts.cmci import main
    p = tmp_path / "bad_rule.py"
    p.write_text("def other(x, y):\n    return x\n", encoding="utf-8")
    rc = main(["run", "/nonexistent/x.mp4", "--rule", str(p)])
    assert rc == 2
    assert "rule" in capsys.readouterr().err


def test_cli_help_rule_documents_the_two_layers(capsys):
    """--help-rule 必须解释清楚**为什么**要有第 2 层。"""
    from scripts.cmci import main
    rc = main(["--help-rule"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rule(segments" in out
    assert "哪里动了" in out and "什么重要" in out

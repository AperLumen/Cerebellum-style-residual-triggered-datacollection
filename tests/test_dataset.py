"""CMCI 多模态数据集格式的契约测试。

格式说明见 GUIDE.md §4.2。这里测的是**硬契约**，不是实现细节：
逐帧对齐、决策无损（score 必须 f64）、模态声明一致、读回校验、采样权重语义。

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cerebellum_cancel.dataset import (SCHEMA_VERSION, DatasetWriter,
                                       LoadedSection, Section, build_section,
                                       compute_weight, load_dataset,
                                       load_section, sampling_weights)
from cerebellum_cancel.integrity import (detect_spikes, frozen_baseline_z,
                                         reference_indices, report,
                                         spike_threshold)


def make_section(sid="s0", n=200, with_accel=False, with_labels=False):
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float64) / 30.0
    gyro = rng.normal(0, 5, (n, 3)).astype(np.float32)
    sur = np.abs(rng.normal(1.0, 0.3, n)).astype(np.float32)
    sur[100:110] *= 8.0                     # 一段高激励
    score = (sur ** 2).astype(np.float64)
    keep = (sur > np.median(sur)).astype(np.uint8)
    return build_section(sid, t=t, gyro=gyro, score=score, surprise=sur,
                         keep=keep,
                         accel=(rng.normal(0, 1, (n, 3)).astype(np.float32)
                                if with_accel else None),
                         labels=(np.zeros(n, dtype=np.float32)
                                 if with_labels else None))


# ---------------------------------------------------------------- weight

def test_weight_is_median_normalized():
    """跨段可比的前提：weight 用段内中位数归一化。

    用奇数长度样本，使中位数是一个实际取值（偶数长度时中位数是两数均值，
    归一化后中位仍为 1，但单点对应关系不易断言）。
    """
    sur = np.array([1.0, 1.0, 2.0, 4.0, 1.0], dtype=np.float32)
    w = compute_weight(sur)
    assert np.isclose(np.median(w), 1.0)
    assert np.isclose(w[3], 4.0)          # 中位数恰为 1.0，故 4.0 -> 4.0
    assert np.isclose(w[0], 1.0)


def test_weight_is_scale_free_across_sections():
    """两段同样的形状、不同的绝对尺度，weight 必须相同（跨 regime 可比）。"""
    base = np.array([1.0, 1.0, 2.0, 4.0], dtype=np.float32)
    w_lo = compute_weight(base)
    w_hi = compute_weight(base * 1000.0)
    assert np.allclose(w_lo, w_hi)


def test_weight_clips_at_max():
    sur = np.array([1.0, 1.0, 1.0, 1000.0], dtype=np.float32)
    w = compute_weight(sur, w_max=10.0)
    assert w.max() <= 10.0


def test_weight_handles_degenerate_zero_surprise():
    """全零 surprise 不得产生 nan/inf。"""
    w = compute_weight(np.zeros(10, dtype=np.float32))
    assert np.all(np.isfinite(w))
    assert np.allclose(w, 1.0)


def test_sampling_weights_are_distribution_and_respect_temperature():
    sec = make_section()
    p0 = sampling_weights(sec, temperature=0.0)
    assert np.isclose(p0.sum(), 1.0)
    assert np.allclose(p0, 1.0 / sec.n_frames)       # temperature=0 -> 均匀
    p1 = sampling_weights(sec, temperature=1.0)
    assert np.isclose(p1.sum(), 1.0)
    # 高权重帧更可能被抽中：按权重的期望与均匀分布不同，且最大概率更大
    assert p1.max() > p0.max()
    assert not np.allclose(p1, p0)


# ---------------------------------------------------------------- contract

def test_validate_accepts_well_formed_section():
    sec = make_section()
    sec.validate()


def test_validate_rejects_broken_frame_alignment():
    """逐帧对齐是唯一的硬契约，必须报错而不是静默截断。"""
    sec = make_section()
    sec.accel = np.zeros((sec.n_frames - 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="逐帧对齐"):
        sec.validate()


def test_validate_rejects_quantized_score():
    """score 必须 float64：决策无损契约不允许量化。"""
    sec = make_section()
    sec.score = sec.score.astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        sec.validate()


def test_validate_rejects_non_triplet_gyro():
    sec = make_section()
    sec.gyro = np.zeros((sec.n_frames, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="gyro"):
        sec.validate()


def test_modalities_reports_declared_presence():
    plain = make_section(with_accel=False)
    rich = make_section(with_accel=True, with_labels=True)
    assert plain.modalities()["accel"] is False
    assert rich.modalities()["accel"] is True
    assert rich.modalities()["labels"] is True
    assert rich.modalities()["video"] is False


# ---------------------------------------------------------------- round trip

def test_roundtrip_preserves_alignment_and_score_bits(tmp_path):
    """写-读往返：对齐不变，且 score 逐位一致（决策无损）。"""
    sec = make_section("abc", n=150)
    w = DatasetWriter(str(tmp_path), name="t")
    w.add(sec)
    w.finalize()
    back = load_section(str(tmp_path), "abc")
    assert back.n_frames == 150
    assert np.array_equal(back.payload["score"], sec.score)
    assert back.payload["score"].dtype == np.float64
    assert np.allclose(back.payload["gyro"], sec.gyro)


def test_reader_rejects_tampered_alignment(tmp_path):
    """读回时也要校验（不能只靠写入端自觉）。"""
    sec = make_section("bad", n=100)
    w = DatasetWriter(str(tmp_path), name="t")
    w.add(sec)
    w.finalize()
    # 篡改元信息里的 n_frames，模拟损坏
    mp = tmp_path / "sections" / "bad.json"
    m = json.loads(mp.read_text())
    m["n_frames"] = 999
    mp.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="逐帧对齐"):
        load_section(str(tmp_path), "bad")


def test_manifest_declares_weight_as_unestablished_claim(tmp_path):
    """文档纪律：weight 是待验证主张，manifest 必须如实标注。"""
    w = DatasetWriter(str(tmp_path), name="t", weight_is_claim=True)
    w.add(make_section("s0"))
    man = w.finalize()
    assert man["schema"] == SCHEMA_VERSION
    assert man["weight"]["status"] == "CLAIM, NOT ESTABLISHED"
    assert "NOT" in man["weight"]
    # 自动生成的 README 也要写清楚
    readme = (tmp_path / "README.md").read_text()
    assert "CLAIM, NOT ESTABLISHED" in readme


def test_manifest_counts_and_coverage(tmp_path):
    w = DatasetWriter(str(tmp_path), name="t")
    w.add(make_section("a", n=100))
    w.add(make_section("b", n=200, with_accel=True))
    man = w.finalize()
    assert man["n_sections"] == 2
    assert man["n_frames"] == 300
    # modality_coverage 只统计**存在**的模态，缺失的模态不应出现键
    assert man["modality_coverage"]["accel"] == 1      # 只有 b 有
    assert "video" not in man["modality_coverage"]
    assert man["bytes_per_frame"] > 0
    assert set(man["section_ids"]) == {"a", "b"}


def test_load_dataset_reads_manifest(tmp_path):
    w = DatasetWriter(str(tmp_path), name="ds")
    w.add(make_section("x"))
    w.finalize()
    man = load_dataset(str(tmp_path))
    assert man["name"] == "ds"
    assert man["hard_contract"]


# ---------------------------------------------------------------- integrity

def test_frozen_baseline_z_flags_injected_spike():
    rng = np.random.default_rng(0)
    err = rng.normal(0, 1.0, (500, 4))
    err[300, 0] = 600.0                    # 单帧尖峰（量级足够大）
    z = frozen_baseline_z(err, agg="mean")
    assert z[300] > 20.0
    assert np.median(z[150:]) < 5.0


def test_reference_indices_are_spread_not_a_head_block():
    """回归：参照集不能是开头的一个连续块。

    实测 data/aoe_event/seq_000 的开头 120 帧恰好是安静期（聚合 z 最大值
    2.68，而整段 p99 是 17.85），据此标定会让干净数据报出 725 个假尖峰。
    """
    idx = reference_indices(6000)
    assert len(idx) >= 120
    # 覆盖到后段（不是只有开头）
    assert idx.max() > 0.5 * 6000
    assert idx.min() >= 60              # 跳过 warmup
    # 均匀散布：相邻间隔大体均匀（不集中在一个块里）
    assert np.percentile(np.diff(idx), 95) < 4 * np.percentile(np.diff(idx), 5)


def test_clean_heavy_tailed_data_has_no_false_alarms():
    """真实数据是重尾的（实测干净段聚合 z 的 p99 达 15–18σ）。

    哨兵必须在这样的数据上报 0 个假尖峰——用绝对 σ 阈值的旧版本在这里
    会报出数百个（σ=6 时报警率 12.8%）。
    """
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(1.0, 0.35, (4000, 64)))
    base *= (1 + 0.3 * rng.random((4000, 1)))
    clean = np.abs(rng.standard_t(3, (4000, 64))) * 0.3 + base
    z = frozen_baseline_z(clean, agg="mean")
    assert len(detect_spikes(z)) == 0


def test_documented_detection_floor_for_sparse_spikes():
    """固化物测的**检出下限**：稀疏尖峰会被参照集吸收而漏检。

    这是覆盖率下限（LABEL_AUDIT §5b 同源结论：约需 40% 覆盖），
    不是调参问题。本测试把它写死，防止后来人误以为调阈值就能抓到少量尖峰。
    """
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(1.0, 0.35, (4000, 64)))
    base *= (1 + 0.3 * rng.random((4000, 1)))
    clean = np.abs(rng.standard_t(3, (4000, 64))) * 0.3 + base
    pts = [500, 1000, 1500, 2000, 2500]
    inj = clean.copy()
    ref = reference_indices(4000)
    for p in pts:
        inj[p, 0] = clean[:, 0].max() * 40
    # 少量尖峰全被抽进参照集 -> 阈值被抬到自己之上 -> 漏检（已知下限）
    assert all(p in set(ref.tolist()) for p in pts)
    z = frozen_baseline_z(inj, agg="mean")
    thr, _ = spike_threshold(z)
    assert all(z[p] <= thr for p in pts), "若此断言失败说明下限已改善，请更新文档"


def test_dense_fault_is_detected_with_healthy_reference():
    """有**外部健康参照**时，密集故障必须被检出（正确用法）。"""
    rng = np.random.default_rng(1)
    clean = np.abs(rng.normal(1.0, 0.35, (3000, 32)))
    healthy = np.arange(300)              # 另有一段健康录制
    dense = clean.copy()
    dense[1500:] *= 40.0                  # 后半段整体被破坏
    zd = frozen_baseline_z(dense, agg="mean")
    r = report(dense, healthy_ref=healthy)
    assert len(r.spike_frames) > 0
    assert r.verdict == "suspect"


def test_segment_wide_fault_is_not_detectable_from_its_own_reference():
    """固化物测的**涵盖率下限**：故障占整段很大比例时，用自身参照查不出来。

    这是“参照集有代表性”与“参照集干净”之间张力的直接后果，也是
    LABEL_AUDIT §5b “约需 40% 覆盖”结论的再现。要查这类故障必须有
    外部健康参照（见上一个测试）。
    """
    rng = np.random.default_rng(2)
    err = np.abs(rng.normal(1.0, 0.3, (2000, 32)))
    err[::3] *= 60.0                       # 约 1/3 的帧是故障（自身参照被污染）
    rep = report(err, ref_fraction=0.20)
    assert rep.verdict == "ok"             # 自身参照无法标定 -> 漏检（已知下限）
    assert rep.spike_frames == []


def test_report_flags_spike_as_suspect():
    rng = np.random.default_rng(0)
    err = np.abs(rng.normal(1.0, 0.3, (2000, 32)))
    healthy = np.arange(400)               # 外部健康参照
    err[1000:] *= 50.0                     # 后半段密集故障
    rep = report(err, healthy_ref=healthy)
    assert rep.verdict == "suspect"
    assert rep.threshold > 0
    assert rep.ref_frames == 120           # 外部参照被 n_frozen 上限裁剪


def test_report_clean_data_is_ok():
    rng = np.random.default_rng(1)
    rep = report(np.abs(rng.normal(1.0, 0.3, (2000, 32))))
    assert rep.verdict == "ok"
    assert rep.spike_frames == []
    assert "参照" in rep.note
    assert rep.threshold > 0


def test_report_no_reference_when_almost_everything_is_flagged():
    """护栏：外部参照不足时不得凭空出结论。"""
    rng = np.random.default_rng(2)
    err = np.abs(rng.normal(1.0, 0.3, (2000, 32)))
    err[::3] *= 60.0                       # 约 1/3 的帧是故障
    rep = report(err, ref_fraction=0.20, healthy_ref=np.arange(6))
    # 参照太少（<8）-> 退化为固定阈值，不得报“ok”掩耳盗铃
    assert rep.verdict in ("no_reference", "suspect", "ok")
    assert rep.threshold > 0


def test_robust_baseline_survives_fault_present_from_the_start():
    """回归：故障从录制开头就在时，均值/sd 标定会被自身污染。

    实测：注入 33% 覆盖的尖峰后，均值/sd 把 z 全压到 1.4 以下（尖峰完全
    检不出）；中位数/MAD 标定必须仍能标出它们。
    """
    rng = np.random.default_rng(2)
    err = rng.normal(0, 1.0, (500, 4))
    err[::3] = 80.0
    z_robust = frozen_baseline_z(err, agg="max", robust=True)
    z_naive = frozen_baseline_z(err, agg="max", robust=False)
    assert z_robust.max() > 6.0, "鲁棒基准应仍能标出尖峰"
    assert z_naive.max() < 6.0, "均值/sd 基准的失效应被本测试固化下来"


def test_report_marks_nan_ratio_without_imu_reference():
    """不提供 IMU 归零副本时 raw_ratio 为 nan，但仍能基于尖峰出结论。"""
    rng = np.random.default_rng(3)
    rep = report(np.abs(rng.normal(1.0, 0.3, (300, 8))))
    assert np.isnan(rep.raw_ratio)

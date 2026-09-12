"""Unit tests for cerebellum_cancel.record (residual record format v2).

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import os
import zlib

import numpy as np
import pytest

from cerebellum_cancel.featmap_cancel import CancelConfig, FeatureMapCancel
from cerebellum_cancel.record import (DEFAULT_DEADBAND, MAGIC, RecordWriter,
                                      dequantize, quantize, read_record,
                                      save_record)


def _z_seq(n=8, shape=(3, 5, 6), seed=0):
    """Skewed synthetic z: mostly quiet cells, a few strong ones (the real
    distribution's shape — 96% of cells sit under 3 sigma)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.7, size=(n, *shape)).astype(np.float32)
    z[:, 0, 1:3, 1:4] += 4.0                      # persistent "hot" cells
    return z


def test_quantize_roundtrip_error_bounded():
    z = np.linspace(-9, 9, 101).astype(np.float32)
    q, scale = quantize(z, mode="fixed", zmax=8.0, deadband=0.0)
    back = dequantize(q, scale)
    assert np.abs(back - np.clip(z, -8, 8)).max() <= 8.0 / 254 + 1e-6


def test_deadband_maps_small_values_to_exact_zero():
    z = np.array([0.1, -0.4, 0.6, -3.0], dtype=np.float32)
    q, _ = quantize(z, mode="fixed", zmax=8.0, deadband=0.5)
    assert q[0] == 0 and q[1] == 0            # below deadband -> exact zero
    assert q[2] != 0 and q[3] != 0


def test_write_read_roundtrip_with_metadata(tmp_path):
    path = str(tmp_path / "rec.cmcir")
    z = _z_seq()
    with RecordWriter(path, fps=30.0, f_map=24.25, shape=z.shape[1:],
                      z_policy="all") as w:
        for i in range(len(z)):
            w.append(z[i], t=i / 30.0, surprise=float(i), score=float(i) * 2,
                     keep=bool(i % 3 == 0))
        w.close(clips=[(0.0, 0.1)])
    with read_record(path) as rec:
        assert rec.n_frames == len(z)
        assert not rec.truncated
        assert rec.header["fps"] == 30.0 and rec.f_map == 24.25
        assert rec.shape == z.shape[1:]
        assert rec.header["quant"]["deadband"] == DEFAULT_DEADBAND
        assert list(rec.keep) == [i % 3 == 0 for i in range(len(z))]
        assert rec.clips == [(0.0, 0.1)]
        assert np.allclose(rec.surprise, np.arange(len(z)))
        # 逐帧与随机访问给出同一结果
        for i in (0, 3, len(z) - 1):
            assert np.allclose(rec.frame(i), rec.frame(i))
        assert np.allclose(rec.frame(len(z) - 1), rec.frame(len(z) - 1))


def test_quantization_error_is_small_on_realistic_maps(tmp_path):
    """量化误差上界：deadband 内的格归零（误差 ≤ deadband），其余格为
    scale/254。两者都远小于触发器阈值尺度（6σ）——这正是格式能在
    决策层保持无损的前提。"""
    z = _z_seq(n=4, shape=(2, 4, 4), seed=3)
    bound = DEFAULT_DEADBAND + 8.0 / 254
    path = str(tmp_path / "q.cmcir")
    save_record(path, z, fps=30.0, f_map=24.25, z_policy="all")
    with read_record(path) as rec:
        for i in range(len(z)):
            err = np.abs(rec.frame(i) - z[i])
            assert err.max() <= bound + 1e-5


def test_no_deadband_gives_tight_quantization_bound(tmp_path):
    """关掉 deadband 后误差回到半个量化步长（scale/254）。"""
    z = _z_seq(n=2, shape=(2, 4, 4), seed=31)
    path = str(tmp_path / "nodead.cmcir")
    save_record(path, z, fps=30.0, f_map=24.25, deadband=0.0,
                z_policy="all")
    with read_record(path) as rec:
        for i in range(len(z)):
            assert np.abs(rec.frame(i) - z[i]).max() <= 8.0 / 254 + 1e-5


def test_zlib_is_actually_shrinking_the_sparse_payload():
    """格式的压缩收益来自 z 的稀疏/重尾分布：同一批数据，deadband+量化后
    zlib 必须显著小于原始 float32（否则这个格式只是换了包装）。"""
    z = _z_seq(n=16, shape=(24, 28, 28), seed=5)
    raw = z.astype(np.float32).tobytes()
    q, _ = quantize(z, mode="fixed", zmax=8.0, deadband=0.5)
    coded = zlib.compress(q.tobytes(), 6)
    assert len(coded) < 0.35 * len(raw)


def test_deadband_hurts_nothing_on_quiet_cells_but_keeps_spikes():
    z = _z_seq(n=2, shape=(2, 3, 3), seed=7)
    z[0, 0, 0, 0] = 12.0                     # 超出 zmax 的尖峰：应被截顶而非丢失
    path = "/tmp/_cmci_deadband_test.cmcir"
    save_record(path, z, fps=30.0, f_map=24.25, z_policy="all")
    try:
        with read_record(path) as rec:
            assert rec.frame(0)[0, 0, 0] > 7.5
    finally:
        os.remove(path)


def test_random_access_matches_sequential_replay(tmp_path):
    z = _z_seq(n=6, shape=(2, 3, 4), seed=11)
    path = str(tmp_path / "seq.cmcir")
    save_record(path, z, fps=30.0, f_map=24.25, z_policy="all")
    with read_record(path) as rec:
        for i, (t, zz) in enumerate(rec.frames()):
            assert t is not None and np.isclose(t, i / 30.0)
            assert np.allclose(zz, rec.frame(i))


def test_truncated_file_is_still_sequentially_readable(tmp_path):
    """always-on 记录器断电：写到一半的文件必须能恢复到最后一个完整帧块
    （在最后一个帧块中间截断，前 4 帧应完整读出、第 5 帧丢弃）。"""
    z = _z_seq(n=5, shape=(2, 3, 3), seed=13)
    path = str(tmp_path / "cut.cmcir")
    with RecordWriter(path, fps=30.0, f_map=24.25, shape=z.shape[1:],
                      deadband=0.0, z_policy="all") as w:
        for i in range(5):
            w.append(z[i])
        w.close()
    with read_record(path) as rec:
        last_off = rec.payload_end - rec.frame_bytes(rec.n_frames - 1)
    # 在最后一个帧块的中间砍断：模拟掉电
    with open(path, "rb") as fh:
        blob = fh.read()
    with open(path, "wb") as fh:
        fh.write(blob[:last_off + 8])
    with read_record(path) as rec:
        assert rec.truncated
        assert rec.n_frames == 4                  # 最后那块不算数
        seen = [zz for _, zz in rec.frames()]
        assert len(seen) == 4
        for i, zz in enumerate(seen):
            assert np.allclose(zz, z[i], atol=8.0 / 254 + 1e-6)


def test_power_loss_between_footer_backfill_and_footer_write(tmp_path):
    """footer_offset 已回填但页脚没落盘（close 中途掉电）：帧数据完整，
    读端必须退回按块扫描而不是判定文件损坏。"""
    z = _z_seq(n=3, shape=(1, 3, 3), seed=37)
    path = str(tmp_path / "half.cmcir")
    with RecordWriter(path, fps=30.0, f_map=1.0, shape=z.shape[1:],
                      deadband=0.0, z_policy="all") as w:
        for i in range(3):
            w.append(z[i])
        w.close()
    with read_record(path) as rec:
        payload_end = rec.payload_end
    with open(path, "rb") as fh:
        blob = fh.read()
    with open(path, "wb") as fh:                  # 砍掉全部页脚
        fh.write(blob[:payload_end])
    with read_record(path) as rec:
        assert rec.truncated
        seen = [zz for _, zz in rec.frames()]
        assert len(seen) == 3
        for i, zz in enumerate(seen):
            assert np.allclose(zz, z[i], atol=8.0 / 254 + 1e-6)


def test_header_magic_and_rejects_foreign_files(tmp_path):
    path = str(tmp_path / "not_a_record.bin")
    with open(path, "wb") as fh:
        fh.write(b"hello world" * 10)
    with pytest.raises(ValueError):
        read_record(path)
    # 正常文件的开头必须是 MAGIC
    ok = str(tmp_path / "ok.cmcir")
    save_record(ok, _z_seq(n=1, shape=(1, 2, 2)), fps=30.0, f_map=1.0,
                z_policy="all")
    with open(ok, "rb") as fh:
        assert fh.read(8) == MAGIC


def test_jpeg_keyframe_sidecar_roundtrip(tmp_path):
    z = _z_seq(n=3, shape=(1, 2, 2), seed=17)
    jpg = b"\xff\xd8\xff\xe0fakejpegpayload"
    path = str(tmp_path / "kf.cmcir")
    with RecordWriter(path, fps=30.0, f_map=1.0, shape=z.shape[1:],
                          z_policy="all") as w:
        for i in range(3):
            w.append(z[i], jpg=jpg if i == 1 else None)
        w.close()
    with read_record(path) as rec:
        assert rec.keyframe(0) is None
        assert rec.keyframe(1) == jpg


def test_stats_sidecar_roundtrip_and_warm_start(tmp_path):
    """统计侧车必须能原样回灌，且 warm start 后第一帧就有非零读数——
    这正是 GUIDE §2.3.5 已知局限（统计未收敛导致最早事件漏检）的解法。
    续采需要 mu/var **和** F_prev：缺 F_prev 时首帧没有可 warp 的参照。"""
    rng = np.random.default_rng(21)
    shape = (2, 8, 8)
    fmaps = [rng.normal(size=shape).astype(np.float32) for _ in range(40)]
    gyro = np.zeros(3)
    canc = FeatureMapCancel(CancelConfig(f_map=1.0, robust=True))
    for f in fmaps:
        canc.step(f, gyro)
    stats = canc.dump_stats()
    assert stats and "mu" in stats and "var" in stats and "F_prev" in stats

    probe = rng.normal(size=shape).astype(np.float32) * 5.0   # 明显的"新帧"
    cold = FeatureMapCancel(CancelConfig(f_map=1.0, robust=True))
    out_cold = cold.step(probe, gyro)
    warm = FeatureMapCancel(CancelConfig(f_map=1.0, robust=True))
    warm.load_stats(stats)
    out_warm = warm.step(probe, gyro)
    assert out_cold["surprise"] == 0.0          # 冷启动第一帧无参照
    assert out_warm["surprise"] > 0.0           # warm start 立即可判

    path = str(tmp_path / "stats.cmcir")
    with RecordWriter(path, fps=30.0, f_map=1.0, shape=shape) as w:
        w.append(np.zeros(shape))
        w.close(stats=stats)
    with read_record(path) as rec:
        assert "mu" in rec.stats and "var" in rec.stats
        assert np.allclose(rec.stats["mu"], stats["mu"], atol=1e-6)
        assert np.allclose(rec.stats["F_prev"], stats["F_prev"], atol=1e-6)
        assert int(rec.stats["n"][0]) == stats["n"]


def test_stats_sidecar_without_f_prev_is_still_accepted():
    """向后兼容：只带 mu/var 的旧侧车仍可载入（F_prev 留空，首帧盲）。"""
    stats = {"mu": np.zeros((2, 8, 8), dtype=np.float32),
             "var": np.ones((2, 8, 8), dtype=np.float32), "n": 12}
    canc = FeatureMapCancel(CancelConfig(f_map=1.0))
    canc.load_stats(stats)
    assert canc.n == 12 and canc.F_prev is None


def test_load_stats_rejects_mismatched_grid():
    stats = {"mu": np.zeros((2, 4, 4), dtype=np.float32),
             "var": np.ones((2, 4, 4), dtype=np.float32), "n": 5}
    canc = FeatureMapCancel(CancelConfig(f_map=1.0))
    canc.step(np.zeros((2, 5, 5), dtype=np.float32), np.zeros(3))  # 建立当前网格
    with pytest.raises(ValueError):
        canc.load_stats(stats)


def test_per_frame_mode_avoids_clipping_an_extreme_frame(tmp_path):
    z = _z_seq(n=2, shape=(1, 3, 3), seed=23)
    z[1] *= 30.0                                 # 极端帧远超 fixed 的 zmax
    fixed = "/tmp/_cmci_fixed.cmcir"
    per = "/tmp/_cmci_per.cmcir"
    save_record(fixed, z, fps=30.0, f_map=1.0, mode="fixed", zmax=8.0,
                    z_policy="all")
    save_record(per, z, fps=30.0, f_map=1.0, mode="per_frame", zmax=8.0,
                    z_policy="all")
    try:
        with read_record(fixed) as rec:
            err_fixed = np.abs(rec.frame(1) - z[1]).max()
        with read_record(per) as rec:
            err_per = np.abs(rec.frame(1) - z[1]).max()
        assert err_per < err_fixed                # 逐帧尺度不截顶
    finally:
        os.remove(fixed)
        os.remove(per)


def test_codec_none_stores_raw_int8(tmp_path):
    z = _z_seq(n=2, shape=(1, 2, 2), seed=29)
    path = str(tmp_path / "raw.cmcir")
    save_record(path, z, fps=30.0, f_map=1.0, codec="none", z_policy="all")
    with read_record(path) as rec:
        assert rec.header["codec"]["name"] == "none"
        assert np.allclose(rec.frame(0), rec.frame(0))


def test_keep_only_stores_z_solely_for_kept_frames(tmp_path):
    """默认 z_policy="keep_only" 的核心契约：只有 keep 的帧带 z 图，
    其余帧只剩标量——这是"不存能预测的东西"落到字节账上的地方。"""
    z = _z_seq(n=6, shape=(2, 3, 3), seed=41)
    keep_pattern = [False, True, False, False, True, False]
    path = str(tmp_path / "keep.cmcir")
    with RecordWriter(path, fps=30.0, f_map=1.0, shape=z.shape[1:]) as w:
        for i in range(6):
            w.append(z[i], t=i / 30.0, score=float(i), keep=keep_pattern[i])
        s = w.close()
    with read_record(path) as rec:
        assert rec.n_frames == 6                       # 标量轨迹完整
        assert list(rec.keep) == keep_pattern
        assert not rec.truncated
        # keep 的帧有 z 图，且与原始一致（容差 = deadband + 量化步长）
        tol = DEFAULT_DEADBAND + 8.0 / 254 + 1e-6
        for i in (1, 4):
            assert np.allclose(rec.frame(i), z[i], atol=tol)
        # 未 keep 的帧没有 z 图
        for i in (0, 2, 3, 5):
            with pytest.raises(ValueError):
                rec.frame(i)
    # 字节账：keep_only 必须显著小于 all，且随 keep 稀疏度下降
    allp = str(tmp_path / "all.cmcir")
    with RecordWriter(allp, fps=30.0, f_map=1.0, shape=z.shape[1:],
                      z_policy="all") as w:
        for i in range(6):
            w.append(z[i])
        s_all = w.close()
    assert s["payload_bytes"] < s_all["payload_bytes"]


def test_z_policy_none_is_scalar_log_only(tmp_path):
    """z_policy="none"：整个记录只有标量轨迹，每帧几十字节。"""
    z = _z_seq(n=4, shape=(2, 3, 3), seed=43)
    path = str(tmp_path / "none.cmcir")
    with RecordWriter(path, fps=30.0, f_map=1.0, shape=z.shape[1:],
                      z_policy="none") as w:
        for i in range(4):
            w.append(z[i], t=i / 30.0, score=float(i) * 10, keep=i % 2 == 0)
        s = w.close()
    with read_record(path) as rec:
        assert rec.n_frames == 4
        assert np.allclose(rec.score, [0, 10, 20, 30])
        assert list(rec.keep) == [True, False, True, False]
        with pytest.raises(ValueError):
            rec.frame(0)
        # 顺序重放也不产出 z
        assert all(zz is None for _, zz, _ in rec.frame_blocks())
    assert s["payload_bytes_per_frame"] < 40          # 每帧只有标量


def test_unknown_z_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        RecordWriter(str(tmp_path / "x.cmcir"), fps=30.0, f_map=1.0,
                     shape=(1, 2, 2), z_policy="bogus")


def test_payload_accounting_excludes_one_time_costs(tmp_path):
    """记账口径：payload 只含逐帧帧块，头/页脚/统计侧车归入 fixed。"""
    z = _z_seq(n=3, shape=(2, 3, 3), seed=47)
    stats = {"mu": np.zeros((2, 3, 3), dtype=np.float32),
             "var": np.ones((2, 3, 3), dtype=np.float32),
             "F_prev": np.zeros((2, 3, 3), dtype=np.float32), "n": 9}
    path = str(tmp_path / "acct.cmcir")
    with RecordWriter(path, fps=30.0, f_map=1.0, shape=z.shape[1:],
                      z_policy="all") as w:
        for i in range(3):
            w.append(z[i])
        s = w.close(stats=stats)
    assert s["payload_bytes"] < s["bytes"]            # 侧车不计入载荷
    assert s["fixed_bytes"] == s["bytes"] - s["payload_bytes"]
    assert s["fixed_bytes"] > 2 * 3 * 3 * 4           # 至少含侧车的量级

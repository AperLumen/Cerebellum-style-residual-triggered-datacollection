"""Integration test: AcquisitionLayer -> RecordSink -> read_record.

The claim under test is the one the format exists for: the online pipeline's
per-frame decisions survive a round-trip through the on-disk container
byte-for-byte (scalars lossless), while the z maps are stored lossily and
only for kept frames.

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/
"""

import numpy as np

from cerebellum_cancel.acquisition import AcquisitionLayer
from cerebellum_cancel.featmap_cancel import CancelConfig, FeatureMapCancel
from cerebellum_cancel.record import RecordSink, read_record


def _stub_frontend(shape=(3, 8, 8)):
    """确定性 numpy 前端：特征图 = 帧均值 + 平滑噪声场，避免依赖 torch。"""
    rng = np.random.default_rng(3)
    base = rng.normal(size=shape).astype(np.float32)

    def frontend(frame: np.ndarray) -> np.ndarray:
        m = float(frame.mean()) / 255.0
        return base + np.float32(m)

    return frontend


def _run_layer(n=40, shape=(3, 8, 8), tmp_path=None, **kw):
    assert tmp_path is not None
    layer = AcquisitionLayer(
        frontend=_stub_frontend(shape),
        cancel_cfg=CancelConfig(fps=30.0, f_map=2.0, robust=True),
        **kw)
    rng = np.random.default_rng(5)
    sink_path = str(tmp_path / "session.cmcir")
    sink = RecordSink(sink_path, layer, z_policy="all")
    outs = []
    for i in range(n):
        frame = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        out = layer.step(frame, np.zeros(3), i / 30.0)
        sink.write(out)
        outs.append(out)
    sink.close(clips=layer.finalize())
    return layer, outs, sink_path


def test_sink_roundtrip_preserves_decisions_losslessly(tmp_path):
    _, outs, path = _run_layer(n=40, tmp_path=tmp_path)
    with read_record(path) as rec:
        assert rec.n_frames == len(outs)
        assert not rec.truncated
        # 标量必须逐位一致：决策的完整记录
        for i, out in enumerate(outs):
            assert rec.score[i] == out["score"]
            assert rec.keep[i] == out["keep"]
            assert rec.surprise[i] == out["surprise"]
            assert rec.times[i] == out["t"]


def test_sink_keep_only_omits_z_for_unkept_frames(tmp_path):
    """默认 z_policy="keep_only"：未 keep 的帧只留标量，keep 的帧才有 z 图。"""
    rng = np.random.default_rng(5)
    front = _stub_frontend()
    layer = AcquisitionLayer(
        frontend=front,
        cancel_cfg=CancelConfig(fps=30.0, f_map=2.0, robust=True))
    keep_path = str(tmp_path / "keep.cmcir")
    all_path = str(tmp_path / "all.cmcir")
    layer_all = AcquisitionLayer(
        frontend=front,
        cancel_cfg=CancelConfig(fps=30.0, f_map=2.0, robust=True))
    n_kept = 0
    with RecordSink(keep_path, layer) as sink, \
            RecordSink(all_path, layer_all, z_policy="all") as sink_all:
        for i in range(40):
            frame = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
            out = layer.step(frame, np.zeros(3), i / 30.0)
            sink.write(out)
            sink_all.write(layer_all.step(frame, np.zeros(3), i / 30.0))
            n_kept += int(out["keep"])
    with read_record(keep_path) as rec:
        assert rec.n_frames == 40
        assert int(rec.keep.sum()) == n_kept
        for i in range(40):
            if not rec.keep[i]:
                try:
                    rec.frame(i)
                except ValueError:
                    continue
                raise AssertionError(f"frame {i} 未 keep 却有 z 图")
        keep_bytes = rec.payload_end
    with read_record(all_path) as rec_all:
        all_bytes = rec_all.payload_end
    assert keep_bytes < all_bytes


def test_sink_sidecar_enables_continuation(tmp_path):
    """RecordSink 落盘的侧车必须能回灌，且续采与不中断一致——
    这是 GUIDE §2.3.5（统计收敛期漏检最早事件）的工程解法。"""
    n = 60
    rng = np.random.default_rng(7)
    shapes = (3, 8, 8)
    front = _stub_frontend(shapes)

    # 一路跑到底（参照）
    full = FeatureMapCancel(CancelConfig(fps=30.0, f_map=2.0, robust=True))
    sur_all = []
    for i in range(n):
        frame = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        sur_all.append(full.step(np.asarray(front(frame)), np.zeros(3))["surprise"])

    # 跑前半段，落盘侧车；随后从侧车续采后半段
    half = FeatureMapCancel(CancelConfig(fps=30.0, f_map=2.0, robust=True))
    rng2 = np.random.default_rng(7)
    for i in range(n // 2):
        frame = rng2.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        half.step(np.asarray(front(frame)), np.zeros(3))

    layer = AcquisitionLayer(
        frontend=front,
        cancel_cfg=CancelConfig(fps=30.0, f_map=2.0, robust=True))
    layer.cancel.load_stats(half.dump_stats())
    path = str(tmp_path / "cont.cmcir")
    with RecordSink(path, layer) as sink:
        for i in range(n // 2, n):
            frame = rng2.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
            sink.write(layer.step(frame, np.zeros(3), i / 30.0))
    with read_record(path) as rec:
        cont = rec.surprise
    ref = np.asarray(sur_all[n // 2:], dtype=np.float32)
    assert cont.shape == ref.shape
    assert np.allclose(cont, ref, rtol=1e-5, atol=1e-5)
    # 且第一帧就有非零读数（冷启动会是 0）
    assert cont[0] > 0


def test_layer_step_exposes_timestamp():
    layer = AcquisitionLayer(
        frontend=_stub_frontend(),
        cancel_cfg=CancelConfig(fps=30.0, f_map=2.0, robust=True))
    out = layer.step(np.zeros((16, 16, 3), dtype=np.uint8), np.zeros(3), 1.25)
    assert out["t"] == 1.25

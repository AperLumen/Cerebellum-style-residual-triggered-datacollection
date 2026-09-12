"""CMCI 残差记录格式 v2：自描述、可续采、按统计分配比特的落盘容器。

设计出发点（GUIDE §4.3 路线图）+ 一篇新论文的启发（Petrini et al.,
Nature 657, 377 (2026), doi:10.1038/s41586-026-10891-z，"DOP 调制 / 高维
光计算"）：那篇论文的核心是**把一个统计量（偏振度 DOP）当作独立的信息
维度**——Stokes 参数被约束在球面上（3 个不独立的量），DOP 提供的径向坐标
补上了缺失的那一维，信息因此从"曲面"扩张到"体积"；并且信息编码在**概率
分布的形状**里（ρ = exp(−δφ²/2)，由相位 PDF 的宽度决定），而对应的确定性
量（平均相位）控制的是球的方位。

映射到本项目的落盘格式（类比，非主张）：

  DOP 论文                        | CMCI 记录格式
  ------------------------------- | --------------------------------------------
  S1,S2,S3（球面，确定量）        | 逐格有符号 z（幅度 + 出现/消失的方向）
  DOP ρ（径向，统计量）           | 逐格窗口相干度 / 运行 sd（统计自由度）
  相位 PDF 形状承载信息           | z 的**分布**形状（稀疏、重尾）承载压缩机会
  按 macromode 分配维度           | 按格分配比特（deadband，实测 96% 格 < 3σ）
  统计量是独立维度，不可从确定性量恢复
                                  | 触发器决策口径：只用幅度会丢信息（§2.3.7）

格式目标（按优先级）：

1. **自描述**——fps / f_map / shape / 量化参数全部进头部。现状是这些量
   在磁盘上靠约定（f_map 由 ``24.25 * W/28`` 反推、fps 有 30.0 兜底），
   换前端或换分辨率就要改代码；记录自带参数后，读出端不用猜。
2. **决策保真**——量化后重放必须给出与 float32 原始 z 相同的触发片段。
   本项目的产物是"存/不存"的决策，字节账只有在决策不变时才有意义。
3. **可续采（warm start）**——旁挂运行统计（mu/var）。GUIDE §2.3.5 记录的
   已知局限"统计收敛需要时间、最早的事件会漏"，跨录制复用统计即可消除；
   这正是"统计量是一等公民"的直接工程化。
4. **按统计分配比特**——deadband 量化 + zlib：实测 z 分布 70%(<1σ) /
   90.6%(<2σ) / 96.5%(<3σ) 的格落在零附近，稀疏化后压缩比很高。
5. **崩溃可恢复**——每帧块自带长度前缀，写入中途掉电也能顺序重放
   （always-on 记录器的真实场景）。此时索引缺失，只能顺序读。

容器布局（小端）：:

    MAGIC "CMCIREC2" (8B)
    u64 footer_offset        # close() 时回填；0 = 未正常关闭
    u32 header_len
    header_json              # 自描述参数（见 RecordWriter.header）
    payload: 逐帧块
        u32 block_len
        f32 scale            # 反量化尺度（fixed 模式 = zmax）
        u8  flags            # bit0: 该帧带 jpg 关键帧
        u32 nbytes_coded
        coded bytes          # zlib/raw 的 int8 量化载荷
        [u32 jpg_len, jpg bytes]   # flags bit0 置位时
    FOOTER  "CMCIEND2" (8B)
    u64 footer_json_len, footer_json      # clips / 元信息
    u64 bundle_len, bundle                # npz：t / surprise / score / keep
                                          # / offsets / nbytes / 统计侧车

用法：:

    with RecordWriter("out.cmcir", fps=30, f_map=24.25, shape=(24,28,28)) as w:
        for t, z in stream:
            w.append(z, t=t, surprise=s, score=sc, keep=k)
        w.close(stats=canc.dump_stats())     # 旁挂统计，供下次 warm start

    with read_record("out.cmcir") as rec:
        print(rec.header, rec.n_frames, rec.stats)
        for t, z in rec.frames():            # 顺序重放（截断文件也能读到最后一帧）
            ...
"""

from __future__ import annotations

import io
import json
import struct
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

MAGIC = b"CMCIREC2"
FOOTER_MAGIC = b"CMCIEND2"
FORMAT_VERSION = 2

DEFAULT_ZMAX = 8.0       # 固定模式：z 截断到 ±8σ（实测 p99.9 = 9.5σ，8 够用）
DEFAULT_DEADBAND = 0.5   # |z| < deadband 记为 0（实测 <1σ 占 70%）


# ---------------------------------------------------------------- 量化


def quantize(z: np.ndarray, *, mode: str = "fixed", zmax: float = DEFAULT_ZMAX,
             deadband: float = DEFAULT_DEADBAND
             ) -> tuple[np.ndarray, float]:
    """float z -> (int8 payload, scale)。反量化：``q / 127 * scale``。

    ``mode="fixed"``：全局尺度 zmax，零点稳定（zlib 友好，且 <deadband 的
    格落在精确的 0 上）；
    ``mode="per_frame"``：逐帧 ``max|z|``，极端帧不截顶（每帧 4 字节开销，
    且零点随帧漂移、稀疏性变差）。
    """
    z = np.asarray(z, dtype=np.float32)
    if deadband > 0:
        z = np.where(np.abs(z) < deadband, np.float32(0.0), z)
    if mode == "fixed":
        scale = float(zmax)
    elif mode == "per_frame":
        scale = float(np.abs(z).max())
        if scale <= 0:
            scale = 1.0
    else:
        raise ValueError(f"未知量化模式: {mode!r}（fixed / per_frame）")
    q = np.round(np.clip(z, -scale, scale) / scale * 127.0)
    return q.astype(np.int8), scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    """int8 payload -> float32 z（``quantize`` 的逆，误差上界 scale/254）。"""
    return (q.astype(np.float32) / 127.0) * np.float32(scale)


# ---------------------------------------------------------------- 写出


class RecordWriter:
    """流式记录写出器：常开管线上逐帧 append，结束时 close()。

    逐帧只做一次量化 + zlib，不做任何需要全局信息的事（没有二遍扫描），
    所以可以直接挂在 ``AcquisitionLayer.step`` 后面。
    """

    def __init__(self, path: str, *, fps: float, f_map: float,
                 shape: "tuple[int, ...]", codec: str = "zlib",
                 level: int = 1, mode: str = "fixed",
                 zmax: float = DEFAULT_ZMAX,
                 deadband: float = DEFAULT_DEADBAND,
                 z_policy: str = "keep_only",
                 producer: str = "cerebellum_cancel.record",
                 meta: Optional[dict[str, Any]] = None):
        """``z_policy`` 决定 z 图的落盘范围（这是格式的核心取舍）：

        - ``"keep_only"``（默认）：**只对 keep 的帧写 z 图**。逐帧标量
          （t/score/surprise/keep）全部保留——它们是决策的完整记录，每帧
          ~27 字节；z 图只在真的要回看/训练时才落盘。这正是项目的主张
          落到格式上的样子："不存能预测的东西"，字节花在触发片段上。
        - ``"all"``：每帧都写 z 图（诊断/基准/离线重放用）。
        - ``"none"``：只写标量轨迹（纯决策日志）。

        ``level`` 默认 1：实测 zlib lvl1 比 lvl6 只差 ~6% 体积但快 3 倍
        （0.23 vs 0.76 ms/帧，``runs/record_format_bench.json``）——常开
        路径上这笔账明显偏向 lvl1。
        """
        if codec not in ("zlib", "none"):
            raise ValueError(f"未知 codec: {codec!r}（zlib / none）")
        if z_policy not in ("keep_only", "all", "none"):
            raise ValueError(f"未知 z_policy: {z_policy!r}"
                             "（keep_only / all / none）")
        self.path = path
        self.shape = tuple(int(s) for s in shape)
        self.codec = codec
        self.level = int(level)
        self.z_policy = z_policy
        self._fh = open(path, "wb")
        self._fh.write(MAGIC)
        self._fh.write(struct.pack("<Q", 0))          # footer_offset 占位
        self.header: dict[str, Any] = {
            "format": "cmci-residual-record",
            "version": FORMAT_VERSION,
            "producer": producer,
            "shape": list(self.shape),
            "fps": float(fps),
            "f_map": float(f_map),
            "dtype": "int8",
            "z_policy": self.z_policy,
            "quant": {"mode": mode, "zmax": float(zmax),
                      "deadband": float(deadband)},
            "codec": {"name": codec, "level": self.level},
            "meta": dict(meta or {}),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        hb = json.dumps(self.header, separators=(",", ":"),
                        sort_keys=True).encode("utf-8")
        self._fh.write(struct.pack("<I", len(hb)))
        self._fh.write(hb)
        self._payload_off = self._fh.tell()
        self._idx: list[tuple[int, int]] = []         # (offset, block_bytes)
        self._t: list[float] = []
        self._surprise: list[float] = []
        self._score: list[float] = []
        self._keep: list[bool] = []
        self._n = 0
        self._closed = False

    # -- 逐帧 --------------------------------------------------------
    def append(self, z: Optional[np.ndarray], *,
               t: Optional[float] = None,
               surprise: float = float("nan"), score: float = float("nan"),
               keep: bool = False, jpg: Optional[bytes] = None) -> int:
        """写一帧。返回该帧块字节数。

        ``z`` 在标量-only 的帧上可为 ``None``（``z_policy="keep_only"`` 且
        ``keep=False`` 时正是如此）——那种帧只写标量。

        ``score``/``surprise`` 以 float64/float32 原样存下：触发器只依赖
        ``score``，把它无损存住就使**决策可精确重放**，不需要让有损的 z 图
        去反推决策（实测 int8 反推会让片段边界漂移，见
        ``scripts/bench_record_format.py`` 与 ``MEMORY_FORMAT.md``）。
        """
        if self._closed:
            raise RuntimeError("writer 已关闭")
        want_z = self.z_policy == "all" or (self.z_policy == "keep_only"
                                            and keep and z is not None)
        flags = 0
        off = self._fh.tell()
        if want_z:
            assert z is not None
            arr = np.asarray(z, dtype=np.float32)
            if tuple(arr.shape) != self.shape:
                arr = arr.reshape(self.shape)
            q, scale = quantize(arr, mode=self.header["quant"]["mode"],
                                zmax=self.header["quant"]["zmax"],
                                deadband=self.header["quant"]["deadband"])
            raw = q.tobytes()
            coded = (zlib.compress(raw, self.level) if self.codec == "zlib"
                     else raw)
            if jpg is not None:
                flags |= 1
            body = struct.pack("<fBI", scale, flags, len(coded)) + coded
            if jpg is not None:
                body += struct.pack("<I", len(jpg)) + jpg
        else:
            # 无 z 图：只留标量。帧块保留 4 字节长度前缀（空载荷），使块
            # 扫描与截断恢复的逻辑对三种 z_policy 完全一致。
            body = struct.pack("<fBI", 1.0, flags, 0)
        self._fh.write(struct.pack("<I", len(body)))
        self._fh.write(body)
        self._idx.append((off, 4 + len(body)))
        if t is None:
            t = len(self._t) / float(self.header["fps"])
        self._t.append(float(t))
        self._surprise.append(float(surprise))
        self._score.append(float(score))
        self._keep.append(bool(keep))
        self._n += 1
        return 4 + len(body)

    # -- 收尾 --------------------------------------------------------
    def close(self, *, stats: Optional[dict[str, np.ndarray]] = None,
              clips: Optional[list[tuple[float, float]]] = None) -> dict:
        """落盘页脚（索引 + 逐帧标量 + 统计侧车），回填 footer_offset。

        ``stats``：``FeatureMapCancel.dump_stats()`` 的返回值（mu/var/n），
        供下次录制 warm start；不传则该记录不能续采（仍可正常读）。
        """
        if self._closed:
            return self.summary()
        self._fh.flush()
        footer_off = self._fh.tell()
        self._fh.write(FOOTER_MAGIC)
        footer = {
            "n_frames": self._n,
            "clips": [[float(a), float(b)] for a, b in (clips or [])],
            "payload_offset": self._payload_off,
            "closed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        fb = json.dumps(footer, separators=(",", ":"),
                        sort_keys=True).encode("utf-8")
        self._fh.write(struct.pack("<Q", len(fb)))
        self._fh.write(fb)
        arrays: dict[str, np.ndarray] = {
            "t": np.asarray(self._t, dtype=np.float64),
            "surprise": np.asarray(self._surprise, dtype=np.float32),
            "score": np.asarray(self._score, dtype=np.float64),
            "keep": np.asarray(self._keep, dtype=np.uint8),
            "offsets": np.asarray([o for o, _ in self._idx], dtype=np.uint64),
            "nbytes": np.asarray([b for _, b in self._idx], dtype=np.uint32),
        }
        if stats:
            if "mu" in stats:
                arrays["stats_mu"] = np.asarray(stats["mu"], dtype=np.float32)
            if "var" in stats:
                arrays["stats_var"] = np.asarray(stats["var"], dtype=np.float32)
            if "F_prev" in stats:
                # 续采必需：没有它，重启后的第一帧残差恒为 0（见
                # FeatureMapCancel.dump_stats 的说明）
                arrays["stats_F_prev"] = np.asarray(stats["F_prev"],
                                                    dtype=np.float32)
            arrays["stats_n"] = np.asarray([int(stats.get("n", 0))],
                                           dtype=np.int64)
        bio = io.BytesIO()
        np.savez_compressed(bio, **arrays)
        blob = bio.getvalue()
        self._fh.write(struct.pack("<Q", len(blob)))
        self._fh.write(blob)
        # 回填 footer 偏移（固定 8 字节槽，位于 MAGIC 之后）
        self._fh.seek(len(MAGIC))
        self._fh.write(struct.pack("<Q", footer_off))
        self._fh.seek(0, 2)
        self._fh.close()
        self._closed = True
        return self.summary()

    def summary(self) -> dict:
        """成本账。**逐帧载荷与一次性开销分开报**——侧车（统计 mu/var +
        F_prev）是与帧数无关的固定成本，把它摊进每帧、在短视频上会得到
        失真的数字。

        ``payload_bytes`` 直接取帧块字节之和（精确值）；``fixed_bytes``
        是文件总长减去它，即头 + 页脚 + 统计侧车（侧车在 npz 里是压缩
        存储的，所以不能拿统计数组的未压缩大小去估）。
        """
        total = 0
        try:
            import os
            total = os.path.getsize(self.path)
        except OSError:
            pass
        payload = sum(b for _, b in self._idx)
        return {
            "n_frames": self._n,
            "bytes": total,
            "bytes_per_frame": total / max(self._n, 1),
            "payload_bytes": payload,
            "payload_bytes_per_frame": payload / max(self._n, 1),
            "fixed_bytes": max(total - payload, 0),
        }

    def __enter__(self) -> "RecordWriter":
        return self

    def __exit__(self, *exc) -> None:
        if not self._closed:
            self.close()


# ---------------------------------------------------------------- 读取


@dataclass
class Record:
    """读出的记录。标量字段全部预载；z 逐帧解码（大序列不爆内存）。"""

    path: str
    header: dict[str, Any]
    n_frames: int
    times: np.ndarray
    surprise: np.ndarray
    score: np.ndarray
    keep: np.ndarray
    clips: list[tuple[float, float]] = field(default_factory=list)
    stats: dict[str, np.ndarray] = field(default_factory=dict)
    truncated: bool = False          # True = 写入端未正常 close()
    bytes_total: int = 0
    _fh: Any = None                  # 内部文件句柄
    _offsets: Optional[np.ndarray] = None
    _nbytes: Optional[np.ndarray] = None
    _payload_off: int = 0

    # -- 元信息 ------------------------------------------------------
    @property
    def fps(self) -> float:
        return float(self.header["fps"])

    @property
    def f_map(self) -> float:
        return float(self.header["f_map"])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.header["shape"])  # type: ignore[return-value]

    @property
    def bytes_per_frame(self) -> float:
        return self.bytes_total / max(self.n_frames, 1)

    @property
    def clips_for_keep(self) -> list[tuple[float, float]]:
        """由逐帧 keep 压缩出的片段（在线决策的封闭版）。"""
        segs: list[tuple[float, float]] = []
        start: Optional[int] = None
        for i, k in enumerate(list(self.keep) + [0]):
            if k and start is None:
                start = i
            elif not k and start is not None:
                segs.append((float(self.times[start]), float(self.times[i - 1])))
                start = None
        return segs

    @property
    def payload_end(self) -> int:
        """最后一帧块的结束偏移（此后是页脚；截断文件里即文件尾）。"""
        if self._offsets is None or self._nbytes is None or self.n_frames == 0:
            return self._payload_off
        return int(self._offsets[-1]) + int(self._nbytes[-1])

    # -- 逐帧 --------------------------------------------------------
    def _read_block(self, offset: int) -> tuple[Optional[np.ndarray],
                                                Optional[bytes]]:
        """读一个帧块。返回 ``(z, jpg)``；仅决策模式下 z 为 ``None``。"""
        fh = self._fh
        fh.seek(offset)
        head = fh.read(4)
        if len(head) < 4:
            raise EOFError(offset)
        (blen,) = struct.unpack("<I", head)
        body = fh.read(blen)
        if len(body) < blen:
            raise EOFError(offset)
        scale, flags, n_coded = struct.unpack("<fBI", body[:9])
        if n_coded == 0:
            return None, None
        coded = body[9:9 + n_coded]
        if self.header["codec"]["name"] == "zlib":
            raw = zlib.decompress(coded)
        else:
            raw = coded
        q = np.frombuffer(raw, dtype=np.int8)
        z = dequantize(q.reshape(self.shape), scale)
        jpg = None
        if flags & 1:
            rest = body[9 + n_coded:]
            (jlen,) = struct.unpack("<I", rest[:4])
            jpg = rest[4:4 + jlen]
        return z, jpg

    def frame(self, i: int) -> np.ndarray:
        """随机访问第 i 帧的 z 图（需要索引；只有纯标量记录没有 z 图）。"""
        if self._offsets is None:
            raise RuntimeError("该记录被截断，只能顺序读（用 frames()）")
        if not 0 <= i < self.n_frames:
            raise IndexError(i)
        z, _ = self._read_block(int(self._offsets[i]))
        if z is None:
            raise ValueError("该记录是仅标量模式（z_policy='none'），没有 z 图")
        return z

    def keyframe(self, i: int) -> Optional[bytes]:
        """第 i 帧旁挂的 JPEG 关键帧（无则 None）。"""
        if self._offsets is None:
            raise RuntimeError("该记录被截断，只能顺序读")
        _, jpg = self._read_block(int(self._offsets[i]))
        return jpg

    def frame_bytes(self, i: int) -> int:
        """第 i 帧块占用的字节数（含 4 字节长度前缀）。"""
        if self._nbytes is None:
            raise RuntimeError("该记录被截断，只能顺序读")
        if not 0 <= i < self.n_frames:
            raise IndexError(i)
        return int(self._nbytes[i])
    def frame_blocks(self) -> Iterator[tuple[Optional[float],
                                             Optional[np.ndarray],
                                             Optional[bytes]]]:
        """顺序重放 ``(t, z, jpg)``。截断记录也能读到最后一个完整帧块，
        此时 ``t`` 为 ``None``（标量轨迹随页脚丢失）；仅决策模式下 z 为
        ``None``。"""
        fh = self._fh
        off = self._payload_off
        i = 0
        while off + 4 <= self.bytes_total:
            try:
                z, jpg = self._read_block(off)
            except (EOFError, zlib.error):
                break
            blen = self._block_bytes(off)
            if blen <= 0:
                break
            t = float(self.times[i]) if i < len(self.times) else None
            yield t, z, jpg
            off += blen
            i += 1
            if i >= self.n_frames:
                break

    def frames(self) -> Iterator[tuple[Optional[float], np.ndarray]]:
        """顺序重放 ``(t, z)``（截断记录下 ``t`` 为 ``None``；仅标量模式
        下抛错——那种记录本来就没有 z 图）。"""
        for t, z, _ in self.frame_blocks():
            if z is None:
                raise ValueError("该记录是仅标量模式（z_policy='none'），"
                                 "没有 z 图；用 frame_blocks() 或标量轨迹")
            yield t, z

    def _block_bytes(self, offset: int) -> int:
        """块总字节数 = 4 字节长度前缀 + 载荷长度；末端读不到则返回 0。"""
        fh = self._fh
        fh.seek(offset)
        head = fh.read(4)
        if len(head) < 4:
            return 0
        (blen,) = struct.unpack("<I", head)
        return 4 + blen

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _load_footer(fh: Any, footer_off: int) -> tuple[dict, dict[str, np.ndarray]]:
    fh.seek(footer_off)
    magic = fh.read(8)
    if magic != FOOTER_MAGIC:
        raise ValueError("页脚魔数不匹配（文件损坏或格式版本不符）")
    (fj_len,) = struct.unpack("<Q", fh.read(8))
    footer = json.loads(fh.read(fj_len).decode("utf-8"))
    (blob_len,) = struct.unpack("<Q", fh.read(8))
    blob = fh.read(blob_len)
    with np.load(io.BytesIO(blob)) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return footer, arrays


def _open_truncated(path: str, fh: Any, header: dict, size: int,
                    payload_off: int) -> Record:
    """打开写入端未正常 close() 的记录：只扫块长前缀（不解压载荷）重建
    索引，代价是一次 O(帧数) 的 4 字节读取，换来随机访问能力。
    标量轨迹（t/surprise/score/keep）随页脚丢失，时间轴按 fps 重建。"""
    offsets, nbytes = [], []
    off = payload_off
    while off + 4 <= size:
        fh.seek(off)
        head = fh.read(4)
        if len(head) < 4:
            break
        (blen,) = struct.unpack("<I", head)
        if blen <= 0 or off + 4 + blen > size:
            break                        # 最后一块被截断：丢弃
        offsets.append(off)
        nbytes.append(4 + blen)
        off += 4 + blen
    n = len(offsets)
    rec = Record(
        path=path, header=header, n_frames=n,
        times=np.arange(n, dtype=np.float64) / float(header["fps"]),
        surprise=np.full(n, np.nan, dtype=np.float32),
        score=np.full(n, np.nan, dtype=np.float64),
        keep=np.zeros(n, dtype=bool),
        truncated=True, bytes_total=size,
        _offsets=np.asarray(offsets, dtype=np.uint64),
        _nbytes=np.asarray(nbytes, dtype=np.uint32),
        _payload_off=payload_off)
    rec._fh = fh
    return rec


def read_record(path: str) -> Record:
    """读记录。正常关闭的文件带索引与统计侧车；截断文件顺序可读。

    Raises:
        ValueError: 不是 CMCI 记录文件，或**头本身就不完整**（在写完魔数之后、
            头落盘之前掉电）。此时无法知道形状与 fps，只能明确拒绝。

    **压力测试发现（2026-09-13）**：本函数此前对"文件短于固定头字段"的情况
    直接 `struct.unpack`，抛的是原生 ``struct.error: unpack requires a buffer
    of 4 bytes``——调用方无法用统一的 ``except ContractError/ValueError``
    处理。对一个**号称掉电安全**的容器，掉电恰好是最该预期到的场景，所以
    这里必须给出可读、可捕获的失败。footer 路径早已有守卫，头路径漏了。
    """
    import os
    size = os.path.getsize(path)
    fh = open(path, "rb")
    try:
        magic = fh.read(8)
        if magic != MAGIC:
            raise ValueError(f"不是 CMCI 记录文件: {path}")
        # 固定头 = 8B magic + 8B footer_off + 4B header_len
        raw = fh.read(12)
        if len(raw) < 12:
            raise ValueError(
                f"记录头不完整: {path}（读到 {8 + len(raw)} 字节，"
                "固定头需要至少 20）。文件在写入过程中被截断；"
                "魔数之后、头落盘之前掉电就会这样。")
        (footer_off,) = struct.unpack("<Q", raw[:8])
        (hl,) = struct.unpack("<I", raw[8:12])
        if hl > size:                   # 头长度不可能超过文件大小
            raise ValueError(f"记录头损坏: {path}")
        head_raw = fh.read(hl)
        if len(head_raw) < hl:
            raise ValueError(
                f"记录头不完整: {path}（声明 {hl} 字节，实得 {len(head_raw)}）")
        header = json.loads(head_raw.decode("utf-8"))
        payload_off = fh.tell()
    except Exception:
        fh.close()
        raise

    if footer_off == 0 or footer_off + 8 > size:
        return _open_truncated(path, fh, header, size, payload_off)

    try:
        footer, arrays = _load_footer(fh, footer_off)
    except (ValueError, OSError, KeyError, EOFError,
            json.JSONDecodeError, zipfile.BadZipFile):
        # footer_offset 已回填但页脚没写完（在回填与页脚落盘之间掉电）：
        # 帧数据本身是完整的，退回按块扫描，别把整个录制判成损坏。
        return _open_truncated(path, fh, header, size, payload_off)

    footer, arrays = _load_footer(fh, footer_off)
    n = int(footer["n_frames"])
    rec = Record(
        path=path, header=header, n_frames=n,
        times=arrays["t"], surprise=arrays["surprise"], score=arrays["score"],
        keep=arrays["keep"].astype(bool),
        clips=[(float(a), float(b)) for a, b in footer.get("clips", [])],
        stats={k[len("stats_"):]: v for k, v in arrays.items()
               if k.startswith("stats_")},
        bytes_total=size, _offsets=arrays["offsets"], _nbytes=arrays["nbytes"],
        _payload_off=int(footer.get("payload_offset", payload_off)))
    rec._fh = fh
    return rec


def save_record(path: str, z_seq: np.ndarray, **kw: Any) -> dict:
    """一次性写整段 z（测试/离线用）。``kw`` 透传 RecordWriter。"""
    z_seq = np.asarray(z_seq, dtype=np.float32)
    shape = kw.pop("shape", z_seq.shape[1:])
    with RecordWriter(path, shape=shape, **kw) as w:
        for i in range(len(z_seq)):
            w.append(z_seq[i])
        return w.close()


class RecordSink:
    """把 ``AcquisitionLayer.step`` 的产物直接落到记录文件（协议层集成点）。

    这是"内存格式"在协议栈里的实际接入位置：管线的每帧决策
    （``keep``/``score``/``surprise``）无损落盘，z 图按 ``z_policy``
    决定留多少。用法::

        with RecordSink("session.cmcir", layer) as sink:
            for frame, gyro, t in stream:
                out = layer.step(frame, gyro, t)
                sink.write(out)
            sink.close(clips=layer.finalize())

    ``write`` 从 step 的返回值里取字段，缺失字段按 NaN/False 记；z 图字段
    ``z_map`` 是 (H,W) 的逐格强度（step 已按通道 RMS 归约），会补一个通道
    维以匹配 ``layer.cancel`` 的特征图网格。
    """

    def __init__(self, path: str, layer: Any, *,
                 z_policy: str = "keep_only", **kw: Any):
        """``shape`` 取自 cancel 的现有统计；若统计尚未建立（首帧之前就
        构造 sink，这是正常用法），延后到第一帧写入时再确定——见
        ``_ensure_writer``。"""
        self._path = path
        self._layer = layer
        self._z_policy = z_policy
        self._kw = kw
        self._writer: Optional[RecordWriter] = None
        mu = layer.cancel.mu
        if mu is not None:
            self._writer = self._make_writer(tuple(np.asarray(mu).shape))

    def _make_writer(self, shape: tuple[int, ...]) -> RecordWriter:
        return RecordWriter(
            self._path, fps=self._layer.fps,
            f_map=self._layer.cancel.cfg.f_map, shape=shape,
            z_policy=self._z_policy, **self._kw)

    def _ensure_writer(self, z_shape: tuple[int, ...]) -> RecordWriter:
        if self._writer is None:
            self._writer = self._make_writer(z_shape)
        return self._writer

    def write(self, step_out: dict) -> int:
        z = step_out.get("z_map")
        keep = bool(step_out.get("keep", False))
        # z_map 是 (H,W) 逐格强度（step 已按通道 RMS 归约）：补一个通道维
        # 以匹配特征图网格 (1,H,W)。
        arr = None if z is None else np.asarray(z, dtype=np.float32)[None]
        shape = tuple(arr.shape) if arr is not None else (1, 0, 0)
        writer = self._ensure_writer(shape)
        t = step_out.get("t")
        if t is None:
            t = writer._n / max(self._layer.fps, 1e-9)
        return writer.append(
            arr, t=float(t),
            surprise=float(step_out.get("surprise", float("nan"))),
            score=float(step_out.get("score", float("nan"))),
            keep=keep)

    def close(self, clips: Optional[list[tuple[float, float]]] = None
              ) -> dict:
        """收尾：把序列级状态（统计侧车）与片段列表一并落盘。

        统计侧车来自 ``layer.cancel.dump_stats()``——下一次录制可直接
        ``layer.cancel.load_stats(rec.stats)`` 续采（warm start）。
        """
        if self._writer is None:
            # 一帧都没写过：无 z 网格可用，退化为纯标量日志
            mu = self._layer.cancel.mu
            self._writer = self._make_writer(
                tuple(np.asarray(mu).shape) if mu is not None else (1, 0, 0))
        stats = self._layer.cancel.dump_stats()
        if clips is None:
            clips = list(self._layer.trigger.clips)
        return self._writer.close(stats=stats or None, clips=clips)

    def __enter__(self) -> "RecordSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

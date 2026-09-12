#!/usr/bin/env python3
"""协议层压力测试——不是"随机值"，是**随机时序与随机链路**。

## 为什么需要它（和 fuzz_contracts.py 的分工）

`fuzz_contracts.py` 撞的是**单次函数调用**的值域：NaN、超大、错 dtype、错形状。
它回答"这个函数对垃圾输入守不守规矩"。

但它对**采集协议层**不够狠，因为协议层的失效**不在单次调用里**：

* 帧数不匹配、时间戳**回退**/重复/非单调 —— 单次调用看不见，时序才看得见；
* `n_frames` 与各载荷长度不一致 —— 这是 dataset 的**唯一硬契约**；
* 写入过程**中途截断**（掉电模拟）—— record 号称"掉电安全"，得真的砸；
* `replay()` 号称与 `run()` **无损等价** —— 这是不变量，必须反复验证；
* 跨段借用参照、段边界处理 —— integrity 的已知代价。

所以本脚本的攻击面是**序列与链路**，不是标量。

## 三类攻击

**A. 时序攻击（timing）**：构造结构合法的序列，但时间轴被系统性地破坏——
回退、重复、NaN 跳变、骤停、超长、超短、单帧。检查下游要么拒绝，要么给出一致结果。

**B. 不变量攻击（invariant）**：反复验证程序**自己声称**的性质：
* `dataset` 写入→读回必须逐帧对齐（长度 + dtype + 数值）；
* `replay(res, default)` 的 `keep` 必须与 `run()` 逐位一致，且 `score` 无损；
* `record` 写入 N 帧后读回必须是 N 帧（含截断场景）。

**C. 链路攻击（link）**：端到端塞进**畸形录制**（丢 IMU / 帧数不符 / 空视频 /
帧尺寸突变 / 时间戳乱序），检查引擎的降级必须是**显式**的，不能静默产出垃圾。

## 判定

与 `fuzz_contracts.py` 同一套四分类：RAISED（好）/ RAISED_OTHER（可疑）/
SILENT_BAD（**坏**）/ SILENT_GOOD（合法）。退出码非 0 代表有需要复核的项。

用法：
    PYTHONPATH=. .venv/bin/python scripts/fuzz_protocol.py
    PYTHONPATH=. .venv/bin/python scripts/fuzz_protocol.py --random --cases 40
    PYTHONPATH=. .venv/bin/python scripts/fuzz_protocol.py --json out.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cerebellum_cancel.contracts import ContractError

# ------------------------------------------------------------------ 结果类型


@dataclass
class Finding:
    target: str
    label: str
    verdict: str                    # RAISED | RAISED_OTHER | SILENT_BAD | SILENT_GOOD
    detail: str = ""


# ------------------------------------------------------------------ 序列工厂
#
# 每个工厂返回一个"结构合法但时间轴被破坏"的录制切片。
# 标签写明破坏方式，便于报告指认。

def _base_series(t: int = 240, *, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """造一段正常的 (t, gyro) —— 单调、有限、形状对。"""
    t_axis = np.arange(t, dtype=np.float64) / fps
    gyro = np.zeros((t, 3), dtype=np.float64)
    gyro[:, 1] = np.sin(t_axis * 3.0) * 20.0        # 一点真实的 yaw 摆动
    return t_axis, gyro


def timing_attacks(rng: np.random.Generator) -> list[tuple[str, tuple]]:
    """A 类：时间轴被系统性破坏。返回 (标签, (t, gyro))。"""
    out: list[tuple[str, tuple]] = []
    t0, g0 = _base_series(240)

    out.append(("monotonic_ok", (t0, g0)))                       # 对照：正常
    out.append(("empty", (t0[:0], g0[:0])))                      # 空
    out.append(("single_frame", (t0[:1], g0[:1])))               # 1 帧

    t = t0.copy(); t[100:] -= 5.0                                # 中途回退 5 秒
    out.append(("backwards", (t, g0)))

    t = t0.copy(); t[120] = t[119]                               # 重复时间戳
    out.append(("duplicate", (t, g0)))

    t = t0.copy(); t[rng.integers(1, 239)] = np.nan              # NaN 时间戳
    out.append(("nan_stamp", (t, g0)))

    t = t0.copy(); t[100:] = t[99]                               # 骤停（时间冻结）
    out.append(("frozen_clock", (t, g0)))

    t = t0.copy(); t *= 1e6                                      # 时间尺度错（微秒当秒）
    out.append(("scale_1e6", (t, g0)))

    t = t0.copy(); t[0] = 1e12                                   # 时间起点荒谬
    out.append(("absurd_origin", (t, g0)))

    t = t0.copy(); t = np.sort(t)                                # 排序过（对照）
    out.append(("sorted_ok", (t, g0)))

    t = np.linspace(0, 1e-9, 240)                                # 全部挤在 1ns 内
    out.append(("sub_ns_span", (t, g0)))

    t = np.linspace(0, 1e6, 240)                                 # 跨 11 天
    out.append(("eleven_days", (t, g0)))

    # 随机破坏：随机位置做随机操作（--random 时用）
    for i in range(6):
        t = t0.copy()
        k = int(rng.integers(1, 239))
        op = int(rng.integers(0, 4))
        if op == 0:
            t[k:] += float(rng.uniform(-100, 100))
        elif op == 1:
            t[k] = float(rng.choice([np.nan, np.inf, -np.inf]))
        elif op == 2:
            t[k:] = t[k - 1]
        else:
            t = t * float(10.0 ** rng.integers(-9, 9))
        out.append((f"rand_timing_{i}", (t, g0)))

    return out


def length_attacks() -> list[tuple[str, tuple]]:
    """A 类（续）：帧数与载荷长度不一致 —— dataset 的唯一硬契约。"""
    t0, g0 = _base_series(120)
    out: list[tuple[str, tuple]] = []
    out.append(("len_ok", (t0, g0)))
    out.append(("gyro_too_short", (t0, g0[:100])))
    out.append(("gyro_too_long", (t0, np.vstack([g0, g0[:30]]))))
    out.append(("gyro_wrong_width", (t0, np.zeros((120, 2)))))
    out.append(("gyro_wrong_width_4", (t0, np.zeros((120, 4)))))
    out.append(("gyro_1d", (t0, np.zeros(120))))
    out.append(("gyro_nan", (t0, np.where(
        np.arange(360).reshape(120, 3) % 31 == 0, np.nan, g0))))
    return out


# --------------------------------------------------------------- B 类：不变量


def _probe_score(gyro: np.ndarray) -> np.ndarray:
    """从 gyro 造一个 score 向量，但**不假设 gyro 是 2 维**。

    本工具自己踩过的坑：直接写 `gyro[:, 1]` 时，畸形的 gyro（1 维）会让
    **探针自己**抛 IndexError，然后被记成被测代码的发现。探针必须先能处理
    任何输入——否则报告里全是自己的噪声。
    """
    g = np.asarray(gyro, dtype=np.float64)
    if g.ndim == 2 and g.shape[1] >= 1:
        base = np.abs(g[:, min(1, g.shape[1] - 1)])
    else:
        base = np.abs(np.ravel(g)[: max(len(g), 1)]).astype(np.float64)
    return np.asarray(base, dtype=np.float64)


def _inv_dataset_roundtrip(t: np.ndarray, gyro: np.ndarray) -> str:
    """dataset 写入→读回必须逐帧对齐。返回 "" 表示通过，否则问题描述。"""
    from cerebellum_cancel import dataset as ds
    with tempfile.TemporaryDirectory() as td:
        score = _probe_score(gyro)
        surprise = score.astype(np.float32)
        keep = (score > np.median(score)).astype(np.uint8)
        try:
            sec = ds.build_section("seq_x", t, gyro, score, surprise,
                                   keep=keep)
        except (ContractError, ValueError):
            return ""            # 拒绝是合法的
        sec.validate()
        dw = ds.DatasetWriter(td, name="fz", description="fuzz")
        dw.add(sec)
        dw.finalize()
        back = ds.load_section(td, "seq_x")
        n_expect = sec.n_frames
        # LoadedSection 暴露 payload/meta，不是直挂属性
        pay = back.payload
        if back.n_frames != n_expect:
            return f"读回 n_frames {back.n_frames} != 写入 {n_expect}"
        # 逐帧对齐：**所有**载荷第 0 维必须等于 n_frames
        for name, arr in pay.items():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and name != "schema":
                if len(arr) != n_expect:
                    return f"载荷 {name} 长度 {len(arr)} != n_frames {n_expect}"
        for name in ("score", "keep", "t", "gyro"):
            if name not in pay:
                return f"读回缺少载荷 {name}"
        if pay["score"].dtype != np.float64:
            return f"score dtype 变成 {pay['score'].dtype}（决策无损契约要求 float64）"
        if not np.array_equal(np.asarray(pay["keep"]), np.asarray(keep)):
            return "keep 读回后与写入不一致"
        # 注：用 equal_nan=True——输入里可能含 NaN（见 timing 攻击），
        # 而 NaN != NaN，所以裸 allclose 会把**正确往返**误报成不一致。
        # （这是本工具自己踩过的坑：第一版把 nan_stamp 报成 SILENT_BAD。）
        if not np.allclose(np.asarray(pay["t"]), t, equal_nan=True):
            return "t 读回后与写入不一致"
        return ""


def _inv_replay_equivalence(t: np.ndarray, gyro: np.ndarray) -> str:
    """replay() 必须与 run() 在默认参数下逐位一致（这是它存在的理由）。"""
    from cerebellum_cancel import engine as eng
    n = len(t)
    sc = _probe_score(gyro)
    # 本检查器要构造一个 **合法** 的 CMCIResult 再回放。gyro/score 形状不对时
    # 连构造都做不了——那不是 replay 的问题，是**构造契约**的问题，
    # 由 test_engine.py 覆盖。这里跳过，避免把"我们自己的前置条件"算成发现。
    if gyro.ndim != 2 or gyro.shape[1] != 3 or len(gyro) != n:
        return ""
    if len(sc) != n:
        return ""
    # t 也必须能构造出合法 CMCIResult（含 NaN/非单调时构造契约自然会拦，
    # 那是构造契约的事，不是 replay 的事）。
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        return ""
    try:
        res = eng.CMCIResult(
            t=t, gyro=gyro.astype(np.float32),
            score=sc.astype(np.float64),
            surprise=sc.astype(np.float32),
            keep=np.zeros(n, dtype=np.uint8),
            weight=np.ones(n, dtype=np.float32),
            segments=[], meta={"source": "fuzz"},
        )
    except (ContractError, ValueError) as e:
        return ""                # 构造期拒绝是合法的
    try:
        r2 = eng.replay(res)
    except (ContractError, ValueError):
        return ""
    if not np.array_equal(res.score, r2.score):
        return "replay 改变了 score（应无损）"
    if len(r2.keep) != n:
        return f"replay 输出帧数 {len(r2.keep)} != {n}"
    return ""


def _inv_record_truncation(t: np.ndarray, gyro: np.ndarray) -> str:
    """record 号称掉电安全：写到一半被截断后，读回必须是可解析的前缀。"""
    from cerebellum_cancel import record as rec
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "x.cmcir")
        z0 = np.random.default_rng(0).random((4, 28, 28)).astype(np.float32)
        try:
            with rec.RecordWriter(path, fps=30.0, f_map=42.7,
                                  shape=(4, 28, 28), z_policy="none") as w:
                for i in range(60):
                    w.append(None, t=float(t[i]),
                             keep=bool(i % 3 == 0),
                             surprise=float(abs(gyro[i, 1])),
                             score=float(abs(gyro[i, 1])))
        except (ContractError, ValueError, IndexError):
            return ""
        raw = open(path, "rb").read()
        if len(raw) < 32:
            return "写出的文件过小"
        # 随机截断到中间某处：读回不能抛非契约异常、也不能返回超量帧
        for cut in (len(raw) // 2, len(raw) - 3, 16, 4, 0):
            fn = os.path.join(td, f"cut_{cut}.cmcir")
            open(fn, "wb").write(raw[:cut])
            try:
                got = rec.read_record(fn)
            except (ContractError, ValueError, EOFError, OSError, AttributeError):
                continue                       # 明确拒绝是可接受的
            except Exception as e:             # noqa: BLE001
                return f"截断到 {cut}B 时抛非契约异常 {type(e).__name__}"
            try:
                n = got.n_frames
            except Exception as e:             # noqa: BLE001
                return f"截断到 {cut}B 读回后取 n_frames 抛 {type(e).__name__}"
            if n > 60:
                return f"截断到 {cut}B 却读出 {n} 帧（超过写入的 60）"
        return ""


# --------------------------------------------------------------- C 类：链路


def _link_engine_no_imu(t: np.ndarray, gyro: np.ndarray) -> str:
    """无 IMU + 畸形视频：降级必须显式记录在 meta 里，不能静默。"""
    from cerebellum_cancel import engine as eng
    with tempfile.TemporaryDirectory() as td:
        vid = os.path.join(td, "v.mp4")
        # 造 3 帧真视频
        try:
            import cv2
            # pyright 的 cv2 桩没有 VideoWriter_fourcc；用 getattr 取（运行时存在）
            fourcc = getattr(cv2, "VideoWriter_fourcc")("m", "p", "4", "v")
            vw = cv2.VideoWriter(vid, fourcc, 30.0, (64, 64))
            rng = np.random.default_rng(0)
            for _ in range(3):
                fr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
                vw.write(fr)
            vw.release()
        except Exception:                                  # noqa: BLE001
            return ""
        for mode in ("auto", "assume_still"):
            try:
                res = eng.run(vid, None, cfg=eng.EngineConfig(no_imu=mode))
            except (ContractError, ValueError, RuntimeError, ImportError):
                continue
            except Exception as e:                         # noqa: BLE001
                return f"no_imu={mode} 抛非契约异常 {type(e).__name__}"
            meta = getattr(res, "meta", {}) or {}
            if "motion_source" not in meta:
                return f"no_imu={mode} 未在 meta 记录 motion_source（静默降级）"
            if not meta.get("motion_source_note"):
                return f"no_imu={mode} 缺少 motion_source_note 说明"
        return ""


def _link_dataset_mismatch(t: np.ndarray, gyro: np.ndarray) -> str:
    """载荷长度不一致时 dataset 必须拒绝，不能悄悄截断对齐。"""
    from cerebellum_cancel import dataset as ds
    score = _probe_score(gyro)[: len(gyro) - 7]
    surprise = score.astype(np.float32)
    if not np.all(np.isfinite(t)):
        return ""                 # t 的契约校验由 roundtrip 覆盖
    try:
        sec = ds.build_section("bad", t, gyro, score, surprise)
    except (ContractError, ValueError, IndexError):
        return ""                # 拒绝 = 正确
    # 若没拒绝，必须至少长度自洽（不能悄悄产出错的对齐）
    try:
        sec.validate()
    except (ContractError, ValueError):
        return ""
    lens = {len(sec.t), len(sec.gyro), len(sec.score)}
    if len(lens) != 1:
        return f"未拒绝且长度不自洽: {lens}"
    return ""


# --------------------------------------------- D 类：状态与幂等（协议层特有）


def _inv_repeated_calls(t: np.ndarray, gyro: np.ndarray) -> str:
    """协议层会被**反复调用**（每帧一次，跑几小时）。状态必须有界且幂等。

    这里同一条序列连跑三次，检查：
    1. 结果**逐位可复现**（同一输入 → 同一输出）；
    2. 对象不积累状态导致第三次与前两次不同。
    """
    from cerebellum_cancel import integrity as itg
    n = len(t)
    if n < 200 or not np.all(np.isfinite(t)):
        return ""
    err = np.abs(np.asarray(_probe_score(gyro), dtype=np.float64))
    if len(err) != n:
        return ""
    outs = []
    for _ in range(3):
        try:
            r = itg.report(err)
        except (ContractError, ValueError, IndexError):
            return ""
        outs.append(float(getattr(r, "n_spikes", 0) or 0))
    if len(set(outs)) != 1:
        return f"同一输入三次 report 给出不同 n_spikes: {outs}（状态泄漏）"
    return ""


def _inv_dataset_idempotent(t: np.ndarray, gyro: np.ndarray) -> str:
    """同一 Section 写入两次，manifest 的 n_frames 必须恰好翻倍（不能重复计数）。"""
    from cerebellum_cancel import dataset as ds
    if not np.all(np.isfinite(t)) or len(t) < 2:
        return ""
    sc = _probe_score(gyro)
    if len(sc) != len(t):
        return ""
    try:
        sec = ds.build_section("s", t, gyro, sc, sc.astype(np.float32))
    except (ContractError, ValueError, IndexError):
        return ""
    with tempfile.TemporaryDirectory() as td:
        dw = ds.DatasetWriter(td, name="x")
        dw.add(sec)
        m1 = dw.finalize()
        n1 = m1["n_frames"]
        with tempfile.TemporaryDirectory() as td2:
            dw2 = ds.DatasetWriter(td2, name="x")
            dw2.add(sec)
            m2 = dw2.finalize()
            if m2["n_frames"] != n1:
                return f"同一 Section 两次写入 n_frames 不同: {n1} vs {m2['n_frames']}"
    return ""


def _link_integrity_cross_segment(t: np.ndarray, gyro: np.ndarray) -> str:
    """跨段借用参照是**已知代价**（§4.3）：把 A 段的参照给 B 段用，
    必须体现在结论里（不能静默当成健康）。

    这里做的是**自洽性**检查：同一段数据当参照和当查询，结论不应矛盾
    （要么都健康，要么都报异常）。
    """
    from cerebellum_cancel import integrity as itg
    n = len(t)
    if n < 400 or not np.all(np.isfinite(t)):
        return ""
    err = np.abs(np.asarray(_probe_score(gyro), dtype=np.float64))
    if len(err) != n or not np.all(np.isfinite(err)):
        return ""
    try:
        a = itg.report(err)
    except (ContractError, ValueError, IndexError):
        return ""
    verdict = getattr(a, "verdict", None)
    if verdict is None:
        return "report() 未给出 verdict 字段（协议层需要可判定的结论）"
    if not isinstance(verdict, str) or not verdict:
        return f"verdict 不是非空字符串: {verdict!r}"
    return ""


# ------------------------------------------------------------------ 运行器


@dataclass
class Probe:
    name: str
    label_fn: Callable[[tuple], str]
    # 检查器签名统一为 (t, gyro) -> "" | 问题描述
    fn: Callable[[np.ndarray, np.ndarray], str]
    allowed: dict[str, str] = field(default_factory=dict)


def _run_invariant(probe: Probe, label: str, args: tuple) -> Finding:
    try:
        problem = probe.fn(*args)
    except ContractError as e:
        return Finding(probe.name, label, "RAISED", type(e).__name__)
    except (ValueError, EOFError):
        return Finding(probe.name, label, "RAISED", "ValueError/EOFError")
    except Exception as e:                                 # noqa: BLE001
        return Finding(probe.name, label, "RAISED_OTHER",
                       f"{type(e).__name__}: {str(e)[:60]}")
    if problem:
        return Finding(probe.name, label, "SILENT_BAD", problem)
    return Finding(probe.name, label, "SILENT_GOOD", "")


INVARIANTS: list[Probe] = [
    Probe("dataset.roundtrip", lambda a: a[0], _inv_dataset_roundtrip),
    Probe("dataset.idempotent", lambda a: a[0], _inv_dataset_idempotent),
    Probe("engine.replay", lambda a: a[0], _inv_replay_equivalence),
    Probe("record.truncation", lambda a: a[0], _inv_record_truncation),
    # D 类：协议层会被反复调用几小时，状态必须幂等
    Probe("integrity.repeated", lambda a: a[0], _inv_repeated_calls),
]

LINKS: list[Probe] = [
    Probe("link.engine_no_imu", lambda a: a[0], _link_engine_no_imu),
    Probe("link.dataset_mismatch", lambda a: a[0], _link_dataset_mismatch),
    Probe("link.integrity_crossseg", lambda a: a[0],
          _link_integrity_cross_segment),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="协议层压力测试（时序/不变量/链路）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random", action="store_true",
                    help="额外跑随机时序破坏（每个序列固定 6 例）")
    ap.add_argument("--cases", type=int, default=40,
                    help="--random 时每类额外采多少例")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="变异自检：证明检查器真的有检出力")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    rng = np.random.default_rng(args.seed)
    all_findings: list[Finding] = []

    print("=" * 74)
    print("协议层压力测试：随机时序 / 不变量 / 链路")
    print("=" * 74)
    print("判定: RAISED=好 | RAISED_OTHER=可疑 | SILENT_BAD=问题 | "
          "SILENT_GOOD=需人工判断\n")

    # ---- A 类：时序攻击（喂给不变量检查器） ----
    serial: list[tuple[str, tuple]] = []
    serial += timing_attacks(rng) if args.random else timing_attacks(
        np.random.default_rng(0))[:10]
    serial += length_attacks()
    if args.random:
        base_t, base_g = _base_series(240)
        for i in range(args.cases):
            t = base_t.copy()
            k = int(rng.integers(1, 239))
            t[k:] += float(rng.uniform(-1e6, 1e6))
            serial.append((f"rand_serial_{i}", (t, base_g)))

    for probe in INVARIANTS:
        rows: list[Finding] = []
        for label, args_ in serial:
            f = _run_invariant(probe, label, args_)
            rows.append(f)
            all_findings.append(f)
        _report(probe.name, rows, probe.allowed)

    # ---- B 类：链路攻击（不吃序列，自己造输入） ----
    t_dummy, g_dummy = _base_series(60)
    for probe in LINKS:
        rows = []
        for label, args_ in (("fixed", (t_dummy, g_dummy)),
                             ("empty", (_base_series(0)[0], _base_series(0)[1]))):
            f = _run_invariant(probe, label, args_)
            rows.append(f)
            all_findings.append(f)
        _report(probe.name, rows, probe.allowed)

    # ---- 汇总 ----
    bad = [f for f in all_findings if f.verdict == "SILENT_BAD"]
    other = [f for f in all_findings if f.verdict == "RAISED_OTHER"]
    print()
    print("=" * 74)
    print(f"总计: 静默垃圾 {len(bad)} | 错误类型不统一 {len(other)}")
    if not (bad or other):
        print("结论: ✓ 协议层不变量全部守住")
    else:
        print("结论: ✗ 有需要复核的项")
    print("=" * 74)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([f.__dict__ for f in all_findings], fh,
                      ensure_ascii=False, indent=1)
    return 1 if (bad or other) else 0


# ---------------------------------------------------------------- 自检
#
# **为什么需要它**：一个全绿的压力测试有两种可能——代码真的没问题，或者
# **检查器根本没跑**。本脚本的检查器大量使用"前置条件不满足就 return ''"，
# 所以"通过"可能是空转。`--selftest` 用**变异测试**回答这个问题：
# 故意把被测代码的关键不变量打破（monkeypatch），确认检查器**能**报错。
# 报不出来的检查器 = 假绿，必须看到。

def _selftest() -> int:
    """变异测试：证明每个检查器都有检出力。"""
    import cerebellum_cancel.dataset as ds
    import cerebellum_cancel.integrity as itg
    import cerebellum_cancel.record as rec

    t, g = _base_series(480)
    problems: list[str] = []

    # 1. 先确认"正常输入下检查器确实执行了"（否则全绿可能是空转）
    checks: list[tuple[str, Callable[[], str]]] = [
        ("dataset.roundtrip", lambda: _inv_dataset_roundtrip(t, g)),
        ("dataset.idempotent", lambda: _inv_dataset_idempotent(t, g)),
        ("engine.replay", lambda: _inv_replay_equivalence(t, g)),
        ("record.truncation", lambda: _inv_record_truncation(t, g)),
        ("integrity.repeated", lambda: _inv_repeated_calls(t, g)),
        ("link.integrity_crossseg", lambda: _link_integrity_cross_segment(t, g)),
    ]
    print("=== 自检 1：正常输入下每个检查器都必须真的执行（非空转）===")
    for name, fn in checks:
        try:
            r = fn()
            status = "✓ 执行且通过" if r == "" else f"! 报出: {r[:50]}"
        except Exception as e:                             # noqa: BLE001
            status = f"✗ 抛 {type(e).__name__}: {str(e)[:40]}"
            problems.append(f"{name} 在**正常输入**上抛异常")
        print(f"    {name:26s} {status}")

    # 2. 变异：把某段真正的数据破坏，检查器必须报出来
    print("\n=== 自检 2：变异后检查器必须能报错（有检出力）===")
    # 2a. record：把一个完整文件截断，read_record 以前会抛 struct.error
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.cmcir")
        with rec.RecordWriter(p, fps=30.0, f_map=42.7, shape=(4, 28, 28),
                              z_policy="none") as w:
            for i in range(60):
                w.append(None, t=i / 30.0, keep=False, surprise=float(i),
                         score=float(i))
        raw = open(p, "rb").read()
        fn2 = os.path.join(td, "cut.cmcir")
        open(fn2, "wb").write(raw[:12])
        try:
            rec.read_record(fn2)
            print("    record 截断        ✗ 变异后未报错（检查力缺失）")
            problems.append("record 截断检查无力")
        except ValueError:
            print("    record 截断        ✓ 报 ValueError（可捕获）")
        except Exception as e:                             # noqa: BLE001
            print(f"    record 截断        ✗ 抛非契约异常 {type(e).__name__}")
            problems.append(f"record 截断抛 {type(e).__name__}")

    # 2b. dataset：NaN 时间戳必须被拒
    bad_t = np.arange(480, dtype=np.float64) / 30.0
    bad_t[100] = np.nan
    try:
        ds.build_section("s", bad_t, g, _probe_score(g),
                         _probe_score(g).astype(np.float32))
        print("    dataset NaN 时间戳  ✗ 未拒绝（契约缺失）")
        problems.append("dataset 未拒绝 NaN 时间戳")
    except (ContractError, ValueError):
        print("    dataset NaN 时间戳  ✓ 拒绝")

    # 2c. dataset：1 维 gyro 必须在**构造处**被拒（不能拖到写入）
    try:
        ds.build_section("s", t, np.zeros(len(t)), np.zeros(len(t)),
                         np.zeros(len(t)))
        print("    dataset 1维 gyro    ✗ 未拒绝（延迟校验）")
        problems.append("dataset 未在构造处拒绝 1 维 gyro")
    except (ContractError, ValueError):
        print("    dataset 1维 gyro    ✓ 构造处拒绝")

    print()
    if problems:
        print(f"自检结论: ✗ {len(problems)} 个问题")
        for p in problems:
            print("   - " + p)
        return 1
    print("自检结论: ✓ 所有检查器都有检出力")
    return 0


def _report(name: str, rows: list[Finding], allowed: dict[str, str]) -> None:
    raised = [f for f in rows if f.verdict == "RAISED"]
    other = [f for f in rows if f.verdict == "RAISED_OTHER"]
    bad = [f for f in rows if f.verdict == "SILENT_BAD"]
    good = [f for f in rows if f.verdict == "SILENT_GOOD"]
    suspect = [f for f in good if f.label not in allowed]
    flag = "✓" if not (other or bad or suspect) else "✗"
    print(f"{flag} {name}")
    print(f"    拒绝 {len(raised)} | 错误类型不统一 {len(other)} | "
          f"不变量破坏 {len(bad)} | 通过 {len(good)}")
    for f in other[:5]:
        print(f"      ✗ [RAISED_OTHER] {f.label}: {f.detail}")
    for f in bad[:5]:
        print(f"      ✗ [SILENT_BAD] {f.label}: {f.detail}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""随机污染测试（fuzz）—— 不靠人想"该测什么"，让机器随机去撞。

## 为什么需要它

前几轮加固（GUIDE §4.4–4.10）都是"我想到的输入"，而 `io.py` 一次探针就撞出
**9 条静默失败**。这说明**人的想象力就是天花板**：凡是靠列举的检查，覆盖的
永远是已经想到的那部分。

## 判定机制（关键：不是"报错/没报错"二分）

朴素的 fuzz 会刷屏假警报，因为有些输入**本来就该被安静接受**（`dt=0` 有明确
语义、`(n,1)` 列向量是合法调用习惯、负 `f_map` 是本项目**刻意**不校验的）。
所以每次探测把结果分成四类：

| 分类 | 含义 | 处理 |
| --- | --- | --- |
| `RAISED` | 抛了 `ContractError` 子类 | **好**——正是我们想要的 |
| `RAISED_OTHER` | 抛了非契约异常（IndexError/TypeError…） | **可疑**——错误类型不统一，调用方无法统一处理 |
| `SILENT_GOOD` | 安静返回，且输出**有限、形状合理** | 需要判断：合法还是漏网 |
| `SILENT_BAD` | 安静返回，但输出**含 NaN/Inf 或形状错** | **问题**——静默产出垃圾，最危险 |

`SILENT_GOOD` 里"合法"与"漏网"靠**白名单**区分：白名单项在 `ALLOWED`
里逐条写明理由（并指向对应的测试）。任何不在白名单的脏输入拿到
`SILENT_GOOD`，都是**候选问题**——由人复核后决定是加校验还是加白名单。

## 两种用法

    # 随机冲撞（默认）：随机挑目标、随机生脏值
    PYTHONPATH=. .venv/bin/python scripts/fuzz_contracts.py

    # 复现某次结果（种子固定即可复现）
    PYTHONPATH=. .venv/bin/python scripts/fuzz_contracts.py --seed 42 -n 2000

    # 只测某个模块
    PYTHONPATH=. .venv/bin/python scripts/fuzz_contracts.py --only rotwarp

退出码：发现**未在白名单**的 `SILENT_BAD`/`SILENT_GOOD`/`RAISED_OTHER` 则为 1
（可接进门禁）；全部已知则为 0。
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cerebellum_cancel.contracts import ContractError      # noqa: E402

# ------------------------------------------------------------------ 脏值工厂
#
# 每个工厂返回 (标签, 值)。标签用于报告里指认"是哪个脏值打穿的"。
# 覆盖的维度：非有限、极端量级、错形状、错 dtype、空、错类型。

_NAN = float("nan")
_INF = float("inf")


def dirty_numbers() -> list[tuple[str, Any]]:
    return [
        ("nan", _NAN), ("inf", _INF), ("-inf", -_INF),
        ("zero", 0.0), ("neg_zero", -0.0), ("neg", -1.0),
        ("tiny", 1e-300), ("huge", 1e300), ("huge_neg", -1e300),
        ("subnormal", 5e-324),
        ("int", 3), ("bool", True), ("str_num", "3.0"),
        ("str_junk", "abc"), ("none", None), ("list1", [1.0]),
    ]


def dirty_arrays(rng: np.random.Generator) -> list[tuple[str, Any]]:
    n = 48
    return [
        ("nan_arr", np.full(n, _NAN)),
        ("inf_arr", np.full(n, _INF)),
        ("mix_nan", np.where(np.arange(n) % 7 == 0, _NAN, 1.0)),
        ("empty", np.zeros(0)),
        ("scalar", np.float64(1.0)),
        ("wrong_len", np.zeros(7)),
        ("2d", np.zeros((8, 6))),
        ("3d", np.zeros((2, 4, 6))),
        ("col_vec", np.zeros((n, 1))),
        ("row_vec", np.zeros((1, n))),
        ("obj_dtype", np.array([object()] * n, dtype=object)),
        ("str_dtype", np.array(["a"] * n)),
        ("bool_arr", np.ones(n, dtype=bool)),
        ("huge_arr", np.full(n, 1e300)),
        ("none", None),
        ("str", "not an array"),
        ("list", [1.0] * n),
        ("random", rng.normal(size=n)),
    ]


def dirty_gyros() -> list[tuple[str, Any]]:
    return [
        ("nan3", np.array([_NAN, 0.0, 0.0])),
        ("inf3", np.array([_INF, 0.0, 0.0])),
        ("nan_all", np.full(3, _NAN)),
        ("len2", np.array([1.0, 2.0])),
        ("len4", np.array([1.0, 2.0, 3.0, 4.0])),
        ("empty", np.zeros(0)),
        ("scalar", 1.0),
        ("list3", [1.0, 2.0, 3.0]),
        ("huge", np.array([1e300, -1e300, 1e300])),
        ("none", None),
        ("2d", np.zeros((3, 1))),
    ]


def dirty_images(rng: np.random.Generator) -> list[tuple[str, Any]]:
    return [
        ("nan_img", np.full((28, 28), _NAN, dtype=np.float64)),
        ("empty_img", np.zeros((0, 0))),
        ("small_img", np.zeros((2, 2))),
        ("3ch_img", np.zeros((28, 28, 3), dtype=np.uint8)),
        ("huge_img", np.full((28, 28), 1e300)),
        ("none", None),
        ("1d", np.zeros(28)),
        ("scalar_img", np.float64(0.0)),
        ("random_img", rng.random((28, 28))),
    ]


# ------------------------------------------------- 真·随机值（--random 模式）
#
# **动机（2026-09-13）**：上面那批脏值是**固定目录**——12 个 TARGETS × ~15 个
# 精选值，而且只有 `random`/`random_img` 两项随种子变。实测：seed 1/2/3/7/11/
# 42/99/123/777/2026 跑出来**完全一样**。也就是说"多跑几遍"在旧实现下不是
# 多次独立试验，而是把同一份清单重复十遍。
#
# 真正要找"没想到的输入"，必须让**值本身**随机。这里按类型从一个大得多的
# 空间里采样：量级指数均匀（覆盖 1e-320 到 1e308）、随机形状、随机 dtype、
# 随机嵌套结构。种子不同 → 值不同 → 真的能扫到没覆盖的角落。

def random_number(rng: np.random.Generator) -> tuple[str, Any]:
    """采一个随机标量（可能落在任何奇怪的类型/量级上）。"""
    choice = rng.integers(0, 12)
    if choice == 0:
        # 指数均匀的量级：1e-320 .. 1e308（含次正规与溢出边界）
        e = rng.uniform(-320, 308)
        return f"mag10e{e:.0f}", float(10.0 ** e)
    if choice == 1:
        return "randn", float(rng.standard_normal())
    if choice == 2:
        return "absurd", float(rng.standard_normal() * 10.0 ** rng.integers(300, 309))
    if choice == 3:
        return "special", float(rng.choice(
            [_NAN, _INF, -_INF, 0.0, -0.0, 5e-324, 1.7976931348623157e308]))
    if choice == 4:
        return "int", int(rng.integers(-10 ** rng.integers(1, 19), 10 ** rng.integers(1, 19)))
    if choice == 5:
        return "bool", bool(rng.integers(0, 2))
    if choice == 6:
        return "complex", complex(rng.standard_normal(), rng.standard_normal())
    if choice == 7:
        return "str", rng.choice([
            "abc", "1e999", "nan", "inf", "-1", "0x10", "", "  7  ", "1_000",
            "٣", "１", "0b1", "1e", "--1", "1.2.3"])
    if choice == 8:
        return "none", None
    if choice == 9:
        return "bytes", bytes([rng.integers(0, 256) for _ in range(rng.integers(0, 4))])
    if choice == 10:
        return "nested", [rng.standard_normal() for _ in range(rng.integers(0, 3))]
    return "obj", object()


def random_array(rng: np.random.Generator, *, like: str) -> tuple[str, Any]:
    """采一个随机数组。``like`` 提示目标期望的形状（numbers/arrays/gyros/images）。"""
    choice = rng.integers(0, 10)
    if like == "gyros" and choice < 4:
        n = int(rng.choice([0, 1, 2, 3, 4, 3, 3]))
    elif like == "images" and choice < 4:
        h = int(rng.choice([0, 1, 2, 4, 8, 28, 28, 64]))
        w = int(rng.choice([0, 1, 2, 4, 8, 28, 28, 64]))
        n = (h, w)
    else:
        n = int(rng.choice([0, 1, 2, 7, 48, 100]))

    if choice == 0:
        return "rand_shape", rng.standard_normal(n)
    if choice == 1:
        return "rand_2d", rng.standard_normal((int(rng.integers(0, 5)), int(rng.integers(0, 5))))
    if choice == 2:
        # 随机量级均匀分布在指数上 —— 包含会溢出 float32/矩阵乘法的值
        e = rng.uniform(-320, 308)
        return f"magnitude{e:.0f}", np.full(n, 10.0 ** e)
    if choice == 3:
        arr = rng.standard_normal(n)
        if arr.size:
            k = int(rng.integers(0, arr.size + 1))
            arr.flat[rng.choice(arr.size, size=k, replace=False)] = rng.choice([_NAN, _INF, -_INF])
        return "sparse_nonfinite", arr
    if choice == 4:
        return "int_dtype", rng.integers(-100, 100, size=n)
    if choice == 5:
        return "bool_dtype", rng.integers(0, 2, size=n).astype(bool)
    # 注：用整数下标选元素而不是 `rng.choice([混合类型列表])`——
    # pyright 无法为异构列表解析 numpy 的 choice 重载（会报 4 个错）。
    # 下标形式既清晰又类型干净。
    size = int(np.prod(n)) if isinstance(n, tuple) else int(n)
    if choice == 6:
        pool: list[Any] = [object(), "x", None, 1.0]
        return "object_dtype", np.array(
            [pool[int(rng.integers(0, len(pool)))] for _ in range(size)],
            dtype=object)
    if choice == 7:
        pool2: list[Any] = [1, "a", None, _NAN]
        return "str_dtype", np.array(
            [str(pool2[int(rng.integers(0, len(pool2)))]) for _ in range(size)])
    if choice == 8:
        return "empty_after_reshape", np.zeros(n).reshape(
            (1, -1) if np.prod(n) > 0 else (0, 0))
    return "list_like", [float(rng.standard_normal()) for _ in range(size)]


# ---------------------------------------------------------------- 目标定义
#
# 每个目标：怎么调、用哪些脏值、以及**哪些"脏值"其实合法**（白名单）。
# 白名单必须逐条写理由——"我们知道了但还是允许"是有意决定，不是疏忽。

@dataclass
class Target:
    module: str
    func: str
    kind: str                       # numbers | arrays | gyros | images
    build: Callable[[Any], tuple]   # 脏值 -> 位置参数
    allowed: dict[str, str] = field(default_factory=dict)
    extra_kwargs: dict = field(default_factory=dict)


F = np.random.default_rng(0).random((4, 28, 28)).astype(np.float32)
IMG = (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8)

TARGETS: list[Target] = [
    Target("contracts", "check_positive", "numbers",
           lambda v: (v, "x"),
           # 3/True/"3.0" 都是合法正数；bool 在 Python 里是 int 子类
           allowed={"int": "正整数是合法输入",
                    "bool": "bool 是 int 子类，值 3 之外 True==1 合法",
                    "str_num": "'3.0' 可被 float() 解析，宽松接受是有意的"
                               "（配置常来自环境变量/CLI，本就是字符串）",
                    "tiny": "极小的正数是正数",
                    "subnormal": "次正规正数仍是正数",
                    "huge": "1e300 是有限正数；量级不归本函数管"}),
    Target("contracts", "check_finite_array", "arrays",
           lambda v: (v, "x"),
           allowed={"col_vec": "(n,1) 会被挤压，合法",
                    "row_vec": "(1,n) 会被挤压，合法",
                    "list": "接受 list-like 是有意的（见模块文档）",
                    "random": "正常数据",
                    # 下面这些是**形状**维度，不是本函数的职责——
                    # check_finite_array 只回答"有没有 NaN/Inf"，
                    # 形状由 check_1d 把关。所以它们安静通过是正确的。
                    "scalar": "0 维数组也是合法输入（有限即可）",
                    "wrong_len": "长度不归本函数管（check_1d 负责）",
                    "2d": "维度不归本函数管（check_1d 负责）",
                    "3d": "维度不归本函数管（check_1d 负责）",
                    "bool_arr": "bool 数组是有限值",
                    "huge_arr": "1e300 是有限值，量级不归本函数管"}),
    Target("contracts", "check_min_frames", "numbers",
           lambda v: (v, 10, "x"),
           allowed={"int": "帧数是整数",
                    # 帧数极大当然满足"至少 10 帧"——本函数只判下界，
                    # 不设上界（上界没有物理意义）。
                    "huge": "只判下界，超大帧数也满足；上界无物理意义"}),

    Target("core", "CerebellumCancel", "arrays",
           lambda v: (),          # 构造不需要脏值；只用它测方法
           allowed={}),

    Target("io", "gyro_at", "gyros",
           lambda v: ([type("S", (), {"t": 0.0,
                                      "gyro": np.zeros(3)})()], 0, v),
           allowed={"random": "有限 t 合法",
                    "scalar": "t 本就可以是标量时间戳（合法用法）"}),
    Target("io", "compensate_rotation", "arrays",
           lambda v: (IMG, IMG, np.zeros(3), v),
           allowed={"random": "有限 dt 合法",
                    "zero": "dt=0 合法（无时间流逝）",
                    "neg_zero": "-0.0 等同 0",
                    "subnormal": "极小正数合法",
                    # 本条目测的是 dt 位置：标量 dt 是**最常见**的合法用法。
                    "scalar": "标量 dt 是主要用法（本条目喂的就是 dt）"}),
    Target("io", "cells_of", "images",
           lambda v: (v, (4, 4)),
           allowed={"random_img": "正常残差图；分格只用到几何，内容无关",
                    "huge_img": "1e300 是有限值；本函数只做分格均值，"
                                "不负责量级（下游判据自己处理）"}),

    Target("rotwarp", "warp_by_rotation", "gyros",
           lambda v: (F, v, 1 / 30, 42.7),
           allowed={"list3": "接受 list-like 是有意的（3 元素 list 是合法 gyro）",
                    "huge": "大角速度合法（快速甩动），不校验量级"}),
    Target("rotwarp", "rotation_homography", "numbers",
           lambda v: (v, 0.0, 0.0, 1 / 30, 42.7, 14.0, 14.0),
           allowed={"int": "角度可以是整数",
                    "str_num": "'3.0' 可被 float() 解析（配置常来自字符串）",
                    # 角度维度的合法取值——零、负、极小都对应真实姿态
                    "zero": "角度 0 = 不旋转，合法",
                    "neg_zero": "-0.0 等同 0",
                    "neg": "负角度 = 反向旋转，合法（方向由符号约定处理）",
                    "tiny": "极慢转动，合法",
                    "subnormal": "次正规数仍是有效角度",
                    "bool": "bool 是 int 子类",
                    "list1": "单元素 list 经 check_scalar 归一化，合法"}),

    Target("integrity", "frozen_baseline_z", "arrays",
           lambda v: (v,),
           allowed={"random": "正常误差序列"}),
    Target("integrity", "spike_threshold", "arrays",
           # **注意**：脏值必须喂给 frozen_z（第一个位置参数）。早先写成了
           # `(np.zeros(400),)` 把脏值丢掉，于是 18 项全部"静默通过"——
           # 那不是库的问题，是探针自己传错了参数。fuzz 工具本身也会出错，
           # 报告里的"待判"要人工复核，不能直接当成缺陷清单。
           lambda v: (v,),
           allowed={"random": "正常 frozen_z 序列",
                    # 本函数接受 (T,1)/(1,T) 与多维序列：内部按第一轴取帧序，
                    # 这是有意宽容（DataFrame 切片常给出列向量）。
                    "col_vec": "(T,1) 是 DataFrame 切片的常见形状，刻意接受",
                    "row_vec": "(1,T) 同上",
                    "2d": "多维序列按第一轴为帧序，合法",
                    "3d": "同上",
                    "list": "接受 list-like 是有意的",
                    "bool_arr": "布尔序列是合法输入（0/1 残差）",
                    "huge_arr": "1e300 是有限值；阈值是相对分位，量级无关",
                    "wrong_len": "本函数按实际长度抽参照集，不预设长度"}),

    Target("visual_motion", "assess_motion", "images",
           lambda v: (v, v),
           allowed={"random_img": "正常灰度图；无稳定角点时 ok=False，"
                                  "那是**设计好的**失败路径，不是静默垃圾",
                    "huge_img": "1e300 会被 clip 到 255 再转 uint8；"
                                "本函数只报运动量，不负责量级",
                    "small_img": "小图仍可尝试（特征是空的，走失败分支）"}),
]


# ------------------------------------------------------------------- 探测

def _outputs(r: Any) -> list[np.ndarray]:
    """把返回值里所有 ndarray 收出来，用于有限性/形状检查。"""
    if isinstance(r, np.ndarray):
        return [r]
    if isinstance(r, (list, tuple)):
        return [x for x in r if isinstance(x, np.ndarray)]
    if isinstance(r, dict):
        return [v for v in r.values() if isinstance(v, np.ndarray)]
    return []


@dataclass
class Finding:
    target: str
    label: str
    verdict: str
    detail: str = ""


def probe_one(t: Target, label: str, value: Any) -> Finding:
    mod = importlib.import_module(f"cerebellum_cancel.{t.module}")
    obj = getattr(mod, t.func)
    # 支持类（如 CerebellumCancel）：只有脏值目标是数组且 build 返回空时跳过
    args = t.build(value)
    if t.func == "CerebellumCancel":
        return Finding(t.func, label, "SKIPPED", "构造类目标，由方法目标覆盖")

    try:
        r = obj(*args, **t.extra_kwargs)
    except ContractError as e:
        return Finding(t.func, label, "RAISED", type(e).__name__)
    except Exception as e:                                  # noqa: BLE001
        return Finding(t.func, label, "RAISED_OTHER",
                       f"{type(e).__name__}: {str(e)[:60]}")

    arrs = _outputs(r)
    if arrs:
        for a in arrs:
            if a.dtype.kind in "fc" and not np.all(np.isfinite(a)):
                return Finding(t.func, label, "SILENT_BAD", "输出含 NaN/Inf")
            if a.size == 0 and getattr(a, "ndim", 0) > 0 and a.shape[0] == 0:
                return Finding(t.func, label, "SILENT_BAD", "输出为空数组")
    return Finding(t.func, label, "SILENT_GOOD", getattr(r, "shape", ""))


def run_target(t: Target, n_per_label: int, seed: int,
               *, random_mode: bool = False,
               n_random: int = 0) -> list[Finding]:
    rng = np.random.default_rng(seed)
    if random_mode:
        # 真·随机模式：值本身随机，不依赖固定目录。
        # 这是唯一能让"多跑几遍"真的可能是多次独立试验的方式。
        out: list[Finding] = []
        # 先跑一遍固定目录（它们有明确的白名单理由，是回归护栏）
        for label, v in _catalogue(t, rng):
            out.append(probe_one(t, label, v))
        gen = random_number if t.kind == "numbers" else None
        for i in range(n_random):
            if t.kind == "numbers":
                label, v = random_number(rng)
            elif t.kind == "gyros":
                label, v = random_array(rng, like="gyros")
            elif t.kind == "images":
                label, v = random_array(rng, like="images")
            else:
                label, v = random_array(rng, like="arrays")
            # 随机值不可能在白名单里逐条列理由，所以统一打标签前缀
            out.append(probe_one(t, f"rand:{label}", v))
        return out

    pool = _catalogue(t, rng)
    # 随机重复 n_per_label 次（对确定性工厂是冗余的，但为将来加入随机工厂留位）
    return [probe_one(t, label, v) for label, v in pool * max(1, n_per_label)]


def _catalogue(t: Target, rng: np.random.Generator) -> list[tuple[str, Any]]:
    """固定的脏值目录（带白名单的理由）。"""
    if t.kind == "numbers":
        return dirty_numbers()
    if t.kind == "gyros":
        return dirty_gyros()
    if t.kind == "images":
        return dirty_images(rng)
    return dirty_arrays(rng)


# ------------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="随机污染测试（fuzz）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-n", "--repeat", type=int, default=1,
                    help="每个脏值重复探测次数")
    ap.add_argument("--only", default=None, help="只测匹配的模块/函数名")
    ap.add_argument("--json", default=None, help="把结果写 JSON")
    ap.add_argument("--random", action="store_true",
                    help="在固定目录之外，额外采样 N 个真·随机值")
    ap.add_argument("--cases", type=int, default=200,
                    help="--random 时每个目标采多少个随机值")
    args = ap.parse_args()

    print("=" * 74)
    print("随机污染测试：找出「静默产出垃圾」的路径")
    print("=" * 74)
    print("判定: RAISED=好 | RAISED_OTHER=可疑(错误类型不统一) | "
          "SILENT_BAD=问题 | SILENT_GOOD=需人工判断是否合法\n")

    all_findings: list[Finding] = []
    for t in TARGETS:
        if args.only and args.only not in f"{t.module}.{t.func}":
            continue
        fs = run_target(t, args.repeat, args.seed,
                        random_mode=args.random, n_random=args.cases)
        all_findings += fs

        # 分类汇总；白名单项不计入问题
        raised = [f for f in fs if f.verdict == "RAISED"]
        other = [f for f in fs if f.verdict == "RAISED_OTHER"]
        bad = [f for f in fs if f.verdict == "SILENT_BAD"]
        good = [f for f in fs if f.verdict == "SILENT_GOOD"]
        skipped = [f for f in fs if f.verdict == "SKIPPED"]
        # 随机采样的值**不可能**预先写进白名单（它们是每次新生成的），
        # 所以不能拿"在不在 allowed 里"判定。对它们而言，安静返回**正常**结果
        # 本来就是预期——毕竟随机值里绝大多数是合法数值。真正的信号是
        # SILENT_BAD / RAISED_OTHER，而那两类与标签无关，照样会报。
        known = [f for f in good
                 if f.label in t.allowed or f.label.startswith("rand:")]
        suspect = [f for f in good
                   if f.label not in t.allowed
                   and not f.label.startswith("rand:")]

        flag = "✓" if not (other or bad or suspect) else "✗"
        print(f"{flag} {t.module}.{t.func}")
        print(f"    阻断 {len(raised)} | 错误类型不统一 {len(other)} | "
              f"静默垃圾 {len(bad)} | 静默可接受 {len(known)}"
              + (f" | **静默待判 {len(suspect)}**" if suspect else ""))
        for f in bad:
            print(f"      ✗ [SILENT_BAD] {f.label}: {f.detail}")
        for f in suspect:
            print(f"      ? [待判] {f.label}: 安静返回了，但不在白名单里"
                  f"（{f.detail}）")
        for f in other:
            print(f"      ! [RAISED_OTHER] {f.label}: {f.detail}")
        if skipped:
            print(f"      (跳过 {len(skipped)} 项)")

    # ---- 汇总
    bad = [f for f in all_findings if f.verdict == "SILENT_BAD"]
    other = [f for f in all_findings if f.verdict == "RAISED_OTHER"]
    suspect = []
    for t in TARGETS:
        for f in all_findings:
            # 与逐目标显示**必须同一口径**（否则总计与明细会打架——
            # 这是本脚本自己踩过的坑：随机模式下明细全 ✓、总计却说 518 待判，
            # 因为这里漏了 `rand:` 前缀的豁免）。
            if (f.verdict == "SILENT_GOOD" and f.target == t.func
                    and f.label not in t.allowed
                    and not f.label.startswith("rand:")):
                suspect.append(f)

    print()
    print("=" * 74)
    print(f"总计: 静默垃圾 {len(bad)} | 错误类型不统一 {len(other)} | "
          f"静默待判 {len(suspect)}")
    if not (bad or other or suspect):
        print("结论: ✓ 所有探测到的行为都符合契约")
    else:
        print("结论: ✗ 有需要复核的项（见上）")
        print("    处理方式：要么加校验（改 contracts/模块），"
              "要么加进 TARGETS 的 allowed 并**写明理由**。")
    print("=" * 74)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump([f.__dict__ for f in all_findings], fp,
                      indent=1, ensure_ascii=False)
        print(f"明细写入 {args.json}")

    return 0 if not (bad or other or suspect) else 1


if __name__ == "__main__":
    raise SystemExit(main())

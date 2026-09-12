"""输入契约与校验——工业化稳定性的单一事实来源。

## 为什么单独一个模块

工业化模块的核心要求不是"功能更多"，而是**失败模式可预测**。具体三条：

1. **非法输入必须报错，不能静默产出错误结果。** 静默失败比崩溃危险得多——它
   会把错误传播到下游（存进数据集、写进日志、触发错误决策），而且事后难以定位。
2. **失败必须是同一种失败。** 同一个非法输入在任何调用路径上应产生同类异常，
   便于调用方统一处理（而不是这里 ValueError、那里 IndexError）。
3. **边界条件要显式，不要靠数值巧合。** 例如"样本太少不足以做统计判断"应当是
   一条明确的规则，而不是"算出来恰好是 0 个报警"。

## 实测背景（这些校验不是假想的）

对一个探针脚本逐项测过现有行为，发现以下**静默通过**的情况：

| 输入 | 修复前 | 修复后 |
| --- | --- | --- |
| `CerebellumCancel.process` 2D 数组（应为 1D） | 静默返回 dict | `ShapeError` |
| `process` 含 NaN | 静默返回并**污染运行统计** | `NonFiniteError` |
| `process` `dt=0` | 静默返回 | 允许（有明确语义，见下） |
| `AdaptiveTrigger.update` `fps=0` | 静默返回 bool | `ConfigError` |
| `update` `score=inf` | 静默返回 bool | `NonFiniteError` |
| `integrity.report` 含 NaN | 静默返回报告 | `NonFiniteError` |
| `integrity.report` 1 帧 | 静默返回报告 | `InsufficientDataError` |

`dt=0` 是**刻意允许**的：`CerebellumCancel._features` 对它已有明确定义（只留偏置
项），且上游在首帧或时间戳重合时会合法地产生它。

## 异常层次

所有异常继承自 `ContractError`，调用方可以只 catch 这一个基类：

    ContractError
    ├── ShapeError            形状/维度不符契约
    ├── NonFiniteError        含 NaN/Inf（会污染在线统计）
    ├── ConfigError           配置参数非法
    └── InsufficientDataError 样本太少，无法给出可靠结论
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 这些检查器**有意**接受 array-like（list/tuple/标量）：它们位于公共入口，
# 调用方不该被迫先手动 `np.asarray`。用 `Any` 而非 numpy.typing 的
# `ArrayLike`，避免把类型检查与 numpy 版本绑定。
ArrayLike = Any


class ContractError(Exception):
    """所有输入契约违规的基类。调用方可只 catch 这一个。"""


class ShapeError(ContractError, ValueError):
    """形状/维度不符契约。

    同时继承 ``ValueError``，使已有的 ``except ValueError`` 调用方不受影响
    ——这是刻意的向后兼容选择。
    """


class NonFiniteError(ContractError, ValueError):
    """输入含 NaN 或 Inf。

    为什么单独成类而不是混进 ``ShapeError``：在线统计（mu/var/neg/w_fast）
    一旦被 NaN 污染，**后续所有帧的判断都会失效且不报错**——这是本项目实测
    过的最隐蔽的失效模式，值得一个可单独捕获的类型。
    """


class ConfigError(ContractError, ValueError):
    """配置参数非法（如 fps<=0、阈值关系不自洽）。"""


class InsufficientDataError(ContractError, ValueError):
    """样本量不足以给出可靠结论。

    典型场景：完整性巡检需要足够帧数去标定参照集分位；帧数太少时"算出来是
    0 个异常"没有统计意义，必须拒绝出结论而不是报"正常"。
    """


def check_finite_array(x: ArrayLike, name: str,
                       allow_empty: bool = False) -> np.ndarray:
    """校验数组非空且全为有限值。

    Args:
        x: 待检查数组（接受 array-like：list/tuple 会被转成 ``float64``）。
        name: 用于错误信息的字段名（便于定位调用点）。
        allow_empty: 是否允许零长度（默认不允许）。
    Returns:
        转换后的 ``float64`` 数组（已强制转换，保证整数输入不会在后续
        浮点运算中被静默截断）。
    Raises:
        InsufficientDataError: 空数组且 ``allow_empty=False``。
        NonFiniteError: 含 NaN/Inf。
    """
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0 and not allow_empty:
        raise InsufficientDataError(f"{name}: 空数组，无法处理")
    if not np.all(np.isfinite(a)):
        n_bad = int(np.count_nonzero(~np.isfinite(a)))
        raise NonFiniteError(
            f"{name}: 含 {n_bad}/{a.size} 个 NaN 或 Inf。"
            "在线统计一旦被污染，后续所有帧的判断都会静默失效——"
            "请在调用前清洗输入，或显式声明该段数据不可用。")
    return a


def check_1d(x: ArrayLike, name: str, expected_len: int | None = None,
             allow_empty: bool = False) -> np.ndarray:
    """校验是 1 维（或可安全展平的标量数组），必要时校验长度。

    ``(n, 1)`` 与 ``(1, n)`` 会被展平——这是有意的宽容：调用方常从
    DataFrame 或切片拿到列向量，强制它 reshape 没有信息增益。
    真正的多维（如 cell 网格 ``(6, 8)`` 被误传进来）会被拒绝，因为静默展平
    会得到**看似正常但语义错误**的结果。
    """
    a = check_finite_array(x, name, allow_empty=allow_empty)
    if a.ndim > 1:
        squeezed = a.squeeze()
        if squeezed.ndim > 1:
            raise ShapeError(
                f"{name}: 期望 1 维（或 (n,1)/(1,n)），实得 shape={a.shape}。"
                "多维输入静默展平会得到语义错误但数值正常的结果，故拒绝。")
        a = squeezed
    if expected_len is not None and len(a) != expected_len:
        raise ShapeError(
            f"{name}: 长度不符，期望 {expected_len}，实得 {len(a)}")
    return a


def check_gyro(gyro: ArrayLike, name: str = "gyro") -> np.ndarray:
    """校验陀螺读数：形状 ``(3,)``、有限、单位是 deg/s。

    只校验形状与有限性——**不校验量级**：真实的陀螺读数在静止时可能很小
    （亚度/秒），而快速甩动可上百；用阈值卡量级会在合法数据上误报。
    """
    g = check_finite_array(gyro, name)
    if g.ndim != 1 or g.size != 3:
        raise ShapeError(
            f"{name}: 期望 3 维角速度 [x,y,z]（deg/s），实得 shape={g.shape}")
    return g


def check_positive(x: float, name: str, allow_zero: bool = False) -> float:
    """校验标量为正（或非负）有限值。"""
    v = float(x)
    if not np.isfinite(v):
        raise NonFiniteError(f"{name}: 必须是有限标量，实得 {x!r}")
    if v < 0 or (v == 0 and not allow_zero):
        rel = "非负" if allow_zero else "正"
        raise ConfigError(f"{name}: 必须是{rel}数，实得 {v}")
    return v


def check_min_frames(n: int, minimum: int, what: str) -> None:
    """校验帧数满足某项分析的最低要求。"""
    if n < minimum:
        raise InsufficientDataError(
            f"{what}: 需要至少 {minimum} 帧，实得 {n}。"
            "样本不足时'未检出异常'没有统计意义，故拒绝出结论。")

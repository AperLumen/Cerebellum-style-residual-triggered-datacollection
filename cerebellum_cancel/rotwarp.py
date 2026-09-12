"""完整旋转 warp：用真实旋转矩阵（含 roll）替代 tan 近似平移。

由来（真机 IMU 实验的实测发现）。在 Galaxy S21 真 gyro 数据上（FOV≈36°、
gyro p95 [pitch 37, yaw 103, roll 47] deg/s，标定逐帧相关 r=+0.958），逐像素
对照三种预测：

| 预测器 | 残差比（<1 = 有用） |
| ------ | ------------------- |
| naive（不补偿） | 1.000 |
| 管线现用（gyro→dx,dy 的 tan 近似平移） | 1.433 |
| 完整旋转矩阵（含 roll，homography） | **0.614** |
| 完整旋转矩阵（不含 roll） | 0.641 |

即：**同一份真 gyro，换一种 warp 实现，结论从"帮倒忙"翻转为"降低 39% 残差"。**
特征图上同口径复核：0.984 vs 0.887（最强旋转帧）/ 1.066 vs 0.958（平静帧）。

为什么现有实现会输（两条独立原因）：

1. **roll 未建模**。`CancelConfig` 只把 gyro[0]/gyro[1] 转成 dx/dy，roll 被
   丢弃。本段 roll p95 47 deg/s，在 28 格网格边缘造成约 1 格位移——丢它是
   纯损失。全旋转模型（``rot3``）比只算 pitch+yaw 的（``rot2``）再省 4%。
2. **平移近似只在中心附近成立，且与旋转混叠**。内容位移并非常数：绕光轴
   旋转给出**线性变化**的位移场、绕横轴给出**二次**场。用常数 (dx,dy) 去近似
   它们，在视野大、角速度高时误差正比于离中心的距离——实测该误差项盖过了
   补偿收益（ratio 1.43）。

实现。用真实的小角度旋转矩阵构造单应：

    R = exp([ω]×)  ,  ω = gyro(rad/frame) 在相机坐标系
    H = K R K⁻¹   ,  K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]

对特征图逐通道做 ``cv2.warpPerspective``（逐通道是因为 OpenCV 把多通道数组的
第一维当空间维）。``f`` 是**特征图像素**焦距，即管线里的 ``f_map``。

成本：warpPerspective 比两次 roll+bllend 贵，但仍是 O(像素数) 的内存操作、
无设备依赖；实测见 ``runs/real_imu_eval.json``。

诚实边界：这是**单段、单设备**的结果（17.9 s、500 帧、Galaxy S21、室内手持）。
它推翻的是"真 IMU 在真实数据上不赚"这个**归因**（此前归给副本质量），
但样本量小，且 Open-AoE 上的负结果（替身 gyro）并未因此自动翻案——那条
链路的失效另有来源（SLAM 副本退化）。两条结论要在文档里分开记。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .contracts import (ConfigError, NonFiniteError, ShapeError,
                        check_scalar,
                        check_finite_array)

# warp_by_rotation **有意**接受 array-like 的 gyro（list/tuple/ndarray）——
# 它是公共入口，早期调用点就直接传 list。与 contracts 模块同一处理方式：
# 注解用 Any，避免把类型检查绑到 numpy 版本。
ArrayLike = Any


def rotation_matrix_zyx(pitch_deg: float, yaw_deg: float, roll_deg: float,
                        dt: float, yaw_sign: float = 1.0,
                        pitch_sign: float = 1.0,
                        roll_sign: float = 1.0) -> np.ndarray:
    """一帧内累积旋转（相机坐标系）的矩阵指数，大角度也稳定。

    参数是**角速度**（deg/s），乘 dt 得到本帧转角。

    ``yaw_sign`` / ``pitch_sign`` / ``roll_sign`` 把存储通道号映射到旋转轴。
    **两个数据源的约定不同，必须显式指定**（实测，见
    ``runs/rotwarp_aoe.json`` 与 MEMORY_FORMAT §8.7）：

    旋转轴 = ``[pitch_sign·pitch, yaw_sign·yaw, roll_sign·roll]``。

    - **真机 IMU**（`data/real_imu`）：``(+1, +1, +1)``——即存储值可直接用，
      无需任何翻转（500 帧符号扫描的全局最优）。
    - **AoE**（`data/aoe_*`）：``(+1, -1, -1)``。注意这与真机段不同，根因是
      两个 prep 脚本的标定目标相反：`prep_aoe.calibrate_signs` 强制
      ``corr(-f·tan(g), dx) > 0``（隐含把 pitch/yaw 都翻过一次），而
      `prep_real_imu.fit_calibration` 只取 ``|corr|`` 最大、不约束符号。

    不过 AoE 的符号差异在本项目里**没有实际后果**：该数据的运动量太小
    （见 DATASET_SIGNS 的注释），任何符号组合下 rotation 都到不了 1.0 以下。
    """
    w = np.radians(np.array([pitch_sign * pitch_deg, yaw_sign * yaw_deg,
                             roll_sign * roll_deg], dtype=np.float64)) * dt
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3, dtype=np.float64)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def rotation_homography(pitch_deg: float, yaw_deg: float, roll_deg: float,
                        dt: float, f: float, cx: float, cy: float,
                        yaw_sign: float = 1.0, pitch_sign: float = 1.0,
                        roll_sign: float = 1.0) -> np.ndarray:
    """内容位移的单应矩阵：``H = K R K⁻¹``（特征图像素坐标）。

    ``R`` 把相机坐标系从 t-1 转到 t；把点 p 的像素坐标同质化后用
    ``K R K⁻¹`` 映射，即得到"上一帧内容在这一帧应出现的位置"。
    符号约定见 ``rotation_matrix_zyx``。
    """
    # **fuzz 发现（2026-09-12，第二轮）**：下面 `np.radians(pitch_deg)` 这类
    # 算术直接作用在入参上。若传入字符串，Python 抛的是
    # `TypeError: can't multiply sequence by non-int`；传入 None 抛
    # `unsupported operand type(s) for *`。两者都不是 ContractError，
    # 调用方无法用统一方式处理。先经 check_scalar 归一化（它内部已包裹
    # 转换异常），后续算术面对的一定是 float。
    _p = check_scalar(pitch_deg, "pitch_deg", "rotation_homography")
    _y = check_scalar(yaw_deg, "yaw_deg", "rotation_homography")
    _r = check_scalar(roll_deg, "roll_deg", "rotation_homography")
    _dt = check_scalar(dt, "dt", "rotation_homography")
    _f = check_scalar(f, "f", "rotation_homography")
    _cx = check_scalar(cx, "cx", "rotation_homography")
    _cy = check_scalar(cy, "cy", "rotation_homography")
    R = rotation_matrix_zyx(_p, _y, _r, _dt,
                            yaw_sign=yaw_sign, pitch_sign=pitch_sign,
                            roll_sign=roll_sign)
    K = np.array([[_f, 0.0, _cx], [0.0, _f, _cy], [0.0, 0.0, 1.0]])
    H = (K @ R @ np.linalg.inv(K)).astype(np.float32)
    # **fuzz 发现（2026-09-12）**：本函数此前对 nan / inf / 1e300 会**静默**
    # 返回一个全 NaN 的单应矩阵——而它是 warp_by_rotation 的底层依赖，
    # 上游只会在残差里看到 NaN，完全定位不到这里。
    # 注意：**光查输入有限是不够的**——1e300 是有限值，但在
    # `np.radians(x)*dt` 与后续矩阵乘法里会溢出。所以这里查的是**结果**。
    if not np.all(np.isfinite(H)):
        for nm, v in (("pitch_deg", _p), ("yaw_deg", _y),
                      ("roll_deg", _r), ("dt", _dt), ("f", _f),
                      ("cx", _cx), ("cy", _cy)):
            # 报出是哪个参数导致溢出，而不是笼统说"结果非有限"
            if not np.isfinite(float(v)) or abs(float(v)) > 1e8:
                raise NonFiniteError(
                    f"rotation_homography: 参数 {nm}={v!r} 会让单应矩阵溢出。"
                    "（量级过大，不是 NaN 也会在计算中溢出——"
                    "实测 1e300 即如此。）")
        raise NonFiniteError(
            "rotation_homography: 单应矩阵含 NaN/Inf，参数组合会导致退化。"
            f"实测输入 pitch={pitch_deg} yaw={yaw_deg} roll={roll_deg} "
            f"dt={dt} f={f}")
    return H


def warp_by_rotation(F_prev: np.ndarray, gyro: ArrayLike, dt: float,
                     f_map: float, use_roll: bool = True,
                     yaw_sign: float = 1.0, pitch_sign: float = 1.0,
                     roll_sign: float = 1.0,
                     border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """按旋转预测 F_t：对 (C,H,W) 特征图逐通道做单应 warp。

    ``gyro`` 顺序与项目一致：[pitch, yaw, roll] deg/s（见 io.load_imu_csv）。
    ``use_roll=False`` 时丢弃 roll——用于消融（实测 roll 贡献约 4%）。
    ``yaw_sign`` / ``pitch_sign`` / ``roll_sign`` 见 ``rotation_matrix_zyx``。

    **必须带 `WARP_INVERSE_MAP`**（本函数历史上漏过一次，代价很大）：
    `cv2.warpPerspective` 的默认语义是 ``dst(x) = src(H⁻¹ x)``，而我们要的是
    ``dst(x) = src(H x)``——即"H 把上一帧的点映射到这一帧的位置"。
    不加这个 flag 会把预测**反向** warp，效果从降残差变成增残差，且用点测试
    才能发现（残差数值上看不出方向错，只会看到"变差了"）。
    这个坑在这条线的开发中连续误导了两轮排查。

    Raises:
        NonFiniteError: ``F_prev`` 或 ``gyro`` 含 NaN/Inf，或 ``dt`` 非有限。
            **这条是本函数最危险的静默失败**：NaN 会经 warp 原样铺满输出，
            此后所有残差统计都是 NaN，而调用方只看到"数值不对"，不知道根源
            在这里。历史上前向路径有同样的问题（见 contracts 模块）。
        ShapeError: ``F_prev`` 不是 3 维 ``(C,H,W)``，或 ``gyro`` 不是 3 维。
        ConfigError: ``f_map == 0``（数学退化，``K`` 不可逆）。

    注：**``f_map`` 为负值不做校验**——本项目尚未给它裁定语义（负值可能被
    用来表达方向翻转，也可能纯属笔误），在没有明确约定的情况下拒绝它会禁掉
    未知的合法用法。值正确性由调用方与 ``DATASET_CONVENTIONS`` 负责。
    """
    if F_prev.ndim != 3:
        raise ShapeError(
            f"F_prev: 期望 3 维 (C,H,W)，实得 shape={F_prev.shape}。"
            "warp 后无法确定通道/空间维度，故拒绝。")
    # 先把 gyro 归一成数组，后面形状/有限性检查与取值都用它
    # （调用方可能传 list；直接 .shape 会 AttributeError）。
    g = check_finite_array(gyro, "gyro")
    if g.shape != (3,):
        raise ShapeError(
            f"gyro: 期望 3 维 [pitch, yaw, roll]（deg/s），实得 shape={g.shape}")
    if not np.isfinite(dt):
        raise NonFiniteError(f"dt: 必须是有限标量，实得 {dt!r}")
    if f_map == 0:
        raise ConfigError("f_map: 不能为 0（K 不可逆，单应退化）")
    # 整体有限性检查：F_prev 含 NaN 会被 warp 原样铺开，污染后续全部统计
    check_finite_array(F_prev, "F_prev")

    c, h, w = F_prev.shape
    roll = float(g[2]) if use_roll else 0.0
    H = rotation_homography(float(g[0]), float(g[1]), roll, dt, f_map,
                            w / 2.0, h / 2.0,
                            yaw_sign=yaw_sign, pitch_sign=pitch_sign,
                            roll_sign=roll_sign)
    out = np.empty_like(F_prev)
    for ch in range(c):
        out[ch] = cv2.warpPerspective(F_prev[ch], H, (w, h),
                                      flags=cv2.INTER_LINEAR
                                      | cv2.WARP_INVERSE_MAP,
                                      borderMode=border)
    return out


# 各数据集的存储约定（实测；见 MEMORY_FORMAT §8.7）
#
# 每条记录的字段：
#   signs —— (yaw_sign, pitch_sign, roll_sign)，把存储通道映射到旋转轴；
#            与 warp_by_rotation 的 WARP_INVERSE_MAP 语义配套生效。
#            注意：符号与 flag 是**联合**约定——`(-s, 有 flag)` 与
#            `(+s, 无 flag)` 在数值上完全等价（已用等价性测试锁定，实测
#            0.9655 == 0.9655）。本表记录的是**带 WARP_INVERSE_MAP 时**该用
#            的值。历史记录里出现过"无 flag 时代"的符号，读数时务必确认
#            当时的 flag 状态，否则会整体差一个负号。
#   lag   —— 预测 F_t 时该取哪一帧的读数：0 = gyro[t]（存储值描述的正是
#            t-1 -> t 这一步），1 = gyro[t-1]（存储值描述的是 t -> t+1）。
#            两个数据源在这里**确实相反**，不能全局二选一。
DATASET_CONVENTIONS: dict[str, dict[str, object]] = {
    # 真机 IMU：lag=1 明显更好（0.835 vs 0.906；500 帧符号/配对联合扫描）。
    # 符号 (-,-,-)：这是"无 flag 时代的 (+,+,+)"换算过来的结果——加 flag
    # 必须整体取反才能保持数值不变（见上面的联合约定）。若误写成 (+,+,+)
    # 会把残差从 0.88 推到 1.27（比不补偿还差），是本线最容易踩的坑。
    "real_imu": {"signs": (-1.0, -1.0, -1.0), "lag": 1},
    # AoE：prep_aoe.trajectory_gyro 存的就是 (t-1 -> t) 的转角，故 lag=0。
    # 逐段独立标定（前半拟合 / 后半评估，12 段）12/12 一致给出 yaw-1 pitch+1
    # roll+1，与既有 scripts/analyze_negatives.py 的标定完全吻合——注意那个
    # 脚本自带 flag，所以这里的符号可以直接对照，无需再取反。
    "aoe": {"signs": (-1.0, 1.0, 1.0), "lag": 0},
}

# 兼容旧名字（历史调用点用的是 DATASET_SIGNS）。
DATASET_SIGNS: dict[str, tuple[float, float, float]] = {
    k: v["signs"] for k, v in DATASET_CONVENTIONS.items()
}  # type: ignore[assignment]


def convention_for(path: str) -> dict[str, object]:
    """按数据路径给出约定 dict（``signs`` 与 ``lag``），避免调用点硬编码。"""
    p = str(path).lower()
    if "real_imu" in p:
        return DATASET_CONVENTIONS["real_imu"]
    if "aoe" in p:
        return DATASET_CONVENTIONS["aoe"]
    return {"signs": (1.0, 1.0, 1.0), "lag": 0}


def signs_for(path: str) -> tuple[float, float, float]:
    """按数据路径给出 (yaw_sign, pitch_sign, roll_sign)。"""
    return convention_for(path)["signs"]  # type: ignore[return-value]


def lag_for(path: str) -> int:
    """按数据路径给出 gyro 配对滞后（0 = gyro[t]，1 = gyro[t-1]）。"""
    return convention_for(path)["lag"]  # type: ignore[return-value]

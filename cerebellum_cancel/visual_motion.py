"""无 IMU 的自运动估计 —— 用视觉替代伴随放电副本。

## 什么时候需要它

引擎的主路径靠 IMU 陀螺流做伴随放电副本（成本 0.033 ms/帧）。但真实数据经常
没有 IMU：老手机、网络视频、只有屏幕录制、或者 IMU 丢包。此时**不能什么都不做**
——现在的 `gyro=0` 等于假设"设备完全静止"，而这恰好是本项目开篇批评的帧差法的
失效模式（GUIDE §0.2：固定监控能用、移动设备全线崩溃）。

所以降级路径必须**真的估计自运动**，而不是假装它不存在。

## 为什么不用现成的稳像库

调研过（`scripts/survey_oss.py`）：视频稳像库（vid.stab ★957、DUT、StabNet）的
目标是**平滑**抖动，让路径看起来像轨道车——与本项目**方向相反**，我们要**保留**
设备预测不了的变化。它们的"全局运动估计"部件思路可借鉴，但整体目标不兼容，
拉进来还要拆掉半边。

## 实测选型（本机 MPS/CPU，120 帧）

| 方法 | 成本 | 相对 IMU |
| --- | --- | --- |
| **稀疏特征 + 仿射（本模块）** | **0.96 ms/帧** | 29x |
| ECC（稠密，精度高） | 1.74 ms/帧 | 53x |
| 相位相关（只有平移） | 0.77 ms/帧 | 23x |
| （IMU 副本，主路径） | 0.033 ms/帧 | 1x |

选稀疏特征 + 仿射的理由：**给完整仿射（含旋转）**，而 GUIDE 已实测"roll 未建模"
是主路径最大的模型误差来源；相位相关只给平移，等于退回主路径已经证伪的近似。

成本 0.96 ms/帧 仍比光流（8.5 ms/帧）便宜 9 倍。结论要诚实说：
**这是降级路径，不是等价替代**——它比 IMU 贵 29 倍，且在某些数据上会失败
（低纹理、快速运动、动态前景占多数时特征匹配会退化）。

## 一个必须记录的负结果（2026-09-12 实测）

在 `data/webcam_rec/video.mp4` 上逐帧对照，视觉仿射补偿**反而更差**：

| 口径 | 残差 | 比值 |
| --- | --- | --- |
| 不补偿 | 2.632 | 1.000 |
| 只补偿平移 | 4.962 | **1.885** |
| 只补偿旋转 | 6.650 | **2.527** |

而估计本身**是可信的**（内点中位 144，MOVEMENT 中位仅 0.29 px / 0.05 deg）。
所以失败**不是**估计精度问题，而是：**这段数据里根本没有可补偿的自运动。**
补偿一个 ~0.3 px 的位移，代价是插值模糊（GUIDE §2.1.3 记录的伪影），净亏。

这与 GUIDE §2.3.3 的既有实测一致——桌面级 egocentric 自运动在 28×28 特征图上
是**亚像素**的，此时任何 warp 都物理中性或负收益。

**⇒ 结论：本模块的价值取决于数据，不取决于算法。** 数据满足以下条件才值得开：
自运动位移**显著大于**插值误差量级（经验上 > 1 特征图像素 ≈ 数 px 源图）。
判断方法：跑一遍看 meta 里的 `motion_source_note` 与残差对比，**不要**因为
"有降级路径"就默认打开它。

## 一个必须保留的诚实信号

`estimate()` 返回 `ok` 标志与 `n_inliers`。**低置信度的估计不该被当成运动读数。**
若匹配失败率高，本模块**明确报告失败**，由调用方决定是拒绝整段、还是退回到
"不作补偿并声明降级"。不静默返回一个瞎猜的位移。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .contracts import ShapeError, check_positive

# 判定"这一帧的运动估计可信"的最低内点数。低于它认为是纹理不足/前景主导，
# 宁可报告失败也不给一个会污染统计的假读数。
MIN_INLIERS = 12

# **可补偿的运动下限**（源图像素）。低于它的运动不值得 warp：
# 补偿的收益小于插值模糊的代价。实测依据（见模块文档负结果）：
# 0.29 px 的位移去补偿，残差反而涨到 1.885x。取 1.0 px 作为保守门槛——
# 这与"特征图上 1 格"同量级（28 格 / 320px ≈ 11 px 源图，故 1 px 源图
# 远小于一格，是**宽松**门槛；真正的判断应看特征图上的位移）。
MIN_COMPENSABLE_SHIFT_PX = 1.0


def assess_motion(prev_gray: np.ndarray, gray: np.ndarray,
                  estimator: "VisualMotionEstimator | None" = None) -> dict:
    """给出**单帧**的运动量诊断，用于回答"这份数据值不值得补偿"。

    这是本模块最该被先调用的函数：负结果（见模块文档）表明，在自运动亚像素的
    数据上开补偿是**净亏**。所以先量，再决定开不开。
    """
    est = (estimator or VisualMotionEstimator()).estimate(prev_gray, gray)
    return {
        "ok": est.ok,
        "shift_px": est.shift_px,
        "n_inliers": est.n_inliers,
        "n_features": est.n_features,
        "above_compensation_floor": est.shift_px >= MIN_COMPENSABLE_SHIFT_PX,
        "floor_px": MIN_COMPENSABLE_SHIFT_PX,
        "advice": (
            "运动高于补偿下限，warp 可能有益"
            if est.shift_px >= MIN_COMPENSABLE_SHIFT_PX else
            "运动低于补偿下限，warp 会因插值模糊而净亏——建议不开补偿"
            if est.ok else
            "估计失败（纹理不足或前景主导），不建议据此判断"),
    }


@dataclass
class MotionEstimate:
    """一帧的相对运动估计（源图 -> 当前图，像素坐标）。"""

    # 2x3 仿射矩阵，把上一帧的点映射到当前帧： p_t = A @ [p_{t-1}, 1]
    affine: np.ndarray | None
    ok: bool
    n_inliers: int
    n_features: int
    # 平移量的模（像素），便于快速判断"这帧动了多少"
    shift_px: float

    def as_dict(self) -> dict:
        return {"ok": self.ok, "n_inliers": self.n_inliers,
                "n_features": self.n_features,
                "shift_px": round(self.shift_px, 3)}


class VisualMotionEstimator:
    """用稀疏特征匹配 + RANSAC 仿射拟合估计帧间全局运动。

    Args:
        max_features: 每帧取多少特征点（越多越稳、越慢）。
        quality: goodFeaturesToTrack 的质量阈值。
        min_distance: 特征点最小间距（像素）。
    """

    def __init__(self, max_features: int = 200, quality: float = 0.01,
                 min_distance: float = 7.0):
        self.max_features = int(max_features)
        self.quality = float(quality)
        self.min_distance = float(min_distance)
        check_positive(self.max_features, "max_features")
        check_positive(self.min_distance, "min_distance")

    def estimate(self, prev_gray: np.ndarray,
                 gray: np.ndarray) -> MotionEstimate:
        """估计 prev_gray -> gray 的全局运动。

        两帧都必须是 2 维灰度（`(H,W)`）——取彩色会静默降级到只在第一个通道
        上匹配，结果难查。这里显式拒绝。
        """
        a = prev_gray
        if a.ndim != 2 or gray.ndim != 2:
            raise ShapeError(
                f"输入必须是 2 维灰度 (H,W)，实得 "
                f"{prev_gray.shape} / {gray.shape}。彩色输入会静默只在"
                "第一个通道上匹配，故拒绝。")
        if prev_gray.shape != gray.shape:
            raise ShapeError(
                f"两帧尺寸必须一致，实得 {prev_gray.shape} vs {gray.shape}")

        p0 = cv2.goodFeaturesToTrack(
            prev_gray, self.max_features, self.quality, self.min_distance)
        if p0 is None or len(p0) < 6:
            return MotionEstimate(None, False, 0, 0 if p0 is None else len(p0),
                                  0.0)

        # pyright 无法解析 cv2 的重载（nextPts=None 是合法的"内部新建"用法），
        # 用 type: ignore 显式记录，而不是改写成不惯用的形式。
        # 注：ignore 必须落在**报错那一行**（参数行），放在函数名行不起作用。
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None)              # type: ignore
        if p1 is None or st is None:
            return MotionEstimate(None, False, 0, len(p0), 0.0)

        ok = st.ravel() == 1
        if ok.sum() < 6:
            return MotionEstimate(None, False, int(ok.sum()), len(p0), 0.0)

        # RANSAC 仿射（partial = 相似变换，含旋转+各向同性缩放）。
        # 用 partial 而不是 full：full 仿射有 6 自由度，在噪声下容易拟合出
        # 剪切形变，而真实手持运动的帧间形变接近相似变换。
        M, inl = cv2.estimateAffinePartial2D(
            p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return MotionEstimate(None, False, 0, len(p0), 0.0)

        n_inl = int(inl.sum()) if inl is not None else int(ok.sum())
        shift = float(np.hypot(M[0, 2], M[1, 2]))
        return MotionEstimate(M.astype(np.float32), n_inl >= MIN_INLIERS,
                              n_inl, len(p0), shift)

    def gyro_equivalent(self, prev_gray: np.ndarray, gray: np.ndarray,
                        f_map: float, dt: float) -> np.ndarray:
        """把视觉运动换算成"等价陀螺读数"（deg/s），喂给现有 warp 路径。

        这样上层（`warp_by_rotation` / `FeatureMapCancel`）**不需要任何改动**——
        降级路径产出的仍是同一个接口，只是来源从 IMU 换成了视觉。

        推导：相似变换的旋转分量 `theta = atan2(M[1,0], M[0,0])` 即本帧转角。
        平移在此**不**折算成角速度——平移不是旋转，硬折算会引入 GUIDE §2.3.5
        实测过的"常深度平移模型"错误。所以只回填旋转，平移留给未来的专职项。
        """
        est = self.estimate(prev_gray, gray)
        out = np.zeros(3, dtype=np.float64)
        if not est.ok or est.affine is None:
            return out
        # **运动太小则不补偿。** 实测：0.29 px 的位移去 warp，残差反而涨到
        # 1.885x——补偿的收益小于插值模糊的代价。返回 0 等于"这一帧不动"，
        # 这是**正确**的保守选择，不是失败。
        if est.shift_px < MIN_COMPENSABLE_SHIFT_PX:
            return out
        th = float(np.arctan2(est.affine[1, 0], est.affine[0, 0]))   # rad
        deg = np.degrees(th) / max(dt, 1e-9)
        # 图像内的 roll 直接对应绕光轴旋转；pitch/yaw 从平移近似不出来，
        # 这里**只填 roll**，并把这一点如实记录（不要假装三个轴都有）。
        out[2] = deg
        return out

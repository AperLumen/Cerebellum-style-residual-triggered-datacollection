"""可插拔采集协议层：帧流 + IMU 流 -> 存储决策 + 残差图。

把触发器包成视觉模型协议栈里的一层：

    前端 (可插拔)      取消层                    触发层
    HWC uint8 帧  ->  特征图  ->  FeatureMapCancel  ->  AdaptiveTrigger
                      (C,H,W)     gyro warp + z 残差      回滞 + 自校准
                                                    |
                                                    v
                              逐帧 {keep, in_clip, surprise, z_map, clips}

接口契约：

- ``frontend``：callable，输入 HWC uint8 帧，输出 (C,H,W) float32 特征图。
  默认内置 ``"mobilenet_v3_small[:3]"``（冻结 backbone 前 3 段，224 输入
  -> (24,28,28)）；换成任何符合签名的 callable 即可替换（协议栈并入时
  换成上游模型自己的特征）。
- ``step()`` 在线逐帧。backbone 前向每帧都跑：冻结前 3 段约 1 ms/帧
  量级（本机 MPS 实测），30 fps 帧预算的 ~3%，常开可接受；筛掉一帧的
  存储/带宽成本远高于此。
- ``finalize()`` 输入结束后调用，返回触发的片段 [(start, end), ...]（秒）。

定位口径：动能不可预测 ≠ 语义重要性。本层的作用是
运动去重 / 带宽预筛 / 片段筛选——剪掉自运动可解释的冗余帧，把存储
花在运动上不可预测的片段上；语义重要性仲裁必须发生在视觉模型侧。
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

from .core import AdaptiveTrigger, TriggerConfig
from .featmap_cancel import (CancelConfig, FeatureMapCancel, maxcell_score)

# HWC uint8 帧 -> (C,H,W) float32 特征图
Frontend = Callable[[np.ndarray], np.ndarray]


def _build_frontend(spec: str, device: str | None) -> Frontend:
    """把 "mobilenet_v3_small[:3]" 这类字符串规格编译成 callable 前端。

    延迟 import torchvision/torch：纯 numpy 前端（测试、协议栈并入后的
    外部特征源）不需要这两个依赖，保持包在非视觉环境下可导入。
    """
    import torch
    import torchvision
    import torchvision.transforms as T
    from PIL import Image

    import re

    m = re.fullmatch(r"([a-zA-Z0-9_]+)(?:\[:(\d+)\])?", spec.strip())
    if m is None:
        raise ValueError(f"无法解析前端规格: {spec!r}（示例: mobilenet_v3_small[:3]）")
    name, depth_s = m.group(1), m.group(2)
    depth = int(depth_s) if depth_s else None
    if not hasattr(torchvision.models, name):
        raise ValueError(f"未知前端模型: {spec!r}（torchvision 无 {name!r}）")

    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = getattr(torchvision.models, name)(weights="DEFAULT").features
    if depth is not None:
        model = model[:depth]
    model = model.to(dev).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def frontend(frame: np.ndarray) -> np.ndarray:
        from typing import cast
        # 与 extract_featmaps.py 同一条变换链；调用方负责给 RGB（PIL 约定）
        x = cast(torch.Tensor, tf(Image.fromarray(frame)))
        y = model(x.unsqueeze(0).to(dev))
        return y[0].cpu().numpy().astype(np.float32)

    return frontend


class AcquisitionLayer:
    """采集协议层：包一层可插拔前端 + FeatureMapCancel + AdaptiveTrigger。

    Args:
        frontend: callable（HWC uint8 -> (C,H,W) float32），或字符串规格
            "mobilenet_v3_small[:3]"（默认，冻结 backbone，与
            extract_featmaps.py 一致）。
        cancel_cfg: CancelConfig（fps、f_map、robust 等）；None 取默认。
        trigger_cfg: TriggerConfig（回滞阈值/前后沿），或一个配好
            warmup/stat_ema 的 AdaptiveTrigger 实例直接注入。
    """

    def __init__(self, frontend: Union[str, Frontend] = "mobilenet_v3_small[:3]",
                 cancel_cfg: CancelConfig | None = None,
                 trigger_cfg: TriggerConfig | AdaptiveTrigger | None = None,
                 device: str | None = None):
        self.frontend: Frontend = (_build_frontend(frontend, device)
                                   if isinstance(frontend, str) else frontend)
        self.cancel = FeatureMapCancel(cancel_cfg or CancelConfig())
        if isinstance(trigger_cfg, AdaptiveTrigger):
            self.trigger = trigger_cfg
        else:
            self.trigger = AdaptiveTrigger(cfg=trigger_cfg or TriggerConfig())
        self.fps = self.cancel.cfg.fps
        self._last_t = 0.0

    def step(self, frame: np.ndarray, gyro: np.ndarray, t: float) -> dict:
        """在线逐帧：前端前向 + cancel.step + trigger.update。

        Args:
            frame: (H,W,3) uint8，RGB 序（cv2 读出的 BGR 先 [:, :, ::-1]）。
            gyro: (3,) deg/s，[pitch, yaw, roll]（与 io.load_imu_csv 同序）。
            t: 帧时间戳（秒，视频时钟）。

        Returns:
            {"keep": bool,          # 在线存储决策：帧在触发片段内
             "in_clip": bool,       # 与 keep 同源（触发器当前输出）
             "surprise": float,     # cancel 的 max-|z|（原始终残差强度）
             "score": float,        # 喂给触发器的 maxcell z^2（已发布读出）
             "z_map": (H,W) 残差图,  # z 按通道求 RMS，峰值帧可叠加原帧可视化
             "clips": list}         # 当前已封闭的片段 [(start, end), ...]
        """
        F = self.frontend(frame)
        out = self.cancel.step(F, np.asarray(gyro, dtype=np.float64))
        z = out["z"]
        score = float(maxcell_score(z[None], self.cancel.cfg.border)[0])
        in_clip = bool(self.trigger.update(t, score, self.fps))
        self._last_t = float(t)
        return {
            "keep": in_clip,
            "in_clip": in_clip,
            "surprise": float(out["surprise"]),
            "score": score,
            "z_map": np.sqrt((z ** 2).mean(axis=0)).astype(np.float32),
            "clips": list(self.trigger.clips),
        }

    def finalize(self) -> list[tuple[float, float]]:
        """结束输入，返回全部触发片段 [(start, end), ...]（秒）。

        finalize 的片段集合是 keep 决策的封闭版：online 的 keep 只反映
        触发器当前输出，片段的右端点要等 off 回滞或 finalize 才确定。
        """
        return self.trigger.finalize(self._last_t)

"""多层残差预测（cascade）——把"光层数增加时性能提升"这条翻译成工程实验。

来源。Petrini et al.（Nature 657, 377, 2026，DOP 调制 / 高维光计算）在 CIFAR-10
上的一个关键观察：**光层是线性的，但增加层数仍然提升性能**。作者把原因归为
偏振（矢量化）引入的有效非线性——每个空间模携带三维矢量，层的"线性"运算作用
在矢量场上，复合出的自由度超过标量情形。

翻译到本项目。本项目的预测器是**线性 warp**（shift/f_map 矩阵平移）。README
已实测一条负结果（§2.3.5）：**常深度平移 warp 是负收益**——加一个线性项没用。
但"层数"这条给出的是另一种可能：如果单层线性 warp 之后的残差里还留着**可预测
结构**，那么**再叠一层预测器**（作用于残差）就可能继续降残差，尽管每一层都是
线性的。这不是"加一个自由项"，而是"换一个复合方式"。

于是得到一个可证伪的问题：

    Q: 在一阶 warp 的残差上再预测一次，残差能不能继续降？
       代价是多少？净收益（考虑级联的存储/算力开销）是正还是负？

实现。``CascadeCancel`` 把 N 个线性预测器串起来，每层预测**上一层的残差**：

    r_0 = F_t                                 （观测）
    r_k = r_{k-1} - P_k(r_{k-1}, a_t)         （第 k 层减掉自己的预测）

第 1 层是既有的 gyro warp（rot90 位移预测）；第 2 层起是**逐格线性预测器**：
用「上一层残差的时序统计 + 动作特征」预测这一层会出现的残差。逐格、在线、
无反向传播——延续项目的局部可塑性原则（GUIDE §3.4 原则三）。

第 2 层的预测子是论文"granule-cell temporal basis"的极简版：把上一层残差的
**慢速指数平均**（该格平时残余多少）作为对下一帧的预测。这正是残差图 lag-1
相关的直接可利用形式。

诚实声明：这是一个**关于层数的工程假设检验**，不是光计算的复现。DOP 论文的
非线性来自矢量化光场；本项目的"再多一层"来自残差的时序结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .featmap_cancel import CancelConfig, FeatureMapCancel, shift_map


@dataclass
class CascadeConfig:
    """级联配置。

    ``n_layers`` 是**预测层数**（1 = 只有 gyro warp，即现状）：
    - 层 1：gyro warp（与 FeatureMapCancel 同口径，用同一份 CancelConfig）
    - 层 ≥2：由 ``layer_kind`` 决定——

    ``layer_kind="temporal"``：**慢速时序水平**。用该格残差的 EMA 预测下一帧
    的残余。对应论文的 granule-cell temporal basis / 负像：吃掉"帧差消不掉的
    持续性残余"（慢漂移、对焦呼吸、曝光适应）。

    ``layer_kind="spatial"``：**中心-周边去相关**。减掉局部空间均值（可看作
    单步的侧抑制/ON-OFF 对抗通道），把局部尖峰从平滑背景里凸显出来。它是
    **无状态**的（不学习），纯几何算子。

    ``slow_ema``：temporal 层的更新率；
    ``spatial_leak``：spatial 层保留多少周边均值（1.0 = 全减，0 = 不作用）；
    ``gate_k``：稳态门——惊讶的格不更新预测（防止把事件学掉）。0 = 不设门。
    """

    cancel: CancelConfig = field(default_factory=CancelConfig)
    n_layers: int = 1
    layer_kind: str = "temporal"
    slow_ema: float = 0.02
    gate_k: float = 4.0
    per_cell_gate: bool = True
    spatial_leak: float = 1.0


def _neighbour_mean(F: np.ndarray) -> np.ndarray:
    """四邻域均值（沿空间轴用 roll 平均，边界环绕——后续读数本就排除边界）。"""
    return 0.25 * (np.roll(F, 1, axis=-1) + np.roll(F, -1, axis=-1)
                   + np.roll(F, 1, axis=-2) + np.roll(F, -1, axis=-2))


class CascadeCancel:
    """N 层线性预测级联，逐格在线，无训练、无反向传播。

    每层输出该层视角下的残差；``surprise`` 读数取**最末层**的 z 分数峰值
    （级联的意义就是让这个读数更干净）。同时返回每层的 z 峰值，用于回答
    "多出来的层到底减掉了多少"。
    """

    def __init__(self, cfg: CascadeConfig | None = None):
        self.cfg = cfg or CascadeConfig()
        if self.cfg.n_layers < 1:
            raise ValueError("n_layers 至少为 1（层 1 = gyro warp）")
        if self.cfg.layer_kind not in ("temporal", "spatial"):
            raise ValueError(
                f"未知 layer_kind: {self.cfg.layer_kind!r}"
                "（temporal / spatial）")
        self.base = FeatureMapCancel(self.cfg.cancel)
        self.reset_layers()

    def reset_layers(self):
        """重置第 2 层起的逐格预测器状态。"""
        self.pred: list[np.ndarray] = []       # 每层一个 (C,H,W) 水平估计
        self.psd: list[np.ndarray] = []        # 每层一个 (C,H,W) 残余尺度
        self.layer_surprise: list[float] = []
        self.n = 0

    def _ensure_layers(self, shape: tuple[int, ...]) -> None:
        need = self.cfg.n_layers - 1
        while len(self.pred) < need:
            self.pred.append(np.zeros(shape, dtype=np.float32))
            self.psd.append(np.ones(shape, dtype=np.float32))

    def step(self, F: np.ndarray, gyro: np.ndarray,
             trans: np.ndarray | None = None) -> dict:
        """喂一帧特征图。

        返回：
        - ``surprise``：末层 z 的峰值（max |z|，与 FeatureMapCancel 同定义）；
        - ``z_final``：**末层的 z 图**——触发读出（``maxcell_score``）应作用在
          它上面，而不是作用在一阶 z 上（否则触发器看的是被级联补偿前的量）；
        - ``layer_surprise``：逐层输入残差的 z 峰值（拆解各层贡献）；
        - ``r``：末层残差（原始尺度，非 z），平静残差口径用它；
        - ``z``：一阶 z（保持与 FeatureMapCancel 的接口语义一致）。
        """
        base_out = self.base.step(F, gyro, trans)
        # base.step() 保证统计已初始化（首帧会建成 zeros/ones）
        assert self.base.var is not None
        sd0 = np.sqrt(self.base.var) + 1e-3
        r = base_out["z"] * sd0                     # 还原一阶残差（原始尺度）
        errmap = np.asarray(r, dtype=np.float32)
        self._ensure_layers(errmap.shape)
        layer_sur = [float(base_out["surprise"])]
        z_final = base_out["z"]

        cur = errmap
        for li in range(self.cfg.n_layers - 1):
            pred = self.pred[li]

            if self.cfg.layer_kind == "spatial":
                # 中心-周边去相关：无状态算子，减掉局部均值（保留 leak 比例的
                # 周边信息）。不学习，所以没有"把事件学掉"的风险。
                pred_new = self.cfg.spatial_leak * _neighbour_mean(cur)
                resid = cur - pred_new
                self.pred[li] = pred_new
                sd = np.sqrt(self.psd[li]) + 1e-3
                z = resid / sd
                # 尺度仍按 always-update 维护，供门控与后续层使用
                a = self.cfg.slow_ema
                self.psd[li] += a * (resid * resid - self.psd[li])
            else:
                # temporal：慢速水平预测 + 稳态门（惊讶的格不学）
                resid = cur - pred
                sd = np.sqrt(self.psd[li]) + 1e-3
                z = resid / sd
                upd = np.abs(z) < self.cfg.gate_k if self.cfg.gate_k > 0 \
                    else np.ones_like(z, dtype=bool)
                if not self.cfg.per_cell_gate:
                    b0 = self.cfg.cancel.border
                    upd = np.full_like(
                        z, bool(np.abs(z[:, b0:-b0, b0:-b0]).max()
                                < self.cfg.gate_k))
                a = self.cfg.slow_ema
                pred += np.where(upd, a * resid, 0.0)
                self.psd[li] += np.where(
                    upd, a * (resid * resid - self.psd[li]), 0.0)
                self.pred[li] = pred

            z_final = z
            b = self.cfg.cancel.border
            layer_sur.append(float(np.abs(z[:, b:-b, b:-b]).max()))
            cur = resid

        self.n += 1
        self.layer_surprise = [round(float(x), 6) for x in layer_sur]
        return {
            "surprise": layer_sur[-1],
            "layer_surprise": layer_sur,
            "z_final": z_final,             # 触发读出作用于此
            "r": cur,                       # 末层残差（原始尺度）
            "z": base_out["z"],             # 一阶 z（既有接口语义）
            "learned": base_out["learned"],
        }

    def dump_stats(self) -> dict:
        """导出全部层状态（含基线层），供落盘续采。"""
        out = dict(self.base.dump_stats())
        for i, (p, s) in enumerate(zip(self.pred, self.psd)):
            out[f"cascade_pred_{i}"] = p
            out[f"cascade_psd_{i}"] = s
        out["cascade_n_layers"] = np.asarray([self.cfg.n_layers], dtype=np.int64)
        return out

    def load_stats(self, stats: dict) -> None:
        """载入 ``dump_stats()`` 的产物。

        形状校验以**基线层**的网格为准（``base.mu`` 的形状来自前端，独立于
        载入内容）；第 2 层起的数组形状必须与它一致，否则拒绝——用载入内容
        去创建状态会让校验永远通过（等于没校验）。
        """
        self.base.load_stats(stats)
        grid = None if self.base.mu is None else self.base.mu.shape
        for i in range(self.cfg.n_layers - 1):
            key = f"cascade_pred_{i}"
            if key not in stats:
                continue
            p = np.asarray(stats[key], dtype=np.float32)
            psd_key = f"cascade_psd_{i}"
            if psd_key not in stats:
                raise ValueError(f"级联状态不完整：缺少 {psd_key}")
            s = np.asarray(stats[psd_key], dtype=np.float32)
            if s.shape != p.shape:
                raise ValueError(
                    f"级联第 {i+2} 层 pred/psd 形状不一致: "
                    f"{p.shape} vs {s.shape}")
            if grid is not None and p.shape != grid:
                raise ValueError(
                    f"级联第 {i+2} 层形状与特征图网格不符: "
                    f"{p.shape} vs {grid}（前端或分辨率变了）")
            self._ensure_layers(p.shape)
            if self.pred[i].shape != p.shape:
                raise ValueError(
                    f"级联第 {i+2} 层与已有状态形状不符: "
                    f"{self.pred[i].shape} vs {p.shape}")
            self.pred[i] = p.copy()
            self.psd[i] = s.copy()


def rollout_cascade(canc: CascadeCancel, fmaps: np.ndarray,
                    gyros: np.ndarray,
                    trans: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跑一段序列，返回 ``(surprise, layer_surprise, r_final)``。

    ``layer_surprise`` 为 (T, n_layers)：第 k 列是第 k+1 层输入的 z 峰值。
    """
    T_ = len(fmaps)
    sur = np.zeros(T_, dtype=np.float32)
    p = max(canc.cfg.n_layers, 1)
    ls = np.zeros((T_, p), dtype=np.float32)
    rs = np.zeros((T_,) + tuple(fmaps.shape[1:]), dtype=np.float32)
    for t in range(T_):
        tv = trans[t] if trans is not None else None
        out = canc.step(fmaps[t], gyros[t], tv)
        sur[t] = out["surprise"]
        ls[t, :len(out["layer_surprise"])] = out["layer_surprise"]
        rs[t] = out["r"]
    return sur, ls, rs

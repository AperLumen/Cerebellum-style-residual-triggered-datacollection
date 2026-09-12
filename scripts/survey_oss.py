#!/usr/bin/env python3
"""调研：与 CMCI 相关的开源方案是否已有成熟实现。

目的不是"找轮子就换"，而是搞清楚**我们有没有在重复造一个别人已经做好的东西**。
按四类候选分别查，用 GitHub API（可复现，且网页搜索被限流/CAPTCHA）。

## 结论（2026-09-12 实测）

**一、没有任何库做我们这件事。** 视频稳像（vid.stab / DUT / StabNet）与视频摘要
（VASNet 等）的目标都与我们**相反**：

| | 目标 | 与 CMCI 的关系 |
| --- | --- | --- |
| 视频稳像 | **平滑**掉抖动（让路径像轨道车） | 反向——我们要**保留**不可预测的变化 |
| 视频摘要 | 找"有代表性/好看"的帧 | 不同判据（美学/覆盖度，非可预测性） |
| 主动学习/coreset | 选信息量大的**样本**（要有标签/模型） | 需要模型或标签；我们要求零训练零标注 |

稳像库反而很值得作为**对照基线**：它同样要做全局运动估计，但不需要 IMU。
于是有一个可证伪的问题——"用视觉运动估计替代 IMU，够不够？"
本项目已有实测答案（GUIDE §2.3.1）：光流残差质量更好（0.070 vs 0.079），
但**贵 100 倍**（8.5 ms vs 0.033 ms），在常开约束下不可接受。

**二、无 IMU 时不该"什么都不做"。** 现在的行为是 gyro=0 → 等于假设设备静止，
而帧差法的失效模式正是本项目开篇的动机（GUIDE §0.2：固定监控能用、移动设备
全线崩溃）。所以降级必须**显式估计自运动**，或**明确声明降级**。

可用的合规方案（本环境实测具备）：`cv2.estimateAffinePartial2D` /
`findTransformECC` / `phaseCorrelate` —— 无需引入任何新依赖。

**三、值得参考但不引入的项目**：
- `georgmartius/vid.stab`（★957，2026-08 更新，C 库，需 ffmpeg 编译）——
  它的**多测量场（measurement fields）**思路可借鉴：在对比度高的区域取小块做
  局部配准，比全局光流便宜且更抗纹理缺失。
- `google-ai-edge/mediapipe`（★36920，Apache-2.0）——若将来要做"手机端实时"，
  它的 IMU/视觉融合与跨平台部署是现成路径。当前不需要。

    PYTHONPATH=. .venv/bin/python scripts/survey_oss.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

# 只查**具体已知**的项目（网页搜索被限流/CAPTCHA，API 更可靠且可复现）
CANDIDATES = {
    "视频稳像/全局运动估计": [
        "georgmartius/vid.stab",          # 经典 C 库，ffmpeg 内置
        "ckmayu/video-stabilization",     # 教学向 Python 实现
        "Annbless/DUTCode",               # 深度稳像，含运动估计
        "XuYicong/StabNet",               # 深度稳像
    ],
    "光流/运动估计": [
        "princeton-vl/RAFT",              # 本项目已用作上限对照
        "autonomousvision/unimatch",      # 半监督光流
        "zhangtao-seu/GMFlow",            # GMFlow
    ],
    "数据集筛选/主动采样": [
        "senerozan/Deep-coresets",         # coreset 选择
        "google/active-learning-playground",
        "Baifeng-Shi/active-learning",     # 主动学习综述配套
        "TrentonBricken/active-learning",
    ],
    "视频摘要/关键帧": [
        "wzhy/VASNet",                     # 视频摘要
        "xiayandi/UnsupVideoSum",          # 无监督视频摘要
        "google/mediapipe",                # 含 IMU/视觉融合工具
    ],
}


def query(full: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{full}",
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "cmci-survey"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_err": f"HTTP {e.code}"}
    except Exception as e:                              # noqa: BLE001
        return {"_err": type(e).__name__}


def main() -> int:
    print("=" * 78)
    print("开源方案调研（数据来自 GitHub API，可复现）")
    print("=" * 78)
    rows = []
    for cat, repos in CANDIDATES.items():
        print(f"\n【{cat}】")
        for full in repos:
            d = query(full)
            if "_err" in d:
                print(f"  {full:38s} -> {d['_err']}")
                continue
            lic = (d.get("license") or {}).get("spdx_id") or "—"
            rows.append({
                "category": cat, "repo": d.get("full_name", full),
                "stars": d.get("stargazers_count", 0),
                "license": lic,
                "pushed": (d.get("pushed_at") or "")[:10],
                "archived": d.get("archived", False),
                "desc": (d.get("description") or "")[:70],
            })
            flag = " [ARCHIVED]" if d.get("archived") else ""
            print(f"  {d.get('full_name', full):38s} "
                  f"★{d.get('stargazers_count', 0):>6}  {lic:12s} "
                  f"{str(d.get('pushed_at'))[:10]}{flag}")
            print(f"      {str(d.get('description'))[:70]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

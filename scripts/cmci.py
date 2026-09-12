#!/usr/bin/env python3
"""CMCI 命令行入口 —— 把一段录制过滤成数据集/索引/表格。

    # 最简：塞进去，出数据集 + 可看的 HTML 索引
    python cmci.py run clip.mp4 --imu imu.csv --out ./out

    # 选格式
    python cmci.py run clip.mp4 --imu imu.csv --out ./out --format dataset,html,jsonl

    # 只看会做什么，不真跑
    python cmci.py run clip.mp4 --imu imu.csv --out ./out --dry-run

    # 描述一个已存在的输出
    python cmci.py info ./out

设计原则：
  * **不猜用户的语义意图。** 默认只做第 1 层（运动去重），并在输出里写明。
    要按业务规则筛，用 `--rule` 指一个 Python 文件（见 `--help-rule`）。
  * **不静默降级。** 没 IMU 会明确警告"退化为帧差法"（能力弱很多），
    而不是假装一切正常。
  * **参数错误在读数据之前报。** 大文件读半天才说参数错了很恼人。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import cast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cerebellum_cancel.contracts import ContractError      # noqa: E402
from cerebellum_cancel.engine import (EngineConfig, Rule,  # noqa: E402
                                      _EXPORTERS, run)

RULE_HELP = """\
第 2 层过滤规则怎么写
=====================

引擎先做第 1 层「运动去重」，再把候选片段交给你的规则函数筛第二遍。
规则文件里定义一个 `rule(segments, frame_view) -> segments`：

    # my_rule.py
    def rule(segments, fv):
        # segments: list[Segment]，字段见 engine.Segment
        # fv: FrameView，只读逐帧数据（t / score / surprise / keep / gyro）
        out = []
        for s in segments:
            if s.peak_score > 20 and s.duration > 0.5:
                s.selection_reason = "rule:high_peak"   # 标上你的名字
                out.append(s)
        return out

然后用 `--rule my_rule.py`。

为什么要有这一层：残差触发器只能回答“哪里动了”，回答不了“什么重要”
（后者在本项目的真实数据上被四轮实验否证）。所以引擎不替你判断语义，
把候选交给你——你在业务侧知道什么才叫“重要”。
"""


class _Invalid(Exception):
    """CLI 参数/规则文件的问题（与 contracts 的领域错误区分开）。"""


def _load_rule(path: str) -> Rule:
    """从 .py 文件加载 rule 函数。错误信息要能定位到具体问题。"""
    if not os.path.exists(path):
        raise _Invalid(f"规则文件不存在: {path}")
    spec = importlib.util.spec_from_file_location("cmci_user_rule", path)
    if spec is None or spec.loader is None:
        raise _Invalid(f"无法加载规则文件（不是合法的 .py？）: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "rule", None)
    if not callable(fn):
        raise _Invalid(
            f"规则文件里没有可调用的 `rule` 函数: {path}\n"
            "签名应为 rule(segments, frame_view) -> segments；"
            "用 --help-rule 看示例。")
    # 用户代码是动态加载的，pyright 只能看到 object；真实签名由
    # RULE_HELP 里写明的契约约束，并在 engine.run 调用时生效。
    return cast(Rule, fn)


def cmd_run(args: argparse.Namespace) -> int:
    # --- 参数校验（在任何 IO 之前）---
    if args.video is None:
        print("需要给出视频路径。", file=sys.stderr)
        return 2
    if args.fps is not None and args.fps <= 0:
        print(f"fps 必须为正，实得 {args.fps}", file=sys.stderr)
        return 2
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    unknown = [f for f in formats if f not in _EXPORTERS]
    if unknown:
        print(f"未知格式 {unknown}；可选 {sorted(_EXPORTERS)}", file=sys.stderr)
        return 2

    rule: Rule | None = None
    if args.rule:
        try:
            rule = _load_rule(args.rule)
        except _Invalid as exc:
            print(str(exc), file=sys.stderr)
            return 2

    # --- 明确的降级警告（不静默）---
    if not args.imu:
        if args.no_imu == "require":
            print("没有提供 IMU（--imu），而 --no-imu=require 要求必须有。",
                  file=sys.stderr)
            return 2
        if args.no_imu == "assume_still":
            print("[警告] 没有 IMU 且 --no-imu=assume_still：将**假设设备静止**。",
                  file=sys.stderr)
            print("       这**只能用于固定机位**——移动设备上自运动会被当成"
                  "真实变化，", file=sys.stderr)
            print("       正是本项目要解决的问题。", file=sys.stderr)
        else:
            print("[注意] 没有提供 IMU，将用**视觉估计**代替伴随放电副本：",
                  file=sys.stderr)
            print("       稀疏特征+RANSAC 仿射，约 0.96 ms/帧（IMU 是 0.033，"
                  "贵 29 倍）。", file=sys.stderr)
            print("       这是一条**降级**路径，不是等价替代；且若自运动是亚像素的，"
                  "补偿会净亏。", file=sys.stderr)
            print("       建议先跑 assess_motion 看运动量（见 GUIDE §4.9）。",
                  file=sys.stderr)

    if args.dry_run:
        print("=== dry-run（不读写任何文件）===")
        print(f"  输入视频: {args.video}")
        print(f"  输入 IMU: {args.imu or '(无 → 退化模式)'}")
        print(f"  输出目录: {args.out}")
        print(f"  格式: {formats}")
        print(f"  前端: {args.frontend}")
        print(f"  max_frames: {args.max_frames}")
        print(f"  第 2 层规则: {args.rule or '(未提供 → 只做运动去重)'}")
        return 0

    cfg = EngineConfig(
        frontend=args.frontend, fps=args.fps, imu_offset=args.imu_offset,
        mirror=args.mirror, max_frames=args.max_frames,
        no_imu=args.no_imu,
        keep_z_maps=("dataset" in formats),   # 只有数据集导出需要 z 图
    )

    # 只有 parquet 需要额外依赖，提前说清楚
    if "parquet" in formats:
        try:
            import pyarrow  # type: ignore  # noqa: F401
        except ImportError:
            print("导出 parquet 需要 pyarrow：pip install pyarrow\n"
                  "（或者把 --format 里的 parquet 去掉，jsonl 零依赖）",
                  file=sys.stderr)
            return 2

    try:
        res = run(args.video, args.imu, cfg=cfg, rule=rule)
    except ContractError as exc:
        print(f"[输入契约] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"[读取失败] {exc}", file=sys.stderr)
        print("提示：路径是帧目录还是视频文件？本工具目前吃视频文件（mp4/mov…）。",
              file=sys.stderr)
        return 3

    s = res.summary()
    print("=== 过滤完成 ===")
    print(f"  帧数        {s['n_frames']}")
    print(f"  时长        {s['duration_s']:.1f} s")
    print(f"  片段        {s['n_segments']} 个，合计 {s['stored_seconds']:.1f} s")
    print(f"  帧级保留    {s['keep_fraction'] * 100:.1f}%")
    print(f"  压缩比      {s['stored_seconds'] / max(s['duration_s'], 1e-9) * 100:.1f}%"
          " 的时长被保留")
    print()
    if rule is None:
        print("  选择依据    **仅第 1 层（运动去重）**")
        print("              被选中 = 设备自身运动预测不了；**不等于**语义重要。")
        print("              要按业务规则筛，用 --rule（见 --help-rule）。")
    else:
        print("  选择依据    第 1 层运动去重 + 你的 --rule 规则")
    print()

    written = res.save(args.out, formats=formats)
    print("=== 产物 ===")
    for k, v in written.items():
        # 用绝对路径显示：相对路径在 --out 指向仓库外时会变成一长串 ../
        p = os.path.abspath(v)
        n = sum(len(f) for _, _, f in os.walk(p)) if os.path.isdir(p) else 1
        print(f"  {k:9s} {p}" + (f"  ({n} 个文件)" if os.path.isdir(p) else ""))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """描述一个已有的输出目录。"""
    import json
    man = os.path.join(args.path, "dataset", "manifest.json")
    if os.path.exists(man):
        m = json.load(open(man, encoding="utf-8"))
        sec = m["sections"][0]
        print(f"=== 数据集 {args.path} ===")
        print(f"  段数      {m['n_sections']}   帧数 {m['n_frames']}")
        print(f"  完整性    {sec['integrity']['verdict']}")
        print(f"  weight    {m['weight']['status']}")
        eng = sec.get("meta", {}).get("engine", {})
        if eng.get("selection_scope"):
            print(f"  选择范围  {eng['selection_scope']}")
    else:
        print(f"{args.path} 里没有 dataset/manifest.json", file=sys.stderr)
        return 1
    idx = os.path.join(args.path, "index.html")
    if os.path.exists(idx):
        print(f"  可视化    {idx}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cmci", description="CMCI：把录制过滤成数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python cmci.py run clip.mp4 --imu imu.csv --out ./out")
    ap.add_argument("--help-rule", action="store_true",
                    help="打印第 2 层规则的写法")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="跑一条录制并导出")
    r.add_argument("video", nargs="?", help="输入视频文件")
    r.add_argument("--imu", default=None, help="IMU csv（不给则退化模式）")
    r.add_argument("--out", default="./cmci_out", help="输出目录")
    r.add_argument("--format", default="dataset,html",
                   help=f"导出格式，逗号分隔。可选 {sorted(_EXPORTERS)}")
    r.add_argument("--frontend", default="mobilenet_v3_small[:3]",
                   help="前端特征（默认 MobileNetV3 前 3 段）")
    r.add_argument("--fps", type=float, default=None,
                   help="覆盖源 fps（图像序列常误报）")
    r.add_argument("--imu-offset", type=float, default=0.0,
                   help="IMU 时间轴偏移（秒，shake-to-sync 求得）")
    r.add_argument("--crop", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="裁剪区域（0~1 的分数）")
    r.add_argument("--mirror", action="store_true",
                   help="视图水平镜像（会翻转 yaw）")
    r.add_argument("--max-frames", type=int, default=None,
                   help="只处理前 N 帧（调试）")
    r.add_argument("--no-imu", default="auto",
                   choices=("auto", "require", "assume_still"),
                   help="没 IMU 时怎么办：auto=视觉估计（默认）/ "
                        "require=报错拒绝 / assume_still=假设静止（仅限固定机位）")
    r.add_argument("--rule", default=None,
                   help="第 2 层业务规则 .py（见 --help-rule）")
    r.add_argument("--dry-run", action="store_true",
                   help="只打印将要做什么，不读写文件")
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("info", help="描述一个已有输出")
    i.add_argument("path")
    i.set_defaults(func=cmd_info)

    args = ap.parse_args(argv)
    if args.help_rule:
        print(RULE_HELP)
        return 0
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

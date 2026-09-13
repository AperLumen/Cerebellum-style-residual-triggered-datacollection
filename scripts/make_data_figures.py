#!/usr/bin/env python3
"""用 PIL 生成数据图（柱状图/折线图等）。

## 为什么数据图用 PIL，结构图用 TikZ

两类图的正确工具不同：

  * **结构图**（`figs/*.tikz` → TikZ）：方框、箭头、流程、映射。要的是精确
    排版和矢量输出，TikZ 是为此设计的。
  * **数据图**（本脚本 → PNG）：柱高、坐标轴、刻度。要的是**从数据算出来**
    的几何——硬编码进 .tikz 源码意味着改一次数据要手改一次坐标，而手改坐标
    必然出错（cost.tikz 的 log10 就是手算的，容易看错一位）。

所以数据图走这里：**读 runs/*.json → 算坐标 → 画**。数字改了重跑即可。

## 本机没有 matplotlib

项目跑在 `requirements.txt` 声明的依赖上，其中**没有 matplotlib**。为了画几张
柱状图引入 matplotlib 会让"零额外依赖"这条界限失守（它是 30+MB 的包，而这里
只需要画矩形和文字）。PIL **已**是依赖（`demo_video.py` 在用），足够了。

代价是坐标轴/刻度要自己算——所以下面 `_BarChart` 把这些收在一处。

## 输出约定（与 TikZ 图一致）

  * 白底不透明 + **无色彩描述**（sRGB）：GitHub 深色模式下透明底会让黑字消失，
    Display P3 描述会让浏览器偏色。两个坑在 build_figures.py 里都踩过。
  * 只写盘当像素真的变了：否则每次 `make check` 都留下幽灵改动。
  * PNG 用 2× 超采样再缩回——PIL 没有抗锯齿的矩形绘制，直接画会有锯齿边。

用法：
    PYTHONPATH=. .venv/bin/python scripts/make_data_figures.py
    PYTHONPATH=. .venv/bin/python scripts/make_data_figures.py --only bytes
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(ROOT, "figs")

SS = 2                       # 超采样倍数（抗锯齿）
PX_W, PX_H = 1800, 1000      # 最终像素尺寸

# ---- 配色：与 figs/_preamble.tex 的语义色一致（读者不用重新学一套）----
C_BASE = (150, 158, 170)     # 基线 / 被对比项（灰）
C_CAM = (48, 96, 168)        # 相机 / 前端（蓝）
C_PRED = (196, 124, 32)      # 预测（橙）
C_RES = (176, 60, 58)        # 残差（红）
C_STORE = (112, 72, 152)     # 存储（紫）
C_OK = (58, 138, 82)         # 成立 / 正结果（绿）——只用在 verdicts 的左栏
C_TXT = (38, 42, 48)
C_MUTE = (120, 128, 138)
C_GRID = (228, 231, 235)


def _font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    """找一个能显示中文的字体。**必须能显示中文**——否则会画成豆腐块。

    macOS 自带 PingFang / Hiragino；找不到就退到 Helvetica（英文图仍可用，
    中文会变方块，此时宁可报错也不要静默画错）。
    """
    candidates = [
        ("/System/Library/Fonts/PingFang.ttc", 0),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/System/Library/Fonts/STHeiti Light.ttc", 0),
        ("/Library/Fonts/Arial Unicode.ttf", 0),
    ]
    for path, idx in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=idx)
            except Exception:                                # noqa: BLE001
                continue
    raise RuntimeError(
        "找不到能显示中文的字体——换成英文图，或补一个字体路径。"
        "不要退到默认位图字体：那会把中文画成方块而没有任何报错。")


class Chart:
    """极简柱状图。PIL 没有绘图 DSL，所有几何都在这里算清楚。

    坐标约定：**数据坐标系**（x 向右、y 向上，y 单位是"柱的单位"）→
    `xy()` 换算到**像素坐标系**（y 向下）。
    """

    def __init__(self, w: int, h: int, pad: tuple[int, int, int, int]):
        self.W, self.H = w * SS, h * SS
        self.img = Image.new("RGB", (self.W, self.H), "white")
        self.d = ImageDraw.Draw(self.img)
        self.L, self.R, self.T, self.B = (p * SS for p in pad)

    # ---- 数据坐标 → 像素坐标 ----
    def set_range(self, ymax: float, n: int) -> None:
        self.ymax, self.n = ymax, n
        self.plot_w = self.W - self.L - self.R
        self.plot_h = self.H - self.T - self.B

    def xy(self, x: float, y: float) -> tuple[float, float]:
        return (self.L + x / self.n * self.plot_w,
                self.T + self.plot_h - y / self.ymax * self.plot_h)

    # ---- 图元 ----
    def hgrid(self, ticks: list[float], fmt: str = "{:g}") -> None:
        f = _font(int(13 * SS))
        for t in ticks:
            _, py = self.xy(0, t)
            self.d.line([(self.L, py), (self.W - self.R, py)],
                        fill=C_GRID, width=1 * SS)
            self.d.text((self.L - 8 * SS, py), fmt.format(t), font=f,
                        fill=C_MUTE, anchor="rm")

    def hline(self, y: float, color: tuple[int, int, int], width: int = 1,
              dashed: bool = False) -> None:
        _, py = self.xy(0, y)
        if not dashed:
            self.d.line([(self.L, py), (self.W - self.R, py)], fill=color,
                        width=width * SS)
            return
        x = self.L
        dash = 6 * SS
        while x < self.W - self.R:
            self.d.line([(x, py), (min(x + dash, self.W - self.R), py)],
                        fill=color, width=width * SS)
            x += dash * 2

    def marker(self, x: float, y: float, r: float,
               color: tuple[int, int, int]) -> None:
        cx, cy = self.xy(x, y)
        rr = r * SS
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color)

    def polyline(self, pts: list[tuple[float, float]],
                 color: tuple[int, int, int], width: int = 2) -> None:
        px = [self.xy(x, y) for x, y in pts]
        if len(px) >= 2:
            self.d.line(px, fill=color, width=width * SS, joint="curve")

    def baseline(self) -> None:
        _, py = self.xy(0, 0)
        self.d.line([(self.L, py), (self.W - self.R, py)],
                    fill=C_TXT, width=2 * SS)

    def bar(self, x: float, y: float, w: float, color: tuple[int, int, int],
            label: str = "", value: str = "", note: str = "") -> None:
        """画一根柱。label 在柱下，value 在柱内顶部，note 在柱内右上。

        **数值标签画在柱内**，不画在柱顶上方：最高的那根柱（f32，75264 B）
        顶到接近 ymax，标签画在上方就会超出画幅被切掉（实测 T0 贴边）。
        柱内空间充足（柱高几百像素），标签放里面既不会溢出、也更贴近数据。
        """
        x0, y0 = self.xy(x, 0)
        x1, y1 = self.xy(x + w, y)
        self.d.rectangle([x0, y1, x1, y0], fill=color)

        if value:
            fv = _font(int(15 * SS), bold=True)
            self.d.text(((x0 + x1) / 2, y1 + 9 * SS), value, font=fv,
                        fill="white", anchor="ma")
        if label:
            fl = _font(int(14 * SS))
            self.d.text(((x0 + x1) / 2, self.H - self.B + 8 * SS), label,
                        font=fl, fill=C_TXT, anchor="ma")
        if note:
            fn = _font(int(12 * SS))
            self.d.text((x1 - 6 * SS, y1 + 30 * SS), note, font=fn,
                        fill="white", anchor="ra")

    def text(self, x: float, y: float, s: str, size: float = 14,
             color: tuple[int, int, int] = C_TXT, bold: bool = False,
             anchor: str = "la", max_w: float | None = None) -> None:
        """画一段文字。``max_w``（数据/像素坐标，画幅内）给定时**自动换行**。

        **换行是必须的，不是便利**：PIL 的 ``draw.text`` 不换行，长句会直接
        画到画幅外面——而画幅外就是裁掉，产物看着只是"少了一截"（实测：
        底部说明换行后的第二行整行丢失，B0 贴边）。宁可这里多几行代码。
        """
        f = _font(int(size * SS), bold)
        if max_w is None:
            self.d.text((x * SS, y * SS), s, font=f, fill=color, anchor=anchor)
            return
        # 按**字符**断行（中文没有空格可断），宽度用 textlength 实量
        limit = max_w * SS
        lines, cur = [], ""
        for ch in s:
            if ch == "\n":
                lines.append(cur); cur = ""
                continue
            if self.d.textlength(cur + ch, font=f) > limit and cur:
                lines.append(cur); cur = ch
            else:
                cur += ch
        if cur:
            lines.append(cur)
        for i, ln in enumerate(lines):
            self.d.text((x * SS, (y + i * size * 1.55) * SS), ln, font=f,
                        fill=color, anchor=anchor)

    def save(self, path: str) -> str:
        # 用 Image.Resampling.LANCZOS 而不是 Image.LANCZOS：后者是 Pillow 9 之前
        # 的模块级别名，pyright 的 PIL 类型存根只认前者（报
        # "LANCZOS is not a known attribute of module PIL.Image"）。
        small = self.img.resize((self.W // SS, self.H // SS),
                                Image.Resampling.LANCZOS)
        # 白底 + 无色彩描述（见模块 docstring）
        if _same_pixels(path, small):
            return "unchanged"
        small.save(path, format="PNG", optimize=True)
        return "updated"


def _same_pixels(path: str, im: Image.Image) -> bool:
    """**看产物不看时间戳**：像素一致就不动文件。

    否则每次 `make check` 都会重写 PNG（PNG 编码器会写时间块），
    git 里就永远挂着一个假改动——GUIDE.pdf 当年正是这么踩的。
    """
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as old:
            if old.size != im.size or old.mode != im.mode:
                return False
            return list(old.getdata()) == list(im.getdata())
    except Exception:                                        # noqa: BLE001
        return False


# ------------------------------------------------------------------ 图 1：字节账

def figure_bytes() -> str:
    r"""记录格式 v2 的字节账（`runs/record_format_bench.json`）。

    这是本项目最该用柱状图讲的一件事：从 f32 到 keep_only 跨了 **4 个数量级**
    （75264 → 3386 B/帧）。文字里那句"9.0% of f16 全量"读者没有尺度感，
    而一根比一根矮一大截的柱子一眼就能看出量级差。

    **对数是必须的**：线性轴上 keep_only（3386）只有 f32（75264）的 4.5%，
    会被压成贴地的一条线，看不出还有差别。
    """
    d = json.load(open(os.path.join(RUNS, "record_format_bench.json"),
                       encoding="utf-8"))
    s = d["summary"]

    # 横轴按"表示"分组：左边是基线（已有做法），右边是本格式的三档
    items = [
        ("f32 z 图\n（内存里）", s["baseline_f32_zmap_B"], C_BASE, 0, "基线"),
        ("f16 z 图\n（落盘惯例）", s["baseline_f16_zmap_B"], C_BASE, 0, ""),
        ("f16 + zlib\n（只做熵编码）", s["fmt_f16_zlib_B"], C_BASE, 0, ""),
        ("int8\n（只做量化）", s["fmt_int8_only_B"], C_PRED, 1, ""),
        ("本格式 all", s["fmt_all_B"], C_CAM, 1, ""),
        ("本格式\nkeep_only", s["fmt_keep_B"], C_STORE, 1, "默认"),
        ("本格式\nnone", s["fmt_none_B"], C_STORE, 1, ""),
    ]
    n = len(items)
    # ymax 比最大值高一点：留出柱顶空隙，否则最高柱会碰画幅顶。
    ymax = 200000.0
    # **底部留白必须按"文字实际占几行"算，不能拍一个数**。
    # 曾经 pad 底部只给 30px、说明文本起点又定在 PX_H-52，于是换行后的第二行
    # 落到 y≈978–999（画幅正好 1000）——**整行被切掉**，而产物看起来正常。
    # 现在：底部 130px 归说明文字（两行 × ~26px + 边距），柱区从 1000-130 起算。
    CH_H = 1000
    BOTTOM = 130
    c = Chart(PX_W, CH_H, (95, 30, 0, BOTTOM))
    c.set_range(ymax, n)

    # 对数刻度：手算位置（PIL 没有对数轴）
    import math
    ticks = [10 ** k for k in range(0, 6)]
    f = _font(int(13 * SS))
    for t in ticks:
        _, py = c.xy(0, t)
        c.d.line([(c.L, py), (c.W - c.R, py)], fill=C_GRID, width=1 * SS)
        lab = f"{t/1000:g}k" if t >= 1000 else f"{int(t)}"
        c.d.text((c.L - 8 * SS, py), lab, font=f, fill=C_MUTE, anchor="rm")

    def logy(v: float) -> float:
        # 把值映射成"对数高度"（ymax 用 1e5 的 log 表示）
        return math.log10(max(v, 1.0)) / math.log10(ymax) * ymax

    c.d.line([(c.L, c.xy(0, 0)[1]), (c.W - c.R, c.xy(0, 0)[1])],
             fill=C_TXT, width=2 * SS)

    for i, (lab, val, col, _, note) in enumerate(items):
        c.bar(i + 0.18, logy(val), 0.64, col,
              label=lab, value=_fmt_bytes(val), note=note)

    # 说明条。**左边距不能是 0**：`text(anchor="la")` 是左上角对齐，传 x=0
    # 会直接贴到画幅左边缘——看着像被裁了。
    c.text(18, 8, "每帧字节数（对数轴）", size=17, bold=True)
    c.text(18, 34, f"来源 runs/record_format_bench.json（6 段 Open-AoE，"
                   f"每段 {d['n_frames_per_seq']} 帧，逐帧载荷口径）",
           size=12.5, color=C_MUTE, max_w=PX_W - 40)
    # 底部说明的 y 从**柱区之下**算起（不写绝对像素）：柱区底边 = CH_H - BOTTOM，
    # 留 14px 间距。这样改 BOTTOM 时文字跟着走，不会又被推出画幅。
    c.text(18, CH_H - BOTTOM + 14,
           f"收益来自「量化」不是熵编码：f16+zlib 几乎等于 f16（尾数是噪声）；"
           f"int8 单独就降到 {s['fmt_int8_only_B']:.0f} B。",
           size=13, color=C_MUTE, max_w=PX_W - 60)
    return c.save(os.path.join(FIGS, "bytes.png"))


def _load(name: str) -> dict:
    """读一个 runs/*.json。**缺文件时报错而不是返回空**：

    图和数字必须对得上。若这里悄悄返回 {}，图会画成一片空白或坐标轴错乱，
    而 `make check` 仍然通过——"图还在，只是内容没了"是最难发现的一类坏法。
    """
    path = os.path.join(RUNS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 不存在——图的数据源丢了。"
            f"先跑产生它的脚本（见 GUIDE 对应小节），不要静默画空图。")
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _fmt_bytes(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.1f}k"
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


# ------------------------------------------------------------------ 图 2：实验状态矩阵

def figure_verdicts() -> str:
    r"""§0.4 那 19 行实验的状态矩阵（✅/⚠️/❌/⬜）。

    这张图补的是一个**读者最容易误读的形状**：那 12 个 ❌ 让表看起来像
    "大部分都失败了"。把 19 行按"定义域内/外"重排后，形状立刻变了——
    **正结果都聚在定义域内，❌ 都聚在定义域外**。图的形状本身就是这个
    估计器适用范围的画像（§0.5 的那段论证其实在讲这件事）。
    """
    # (编号, 一句话, 状态, 是否在定义域内)
    rows = [
        (1,  "合成下游训练对比", "ok", True),
        (2,  "成本画像", "ok", True),
        (3,  "分辨率扫描", "ok", True),
        (4,  "光流上限与 RAFT 对照", "ok", True),
        (5,  "CPU/MPS 可移植性", "ok", True),
        (17, "AdaptiveTrigger 闸门锁死（缺陷修复）", "ok", True),
        (6,  "真实桌面触发", "warn", True),
        (7,  "高运动真实对", "warn", True),
        (14, "光流仲裁级联", "warn", True),
        (15, "iPhone 真 gyro 通路", "todo", True),
        (8,  "448px 真实交叉", "no", False),
        (9,  "平移建模（常深度）", "no", False),
        (10, "真实数据下游价值", "no", False),
        (11, "稀疏物理事件 regime", "no", False),
        (12, "z 图读出端升级", "no", False),
        (13, "行走级 gyro 价值", "no", False),
        (16, "推理期守门", "no", False),
        (18, "事件「时刻」信号（24 格）", "no", False),
        (19, "学习式上界（用标注）", "no", False),
    ]
    # 分两栏：左 = 定义域内，右 = 定义域外（这是图要传达的关键对比）
    inner = [r for r in rows if r[3]]
    outer = [r for r in rows if not r[3]]

    # ---- 画幅由**行数**算出来，不拍脑袋 ----
    # 曾把 CH 写死 1000、两栏各画各的，于是行数少的右栏（9 行 vs 左栏 10 行）
    # 底部空出 504px 宽的一条——看着像图没画完。
    # 现在：行高 × max(栏行数) + 表头 + 页脚 → 画幅高；两栏共用同一组 y 坐标，
    # 行数少的那栏底部自然留白（可接受），但**两栏顶部与行距完全对齐**。
    MARGIN, GUTTER, PANEL_W = 18, 26, 860
    PX_W_ = MARGIN * 2 + PANEL_W * 2 + GUTTER
    ROW_H = 40
    HEAD_H = 68          # 标题 + 副标题
    FOOT_H = 96          # 底部两行说明
    n_rows = max(len(inner), len(outer))
    CH = HEAD_H + n_rows * ROW_H + FOOT_H

    c = Chart(PX_W_, CH, (40, 30, 30, 30))
    c.img = Image.new("RGB", (PX_W_ * SS, CH * SS), "white")
    c.d = ImageDraw.Draw(c.img)

    COL = {"ok": C_OK, "warn": (196, 140, 40), "no": C_RES, "todo": C_BASE}
    MARK = {"ok": "成立", "warn": "部分", "no": "域外", "todo": "未测"}

    def panel(x0, title, items, subtitle, tcol):
        c.text(x0, 10, title, size=18, bold=True, color=tcol)
        c.text(x0, 38, subtitle, size=12.5, color=C_MUTE,
               max_w=PANEL_W - 20)
        y = HEAD_H
        for num, name, st, _ in items:
            h = ROW_H
            c.d.rectangle([x0 * SS, y * SS,
                           (x0 + PANEL_W - 20) * SS, (y + h - 7) * SS],
                          fill=(246, 247, 249))
            c.d.rectangle([x0 * SS, y * SS,
                           (x0 + 6) * SS, (y + h - 7) * SS],
                          fill=COL[st])
            c.text(x0 + 16, y + 8, f"{num:>2}", size=12.5, color=C_MUTE)
            c.text(x0 + 46, y + 6, name, size=14)
            c.text(x0 + PANEL_W - 30, y + 7, MARK[st], size=12.5,
                   color=COL[st], bold=True, anchor="ra")
            y += h
        return y

    panel(MARGIN, "定义域内：它擅长的事",
          inner, "运动去重 / 带宽预筛——这些主张站得住", C_OK)
    panel(MARGIN + PANEL_W + GUTTER, "定义域外：它不负责的事",
          outer, "语义重要性 / 事件时刻——这些是边界测量，不是失败",
          C_STORE)

    # 竖分隔线：终点必须**跟着行数算**。曾硬编码 830，而画幅高现在由行数
    # 决定（10 行 → 68+400+96=564），于是线画到了画幅外、看起来像底部贴边。
    #
    # 注：右栏比左栏少一行（9 vs 10），所以右栏底部会空出一条（实测 x1205-1709
    # 约 504px）。**这是内容的不对称，不是排版错误**——两栏的顶部与行距完全
    # 对齐，强行拉伸右栏行高会把两栏的行距搞得不一致，反而更乱。留白在此可接受。
    mid = MARGIN + PANEL_W + GUTTER / 2
    c.d.line([(mid * SS, 60 * SS),
              (mid * SS, (HEAD_H + n_rows * ROW_H) * SS)],
             fill=C_GRID, width=2 * SS)

    c.text(MARGIN, CH - FOOT_H + 20,
           "图的形状本身就是结论：正结果聚在左边，❌ 聚在右边——"
           "不是估计器弱，而是右栏那些问题不在它的定义域里。",
           size=13.5, color=C_TXT, max_w=PX_W_ - MARGIN * 2)
    c.text(MARGIN, CH - FOOT_H + 52,
           "来源 GUIDE.md §0.4（19 行实验总表）；详表与复现命令见 §2.3 各节。",
           size=12, color=C_MUTE, max_w=PX_W_ - MARGIN * 2)
    return c.save(os.path.join(FIGS, "verdicts.png"))


def figure_gate() -> str:
    r"""守门捕获率 vs 预算（`runs/gate_operating_curve.json`）。

    这张图要说清的是一件**反直觉**的事：把残差当"唤醒信号"用，**每一档预算
    都输给随机撒帧**。数字在表里是六行两位小数，在图上是两条一眼分开的曲线。

    三种语义画在一起，因为**它们的对比才是结论**：
      * **唤醒语义**（`wake_gate` vs `wake_rand`）：残差选帧打不赢随机；
      * **存储语义**（`clip_gate`，onset 判据）：≤5% 预算下**恒为 0.000**
        ——残差连事件的时间定位都不携带；
      * **oracle**：允许"先看答案再选帧"的上界，用来证明实验台有检出力，
        所以上面的"打不赢"不是台子坏了。
    """
    d = _load("gate_operating_curve.json")
    t = d["preds"]["shift"]["table"]
    n = len(t)
    budgets = [r["budget"] for r in t]
    wake_gate = [r["wake_gate_capture"] for r in t]
    wake_rand = [r["wake_rand_capture_mean"] for r in t]
    clip_gate = [r["clip_gate_capture"] for r in t]
    oracle = [r["oracle_capture"] for r in t]

    PX_W, CH_H = 1900, 1000
    BOTTOM = 170
    c = Chart(PX_W, CH_H, (110, 40, 0, BOTTOM))
    c.set_range(1.05, n)

    for k in range(0, 6):
        c.hgrid([k / 5], fmt="{:.1f}")

    baseline_py = c.xy(0, 0)[1]
    for i, b in enumerate(budgets):
        x, _ = c.xy(i + 0.5, 0)
        c.d.text((x, baseline_py + 52 * SS), _budget_lab(b),
                 font=_font(int(13 * SS)), fill=C_TXT, anchor="ma")

    c.baseline()

    # 唤醒语义：成对柱，直接比高矮（这是结论本身）
    for i in range(n):
        c.bar(i + 0.16, wake_gate[i], 0.32, C_RES,
              value=f"{wake_gate[i]:.2f}")
        c.bar(i + 0.50, wake_rand[i], 0.32, C_BASE,
              value=f"{wake_rand[i]:.2f}")

    # 存储语义（onset 判据）：≤5% 恒为 0，画成贴地小点
    for i in range(n):
        c.marker(i + 0.32, max(clip_gate[i], 0.012), 5, C_STORE)

    # oracle 上界参考线
    c.polyline([(i + 0.5, o) for i, o in enumerate(oracle)],
               (120, 160, 120), width=2)

    # 图例放**柱区内部右上**（不能在 CH_H*0.93——那是画幅坐标，会落到
    # 底部说明文字那一带，距底只剩 7px）。用柱区顶边往下 12px 起算。
    lx = 0.58 * PX_W
    ly = c.T / SS + 12
    c.text(lx, ly, "■ 唤醒·残差选帧", size=13, color=C_RES)
    c.text(lx, ly + 26, "■ 唤醒·随机撒帧（零假设）", size=13, color=C_BASE)
    c.text(lx, ly + 52, "● 存储·onset 捕获（≤5% 恒为 0）", size=13,
           color=C_STORE)
    c.text(lx, ly + 78, "— oracle 上界（允许看答案）", size=13,
           color=(104, 144, 104))

    c.text(18, 8, "守门捕获率 vs 帧预算（10 段事件数据）", size=17, bold=True)
    c.text(18, 34, "捕获率 = 被唤醒的帧命中事件区间的比例，越高越好。"
                   "随机撒帧是零假设：5% 的帧撒进 1s 区间期望覆盖已约 79%"
                   "（生日问题），所以它本来就很高。",
           size=12.5, color=C_MUTE, max_w=PX_W - 40)
    c.text(18, CH_H - BOTTOM + 74,
           "结论：唤醒语义下每一档预算残差都输给随机撒帧；存储语义下 ≤5% "
           "预算的 onset 捕获恒为 0.000——残差连事件的时间定位都不携带。",
           size=13, color=C_TXT, max_w=PX_W - 60)
    return c.save(os.path.join(FIGS, "gate.png"))


def _budget_lab(b: float) -> str:
    return f"{b*100:g}%"


def figure_zdist() -> str:
    r"""逐格 z 的累积占比（解释 deadband 为什么几乎无损）。

    分布绝大部分质量挤在低幅区，所以把低幅格**精确归零**几乎不丢信息，
    而 zlib 随后的收益正是来自这些被归零的"安静的格"。

    ## 为什么横轴用**分位数**而不是 σ

    σ 是**拟合出来的量**——它假定了一个宽度。而这组数据恰恰以"不服从那个
    宽度"为特征（重尾）。用 σ 当横轴等于把"分布是正态的"这个被证伪的前提
    画进了图上，读者会顺手把 σ 当成天然单位。

    分位数是**数据自身的性质**：第 p 分位就是"有 p 比例的格不超过它"，
    不依赖任何分布假设，换数据集也照样成立。

    换算关系（实测 6 段 Open-AoE）：<1σ 覆盖 70.2% ⟺ 该幅度落在 p70，
    依次 p91 / p97 / p99.5。图注里保留 σ 的对照，因为 GUIDE §2.3.3 与
    MEMORY_FORMAT §1 是用 σ 写的，读者需要能对上号。
    """
    # 四档来自 GUIDE §2.3.3 的实测（6 段 Open-AoE，robust 统计）
    # (分位标签, 累积比例, 等价的 σ 表述——只进图注，不画在图里)
    steps = [
        ("p70", 0.702, "1σ"),
        ("p91", 0.906, "2σ"),
        ("p97", 0.965, "3σ"),
        ("p99.5", 0.995, "6σ"),
    ]

    PX_W, CH_H = 1500, 900
    BOTTOM = 150
    c = Chart(PX_W, CH_H, (100, 40, 30, BOTTOM))
    n = len(steps)
    c.set_range(1.0, n)

    for k in range(0, 6):
        c.hgrid([k / 5], fmt="{:.1f}")

    for i, (lab, v, _) in enumerate(steps):
        c.bar(i + 0.2, v, 0.6, C_CAM, label=lab, value=f"{v*100:.1f}%")

    c.baseline()
    c.text(18, 8, "逐格 z 的累积占比（纵轴 = 落在该分位内的格的比例）",
           size=17, bold=True)
    c.text(18, 34, "横轴用分位数而不是 σ：σ 是拟合出来的宽度，而这组数据"
                   "恰恰以「不服从那个宽度」为特征（重尾）；分位数是数据自身的"
                   "性质，不依赖分布假设。",
           size=12.5, color=C_MUTE, max_w=PX_W - 40)
    c.text(18, CH_H - BOTTOM + 14,
           "读法：70% 的格落在 p70 之内（≈1σ），97% 落在 p97 之内（≈3σ）。"
           "所以 deadband 把低幅格精确归零几乎不丢信息——zlib 的收益正来自"
           "这些被归零的「安静的格」。尾部要一直拉到 p99.9 才收住"
           "（那一档 ≈9.5σ，在正态分布下极罕见），这就是阈值不能按正态设的原因。",
           size=13, color=C_MUTE, max_w=PX_W - 60)
    return c.save(os.path.join(FIGS, "zdist.png"))


def figure_resolution() -> str:
    r"""分辨率扫描：gyro 收益随特征图分辨率单调增长（`runs/flow_eval_res*.json`）。

    这张图是本项目**少数几个单调、可复现的正结果**之一：分辨率越高，自运动在
    像素上占的位移越大、越不该被当成"变化"，于是"减掉自运动"的收益越大。
    val 上 −41.2% → −52.2% → −57.7%，**单调未饱和**。

    用**相对 naive 的降幅**（%）而不是绝对残差：三档分辨率的特征图尺度不同，
    绝对残差不可直接比（这是本项目反复强调的口径问题）。

    同时画 flow 的降幅作为**质量上限**参照：gyro 与 flow 的差距随分辨率收窄
    （val 上 41.2→57.7 vs 51.6→69.1），说明 gyro 的模型误差在高分辨率下被
    稀释，而不是"gyro 变强了"。
    """
    res = (224, 336, 448)
    val_gain, flow_gain = [], []
    for r in res:
        d = _load(f"flow_eval_res{r}.json")
        q = d["quality"]["val"]
        na = q["naive"]["calm_resid"]
        val_gain.append((q["gyro"]["calm_resid"] - na) / na * -100)
        flow_gain.append((q["flow"]["calm_resid"] - na) / na * -100)

    PX_W, CH_H = 1600, 950
    BOTTOM = 160
    c = Chart(PX_W, CH_H, (110, 40, 30, BOTTOM))
    n = len(res)
    ymax = 80.0
    c.set_range(ymax, n)

    for k in range(0, 9):
        c.hgrid([k * 10], fmt="{:.0f}%")

    for i in range(n):
        c.bar(i + 0.16, val_gain[i], 0.34, C_RES,
              value=f"{val_gain[i]:.1f}%")
        c.bar(i + 0.54, flow_gain[i], 0.34, C_BASE,
              value=f"{flow_gain[i]:.1f}%")

    baseline_py = c.xy(0, 0)[1]
    for i, r in enumerate(res):
        x, _ = c.xy(i + 0.5, 0)
        c.d.text((x, baseline_py + 52 * SS),
                 f"{r}px\n特征图", font=_font(int(13 * SS)), fill=C_TXT,
                 anchor="ma", align="center")

    c.baseline()
    lx, ly = 0.52 * PX_W, c.T / SS + 12
    c.text(lx, ly, "■ gyro（IMU 副本，成本 0.03 ms）", size=13, color=C_RES)
    c.text(lx, ly + 26, "■ flow（DIS 光流，成本 3.9 ms）", size=13,
           color=C_BASE)

    c.text(18, 8, "分辨率扫描：残差相对 naive 的降幅（val 集，越高越好）",
           size=17, bold=True)
    c.text(18, 34, "画相对降幅而不是绝对残差：三档的特征图尺度不同，"
                   "绝对残差不可直接比。flow 是质量上限参照——"
                   "gyro 与它的差距随分辨率收窄，说明是模型误差被稀释，"
                   "而不是 gyro 本身变强了。",
           size=12.5, color=C_MUTE, max_w=PX_W - 40)
    c.text(18, CH_H - BOTTOM + 14,
           f"结论：gyro 收益随分辨率单调增长（{val_gain[0]:.1f}% → "
           f"{val_gain[1]:.1f}% → {val_gain[2]:.1f}%），未饱和——"
           f"这是「伴随放电的价值要看 regime」这一判断的正面证据。",
           size=13, color=C_TXT, max_w=PX_W - 60)
    return c.save(os.path.join(FIGS, "resolution.png"))


def figure_flowlimit() -> str:
    r"""光流质量上限：DIS vs RAFT（`runs/raft_eval_*.json`）。

    这张图要否掉一个很自然的质疑："你的光流基线是不是太弱了？换个 SOTA
    估计器说不定就有收益了。"——答案是**不会**：把 DIS 换成 RAFT（约 20 倍
    成本），残差几乎不动。

    所以"光流只能把残差降到这里"**不是估计器的问题**，而是这个任务在光流这
    条路线上的**上限**。这条结论支撑了 §1.1 的判断：瓶颈在模型类，不在估计精度。

    设计上**质量与成本必须分成两个面板**：合成一张"质量/成本"复合图会把两件
    事搅在一起，而它们恰恰是分开才有说服力（质量几乎一样，成本差 20 倍）。

    两组数据都画（合成 val / 真实高运动）：它们的**绝对残差差 2 倍**
    （0.106 vs 0.052），而结论完全一致——这正是"结论不依赖数据分布"的证据。
    """
    import statistics as st

    groups = [
        ("合成 val", "raft_eval_data_dataset_val.json"),
        ("真实高运动", "raft_eval_data_aoe_motion_val.json"),
    ]
    rows = []
    for label, fn in groups:
        d = _load(fn)
        rows.append((
            label,
            st.median([v["dis_flow"] for v in d.values()]),
            st.median([v["raft_flow"] for v in d.values()]),
            st.median([v["dis_ms"] for v in d.values()]),
            st.median([v["raft_ms"] for v in d.values()]),
        ))

    PX_W, CH_H = 1800, 880
    BOTTOM = 170
    c = Chart(PX_W, CH_H, (110, 40, 0, BOTTOM))

    # ---- 两个面板共享 x 轴：每组两根柱（DIS / RAFT）----
    # 左面板画质量，右面板画成本，中间用竖线隔开。
    # 用同一套 x 坐标（0..4），靠 y 轴的量纲不同来区分两件事。
    n = len(rows) * 2

    # 质量面板的 y 轴在左，成本面板的 y 轴需要另标——这里用**两个独立
    # 坐标系**太复杂，改成：只画质量，成本作为柱上标注。
    # 权衡：质量是"结论"，成本是"代价"，后者用一个数字就够，不必占一个面板。
    c.set_range(0.14, n)
    for k in range(0, 6):
        c.hgrid([k * 0.03], fmt="{:.2f}")

    for i, (label, dis, raft, dms, rms) in enumerate(rows):
        c.bar(i * 2 + 0.18, dis, 0.64, C_PRED, value=f"{dis:.4f}")
        c.bar(i * 2 + 1.18, raft, 0.64, C_CAM, value=f"{raft:.4f}")
        # 每对柱下方标"×N 成本"，把成本信息压进质量面板
        x, _ = c.xy(i * 2 + 1.0, 0)
        ratio = rms / dms
        c.d.text((x, (CH_H - BOTTOM + 52) * SS), f"RAFT 贵 {ratio:.0f}×",
                 font=_font(int(12 * SS)), fill=C_MUTE, anchor="ma")

    baseline_py = c.xy(0, 0)[1]
    for i, (label, *_rest) in enumerate(rows):
        x, _ = c.xy(i * 2 + 1.0, 0)
        c.d.text((x, baseline_py + 52 * SS), label,
                 font=_font(int(13 * SS)), fill=C_TXT, anchor="ma")

    c.baseline()
    lx, ly = 0.55 * PX_W, c.T / SS + 12
    c.text(lx, ly, "■ DIS（传统方法，约 6-7 ms）", size=13, color=C_PRED)
    c.text(lx, ly + 26, "■ RAFT（SOTA，约 140 ms）", size=13, color=C_CAM)

    c.text(18, 8, "光流质量上限：DIS vs RAFT（残差，越低越好）", size=17,
           bold=True)
    c.text(18, 34, "每对柱的差值 = 换成 SOTA 估计器的全部收益：合成 val 差 0.7%，"
                   "真实高运动差 4.3%——而成本差约 20 倍。"
                   "所以「光流只能把残差降到这里」不是估计器的问题，是这条路线的上限。",
           size=12.5, color=C_MUTE, max_w=PX_W - 40)
    c.text(18, CH_H - BOTTOM + 14,
           "两组数据的绝对残差差 2 倍（0.106 vs 0.052），而「换估计器没用」这条"
           "结论完全一致——这是结论不依赖数据分布的证据。",
           size=13, color=C_MUTE, max_w=PX_W - 60)
    return c.save(os.path.join(FIGS, "flowlimit.png"))


FIGURES = {"bytes": figure_bytes, "verdicts": figure_verdicts,
           "gate": figure_gate, "zdist": figure_zdist,
           "resolution": figure_resolution, "flowlimit": figure_flowlimit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(FIGURES),
                    help="只生成一张（调试用）")
    args = ap.parse_args()

    names = [args.only] if args.only else sorted(FIGURES)
    failed = []
    print(f"=== 生成 {len(names)} 张数据图 ===")
    for n in names:
        try:
            how = FIGURES[n]()
        except Exception as e:                               # noqa: BLE001
            print(f"  ✗ {n}: {type(e).__name__}: {e}")
            failed.append(n)
            continue
        with Image.open(os.path.join(FIGS, f"{n}.png")) as im:
            px = f"{im.size[0]}x{im.size[1]}"
        print(f"  ✓ {n}.png  {px}  {'（未变）' if how == 'unchanged' else '（已更新）'}")

    if failed:
        print(f"\n{len(failed)} 张失败: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\n全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

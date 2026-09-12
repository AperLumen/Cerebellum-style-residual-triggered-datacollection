#!/usr/bin/env python3
"""把 GUIDE.md 转成带书签目录、可点击跳转的 PDF。

为什么不用 pandoc：本机没有装，且 GUIDE 的 GFM 表格 + emoji 状态标记 + 行内
代码混合得很密（256 行表格、280+ 处标记），通用转换器对 emoji 的处理反而要
额外打补丁。这个脚本只针对 GUIDE.md 的**实际结构**做转换，范围可控。

生成的 PDF 有三条跳转路径：
  1. PDF 书签/大纲（阅读器的侧栏）——`\\section`/`\\subsection` 自动生成；
  2. 目录页（`\\tableofcontents`）——条目可点击；
  3. 文内的交叉引用（§x.y.z 会被自动变成指向对应小节的超链接）。

用法：
    PYTHONPATH=. .venv/bin/python scripts/build_guide_pdf.py
    # 默认 xelatex 编译两遍（第二遍为了让目录页码/书签收敛）
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "GUIDE.md")
TEX = os.path.join(ROOT, "guide.tex")
OUT = os.path.join(ROOT, "GUIDE.pdf")
XELATEX = "/Users/luyu/Library/TinyTeX/bin/universal-darwin/xelatex"

# emoji 状态标记 → 可排版的文本标记（ctex/xelatex 下 emoji 字体不可靠）
# 注意：**替换必须一次成型**。若 A 的替换串里含有 B 的源字符，B 的规则会
# 再命中一次，产出 `$\times$ $\times$` 这类重复（实测踩过）。因此：
#   1. 所有替换串都不含任何源字符（尤其不要用 ×/− 写 LaTeX）；
#   2. 用单遍扫描（dict.get）而不是循环 str.replace。
EMOJI_MAP = {
    "\u2705": r"{\bfseries [\u6210\u7acb]}",        # ✅
    "\u274c": r"{\bfseries [\u5426]}",              # ❌
    "\u26a0\ufe0f": r"{\bfseries [\u4fdd\u7559]}",  # ⚠️
    "\u26a0": r"{\bfseries [\u4fdd\u7559]}",         # ⚠
    "\u2b1c": r"{\bfseries [\u672a\u6d4b]}",        # ⬜
    "\u2713": r"$\checkmark$",                 # ✓
    "\u2717": r"$\times$",                   # ✗
    "\u2192": r"$\rightarrow$",              # →
    "\u2190": r"$\leftarrow$",               # ←
    "\u2194": r"$\leftrightarrow$",          # ↔
    "\u21d2": r"$\Rightarrow$",              # ⇒
    "\u2248": r"$\approx$",                  # ≈
    "\u00b1": r"$\pm$",                      # ±
    "\u00d7": r"$\times$",                   # ×
    "\u00b7": r"$\cdot$",                    # ·
    "\u2264": r"$\le$",                      # ≤
    "\u2265": r"$\ge$",                      # ≥
    "\u2260": r"$\neq$",                     # ≠
    "\u2208": r"$\in$",                      # ∈
    "\u221a": r"$\surd$",                    # √    "\u03c3": r"$\sigma$",                   # σ
    "\u03bc": r"$\mu$",                      # μ
    "\u03a3": r"$\Sigma$",                   # Σ
    "\u0394": r"$\Delta$",                   # Δ
    "\u03b8": r"$\theta$",                   # θ
    "\u03b1": r"$\alpha$",                   # α
    "\u03b2": r"$\beta$",                    # β
    "\u03c4": r"$\tau$",                     # τ
    "\u03bb": r"$\lambda$",                  # λ
    "\u03c0": r"$\pi$",                      # π
    "\u00b2": r"$^2$",                       # ²
    "\u00b3": r"$^3$",                       # ³
    "\u207b": r"$^-$",                       # ⁻
    "\u2212": r"$\text{-}$",                 # − U+2212
    "\u2044": "/",
}

# LaTeX 特殊字符，正文里必须转义
TEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# 长路径/长标识符在表格单元格里溢出严重：允许在这些位置断行
LONG_TOKEN = re.compile(
    r"(?<![{\\w])([A-Za-z_][\w./\\-]{12,})(?![\w}])")


def tex_escape(s: str) -> str:
    return "".join(TEX_ESCAPE.get(c, c) for c in s)


def apply_emoji(s: str) -> str:
    """单遍扫描替换 emoji 与数学符号。

    **必须单遍**：若用 `for k,v: s = s.replace(k,v)`，当某个替换串里含有
    另一个源字符时会被二次命中（`✗` 和 `×` 都映射到 `$\\times$`，循环
    写法会产出 `$\\times$ $\\times$`）。单遍扫描保证每个源字符只处理一次。
    """
    if not any(ord(c) > 0x2000 for c in s):
        return s
    out = []
    for c in s:
        out.append(EMOJI_MAP.get(c, c))
    return "".join(out)


def slugify_inline(s: str, in_table: bool = False) -> str:
    """把行内 Markdown 转成 LaTeX。

    Args:
        in_table: 处于 longtable 的 ``p{}`` 单元格内。此模式下**禁用**
            ``\\allowbreak`` 与嵌套的字号分组——``p{}`` 里 ``\\allowbreak``
            嵌套在 ``{\\bfseries ...}`` 内会让 LuaTeX/TeX 的对齐钩子出错
            （实测报 Missing } / Extra alignment tab）。表格里靠列宽而不是
            断行点来控制溢出。
    """
    # 占位符里的内容**已是 LaTeX**，必须整体保护起来，不能再被后面的
    # 转义或正则碰到。用「NUL 包裹序号」是因为它不可能出现在正文里。
    # 顺序很关键：**先抽出所有 LaTeX 片段 → 再对剩余纯文本转义**。
    # 反过来（先转义再套命令/链接）会把刚插入的命令也转义掉，
    # 产出字面的 "\textbackslash{}textbf\{"——实测踩过。
    holders: list[str] = []

    def stash_piece(latex: str) -> str:
        holders.append(latex)
        return f"\x00{len(holders) - 1}\x00"

    # 1) 行内代码
    def stash_code(m: re.Match) -> str:
        code = m.group(1)
        esc = tex_escape(code)
        if in_table:
            # 表格里不断行，也不加字号分组（避免与列宽钩子打架）
            return stash_piece(r"\texttt{" + esc + r"}")
        # 正文：在断行点前插哨兵再转义——顺序反了会把 \_ 拆成 \ + _，
        # 那个 _ 会逃出等宽盒子变成数学下标（实测踩过）。
        py = re.sub(r"([/_.\-])", lambda mm: mm.group(1) + "\x02", code)
        esc2 = tex_escape(py).replace("\x02", r"\allowbreak{}")
        return stash_piece(r"{\ttfamily\small " + esc2 + r"}")

    s = re.sub(r"`([^`]+)`", stash_code, s)

    # 2) 行内公式：原样保留
    s = re.sub(r"\$([^$\n]+)\$", lambda m: stash_piece(m.group(0)), s)

    # 3) 链接 [text](url)
    #    `\href{url}{text}` 的 **url 参数是 verbatim**（转义反而会进 URL），
    #    但 **text 参数是文本模式**，下划线/百分号必须转义——不然
    #    `runs/cascade_scan.json` 里的 `_` 会报 "Missing $"（实测踩过）。
    #    只跳过对 text 里已有占位符的转义。
    def stash_link(m: re.Match) -> str:
        text, url = m.group(1), m.group(2)
        if url.startswith("#"):
            return text
        # url：百分号是 URL 编码的一部分，不能转义成 \%
        url_l = url.replace("#", r"\#")
        parts = re.split(r"(\x00\d+\x00)", text)
        txt = "".join(p if re.fullmatch(r"\x00\d+\x00", p) else tex_escape(p)
                      for p in parts)
        return stash_piece(rf"\href{{{url_l}}}{{{txt}}}")

    s = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", stash_link, s)

    # 加粗 / 斜体。用 {\bfseries ...} 而非 \textbf{...}：\textbf 在
    #    longtable 单元格结尾会触发标点修正钩子 \check@nocorr@，报
    #    "Forbidden control sequence"（实测踩过）。
    #    **关键：内层文本必须先转义**。曾经直接 stash 原文，导致粗体里的
    #    `%` 未转义——`%` 会注释掉整行，报 "Missing } / Extra alignment
    #    tab"；粗体里的 `_` 未转义则会报 "Missing $"。两处都踩过。
    def wrap_bold(m: re.Match) -> str:
        inner = m.group(1)
        # 内层可能含占位符（行内代码/公式/链接）——它们已是 LaTeX，不能再转义。
        # 用占位符切分，只转义真文本部分。
        parts = re.split(r"(\x00\d+\x00)", inner)
        esc = "".join(p if re.fullmatch(r"\x00\d+\x00", p) else tex_escape(p)
                      for p in parts)
        return stash_piece(r"{\bfseries " + esc + "}")

    s = re.sub(r"\*\*([^*\x00]*(?:\x00\d+\x00[^*\x00]*)*)\*\*", wrap_bold, s)
    def wrap_italic(m: re.Match) -> str:
        inner = m.group(1)
        parts = re.split(r"(\x00\d+\x00)", inner)
        esc = "".join(p if re.fullmatch(r"\x00\d+\x00", p) else tex_escape(p)
                       for p in parts)
        return stash_piece(r"{\itshape " + esc + "}")

    s = re.sub(r"(?<![\w*])\*([^*\n\x00]+)\*(?![\w*])", wrap_italic, s)

    # 5) 剩余纯文本转义（占位符是 \x00N\x00，转义表不含这两者）
    s = tex_escape(s)

    # 6) emoji/符号替换（表格里也允许，但标记本身不再嵌套字号组）
    s = apply_emoji(s)

    # 7) 还原。**由内向外**：先把最深层的 stash 展开，再展开外层的，
    #    避免「外层占位符先展开、其内容又被误当成占位符」的问题。
    for _ in range(8):
        before = s
        for i in range(len(holders) - 1, -1, -1):
            s = s.replace(f"\x00{i}\x00", holders[i])
        if s == before:
            break

    return s


# --------------------------------------------------------------- 标题与锚点

# 小节号 → label，供 §x.y.z 交叉引用
SECTION_LABELS: dict[str, str] = {}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 形如 "## §4.4 输入契约：..." 或 "### 4.4.1 先量再改：..."
NUM_HEADING = re.compile(r"^§?\s*(\d+(?:\.\d+)*)")


def register_heading(level: int, text: str) -> str | None:
    """记录标题编号对应的 label，返回编号（用于交叉引用）。"""
    m = NUM_HEADING.match(text.strip())
    if not m:
        return None
    num = m.group(1)
    SECTION_LABELS.setdefault(num, f"sec:{num.replace('.', '_')}")
    return num


def render_heading(level: int, text: str) -> str:
    raw = text.strip()
    num = register_heading(level, raw)

    # **层级按编号深度决定，不按 Markdown 的 # 数量。**
    # GUIDE 里 §4（`##`）、§4.1（`###`）、§4.4（`##`）的 # 数量与语义层级
    # 不一致（§4.4 是 §4 的子节却写成二级）。若照搬 # 数量，§4.4 会和 §4
    # 平级，书签树就错了。这里统一以编号的段数（"4" / "4.1" / "4.4.1"）为准。
    if num:
        depth = num.count(".")          # 0 → 一级, 1 → 二级, 2 → 三级
        cmd = {0: "section", 1: "subsection", 2: "subsubsection"}.get(
            depth, "paragraph")
        body = slugify_inline(raw)
        label = SECTION_LABELS[num]
        return f"\\{cmd}{{{body}}}\\label{{{label}}}"

    body = slugify_inline(raw)
    # 无编号标题：按 # 数量给一个合理的层级（目录/说明性标题）
    cmd = {1: "section", 2: "subsection", 3: "subsubsection"}.get(
        level, "paragraph")
    return f"\\{cmd}*{{{body}}}"


def linkify_section_refs(s: str) -> str:
    """把正文里的 §x.y.z / §x.y 变成指向对应小节的超链接。"""
    def repl(m: re.Match) -> str:
        num = m.group(1)
        label = SECTION_LABELS.get(num)
        if label:
            return rf"\hyperref[{label}]{{§{num}}}"
        # 也试前缀（§5.2 可能只有 5.2.x）
        for k in sorted(SECTION_LABELS, key=len, reverse=True):
            if k.startswith(num + "."):
                return rf"\hyperref[{SECTION_LABELS[k]}]{{§{num}}}"
        return m.group(0)

    return re.sub(r"§\s?(\d+(?:\.\d+)*)", repl, s)


# ----------------------------------------------------------------- 表格

def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    # 处理转义竖线 \|
    cells, cur, i = [], "", 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if line[i] == "|":
            cells.append(cur)
            cur = ""
            i += 1
            continue
        cur += line[i]
        i += 1
    cells.append(cur)
    return [c.strip() for c in cells]


def is_sep_row(line: str) -> bool:
    return bool(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line)) and "-" in line


def render_table(rows: list[str]) -> str:
    header = split_row(rows[0])
    ncol = len(header)
    body = [split_row(r) for r in rows[2:] if not is_sep_row(r)]
    body = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol]
            for r in body]

    # 列宽按内容长度分配（长文本列给更多权重）
    weights = []
    for c in range(ncol):
        lens = [len(header[c])] + [len(r[c]) for r in body]
        # 用中位数而非最大值做"典型长度"，避免单个超长路径吃掉整列预算
        lens_sorted = sorted(lens)
        typical = lens_sorted[len(lens_sorted) // 2]
        weights.append(max(6, min(40, typical + max(lens) // 4)))

    # **扣掉 \tabcolsep 的固定开销**：n 列 = 2n 个 \tabcolsep，默认 6pt 每个。
    # 不扣的话每列宽度之和 + 表间距会超过 \textwidth，实测最大溢出 110pt。
    # 这里先估出可用于 p{} 宽度的净文本宽度（保守按 14pt/列 计）。
    tex_w = 597.0                       # a4paper 2.0cm 边距下的 \textwidth (pt)
    overhead = 2 * ncol * 6.0           # \tabcolsep
    usable = max(tex_w - overhead - 4.0, 0.5 * tex_w)
    total = sum(weights)
    # 归一化到可用宽度，单位换算成 \textwidth 的倍数
    widths = [max(0.045, w / total) for w in weights]
    s = sum(widths)
    widths = [w / s * (usable / tex_w) for w in widths]

    spec = "".join(
        rf">{{\raggedright\arraybackslash}}p{{{w:.4f}\textwidth}}"
        for w in widths)

    # 列间距收到 4pt，进一步争取宽度
    out = [r"{\setlength{\tabcolsep}{4pt}",
           r"\begin{longtable}{" + spec + "}", r"\toprule"]
    out.append(" & ".join(tex_escape(c) for c in header) + r" \\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    for r in body:
        cells = []
        for c in r:
            v = slugify_inline(c, in_table=True)
            cells.append(v.replace("\n", " ").strip())
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}}")
    return "\n".join(out)


# ----------------------------------------------------------------- 主转换

def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_code = False
    code_buf: list[str] = []
    in_math = False
    math_buf: list[str] = []
    table_buf: list[str] = []
    quote_buf: list[str] = []
    in_list = False
    list_depth = 0

    def flush_table():
        nonlocal table_buf
        if table_buf:
            out.append(render_table(table_buf))
            out.append("")
            table_buf = []

    def flush_quote():
        nonlocal quote_buf
        if quote_buf:
            inner = " ".join(slugify_inline(x) for x in quote_buf)
            inner = linkify_section_refs(inner)
            out.append(r"\begin{quote}\small " + inner + r"\end{quote}")
            out.append("")
            quote_buf = []

    def flush_list():
        nonlocal in_list, list_depth
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False
            list_depth = 0

    while i < len(lines):
        line = lines[i]

        # --- 代码块
        if line.startswith("```"):
            if not in_code:
                flush_table(); flush_quote(); flush_list()
                in_code = True
                code_buf = []
            else:
                in_code = False
                out.append(r"\begin{quote}\ttfamily\small")
                out.append(r"\begin{verbatim}")
                out.extend(code_buf)
                out.append(r"\end{verbatim}")
                out.append(r"\end{quote}")
                out.append("")
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # --- 行间公式
        if line.strip() == "$$":
            if not in_math:
                flush_table(); flush_quote(); flush_list()
                in_math = True
                math_buf = []
            else:
                in_math = False
                out.append(r"\begin{equation*}")
                out.extend(math_buf)
                out.append(r"\end{equation*}")
                out.append("")
            i += 1
            continue
        if in_math:
            math_buf.append(line)
            i += 1
            continue

        # --- 表格
        if line.lstrip().startswith("|"):
            flush_quote(); flush_list()
            table_buf.append(line)
            i += 1
            continue
        else:
            flush_table()

        # --- 引用块（> 或 HTML 备注）
        if line.startswith(">"):
            quote_buf.append(line.lstrip("> ").rstrip())
            i += 1
            continue
        else:
            flush_quote()

        # --- 分隔线
        if re.match(r"^\s*---+\s*$", line):
            flush_list()
            out.append(r"\medskip\hrule\medskip")
            out.append("")
            i += 1
            continue

        # --- 标题
        m = HEADING_RE.match(line)
        if m:
            flush_list()
            level = len(m.group(1))
            out.append("")
            out.append(render_heading(level, m.group(2)))
            out.append("")
            i += 1
            continue

        # --- 列表
        lm = re.match(r"^(\s*)([-*])\s+(.*)$", line)
        if lm:
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"\item " + slugify_inline(lm.group(3)))
            i += 1
            continue
        nm = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if nm:
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"\item " + slugify_inline(nm.group(3)))
            i += 1
            continue
        flush_list()

        # --- 空行
        if not line.strip():
            out.append("")
            i += 1
            continue

        # --- 普通段落
        s = slugify_inline(line)
        s = linkify_section_refs(s)
        out.append(s)
        out.append("")
        i += 1

    flush_table(); flush_quote(); flush_list()
    return "\n".join(out)


PREAMBLE = r"""% 由 scripts/build_guide_pdf.py 从 GUIDE.md 生成——不要直接编辑本文件。
% 编译:xelatex(GUIDE 含中文与行间公式)
\documentclass[UTF8,11pt,a4paper]{ctexart}

\usepackage[margin=2.0cm]{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}

% PDF 书签与可点击目录：这些是"快速跳转"的实现
\hypersetup{
  bookmarksnumbered=true,     % 书签带编号
  bookmarksopen=true,         % 打开时就展开
  bookmarksopenlevel=2,       % 展开到二级
  colorlinks=true,
  linkcolor=blue!55!black,
  urlcolor=teal!70!black,
  citecolor=blue!55!black,
  pdftitle={残差触发数据采集——项目指南},
  pdfauthor={},
  pdfsubject={CMCI / cerebellum_cancel 项目指南},
  pdfkeywords={residual trigger, corollary discharge, IMU, data acquisition}
}

% 表格里的长路径可以断行，避免溢出
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}

% 页码放页脚居中（不依赖 fancyhdr，TinyTeX 精简安装里没有该宏包）
\pagestyle{plain}

\begin{document}

% 用 ctexart + xelatex 时 hyperref 的书签对 \section 自动生效

\title{\bfseries 残差触发数据采集\\[0.3em]\large 从小脑式预测取消到一个采集协议层\\[0.6em]
\normalsize\mdseries 项目指南(GUIDE)}
\author{}
\date{2026 年 9 月}

\maketitle
\thispagestyle{empty}

\vspace{1em}
\noindent\rule{\textwidth}{0.4pt}
\small
\textbf{怎么用这份 PDF。} 左栏(PDF 书签/大纲)是全文结构,可逐级展开;下方目录条目、
以及正文里所有 \texttt{§x.y} 形式的引用都是\textbf{可点击}的,点击直接跳到对应小节;
正文里的 URL 与仓库路径也可点击。
\normalsize
\noindent\rule{\textwidth}{0.4pt}

\newpage
\tableofcontents
\newpage

"""


def main() -> int:
    if not os.path.exists(SRC):
        print(f"找不到 {SRC}", file=sys.stderr)
        return 1

    md = open(SRC, encoding="utf-8").read()

    # 去掉文档顶部的重复标题块（§0 之前的引文块保留）
    md = re.sub(r"^#{1}\s+.*?\n", "", md, count=1)

    body = convert(md)
    tex = PREAMBLE + body + "\n\\end{document}\n"

    # --- 自检：转换后不得残留未展开的 LaTeX 命令字面量 ---
    # 曾经因为「先转义再套命令」的顺序错误，产出
    # `\textbackslash{}textbf\{...\}` 这类字面量——排版能过，但正文里
    # 明晃晃一串反斜杠。这里在写盘前就拦住，不让它进 PDF。
    leaks = []
    for pat, why in [
        (r"\\textbackslash\{\}", "反斜杠被转义（命令顺序错）"),
        (r"\\textbackslash\{\}textbf", "\\textbf 被二次转义（经典顺序 bug）"),
        (r"\\textbackslash\{\}bfseries", "\\bfseries 被二次转义"),
        (r"\\textbackslash\{\}href", "\\href 被二次转义"),
        (r"\\textbackslash\{\}emph", "\\emph 被二次转义"),
        (r"[\x00\x01\x02]", "占位符未还原（stash 丢失）"),
    ]:
        n = len(re.findall(pat, tex))
        if n:
            leaks.append(f"  ✗ {why}: {n} 处（pattern={pat!r}）")
    # \textbf 用在 longtable 单元格里会触发标点修正钩子 -> \check@nocorr@
    # 报 "Forbidden control sequence"。转换器统一用 {\bfseries ...} 规避，
    # 这里确认没有 \textbf 漏进表格。
    blocks = re.findall(r"\\begin\{longtable\}.*?\\end\{longtable\}",
                        tex, flags=re.S)
    n_tb = sum(len(re.findall(r"\\textbf\{", b)) for b in blocks)
    if n_tb:
        leaks.append(f"  ✗ 表格里出现 \\textbf{{}} × {n_tb}（会触发 "
                     f"\\check@nocorr@；应用 {{\\bfseries}}）")
    if leaks:
        print("转换自检失败——未写盘：", file=sys.stderr)
        print("\n".join(leaks), file=sys.stderr)
        return 4

    with open(TEX, "w", encoding="utf-8") as fp:
        fp.write(tex)
    print(f"写入 {TEX} ({len(tex)} 字符)")

    xelatex = XELATEX if os.path.exists(XELATEX) else shutil.which("xelatex")
    if not xelatex:
        print("找不到 xelatex，只生成了 .tex", file=sys.stderr)
        return 2

    # **可复现构建**：PDF 内嵌 CreationDate/ModDate，默认取当前时间，于是每次
    # 重建都产生字节不同的产物——即使正文一字未改。实测：内容逐字相同
    # （47 页、extract_text 完全一致）而 git 仍报 GUIDE.pdf 已修改。
    # 由于 `make check` 每次都会重建，这会让工作区永远带着一个假改动，
    # 真正的文档变更反而淹没在噪声里。
    #
    # 解法是标准机制 SOURCE_DATE_EPOCH：TeX Live 与多数工具链都遵守它，
    # 用它固定时间戳。取一个固定值（本指南首次发布日）而不是"当前时间"，
    # 这样构建产物只取决于输入内容。
    env = dict(os.environ)
    env.setdefault("SOURCE_DATE_EPOCH", "1789257600")   # 2026-09-12 00:00 UTC
    env.setdefault("FORCE_SOURCE_DATE", "1")            # 让 TeX 真正采用该值

    # 两遍：第一遍收集目录/书签，第二遍写入页码
    for run in (1, 2):
        p = subprocess.run(
            [xelatex, "-interaction=nonstopmode", "-halt-on-error",
             os.path.basename(TEX)],
            cwd=ROOT, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            print(f"xelatex 第 {run} 遍失败", file=sys.stderr)
            tail = "\n".join(p.stdout.splitlines()[-40:])
            print(tail, file=sys.stderr)
            return 3
        print(f"xelatex 第 {run} 遍完成")

    built = os.path.join(ROOT, "guide.pdf")
    if os.path.exists(built):
        shutil.move(built, OUT)
        print(f"生成 {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

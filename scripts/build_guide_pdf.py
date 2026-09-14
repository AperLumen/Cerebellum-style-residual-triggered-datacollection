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

退出码（**每条对应一类"编译成功但产物是坏的"的静默失效**）：
    0  成功
    1  GUIDE.md 不存在
    2  找不到 xelatex（只生成了 .tex）
    3  xelatex 编译失败
    4  转换自检：表格里出现 ``\\textbf{}``（会触发 ``\\check@nocorr@``）
    5  转换自检：有符号没进 EMOJI_MAP（会按 CJK 字体渲染，与数学符号不匹配）
    6  排版自检：内容超出页边距（Overfull \\hbox）
    7  排版自检：字符在 PDF 里没有出来（xelatex 静默跳过）
    8  排版自检：PDF 书签里有丢失字符的痕迹（连续空白 = token 被删）
    9  排版自检：PDF 元数据里的下划线被吃掉（``cerebellum_cancel`` →
       ``cerebellumcancel``）

后四条都要求**先编译出 PDF 再回读产物**（书签、元数据、Overfull、Missing
character 都是 TeX 排版决定的，转换器只能猜）。这是刻意的：本文件遭遇过的
每一类 bug，共同点都是 **xelatex 退出码 0 而 PDF 是坏的**，只有检查产物才能
发现。详见 GUIDE §4.8。
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

# 图：Markdown 里写标准图片语法 `![图 N：说明](figs/name.png)`。
#   * GitHub / 编辑器：直接显示同名 .png（位图），无需任何额外约定；
#   * PDF：**结构图**换成 `figs/name.pdf`（矢量）；**数据图**没有 .pdf
#     （PIL 生成，见 scripts/make_data_figures.py），直接用 PNG。
# 两种都支持，判据是"同名 .pdf 是否存在"——这样加图时不需要在两处登记。
IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\((figs/[A-Za-z0-9_-]+)\.png\)\s*$")

# 图缩放上限：宽图按版心缩，高图按高限缩（矢量缩放不损质量）。
#
# **高度上限从 0.82 降到 0.70**（2026-09-13，配合选择性分栏）。
# 分栏后图落在**栏内**，而栏比整页窄得多；`mapping.pdf` 是竖长图，
# 按宽缩完仍然很高，加上图注就顶出页面——实测图注最后一行 y=842.4，
# 而页高 841.9，**超出 0.5pt**（文字被切掉一条边，非常难发现）。
# 降到 0.70 后图注完整落在页内。
# 单栏时 0.82 没问题——这个值是为分栏调的，改回来要先确认图不越界。
FIG_WIDTH = r"\linewidth"
FIG_HEIGHT = r"0.70\textheight"

# emoji 状态标记 → 可排版的文本标记（ctex/xelatex 下 emoji 字体不可靠）
# 注意：**替换必须一次成型**。若 A 的替换串里含有 B 的源字符，B 的规则会
# 再命中一次，产出 `$\times$ $\times$` 这类重复（实测踩过）。因此：
#   1. 所有替换串都不含任何源字符（尤其不要用 ×/− 写 LaTeX）；
#   2. 用多字符键优先 + 单遍扫描（dict.get）而不是循环 str.replace。
#
# **中文标记不能写成 r"...\uXXXX..."**：原始字符串不解析 \u 转义，于是
# TeX 收到的是字面 `\u6210\u7acb`，而 `\u` 在 LaTeX 里是**短音符重音命令**
# （breve accent）——读者看到的是"给数字 6 加一个 ˇ 帽"这种垃圾（PDF 抽文
# 实测 `6̆2107̆acb`，15 处）。整个 §0.4 实验总表的状态列都是这样，而成因
# 是四个字符的 `r` 前缀。**编译成功、退出码 0、每项自动化检查全绿**。
# 所以这里的值必须是真汉字（不加 r 前缀），只对含反斜杠的 LaTeX 命令用 r。
EMOJI_MAP = {
    # ---- 状态列（§0.4 实验总表）----------------------------------
    # **这组词不能读成"成功 / 失败"。** ❌ 原本映射到「否」，于是在 PDF 里
    # 印出一个"否"字——而 GUIDE §0.5 花了整整两段说明"那些 ❌ **不是**这个
    # 方法失败了"。表格与正文自相矛盾（读者看到 `[否]` 就会那样读）。
    # 现在改成中性的领域词：**成立 / 部分 / 域外 / 未测**，含义写在图例里。
    "\u2705": "{\\bfseries [成立]}",        # ✅
    "\u274c": "{\\bfseries [域外]}",        # ❌
    "\u26a0\ufe0f": "{\\bfseries [部分]}",  # ⚠️
    "\u26a0": "{\\bfseries [部分]}",        # ⚠
    "\u2b1c": "{\\bfseries [未测]}",        # ⬜
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
    "\u2261": r"$\equiv$",                   # ≡
    "\u2208": r"$\in$",                      # ∈
    # √ 与 σ 之间**必须换行**：曾被写成同一行，σ 的映射被行尾注释吞掉，
    # 于是它在 guide.tex 里以裸 Unicode 进入正文——ctex 会把它交给 CJK
    # 后备字体渲染，字号/基线/字宽都按汉字来，与邻接的 $\mu$ 不匹配
    # （读者看到的是\sigma与\mu大小不一致）。10 处受影响，肉眼很容易
    # 归因成"字体就是这样"，实际是这行折行事故。
    "\u221a": r"$\surd$",                    # √
    "\u03c3": r"$\sigma$",                   # σ
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

    # ---- 2026-09-13 补：圈号 / 星号 / 制表符号 ----------------
    # **这一组是被"静默丢字"抓出来的**（用户报"有的字没有出来"）。
    # 判据是 `guide.log` 里的 `Missing character: There is no X`——xelatex
    # **不报错、不退码、PDF 照生**，只是那个字符的位置**什么都没有**。
    # 实测全文 71 处，其中圈号 ①–⑦ 占 55 处（全部来自 §4.8.7 的七个小节
    # 标题与 §4.7.1.1 的列表），读者看到的是空缺。
    #
    # 教训：**新增没映射过的字符时，"编译成功"给不出任何信号。**
    # 唯一可靠的判据是扫 log 里的 Missing character（`build_guide_pdf.py`
    # 的排版自检已包含这条）。
    "\u2460": "(1)",                         # ①
    "\u2461": "(2)",                         # ②
    "\u2462": "(3)",                         # ③
    "\u2463": "(4)",                         # ④
    "\u2464": "(5)",                         # ⑤
    "\u2465": "(6)",                         # ⑥
    "\u2466": "(7)",                         # ⑦
    "\u2467": "(8)",                         # ⑧
    "\u2468": "(9)",                         # ⑨
    "\u2469": "(10)",                        # ⑩
    "\u2605": r"$\star$",                    # ★
    "\u27e8": r"$\langle$",                  # ⟨
    "\u27e9": r"$\rangle$",                  # ⟩
    # 制表/框线类只出现在代码块里（verbatim 原样输出，不受本表影响），
    # 但表格或正文里可能用到，给一个等宽的近似写法。
    "\u2500": "---",                         # ─
    "\u251c": "|--",                         # ├
    "\u2514": "`--",                         # └
}

# EMOJI_MAP 里的多字符键，**必须先于逐字符扫描处理**（按长度降序，先长后短）。
# 存在的理由：`⚠️` 是"基字符 U+26A0 + 变体选择符 U+FE0F"两个码位，逐字符
# 扫描时它永远匹配不上整键，只会命中单字符的 `⚠`，把 `U+FE0F` 留在原地——
# 那是个 TeX 排不出的字符，实测撑爆表格列（26 处 Overfull）而编译仍成功。
_MULTI_CHAR_KEYS = sorted((k for k in EMOJI_MAP if len(k) > 1),
                          key=len, reverse=True)

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
    # `<` 与 `>` **在 LaTeX 里不是普通字符**：文本模式下它们渲染成西班牙语的
    # 倒置标点 `¡` / `¿`，字宽也不同。曾经漏了这两条，于是 `rule:<name>`
    # 这类字面量在 PDF 里显示成 `rule:¡name¿`——而编译完全成功。
    # 用数学模式的 \textless/\textgreater：字宽贴近原字符，且在任何字体下
    # 都有定义（直接用 `<` 在数学模式下会变成关系符，两侧多出间距）。
    "<": r"\textless{}",
    ">": r"\textgreater{}",
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

    **快速路径的判据必须是"该字符是否是 EMOJI_MAP 的键"，不能是"码位是否
    大于 U+2000"**。后者曾让一整类符号静默漏替换：`±`(U+00B1)、`×`(U+00D7)、
    `²`(U+00B2)、`·`(U+00B7)、`−`(U+2212)——它们全在 U+2000 以下，于是
    **只要某个字符串里没有汉字（例如纯数字表格单元格 "0.162±0.077"），整串
    的符号替换就被跳过**。后果同样是静默降级：漏掉的字符交给 CJK 后备字体
    渲染，字号与邻接的 `$\\mu$` 不一致，而编译永远成功。
    实测该 bug 让 GUIDE.md 里约 100 处符号未被替换。
    """
    if not any(c in EMOJI_MAP for c in s):
        return s
    # **先处理多字符键，再做逐字符替换**。逐字符扫描永远匹配不上 `⚠️`
    # 这类"基字符 + 变体选择符 U+FE0F"的组合键，于是它会被拆成
    # `⚠`（命中单字符键）+ 一个**孤立的 U+FE0F**。后者在 TeX 里是一个
    # 无法排版的字符，实测把 §0.4 状态列撑爆（26 处 Overfull，最大 28pt），
    # 而 xelatex 退出码仍是 0——`make check` 全绿。
    #
    # 顺序不能反：先逐字符扫的话，`⚠` 已被替换成 `{\bfseries [保留]}`，
    # 后面的 `\ufe0f` 就再也找不到配对的基字符了。
    for k in _MULTI_CHAR_KEYS:
        if k in s:
            s = s.replace(k, EMOJI_MAP[k])
    out = []
    for c in s:
        out.append(EMOJI_MAP.get(c, c))
    return "".join(out)


def _add_breakpoints(text: str, symbols: bool, reopen: str = "") -> str:
    r"""转义文本，插入可断点，按需替换符号。返回的**已是 LaTeX**。

    三类断点，缺一不可：

    1. **符号后**（``/ _ . -``）：``drop/throw/toss/break``、``runs/x.json``。
    2. **驼峰边界**：``InsufficientDataError`` → ``Insufficient|Data|Error``。
       只加第 1 类时，一个 21 字符的驼峰标识符仍是**不可断的长词**，实测把
       整行推出页边距 11.7pt（它所在的列根本装不下 120pt 的词）。
    3. **下标 `[` 前**：``features[:3]``。

    只在"下一个小写字母跟在大写字母之后"处断（`Data|Error`），不在 ALLCAPS
    中间断（`R|ANSAC` 会很难看）——准确说：`小写→大写` 与 `大写→大写后跟小写`
    （`HTTPServer` → `HTTP|Server`）两处。

    Args:
        symbols: 是否跑 `apply_emoji`。
            * 粗体标识符（表格里，如 `selection_reason`）→ **True**，它周围
              可能有 `★`/`±` 之类；
            * 行内代码 `` `x` `` → **False**，代码里的字符是**故意字面**的，
              替换成 `$\times$` 会把等宽文本拆断，且 `_`
              会被当成数学下标。
        reopen: 断点处要**重开**的分组（默认空 = 旧行为，只插 `\allowbreak`）。
            行内代码传 ``"{\\ttfamily\\small "``——**开括号要算进去**，
            因为断点会先把它前面那个分组闭合掉。理由见 return 处。
    """
    # 先插哨兵，再整体转义，最后把哨兵换成 \allowbreak
    # —— 顺序反了会把 `\_` 拆成 `\` + `_`，那个 `_` 会变成数学下标（踩过）。
    #
    # 用 lambda 而不是 `r"\1\x02"` 作替换串：`re.sub` 的替换串会把
    # `\x02` 当成**正则转义**去解析，直接抛 `bad escape \x`（实测）。
    def mark(m: re.Match) -> str:
        return m.group(1) + "\x02"

    marked = re.sub(r"([/_.\-])", mark, text)
    marked = re.sub(r"([a-z0-9])(?=[A-Z][a-z])", mark, marked)
    marked = re.sub(r"([A-Z])(?=[A-Z][a-z])", mark, marked)
    # **反斜杠也是断点**（2026-09-13 加，配合选择性分栏）。
    #
    # 病因：分栏后正文栏宽减半（483.7pt → 236.9pt），放不下的长 token 就从
    # "勉强"变成"确定溢出"。GUIDE 里有一处行内代码是**两个 LaTeX 控制序列**：
    #     `\phantomsection\addcontentsline{toc}{...}`
    # 它渲染出来 **41 个等宽字符**，而半栏刚好只放得下 41 个——**一个字都
    # 断不开**，于是整行溢出 35.2pt，文字被切出页面。
    # 它内部本来就有天然断点（两个控制序列之间、`{` 之前），只是旧规则
    # 只认 `/ _ . -` 与驼峰，**不认反斜杠**。
    #
    # 只在"反斜杠后面跟字母"处断（即控制序列的**开始**），不在别处断——
    # 这样 `\foo\bar` 会在 `\bar` 前面断开，而 `\%` 这类单字符命令不受影响。
    # 注意用**捕获组**而不是 lookbehind：`mark()` 要读 `m.group(1)`
    # （第一版写成 lookbehind，直接 `IndexError: no such group`）。
    marked = re.sub(r"([A-Za-z0-9{}])(?=\\[A-Za-z])", mark, marked)
    marked = marked.replace("[", "\x02[")
    out = tex_escape(marked)
    if symbols:
        # 在 tex_escape **之后**跑：apply_emoji 产出的是 LaTeX 命令
        # （`$\times$` 里有 `$` 和 `\`），先跑会被转义坏掉。
        out = apply_emoji(out)
    # **断点必须"闭合盒子 + 重开盒子"**（2026-09-13 修，配合选择性分栏）。
    #
    # `\allowbreak{}` 单独放着**不管用**：它前面的 `\textbackslash{}` 是个盒子，
    # 盒子与前后字符之间没有胶（glue），TeX 在那里不能断行。实测四种写法，
    # 只有"闭合分组 + `\allowbreak` + 重开同名分组"能让长控制序列在窄栏里折行。
    #
    # ⚠️ **两个坑都踩过，注释留在这里**：
    #
    # 1. 只写 `}\allowbreak{}{\ttfamily\small `（少一个开括号）→ 整个文档
    #    当场 `! Extra }, or forgotten \endgroup.`，PDF 只出 2 页。
    # 2. 把 `}` 写死在替换串里（不判断 reopen 是否为空）→ **表格/粗体那条
    #    路径**也跟着吐 `}`，正文里出现 `13.}\allowbreak{}7\%`，
    #    同样报 Extra }。这两条路径共用本函数，改这里必须两条都验。
    # **断点不能解决一切**（2026-09-13 实测）：`\textbackslash{}` 是个**定宽
    # 盒子**，盒子两侧没有胶（glue），所以"盒子 + 断点 + 盒子"仍然断不开——
    # 实测 `\allowbreak{}` / `\hspace{0pt}` / `\linebreak[0]` 三种分隔符
    # 在该处**都无效**（`\char92` 能断但渲染成组合重音，不能用）。
    #
    # 但这类"长控制序列"的溢出**本身就是碰运气**：同一段标记，前缀空时 0 溢出、
    # 前缀是"未编号标题补上 "时溢出、前缀更长时又 0 溢出——因为 TeX 只是
    # 在凑每行的断点。所以对策不是继续调 separator，而是让整段的自然断点变多。
    #
    # 这里保留 `reopen`（它对**非盒子**的断点确实有效，比如 `runs/x.json`
    # 里的 `/`），并给文档加 `\emergencystretch` 兜底（见 PREAMBLE）。
    if not reopen:
        return out.replace("\x02", r"\allowbreak{}")
    return out.replace("\x02", "}\\allowbreak{}" + reopen)


_INLINE_CODE_OPEN = r"{\ttfamily\small "


def _tt_breakable(code: str) -> str:
    """行内代码的断点版本（**不**替换符号——代码要保持字面）。

    但等宽字体（lmmono）里**没有**的字形必须换成 ASCII 等价写法，
    否则 xelatex **静默丢字**（退出码 0、PDF 该位置空白）。详见
    ``VERBATIM_FALLBACK``：实测 ``raw_ratio ≈ 1`` 这类行内代码里的
    ``≈`` 就是这种情形——正文别的 ``≈`` 走 ``$\\approx$`` 没问题，
    但行内代码刻意不替换符号，于是撞上字体缺字。
    """
    return _add_breakpoints(_mono_safe(code), symbols=False,
                            reopen=_INLINE_CODE_OPEN)


def _mono_safe(text: str) -> str:
    """把等宽字体排不出的字符换成 ASCII 等价写法（只用于排版层）。

    注意 ``~`` 在 LaTeX 文本模式下会变成不换行空格（甚至成了重音符），
    所以 ``≈ -> ~=`` 里的 `~` 会被 ``tex_escape`` 转成 ``\textasciitilde{}``
    ——那是**正确**行为（读者看到 ``~=``），不是问题。

    **多字符键要先处理**（与 `apply_emoji` / `pdf_bookmark_text` 同理）：
    ``⚠️`` 是"基字符 U+26A0 + 变体选择符 U+FE0F"，逐字符扫描只会命中
    ``⚠``，把 U+FE0F 留在原地——而那是个排不出的字符，照样静默丢。
    """
    for k in _MULTI_CHAR_KEYS:
        if k in text:
            text = text.replace(k, VERBATIM_FALLBACK.get(k, k))
    return "".join(VERBATIM_FALLBACK.get(c, c) for c in text)


# 等宽字体（lmmono）里**没有**的字形 -> ASCII 替代。
# 这些只用于**排版层**（verbatim 代码块与行内代码），因为 `EMOJI_MAP` 的数学
# 替换（`$\approx$`）在 verbatim 里会被原样打印成 LaTeX 源码，所以不能复用。
#
# 为什么需要它（2026-09-13，用户报"有的字没有出来"）：xelatex 遇到字体里
# 没有的字符会**静默跳过**——不报错、退出码 0、PDF 照生，那个位置就是空的。
# 实测全文 71 处，其中代码块里的框线字符（`├ ─ └`）与数学符号（`≈ ⇒ ≡`）
# 占 19 处。README 里渲染正常（GitHub 有完整字体），只有 PDF 缺字。
#
# **这份表的依据是实测，不是推测**（2026-09-13 补全）。方法：把 `EMOJI_MAP`
# 的全部非 ASCII 键逐个用 `{\ttfamily\small X}` 包起来编译一次，扫 log 里的
# `Missing character`。结果是 52 个键里 **40 个在等宽字体里没有字形**，
# 其中本表原先只覆盖了 12 个——**另外 28 个是"只要有人把它们写进行内代码就
# 会静默丢字"的定时炸弹**（写这一节时就当场踩到：`① ② ③` 包在反引号里，
# 自检 4 立刻报 3 处丢字）。
#
# 实测排不出的 40 个：αβθλμπστ⁻↔⇒∈≈≠≡≤≥①②③④⑤⑥⑦⑧⑨⑩─└├★⚠✅✓✗❌⟨⟩⬜
# 实测**能**排出的 12 个：±²³·×ΔΣ⁄←→−√
#
# 那 12 个里，本表保留了 4 个（`± × ← →`）没用到的条目。这是**刻意的保守
# 选择**：它们换成 ASCII 也语义正确（`+/-`、`x`、`<-`、`->`），而如果将来
# 字体配置变了，多留一条不会造成丢字，少留一条会。
VERBATIM_FALLBACK = {
    "\u2500": "-",      # ─
    "\u2502": "|",      # │
    "\u251c": "|",      # ├
    "\u2514": "`",      # └
    "\u251d": "|",      # ┝
    "\u253c": "+",      # ┼
    "\u2248": "~=",     # ≈
    "\u21d2": "=>",     # ⇒
    "\u2261": "==",     # ≡
    "\u2260": "!=",     # ≠
    "\u2264": "<=",     # ≤
    "\u2265": ">=",     # ≥
    "\u27e8": "<",      # ⟨
    "\u27e9": ">",      # ⟩
    "\u2192": "->",     # →
    "\u2190": "<-",     # ←
    "\u00b1": "+/-",    # ±
    "\u00d7": "x",      # ×
    # ---- 2026-09-13 按实测补全的 28 条 --------------------------
    # 这 28 个原先只在 EMOJI_MAP 里（那走的是数学模式，正文没问题），
    # 一旦出现在**行内代码或代码块**里就会静默丢字。
    "\u2194": "<->",    # ↔
    "\u2208": "in",     # ∈
    "\u221a": "sqrt",   # √
    "\u03b1": "alpha",  # α
    "\u03b2": "beta",   # β
    "\u03b8": "theta",  # θ
    "\u03bb": "lambda",  # λ
    "\u03bc": "mu",     # μ
    "\u03c0": "pi",     # π
    "\u03c3": "sigma",  # σ
    "\u03c4": "tau",    # τ
    "\u207b": "^-",     # ⁻
    "\u2460": "(1)",    # ①
    "\u2461": "(2)",    # ②
    "\u2462": "(3)",    # ③
    "\u2463": "(4)",    # ④
    "\u2464": "(5)",    # ⑤
    "\u2465": "(6)",    # ⑥
    "\u2466": "(7)",    # ⑦
    "\u2467": "(8)",    # ⑧
    "\u2468": "(9)",    # ⑨
    "\u2469": "(10)",   # ⑩
    "\u2605": "*",      # ★
    "\u26a0\ufe0f": "!",    # ⚠️（多字符键：基字符 + 变体选择符）
    "\u26a0": "!",      # ⚠
    "\u2705": "[OK]",   # ✅
    "\u2713": "v",      # ✓
    "\u2717": "x",      # ✗
    "\u274c": "[X]",    # ❌
    "\u2b1c": "[ ]",    # ⬜
}


def _verbatim_safe(lines: list[str]) -> list[str]:
    """把代码块里等宽字体排不出的字符换成 ASCII 等价写法。

    **只改排版产物，不改 GUIDE.md**：源文件里保留原字符（GitHub 与编辑器
    显示正常，且那才是作者写的），只有进 PDF 的这一份降级。
    判据来自 `guide.log` 的 `Missing character`（见 main 的自检 4）。
    """
    return ["".join(VERBATIM_FALLBACK.get(c, c) for c in ln) for ln in lines]


def _is_block_start(line: str) -> bool:
    """这一行是否**必须**独占处理（不得与上一行合并）。"""
    s = line.strip()
    if not s:
        return True
    if s.startswith(("#", "|", ">", "```", "![")):
        return True
    if s in ("$$",):
        return True
    if s.startswith(("- ", "* ", "+ ")):
        return True
    if re.match(r"^\d+\.\s", s):
        return True
    if re.match(r"^-{3,}$", s):
        return True
    return False


def _is_list_item(line: str) -> bool:
    s = line.strip()
    return bool(s.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", s))


def join_soft_wraps(md: str) -> str:
    """把 Markdown 的**软换行**合并成一行（否则每一行会变成独立段落）。

    ## 为什么必须做这一步

    本仓库的 GUIDE.md 是按 ~72 字符**硬折行**写的（编辑器折行习惯），
    但 Markdown 的语义是：**单个换行 = 软换行，会合并成一个段落**。
    GitHub 上显示正常，而本转换器（`convert`）是**逐行**处理的——每行都
    `out.append(s)` + `out.append("")`，于是：

    | 症状 | 实测 |
    | --- | --- |
    | 段落被拆碎 | 2218 行 → PDF 里 2218 个段落；其中 619 行本应属于 **201** 个段落 |
    | 跨行 `**粗体**` 匹配不上 | **39 处**（`**` 分别落在两行，正则 `[^*]+` 跨不过换行）→ **88 个字面星号印进 PDF，分布在 26/68 页** |
    | 列表续行逃出 `itemize` | **52 处**（续行不再是 `\\item`，被当成普通段落，`\\end{itemize}` 把它切在外面） |

    第一个症状是"每行太短、总在分行"的观感问题；**第二个是真正的排版事故**
    ——读者会在正文里看到 `**` 星号。三个症状同一个根因，所以在这一层修。

    ## 合并规则

    * 连续的非空、非块起始行 → 用**一个空格**连接（中文之间不需要空格，但
      源文件里行尾是标点、行首是汉字的情况占多数；加空格不会显示成空隙，
      TeX 会吃掉行首空白）。
    * **列表项**：其后的缩进续行合并到同一个 `\\item` 里（这些行本来就不会
      被识别为 item，所以此前直接漏出列表）。
    * **引用块**：`>` 后的内容先脱去 `> ` 前缀再合并，最后统一加回。
    * 代码围栏 / 表格 / 图 / 标题 / 分隔线 / 行间公式一律**不合并**。
    * 行尾**两个空格**是 Markdown 的显式硬换行——保留（不合并）。
    """
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_code = False

    def hard_break(line: str) -> bool:
        return line.endswith("  ")

    while i < len(lines):
        line = lines[i]

        # 代码围栏内原样保留（围栏本身也不动）
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue

        # 引用块：整块一起处理（脱前缀 -> 合并 -> 加回前缀）
        if line.lstrip().startswith(">"):
            buf: list[str] = []
            while (i < len(lines) and lines[i].lstrip().startswith(">")):
                content = lines[i].lstrip()[1:]
                if content.startswith(" "):
                    content = content[1:]
                # 引用块内部可能嵌列表：那里不合并（保留 item 结构）
                buf.append(content)
                i += 1
            out.extend(_join_plain_block(buf, hard_break, prefix="> "))
            continue

        # 普通段落 / 列表项：向后吸收续行
        if line.strip() and not _is_block_start(line):
            buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip() \
                    and not _is_block_start(lines[i]) \
                    and not hard_break(buf[-1]):
                buf.append(lines[i].strip())
                i += 1
            out.append(" ".join(buf))
            continue

        # 列表项：吸收缩进续行（这些行不会被 _is_list_item 命中）
        if _is_list_item(line):
            head = line.rstrip()
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    # 空行后若仍是缩进续行（≥2 空格），列表继续
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and (len(lines[j]) - len(
                            lines[j].lstrip())) >= 2 \
                            and not _is_list_item(lines[j]) \
                            and not _is_block_start(lines[j]):
                        i = j
                        continue
                    break
                if _is_list_item(nxt) or _is_block_start(nxt):
                    break
                head += " " + nxt.strip()
                i += 1
            out.append(head)
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _join_plain_block(buf: list[str], hard_break, prefix: str = "") -> list[str]:
    """把引用块的内容行合并成若干行（列表项仍各自独立）。"""
    out: list[str] = []
    cur: list[str] = []
    for ln in buf:
        if not ln.strip():
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append("")
            continue
        if _is_list_item(ln) or _is_block_start(ln):
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append(ln)
            continue
        if cur and hard_break(cur[-1]):
            out.append(" ".join(cur))
            cur = []
        cur.append(ln.strip())
    if cur:
        out.append(" ".join(cur))
    return [prefix + x if x else "" for x in out]


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
        if in_table:
            # 表格单元格里**允许在断点插入 \allowbreak**。这曾经是禁止的
            # （旧注释说会触发对齐钩子），但实测证实：只要 \allowbreak 写在
            # `\texttt{}` **内部**、且不嵌套在 `{\bfseries ...}` 里，就是安全的
            # （最小复现：编译通过、0 处 Overfull）。
            #
            # 为什么必须允许：表格里的长标识符（`AcquisitionLayer`、
            # `InsufficientDataError`）无法断行时会把整行推出页边距——
            # §5.3 那张表实测溢出 90.9pt（约 3.2cm），读者看到的是被切掉的
            # 文字。靠调列宽治不好：无论给多宽，总有一个 token 比列宽长。
            #
            # **必须包 `{\ttfamily ...}`**（2026-09-13 修）：此前表格里的行内
            # 代码**没有等宽标记**——它按正文字体渲染，与正文里的同一段代码
            # 长得不一样（读者看不出那是代码）。而且没有任何标记能把
            # "代码里的字面 `**`" 与"漏转的粗体标记"区分开，护栏无从下手。
            # 用**分组形式** `{\ttfamily ...}` 而非 `\texttt{...}`：前者与
            # 正文路径一致，且不触发 longtable 的标点修正钩子。
            return stash_piece(r"{\ttfamily " + _tt_breakable(code) + r"}")
        # 正文：在断行点前插哨兵再转义——顺序反了会把 \_ 拆成 \ + _，
        # 那个 _ 会逃出等宽盒子变成数学下标（实测踩过）。
        return stash_piece(r"{\ttfamily\small " + _tt_breakable(code) + r"}")

    s = re.sub(r"`([^`]+)`", stash_code, s)

    # 2b) **相邻行内代码之间的分隔符允许断行**（2026-09-13 补）。
    #
    # `_tt_breakable` 只在**代码内部**插断点，所以
    # `` `implementation`/`tests`/`enforced_in` `` 这种写法里，两个等宽盒子
    # **之间**的 `/` 两侧都没有断点——TeX 只在空格或连字符处断行，于是
    # 整串 `implementation/tests/enforced_in` 成了一个不可断单元。
    # 实测它是全表最后一处 Overfull（8.4pt），而它表面上完全像"内容太长"，
    # 很容易误判成列宽不够（那一列实际有 163.5pt，远超需要的 80.5pt）。
    #
    # 只在**两侧都是占位符**时插断点：那说明是"代码 / 代码"的结构，
    # 断在这里语义正确；纯文本里的 `/` 不动（避免正文多出无谓断点）。
    #
    # **必须走 stash_piece**，不能直接写 `\allowbreak{}`：这一步在
    # `tex_escape` **之前**，直接写会被转义成 `\textbackslash{}allowbreak\{\}`
    # （实测：转换自检当场报"反斜杠被转义 10 处"）。
    if in_table:
        s = re.sub(
            r"(\x00\d+\x00)([/|、])(?=\x00\d+\x00)",
            lambda m: (m.group(1) + stash_piece(r"\allowbreak{}")
                       + m.group(2)), s)

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
        #
        # **内层也必须跑 apply_emoji**：外层 `s` 上的那次替换看不到这里的
        # 内容（此刻它已被换成占位符）。曾经漏了这一步，于是**所有加粗文字
        # 里的符号都不被替换**——`**3× 更快**` 里的 `×` 会以裸 Unicode 进正文，
        # 交给 CJK 后备字体渲染（字号与邻接的 `$\pm$` 不一致）。
        # 这是 `×` 在文档里大面积漏替换的真正原因，比行尾注释吞映射更隐蔽：
        # 编译成功、只有观感差异，而漏掉的只在粗体里。
        #
        # **表格与正文都要插断点**（2026-09-13 改）。
        #
        # 原注释的理由是"正文有 17cm，放得下，插了只是多无用断点"——那条
        # 判断建立在**段落被拆碎**的前提上：每行只有 ~72 字符，任何 token
        # 都到不了页边距。修好软换行合并（见 `join_soft_wraps`）之后，
        # 段落恢复成完整长度，**粗体随即成了正文里唯一没有断点的构造**
        # ——行内代码走 `_tt_breakable`，链接走 `\href`，普通长 token 也
        # 在表格外靠连字符/空格断，只有粗体是"整体不可断"。
        # 实测：合并后立刻出现 1 处 `Overfull \hbox (37.1pt)`，正是
        # `**seed 1/2/3/7/11/42/99/123/777/2026 跑出来完全一样**` 里的长
        # token——它在修复前因为 `**` 跨行而**根本没被当成粗体**，所以从未
        # 被测出。断点是 `\allowbreak{}`，只在需要时才断，正常文本无副作用。
        parts = re.split(r"(\x00\d+\x00)", inner)
        chunks: list[str] = []
        for p in parts:
            if re.fullmatch(r"\x00\d+\x00", p):
                chunks.append(p)
            else:
                chunks.append(_add_breakpoints(p, symbols=True))
        return stash_piece(r"{\bfseries " + "".join(chunks) + "}")

    s = re.sub(r"\*\*([^*\x00]*(?:\x00\d+\x00[^*\x00]*)*)\*\*", wrap_bold, s)
    def wrap_italic(m: re.Match) -> str:
        inner = m.group(1)
        parts = re.split(r"(\x00\d+\x00)", inner)
        # 同 wrap_bold：内层必须自己跑一遍 apply_emoji（外层那次看不到它）
        esc = "".join(p if re.fullmatch(r"\x00\d+\x00", p)
                       else apply_emoji(tex_escape(p))
                       for p in parts)
        return stash_piece(r"{\itshape " + esc + "}")

    s = re.sub(r"(?<![\w*])\*([^*\n\x00]+)\*(?![\w*])", wrap_italic, s)

    # 5) 剩余纯文本转义（占位符是 \x00N\x00，转义表不含这两者）
    #
    # 表格里额外给"带点的文件名"插断点：`GUIDE.pdf`、`manifest.json` 这类
    # token 在普通文本里**不可断**（TeX 只在空格与连字符处断行），于是会把
    # 窄列顶出页边距——实测最后一处溢出（4.77pt）就是 `重建 GUIDE.pdf`
    # 这一格的 `GUIDE.pdf` 撑出去的。
    # 只在表格里做：正文有 17cm 宽度，长文件名基本放得下，插进去反而增加
    # 无谓的断行点。
    if in_table:
        s = re.sub(r"(?<=[A-Za-z0-9_])\.(?=[A-Za-z])", "\x03", s)
    s = tex_escape(s)
    if in_table:
        s = s.replace("\x03", r"\allowbreak{}")

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

# **标题层级栈**：[Markdown 的 `#` 数量, 渲染出的深度]。
#
# 为什么需要它（2026-09-13 修，用户报"目录有问题"）：
# 无编号标题原先**只按 `#` 数量**决定层级，于是 §4.7.3（`###`，编号深度 2 →
# depth 2）下面那几个 `####` 子标题（`① 逐行处理 → ...`）被渲染成 depth 2 的
# 兄弟节点，而不是 depth 3 的孩子。侧栏里看起来就像 §4.7.2、§4.7.3、
# `① 逐行处理` 是平级的三个小节——**内容归属完全读错了**。
#
# 正确的规则是经典的两条：
#   1. **编号标题**：深度 = 编号的段数（`4`→0, `4.1`→1, `4.4.1`→2）；
#   2. **无编号标题**：深度 = 栈里最近一个更浅的标题的深度 + 1。
# 弹栈条件是 `md_level >= 当前 level`（同样的 `#` 数量视为同级，不管编号）。
_MD_STACK: list[tuple[int, int]] = []


def _rendered_depth(level: int, num: str | None) -> int:
    """算出这个标题应该渲染到第几层（0=section, 1=subsection, ...）。"""
    while _MD_STACK and _MD_STACK[-1][0] >= level:
        _MD_STACK.pop()
    if num is not None:
        depth = num.count(".")
    elif _MD_STACK:
        depth = _MD_STACK[-1][1] + 1
    else:
        # 还栈是空的（文档里第一个就是无编号标题）——退回按 `#` 数量估。
        depth = {1: 0, 2: 1, 3: 2}.get(level, 3)
    depth = min(depth, 3)           # LaTeX 最深到 paragraph(3)
    _MD_STACK.append((level, depth))
    return depth


# depth → LaTeX 命令名。**顺序不能改**：tocdepth=4 已放开 paragraph。
_DEPTH_CMD = {0: "section", 1: "subsection", 2: "subsubsection",
              3: "paragraph"}


def register_heading(level: int, text: str) -> str | None:
    """记录标题编号对应的 label，返回编号（用于交叉引用）。"""
    m = NUM_HEADING.match(text.strip())
    if not m:
        return None
    num = m.group(1)
    SECTION_LABELS.setdefault(num, f"sec:{num.replace('.', '_')}")
    return num


# 书签标题里能安全出现的字符。**必须逐字白名单，不能只靠"去掉 $ 和反斜杠"**。
#
# 为什么需要这个函数（2026-09-13 发现，用户报"目录有问题"）：
# PDF 书签标题走的是 hyperref 的 `\texorpdfstring` 之外的**第二遍展开**，
# 它把 TeX 当纯文本读。任何"需要排版才成立"的东西（`$...$` 数学模式、
# `\rightarrow`、`\bfseries`）它都**静默丢弃**——只留一条
# `Package hyperref Warning: Token not allowed in a PDF string` 在 log 里，
# 退出码仍然是 0。
#
# 实测后果：13 条警告，都是同一类标题。`→` 被映射成 `$\rightarrow$` 进正文
# （PDF 上是对的），但书签标题变成「1.1 核心想法：伴随放电  副本，负像
# 运行统计，残差  触发」——**三个箭头全没了，只剩双空格**（侧栏里看上去
# 像排版事故）。§2.2「启发 → 工程问题总表」同理。
#
# 修法：给 `\section[书签]{正文}` 的**可选参数**单独提供一份"纯文本版"标题。
# 正文里照样用 `→` 的数学渲染（好看），书签里用 `->`（不会丢）。
# 这是 hyperref 设计的标准用法，只是很容易忘记写。
def pdf_bookmark_text(raw: str) -> str:
    r"""把 Markdown 标题转成**书签安全**的形式。

    与 `slugify_inline` 的分工：后者产出"排版用的正文"，可以有 `$...$`、
    `{\ttfamily ...}`；这里产出"PDF 书签里显示的一行字"。

    **它不是"纯文本"，是一个受限的 TeX 上下文。** 这一点是实测出来的
    （`/tmp` 下的最小复现，一次只变一个量），结论列在下面 `_BOOKMARK_SAFE`：
    有些东西**必须**转义（`_ & % # $ { }`），有些东西**必须**防连字
    （`--` → `–`），有些东西反而是普通字符（`< > ^ ~`）。

    **为什么不能省这一层**（2026-09-13 发现，用户报"目录有问题"）：
    PDF 书签走的是 hyperref 的第二遍展开，它把标题当"PDF 字符串"读。
    `$...$`、`\rightarrow`、`\bfseries` 这类"需要排版才成立"的 token
    **直接被删掉**——只留一条 `Token not allowed in a PDF string` 警告，
    退出码仍是 0。实测正文有 13 处标题含 `→`，正文里印得好好的，
    侧栏里变成「伴随放电　副本」这种双空格（箭头没了）。§2.2 同理。

    Returns:
        可直接作为 `\texorpdfstring` 第二个参数、或 `\addcontentsline`
        参数的安全字符串。
    """
    s = raw
    s = re.sub(r"`([^`]+)`", r"\1", s)               # 行内代码 → 去反引号
    s = s.replace("**", "").replace("*", "")          # 粗体标记 → 去星号
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)    # 链接 → 只留文字

    # **先处理多字符键**，与 `apply_emoji` 同理（那里有详细说明）：
    # `⚠️` 是"基字符 U+26A0 + 变体选择符 U+FE0F"。逐字符扫描只会命中
    # `⚠`，把 U+FE0F 留在原地——那是个排不出的字符（本例实测产出
    # `[部分]️`，末尾那个不可见的选择符会被 xelatex 静默跳过）。
    # 顺序不能反：先逐字符扫的话，`⚠` 已被换成 `[部分]`，后面的
    # U+FE0F 就再也找不到配对的基字符了。
    for k in _MULTI_CHAR_KEYS:
        if k in s:
            s = s.replace(k, BOOKMARK_FALLBACK[k])
    s = "".join(BOOKMARK_FALLBACK.get(c, c) for c in s)

    out = []
    for c in s:
        if c == "\\":
            out.append(r"\textbackslash{}")
        elif c in "_&%#$":
            out.append("\\" + c)          # 必须转义：实测 `\_` 显示为 `_`，
                                          # 而裸 `_` 会触发 subscript 警告并被丢
        elif c in "{}":
            out.append("\\" + c)
        elif c == "-":
            out.append("-")               # 连字单独按串处理（见下）
        else:
            out.append(c)
    s = "".join(out)
    # `--` 在 TeX 里是连字符（en dash）连字：书签里会显示成 `–`。
    # 标题里 `--random`（命令行旗标）就这样被印成了「–random」——不是崩溃，
    # 但读者会以为我们写错了旗标。实测 `-{}-` 能断开连字且显示为 `--`。
    # 放在最后做：此时剩下的 `-` 都是字面连字符，不会有转义里的 `-`。
    return s.replace("--", "-{}-")


# 书签标题里的字符替换：**必须是纯文本**，不能含 `$` 或反斜杠命令。
# 与 `VERBATIM_FALLBACK` 同样思路（排版层降级），但用途不同：
# 那个是给等宽字体缺字形用的，这个是给"PDF 字符串里不允许的 token"用的。
#
# **必须对 `EMOJI_MAP` 全覆盖**——有专门的守卫测试盯着（见
# `tests/test_guide_convert.py::test_bookmark_fallback_covers_emoji_map`）。
# 漏一个的后果不是"书签里少个符号"，而是那个字符以**裸 Unicode 走进
# `.toc` 文件**、再被 TeX 当成"当前字体里没有的字形"跳过：
# 自检 4 会报 Missing character，而报的位置是**目录页**——指向的地方
# 和真正的成因（BOOKMARK_FALLBACK 少了一条）隔着两层。
# 这个坑是加 `\addcontentsline` 支持无编号标题时**当场踩到的**：
# §4.7.3 那三个 `① ② ③` 小节标题立刻在目录页丢了字。
BOOKMARK_FALLBACK = {
    "\u2192": "->",     # →
    "\u2190": "<-",     # ←
    "\u2194": "<->",    # ↔
    "\u21d2": "=>",     # ⇒
    "\u2248": "~=",     # ≈
    "\u2261": "==",     # ≡
    "\u2260": "!=",     # ≠
    "\u2264": "<=",     # ≤
    "\u2265": ">=",     # ≥
    "\u2208": "in",     # ∈
    "\u00b1": "+/-",    # ±
    "\u00d7": "x",      # ×
    "\u00b7": ".",      # ·
    "\u2212": "-",      # −
    "\u221a": "sqrt",   # √
    "\u2044": "/",      # ⁄
    "\u03c3": "sigma",  # σ
    "\u03bc": "mu",     # μ
    "\u03a3": "Sigma",  # Σ
    "\u0394": "Delta",  # Δ
    "\u03b8": "theta",  # θ
    "\u03b1": "alpha",  # α
    "\u03b2": "beta",   # β
    "\u03c4": "tau",    # τ
    "\u03bb": "lambda",  # λ
    "\u03c0": "pi",     # π
    "\u00b2": "^2",     # ²
    "\u00b3": "^3",     # ³
    "\u207b": "^-",     # ⁻
    # 圈号：正文里 ① 走 `(1)`，书签里同样用 `(1)`（纯 ASCII，最保险）
    "\u2460": "(1)",    # ①
    "\u2461": "(2)",    # ②
    "\u2462": "(3)",    # ③
    "\u2463": "(4)",    # ④
    "\u2464": "(5)",    # ⑤
    "\u2465": "(6)",    # ⑥
    "\u2466": "(7)",    # ⑦
    "\u2467": "(8)",    # ⑧
    "\u2468": "(9)",    # ⑨
    "\u2469": "(10)",   # ⑩
    "\u2605": "*",      # ★
    "\u27e8": "<",      # ⟨
    "\u27e9": ">",      # ⟩
    # 制表符号：只在代码块里出现，标题里不该有；留着是为了"覆盖完整"这条
    # 不变式本身可被机械校验，不必判断"这个字符会不会出现在标题里"
    "\u2500": "---",    # ─
    "\u2502": "|",      # │
    "\u251c": "|--",    # ├
    "\u2514": "`--",    # └
    "\u251d": "|--",    # ┝
    "\u253c": "+",      # ┼
    "\u2705": "[成立]",     # ✅
    "\u274c": "[域外]",     # ❌
    "\u26a0\ufe0f": "[部分]",   # ⚠️
    "\u26a0": "[部分]",       # ⚠
    "\u2b1c": "[未测]",     # ⬜
    "\u2713": "v",      # ✓
    "\u2717": "x",      # ✗
}


def render_heading(level: int, text: str) -> str:
    raw = text.strip()
    num = register_heading(level, raw)
    depth = _rendered_depth(level, num)
    cmd = _DEPTH_CMD[depth]

    # **层级按编号深度决定，不按 Markdown 的 # 数量。**
    # GUIDE 里 §4（`##`）、§4.1（`###`）、§4.4（`##`）的 # 数量与语义层级
    # 不一致（§4.4 是 §4 的子节却写成二级）。若照搬 # 数量，§4.4 会和 §4
    # 平级，书签树就错了。编号标题统一以编号的段数（"4" / "4.1" / "4.4.1"）为准；
    # 无编号标题按栈（见 `_rendered_depth`）。
    #
    # **标题一律走 `\texorpdfstring{正文}{书签}`**（2026-09-13 修）。
    # 正文那半可以有任何排版（`$→$`、`{\ttfamily ...}`）；书签那半必须是
    # 纯文本。若不分开给，hyperref 把正文直接当 PDF 字符串读，遇到
    # "不属于字符串上下文"的 token 就**静默丢弃**——只留一条 warning 在
    # log 里，退出码仍是 0。实测 13 处，全是标题里的 `→`：
    # 侧栏显示成「1.1 核心想法：伴随放电  副本，负像  运行统计，残差  触发」，
    # 双空格处就是被丢掉的箭头。用户报的"目录有问题"有一部分是这条。
    if num:
        body = slugify_inline(raw)
        return (f"\\{cmd}{{{texorpdfstring(body, pdf_bookmark_text(raw))}}}"
                f"\\label{{{SECTION_LABELS[num]}}}")

    # 无编号标题：**必须用带星号的版本，但也要自己补一条目录/书签**。
    #
    # `\section*` 不进目录、也不进 PDF 书签。本文唯一的无编号二级标题是
    # 「先看这里：五分钟跑通」——它**整条从侧栏消失了**（实测：书签树里
    # §0 上面什么都没有），而它恰恰是全篇的入口小节。
    # 补法是 `\addcontentsline` + `\phantomsection`：前者造目录条目，
    # 后者给一个锚点，否则点击目录条目会跳到**上一页**的某处（实测偏一页）。
    #
    # `\addcontentsline` 的参数**同时喂给"排印的目录页"和"PDF 书签"**，
    # 两边的需求还不同，所以同样要用 `\texorpdfstring` 分开给：
    #   * 目录页那半走 `slugify_inline`——它把 `①` 换成 `(1)`。实测目录页
    #     用**粗体罗马字体**排标题，那个字体里**没有** `①` 的字形，`\addcontentsline`
    #     里的裸 `①` 会被静默跳过（自检 4 报在目录页，成因却在这一行）。
    #   * 书签那半走 `pdf_bookmark_text`。
    # 实测：这样写 TokenWarn=0、MissingChar=0，两边都正常。
    body = slugify_inline(raw)
    dual = texorpdfstring(body, pdf_bookmark_text(raw))
    return (f"\\phantomsection\\addcontentsline{{toc}}{{{cmd}}}{{{dual}}}"
            f"\\{cmd}*{{{dual}}}")


def texorpdfstring(body: str, bookmark: str) -> str:
    r"""`\texorpdfstring{正文}{书签}`——同一标题的两副面孔。

    单独包一层是为了**可在测试里直接断言**："书签那一半里不含 `$`、不含
    反斜杠命令"。散在 f-string 里就没法测。
    """
    return f"\\texorpdfstring{{{body}}}{{{bookmark}}}"


def render_figure(caption: str, stem: str) -> str:
    """Markdown 图片 → LaTeX figure 环境。

    **优先用同名矢量 PDF**（结构图由 build_figures.py 从 .tikz 生成）；
    没有 PDF 时用 PNG（数据图由 make_data_figures.py 用 PIL 生成）。
    判据是文件是否存在，不靠额外的登记表——少一处会忘的地方。

    用 `[H]`（float 宏包）而不是 `[htbp]`：本文表格多、篇幅长，浮动体在
    一份 60+ 页的正文里会漂到离引用处很远的地方——读者看到图时已经忘了
    前文在说什么。图是正文的骨架，必须钉在原位。
    """
    vec = os.path.join(ROOT, stem + ".pdf")
    src = f"{stem}.pdf" if os.path.exists(vec) else f"{stem}.png"
    # **图与图注必须绑成一盒，不许在中间断页**（2026-09-13，配合分栏）。
    # 分栏后图落在栏内，`[H]` 把它钉在原位；如果栏里剩余高度不够，
    # TeX 会把图注挤到页面下边界之外——实测图注最后一行 y=842.4，
    # 而页高 841.9，**文字被切掉**（`Overfull \vbox` 报 76pt 就是这个）。
    # `\noindent\begin{minipage}` 让"图 + 图注"成为一个不可分割的整体，
    # 放不下就整体移到下一栏，而不是被切开。
    return "\n".join([
        r"\noindent\begin{minipage}{\linewidth}",
        r"\centering",
        rf"\includegraphics[width={FIG_WIDTH},height={FIG_HEIGHT},"
        rf"keepaspectratio]{{{src}}}",
        rf"\captionof{{figure}}{{{slugify_inline(caption)}}}",
        r"\end{minipage}",
        "",
    ])


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


# 字符**显示宽度**（半角为单位）。用实测值，不要凭感觉拍。
#
# 实测方法（本机 xelatex + `\sbox`，11pt ctexart，`\textwidth` = 483.7pt）：
#
#     汉字 ×10 = 109.5pt  ->  **10.95pt / 字**
#     `a`  ×10 =  54.7pt  ->   5.475pt / 字
#     `A`  ×10 =  82.1pt  ->   8.21pt / 字
#     数字 ×10 =  54.7pt  ->   5.475pt / 字
#     等宽 `m`/`M` ×10 = 57.5pt -> 5.75pt（与字号无关的等宽网格）
#
# 结论：**一个汉字 = 2 个半角单位**（与 ctex 的 CJK 字体度量一致），
# 数字/小写字母 = 1，大写字母 ≈ 1.5。
#
# 为什么必须区分：`len()` 把它们都算 1，于是纯汉字列的列宽权重被低估
# **一半**——实测"神经科学概念"占的列只拿到 61.7pt，而 6 个汉字需要
# 6×10.95 = 65.7pt，差 4pt 就放不下，于是被排成"神经科学概"+"念"。
# 读者看到的是"表格串行、一个字换行"。
_CJK_WIDTH = 2.0      # 汉字（含全角标点）相对半角
_UPPER_WIDTH = 1.5    # 大写拉丁字母偏宽（82.1/54.7 ≈ 1.5）


def _display_width(s: str) -> float:
    """字符串的**显示宽度**（以半角字符为单位）。

    用于表格列宽分配。这个函数必须量**排版后**的宽度，而不是源文本的
    字符数——**两次栽在同一件事上**，所以两次都写在下面：

    1. **`len()` 不认汉字**（2026-09-13 修）：`len("汉") == len("a") == 1`，
       而汉字实际占两个半角宽。纯汉字列被低估一半，6 字表头被排成
       "5 字 + 1 字"。所以有了 `_CJK_WIDTH`。
    2. **源文本有"排版后会变宽"的字符**（2026-09-13 修）：`✅` 在源文件里
       是 1 个字符，但 `apply_emoji` 把它渲染成 `{\\bfseries [成立]}`
       ——**4 个可视字符 + 粗体**。按源文本量只有 1.0，实际要 6.0，
       **低估 6 倍**。

    第 2 条的后果是可见的：§0.4 那张 19 行总表的"状态"列权重算出来是 6.0
    （下限），归一化后列宽 29.2pt，减掉 `\\tabcolsep` 只剩 21.2pt，
    而 `[成立]` 在 `\\footnotesize` 下要 21.5pt——**差不到 1pt，放不下**，
    于是每一格都被折成 `[成` + `立]`。实测全文 **24 处**这种"括号裂开"。
    """
    # **先做与排版一致的替换**，再量宽度：emoji/符号会展开成更宽的文本。
    # 必须在去掉 Markdown 标记**之前**做——`apply_emoji` 认的是源字符。
    t = apply_emoji(s)
    # **剥掉 LaTeX 命令只留"看得见的字"**。`apply_emoji` 产出的是
    # `{\bfseries [成立]}` 这种片段，如果只把反斜杠跳掉，`bfseries` 那 8 个
    # 字母会被当成正文量进去（实测 17.0，而真正的可视宽度只有 6.0）——
    # 那是**反向**的错，会把窄列撑宽、挤掉别的列。
    t = re.sub(r"\\[A-Za-z@]+\*?", "", t)       # 控制序列名（\bfseries 等）
    t = re.sub(r"\\[^A-Za-z]", "", t)           # 单字符命令（\\ \{ \} 等）
    t = t.replace("{", "").replace("}", "")     # 分组花括号不占宽
    # 再去掉不产生宽度的 Markdown 标记
    t = re.sub(r"`([^`]*)`", r"\1", t)          # 行内代码：只算内容
    t = re.sub(r"\*\*([^*]*)\*\*", r"\1", t)    # 粗体
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)   # 链接：只算文字
    t = re.sub(r"[*_`]", "", t)                 # 残留的标记字符
    # 控制序列名被剥掉后会留下一个空格（`\bfseries [成立]` → ` [成立]`），
    # 而 TeX 会吃掉行首空白——量进去就多算了 1.0（实测 `[成立]` 变 7.0）。
    t = t.strip()
    total = 0.0
    for ch in t:
        o = ord(ch)
        if o > 0x2E80:                          # CJK 及全角
            total += _CJK_WIDTH
        elif "A" <= ch <= "Z":
            total += _UPPER_WIDTH
        elif o < 0x20:                          # 控制字符不占宽
            continue
        elif ch == "$":                         # 数学定界符不占宽
            continue
        else:
            total += 1.0
    return total


COL_WEIGHT_MIN = 12.0
COL_WEIGHT_MAX = 50.0


def column_weights(header: list[str], body: list[list[str]]) -> list[float]:
    """按内容**显示宽度**分配列权重（长文本列给更多权重）。

    单独抽成函数，是为了让守卫能调到**真实实现**。
    2026-09-13 踩的坑：测试文件里自己复制了一份同样的算式，于是"注入验证"
    永远注入不进去——把 `second * 0.55` 删掉，守卫**照样是绿的**，因为它
    量的是副本。抽出来之后注入就能让它变红。
    """
    weights = []
    for c in range(len(header)):
        lens = [_display_width(header[c])] + [_display_width(r[c])
                                              for r in body]
        # 用中位数而非最大值做"典型长度"，避免单个超长路径吃掉整列预算
        lens_sorted = sorted(lens)
        typical = lens_sorted[len(lens_sorted) // 2]
        w = typical + max(lens) // 4
        # **给"第二长条目"留出余地**（2026-09-13 加）。
        #
        # 什么时候这条才真的起作用？——**表头中等、绝大多数格极短、
        # 第二长与最长同量级**的列。中位数被那一堆极短的格压到很低
        # （如 4），于是 `中位 + max//4` 只给出 9，而第二长是 20，
        # 要靠 `20 × 0.55 = 11` 才顶得上去。
        # GUIDE.md:689（"IMU 有价值 / 无价值"那张表）就是这个形状。
        #
        # **作用范围（实测，216 列）**：在**截断前**改变 `w` 的有 9 列，
        # 在 `max(6, min(50, …))` **截断后**真正改变最终权重的有 7 列。
        #
        # 为什么是 0.55 而不是更大：最大的那条（常是长路径/长标识符）该允许折行，
        # 全按它分配会挤掉别的列（这就是当初"列宽下界"失败的原因）。
        # 取一个中间值——保证"第二长"能放下。
        #
        # **警告（2026-09-13 更正）**：本节早先的注释声称这条修的是
        # `| 词 | 这里指什么 | 打个比方 |` 表里的 `**特征图**` 折行，
        # 并称那一列"拿到 55.6pt"。**实测证伪**：
        #     该列 lens = [2.0, 4.5, 6.0, 6.0]
        #     中位 = 6.0，`中位 + max//4` = 7.0，`第二长 × 0.55` = 3.3
        #     → 最终权重 **7.0**（来自中位数那一路，兜底没参与）
        #     归一化后列宽 47.5pt − 2×4pt(\tabcolsep) = 39.5pt
        #     `\footnotesize` 下一个汉字 9.0pt → 放得下 4 个字
        #     `特征图` 只要 27pt，**放得下，根本不折行**。
        # 也就是说这条注释里的病因、数字、结论**三样都是错的**，
        # 而它一度把我引到错误的测试构造上，白费一轮排查。
        # 教训：注释里写"这条修了什么"，必须能被实测复现；
        #      **写不出实测数字的因果说明，就是猜测**。
        second = lens_sorted[-2] if len(lens_sorted) >= 2 else lens_sorted[-1]
        w = max(w, second * 0.55)
        # **上限 50 而不是 40**（2026-09-13 实测调整）。
        #
        # 病因：一张 3 列的表里，宽列的需求量常常都超过 40 → **一起顶到上限**，
        # 于是"权重"退化成"每列一样宽"，而窄列（表头只有 1–2 个汉字）拿到的
        # 绝对宽度不够，那一格的文字被折成"…+ 孤零零一个字"。
        # 实测 `| 词 | 这里指什么 | 打个比方 |` 那张表：后两列的原始需求是
        # 89 和 45，都被压成 40，总权重 86；短列只占 6/86。
        #
        # 放宽到 50 后的**实测对比**（同一份 GUIDE，单变量）：
        #   cap 40 → 孤字 22 个、0 处溢出
        #   cap 50 → 孤字 17 个、0 处溢出   ← 采用
        #   cap 60 → 孤字 22 个、1 处溢出（开始把窄列挤爆）
        # 页数三者都是 66，不影响篇幅。
        #
        # **下限 12 而不是 6**（2026-09-13 实测调整，配合上一段的显示宽度修正）。
        #
        # 病因和上限那条同源但方向相反：下限太松时，窄列（表头只有 1–2 个字）
        # 拿到的绝对宽度不够，**最长的那一格**放不下。实测两处：
        #   `| 词 | … |` 表：`特征图` 裂成 `特征` + `图`
        #   `| 输入 | 修复前 | … |` 表：`返回` + `ndarray` 裂开
        #
        # **实测对比**（同一份 GUIDE，单变量，量"表格折行残留"处数）：
        #   min  6 → 13 处、0 溢出、66 页
        #   min 10 → 12 处、0 溢出、66 页
        #   min 12 → 11 处、0 溢出、66 页   ← 采用（修掉上面两处）
        #   min 16 → 10 处、0 溢出、66 页
        # 取 12 而不是 16：12 已经修掉那两处**具体的**裂开，16 只多修一处、
        # 却把更多预算从宽列拿走。判据是"修掉已知病灶"，不是"数字最小"。
        #
        # **试过但失败的方向**（别重试）：把下限改成"最长格 × 0.35/0.45/0.55"
        # ——折行残留反而从 13 涨到 **32**。这和当年"列宽下界"那次失败同源：
        # 按**最长格**保底会把宽度从别的列抢走，总量守恒，只是把病换了个地方。
        weights.append(max(COL_WEIGHT_MIN, min(COL_WEIGHT_MAX, w)))
    return weights


def render_table(rows: list[str]) -> str:
    header = split_row(rows[0])
    ncol = len(header)
    body = [split_row(r) for r in rows[2:] if not is_sep_row(r)]
    body = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol]
            for r in body]

    # 列宽按内容**显示宽度**分配（长文本列给更多权重）。
    #
    # **这就是"表格串行、一个字换行"的根因**（2026-09-13 修）。
    # 原来用 `len()`，它把汉字与拉丁字母都算 1 —— 但实测（本机 `\sbox` 量）：
    #
    #     汉字 10.95pt   Latin `a` 5.475pt   数字 5.475pt
    #
    # 即**一个汉字恰好占两个半角宽**。于是纯汉字列的权重被低估一半：
    # 5 列表"神经科学概念"那列只拿到 61.7pt，而 6 个汉字需要 65.7pt
    # ——差 4pt 放不下，表头被排成"神经科学概"+"念"。整张表因此每一格
    # 都多折几行，读者看到的就是"一个字一行"。
    #
    # 只改这一个度量，不动别的参数。我试过再加"列宽下界"和"列太多就缩小
    # 字号"两层，**反而更差**（页数 68→69、短行 249→269）——那是在用
    # 更复杂的办法掩盖一个度量错误。
    weights = column_weights(header, body)

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

    # 列间距收到 4pt；表格整体用 \footnotesize。
    #
    # **为什么表格用更小的字号**（2026-09-13，用户报"表格串行太严重"）：
    # 这是本项目表格最集中、最难看的排版问题——读者看到大量"一行只有
    # 1–2 个汉字"的碎片行。成因是结构性的：GUIDE 的表格常是 4–6 列、
    # 每格又是 15–25 个汉字的完整句子，而 \textwidth 只有 483.7pt。
    # 11pt 下一个汉字 10.95pt，5 列表分下来每列一行只放得下 6–9 个字，
    # 于是每格都要折 2–3 行，末行剩 1–2 个字。
    #
    # 调列宽只能**重新分配**那点宽度，改变不了总量；字号才是真的把每个字
    # 变窄（\footnotesize 下 10.95 -> 9.3pt，一行多放 18%）。实测四种组合
    # （同一份 GUIDE，只改这两处，用"一行 1–2 个汉字"的行数计）：
    #
    #     原始（len 度量, 11pt）          211
    #     仅显示宽度度量                  193
    #     仅 \footnotesize                202
    #     显示宽度度量 + \footnotesize    **126**   <- 采用
    #
    # 代价是表格字比正文小一号。这是**有意的取舍**：表格是查阅用的，
    # 密度比字号重要；正文不受影响。
    out = [r"{\footnotesize\setlength{\tabcolsep}{4pt}",
           r"\begin{longtable}{" + spec + "}", r"\toprule"]
    out.append(" & ".join(slugify_inline(c, in_table=True) for c in header)
               + r" \\")
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
                # 代码块字号用 \footnotesize 而不是 \small：实测 `quote` 环境
                # 里的可用宽度是 428.9pt，而本仓库最长的命令行（89 字符的
                # pytest 命令）在 \small 下需要 431.3pt——**只差 2.4pt 就
                # 溢出**，任何一次微小的内容改动都会让它越界。\footnotesize
                # 下同样的行只要 392.2pt，留出 37pt 余量。
                # 代价是字略小，但等宽代码本来就在视觉上偏大，反而更协调。
                out.append(r"\begin{quote}\ttfamily\footnotesize")
                out.append(r"\begin{verbatim}")
                out.extend(_verbatim_safe(code_buf))
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

        # --- 图。**必须放在链接规则之前判**：`![cap](url)` 里的
        #     `[cap](url)` 会被链接正则先命中，变成普通链接——图就静默
        #     消失了，而且转换"成功"，最难发现。
        fim = IMAGE_RE.match(line.strip())
        if fim:
            flush_table(); flush_quote(); flush_list()
            out.append(render_figure(fim.group(1), fim.group(2)))
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
\usepackage{float}       % [H]：图钉在原位（本文表格多，浮动体会漂远）
\usepackage{caption}
\usepackage{hyperref}
\usepackage{xcolor}

% 图注小一号、加粗"图 N"，与正文区分开
\captionsetup{font=small,labelfont=bf,skip=4pt}
\renewcommand{\figurename}{图}

% PDF 书签与可点击目录：这些是"快速跳转"的实现
\hypersetup{
  % **必须 false**：GUIDE 的章节号（§4.7）是**写在标题文字里**的，LaTeX 若
  % 再自动编一次，书签就变成"6.6 §4.7 自动化兜底"这种两套编号打架的样子
  % （实测：读者会以为有两个目录）。编号由正文自带，不要自动加。
  bookmarksnumbered=false,
  bookmarksopen=true,         % 打开时就展开
  bookmarksopenlevel=3,       % 展开到三级（x.y.z 也展开）
  colorlinks=true,
  linkcolor=blue!55!black,
  urlcolor=teal!70!black,
  citecolor=blue!55!black,
  pdftitle={残差触发数据采集——项目指南},
  pdfauthor={},
  % **元数据里的下划线必须转义**（2026-09-13 修）。`pdfsubject` 原本写的是
  % 裸 `cerebellum_cancel`，hyperref 把 `_` 当数学下标 → `removing
  % 'subscript'` → PDF 元数据里变成 **`cerebellumcancel`**（下划线消失）。
  % 编译退出码 0、正文一字不错、只有文件属性里坏掉——又一个"没人会看的地方"。
  % 读回来的判据：`PdfReader(pdf).metadata["/Subject"]` 必须含下划线。
  % （同样的静默丢字机制在书签标题里也发生过，见 pdf_bookmark_text。）
  pdfsubject={CMCI / cerebellum\_cancel 项目指南},
  pdfkeywords={residual trigger, corollary discharge, IMU, data acquisition}
}

% **让四级标题也进书签**：默认 \paragraph 不进书签，于是 §4.7.1.1 这类四级
% 小节在侧栏里整个消失（实测踩到）。tocdepth=4 把它收回来。
% 注意这些 \setcounter 必须留在 preamble —— 放进 \begin{document} 之后
% 会让 xelatex 直接报 "No pages of output"。
\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{0}   % 不自动编号（GUIDE 的编号写在标题文字里）

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


def _strip_tex_group(s: str, cmd: str) -> str:
    r"""删掉形如 ``{\cmd ...}`` 的整个花括号分组（含正确配对的嵌套）。

    为什么需要它：``{\ttfamily\small raw\_ratio ≈ 1}`` 里的花括号是**分组**，
    不是命令参数——它包住的是"从这里开始用等宽字体"，值一直延伸到配对的
    ``}``。用 `\{\\ttfamily[^}]*\}` 去匹配会在第一个 `}` 处停下，漏掉真正的
    内容（实测这个误写让 6 处等宽文本仍被报出来）。

    这里按花括号配对数扫描，不用正则（正则处理嵌套配对本就不可靠）。
    """
    key = "{" + cmd
    out, i = [], 0
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        depth, k = 0, j
        while k < len(s):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1     # 跳过整个分组


def _find_unmapped_symbols(tex_body: str) -> list[tuple[str, int]]:
    r"""找出会漏进正文的 Unicode 符号：返回 [(字符, 次数)]。

    为什么要有这条守卫：漏映射**不会**让编译失败，只会让字符按 CJK 后备字体
    渲染——字号、基线、字宽全按汉字来，于是 `σ` 比旁边的 `$\mu$` 大一号，
    而且它周围的间距是汉字间距。产物照样"生成成功"，问题只体现在观感上，
    极容易被归因成"字体就是这样"。

    扫描范围刻意**只包含**不该由 CJK 字体承担的符号类字符：
      * 希腊字母（U+0370–U+03FF）：数学变量，必须走数学字体；
      * 数学运算符（U+2200–U+22FF）：同上；
      * 常用上下标与比较符（U+00B1/U+00D7/U+00B2/U+00B3/U+207B/U+2260 等）。

    中文（CJK）**不在**范围内——那些正是要交给 ctex 的。
    ``verbatim`` 环境与注释行也跳过：代码块里出现希腊字母是合法的，
    不能因此报错。
    """
    # 去掉 verbatim 环境（代码块）与 TeX 注释行，避免误报
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "",
                  tex_body, flags=re.S)
    body = re.sub(r"(?m)^\s*%.*$", "", body)
    # 行内代码（\texttt{} / {\ttfamily ...}）里的字符是**故意字面**的，不能
    # 替换成数学符号——`{\ttfamily raw\_ratio ≈ 1}` 里的 ≈ 换成 $\approx$
    # 会把等宽文本拆成两段，代码读起来就断了。所以它们也不该被这条守卫报出来。
    #
    # 注意 {\ttfamily\small ...} 是**分组**而非命令参数：花括号只包住命令名
    # 后面的样式切换，值在分组里直到配对的 `}`。所以按"整组"剥，不能按
    # `\ttfamily{...}` 的形状去匹配（实测这样匹配不到，仍会误报 6 处）。
    body = re.sub(r"\\texttt\{[^{}]*\}", "", body)
    body = _strip_tex_group(body, r"\ttfamily")

    def risky(ch: str) -> bool:
        o = ord(ch)
        return (0x0370 <= o <= 0x03FF        # 希腊字母
                or 0x2200 <= o <= 0x22FF     # 数学运算符
                or ch in "\u00b1\u00d7\u00b2\u00b3\u207b\u2260\u2264\u2265"
                or o == 0x2044)              # 分数斜杠

    counts: dict[str, int] = {}
    for ch in body:
        if risky(ch):
            counts[ch] = counts.get(ch, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def _check_bookmark_text(pdf_path: str) -> list[str]:
    r"""检查 PDF 里真正的书签文字，返回"有丢失痕迹"的那些。

    **这条守卫的判据是产物，不是日志。** 上一节的自检 5 只看 hyperref 的
    警告——但警告与结果不是一一对应的：`\texorpdfstring{正文}{书签}` 在展开
    **正文**那一半时也会触发同样的警告（hyperref 要检查它），而书签那半是
    干净的。反过来也有——某些 token 不报警告、照样从书签里消失。

    真正的判据是：**书签里出现了"正文有的东西，书签里没有"的痕迹。**
    最可靠、也最容易自动化的一个是**连续两个以上空格**：中文排版里不需要
    空格，而"箭头被删掉、两侧的文字挤到一起"恰恰就是这个形态：

        正文  1.1 核心想法：伴随放电 → 副本，负像 → 运行统计，残差 → 触发
        书签  1.1 核心想法：伴随放电  副本，负像  运行统计，残差  触发
                                    ↑↑                      ↑↑

    实测 13 处，全部是这种"双空格"。所以这条守卫抓得准。

    加这条守卫的原因：这已经是**同一类静默失效的第三次**了（段落被拆碎、
    字符被丢弃、现在是书签被截断）。共同点都是"编译成功、退出码 0、
    正文看起来正常"，只有侧栏/目录这种**平时没人细看的地方**坏掉。
    """
    if not os.path.exists(pdf_path):
        return []
    try:
        from pypdf import PdfReader
    except ImportError:
        return []          # pypdf 只用于开发期自查，缺了不该让构建失败
    try:
        reader = PdfReader(pdf_path)
        titles: list[str] = []

        def walk(items) -> None:
            for it in items:
                if isinstance(it, list):
                    walk(it)
                else:
                    titles.append(str(it.title))

        walk(reader.outline)
    except Exception:
        return []

    bad = []
    for t in titles:
        if re.search(r"\S\s{2,}\S", t):
            bad.append(t)
    return bad


def _check_pdf_metadata(pdf_path: str) -> list[str]:
    r"""检查 PDF 元数据里的下划线有没有被静默吃掉。

    这是**同一类静默失效第四次**冒头（前三次：段落被拆碎、字符被丢、
    书签被截断）。机制完全相同——hyperref 把值当"PDF 字符串"读，遇到
    `_` 就当成数学下标，于是 `removing 'subscript'` 然后继续。

    实测：`pdfsubject={CMCI / cerebellum_cancel 项目指南}` 在 PDF 属性里
    变成 **`cerebellumcancel`**（下划线消失）。编译成功、正文无误、所有
    其它检查全绿——只有文件属性里坏掉，而那是几乎没人会看的地方。

    判据：元数据里出现过的下划线，读回来必须还在。
    """
    if not os.path.exists(pdf_path):
        return []
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        meta = PdfReader(pdf_path).metadata or {}
    except Exception:
        return []
    # 这些值里**应该**有下划线（源码里写的就是带下划线的标识符）
    expect = {"Subject": "cerebellum_cancel"}
    bad = []
    for key, want in expect.items():
        got = str(meta.get("/" + key, ""))
        if want.replace("_", "") in got.replace(" ", "") and want not in got:
            bad.append(f"/{key} = {got!r}（少了 {want!r} 里的下划线）")
    return bad


def md_to_tex_body(md: str) -> str:
    """完整转换管线：Markdown 原文 → TeX 正文（不含导言区）。

    **抽成独立函数是为了可测**（2026-09-13）：此前"去标题 / 合并软换行 /
    convert"这三步直接写在 `main()` 里，测试只能自己重演那三步——于是
    "`main()` 里漏掉 `join_soft_wraps`"这类**接线级** bug 测不出来
    （实测：把 `main()` 里的合并在注入测试中删掉，40 个转换器测试全绿）。
    现在测试与生产走同一条路。
    """
    # 去掉文档顶部的重复标题块（§0 之前的引文块保留）
    md = re.sub(r"^#{1}\s+.*?\n", "", md, count=1)
    # **重置标题层级栈**：`_MD_STACK` 是模块级状态，跨调用会残留
    # （`md_to_tex_body` 在同一进程里被调用多次——测试、重复转换）。
    # 不清空的话，第二次转换会接着上一次的栈算层级，产出与首次不同的
    # 结果——典型的"测试之间互相污染"，而且症状是"层级偶尔错一层"，
    # 非常难查。放在这里而不是 main()：任何调用点都自动正确。
    _MD_STACK.clear()
    # **先把软换行合并掉**。必须在 convert 之前、且不能放进 convert 内部：
    # convert 是"逐行"的状态机，合并后的行才是它的语义单位。
    # 不合并的后果见 join_soft_wraps 的 docstring——最严重的是跨行
    # `**粗体**` 匹配不上，88 个字面星号印进 PDF。
    md = join_soft_wraps(md)
    return convert(md)


# =============================================== 分栏（已定：整篇单栏）
#
# 注：本节原为"选择性两栏"实现，2026-09-14 用户改定——**不要分栏，全部横着
# （单栏）**，"分栏了之后看起来很难受"。`make_two_column` 已改为恒等函数；
# 下方 `_TWOCOL_*` 常量随之停用、留作历史（未来再想分栏时连同坑实录参考）。
#
# **原理由（2026-09-13，用户要求"可以选择性地两栏"）：**
#
# 整篇 `twocolumn` 做不到——文档里有 65 张 `longtable`，它会直接报
# `! Package longtable Error: longtable not in 1-column mode.`（实测；
# 而且必须加 `-halt-on-error` 才看得见：不加时退出码仍是 0、PDF 照生）。
#
# 但把**正文**分栏、**表格与代码块**留在栏外（通栏）是可行的，效果也明显：
#   单栏：      66 页，正文行中位 **45** 字
#   选择性两栏：61 页，正文行中位 **22** 字
# 用户报的"还剩几个字就换行"正是长行造成的——行越短，末行残字的突兀越轻。
#
# **关键约束：`longtable` 不能进 `multicols`**。所以这里**不改写 `longtable`**。
# 早先试过把它转成 `tabular` + 浮动 `table` / 非浮动通栏块，引入了两类事故：
#   1. `.replace("\\endhead","")` 留下**空行**，而 `tabular` 里的空行 = `\par`
#      = 非法 → 报 `Argument of \@xnext has an extra }.`（这个坑花了一轮排查）；
#   2. 转成非浮动 `\par\noindent...` 时，**表格行会被静默截掉**——实测
#      `[成立]` 从 15 次变 14 次，因为 `\vbox` 溢出 305pt 把内容推出了页面。
# 保持 `longtable` 原样、只把它留在 `multicols` 外面，这两个坑都不存在。
_TWOCOL_FULLWIDTH = re.compile(
    r"\{\\footnotesize.*?\\end\{longtable\}\}"                  # 1 表格（长表）
    r"|\\begin\{quote\}\\ttfamily\\footnotesize\n"
    r"\\begin\{verbatim\}.*?\\end\{verbatim\}\n\\end\{quote\}"  # 2 代码块
    r"|\\noindent\\begin\{minipage\}\{\\linewidth\}.*?\\end\{minipage\}"  # 3 图
    r"|\\begin\{equation\*\}.*?\\end\{equation\*\}"              # 4 公式
    r"|^\\phantomsection\\addcontentsline\{toc\}\{(?:sub)*section\}[^\n]*"  # 5a 无编号标题
    r"|^\\(?:sub)*section\*?\{[^\n]*",                              # 5b 编号标题
    re.S | re.M)

# 6 短正文段：可见字数低于这个阈值就不分栏。栏宽半页、一行约 22 字，
# 120 字 ≈ 每栏 3 行——再短的内容分栏后会出现「一栏满、一栏空」。
_TWOCOL_MIN_CHARS = 120


def _plain_len(seg: str) -> int:
    """粗略估计一段的**可见字数**（扣掉 TeX 命令与定界符的占用）。"""
    s = re.sub(r"\\[a-zA-Z@]+\*?", "", seg)     # 命令名（allowbreak 等）
    s = re.sub(r"[{}$\\&]", "", s)              # 分组/数学/转义定界符
    return len(re.sub(r"\s", "", s))

# 代码块里有一行含超长行内代码（`\phantomsection\addcontentsline{toc}{...}`），
# 它在**半栏**里放不下（实测 71.9pt 溢出、文字被切出页面）。这里给它补断点。
# 只在分栏版里加——单栏放得下，不需要动。
_TWOCOL_BREAKABLE = (
    r"{\ttfamily\small \textbackslash{}phantomsection\textbackslash{}"
    r"addcontentsline\{toc\}\{.\allowbreak{}.\allowbreak{}.\allowbreak{}\}}",
    r"{\ttfamily\small \textbackslash{}phantomsection\allowbreak"
    r"\textbackslash{}addcontentsline\allowbreak\{toc\}\{\allowbreak"
    r".\allowbreak.\allowbreak.\allowbreak.\}}",
)


def make_two_column(body: str) -> str:
    """恒等返回——2026-09-14 起 GUIDE.pdf 整篇单栏，不再分栏。

    保留函数名与签名：调用点与测试都引用它，恒等实现让"单栏"成为
    可断言的行为（全文不得出现 `multicols`）。
    """
    return body


def main() -> int:
    if not os.path.exists(SRC):
        print(f"找不到 {SRC}", file=sys.stderr)
        return 1

    body = make_two_column(md_to_tex_body(open(SRC, encoding="utf-8").read()))
    preamble = PREAMBLE  # 单栏（2026-09-14）：multicol 随分栏一并撤下
    tex = preamble + body + "\n\\end{document}\n"

    # --- 自检：转换后不得残留未展开的 LaTeX 命令字面量 ---
    # 曾经因为「先转义再套命令」的顺序错误，产出
    # `\textbackslash{}textbf\{...\}` 这类字面量——排版能过，但正文里
    # 明晃晃一串反斜杠。这里在写盘前就拦住，不让它进 PDF。
    #
    # **但行内代码必须排除**（2026-09-13 修）：GUIDE 里有大量
    # `` `\item` ``、`` `\href` ``、`` `Overfull \hbox` `` 这类**故意展示
    # LaTeX 字面量**的代码片段——文档要讲转换器，就必须能引用这些命令。
    # 它们渲染成等宽文本是**正确**的，不是泄漏。此前不排除，于是
    # §4.8.8 一写就被误报成 3 处泄漏、整个 PDF 拒绝生成。
    # 判据：注释/代码块（verbatim）与 `\ttfamily` 分组内都不是泄漏现场。
    scan = tex
    scan = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "",
                  scan, flags=re.S)
    scan = _strip_tex_group(scan, r"\ttfamily")

    leaks = []
    for pat, why in [
        (r"\\textbackslash\{\}", "反斜杠被转义（命令顺序错）"),
        (r"\\textbackslash\{\}textbf", "\\textbf 被二次转义（经典顺序 bug）"),
        (r"\\textbackslash\{\}bfseries", "\\bfseries 被二次转义"),
        (r"\\textbackslash\{\}href", "\\href 被二次转义"),
        (r"\\textbackslash\{\}emph", "\\emph 被二次转义"),
        (r"[\x00\x01\x02]", "占位符未还原（stash 丢失）"),
    ]:
        n = len(re.findall(pat, scan))
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

    # --- 自检 2：Unicode 字母/符号必须全部被 EMOJI_MAP 消化 ---
    # 漏映射的字符会以裸 Unicode 进正文，交给 ctex 的 CJK 后备字体渲染：
    # 字号、基线、字宽都按汉字来，与邻接的数学符号不匹配。这类错误**编译
    # 永远成功**，只是排版变丑，所以没人会收到信号。
    # 这条守卫是 σ 被行尾注释吞掉那次事后加的（10 处受影响，看了一个月）。
    # 判据：正文（非注释、非 verbatim）里不允许残留 EMOJI_MAP 覆盖范围内的字符。
    unmapped = _find_unmapped_symbols(body)
    if unmapped:
        leaks = [f"  ✗ 未映射的符号（会在 PDF 里按 CJK 字体渲染，与数学符号不匹配）:"
                 f" {', '.join(f'{c!r}(U+{ord(c):04X})×{n}' for c, n in unmapped)}"]
        print("转换自检失败——未写盘：", file=sys.stderr)
        print("\n".join(leaks), file=sys.stderr)
        print("  处理：在 scripts/build_guide_pdf.py 的 EMOJI_MAP 里加上映射"
              "（注意别把它写在别的条目的行尾注释里）", file=sys.stderr)
        return 5

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

    # --- 自检 3：排版溢出 ---
    # **这是本项目最该有、却一直缺席的一条**：Overfull \hbox 意味着内容被
    # 挤到页边距之外——读者看到的是被切掉的数字（§0.4 那张表实测 20 处，
    # 最大 28.4pt ≈ 1cm），而 xelatex **退出码仍是 0**，`make check` 全绿。
    # 也就是说：排版坏了整整一个版本，而所有自动化检查都通过。
    #
    # 之所以放在**编译之后**（不是转换自检里）：Overfull 是 TeX 排版决定的，
    # 只有真编译才知道；转换器只能估列宽，估得准不准要 TeX 来裁决。
    log_path = os.path.join(ROOT, "guide.log")
    if os.path.exists(log_path):
        log = open(log_path, encoding="utf-8", errors="replace").read()

        # --- 自检 4：**静默丢字**（2026-09-13 新增）---
        # `Missing character: There is no X (U+....) in font [...]` 意味着
        # 那个字符在 PDF 里**什么都没有**——不是方框、不是乱码，是空缺。
        # xelatex 退出码 0、Overfull 也可能为 0，所以此前完全无信号：
        # 实测全文 71 处，其中圈号 ①–⑦ 占 55 处（§4.8.7 的七个小节标题
        # 与 §4.7.1.1 的编号整个消失），用户报的"有的字没有出来"就是这条。
        #
        # 判据要**排除代码块**：`verbatim` 里排不出的字符也是同样现象，
        # 但那里的内容是"照原样展示"，改它反而错。本检查只关心正文/表格。
        miss = re.findall(
            r"Missing character: There is no (\S+) \(U\+([0-9A-Fa-f]+)\)",
            log)
        if miss:
            # 统计 + 去重；字体名里带 lmmono 的多来自 verbatim，单独提示
            from collections import Counter
            cnt = Counter(f"{ch} (U+{cp})" for ch, cp in miss)
            print(f"\n排版自检：{len(miss)} 处字符**在 PDF 里没有出来**"
                  f"（xelatex 静默跳过，退出码 0）：", file=sys.stderr)
            for k, v in cnt.most_common(10):
                print(f"  ✗ {k} × {v}", file=sys.stderr)
            print("  处理：把这些字符加进 EMOJI_MAP（给一个能排版的替换），"
                  "或改用能显示它们的写法。\n"
                  "  注意：代码块（verbatim）里的排不出字符是同样现象，"
                  "若确认来自代码块可在此处放行。", file=sys.stderr)
            return 7
        print("排版自检：0 处丢字 ✓")

        # --- 自检 5：**书签标题被静默截断/丢字**（2026-09-13 新增）---
        # `Package hyperref Warning: Token not allowed in a PDF string` 意味着
        # hyperref 在读标题、准备写进 PDF 书签时，遇到了不能在"PDF 字符串"
        # 上下文中表示的 token——它**直接删掉那一段**，然后继续。
        # 退出码 0、Overfull 0、Missing character 0：正文里那个符号好端端地
        # 印着（它是 `$\\rightarrow$`），只有侧栏的目录名字少了东西。
        #
        # 最典型的形态是"箭头消失、留下双空格"：
        #   侧栏  「1.1 核心想法：伴随放电  副本，负像  运行统计，残差  触发」
        #   正文  「1.1 核心想法：伴随放电 → 副本，负像 → 运行统计，残差 → 触发」
        # 读者会以为目录坏了（用户就是这么报的）。实测 13 处。
        #
        # 判据不是"警告数为 0"——`\texorpdfstring` 在展开**正文**那一半时
        # 也会触发同样的警告（hyperref 要检查它），而书签那半是干净的。
        # 真正的判据在下面自检 6：**看产物里有没有丢失的痕迹**。
        # 这里只把警告列出来当提示，不当作失败判据。
        tok = re.findall(r"Token not allowed in a PDF string[^\n]*\n"
                         r"\(hyperref\)\s+removing `([^']*)' on input line (\d+)",
                         log)
        if tok:
            print(f"\n排版自检（提示）：{len(tok)} 处 token 未进入 PDF 书签。"
                  f"若侧栏对应条目不正常，检查 render_heading 是否漏了 "
                  f"texorpdfstring 的书签参数：", file=sys.stderr)
            for name, line in tok[:10]:
                print(f"  · 去掉 {name!r}（guide.tex 第 {line} 行）",
                      file=sys.stderr)

        boxes = [" ".join(m.split())
                 for m in re.findall(r"(Overfull \\hbox[^\n]*?pt too wide)",
                                     log)]
        if boxes:
            print(f"\n排版自检：{len(boxes)} 处内容超出页边距——"
                  f"编译成功但排版是坏的：", file=sys.stderr)
            for b in boxes[:8]:
                print("  ✗ " + b, file=sys.stderr)
            if len(boxes) > 8:
                print(f"  … 另有 {len(boxes) - 8} 处", file=sys.stderr)
            print("  处理：调 render_table 的列宽权重，或把过宽的表格拆开"
                  "（§0.4 的 19 行大表就是靠拆分解决的）", file=sys.stderr)
            return 6
        # **成功消息只能在真检查之后打印**（2026-09-13 修）。
        # 原来这行 "0 处溢出 ✓" 写在上面的 hbox 检查**之前**，于是无论
        # 有没有溢出都会先打印一次"0 处溢出"——**一个恒真的自检输出**，
        # 比没有更坏：它让人以为已经查过了。实测当时 log 里明明有
        # `Overfull \hbox (35.2pt too wide)`，屏幕上却写着"0 处溢出"。
        print(f"排版自检：0 处 hbox 溢出（未超出页边距）✓")

        # --- 自检 6：**书签里出现双空格/断裂**（2026-09-13 新增）---
        # 这条检查的是**产物**（PDF 里真正的书签文字），不是 log。
        # 上面那条提示只看症状，这条看结果——即使 hyperref 不再警告，
        # 只要侧栏里出现「放电  副本」这种断裂，就说明还是有 token 被丢了。
        bad = _check_bookmark_text(OUT)
        if bad:
            print(f"\n排版自检：{len(bad)} 条 PDF 书签里有丢失字符的痕迹"
                  f"（连续空白 = 某个 token 被静默删掉）：", file=sys.stderr)
            for t in bad[:8]:
                print(f"  ✗ {t}", file=sys.stderr)
            if len(bad) > 8:
                print(f"  … 另有 {len(bad) - 8} 条", file=sys.stderr)
            print("  处理：检查 render_heading —— 带编号和未编号的标题都必须"
                  "走 `\\texorpdfstring{正文}{书签纯文本}`，书签那半不能含 "
                  "`$`、`\\` 命令或数学符号。", file=sys.stderr)
            return 8

        # --- 自检 7：**元数据里的下划线被吃掉**（2026-09-13 新增）---
        # 与自检 6 同源（都是 hyperref 的 PDF 字符串上下文），但报在另一个
        # 地方：文件属性。`cerebellum_cancel` → `cerebellumcancel`。
        badmeta = _check_pdf_metadata(OUT)
        if badmeta:
            print("\n排版自检：PDF 元数据里的下划线被静默丢掉"
                  "（hyperref 把它当数学下标）：", file=sys.stderr)
            for b in badmeta:
                print(f"  ✗ {b}", file=sys.stderr)
            print("  处理：在 \\hypersetup 的 pdfsubject/pdfkeywords 里把 `_` "
                  "写成 `\\_`。", file=sys.stderr)
            return 9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

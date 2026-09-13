"""`scripts/build_guide_pdf.py` 的转换器回归测试。

为什么这个文件必须存在：**这一层的 bug 全部是静默的**。它们不报错、不让
xelatex 失败、`make check` 依旧全绿，只在 PDF 里留下错误的内容——而 PDF 是
人类唯一会真正阅读的产物，也是最不容易被任何自动化检查碰到的地方。

实测踩过的四类（每一个都在下面有对应测试）：

  1. **原始字符串吞掉 Unicode 转义**：`r"{\\bfseries [\\u6210\\u7acb]}"` 里的
     `\\u6210` 是**字面反斜杠序列**，而 `\\u` 在 LaTeX 里是短音符重音命令
     （breve accent）。读者看到的是"给数字 6 加一个 ˇ 帽"（PDF 抽文实测
     `6̆2107̆acb`），**整个 §0.4 实验总表的状态列都是垃圾**，成因只是四个
     字符的 `r` 前缀。已存在很久，无人发现。
  2. **多字符 emoji 键永远匹配不上**：`⚠️` 是"基字符 U+26A0 + 变体选择符
     U+FE0F"，逐字符扫描只会命中单字符的 `⚠`，把 `U+FE0F` 留在原地——
     那是个 TeX 排不出的字符，把表格列顶出页边距（26 处 Overfull），
     而退出码仍是 0。
  3. **粗体/斜体内部不跑符号替换**：`**3× 更快**` 里的 `×` 以裸 Unicode 进
     正文，交给 CJK 后备字体渲染（字号与邻接的 `$\\pm$` 不一致）。
  4. **行尾注释吞掉映射条目**：`"\\u221a": ..., # √    "\\u03c3": ...` 让 σ
     那一条整行消失。

测试只调用纯函数（`convert` / `apply_emoji` / `slugify_inline`），
**不编译 LaTeX**——编译由 `check_all.py` 的独立步骤负责，那样失败原因更好定位。
"""
from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_guide_pdf import (BOOKMARK_FALLBACK, EMOJI_MAP,  # noqa: E402
                            render_heading,
                            _find_unmapped_symbols, _MULTI_CHAR_KEYS,
                            apply_emoji, convert, join_soft_wraps,
                            md_to_tex_body, pdf_bookmark_text, slugify_inline)


# ============================================ 0. 软换行合并（2026-09-13）
#
# 这一组守住"每行太短、总在分行"那个观感的**根因**，以及它连带的两个
# 真事故。GUIDE.md 按 ~72 字符硬折行写，而 Markdown 的语义是"单换行 =
# 软换行、合并成一段"——GitHub 正常，而本转换器逐行处理，于是：
#
#   段落被拆碎 2218 行 -> 2218 段（应 1751 段，其中 619 行本属 201 段）
#   跨行 **粗体** 39 处匹配不上 -> 88 个字面星号印进 PDF（26/68 页）
#   列表续行逃出 itemize 52 处 -> 项目符号下孤零零一段字


def test_soft_wrapped_lines_merge_into_one_paragraph():
    md = "这是一个很长的段落，\n它被硬折行写成了两行。\n"
    out = join_soft_wraps(md)
    assert out.strip().count("\n") == 0, f"未合并成一行: {out!r}"
    assert "两行" in out and "段落" in out


def test_blank_line_still_separates_paragraphs():
    md = "第一段。\n\n第二段。\n"
    out = join_soft_wraps(md)
    assert out.count("\n\n") == 1, f"空行应保留段落分隔: {out!r}"


def test_bold_spanning_lines_gets_paired():
    r"""跨行 `**粗体**` 必须被合并，否则正则匹配不上、字面星号进 PDF。

    这是本组里最严重的：**读者会在正文里看到 `**` 星号**。
    """
    md = "结论是**这条\n边界量准**，不是证伪。\n"
    joined = join_soft_wraps(md)
    assert joined.count("**") == 2, f"星号未成对: {joined!r}"
    tex = convert(joined)
    assert "**" not in tex, f"字面星号进了 TeX: {tex!r}"
    assert r"\bfseries" in tex


def test_list_continuation_merges_into_item():
    """列表项的缩进续行必须并进同一个 `\\item`。

    修复前续行不是 item，于是 `\\end{itemize}` 把它切在列表外面——
    正文里出现"项目符号下孤零零一段字"。
    """
    md = "1. **自检要看产物。** 这条有代价：\n   锚点命中要单独验。\n"
    joined = join_soft_wraps(md)
    assert joined.count("\n") == 1, f"续行未并入 item: {joined!r}"
    tex = convert(joined)
    items = re.findall(r"\\item (.*)", tex)
    assert len(items) == 1, f"应只有 1 个 item: {items!r}"
    assert "锚点命中要单独验" in items[0], f"续行未进 item: {items[0]!r}"


def test_quote_block_merges_inner_lines():
    """多行引用块要合并（此前逐行 join，跨行标记永远匹配不上）。"""
    md = "> 引用第一行**加粗\n> 跨行**继续。\n"
    joined = join_soft_wraps(md)
    tex = convert(joined)
    assert "**" not in tex, f"引用块里的字面星号: {tex!r}"
    assert r"\bfseries" in tex


def test_code_fence_content_is_never_merged():
    """代码块内部**不得**合并——那里的换行是内容的一部分。"""
    md = "```\nline one\nline two\n```\n"
    out = join_soft_wraps(md)
    assert out == md, f"代码块被改动了:\n{out!r}\n期望:\n{md!r}"
    assert "line one\nline two" in out


def test_table_rows_are_never_merged():
    md = "| a | b |\n| -- | -- |\n| 1 | 2 |\n"
    out = join_soft_wraps(md)
    assert out.count("\n") == 3, f"表格被合并了: {out!r}"


def test_headings_and_figures_are_not_merged():
    md = "# 标题\n正文第一行\n正文第二行\n![图](figs/x.png)\n"
    out = join_soft_wraps(md)
    lines = [ln for ln in out.split("\n") if ln.strip()]
    assert lines[0].startswith("# 标题")
    assert "正文第一行 正文第二行" in lines[1], lines
    assert lines[2].startswith("![")


def test_table_header_markdown_is_converted():
    """表头必须走 slugify_inline——修复前只做 tex_escape，标记原样进 PDF。

    实测：`| 术语 | 指什么 | **不**指什么 |` 的表头印出字面
    `**不**指什么`，是全文最后残留的 6 个星号。
    """
    md = "| a | **不**指什么 |\n| -- | -- |\n| 1 | 2 |\n"
    tex = convert(md)
    assert "**" not in tex, f"表头字面星号: {tex!r}"
    assert r"\bfseries" in tex


def test_whole_guide_has_no_literal_asterisks_after_convert():
    """端到端：真 GUIDE.md 走**生产管线**后不得残留字面 `**`。

    这条是"用户看得见"的最终判据——86 个字面星号曾经分布在 26/68 页。
    走 `md_to_tex_body`（而不是自己重演那三步）才能覆盖**接线**：
    实测过"main() 里漏掉 join_soft_wraps"这种 bug 在自演三步的测试下
    全绿。
    """
    path = os.path.join(ROOT, "GUIDE.md")
    if not os.path.exists(path):
        pytest.skip("GUIDE.md 不存在")
    tex = md_to_tex_body(open(path, encoding="utf-8").read())
    # 先移除**全部**行内代码渲染与代码块——那里出现 `**` 是**正确**的
    # （作者在引用字面的星号，例如“字面 `**` 有 86 处”）。
    # 用生产代码的 `_strip_tex_group` 剥分组，而不是自己写正则：
    # 表格路径是 `{\ttfamily ...}`、正文路径是 `{\ttfamily\small ...}`，
    # 自写正则只覆盖后者，会把表格里的代码误判成泄漏。
    from build_guide_pdf import _strip_tex_group
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", tex,
                  flags=re.S)
    for macro in (r"\ttfamily", r"\ttfamily\small"):
        body = _strip_tex_group(body, macro)
    body = "\n".join(ln for ln in body.split("\n")
                     if not ln.lstrip().startswith("%"))
    n = body.count("**")
    assert n == 0, f"转换后仍有 {n} 处字面星号（读者会看到）"


def test_pipeline_actually_merges_soft_wraps():
    """生产管线必须真的调用了合并——守住"接线"而不只是函数本身。

    判据锚定独立真值：真 GUIDE.md 的输出行数必须显著少于源行数。
    若 main()/md_to_tex_body 丢了 join_soft_wraps，这里立刻转红。
    """
    path = os.path.join(ROOT, "GUIDE.md")
    if not os.path.exists(path):
        pytest.skip("GUIDE.md 不存在")
    raw = open(path, encoding="utf-8").read()
    n_src = len(raw.splitlines())
    body = md_to_tex_body(raw)
    # 段落级合并后，TeX 里的空行数应显著少于源行数
    n_blank = sum(1 for ln in body.split("\n") if not ln.strip())
    assert n_blank < n_src * 0.8, (
        f"管线似乎没合并软换行: 源 {n_src} 行，TeX 空行 {n_blank}"
        f"（未合并时每行后面都有一个空行，约等于 {n_src}）")


def test_whole_guide_paragraph_count_is_sane():
    r"""端到端：软换行必须真的被合上——判据是**合并这一步之后还剩多少相邻正文行**。

    **必须走生产入口 `md_to_tex_body`，不能自己调 `join_soft_wraps`。**
    这条守卫前后失败三次，每次原因不同、都值得记：

    1. **判据是比值**：`n_joined < n_src * 0.85` 把"合并有效"和"源文件有多长"
       绑死。2026-09-13 做段落合并后源文件变短（2315 → 1959 行），比值涨到
       0.88，守卫变红——**合并逻辑一个字没改，是判据过期了**。
    2. **绕过了生产路径**：改判据后仍自己调 `join_soft_wraps(raw)`，于是把
       `main()` 里那行 `md = join_soft_wraps(md)` 删掉时，守卫**照样绿**
       （注入验证当场抓到）。它守的是"这个函数是对的"，不是"管线真的调了它"。
    3. **拿 Markdown 和 LaTeX 比**：改成量 `md_to_tex_body` 的输出后，
       输出的相邻行数是源文件的 4.8 倍——因为输出里多了 `\begin{verbatim}`
       这类**块级 LaTeX 行**，而我的 `BLOCK` 正则只认 Markdown 的块标志。
       拿两种格式的行去比，比出来的是格式差异，不是合并效果。

    现在只量 **`join_soft_wraps` 这一步的输入与输出**（两边都是 Markdown，
    可比），且从生产入口拿到输入——这样既走对了路径，又不会掉进格式陷阱。
    """
    path = os.path.join(ROOT, "GUIDE.md")
    if not os.path.exists(path):
        pytest.skip("GUIDE.md 不存在")
    raw = open(path, encoding="utf-8").read()

    # **从生产入口取"合并前"的那份**：`md_to_tex_body` 第一步会去掉一级标题，
    # 我们用同样的规则复现，确保两边可比。
    md = re.sub(r"^#{1}\s+.*?\n", "", raw, count=1)
    joined = join_soft_wraps(md)

    # 块级行：标题/表格/引用/代码围栏/图/分隔线/列表
    BLOCK = re.compile(r"^(#{1,6} |\||> |```|!\[|-{3,}|[-*] |\d+\. )")

    def leftover(text: str) -> int:
        L = text.split("\n")
        n = 0
        for i in range(1, len(L)):
            a, b = L[i - 1], L[i]
            if not a.strip() or not b.strip():
                continue          # 空行隔开 → 本就是两段
            if BLOCK.match(a) or BLOCK.match(b):
                continue          # 块级结构不算软换行
            n += 1
        return n

    n_after = leftover(joined)
    n_before = leftover(md)
    assert n_after < n_before * 0.5, (
        f"合并后仍剩 {n_after} 处相邻正文行（合并前 {n_before}）——"
        f"软换行没有被合上")


# ===================================== 0b. 静默丢字（2026-09-13）
#
# 用户报"有的字没有出来"。根因：xelatex 遇到字体里没有的字符会**静默跳过**
# ——不报错、退出码 0、PDF 照生，那个位置就是空的。实测全文 71 处：
# 圈号 ①–⑦ 占 55 处（§4.8.7 七个小节标题整个没有编号），代码块里的框线
# （├ ─ └）与数学符号（≈ ⇒ ≡）占 19 处。
#
# 这一类**必须靠 log 里的 `Missing character` 发现**——"编译成功"给不出
# 任何信号，PDF 抽文里那个字符干脆不存在。


def test_circled_numbers_are_mapped():
    r"""圈号 ①–⑦ 必须有映射，否则在 PDF 里**整个消失**。

    实测：§4.8.7 的七个小节标题写成 `#### ① ...`，PDF 里编号位置空白
    （`Missing character: There is no ① (U+2460) in font [lmroman10]`）。
    """
    for ch, want in [("\u2460", "(1)"), ("\u2461", "(2)"), ("\u2462", "(3)"),
                     ("\u2463", "(4)"), ("\u2464", "(5)"), ("\u2465", "(6)"),
                     ("\u2466", "(7)")]:
        assert ch in EMOJI_MAP, f"{ch} 不在 EMOJI_MAP 里（PDF 会丢字）"
        assert apply_emoji(ch) == want, f"{ch} -> {apply_emoji(ch)!r}"


def test_no_unmapped_droppable_chars_in_docs():
    r"""真 GUIDE.md 里**不得**出现既无映射、又不在代码块里的"可丢字符"。

    这是"读者会看到空缺"的直接判据。列出已知的易丢类别（圈号/框线/星号/
    数学角括号），要求它们在非代码位置都能被替换掉。
    """
    from build_guide_pdf import _is_block_start  # noqa: F401
    path = os.path.join(ROOT, "GUIDE.md")
    if not os.path.exists(path):
        pytest.skip("GUIDE.md 不存在")
    risky = "\u2460\u2461\u2462\u2463\u2464\u2465\u2466\u2467\u2468\u2469" \
            "\u2605\u27e8\u27e9\u2500\u2502\u251c\u2514\u253c"
    md = open(path, encoding="utf-8").read()
    in_code = False
    offenders = []
    for n, ln in enumerate(md.split("\n"), 1):
        if ln.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for ch in risky:
            if ch in ln and ch not in EMOJI_MAP:
                offenders.append((n, ch, ln.strip()[:60]))
    assert not offenders, (
        f"有 {len(offenders)} 处字符在 PDF 里会消失:\n"
        + "\n".join(f"  行 {n}: {ch!r} ← {t}" for n, ch, t in offenders[:8]))


def test_verbatim_fallback_covers_mono_gaps():
    r"""代码块里的框线/数学符号要有 ASCII 兜底（等宽字体没有这些字形）。"""
    from build_guide_pdf import VERBATIM_FALLBACK, _verbatim_safe
    for ch in "\u2500\u251c\u2514\u2248\u21d2\u2261\u27e8\u27e9":
        assert ch in VERBATIM_FALLBACK, f"{ch!r} 缺 ASCII 兜底（PDF 会丢字）"
    out = _verbatim_safe(["\u251c\u2500\u2500 x", "\u2248 1"])
    assert out[0].startswith("|--"), out
    assert "~=" in out[1], out
    # 兜底只作用于**ASCII 安全**：普通字符不许被动
    assert _verbatim_safe(["plain text 123"]) == ["plain text 123"]


def test_verbatim_fallback_covers_emoji_map():
    r"""**`VERBATIM_FALLBACK` 必须覆盖 `EMOJI_MAP` 里所有排不出的键。**

    这条是补上的，因为第一版守卫守不住（2026-09-13 实测）：
    原先只断言 8 个**手挑的**字符有兜底——`\u2500 \u251c \u2514 \u2248
    \u21d2 \u2261 \u27e8 \u27e9`，全都已经在了，于是它就是一条恒真断言。

    真实现状：`EMOJI_MAP` 有 52 个非 ASCII 键，逐个用 `{\ttfamily\small X}`
    编译后实测 **40 个在等宽字体里没有字形**，而兜底表只覆盖了 12 个。
    另外 **28 个是定时炸弹**：正文里走数学模式（没问题），但只要有人把它们
    写进行内代码或代码块，就会静默丢字。写 §4.7.3 时当场踩到——
    `① ② ③` 包在反引号里，自检 4 立刻报 3 处丢字。

    判据用**集合差**而不是抽查：抽查在"加新符号"时不会变红，集合差会。
    例外：少量字符经实测等宽字体**能**排出（`± ² ³ · × Δ Σ ⁄ ← → − √`），
    它们本来不需要兜底——但本表刻意保留了其中几条作为保守冗余，所以
    这里只查**单向**包含（EMOJI_MAP 的排不出键 ⊆ VERBATIM_FALLBACK），
    不要求反向相等。
    """
    from build_guide_pdf import VERBATIM_FALLBACK
    # 实测等宽字体（lmmono）**没有**字形的那 40 个（逐个编译验证过）
    mono_missing = "αβθλμπστ⁻↔⇒∈≈≠≡≤≥①②③④⑤⑥⑦⑧⑨⑩─└├★⚠✅✓✗❌⟨⟩⬜"
    missing = sorted(k for k in EMOJI_MAP
                     if k in mono_missing and k not in VERBATIM_FALLBACK)
    assert not missing, (
        "VERBATIM_FALLBACK 缺少等宽字体排不出的键（写进行内代码/代码块就丢字）："
        + ", ".join(f"{k!r}(U+{ord(k[0]):04X})" for k in missing))
    # 那 40 个必须**全部**在表里——这条才是真正的不变式
    covered = [c for c in mono_missing if c in VERBATIM_FALLBACK]
    assert len(covered) == len(mono_missing), (
        f"等宽字体排不出的 {len(mono_missing)} 个字符里只有 {len(covered)} 个"
        f"有兜底，缺：{''.join(c for c in mono_missing if c not in VERBATIM_FALLBACK)}")


def test_mono_safe_handles_multi_char_keys():
    r"""`_mono_safe` 也要先处理多字符键——否则留下孤立的变体选择符 U+FE0F。

    与 `apply_emoji` / `pdf_bookmark_text` 是同一个坑的第三处。
    """
    from build_guide_pdf import _mono_safe
    out = _mono_safe("\u26a0\ufe0f warning")
    assert "\ufe0f" not in out, f"留下了孤立的变体选择符: {out!r}"
    assert "!" in out, out


def test_inline_code_gets_mono_fallback():
    r"""行内代码里的 ≈ / ≡ 也要兜底——它们刻意不走数学替换，于是撞字体缺字。

    实测 `` `raw_ratio ≈ 1` `` 里的 `≈` 在 PDF 里是空的。
    注意 `~` 会被 tex_escape 转成 `\textasciitilde{}`（正确：读者看到 `~=`）。
    """
    tex = slugify_inline("`raw_ratio ≈ 1`")
    assert "\u2248" not in tex, f"行内代码里的 ≈ 未兜底: {tex!r}"
    assert "textasciitilde" in tex, f"≈ 未换成 ~=: {tex!r}"
    tex2 = slugify_inline("`(-s) ≡ (+s)`")
    assert "\u2261" not in tex2 and "==" in tex2, tex2


# ============================================ 0c. 表格列宽（2026-09-13）
#
# 用户报"表格串行太严重，比如'是否'经常一个字换行"。根因是列宽按
# `len()` 分配，而**一个汉字占两个半角宽**（实测 10.95pt vs 5.475pt）——
# 纯汉字列的权重因此被低估一半，拿到列宽后每行只放得下 5–6 个字，
# 一个 6 字表头就被排成"5 字 + 1 字"。
#
# 加上表格整体用 `\footnotesize` 后，实测"一行只有 1–2 个汉字"的行数
# 从 211 降到 126（-40%），页数 66 -> 63。


def test_display_width_counts_cjk_as_two_halves():
    r"""汉字必须按 2 个半角单位计（实测 10.95pt vs 5.475pt）。"""
    from build_guide_pdf import _display_width
    assert _display_width("汉") == 2.0
    assert _display_width("a") == 1.0
    assert _display_width("1") == 1.0
    assert _display_width("汉a") == 3.0
    # 大写字母偏宽（实测 82.1/54.7 ≈ 1.5）
    assert _display_width("A") == 1.5


def test_display_width_ignores_markdown_markers():
    r"""Markdown 标记排版后不占宽度，算进去会让加粗的格子被高估。"""
    from build_guide_pdf import _display_width
    assert _display_width("**粗**") == 2.0         # 只算「粗」= 2
    assert _display_width("`code`") == 4.0         # 只算 code = 4
    assert _display_width("[文本](url)") == 4.0    # 只算「文本」，不算 url


def test_display_width_measures_what_is_typeset_not_the_source():
    r"""**宽度要量排版后的文字，不是源文件的字符**。

    这是"低估宽度"这一类错误的**第二种形态**（第一种是 `len()` 不认汉字）。
    源文件里 `✅` 是 **1 个字符**，但 `apply_emoji` 把它渲染成
    `{\bfseries [成立]}`——**4 个可视字符**。按源文本量只有 1.0，实际要 6.0，
    **低估 6 倍**。

    后果是可见的：§0.4 那张 19 行总表的"状态"列权重被算成下限，实际宽度
    差不到 1pt 放不下 `[成立]`，于是**每一格都裂成 `[成` + `立]`**——
    实测全文 24 处。

    **注入验证**：把 `_display_width` 里那行 `t = apply_emoji(s)` 删掉，
    这条必须变红。
    """
    from build_guide_pdf import _display_width
    # 四个状态标记：源 1 个字符，渲染成 4 个可视字符
    for mark, rendered in (("✅", "[成立]"), ("❌", "[域外]"),
                           ("⚠️", "[部分]"), ("⬜", "[未测]")):
        w_mark = _display_width(mark)
        w_rendered = _display_width(rendered)
        assert w_mark == w_rendered, (
            f"`{mark}` 量出 {w_mark}，而它实际渲染成 `{rendered}`（{w_rendered}）"
            f"——宽度没按排版后的文字算，这一列会被低估")
    # 而且量出来的数必须**够大**：`[成立]` = `[`+`成`×2+`立`×2+`]` = 6.0。
    # 光有"两者相等"还不够——两者都是 1.0 也会相等（apply_emoji 被短路时）。
    assert _display_width("✅") == 6.0, (
        f"`✅` 量出 {_display_width('✅')}，应为 6.0（`[成立]` 的宽度）")


def test_display_width_strips_latex_commands_and_braces():
    r"""**剥掉 LaTeX 命令，但只剥命令本身**。

    上一条把 `apply_emoji` 的输出喂进来之后，必须把 `\bfseries` 这类控制序列
    去掉——否则量的是**反过来的错**：那 8 个字母被当成正文，`[成立]` 会量成
    **17.0**（真正的可视宽度 6.0），于是窄列被撑宽、挤掉别的列。

    两个坑一起记：
      * **没剥** → 17.0（高估 2.8 倍）
      * **剥了但没 `strip()`** → 7.0（`\bfseries` 后面那个空格被算成 1.0，
        TeX 会吃掉行首空白，不该算）
    """
    from build_guide_pdf import _display_width
    assert _display_width("{\\bfseries [成立]}") == 6.0
    assert _display_width("\\bfseries x") == 1.0        # 命令名不算，`x` 算 1
    assert _display_width("\\textbf{粗}") == 2.0        # 花括号不占宽
    # 数学定界符不占宽：`$\\times$` 只算 `\times` 剥完剩下的一点点
    assert _display_width("$\\times$") == 0.0


def test_column_weight_floor_is_high_enough_for_a_short_header_column():
    r"""列权重的**下限**要够高，否则窄列最长的格子会被折开。

    病因：下限太松（曾是 6.0）时，表头只有 1–2 个字的列拿到的绝对宽度不够。
    实测两处裂开（都在 PDF 里逐字核过）：

        `| 词 | 这里指什么 | 打个比方 |` 表 → `特征图` 裂成 `特征` + `图`
        `| 输入 | 修复前 | 危险之处 | 修复后 |` 表 → `返回` + `ndarray` 裂开

    这两张表的形状是**共性的**：表头很短（决定中位数），但**表体里有一个
    明显更长的格**。所以判据不是在某个具体表上断言，而是断言**这个形状**
    下窄列能拿到足够权重。

    实测（单变量，同一份 GUIDE）：min 6 → 13 处裂开；min 10 → 12 处；
    **min 12 → 11 处（修掉上面两处）**；min 16 → 10 处。取 12。
    """
    from build_guide_pdf import COL_WEIGHT_MIN, column_weights

    # 形状一：`| 词 | 这里指什么 | 打个比方 |`
    hdr = ["词", "这里指什么", "打个比方"]
    body = [["IMU", "x" * 30, "y" * 20],
            ["特征图", "x" * 30, "y" * 20],
            ["z 分数", "x" * 30, "y" * 20],
            ["残差", "x" * 30, "y" * 20]]
    w = column_weights(hdr, body)
    # `特征图` = 3 个汉字 = 6.0 单位。权重经过归一化后还要减 \tabcolsep，
    # 所以这里只断言下限本身够高（归一化是另一层，见 render_table）。
    assert w[0] >= 6.0, f"第一列权重 {w[0]} 连 `特征图`(6.0) 都到不了"

    # 形状二：`| 输入 | 修复前 | 危险之处 | 修复后 |`
    hdr2 = ["输入", "修复前", "危险之处", "修复后"]
    body2 = [["`.ravel()` 展平", "静默返回", "a" * 20, "b" * 15],
             ["网格参数错", "静默返回", "a" * 20, "b" * 15],
             ["时间边界", "静默返回 bool", "a" * 20, "b" * 15],
             ["统计量", "返回 ndarray", "a" * 20, "b" * 15]]
    w2 = column_weights(hdr2, body2)
    assert w2[1] >= 8.0, (
        f"「修复前」列权重 {w2[1]} 到不了 `静默返回`(8.0)——会裂成两行")

    assert COL_WEIGHT_MIN >= 12.0, (
        f"COL_WEIGHT_MIN = {COL_WEIGHT_MIN}，低于实测采用的 12.0——"
        f"窄列会重新开始裂开（实测 6.0 时有 13 处，12.0 时 11 处）")


def test_table_uses_display_width_for_columns():
    r"""表格列宽必须走**显示宽度**——纯汉字列不该被算得只有一半宽。

    判据要能**区分**两种度量（这是这个测试存在的意义）：用一张"一列长中文、
    一列长拉丁"的表，两者给出的分配完全不同。

        表头：`神经科学概念对应关系`（10 汉字） vs `LatinTokenName`（14 字母）

        len() 度量        -> [190.8pt, 270.2pt]   （把 10 汉字当 10 个单位）
        _display_width    -> [265.0pt, 196.0pt]   （当 20 个单位）

    修复前是前者：汉字列只拿到 190.8pt，而 10 个汉字需要 10×10.95 = 109.5pt
    ——看似够，但同列还有更长的内容时就会折成"一个字一行"。
    """
    from build_guide_pdf import render_table
    rows = ["| 神经科学概念对应关系 | LatinTokenName |",
            "| --- | --- |",
            "| 短 | x |"]
    tex = render_table(rows)
    ws = [float(x) for x in re.findall(r"p\{([\d.]+)\\textwidth\}", tex)]
    assert len(ws) == 2
    cjk_pt = ws[0] * 483.7
    # 汉字列必须拿到**更多**宽度（10 汉字 = 20 半角单位 > 14 字母）
    assert cjk_pt > ws[1] * 483.7, (
        f"汉字列未拿到应有宽度: 汉字列 {cjk_pt:.1f}pt vs 拉丁列 "
        f"{ws[1]*483.7:.1f}pt —— 权重按 len() 算了（汉字只算 1 个单位）")
    assert cjk_pt > 240.0, f"汉字列太窄: {cjk_pt:.1f}pt"


def test_table_uses_smaller_font():
    r"""表格整体用 \footnotesize——这是压掉碎片行的主要手段（实测 -40%）。"""
    from build_guide_pdf import render_table
    tex = render_table(["| a | b |", "| --- | --- |", "| 1 | 2 |"])
    assert r"\footnotesize" in tex, "表格未用 \\footnotesize"


def test_table_braces_are_balanced():
    r"""表格输出里的花括号必须配平（组开 `{` 必须有关闭 `}`）。

    注：`\end{longtable}}` 里的**双花括号是正确且必要的**——外层那个
    `}` 关闭 `{\footnotesize\setlength{...}` 开的分组。曾经在用
    `\begingroup` 改写时漏改收尾，导致括号不匹配。
    """
    from build_guide_pdf import render_table
    tex = render_table(["| a | b |", "| --- | --- |", "| 1 | 2 |"])
    assert tex.count("{") == tex.count("}"), (
        f"花括号不配平: {{ x{tex.count('{')} vs }} x{tex.count('}')}")


# ------------------------------------------------------- 1. 原始字符串陷阱

def test_emoji_map_values_have_no_literal_backslash_u():
    r"""**EMOJI_MAP 的中文值必须是真汉字**，不能是字面 `\uXXXX`。

    这是本项目最严重的一次排版事故：状态列的四个 emoji 全用了 r"..."，
    于是 `\u6210` 以字面形式进了 TeX，而 `\u` 是 LaTeX 的短音符重音命令。
    §0.4 整张表的状态列渲染成 `6̆2107̆acb`（给数字加帽），持续了多个版本
    都没人发现——因为**编译成功、退出码 0、每项检查全绿**。
    """
    for key, val in EMOJI_MAP.items():
        assert r"\u" not in val, (
            f"EMOJI_MAP[{key!r}] 的值含字面 \\u 转义: {val!r}\n"
            f"  TeX 会把 \\u 当成短音符重音命令，产出乱码。\n"
            f"  修法：值用普通字符串（真汉字）+ 双反斜杠写 LaTeX 命令，"
            f"不要加 r 前缀整段原始化。")


def test_status_markers_render_as_chinese():
    r"""四个状态标记必须产出**中文领域词**，而不是转义序列、也不是"是否"。

    ❌ 曾经映射成「否」——那让 PDF 直接印出一个"否"字，而 §0.5 花两段说明
    "那些 ❌ 不是这个方法失败了"。**表格与正文自相矛盾**。
    现在统一成中性的：成立 / 部分 / 域外 / 未测。
    """
    expect = {"\u2705": "成立", "\u274c": "域外",
              "\u26a0": "部分", "\u2b1c": "未测"}
    for src, word in expect.items():
        out = apply_emoji(src)
        assert word in out, f"{src!r} → {out!r}，里面没有 {word!r}"
    # ❌ 不得再产出"否"（那正是 §0.5 反对的读法）
    assert "否" not in apply_emoji("\u274c"), "❌ 又印成了「否」"


# ------------------------------------------- 2. 多字符 emoji 与变体选择符

def test_multichar_keys_exist_and_are_ordered():
    """多字符键必须存在且按长度降序——先长后短才不会拆错。"""
    assert _MULTI_CHAR_KEYS, "EMOJI_MAP 里应当有多字符键（如 ⚠️）"
    assert _MULTI_CHAR_KEYS == sorted(_MULTI_CHAR_KEYS,
                                      key=len, reverse=True)


@pytest.mark.parametrize("text", [
    "\u26a0\ufe0f",             # ⚠️ 基字符 + 变体选择符
    "前 \u26a0\ufe0f 后",
    "\u26a0",                   # 裸 ⚠ 也要能处理
])
def test_no_variation_selector_leaks(text):
    r"""变体选择符 U+FE0F **不能漏进正文**。

    它是"给前一个字符换字形"的修饰符，TeX 排不出来。漏进去会撑爆表格列
    （实测 26 处 Overfull 的最大贡献者），而编译仍然成功。
    """
    out = apply_emoji(text)
    assert "\ufe0f" not in out, (
        f"{text!r} → {out!r} 残留了 U+FE0F。\n"
        f"  成因：apply_emoji 逐字符扫描，而 ⚠️ 是多字符键，永远匹配不上，"
        f"于是被拆成 ⚠ + 孤立的 U+FE0F。修法：先做多字符替换再逐字符扫。")


@pytest.mark.parametrize("text", [
    "0.162\u00b10.077",         # ± 在数字中间（纯 ASCII 串，无汉字）
    "3\u00d7 更快",              # ×
    "x\u00b2",                  # ²
    "5% \u2264 10%",            # ≤
    "\u221a2",                  # √
    "a \u2261 b",               # ≡
])
def test_symbols_replaced_even_without_cjk(text):
    """**纯 ASCII 串里的符号也必须被替换**。

    快速路径曾写成 `if not any(ord(c) > 0x2000 for c in s): return s`，
    于是只要字符串里没有汉字就整串跳过——而 `±`(U+00B1)、`×`(U+00D7)、
    `²`(U+00B2)、`·`(U+00B7)、`−`(U+2212) **全在 U+2000 以下**。
    影响约 100 处，包括表格里所有形如 `0.162±0.077` 的数字。
    """
    out = apply_emoji(text)
    assert out != text, f"{text!r} 完全没被替换（快速路径判据错）"
    assert "$" in out, f"{text!r} → {out!r} 没有产出数学命令"


# ------------------------------------------------- 3. 粗体/斜体内部的符号

def test_symbols_inside_bold_are_replaced():
    r"""`**3× 更快**` 里的 `×` 必须被替换。

    `wrap_bold` 把内容换成占位符后，外层那次 `apply_emoji` 已经看不到它了，
    所以内层必须自己跑一遍。漏了这步则**所有加粗文字里的符号都漏替换**，
    而且是静默的（只影响观感）。
    """
    tex = convert("**3\u00d7 更快**")
    assert "\u00d7" not in tex, f"粗体里的 × 漏进 TeX: {tex!r}"
    assert r"$\times$" in tex


def test_symbols_inside_italic_are_replaced():
    tex = convert("*\u00b1 5%*")
    assert "\u00b1" not in tex, f"斜体里的 ± 漏进 TeX: {tex!r}"


# ----------------------------------------------------- 4. 行尾注释吞映射

def test_every_greek_letter_in_docs_is_mapped():
    r"""**进 PDF 的文档**里出现的希腊字母必须都在 EMOJI_MAP 里。

    曾经 `"\u221a": r"$\surd$",` 后面同一行挤进了 σ 的映射，而它在 `#` 注释
    之后——于是 σ 整条失效，在 `guide.tex` 里以裸 Unicode 进正文，交给
    ctex 的 CJK 后备字体渲染（字号/基线按汉字，与邻接的 `$\mu$` 不一致）。
    13 处受影响，很容易被归因成"字体就是这样"。

    检查范围**只含真正进 PDF 的源文档**（`GUIDE.md`）。`MEMORY_FORMAT.md`
    与 README 不经过这个转换器（README 在 GitHub 上由浏览器渲染），
    对它们提这条要求是**假警报**——而假警报会让人开始忽略这条守卫，
    那比没有守卫更糟。
    """
    p = os.path.join(ROOT, "GUIDE.md")
    missing: dict[str, int] = {}
    for ch in open(p, encoding="utf-8").read():
        o = ord(ch)
        greek = 0x0370 <= o <= 0x03FF
        math = 0x2200 <= o <= 0x22FF
        if (greek or math) and ch not in EMOJI_MAP:
            missing[ch] = missing.get(ch, 0) + 1
    assert not missing, (
        "GUIDE.md 里出现但 EMOJI_MAP 没有映射的符号"
        "（在 PDF 里会按 CJK 字体渲染，与邻接数学符号不一致）:\n"
        + "\n".join(f"  {ch!r} U+{ord(ch):04X} ×{n}"
                    for ch, n in sorted(missing.items(), key=lambda kv: -kv[1])))


# ------------------------------------------------------- 5. 断行点正确性

@pytest.mark.parametrize("code,want_break_at", [
    ("AcquisitionLayer", "Acquisition"),   # 驼峰边界
    ("InsufficientDataError", "Insufficient"),
    ("runs/a.json", "runs/"),              # 符号后
    ("features[:3]", "[:"),               # 下标前
])
def test_table_code_gets_breakpoints(code, want_break_at):
    r"""表格里的长标识符必须能断行。

    它们无法断行时会把整行顶出页边距——实测 §5.3 溢出 90.9pt（≈3.2cm），
    读者看到的是被切掉的文字。`\allowbreak` 写在 `\texttt{}` 内部是安全的
    （最小复现验证过），旧注释里"表格禁用 \allowbreak"的说法只对**嵌套在
    {\bfseries ...} 里**的情况成立。
    """
    tex = slugify_inline("`" + code + "`", in_table=True)
    assert r"\allowbreak" in tex, f"{code!r} 没有断点: {tex!r}"
    assert want_break_at in tex


def test_plain_filenames_in_tables_get_breakpoints():
    r"""表格里**普通文本**的带点文件名也要能断。

    `GUIDE.pdf` 在普通文本里不可断（TeX 只在空格与连字符处断行），
    这就是最后一处溢出（4.77pt）的成因。
    """
    tex = slugify_inline("重建 GUIDE.pdf", in_table=True)
    assert r"\allowbreak" in tex, f"`GUIDE.pdf` 没有断点: {tex!r}"


def test_body_text_gets_no_extra_breakpoints():
    """正文不该被插入多余的断点——那里有 17cm 宽度，文件名放得下。"""
    tex = slugify_inline("重建 GUIDE.pdf", in_table=False)
    assert r"\allowbreak" not in tex, f"正文被插了断点: {tex!r}"


# ------------------------------------------------- 6. 守卫自身的精确性

def test_unmapped_guard_reports_body_but_excuses_code():
    """`_find_unmapped_symbols` 必须**该报的报、该放过的放过**。

    误报（把代码块里的 α 当漏映射）会让人开始忽略这条守卫；漏报则等于没有
    守卫。两者都会让上一轮那些事故重新发生。
    """
    sig, eqv, times = "\u03c3", "\u2261", "\u00d7"
    # 应当报出
    assert _find_unmapped_symbols(f"正文 {times} {sig}")
    assert _find_unmapped_symbols(r"{\bfseries 3" + times + "}")
    # 应当放过
    assert not _find_unmapped_symbols(
        r"\begin{verbatim} a " + times + r" \end{verbatim}")
    assert not _find_unmapped_symbols(r"\texttt{a " + times + r" b}")
    assert not _find_unmapped_symbols(
        r"{\ttfamily\small raw " + eqv + r" 1}")
    assert not _find_unmapped_symbols("% " + times + " 注释里的\n正文")


def test_conversion_is_idempotent_on_symbols():
    """再跑一次替换不应产出 `$\\times$ $\\times$` 这类重复。

    `✗` 与 `×` 都映射到 `$\\times$`，若用 `for k,v: s.replace(k,v)` 循环，
    刚替换出的结果会被下一条规则再命中（实测踩过）。单遍扫描正是为此。
    """
    once = apply_emoji("\u2717 \u00d7")
    twice = apply_emoji(once)
    assert once == twice, f"二次替换改变了结果:\n  1 次: {once!r}\n  2 次: {twice!r}"
    assert once.count(r"$\times$") == 2, once


# -------------------------------------------- 7. 尖括号与表格内粗体（2026-09-13）

def test_angle_brackets_are_escaped():
    r"""`<` 与 `>` **在 LaTeX 里不是普通字符**：文本模式下渲染成西班牙语倒置
    标点 `¡` / `¿`。曾经漏了这两条，于是 `rule:<name>` 在 PDF 里显示成
    `rule:¡name¿`——而编译完全成功、所有检查全绿。

    这类"字符在目标语言里另有含义"的问题，是这一层最容易反复踩的坑：
    TeX 的文本模式特殊字符远不止 `\ & % $ # _ { } ~ ^`。
    """
    for src, must_not_contain in (("rule:<name>", "<"),
                                  ("a>b", ">"),
                                  ("a<b", "<")):
        tex = slugify_inline(src)
        assert must_not_contain not in tex, (
            f"{src!r} → {tex!r}，裸 {must_not_contain!r} 会渲染成倒置标点")
    assert r"\textless{}" in slugify_inline("a<b")
    assert r"\textgreater{}" in slugify_inline("a>b")


def test_bold_identifiers_in_tables_get_breakpoints():
    r"""表格里**加粗的标识符**也必须能断行。

    加粗内容在表格里通常是标识符（`selection_reason`、`AcquisitionLayer`），
    而它们不会自动断行——`selection_reason` 在粗体等宽下要约 104pt，
    而术语表第一列只有 86.6pt，整行顶出页边距 5.5pt。
    `wrap_bold` 曾只做转义+符号替换、不加断点。
    """
    tex = slugify_inline("**selection_reason**", in_table=True)
    assert r"\allowbreak" in tex, f"粗体标识符没有断点: {tex!r}"
    assert r"\bfseries" in tex


def test_body_bold_gets_breakpoints():
    r"""正文里的粗体**也要**插断点（2026-09-13 改，曾是"表格才插"）。

    原设计判断"正文有 17cm，放得下"，但那建立在**段落被拆碎**的前提上
    （每行 ~72 字符）。修好软换行合并后段落恢复完整长度，粗体就成了正文里
    唯一没有断点的构造——实测立刻出现 1 处 `Overfull \hbox (37.1pt)`，
    来自 `**seed 1/2/3/7/11/42/99/123/777/2026 ...**` 里的长 token。
    该粗体在修复前因 `**` 跨行而**根本没生效**，所以从未被测出。
    """
    tex = slugify_inline("**seed 1/2/3/7/11/42/99/123/777/2026**", in_table=False)
    assert r"\allowbreak" in tex, f"正文粗体没插断点: {tex!r}"


def test_body_bold_without_long_tokens_has_no_breakpoint():
    """短粗体不该出现断点——断点只在真正有断点字符时才插入。"""
    tex = slugify_inline("**重要**", in_table=False)
    assert r"\allowbreak" not in tex, f"短粗体被插了无谓断点: {tex!r}"


def test_code_inside_bold_keeps_symbols_literal():
    r"""粗体里的**行内代码**要保持字面——不能因为走了粗体路径就把
    `x_a` 的 `_` 变成数学下标、或把 `a×b` 的 `×` 换成 `$\times$`
    （那会拆断等宽文本）。粗体路径按占位符切分，只对真文本部分插断点。

    **判据在 2026-09-13 改过一次**：原来断言输出里含 `a\_b` 这个**精确串**，
    但那天给断点加了"闭合分组 + 重开分组"之后，输出变成
    `{\ttfamily a\_}\allowbreak{}{\ttfamily\small b}`——`a\_` 与 `b` 之间
    多了一层分组边界，**字面性完全保持**，只是精确串不再匹配。
    也就是说旧判据测的是"实现细节长什么样"，不是"语义有没有保持"。
    现在改成三条语义判据：`\_` 仍是转义下划线、没有 `$`（没被当数学）、
    花括号配平。
    """
    tex = slugify_inline("**`a_b` 与 selection_reason**", in_table=True)
    assert r"\bfseries" in tex
    # `_` 必须是**转义**形态（`\_`），不能是裸 `_`（会变数学下标）
    assert r"\_" in tex, tex
    assert "_" not in tex.replace(r"\_", ""), f"有裸下划线（会变数学下标）: {tex}"
    # 代码里的字符保持字面：不能出现 `$...$`（那说明被当数学处理了）
    assert "$" not in tex, f"行内代码里混进了数学模式: {tex}"
    # 断点插在分组边界上，必须配平
    assert tex.count("{") == tex.count("}"), f"花括号不配平: {tex}"


def test_breakpoints_do_not_break_underscore_escapes():
    r"""断点绝不能把 `\_` 拆成 `\` + `_`。

    顺序必须是"先插哨兵 → 再整体转义 → 最后把哨兵换成 \\allowbreak"。
    反过来做，刚转义出的 `\_` 会被哨兵规则命中，那个裸 `_` 逃出等宽盒子
    变成数学下标（实测踩过）。
    """
    tex = slugify_inline("**a_b_c**", in_table=True)
    assert r"\_\allowbreak{}b" in tex, tex
    assert not re.search(r"(?<!\\)\\_?[^\\]_", tex.replace(r"\_", "")), tex


# --------------------------------------------------------------- 书签标题
#
# PDF 书签标题走的是 hyperref 的"第二遍展开"：它把标题当**纯文本**读，
# 遇到 `$...$` 数学模式、`\rightarrow`、`\bfseries` 这类"需要排版才成立"的
# token 就**直接删掉**，只留一条 `Token not allowed in a PDF string` 警告。
# 退出码 0、正文里的符号好端端地印着，只有侧栏的目录名坏了。
# 实测 13 处，全是标题里的 `→`（侧栏显示成双空格：
# "伴随放电　副本"）。用户报的"目录有问题"有一部分就是这条。


def test_bookmark_text_has_no_math_markers():
    r"""书签文本里不能出现 `$`——那是被删掉的直接原因。"""
    tex = pdf_bookmark_text("1.1 核心想法：伴随放电 → 副本，负像 → 运行统计")
    assert "$" not in tex, f"书签里还有数学模式标记: {tex!r}"
    assert "\\rightarrow" not in tex, f"书签里还有数学命令: {tex!r}"


def test_bookmark_text_keeps_the_arrow_as_ascii():
    r"""箭头要变成**看得见的 ASCII**，而不是消失。

    这是这条 bug 的用户可见形态：箭头被丢后留下双空格，读者以为目录坏了。
    """
    tex = pdf_bookmark_text("2.2 启发 → 工程问题总表")
    assert "->" in tex, tex
    assert BOOKMARK_FALLBACK["\u21d2"] == "=>"
    assert not re.search(r"\s{2,}", tex), f"书签里出现了双空格: {tex!r}"


def test_bookmark_text_strips_markdown_markers():
    r"""书签里不该有 `**`、反引号、链接语法——那些是 Markdown 的字面量。"""
    tex = pdf_bookmark_text("4.5 `rotwarp` 的输入契约（**最重要**的那层）")
    assert "**" not in tex, tex
    assert "`" not in tex, tex
    assert "rotwarp" in tex, tex


def test_bookmark_text_escapes_underscore():
    r"""下划线**必须**转义成 `\_`——否则会被当成数学下标而丢掉。

    实测（/tmp 下的最小复现）：书签里写 `A\_B\_esc` 显示为 `A_B_esc`；
    写裸 `A_B_raw` 会触发 `removing 'subscript'` 警告，显示为 `AB_raw`
    （下划线消失）。§4.7.1.2 的 `fuzz_protocol.py` 就是这条。
    """
    tex = pdf_bookmark_text("4.7.1.2 ...  `fuzz_protocol.py`")
    assert r"fuzz\_protocol.py" in tex, tex


def test_bookmark_text_breaks_the_en_dash_ligature():
    r"""`--` 必须断开连字——否则书签里会显示成 en dash `–`。

    实测：书签写 `--random` 显示为 `–random`（命令行旗标被改写了）；
    写 `-{}-random` 显示为 `--random`。`p50--p99` 同理。
    """
    tex = pdf_bookmark_text("4.7.1.1 固定目录不是随机：--random 才发现")
    assert "-{}-random" in tex, tex
    assert "--" not in tex.replace("-{}-random", ""), tex


def test_bookmark_text_escapes_latex_specials():
    r"""`& % # $ _ { }` 都要转义——它们在书签里是活动字符。

    实测：`&` `%` `#` `$` 转义后显示正常；`{}` 不转义会破坏参数边界。
    """
    tex = pdf_bookmark_text("A & B 100% #tag $x {y} _z")
    for ch in "&%#$":
        assert "\\" + ch in tex, f"{ch!r} 未转义: {tex!r}"
    assert "\\_z" in tex, tex


def test_bookmark_fallback_covers_emoji_map():
    r"""**BOOKMARK_FALLBACK 必须覆盖 EMOJI_MAP 的每个非 ASCII 键。**

    这条不变式是实测踩出来的：加 `\addcontentsline` 支持无编号标题后，
    §4.7.3 那三个 `① ② ③` 小节标题在**目录页**丢了字（自检 4 报
    Missing character，位置在目录页，而真正的成因是 BOOKMARK_FALLBACK
    少了一条）——报错点和成因隔了两层。

    漏一个的后果不止"书签少个符号"：那个字符以**裸 Unicode 走进 .toc**，
    被 TeX 当成"当前字体里没有的字形"跳过，于是自检 4 在目录页报错。

    判据用"EMOJI_MAP 的键集合 - BOOKMARK_FALLBACK 的键集合"，而不是
    抽查几个熟悉的符号——后者在加新符号时不会变红。

    注意 EMOJI_MAP 里有**多字符键**（`⚠️` = U+26A0 + U+FE0F），
    所以判"非 ASCII"要看**整个键的任一字符**是否非 ASCII，不能用 ord(键)。
    """
    missing = sorted(k for k in EMOJI_MAP
                     if any(ord(c) > 127 for c in k)
                     and k not in BOOKMARK_FALLBACK)
    assert not missing, (
        "BOOKMARK_FALLBACK 缺少 EMOJI_MAP 里的键（会以裸 Unicode 进 .toc，"
        "在目录页静默丢字）："
        + ", ".join(f"{k!r}(U+{ord(k[0]):04X})" for k in missing))


def test_bookmark_fallback_values_are_pure_text():
    r"""每个替换值都必须是**纯文本**—— 不能含 `$` 或反斜杠。

    含 `$` 的值照样会被 hyperref 删（那就白换了）；更深一层：
    `\alpha` 这类命令在 PDF 字符串里会**原样印成"alpha"带个反斜杠"，
    或者被当成控制序列整个丢掉。
    """
    for k, v in BOOKMARK_FALLBACK.items():
        assert "$" not in v, f"{k!r} → {v!r} 含数学模式"
        assert "\\" not in v, f"{k!r} → {v!r} 含反斜杠命令"

def test_bookmark_text_has_nothing_unrepresentable():
    r"""书签文本里不能剩下任何"排不出的"符号。

    判据要**区分汉字与符号**：汉字可以直接出现在书签里（hyperref 会转成
    UTF-16，实测正常：「先看这里：五分钟跑通」显示无误）。排不出的是
    **符号区**的东西——它们在 PDF 字符串里要么被丢，要么触发警告。

    用"是否是 BOOKMARK_FALLBACK 的键"做判据（而不是"码位是否 > 127"），
    这样加新符号时这条守卫会自动变红——这正是它该有的行为。

    还有一个坑：**变体选择符 U+FE0F**。`⚠️` 是"基字符 + U+FE0F"，
    逐字符扫描只命中 `⚠`，把 U+FE0F 留在原地（实测产出 `[部分]️`）。
    所以单独把 U+FE0F 也列进判据。
    """
    src = ("→ ← ↔ ⇒ ≈ ≡ ≠ ≤ ≥ ∈ ± × · − √ ⁄ σ μ Σ Δ θ α β τ λ π ² ³ ⁻ "
           "①②③④⑤⑥⑦⑧⑨⑩ ★ ⟨ ⟩ ✅ ❌ ⚠️ ⬜ ✓ ✗")
    tex = pdf_bookmark_text(src)
    leaked = [c for c in tex
              if c in EMOJI_MAP or c == "\ufe0f"]
    assert not leaked, (
        f"书签里残留了排不出的符号 {leaked!r}（U+{[f'{ord(c):04X}' for c in leaked]}）"
        f"——会把它们加进 BOOKMARK_FALLBACK：{tex!r}")
    # 中文必须原样保留（不能被当成符号丢掉）
    assert pdf_bookmark_text("先看这里：五分钟跑通") == "先看这里：五分钟跑通"

def test_render_heading_gives_bookmarks_a_plain_text_twin():
    r"""带编号标题必须走 `\texorpdfstring{正文}{书签}`。

    这是 hyperref 设计的标准用法。不做的话，正文里好看的 `$\rightarrow$`
    在书签里就是消失的箭头。
    """
    tex = render_heading(3, "2.2 启发 → 工程问题总表")
    assert r"\texorpdfstring" in tex, tex
    assert "->" in tex, f"书签那一半没有 ASCII 箭头: {tex!r}"
    assert r"$\rightarrow$" in tex, f"正文那一半丢了数学箭头: {tex!r}"


def test_unnumbered_heading_gets_a_toc_entry_and_anchor():
    r"""**无编号标题也必须进目录/书签**——它是全篇的入口小节。

    `\section*` 不进目录也不进书签。实测「先看这里：五分钟跑通」整条从
    侧栏消失了（书签树里 §0 上面什么都没有），而它恰恰是读者最先要找的。

    `\phantomsection` 不能省：它给 `\addcontentsline` 一个锚点，
    否则点击目录条目会跳到**上一页**的某处（实测偏一页）。
    """
    tex = render_heading(2, "先看这里：五分钟跑通")
    assert r"\addcontentsline{toc}" in tex, tex
    assert r"\phantomsection" in tex, f"缺锚点会跳错页: {tex!r}"
    assert r"\subsection*" in tex, tex
    assert "先看这里" in tex, tex
    # 目录条目的层级要是 subsection（与 §0 的子节同级，不是与 §0 同级）
    assert r"\addcontentsline{toc}{subsection}" in tex, tex


def test_unnumbered_toc_entry_typesets_symbols():
    r"""无编号标题的**目录条目**也要走 `\texorpdfstring`，且排印那半要
    用 `slugify_inline`（把 `①` 换成 `(1)`）。

    实测教训：`\addcontentsline` 的参数同时喂给"排印的目录页"和"PDF 书签"。
    目录页用**粗体罗马字体**排标题，那字体里没有 `①` 的字形——裸 `①`
    会被静默跳过（自检 4 报在目录页，而真正的成因在这一行）。
    只给书签那一半做处理是不够的。
    """
    tex = render_heading(3, "① 逐行处理")
    # 排印那半：① 已变成 (1)
    assert "(1)" in tex, tex
    # 目录条目本身也要带 texorpdfstring
    m = re.search(r"\\addcontentsline\{toc\}\{subsubsection\}\{(.*)\}"
                  r"\\subsubsection\*", tex, re.S)
    assert m, f"目录条目结构不对: {tex!r}"
    assert r"\texorpdfstring" in m.group(1), m.group(1)
    assert "①" not in m.group(1), f"目录条目里还有排不出的 ①: {m.group(1)!r}"


def test_unnumbered_document_title_is_not_a_toc_entry():
    r"""文档大标题（`#` 一级）不该进目录——`\maketitle` 已经排了它。

    但转换管线里**先去掉了文档标题**（`md_to_tex_body` 的第一步），所以
    这里验证的是：真正被渲染的一级标题（若存在）会拿到 section 级条目。
    """
    tex = render_heading(1, "附：一个没有编号的一级标题")
    assert r"\addcontentsline{toc}{section}" in tex, tex


# --------------------------------------------------- 标题层级（目录结构正确性）
#
# 用户报的"就是目录有问题"有一部分是这里：无编号标题原先**只按 `#` 数量**
# 定层级，于是 §4.7.3（编号深度 2）下面的 `####` 子标题被渲染成同样的深度 2，
# 成了 §4.7.2 的**兄弟**而不是 §4.7.3 的**孩子**。侧栏里看起来就像
# 「4.7.2 / 4.7.3 / ① 逐行处理」三个平级小节——内容归属完全读错了。
#
# 正确规则（经典两条）：
#   1. 编号标题：深度 = 编号段数（`4`→0, `4.1`→1, `4.4.1`→2）；
#   2. 无编号标题：深度 = 栈里最近一个更浅的标题深度 + 1。


def _depth_of(tex: str) -> int:
    """从 render_heading 的输出里读出渲染到第几层。"""
    m = re.search(r"\\(section|subsection|subsubsection|paragraph)\*?\{", tex)
    assert m, f"认不出命令: {tex!r}"
    return {"section": 0, "subsection": 1, "subsubsection": 2,
            "paragraph": 3}[m.group(1)]


def test_numbered_heading_depth_comes_from_number_not_hashes():
    r"""编号标题的层级看**编号段数**，不看 `#` 数量。

    GUIDE 里 `## §4.4`（编号深度 1）实际是 §4 的子节却写成二级标题；
    照搬 `#` 数量会让 §4.4 与 §4 平级，书签树就错了。
    """
    from build_guide_pdf import _MD_STACK
    _MD_STACK.clear()
    assert _depth_of(render_heading(2, "§4 正向交付")) == 0
    # 三级标题但编号深度 1 → 应是 subsection，不是 subsubsection
    assert _depth_of(render_heading(3, "4.1 记录格式")) == 1
    # 二级标题但编号深度 1 → 同样是 subsection（与上一行一致）
    assert _depth_of(render_heading(2, "4.4 输入契约")) == 1
    assert _depth_of(render_heading(3, "4.4.1 先量再改")) == 2


def test_unnumbered_heading_nests_under_its_parent():
    r"""**无编号标题必须成为父节的孩子**——这是用户报的目录问题之一。

    实测症状：§4.7.3（depth 2）下的 `#### ① 逐行处理` 被渲染成 depth 2，
    在侧栏里与 §4.7.2、§4.7.3 平级 → 读者以为那四段是独立小节。
    """
    from build_guide_pdf import _MD_STACK
    _MD_STACK.clear()
    assert _depth_of(render_heading(3, "4.7.3 文档工具链")) == 2
    # `####` 子标题 → 应比父节深一层
    assert _depth_of(render_heading(4, "① 逐行处理")) == 3
    # 同级的下一个 `####` 仍是 3（不该继续加深）
    assert _depth_of(render_heading(4, "② 汉字被当成半角宽")) == 3


def test_heading_level_stack_pops_on_same_or_shallower_hash():
    r"""遇到同深或更浅的 `#` 要弹栈，否则层级会越走越深。"""
    from build_guide_pdf import _MD_STACK
    _MD_STACK.clear()
    assert _depth_of(render_heading(2, "§0 为什么")) == 0
    assert _depth_of(render_heading(3, "0.1 问题")) == 1
    assert _depth_of(render_heading(4, "无编号子项")) == 2
    # 回到 `##` → 同级弹栈，新 section 仍是 0
    assert _depth_of(render_heading(2, "§1 方法")) == 0
    assert _depth_of(render_heading(3, "1.1 核心想法")) == 1
    # 再来一个 `###` 的无编号标题 → 应是 1（与 1.1 同级），不是 2
    assert _depth_of(render_heading(3, "无编号兄弟")) == 1


def test_unnumbered_toc_level_matches_its_sectioning_level():
    r"""**目录条目的层级必须与排版命令的层级一致**——这是真正的那条 bug。

    实测踩到的形态：无编号标题算出的排版命令是 `\paragraph`（第 4 层），
    但 `\addcontentsline` 被硬写成 `{subsubsection}`（第 3 层）。
    于是**侧栏里那个条目跑到上一层去了**：§4.7.3 下面的
    `#### ① 逐行处理` 与 §4.7.2、§4.7.3 显示成平级的三个小节，
    读者会以为那四段（① ② ③ ④）是独立小节而不是 §4.7.3 的内容。

    判据：同一个 `render_heading` 输出里，`\addcontentsline{toc}{X}` 的 X
    必须等于 `\X*` 的命令名。**只比较一致性**，不写死具体层数——
    这样改层级策略时这条守卫依然有效。
    """
    from build_guide_pdf import _MD_STACK
    _MD_STACK.clear()
    render_heading(3, "4.7.3 文档工具链")
    tex = render_heading(4, "① 逐行处理 → 段落被拆碎")
    m = re.search(r"\\addcontentsline\{toc\}\{(\w+)\}", tex)
    assert m, f"没有目录条目: {tex!r}"
    toc_cmd = m.group(1)
    sec = re.search(r"\\(\w+)\*\{", tex)
    assert sec, f"没有排版命令: {tex!r}"
    assert toc_cmd == sec.group(1), (
        f"目录条目层级 {toc_cmd!r} 与排版命令 {sec.group(1)!r} 不一致——"
        f"侧栏里这一条会跑到别的层级去: {tex!r}")


def test_heading_depth_is_capped_at_paragraph():
    r"""层级最深到 `\paragraph`（LaTeX 没有更深的标准层级）。

    超过 3 的深度若直接拼命令名会产出 `\foobar{}`，编译报 undefined
    control sequence——不是静默的，但会在很晚才炸。这里夹住。
    """
    from build_guide_pdf import _MD_STACK
    _MD_STACK.clear()
    render_heading(2, "§0 x")
    render_heading(3, "0.1 x")
    render_heading(4, "0.1.1 x")
    tex = ""
    for _ in range(3):
        tex = render_heading(5, "很深的无编号标题")
    assert _depth_of(tex) == 3, tex


def test_md_to_tex_body_resets_heading_stack():
    r"""**每次转换都要重置层级栈**，否则同一进程里转两次结果不同。

    `_MD_STACK` 是模块级状态；测试与生产都可能在同进程里多次调用
    `md_to_tex_body`。

    **判据要看"栈是否被清空"这个可观测的结果，而不是纠结具体层数。**
    这条守卫改了三版才有效，值得记下来为什么：

    * 第 1、2 版拿"同一个文档转两次结果相同"作判据——**恒真**，因为
      弹栈逻辑本身会把残留状态消化掉；
    * 第 3 版拿"首个标题的层级"作判据，又写错了期望值——`####` 在栈里
      有 `(2,0)` 时**本来就**该算 depth 1（它是 `##` 的孩子，深度合理）。

    真正干净、也真的能区分对错的判据是：**转换开始时栈必须是空的**。
    这直接对应那条不变式，不依赖任何层级策略的细节——将来改层级算法
    这条守卫依然有效。
    """
    from build_guide_pdf import _MD_STACK, md_to_tex_body

    # **判据的形状要能区分对错**（这条守卫前三版都恒绿）。
    # 做法：先把栈弄脏成 `[(2, 0)]`（模拟上一次转换的残留），再转换一份
    # 首标题 `#` 数量为 3 的文档。
    #
    #   * 正确（有重置）：栈 = [(3, 1)]            —— 只有本文档的标题
    #   * 错误（无重置）：栈 = [(2, 0)]            —— 残留项没被清掉，
    #                                                而新标题的 3 >= 2 弹不动它
    #
    # 两者**内容不同**，所以可区分。前几版之所以恒绿，是因为选的文档
    # 恰好让残留项被后来的弹栈消化掉，最终栈一模一样。
    _MD_STACK.clear()
    _MD_STACK.append((2, 0))
    md_to_tex_body("### 0.9 X\n\ntext\n")
    assert _MD_STACK == [(3, 1)], (
        f"转换结束后栈里只能有本文档的标题，实得 {_MD_STACK!r}"
        f"——`md_to_tex_body` 忘了重置 `_MD_STACK`（残留项会污染层级判定）")


# ------------------------------------------- 表格列宽：不许出现"孤字换行"
#
# 用户第二次提这条："表格串行太严重，比如'是否'经常一个字换行。"
# 第一轮修了**度量**（len() 不认汉字宽度）和**字号**（表格改 footnotesize），
# 但没从根上解决：`render_table` 给每列算权重时，宽列的需求量常一起超过上限，
# **同时被截断**，于是"权重"退化成"每列一样宽"——窄列（表头只有 1–2 个汉字）
# 拿到的绝对宽度不够，那一格被折成"…" + 孤零零一个字。
#
# 实测（单变量）：
#   cap 40 → 孤字 22 个 / 0 溢出
#   cap 50 → 孤字 17 个 / 0 溢出   ← 采用
#   cap 60 → 孤字 22 个 / 1 溢出


def _table_weights(header, body):
    """调用**生产实现**的列权重（不是副本）。

    曾经这里自己复制了一份算式——结果"注入验证"永远注入不进去：把生产代码里的
    `second * 0.55` 删掉，守卫照样是绿的，因为它量的是这份副本。
    现在直接 import `column_weights`，注入就能让它变红（2026-09-13 修）。
    """
    from build_guide_pdf import column_weights
    return column_weights(header, body)


def _naive_weights(header, body):
    """"没有第二长兜底"的对照算法——用来做反向断言。"""
    from build_guide_pdf import _display_width
    out = []
    for c in range(len(header)):
        lens = [_display_width(header[c])] + [_display_width(r[c])
                                              for r in body if c < len(r)]
        ls = sorted(lens)
        out.append(max(6, min(50, ls[len(ls) // 2] + max(ls) // 4)))
    return out


def test_column_weights_stay_content_dependent():
    r"""**权重必须随内容变**——不能所有列都顶到同一个上限。

    这条守卫改了三版，前两版都是恒真的，原因值得记：

    * 第 1 版断言"窄列权重 ≥6"。但窄列本来就是 `max(6, …)` 兜住的，
      把上限从 50 退回 40，窄列**照样是 6** → 断言恒真。
    * 第 2 版断言"三列权重不全等"。我用的构造里只有两列超限、第三列很小，
      所以对任何上限都不全等 → 同样恒真。

    真正会暴露退化的是：**某一列的内容明显比另一列长，但两者权重相同**。
    那说明上限把差异吃掉了——列宽与内容脱钩，必然有一列被排坏。
    构造：两列内容长度差 10 倍，权重必须**不相等**。
    """
    # 左列内容很长（>上限），右列中等——两者都在"会被上限碰到"的量级，
    # 但需求仍然差 10 倍。
    hdr = ["长" * 60, "中" * 6]
    body = [["长" * 60, "中" * 6], ["长" * 60, "中" * 6]]
    w = _table_weights(hdr, body)
    assert w[0] > w[1], (
        f"第一列内容长度是第二列的 10 倍，权重却被压成 {w}——"
        f"上限把内容差异吃掉了（第 1 列 {w[0]} vs 第 2 列 {w[1]}）")
    # 而且窄列要真的放得下它自己的内容（6 个汉字 ≈ 12 单位）
    assert w[1] >= 12, f"第二列权重 {w[1]} 放不下它自己的 6 个汉字: {w}"


def test_narrow_column_beats_the_median_trap():
    r"""**中位数会漏掉长条目**——窄列必须按"第二长"兜底。

    实测病灶（GUIDE §0.4 那张 19 行总表的"实验"列）：
        hdr 长度 4，body 长度 [114, 28, 8, 30, 22, …]
        → 中位 30，加上 max/4 得 59.5；而第二长是 112.5
        → 不放兜底，权重只有 59.5（放不下那一条），折行严重。
    兜底项 `max(w, second * 0.55)` 把它抬到 61.9。

    真实 GUIDE 上该项生效 **9 次**（216 列里），所以不是死代码。

    **这条守卫带反向断言**：同时算出"没有兜底"的版本，要求它在同一个构造上
    确实更小——否则这个用例区分不了两种算法，等于恒真。
    （前两版就是栽在这里：构造选得太温和，两种算法给出同样的权重。）
    """
    from build_guide_pdf import _display_width
    # 忠实复现**真实病灶列**的形状（GUIDE.md:689 那张"IMU 有价值 / 无价值"表的
    # "真机段（IMU 有价值）"列）：**表头中等，绝大多数格极短，第二长与最长同量级**。
    # 这正是中位数失效的形状——中位是 4（那些极短的格），第二长却是 20。
    #
    # 构造有三个约束，前几版就是栽在这里（两种算法给出同样权重 → 守卫恒真）：
    #   * 中位要**小**，使 `中位 + max//4` 够不到第二长；
    #   * 第二长要**大**，使 `第二长 × 0.55` 超过前者；
    #   * 两者都不能撞 50 上限——**一撞上限兜底就白做**，两种算法都得 50。
    hdr = ["一二三四五六七八九十一"]                  # 22.0 宽
    body = [["一二三四五六七八九十"]]                 # 20.0 ← 最长
    body += [["二一"]] * 7                            # 4.0 ×7 ← 把中位压到 4
    body += [["一二三四五六七"]]                       # 14.0
    w = _table_weights(hdr, body)
    naive = _naive_weights(hdr, body)

    assert w[0] > naive[0], (
        f"兜底没有生效：有权重 {w[0]}，去掉兜底是 {naive[0]}——构造区分不了？")
    assert w[0] < 50, (
        f"权重 {w[0]} 撞上了上限 50——那样兜底与不兜底结果相同，"
        f"这条守卫会变成恒真。换一个更温和的构造。")
    # 兜底的目的就是这个：让第二长那一条放得下
    lens = sorted([_display_width(hdr[0])] + [_display_width(r[0]) for r in body])
    assert w[0] >= lens[-2] * 0.55, (
        f"权重 {w[0]} 低于「第二长 × 0.55」={lens[-2] * 0.55:.1f}")

# ============================ 选择性分栏的结构约束（2026-09-13）
#
# 用户要求"可以选择性地两栏"。分栏能成立靠三条不变量，每条都踩过坑。


def test_two_column_keeps_tables_out_of_columns():
    r"""**`longtable` 绝不能进 `multicols`**——它会直接报
    `! Package longtable Error: longtable not in 1-column mode.`

    这条坑的可怕之处：**不加 `-halt-on-error` 时退出码仍是 0、PDF 照生**
    （只是少了内容），我一度以为编译成功了。

    **注入验证**：让 `_TWOCOL_FULLWIDTH` 不匹配表格（比如去掉 `longtable`
    那一支），这条必须变红。
    """
    from build_guide_pdf import make_two_column, md_to_tex_body
    tex = make_two_column(md_to_tex_body(open(os.path.join(ROOT, "GUIDE.md"), encoding="utf-8").read()))
    # 把正文按 multicols 分段，检查每段内部**没有** longtable
    depth = 0
    bad = []
    for line in tex.split("\n"):
        if "\\begin{multicols}" in line:
            depth += 1
        elif "\\end{multicols}" in line:
            depth -= 1
        elif depth > 0 and "\\begin{longtable}" in line:
            bad.append(line[:60])
    assert not bad, f"有 longtable 落在 multicols 内部: {bad[:3]}"


def test_two_column_wraps_prose_but_leaves_blocks_full_width():
    r"""**正文要包起来、通栏块要留在外面**。

    两个方向都要查，因为两种错误都很隐蔽：
      * 正文没包 → 分栏根本没生效（页数不变，看起来"成功了"）；
      * 通栏块被包 → `longtable` 报错（见上一条）。

    **注入验证**：把 `make_two_column` 的包裹关掉，这条必须变红。
    """
    from build_guide_pdf import make_two_column, md_to_tex_body
    tex = make_two_column(md_to_tex_body(open(os.path.join(ROOT, "GUIDE.md"), encoding="utf-8").read()))
    n_col = tex.count("\\begin{multicols}")
    n_tab = tex.count("\\begin{longtable}")
    assert n_col > 50, f"只切出 {n_col} 个分栏段——分栏没生效？"
    assert n_tab > 50, f"只剩 {n_tab} 张表——表格被改写了？"


def test_two_column_keeps_headings_figures_equations_out_of_columns():
    r"""**标题、图、公式必须留在通栏，不得被包进 `multicols`**。

    2026-09-13 二轮用户反馈"不是全部都两栏"：第一版把标题、图、公式
    全裹进了半栏——标题在栏里够不着版心宽度、图被缩到栏宽。这条钉死
    回归点：**section/subsection 级标题、figure 的 minipage、`equation*`
    显示公式，一律不得出现在 `multicols` 内部**。

    （`paragraph` 级小标题是例外：它字号小、专为正文段落服务，
    本就该跟着正文进栏——所以这里只查 `section`，不查 `paragraph`。）

    **注入验证**：把 `_TWOCOL_FULLWIDTH` 里 5b 那一支（编号标题）删掉，
    这条必须变红。
    """
    import re
    from build_guide_pdf import make_two_column, md_to_tex_body
    tex = make_two_column(md_to_tex_body(open(os.path.join(ROOT, "GUIDE.md"), encoding="utf-8").read()))
    depth = 0
    bad = []
    for line in tex.split("\n"):
        if "\\begin{multicols}" in line:
            depth += 1
        elif "\\end{multicols}" in line:
            depth -= 1
        elif depth > 0:
            # section/subsection 级标题（排除更小的 paragraph）。
            # 两种形态：无编号 `\phantomsection\addcontentsline{toc}{section|subsection}...`
            # 与编号 `\section{...} / \subsection{...}`；paragraph 级刻意放行。
            if re.match(r"^\\phantomsection\\addcontentsline\{toc\}\{(?:sub)*section\}", line) \
               or re.match(r"^\\(?:sub)*section\*?\{", line):
                bad.append(("标题", line[:60]))
            if "\\includegraphics" in line:
                bad.append(("图", line[:60]))
            if "\\begin{equation*}" in line:
                bad.append(("公式", line[:60]))
    assert not bad, f"有通栏元素被包进 multicols: {bad[:3]}"


def test_two_column_breakpoints_keep_braces_balanced():
    r"""断点会**闭合再重开**分组，必须配平。

    这条看着像形式检查，其实是踩过两次真坑换来的：
      1. 重开串少一个开括号 → 整篇 `! Extra }, or forgotten \endgroup.`，
         PDF 只出 2 页；
      2. `}` 写死在替换串里没判断 `reopen` 是否为空 → **表格/粗体那条
         共用路径**也吐 `}`，正文出现 `13.}\allowbreak{}7\%`，同样报错。

    两条都不会在转换阶段报错，只有真编译才炸——所以必须在这里钉住。
    """
    from build_guide_pdf import _tt_breakable
    for s in (r"\phantomsection\addcontentsline{toc}{...}",
              "runs/downstream.json", "InsufficientDataError",
              "a_b", "features[:3]"):
        out = _tt_breakable(s)
        # 去掉 `\{` `\}`（转义的可见花括号）再数，否则会误判
        probe = out.replace(r"\{", "").replace(r"\}", "")
        assert probe.count("{") == probe.count("}"), (
            f"{s!r} 的断点版本花括号不配平: {out!r}")

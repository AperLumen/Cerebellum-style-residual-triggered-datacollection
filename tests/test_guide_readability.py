"""GUIDE 的**可读性**守卫：专业术语第一次出现时必须有解释。

为什么需要这个文件
------------------

用户的原话是："把它当成一本大学的基础知识教科书，是一篇刚刚高中毕业的学生也
能轻松看懂的指南。" 这条要求落成可机械检查的东西并不容易——"好懂"没法直接测。
但它有一个**可以测的代理量**：**术语墙**。

实测的病灶位置很具体（2026-09-13 诊断）：`AP` / `AUC` / `oracle` / `regime` /
`LOO` 五个词**全部首现于 §0.4 那张表的第 132–152 行**，而解释它们的术语表在
第 2246 行。也就是说读者在**第一个实质章节**就撞上一堵墙，而字典在 2000 行之后。
这不是"写得不够好"，是**结构问题**：解释与使用离得太远，读者只会放弃。

所以本文件的判据不是"文风好不好"，而是那条结构不变式：

    **任何一个硬术语，在它首次出现的位置附近，必须能找到一句解释。**

"附近"取 ±80 行（够覆盖"表前的术语小抄"这种写法：解释在表前一点，术语在表里）。
"解释"的判据是**同时**满足两条：

  1. 那一行是**定义句**的形状（含 `——` 破折号解释、`指`、`即`、`= `、
     `原意`、`意思是`、`读作`、`打比方` 之类）；
  2. 那个术语**出现在同一行**。

只查"术语表里有"是不够的——那正是原来的问题（有，但在 2246 行）。

这个文件与 `test_guide_convert.py` 的分工：那边管**转换器**（Markdown→LaTeX
的机制），这边管**内容**（术语有没有解释、白话够不够）。两者都产出"用户能读的
PDF"，但坏的层完全不同。
"""
from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(ROOT, "GUIDE.md")

# 生产转换器住在 scripts/ 下，要显式加进路径才能导入（与
# tests/test_guide_convert.py 同样的做法）。
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# 解释句的形状。**必须宽松**：定义句的写法很多样，判据太严会把好句子误判成
# 没解释，然后逼着作者去凑格式——那是**测试在指挥内容**，是反的。
# 只要求"看起来在解释"的任何一个信号。
DEFINITION_MARKERS = (
    "——", "—— ", "指", "即", "= ", "原意", "意思是", "读作", "打比方",
    "比方", "比如", "就是", "又称", "简称", "俗称", "：" ,
)

# 硬术语表：高中毕业生不会认识、且本项目**大量使用**的词。
# 维护规则：**只加"读者不解释就一定看不懂"的词**，不要把通用词（如"阈值"）
# 塞进来——判据太宽会让守卫变成噪声，然后被无视。
HARD_TERMS = [
    # 评估指标（这些是 §0.4 那张表的墙）
    "AP", "AUC", "oracle", "regime", "LOO", "mAP",
    # 统计
    "p95", "零假设", "分位数", "重尾", "交叉验证",
    # 视觉 / 信号
    "光流", "亚像素", "backbone", "特征图", "z 分数", "量化", "deadband",
    "单应", "homography",
    # 神经科学 / 方法
    "伴随放电", "可塑性", "负像", "副本",
    # 工程
    "MPS", "zlib", "kNN",
]

# 允许"先出现、后解释"的例外：给出理由，可审计。
# **不要用它来掩盖问题**——它只用于"该词在这个位置是英文原文的一部分、
# 中文解释在下一段"这类真正的情形。
ALLOW = {
    # §0.4 的表头里 `AP` 与 `AUC` 并排出现，解释在表前的圆点列表里
    # （同段落，但不在同一行）。守卫的 ±80 行窗口能覆盖，无需例外。
}

WINDOW = 80


def _read(path: str) -> str:
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} 不存在")
    return open(path, encoding="utf-8").read()


def _lines() -> list[str]:
    return _read(GUIDE).split("\n")


def _looks_like_definition(line: str, term: str) -> bool:
    """这一行是否像"在解释 term"。"""
    if term not in line:
        return False
    return any(m in line for m in DEFINITION_MARKERS)


def _first_index(lines: list[str], term: str) -> int | None:
    for i, ln in enumerate(lines):
        if term in ln:
            return i
    return None


@pytest.mark.parametrize("term", HARD_TERMS)
def test_hard_term_is_explained_near_first_use(term: str):
    r"""**每个硬术语首现处附近必须有一句解释。**

    这是本文件存在的主要理由。实测病灶：`AP`/`AUC`/`oracle`/`regime`/`LOO`
    首现在 §0.4 的表里（行 132–152），而术语表在行 2246——读者在第一个实质
    章节撞墙，字典在 2000 行之后。

    Args:
        term: 待检查的术语。
    """
    lines = _lines()
    i = _first_index(lines, term)
    if i is None:
        pytest.skip(f"{term} 在 GUIDE.md 里没出现（可以把它从 HARD_TERMS 删掉）")
    if term in ALLOW:
        pytest.skip(f"{term}: {ALLOW[term]}")

    lo = max(0, i - WINDOW)
    hi = min(len(lines), i + WINDOW + 1)
    found = [j for j in range(lo, hi)
             if _looks_like_definition(lines[j], term)]
    assert found, (
        f"术语 {term!r} 首次出现在第 {i+1} 行，但前后 {WINDOW} 行内找不到"
        f"一句解释：\n"
        f"    第 {i+1} 行: {lines[i].strip()[:100]}\n"
        f"  修法：在该处**就地**加一句白话解释（「X 是指……」），而不是只把"
        f"它收进 §6 术语表——解释与使用离太远，读者会放弃。\n"
        f"  （若这个位置是英文原文、解释在下一段，把理由写进 ALLOW 字典。）")


def test_first_substantive_section_has_no_bare_metric_wall():
    r"""**§0.4 表格之前**必须先有一段评估术语说明。

    这条比上面那条更具体地钉住实测病灶：那张 19 行的表是全文的"成绩单"，
    但也是术语最密集的地方。如果表前没有小抄，读者会在第一个实质章节放弃。

    判据锚定"表前有没有一个解释块"，而不是"表里有没有看不懂的词"——
    后者无法机械判定，前者可以。
    """
    text = _read(GUIDE)
    lines = text.split("\n")
    # 找 §0.4 那张表的第一行（以 `| #` 开头的表头）
    tbl = next((i for i, ln in enumerate(lines)
                if ln.startswith("| #") and "实验" in ln), None)
    assert tbl is not None, "找不到 §0.4 的实验总表"
    # 表前 60 行内必须解释过这五个词
    head = lines[max(0, tbl - 60):tbl]
    joined = "\n".join(head)
    missing = [t for t in ("AP", "AUC", "oracle", "regime")
               if not any(_looks_like_definition(ln, t) for ln in head)]
    assert not missing, (
        f"§0.4 的表（第 {tbl+1} 行）之前没有解释这些词：{missing}\n"
        f"  实测教训：这五个词原先全部首现在表里，术语表在 2000 行之后。")


def _section_body(start_pattern: str, limit: int = 20000) -> str:
    r"""取某一节的正文，**在下一个同级标题处截断**。

    **为什么必须截断**（踩过的坑）：第一版直接取 `heading 之后 N 字符`，
    于是窗口越界读进了下一节。注入验证时把 §0.5 的白话概括整段删掉，
    测试**照样全绿**——因为它读到了 §1.1 里的"打个比方"。窗口越界会让守卫
    变成恒真断言，这是"弱守卫"最常见的形态之一。

    **两个必须注意的细节**：

    1. `start_pattern` 要匹配**行首**（调用方给的正则要带 `^` 且用 `re.M`）。
       GUIDE 里 §0.5 在一处 git 风格的检查清单里也被提到，不锚行首会先命中
       那一行，截断出 8 个字符的空节（实测踩过）。
    2. 截断点是**同级或更浅的下一个标题**。§0.5 自己的子标题是 `####`，
       用 `#{2,4}` 会在第一个子标题处就截断——那样反而测不到正文。
       所以只认 `##` 与同级 `###`，不认更深的 `####`。
    """
    text = _read(GUIDE)
    m = re.search(start_pattern, text, re.M)
    assert m, f"找不到匹配 {start_pattern!r} 的标题"
    # 从标题行的**行首**开始取正文（`m.start()` 而非 `m.end()`，
    # 前面也不会有缩进，所以直接切）
    rest = text[m.end():]
    nxt = re.search(r"(?m)^#{2,3}\s+\S", rest)
    body = rest[:nxt.start()] if nxt else rest
    return body[:limit]


def test_zero_shot_explanation_uses_plain_language():
    r"""§0.5（解释那 12 个 ❌ 的地方）必须给出**一句人话概括**。

    用户抱怨的核心是"标了一堆专业术语上来，却没有告诉你这东西最简单直接的
    怎么做"。§0.5 是全文最需要人话的地方——它要解释"为什么 12 个实验结果
    是负的，但项目没失败"，这件事对外行是反直觉的。

    判据：该节**自己的正文**（在下一个标题处截断）要有一句带"人话 / 就像 /
    说白了"之类标记的概括句。
    """
    body = _section_body(r"^###\s*0\.5\s+")
    markers = ("人话", "打个比方", "就像", "说白了", "好比", "相当于")
    assert any(k in body for k in markers), (
        "§0.5 缺少一句白话概括（应有一句'人话'把结论先说清，再展开细节）")


def test_domain_scope_is_metaphorically_explained():
    r"""§0.5 的"定义域外"必须有一个**日常类比**，不能只给术语。

    "用相机侧的量去回答世界侧的语义问题"这句话对外行等于没说。
    实测有效的写法是给一个跨领域的类比，让人立刻抓住"工具没坏、是问错了"。

    **判据必须锚定"类比结构"，不能只锚定"出现了某个名词"。** 这条守卫
    改了两版都是恒真，两次的原因不同、都值得记：

    * 第 1 版用 `("就像", "好比", ...)` —— §0.5 别处有个无关的"就像"，
      注入后照样绿。
    * 第 2 版用 `("体重秤", "尺子", ...)` 这类名词 —— **同一个名词可以
     出现在非类比的句子里**（实测："我们知道这把尺子量到哪儿为止"是个
      比喻性说法，但它在讲别的事）。注入掉真正的类比句后，那个名词还在，
      守卫依旧绿。

    所以判据要**同时**满足两条：
      1. 出现"日常物件"名词；
      2. 该名词附近有**比较/替代结构**（"拿 X 去量 Y"、"X 量 Y"、
         "用 X 去回答 Y"之类）——那才是类比。
    """
    body = _section_body(r"^###\s*0\.5\s+")
    analogies = ("体重秤", "量身高", "温度计", "血压计", "体温计", "天平",
                 "称重", "卷尺", "卡尺")
    # 类比结构：物件 + 被错用的对象。用"同一句里出现动作词"来近似。
    comparators = ("量", "称", "测")
    hits = []
    for a in analogies:
        for m in re.finditer(re.escape(a), body):
            seg = body[max(0, m.start() - 40):m.end() + 40]
            if any(c in seg for c in comparators):
                hits.append(a)
                break
    assert hits, (
        "§0.5 对'定义域外'只给了术语解释，缺一个**完整的日常类比**——"
        "这一类概念对外行必须借一个熟悉的东西来说明。\n"
        f"  （需要：{analogies} 之一 + 「量/称/测」之类的错用结构同时出现；"
        f"只有名词不算，实测那样会让守卫恒真）")


# ================================== 段落边界：行内不得藏着新段落
#
# **这条守卫是被一个真 bug 逼出来的**（2026-09-13）：做段落合并时，我把
# "软换行"和"段落边界"混为一谈，于是有 5 处**真正的段落起点被吞进上一行**——
# 最明显的是 `**意义。** 这个缺陷对所有用触发器的实验都有影响…`，
# 它挤在上一段末尾，读者看不出那是新的一段。
#
# 这类损伤**没有任何机械信号**：行数变了、PDF 也照样编出来、抽文一致、
# 0 处溢出。唯一的判据是"一段里不该出现前一段的句末标点紧跟领句粗体"。
#
# 判据的形态很关键。第一版想查"行内出现 `。 **`"，但那个形态在 **引用块和
# 列表里是合法的**（引用块常把两句并在一行）。所以这里只查**普通散文行**
# （不以 #/|/>/空白+列表符号开头），且要求句号和粗体之间正好一个空格。

_LEAD_IN_SPLIT = re.compile(r"[。！？]\s\*\*[^*。！？]{1,24}[。！？]\*\*")


def _prose_lines():
    """GUIDE 里的普通散文行（排除标题/表格/引用/列表/代码/公式）。"""
    lines = open(GUIDE, encoding="utf-8").read().split("\n")
    out = []
    in_fence = False
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not s:
            continue
        if s[0] in "#|>$-" or re.match(r"^[-*+]\s", s) or re.match(r"^\d+\.\s", s):
            continue
        out.append((i, l))
    return out


def test_no_paragraph_start_is_buried_mid_line():
    r"""**段落起点不能藏在行中间**——`…句子。**小结。** 正文…` 这种形态是错的。

    实测的 5 处病灶（2026-09-13 修）长得都一样：
        `…既不能证明、也不能否证。**怎么才能测出筛选价值。** 需要两个条件…`
    前一句属于上一段，后一句是**新一段的领句**，两者被并进同一行——读者
    读不出换段。

    **注入验证**：把任意一处修好的空行再删掉，这条必须变红。
    """
    bad = []
    for n, line in _prose_lines():
        for m in _LEAD_IN_SPLIT.finditer(line):
            seg = line[max(0, m.start() - 24):m.end() + 24]
            bad.append(f"  行{n}: …{seg}…")
    assert not bad, (
        "发现段落起点被并进上一行（句末标点 + 空格 + 领句粗体）：\n"
        + "\n".join(bad)
        + "\n  修法：在那句句末标点之后**补一个空行**，让它成为独立段落。")


def test_source_hard_wrapping_is_neutralised_by_the_pipeline():
    r"""**源文件的硬折行不会漏进 PDF**——这是"换行"这件事的真正不变式。

    用户报："很多地方其实没有必要换行。" 但要分清**两个不同的东西**：

    * **源文件 `GUIDE.md` 是按 ~72 字符硬折行的**——这是 Markdown 的常规
      写法，GitHub 上显示正常（单换行 = 软换行，会合并）。**它本身不是缺陷**。
    * **PDF 是给人读的产物**，如果转换器不合并软换行，每行就变成一个独立
      段落，读者看到满页半截行。**这才是缺陷。**

    实测（2026-09-13）：合并前 PDF 里段落被拆成 2218 段、中位行长 43 字符；
    合并后中位 **99**、段内非末行只有 **2.4%** 填充不足 80%。

    ## 为什么这条守卫不直接量源文件的行长

    我第一版就是量源文件："散文行中位 ≥70 字符"。**那条判据是错的**，而且
    错得很典型——它把"源文件怎么写"当成了"读者看到什么"。实测源文件中位
    48 字符，那条守卫**永远红**，除非把源文件也重排一遍。但重排源文件：
      * 收益为零（PDF 已经合并了）；
      * 代价是把 `git diff` 变成整篇重写，真正的改动再也看不出来。
    所以判据锚在**产物的性质**上：硬折行必须被管线中和掉。

    实测判据：数**转换输出里的"散文段落"个数**。段落 = 连续的非空、非
    LaTeX/表格行。合并之后，原来被硬折行拆开的每一截都会并回同一段，
    所以段数应当**显著少于不合并时**。

    实测：不合并 **392** 段 → 合并 **233** 段。

    **"合并后"那一半必须走 `md_to_tex_body`，不能自己调 `join_soft_wraps`。**
    这个坑我在 `test_guide_convert.py` 里已经踩过一次并写下了教训，结果在这里
    **又踩了一遍**：第一版自己调 `convert(join_soft_wraps(raw))`，于是把
    `md_to_tex_body` 里那行 `md = join_soft_wraps(md)` 删掉时——守卫**照样绿**。
    它守的是"这个函数是对的"，不是"管线真的调了它"。

    **走过两次弯路，都记在这里**（两次都是"看着合理的判据其实恒真"）：
    1. 第一版量**源文件**的行长（"中位 ≥70 字符"）。判据本身是错的：
       源文件按 72 字符硬折行是 Markdown 的常规写法，**收益在 PDF 不在这**，
       而那条守卫**永远红**，除非把源文件也重排一遍。
    2. 第二版量"输出里的多行块"。**恒真**——`convert()` 给每一行后面都补
       一个空行，所以输出里**根本不可能有多行块**（实测合并前后都是 3）。
       也就是说那个判据永远绿，跟合并有没有生效无关。
    """
    raw = re.sub(r"^#{1}\s+.*?\n", "",
                 open(GUIDE, encoding="utf-8").read(), count=1)

    def prose_paragraphs(text: str) -> int:
        n, prev = 0, False
        for line in text.split("\n"):
            s = line.strip()
            is_prose = bool(s) and not s.startswith(("\\", "|", "&", "{", "}")) \
                and "$" not in s
            if is_prose and not prev:
                n += 1
            prev = is_prose
        return n

    from build_guide_pdf import convert, md_to_tex_body
    merged = prose_paragraphs(md_to_tex_body(raw))   # ← 生产路径
    unmerged = prose_paragraphs(convert(raw))        # ← 对照：跳过合并

    assert unmerged > 300, (
        f"不合并时只有 {unmerged} 段——判据前提不成立（应 >300）")
    assert merged < unmerged * 0.75, (
        f"合并输出 {merged} 段 vs 不合并 {unmerged} 段——比例不够低，"
        f"说明硬折行没有被真正中和。\n"
        f"  检查 `md_to_tex_body` 里的 `join_soft_wraps` 是否还在调用。"
        f"（实测正常值：392 → 233，比例 0.59）")

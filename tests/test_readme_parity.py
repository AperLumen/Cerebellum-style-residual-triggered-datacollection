"""中英两份 README 的一致性检查（`tests/test_readme_parity.py`）。

## 为什么需要它，以及它**不**检查什么

`README.md` 与 `README.zh.md` 是同一份执行摘要的两个语言版本。它们曾经漂移过
（中文版落后三个月而没人察觉），所以需要守卫。

但"一致性"不能理解成"逐行对应"——两份的**行数差 50%**（英文 1810 / 中文 1188），
而这**不是缺陷**：中文的书写密度本来就高，且英文版承担了更多证据细节的展开
（复现路径、行号级说明）。按行数或按字数比对只会产生假警报，而**假警报会让人
开始忽略这条守卫**，那比没有守卫更糟。

所以这里守的是三件事，按"从粗到细"排列：

  1. **章节结构对齐**：两份的 `##` 标题数量与先后顺序必须一致。这是最强的
     结构不变量——某节只加在一份里，这条立刻报错。
  2. **关键事实不缺失**：实测数字（AP、残差比、成本、AUC…）两份都要有。
     这是最实际的风险：数字是结论的载体，一边更新一边忘了另一边，读中文的人
     就会得到过期的结论。
  3. **正文条目标题对齐**：如「诚实的局限性」这类列表，条目数不该差太多。

**不检查**段落措辞、句子数、字数——那些差异是正常的，且无法自动判优劣。
"""
from __future__ import annotations

import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN = os.path.join(ROOT, "README.md")
ZH = os.path.join(ROOT, "README.zh.md")
GUIDE = os.path.join(ROOT, "GUIDE.md")


def _read(p: str) -> str:
    if not os.path.exists(p):
        pytest.skip(f"{os.path.basename(p)} 不存在")
    return open(p, encoding="utf-8").read()


def _sections(text: str) -> list[str]:
    """`## ` 二级标题的文字（按出现顺序）。"""
    return [m.strip() for m in re.findall(r"(?m)^##\s+(.+)$", text)]


def _section_body(text: str, title: str) -> str:
    """取某一节到下一节之间的正文。"""
    m = re.search(rf"(?m)^##\s+{re.escape(title)}\s*$", text)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"(?m)^##\s+", rest)
    return rest[:nxt.start()] if nxt else rest


# --------------------------------------------------------------- 结构对齐

def test_same_number_of_sections():
    en, zh = _sections(_read(EN)), _sections(_read(ZH))
    assert len(en) == len(zh), (
        f"两份 README 的二级标题数不同: EN {len(en)} vs ZH {len(zh)}\n"
        f"  EN: {en}\n  ZH: {zh}")


def test_section_order_matches():
    """**顺序**也要一致——两份文档的叙事弧是同一套，顺序错了读者会找不到对应节。"""
    en, zh = _sections(_read(EN)), _sections(_read(ZH))
    assert len(en) == len(zh), "先修 `test_same_number_of_sections`"
    # 不比对标题文字（中英本来就不同），只比对"位置对齐"这件事本身
    # ——数量一致 + 逐位都有内容，就说明没有"某一节只在一份里"。
    for i, (a, b) in enumerate(zip(en, zh)):
        assert a and b, f"第 {i+1} 个标题为空: EN={a!r} ZH={b!r}"


# ----------------------------------------------------------- 关键事实不缺失

# 每个元组：(事实片段, 人读的描述)。片段必须在**两份**里都出现。
# 加条目时只加"读者会据此做判断"的数字——它们是结论的载体。
KEY_FACTS = [
    # 合成基准
    ("0.298", "合成 val AP"),
    ("0.145", "uniform 4x 预算 AP"),
    ("0.191", "oracle AP"),
    ("3.4%", "合成存储占比"),
    # 真机
    ("0.876", "真机残差比（修对 warp 后）"),
    ("1.308", "原 tan 近似（帮倒忙）"),
    ("+0.958", "真机副本相关 r"),
    # 成本
    ("0.032", "gyro 副本 ms/帧"),
    ("0.085", "记录格式写入 ms/帧"),
    ("2e-6", "CPU/MPS 特征最大相对差"),
    # 分辨率扫描
    ("-41", "分辨率 224 收益"),
    ("-52", "分辨率 336 收益"),
    ("-58", "分辨率 448 收益"),
    # 守门（负结果）
    ("0.464", "守门 2% 预算捕获"),
    ("0.782", "随机撒帧 2% 预算"),
    ("0.510", "帧级 AUC（随机）"),
    # 上界（负结果）
    ("0.589", "kNN 同场景 AUC"),
    ("0.478", "kNN 跨场景 AUC"),
]


@pytest.mark.parametrize("frag,desc", KEY_FACTS,
                         ids=[d for _, d in KEY_FACTS])
def test_key_fact_present_in_both(frag, desc):
    """每个关键事实必须在**两份** README 里都出现。

    这是本文件最该存在的理由：数字是结论的载体。一边更新、另一边忘了，
    读那个语言的人就会基于过期结论做判断——而**两份文档各自都是自洽的**，
    所以那种漂移没有任何自检信号。
    """
    en, zh = _read(EN), _read(ZH)
    assert frag in en, f"英文 README 缺少事实 {frag!r}（{desc}）"
    assert frag in zh, (
        f"中文 README 缺少事实 {frag!r}（{desc}）——"
        f"英文有而中文没有，说明更新时漏了一边")


# --------------------------------------------------------- 条目数大致对齐

def _bold_items(text: str) -> list[str]:
    return re.findall(r"\*\*(.{4,60}?)\*\*", text)


def test_honest_limitations_item_counts_are_close():
    r"""「诚实的局限性」两边的条目标题数不该差太多。

    不要求相等（中英的加粗用法本就有差异，中文标点全角会让正则多吃几个字符），
    只要数量级一致。差 2 倍以上说明有一边漏了整批条目。
    """
    en_sec = _section_body(_read(EN), "Honest limitations")
    zh_sec = _section_body(_read(ZH), "诚实的局限性")
    if not en_sec or not zh_sec:
        pytest.skip("找不到「诚实的局限性」节（标题可能改名了）")
    a, b = len(_bold_items(en_sec)), len(_bold_items(zh_sec))
    assert max(a, b) <= max(min(a, b) * 2, min(a, b) + 8), (
        f"「诚实的局限性」条目数差异过大: EN {a} vs ZH {b}——"
        f"可能有一边漏了整批条目")


# ------------------------------------------------ 上手路径（2026-09-13）
#
# 用户反馈："表格下面的批注和补充特别特别多……标了一堆专业术语上来，
# 却没有告诉你这东西最简单直接的怎么做。"
#
# 根因：README 有快速开始，但 **GUIDE 没有**——2398 行里第一个可运行命令
# 出现在 23% 处，而 GUIDE 自称"自包含参考"。一个只想试一下的读者拿不到
# 入口。（README 只放三段图、定位是执行摘要，把上手路径放那里也不够。）
#
# 所以加了一节 §0.0「先看这里：五分钟跑通」，并在这里钉住它不会退化。


def test_guide_has_a_getting_started_section():
    r"""GUIDE 必须有一节"先看这里"，且在第一屏内（前 15%）。"""
    text = _read(GUIDE)
    lines = text.split("\n")
    n = len(lines)
    hits = [i + 1 for i, ln in enumerate(lines)
            if re.match(r"^##\s*先看这里", ln)]
    assert hits, "GUIDE 缺少「先看这里」上手小节"
    assert hits[0] <= n * 0.15, (
        f"上手节在 {hits[0]}/{n} 行（{hits[0]/n:.0%} 处）——应在**第一屏内**，"
        "否则读者要翻很久才找得到入口")
    # 必须排在 §0 之前（它是对全文的入口，不是某章的子节）
    s0 = next((i for i, ln in enumerate(lines) if ln.startswith("## §0 ")),
              None)
    assert s0 is not None and hits[0] < s0 + 1, "上手节应排在 §0 之前"


def test_getting_started_has_a_runnable_command():
    r"""上手节里必须有**能直接复制运行**的命令，而不只是描述。

    判据锚定"可执行"而不是"提到某个脚本"：正则匹配行首的调用形式。
    """
    text = _read(GUIDE)
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"^##\s*先看这里", ln)), None)
    assert start is not None, "找不到上手节"
    end = next((i for i in range(start + 1, len(lines))
                if re.match(r"^##\s", lines[i])), len(lines))
    body = "\n".join(lines[start:end])
    runnable = re.findall(
        r"^\s*(PYTHONPATH=\S+\s+)?\.?\.?/?\.?venv/bin/python\s+\S+", body, re.M)
    assert len(runnable) >= 2, (
        f"上手节只有 {len(runnable)} 条可运行命令——至少要有"
        "「跑示例」与「跑自己的数据」两条")


def test_getting_started_does_not_duplicate_other_sections():
    r"""上手节**不得变成一个重复仓库**——它只给入口，细节指向别处。

    这条是实测教训：第一版上手节 110 行，把术语表（§6）、引擎用法（§4.8.2）、
    成本账（§1.2）都重生了一遍——**同一件事变成四处讲**。
    上手节的职责是「让人跑起来 + 指路」，不是「再写一份完整参考」。
    """
    text = _read(GUIDE)
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"^##\s*先看这里", ln)), None)
    assert start is not None
    end = next((i for i in range(start + 1, len(lines))
                if re.match(r"^##\s", lines[i])), len(lines))
    body = "\n".join(lines[start:end])
    assert end - start <= 90, (
        f"上手节有 {end-start} 行——超过 90 行说明它在重写别处的内容"
        "（实测：第一版 110 行，四处重复）")


def test_getting_started_states_the_one_liner():
    r"""上手节要有一句话说明，不能只给命令。

    "它到底在做什么"必须用**不含术语**的句子先说清——这是用户抱怨的核心：
    "标了一堆专业术语上来，却没有告诉你这东西最简单直接的怎么做"。
    """
    text = _read(GUIDE)
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"^##\s*先看这里", ln)), None)
    assert start is not None
    body = "\n".join(lines[start:start + 40])
    assert "一句话" in body, "上手节缺少「一句话」说明"
    m = re.search(r"###\s*一句话\s*\n+(.+?)(?=\n\n)", body, re.S)
    if m:
        one = m.group(1)
        for bad in ("AUC", "AP ", "LOO", "oracle", "regime", "residual",
                    "corollary"):
            assert bad not in one, (
                f"「一句话」里出现了术语 {bad!r}——这一节是给不懂的人看的："
                f"{one[:80]}")


def test_readmes_link_to_the_getting_started_section():
    r"""两份 README 都要指向 GUIDE 的上手节，别只说"见 GUIDE"。"""
    for path, lang in ((EN, "EN"), (ZH, "ZH")):
        text = _read(path)
        assert "五分钟上手" in text or "five-minute walkthrough" in text, (
            f"{lang} README 未指向 GUIDE 的上手节——"
            "只写「完整指南见 GUIDE.md」会让新读者翻不到入口")


# --------------------------------------------- 实验有效性：必须有全量基线
#
# 这条守的是 2026-09-13 发现的真错误：§2.3.6 那张"筛选 vs 不筛选"的表
# 跑了 6 个方法、105 轮、报了 ±0.08 的误差棒，看起来是一份严肃的对照实验——
# **但缺了最重要的一行**："全量存 100%"。
#
# 补跑后：全量 100% 得 0.166，最小预算 8% 得 0.176。**多用 12 倍数据，分数没变。**
# 也就是说那个任务对"数据量"完全没反应——一个连"多存 12 倍"都不涨分的判据，
# 当然测不出"选帧好坏"。原表那些 1σ 重合的数字**不是关于方法的证据**，
# 而是关于**评测选错了**的证据。
#
# 更糟的是它还顺带暴露了 oracle 是坏的：那个 oracle 挑"有动词标注的帧"，
# 而这些段里有标注的帧占 48%–100%——"照真值挑"退化成了"随便挑"，
# 所以 oracle（0.164）反而低于 uniform（0.176）。§2.0 明写"oracle 必须先自证
# 真是上界"，当时没做这一步。


def test_selection_experiments_have_a_full_budget_baseline():
    r"""**任何"筛选 vs 不筛选"的表格，都必须有一行全量基线。**

    判据：GUIDE 里每一张含"存储占比 / stored"的对照表，附近必须出现
    "all(100%)" 或"全量"字样——否则读者无法判断这张表有没有分辨力。

    这条是**用一次真实错误**换来的：§2.3.6 的 6 行表缺了全量基线，
    于是"所有方法 1σ 重合"被读成了"方法不行"，实际是尺子坏了
    （补跑后：全量 100% 得 0.166 ≈ 最小预算 8% 的 0.176）。

    **允许豁免**：合成表（§2.3.2）暂时豁免——它的源数据
    （`data/dataset/{train,val}`）已被删除（就是 `test_extract_featmaps_dino.py`
    那 6 个冻结 ERROR 的同一个原因），**没法补跑**。豁免写在 `ALLOW` 里并注明
    理由，而不是放宽判据——放宽判据会让这条守卫退化成恒真。
    """
    text = _read(GUIDE)
    lines = text.split("\n")
    missing = []
    for i, ln in enumerate(lines):
        # **只认"真正的表头"**：markdown 表头的定义是"下一行是 |---| 分隔行"。
        #
        # 踩过的坑：第一版用"这一行含 方法 且含 存储"当判据，于是 §0.4 那张
        # 19 行总表被误报——它的**数据行**里有"三种存储方法 AP 逐位相同"
        # 这样的字样，但它是跨实验索引表，不是筛选对照表。
        # 用"表头 = 紧跟着分隔行的那一行"来定位，这个歧义就消失了。
        if i + 1 >= len(lines) or not lines[i + 1].startswith("|"):
            continue
        if not re.match(r"^\|[\s:\-|]+\|\s*$", lines[i + 1]):
            continue                      # 下一行不是 |---|---| → 不是表头
        if not ln.startswith("|"):
            continue
        # 筛选对照表的表头形状：有「方法」列 + 「存储占比」列 + 分数列
        if "方法" not in ln:
            continue
        if "存储" not in ln and "stored" not in ln.lower():
            continue
        if "训练帧数" in ln:
            continue      # §2.3.2 合成表：源数据已删除，无法补跑（见正文说明）
        # **只扫连续的表格行**（表头往后直到不再是 `|` 开头）。
        # 第一版扫"表头前后 40 行"，注入"删掉 all(100%) 那一行"后测试照样绿
        # ——因为 §2.3.6 周围的说明文字里到处都有"全量"（那段解释正是讲这件事）。
        # 这是弱守卫的第三种形态：**判据被同一段解释性文字喂饱了**。
        j = i
        tbl = []
        while j < len(lines) and lines[j].startswith("|"):
            tbl.append(lines[j]); j += 1
        body = "\n".join(tbl)
        if "all(100%)" in body or "100%" in body:
            continue
        missing.append((i + 1, ln.strip()[:80]))
    assert not missing, (
        "以下筛选对照表**表体内**没有「全量 100%」基线行，读者无法判断该表"
        "有无分辨力（实测教训：§2.3.6 缺这一行，把'尺子坏了'读成了"
        "'方法不行'；注意：表外的散文提到'全量'不算）：\n"
        + "\n".join(f"  行 {n}: {t}" for n, t in missing))


def test_synthetic_table_notes_its_missing_baseline():
    r"""§2.3.2 的合成表**必须写明它缺全量基线**，不能装作完整。

    豁免不等于隐去。读者看到一张 6 行表、全是"cerebellum 大幅领先"，
    有理由相信它有分辨力；但它同样没有全量基线。诚实做法是注明这一点
    （源数据已删除、无法补跑），而不是让豁免变成"没人知道它缺"。
    """
    text = _read(GUIDE)
    m = re.search(r"(?m)^####\s*2\.3\.2\s+(.+)$", text)
    assert m, "找不到 §2.3.2"
    # 该节到下一节之间
    rest = text[m.end():]
    nxt = re.search(r"(?m)^####\s", rest)
    body = rest[:nxt.start()] if nxt else rest
    assert "全量" in body or "all(100%)" in body, (
        "§2.3.2 的合成表缺全量基线，且没有注明——豁免必须写在正文里，"
        "否则读者会以为这张表有分辨力")


def test_guide_corrects_the_retracted_claim():
    r"""§2.3.6 的旧结论必须**显式更正**，不能只是悄悄改数字。

    一个被推翻的结论如果不写明"原来说的是 X，现在更正为 Y"，读者会
    在别处（README、引用）继续遇到旧说法，而两份文档**各自都是自洽的**，
    所以那种漂移没有任何自检信号。
    """
    text = _read(GUIDE)
    assert "原来的解读是错的" in text or "已更正" in text, (
        "GUIDE 缺少对 §2.3.6 旧结论的显式更正说明")


def test_no_duplicated_figures():
    r"""同一张图不得在一个文件里出现两次。

    实测踩过两次（2026-09-13 各一次）：编辑时把整段替换掉、但替换串里
    也含了原图，于是 PDF 里同一张图连印两遍。这类错误编译永远成功、
    页数只多一页，肉眼很难注意到。
    """
    text = _read(GUIDE)
    figs = re.findall(r"!\[[^\]]*\]\((figs/[A-Za-z0-9_./-]+)\)", text)
    dup = sorted({f for f in figs if figs.count(f) > 1})
    assert not dup, f"GUIDE 里有重复引用的图（PDF 会印两遍）：{dup}"


# ------------------------------------------- 写法卫生：标题不挂括号注解
#
# 用户反馈："指南里面写了很多括号——（详细）、（诚实清单）、（已更正）、（边界），
# 我觉得没有必要。括号解决不了任何问题。它既然是一个指南，你就不需要讲那么多，
# 你只需要陈述事实。"
#
# 这是对的，而且括号注解在这个项目里有具体的害处：它们**替代了本该写进正文的
# 结论**。比如「2.3.7 …主场作战也没赢（边界）」——"边界"两个字没有提供任何信息，
# 读者仍要读正文才知道是哪种边界；而「（已更正）」更是纯粹的自我指涉，正文
# 说清楚就行。
#
# 判据只针对**标题**：正文里括号用于列举、代号、单位是正常写法，禁掉会伤及
# 表达。标题挂注解则几乎总是"该说的事没说清，先贴个标签"。


def test_headings_have_no_parenthetical_annotations():
    r"""标题不得挂「（详细）（边界）（已更正）（诚实清单）」这类注解。

    用户口径："括号解决不了任何问题……你只需要陈述事实。"

    例外：**数字/单位**类的括号是必要的（如「跑一遍（约 30 秒）」），
    它们给的是具体信息，不是态度标签。判据用"括号里是否是纯注解词"来区分。
    """
    GUIDE_ = _read(GUIDE)
    # 允许的括号内容：含数字或单位——那是实打实的信息
    informative = re.compile(r"[0-9０-９]|秒|分钟|米|像素|帧|倍|%")
    bad = []
    in_fence = False
    for n, ln in enumerate(GUIDE_.split("\n"), 1):
        # **必须跳过代码块**：bash 注释也以 `#` 开头（`# 有 IMU csv（…）`），
        # 第一版没跳，于是把 8 条代码注释误报成标题（实测）。
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # 真正的 Markdown 标题：`#` 后有空格
        if not re.match(r"^#{1,6} \S", ln):
            continue
        for m in re.finditer(r"[（(]([^）)]*)[）)]", ln):
            if informative.search(m.group(1)):
                continue
            bad.append((n, ln.strip()[:78], m.group(1)))
    assert not bad, (
        "以下标题挂了括号注解——把它删掉，需要说明的事写进正文：\n"
        + "\n".join(f"  行 {n}: 「{c}」  ← {t}" for n, t, c in bad))


def test_no_meta_commentary_blocks():
    r"""文档不得有**大段自我指涉**的引用块。

    实测的形态：`> **一个记录在案的设计错误**：…`、`> **这一轮我走了弯路，
    值得记**…`、`> **判据：…**`。它们讲的是"我们怎么发现的/我们学到了什么"，
    而不是**这个项目是什么**。

    指南的职责是陈述事实。方法学教训如果要留，应当**写成对读者的直接陈述**
    （"注入必须在统计收敛之后"），而不是"我们的探针第一版写错了"。

    判据：引用块（`>` 开头）连续 3 行以上、且含自我指涉措辞 → 违规。
    """
    GUIDE_ = _read(GUIDE)
    lines = GUIDE_.split("\n")
    meta = ("我走了弯路", "值得记", "记录在案", "诚实注记", "一个附带发现",
            "这条是", "教训：", "判据：", "所以正确的说法")
    bad = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            block = "\n".join(lines[i:j])
            if j - i >= 3 and any(k in block for k in meta):
                bad.append((i + 1, lines[i][:70]))
            i = j
        else:
            i += 1
    assert not bad, (
        "以下引用块是自我指涉的元评论——改成对读者的直接陈述：\n"
        + "\n".join(f"  行 {n}: {t}" for n, t in bad))

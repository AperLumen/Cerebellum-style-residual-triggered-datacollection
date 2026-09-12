#!/usr/bin/env python3
"""校验 GUIDE.pdf 的可跳转性与内容完整性（生成后自检）。

分三层验证，因为"编译成功"不等于"产物正确"：
  1. **书签树**：层级是否与文档语义一致（曾出现 §4.4 与 §4 平级的 bug）。
  2. **可点击链接**：目录条目与正文 §x.y 交叉引用是否真的带 GoTo 动作。
  3. **内容完整性**：从 PDF 反抽文本，逐条核对关键锚点是否都在——且要
     先把空白去掉再匹配，否则 \\allowbreak 造成的断行会被误判成"缺失"。

两个已踩过的坑，写在这里免得后人重复归因：
  * **锚点必须先在 GUIDE.md 里存在**。曾拿 `MIN_ANALYSIS_FRAMES` 当锚点，
    它在 GUIDE.md 里根本没有（只在代码与 MEMORY_FORMAT 里），于是报"PDF
    缺失"——其实是自检清单错了。本脚本现在会先核对源文档。
  * **pypdf 的纯文本抽取会把 `\\textbf` 这类命令按字体编码吐出来**，看起来
    像 LaTeX 泄漏，实际渲染是对的。判断"是否泄漏"要 grep `guide.tex`，
    不要用 PDF 抽出的文本。因此本脚本不在锚点里放带格式的整句。
"""
from __future__ import annotations

import os
import re
import sys

from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "GUIDE.pdf")
MD = os.path.join(ROOT, "GUIDE.md")

# 必须出现在 PDF 里的锚点。**全部取自 GUIDE.md 实际存在的文本**——
# 加锚点前先确认源文档里有它（本脚本会做这个核对）。
ANCHORS = [
    # 章节结构
    "§0", "§1", "§2", "§3", "§4", "§4.4", "§5", "§5.3",
    # 代码/模块名（表里的长标识符，最容易被断行破坏）。
    # 注意：只放 GUIDE.md 里**真实出现过**的路径。`contracts.py` 的全路径在
    # GUIDE.md 里没有（§4.4 只写模块名），拿它当锚点会误报"PDF 缺失"。
    "cerebellum_cancel/integrity.py",
    "cerebellum_cancel/dataset.py",
    "tests/test_contracts.py",
    "InsufficientDataError",
    "ContractError",
    # 关键数字
    "173 passed", "0.205", "0.098", "10.92", "0.033",
    # 关键结论
    "insufficient_data", "no_reference", "dt=0",
    "结论：该说与不该说的话", "三个必须保留的护栏",
    "不在残差这个量的定义域里", "生日问题",
]


def flat_text(reader: PdfReader) -> str:
    """抽取全文并去掉所有空白——用于抵抗 \allowbreak 造成的断行。"""
    txt = "".join((p.extract_text() or "") for p in reader.pages)
    return re.sub(r"\s+", "", txt)


def walk_outline(items):
    out = []

    def rec(nodes, depth):
        for n in nodes:
            if isinstance(n, list):
                rec(n, depth + 1)
            else:
                out.append((depth, n.title))

    rec(items, 0)
    return out


def main() -> int:
    if not os.path.exists(PDF):
        print(f"找不到 {PDF}", file=sys.stderr)
        return 1

    r = PdfReader(PDF)
    flat = flat_text(r)
    ok = True

    print(f"=== GUIDE.pdf 自检 ===")
    print(f"页数: {len(r.pages)}   体积: {os.path.getsize(PDF) / 1024:.0f} KB")
    print()

    # --- 1. 书签树
    outline = walk_outline(r.outline)
    depth = max((d for d, _ in outline), default=-1) + 1
    print(f"[1] 书签: {len(outline)} 条, {depth} 层")
    if len(outline) < 40:
        print("    ✗ 书签过少，层级可能没生效")
        ok = False
    else:
        print("    ✓ 数量正常")
    # §4.4 必须挂在 §4 之下（曾出过平级 bug）
    top = [t for d, t in outline if d == 0]
    if any("§4.4" in t for t in top):
        print("    ✗ §4.4 被当成一级标题（层级 bug 复发）")
        ok = False
    else:
        print("    ✓ §4.4 层级正确（不是一级）")
    print()

    # --- 2. 可点击链接
    internal = external = 0
    for p in r.pages:
        for a in (p.get("/Annots") or []):
            act = a.get_object().get("/A")
            if not act:
                continue
            s = act.get("/S")
            if s == "/GoTo":
                internal += 1
            elif s == "/URI":
                external += 1
    print(f"[2] 可点击链接: 内部(GoTo) {internal}, 外部(URI) {external}")
    if internal < 50:
        print("    ✗ 内部跳转链接过少（目录/交叉引用可能没连上）")
        ok = False
    else:
        print("    ✓ 目录与 §x.y 交叉引用都已连通")
    print()

    # --- 3. 内容完整性
    # 先核对锚点在**源文档**里存在。若源文档都没有，那是自检清单写错了，
    # 不该报成"PDF 缺失"（这个误判真的发生过）。
    src = ""
    if os.path.exists(MD):
        src = re.sub(r"\s+", "", open(MD, encoding="utf-8").read())

    bad_anchor, missing = [], []
    for a in ANCHORS:
        key = re.sub(r"\s+", "", a)
        if src and key not in src:
            bad_anchor.append(a)
        elif key not in flat:
            missing.append(a)

    print(f"[3] 内容锚点: {len(ANCHORS)} 条")
    if bad_anchor:
        print(f"    ! 自检清单有误（源文档 GUIDE.md 里就没有）: {bad_anchor}")
        ok = False
    for m in missing:
        print(f"    ✗ PDF 缺失（源文档里有）: {m}")
    if missing:
        ok = False
    if not bad_anchor and not missing:
        print(f"    ✓ 全部 {len(ANCHORS)} 条命中")
    print()

    print("结论:", "✓ 通过" if ok else "✗ 有问题")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""图构建链路（`scripts/build_figures.py` + `figs/*.tikz`）的回归测试。

图在这个项目里是**文档的骨架**，不是装饰：GUIDE 的 §0.3 与 §1.1 用它们承载
方法的结构（谁在谁之前、哪些量是并行的），这两处文字很难替代。所以图的构建
必须具备和代码同一档的可验证性——"编译过了"远不等于"图是对的"。

这里盯的是三个**静默**失效模式（每一个都真的发生过，且都不报错）：

  1. **贴边裁切**：`\\parindent` 被 ctex 重置、`\\topskip=0` 把基线钉在顶边，
     于是图贴左边缘、上边缘被切掉一截。产物照样生成，看着像"图就是这么画的"。
  2. **透明底**：sips 转出的 PNG 背景全透明。GitHub **深色模式**下透明处渲染成
     深色，黑字消失——浅色模式完全正常，所以极难被发现。
  3. **过期图**：`figures` 段登记了 4 个产物却不生成，或文档引用的图不存在，
     两者都会让"图已经没了"在审计里看起来是"有覆盖"。

测试只读产物（不重编译），因此快；重建由 `check_all.py` 的独立步骤负责——
**"有没有画对"和"有没有重新生成"是两件事**，混在一起会让失败原因难以定位。
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")

DOCS = ["GUIDE.md", "README.md", "README.zh.md"]
IMG_RE = re.compile(r"!\[[^\]]*\]\((figs/[A-Za-z0-9_-]+)\.png\)")


def _sources() -> list[str]:
    """TikZ 结构图的源（figs/*.tikz）。"""
    if not os.path.isdir(FIGS):
        return []
    return sorted(f[:-5] for f in os.listdir(FIGS)
                  if f.endswith(".tikz") and not f.startswith("_"))


def _data_figures() -> list[str]:
    """PIL 数据图（只有 .png，没有 .tikz / .pdf）。

    **两类图的产物集不同，这是决策不是遗漏**：结构图的几何是手写的，
    适合矢量 PDF；数据图的几何要从 runs/*.json 算出来，用 PIL 生成位图
    （理由见 scripts/make_data_figures.py 的模块文档）。
    所以测试不能对两类图要求同一套文件。
    """
    import re as _re
    src = os.path.join(ROOT, "scripts", "make_data_figures.py")
    if not os.path.exists(src):
        return []
    text = open(src, encoding="utf-8").read()
    blk = text.split("FIGURES = {", 1)[-1]
    return sorted(_re.findall(r'"(\w+)": figure_', blk))


def _referenced() -> dict[str, list[str]]:
    """{"figs/<name>": [引用它的文档]}。"""
    out: dict[str, list[str]] = {}
    for doc in DOCS:
        p = os.path.join(ROOT, doc)
        if not os.path.exists(p):
            continue
        for stem in IMG_RE.findall(open(p, encoding="utf-8").read()):
            out.setdefault(stem, []).append(doc)
    return out


SOURCES = _sources()
DATA_FIGURES = _data_figures()
REFERENCED = _referenced()
# 所有图（两种来源合起来）——用在与来源无关的检查里（透明度/裁切/引用）
ALL_FIGURES = SOURCES + DATA_FIGURES


# --------------------------------------------------------------- 源与登记

def test_figures_dir_exists():
    assert os.path.isdir(FIGS), "figs/ 不存在——图是这个文档项目的核心资产"


def test_at_least_one_figure_source():
    assert SOURCES, "figs/ 下没有 .tikz 源"


def test_every_source_has_both_products():
    """结构图（TikZ）必须同时有 .pdf（进 PDF）与 .png（进 README）。

    数据图不在此列——它们只有 .png，见 `_data_figures` 的说明。
    """
    for name in SOURCES:
        for ext in (".pdf", ".png"):
            path = os.path.join(FIGS, name + ext)
            assert os.path.exists(path), (
                f"{name}{ext} 缺失——跑 scripts/build_figures.py 生成"
                f"（.pdf 进 GUIDE，.png 进 README）")


def test_every_data_figure_has_a_png():
    """数据图至少要有 .png（唯一的产物）。"""
    for name in DATA_FIGURES:
        assert os.path.exists(os.path.join(FIGS, name + ".png")), (
            f"{name}.png 缺失——跑 scripts/make_data_figures.py 生成")


def test_manifest_registers_every_source_and_product():
    """manifest.figures 与磁盘必须双向一致——只登记不生成、或只生成不登记，
    都意味着这份索引已经不能回答"图从哪来"。"""
    m = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
    fig = m.get("doc_figures")
    assert isinstance(fig, dict), "manifest.json 缺少 doc_figures 段"

    reg_src = set(fig.get("sources") or [])
    reg_prod = set(fig.get("products") or [])
    reg_style = set(fig.get("styles") or [])
    # 结构图的源是 .tikz；数据图的源是脚本里的一个函数，登记成
    # "scripts/make_data_figures.py#<name>"——**不能拿它当文件路径去 os.path.exists**，
    # 否则这条检查会因为一个格式约定而误报。
    disk_src = {f"figs/{n}.tikz" for n in SOURCES}
    disk_src |= {f"scripts/make_data_figures.py#{n}" for n in DATA_FIGURES}
    disk_prod = {f"figs/{n}.png" for n in ALL_FIGURES}
    disk_prod |= {f"figs/{n}.pdf" for n in SOURCES}

    assert disk_src <= reg_src, f"有图源未登记: {sorted(disk_src - reg_src)}"
    assert disk_prod <= reg_prod, f"有产物未登记: {sorted(disk_prod - reg_prod)}"
    # 反过来：登记了就要存在。**共享样式单独归 styles**——
    # `_preamble.tex` 不是"一张图"，混进 sources 会让"每张图都有产物"
    # 这条不变量的含义变模糊。
    assert reg_src <= disk_src, f"登记了不存在的图源: {sorted(reg_src - disk_src)}"
    assert reg_prod <= disk_prod, f"登记了不存在的产物: {sorted(reg_prod - disk_prod)}"
    for s in reg_style:
        assert os.path.exists(os.path.join(ROOT, s)), f"登记了不存在的样式: {s}"
        assert not s.endswith(".tikz"), f"{s} 是图源而非样式，应放在 sources"
    for key in ("builder", "doc"):
        assert fig.get(key), f"figures.{key} 为空"    # 每张图都要写清"它在解释什么"——manifest 是深度复现索引，图没有 purpose
    # 时读者只能看图猜，而图的取舍理由（为什么用对数轴、为什么拆两层）恰恰是
    # 最不该丢的信息。
    purpose = fig.get("purpose") or {}
    all_names = set(SOURCES) | set(DATA_FIGURES)
    assert set(purpose) == all_names, (
        f"doc_figures.purpose 未覆盖全部图: "
        f"缺 {sorted(all_names - set(purpose))}，"
        f"多 {sorted(set(purpose) - all_names)}")

# ----------------------------------------------------------- 文档引用一致

def test_every_source_is_referenced_somewhere():
    """每张图都必须有文档引用——画了不用的图是纯负担：它要维护、要重建、
    还会让 `make check` 多跑一遍。"""
    unused = [n for n in ALL_FIGURES if f"figs/{n}" not in REFERENCED]
    assert not unused, f"画了但没有任何文档引用: {unused}"


def test_readme_holds_only_the_orientation_figures():
    """README 是**执行摘要**，不是完整参考（那个角色是 GUIDE）。

    所以它只放"读者需要它才能开始读"的定向图：总览、总架构、电鱼映射。
    而 §4 那些工程细节图（成本账、记录格式、两层过滤）只进 GUIDE。
    这条不变量值得钉住：README 已经 1700+ 行，再加图会让它彻底变成第二份
    GUIDE，而两份长篇文档必然分叉（README.zh 落后三个月的旧病）。
    """
    readme_figs = {s for s, ds in REFERENCED.items() if "README.md" in ds}
    guide_figs = {s for s, ds in REFERENCED.items() if "GUIDE.md" in ds}
    assert readme_figs <= guide_figs, (
        f"README 引用了 GUIDE 没有的图: {sorted(readme_figs - guide_figs)}")
    assert readme_figs == {"figs/overview", "figs/arch", "figs/mapping"}, (
        f"README 的图集变了: {sorted(readme_figs)}——若确实要加，"
        f"同步更新这条断言并想清楚 README 的定位是否还成立")

def test_figure_text_is_kept_out_of_the_canvas():
    """**图里只放视觉信息，句子放图注。**

    这条是实测逼出来的：overview 图初版把 README 的整段文字塞进画布，结果是
    "一屏小字"——读者要逐行读完才知道在说什么，比直接读正文还累。
    图的价值是"一眼看出结构"，文字的价值是"精确表述"，混在一起会同时损失两者。

    判据用**每行最大宽度**而不是总字数：长句必然撑宽，而短标签（框名、数字、
    "被自运动淹没"这类 4-6 字的注）不会。所以只要没有一行超过阈值，
    就说明没有"成句的论述"。
    """
    import subprocess
    import tempfile

    tikz = os.path.join(FIGS, "overview.tikz")
    if not os.path.exists(tikz):
        pytest.skip("没有 overview.tikz")
    text = open(tikz, encoding="utf-8").read()
    # 取所有 node 内容里的中文串，逐行看长度
    longest = 0
    worst = ""
    for line in text.splitlines():
        stripped = line.strip()
        # 跳过注释、样式定义、坐标行
        if stripped.startswith("%") or not stripped:
            continue
        cjk = "".join(c for c in stripped if "\u4e00" <= c <= "\u9fff")
        if len(cjk) > longest:
            longest, worst = len(cjk), stripped[:78]
    assert longest <= 12, (
        f"overview.tikz 里有 {longest} 个连续汉字的行，看起来是句子而非标签：\n"
        f"    {worst}\n"
        f"  图里只放标签/数字/结构；解释移到图注（README 里图下面那段）。")


def test_every_reference_resolves_to_a_source():
    known = {f"figs/{n}" for n in ALL_FIGURES}
    dangling = [s for s in REFERENCED if s not in known]
    assert not dangling, f"文档引用了不存在的图: {dangling}"


def test_manifest_has_no_duplicate_keys():
    """**JSON 重复键会静默丢数据**：`json.load` 只保留最后一个同名的键，
    前面的内容无声无息地消失——既不报错也不警告。

    这个坑真的踩到了：给 manifest 加图件登记时用了 `figures` 作键名，而
    顶层已有一个 `figures`（登记 `runs/` 下的产物图），于是那份索引被整个
    覆盖。发现它靠的是偶然翻到旧版本，不是任何检查——所以补上这条。
    """
    pairs: list[tuple[str, object]] = []

    def hook(items):
        pairs.extend(items)
        seen: dict[str, object] = {}
        for k, v in items:
            if k in seen:
                dup.append(k)
            seen[k] = v
        return seen

    dup: list[str] = []
    with open(os.path.join(ROOT, "manifest.json"), encoding="utf-8") as fp:
        json.load(fp, object_pairs_hook=hook)
    assert not dup, (
        f"manifest.json 有重复键 {sorted(set(dup))}——json.load 只保留最后一个，"
        f"同名的那份内容会被静默丢弃")
    """中文与英文 README 面向同一批读者，图集不该分叉——README.zh 曾经
    在内容上落后 README 三个月而没人察觉。"""
    for doc in ("README.md", "README.zh.md"):
        if doc not in {d for ds in REFERENCED.values() for d in ds}:
            pytest.skip(f"{doc} 未引用任何图")
    en = {s for s, ds in REFERENCED.items() if "README.md" in ds}
    zh = {s for s, ds in REFERENCED.items() if "README.zh.md" in ds}
    assert en == zh, f"两份 README 的图集不一致: EN={sorted(en)} ZH={sorted(zh)}"


# ------------------------------------------------------------- 产物质量

@pytest.mark.skipif(not ALL_FIGURES, reason="没有图")
@pytest.mark.parametrize("name", ALL_FIGURES or ["<none>"])
def test_png_is_opaque_white_background(name):
    """**深色模式守卫**：透明背景的 PNG 在 GitHub 深色模式下会把黑字吞掉。

    这是最隐蔽的一类坏法——浅色模式下完全正常，所以只有显式检查模式
    才能发现。
    """
    path = os.path.join(FIGS, name + ".png")
    assert os.path.exists(path), f"{name}.png 缺失"
    with Image.open(path) as im:
        assert im.mode in ("RGB", "L"), (
            f"{name}.png 是 {im.mode}（带 alpha）——必须合成白底，否则 "
            f"GitHub 深色模式下透明处变黑、黑字消失")
        arr = np.asarray(im.convert("RGB"))
    # 四角与四边中点必须是背景（白）。用 numpy 而不是 getpixel：后者的返回
    # 类型是 `float | tuple | None` 的联合（随 mode 变），pyright 无法收窄。
    h, w = arr.shape[:2]
    probes = arr[[0, 0, h - 1, h - 1, 0, h - 1, h // 2, h // 2],
                 [0, w - 1, 0, w - 1, w // 2, w // 2, 0, w - 1]]
    assert probes.min() > 240, (
        f"{name}.png 的边缘不是白的（min={probes.min()}）——"
        f"背景没合成白底，或图被裁到了画幅边缘")


@pytest.mark.skipif(not ALL_FIGURES, reason="没有图")
@pytest.mark.parametrize("name", ALL_FIGURES or ["<none>"])
def test_png_has_no_color_profile(name):
    """Display P3 描述会让浏览器重解释 RGB 数值 → 图与正文颜色不一致。"""
    with Image.open(os.path.join(FIGS, name + ".png")) as im:
        assert "icc_profile" not in im.info, (
            f"{name}.png 带色彩描述（sips 会写 Display P3），会导致偏色")


@pytest.mark.skipif(not ALL_FIGURES, reason="没有图")
@pytest.mark.parametrize("name", ALL_FIGURES or ["<none>"])
def test_png_is_not_cropped_at_the_edges(name):
    """**贴边裁切守卫**：内容碰到画幅边缘就说明排版没居中/留白不够。

    判据是"最外 2px 无墨"，因为线宽与抗锯齿会占掉约 1px——留白真的够时，
    边缘一定是纯背景色。
    """
    with Image.open(os.path.join(FIGS, name + ".png")) as im:
        a = np.asarray(im.convert("RGB"))
    ink = a.mean(axis=2) < 245
    assert ink.any(), f"{name}.png 是空白图"
    edge = np.concatenate([ink[:2].ravel(), ink[-2:].ravel(),
                           ink[:, :2].ravel(), ink[:, -2:].ravel()])
    ys, xs = np.where(ink)
    pad = f"留白 L{xs.min()} R{a.shape[1] - 1 - xs.max()} " \
          f"T{ys.min()} B{a.shape[0] - 1 - ys.max()}"
    assert not edge.any(), (
        f"{name}.png 有内容贴到画幅边缘（{pad}）——图的排版没居中，"
        f"或 PAD_CM 对小画幅不够")


@pytest.mark.skipif(not ALL_FIGURES, reason="没有图")
@pytest.mark.parametrize("name", ALL_FIGURES or ["<none>"])
def test_png_is_readable_at_github_width(name):
    """图在 README 里按内容宽度显示，太小等于看不清。

    1000px 是判据而非审美：GitHub 正文列宽约 880 CSS px，2x 屏需要
    ≥1760 物理像素才不糊；这里放宽到 1000 是为了不把"故意做小的图"判死。
    """
    with Image.open(os.path.join(FIGS, name + ".png")) as im:
        w, h = im.size
    assert max(w, h) >= 1000, f"{name}.png 只有 {w}x{h}，在 README 里会糊"


@pytest.mark.skipif(not SOURCES, reason="没有图源")
@pytest.mark.parametrize("name", SOURCES or ["<none>"])
def test_source_declares_its_purpose(name):
    """每个图源开头要写清"这张图在解释什么"。

    这条看起来像风格要求，实际是防止图退化成装饰：写不出目的说明时，
    这张图对读者大概率没有增量信息。
    """
    text = open(os.path.join(FIGS, name + ".tikz"), encoding="utf-8").read()
    head = "\n".join(text.splitlines()[:12])
    assert "图" in head or "Figure" in head, (
        f"{name}.tikz 开头没有说明这张图在解释什么")

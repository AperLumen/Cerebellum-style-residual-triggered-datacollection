#!/usr/bin/env python3
r"""把 figs/*.tikz 编译成 figs/*.pdf（矢量，进 GUIDE.pdf）与 figs/*.png（位图，进 README）。

为什么要这个脚本，而不是"把图直接写进 GUIDE.md"：

  1. **一份源，两处用**。GUIDE.md 要的是嵌进 PDF 的矢量图，README.md 要的是
     浏览器能显示的位图。同一个 TikZ 源同时产出两者，图不会漂移。
     文档里两份都写，语法是标准的 `![说明](figs/name.png)`——GitHub 显示位图，
     `build_guide_pdf.py` 转 PDF 时换成同名矢量 PDF。
  2. **图能单独校验**。图写在 GUIDE.md 里时，"图有没有被裁掉"只能靠肉眼看
     GUIDE.pdf 里那一页；独立编译时它就是一张一页的 PDF，可以自动查。

渲染链路（本机实测确定。TinyTeX 精简安装**没有** standalone / preview /
pdfcrop 三个宏包，所以不能用它们裁画幅）：

    第一遍  ctexart + A4 纸上排版，用 \sbox 量出 tikzpicture 的 \wd/\ht/\dp
            —— 这是 TeX 自己算出的边界，不是从日志里猜的
    第二遍  用实测画幅（+PAD_CM 留白）重排，得到紧致的一页
    导出    xelatex → figs/<name>.pdf（矢量）；sips → figs/<name>.png（位图）

**为什么必须测而不是声明画幅**：画幅是个编译期常量（TeX 排版前不知道
tikzpicture 有多大），但手写一个尺寸声明等于让作者猜——**给小了会静默裁掉
内容**，给大了留一圈白边。测量法两个毛病都没有，且第一轮放在 A4（比任何图都
大）上，所以那一轮绝不会裁。

**两条静默失败已在此脚本里堵掉，改代码时别退回去**：
  * 贴边裁切：`\parindent` 被 ctex 重置、`\topskip=0` 把基线钉在顶边，两者
    造成**不对称**的裁切（左边没白边、上边缺一块），看着像"图就是这么画的"。
    渲染模板用 `\vbox to \textheight{\vfil\centerline{...}\vfil}` 双向居中。
  * 透明底：sips 从 PDF 转出的 PNG 背景全透明，GitHub 深色模式下黑字消失。
    必须合成白底并剥掉 Display P3 色彩描述（否则浏览器按 P3 解释会偏色）。

可复现性：xelatex 遵守 SOURCE_DATE_EPOCH（GUIDE.pdf 已依赖这一点）。
PNG 走 sips，它会在文件里写元数据，于是"像素完全一样但字节不同"。因此 PDF
与 PNG 的写盘判据都是**逐字节/逐像素比较**而不是时间戳——内容没变就不动文件，
避免 `make check` 每次都留下一个假改动（这个坑 GUIDE.pdf 踩过）。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
BUILD = os.path.join(FIGS, ".build")
XELATEX = "/Users/luyu/Library/TinyTeX/bin/universal-darwin/xelatex"

# README 里图的最大像素边。1800 在 2x 屏上够清晰，单张约 200-400 KB。
PNG_MAX_DIM = 1800

# 边界留白：TikZ 的 bounding box 不含线宽/阴影的外扩量，紧贴会有"缺边"观感。
PAD_CM = 0.12

# 第一遍在 A4 上量尺寸；A4 比任何一张图都大，所以这一轮绝不会裁内容。
MEASURE_PAPER = (21.0, 29.7)

WRAPPER = r"""% 由 scripts/build_figures.py 生成——不要直接编辑本文件。
\documentclass[UTF8]{{ctexart}}
\usepackage[paperwidth={w}cm,paperheight={h}cm,margin=0pt]{{geometry}}
\input{{_preamble.tex}}
{body}\end{{document}}
"""

# 测量阶段：把图装进 \sbox 量 \wd/\ht/\dp——这是 TeX 自己算出的边界，
# 不是从日志里猜的。量完 \typeout 报数并停止排版。
MEASURE_BODY = r"""\newsavebox{{\figmeasurebox}}
\begin{{document}}
\sbox{{\figmeasurebox}}{{\input{{{name}.tikz}}}}
\typeout{{FIGMEASURE \the\wd\figmeasurebox\space \the\ht\figmeasurebox\space \the\dp\figmeasurebox}}
"""

# 渲染阶段：**必须显式居中**，不能让 TeX 按基线/\topskip 自行摆放。
#   水平——\parindent 会被 ctex 在 \begin{{document}} 时重置，内容贴左边缘；
#   垂直——\topskip=0 把基线钉在顶边，box 在基线上方的部分被版面切掉。
# 两者都是**不对称的静默裁切**（左边没白边、上边缺一块），看着就像
# "图本来就是这么画的"。用 \vbox to \textheight + \centerline 双向居中。
RENDER_BODY = r"""\begin{{document}}
\vbox to \textheight{{\vfil\centerline{{\input{{{name}.tikz}}}}\vfil}}%
"""

# 测量：把图放进一个 box 里量 \wd/\ht/\dp——这是 TeX 自己算出来的边界，
# 不是从日志里猜的。量完立即报数并停止，不产生页面。


def sources() -> list[str]:
    if not os.path.isdir(FIGS):
        return []
    return sorted(f[:-5] for f in os.listdir(FIGS)
                  if f.endswith(".tikz") and not f.startswith("_"))


def xelatex() -> str | None:
    if os.path.exists(XELATEX):
        return XELATEX
    return shutil.which("xelatex")


def write_wrapper(name: str, w: float | str, h: float | str,
                  measure: bool) -> str:
    """写包装文档。measure=True 用 A4 纸（比任何图都大，不会裁），
    measure=False 用实测画幅做紧致排版。"""
    tex = os.path.join(BUILD, f"{name}.tex")
    tpl = MEASURE_BODY if measure else RENDER_BODY
    with open(tex, "w", encoding="utf-8") as fp:
        fp.write(WRAPPER.format(w=w, h=h, body=tpl.format(name=name)))
    return tex


def run_xelatex(name: str, exe: str, env: dict,
                halt: bool = True) -> tuple[int, str]:
    cmd = [exe, "-interaction=nonstopmode",
           "-output-directory=" + BUILD, f"{name}.tex"]
    if halt:
        cmd.insert(1, "-halt-on-error")
    p = subprocess.run(cmd, cwd=FIGS, capture_output=True, text=True, env=env)
    log = p.stdout or ""
    logpath = os.path.join(BUILD, f"{name}.log")
    if os.path.exists(logpath):
        log += open(logpath, encoding="utf-8", errors="replace").read()
    return p.returncode, log


def measure(name: str, exe: str, env: dict) -> tuple[int, float, float, str]:
    """第一遍：在 A4 上排版，用 \\sbox 量出 tikzpicture 的真实尺寸（cm）。

    量的是 TeX 自己算出的 box 边界，不是从日志猜的——所以图永远不会被裁。
    A4 比任何一张图都大，这一轮不产生页面（量完就 \\end{document}）。
    """
    write_wrapper(name, *MEASURE_PAPER, measure=True)
    rc, log = run_xelatex(name, exe, env, halt=False)
    m = re.search(r"FIGMEASURE\s*([\d.-]+)pt\s*([\d.-]+)pt\s*([\d.-]+)pt", log)
    if not m:
        return 1, 0.0, 0.0, log
    wd, ht, dp = (float(g) for g in m.groups())
    return 0, wd / 28.45274, (ht + dp) / 28.45274, log


def overfull(log_text: str) -> str | None:
    r"""画幅不够时 TeX 报 Overfull \hbox——内容会被裁掉或压到画幅外。

    这是本脚本最该拦的一类失败：图被裁了以后**看起来只是"少了一块"**，
    容易归因成"TikZ 写错了"而不是"画幅给小了"。
    """
    m = re.search(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log_text)
    if m:
        return f"内容比画幅宽 {float(m.group(1)):.1f}pt（测量环节没生效？）"
    m = re.search(r"Overfull \\vbox \(([\d.]+)pt too high\)", log_text)
    if m:
        return f"内容比画幅高 {float(m.group(1)):.1f}pt（测量环节没生效？）"
    return None


def _tail(log: str, n: int = 18) -> str:
    """TeX 的报错在长日志末尾，且真错误行以 ! 开头——优先抓它。"""
    lines = log.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("!"):
            return "\n    ".join(lines[i:i + n])
    return "\n    ".join(lines[-n:])


def to_png(pdf: str, png: str) -> tuple[bool, str]:
    """PDF → PNG（macOS 自带 sips，无需 ImageMagick/poppler）。

    两步后处理都是必需的，不是洁癖：

    1. **合成到白底**。sips 从 PDF 转出来的 PNG 带 alpha 通道、背景全透明。
       GitHub README 在深色模式下会把透明处渲染成深色，于是黑字消失——
       图在浅色模式下正常、深色模式下看不见，是最难被发现的坏法。
    2. **剥掉色彩描述并转 sRGB**。sips 会写入 Display P3 描述，浏览器按 P3
       解释同一组 RGB 数值就会偏色（同仓库其它产物没有这个描述，图会和
       正文颜色不一致）。
    """
    exe = shutil.which("sips")
    if not exe:
        return False, "找不到 sips（macOS 自带；其它平台请装 poppler 的 pdftoppm）"
    tmp = png + ".raw"
    p = subprocess.run([exe, "-s", "format", "png", "-Z", str(PNG_MAX_DIM),
                        pdf, "--out", tmp], capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(tmp):
        return False, (p.stderr or p.stdout or "").strip()[:200]

    try:
        from PIL import Image
        with Image.open(tmp) as im:
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[3])
            staged = png + ".tmp"
            flat.save(staged, format="PNG", optimize=True)
        os.remove(tmp)
    except Exception as e:                                  # noqa: BLE001
        return False, f"白底/SRGB 后处理失败: {e}"

    # **看产物，不看时间戳**：像素一致就不动已有文件（sips 每次都写元数据，
    # 直接覆盖会让 git 里出现"图变了"的假改动）。
    if pixels_equal(png, staged):
        os.remove(staged)
        return True, "unchanged"
    os.replace(staged, png)
    return True, "updated"


def pixels_equal(a: str, b: str) -> bool:
    if not (os.path.exists(a) and os.path.exists(b)):
        return False
    try:
        from PIL import Image, ImageChops
        ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
        if ia.size != ib.size:
            return False
        return ImageChops.difference(ia, ib).getbbox() is None
    except Exception:
        return False


def bytes_equal(a: str, b: str) -> bool:
    if not (os.path.exists(a) and os.path.exists(b)):
        return False
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def main() -> int:
    names = sources()
    if not names:
        print(f"{FIGS} 下没有 .tikz 源", file=sys.stderr)
        return 1

    exe = xelatex()
    if not exe:
        print("找不到 xelatex", file=sys.stderr)
        return 2

    os.makedirs(BUILD, exist_ok=True)
    env = dict(os.environ)
    # 与 GUIDE.pdf 同一套可复现构建机制（见 build_guide_pdf.py 的说明）
    env.setdefault("SOURCE_DATE_EPOCH", "1789257600")
    env.setdefault("FORCE_SOURCE_DATE", "1")

    failed = []
    print(f"=== 编译 {len(names)} 张图 ===")
    for name in names:
        # --- 第一遍：量出 tikzpicture 的真实尺寸
        rc, w, h, log = measure(name, exe, env)
        if rc != 0:
            print(f"  ✗ {name}: 测量失败（图本身编译不过）")
            print("    " + _tail(log))
            failed.append(name)
            continue

        # --- 第二遍：按实测画幅紧致排版（+留白，否则线宽会被切掉）
        w += 2 * PAD_CM
        h += 2 * PAD_CM
        write_wrapper(name, f"{w:.4f}", f"{h:.4f}", measure=False)
        rc, log = run_xelatex(name, exe, env)
        if rc != 0:
            print(f"  ✗ {name}: xelatex 失败")
            print("    " + _tail(log))
            failed.append(name)
            continue
        bad = overfull(log)
        if bad:
            print(f"  ✗ {name}: {bad}")
            failed.append(name)
            continue

        built = os.path.join(BUILD, f"{name}.pdf")
        out_pdf = os.path.join(FIGS, f"{name}.pdf")
        # PDF 走 xelatex + SOURCE_DATE_EPOCH，字节级稳定；仍比一次内容，
        # 避免无谓地刷新 mtime。
        same = bytes_equal(out_pdf, built)
        if not same:
            shutil.copyfile(built, out_pdf)

        ok, how = to_png(out_pdf, os.path.join(FIGS, f"{name}.png"))
        if not ok:
            print(f"  ✗ {name}: PDF→PNG 失败: {how}")
            failed.append(name)
            continue
        try:
            from PIL import Image
            with Image.open(os.path.join(FIGS, f"{name}.png")) as im:
                px = f"{im.size[0]}x{im.size[1]}"
        except Exception:
            px = "?"
        print(f"  ✓ {name}  {w - 2*PAD_CM:.2f}x{h - 2*PAD_CM:.2f}cm  →  "
              f"{name}.pdf{'（未变）' if same else '（已更新）'}, "
              f"{name}.png {px} {'（未变）' if how == 'unchanged' else '（已更新）'}")

    if failed:
        print(f"\n{len(failed)} 张图失败: {', '.join(failed)}", file=sys.stderr)
        return 3
    print("\n全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

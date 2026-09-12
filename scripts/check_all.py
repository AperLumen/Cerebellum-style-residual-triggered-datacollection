#!/usr/bin/env python3
"""一条命令跑完全部验收：测试 → 类型检查 → 文档一致性 → PDF 产物。

## 为什么需要它

工业化稳定性的最后一环是**兜底**：前面那些契约（§4.4–4.6）只有在每次改动后
都真的跑过才有意义。此前这些步骤散在人的记忆里——"记得跑 pytest、记得跑
pyright、改了 GUIDE 记得重建 PDF"。凡是靠记性的检查，迟早会漏。

## 设计原则

1. **单一入口**：`python scripts/check_all.py` 就是全部。不要求记多个命令。
2. **快速失败**（默认）：第一个失败就停，输出聚焦在错处。`--all` 可以跑完所有
   步骤再汇总——用于提交前看全貌。
3. **确定性**：不依赖网络、不依赖 `runs/`（那些是 gitignore 的产物）。
4. **已知基线要显式**：`tests/test_extract_featmaps_dino.py` 有 6 个既存 ERROR
   （依赖已删除的合成帧）。把它写成**基线**而不是"忽略错误"——数量对不上就
   报警，避免既存问题掩盖新问题。
5. **能自动修的就修**：`--fix` 会重建 GUIDE.pdf（纯派生物）。

用法：
    PYTHONPATH=. .venv/bin/python scripts/check_all.py
    PYTHONPATH=. .venv/bin/python scripts/check_all.py --all
    PYTHONPATH=. .venv/bin/python scripts/check_all.py --fix
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")

# 既存失败基线：数量对不上就报警（不是"忽略"）。
# 来源：tests/test_extract_featmaps_dino.py 依赖已删除的合成帧。
KNOWN_TEST_ERRORS = {
    "tests/test_extract_featmaps_dino.py": 6,
}

# 类型检查分两档，因为要求不同：
#   * `cerebellum_cancel/` + `tests/` 是**交付物**，必须 0 错误——它们会被
#     别人 import，类型错误会传导出去。
#   * `scripts/` 里有一部分是**一次性探查脚本**（比如 analyze_negatives.py，
#     numpy 索引/cv2 重载在 pyright 下必然报错，修它没有收益）。对这些，
#     要求"不新增"而不是"清零"。
STRICT_PYRIGHT = ["cerebellum_cancel/", "tests/"]
LOOSE_PYRIGHT = ["scripts/"]
# scripts/ 的既存错误数基线（改动它应该是有意的，所以数量变了要显式更新）
KNOWN_PYRIGHT_ERRORS_SCRIPTS = 4


class Step:
    def __init__(self, name: str, cmd: list[str], *, cwd: str = ROOT,
                 note: str = ""):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.note = note
        self.rc = 0
        self.out = ""
        self.secs = 0.0


def run(step: Step, env_extra: dict[str, str] | None = None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    p = subprocess.run(step.cmd, cwd=step.cwd, env=env,
                       capture_output=True, text=True)
    step.secs = time.time() - t0
    step.rc = p.returncode
    step.out = (p.stdout or "") + (p.stderr or "")


# --------------------------------------------------------------------- 步骤

def step_tests() -> Step:
    return Step("单元测试", [PY, "-m", "pytest", "tests/", "-q"])


def step_pyright() -> Step:
    return Step("类型检查", [PY, "-m", "pyright",
                              *STRICT_PYRIGHT, *LOOSE_PYRIGHT])


def step_engine_smoke() -> Step:
    """引擎端到端冒烟：真的跑一条录制并导出，确认产物存在。

    为什么单列一步而不是只靠单测：单测用注入式 source（快、无依赖），
    永远碰不到真实的解码器、真实的前端、真实的落盘路径。这一条是**唯一**
    会真正读一个视频文件的检查——历史上 shape 契约不一致（read_gray 给
    (H,W) 而 step 要 (H,W,3)）就是单测发现不了、只有真跑才暴露的那类问题。
    """
    return Step("引擎端到端冒烟", [PY, "-c", ENGINE_SMOKE])


ENGINE_SMOKE = r'''
import os, shutil, sys, tempfile
import numpy as np
from cerebellum_cancel.engine import run, EngineConfig

# 优先用仓库里真实存在的短视频；没有就跳过（CI 上可能没有 data/）。
cands = ["data/webcam_rec/video.mp4", "data/synthetic/frames"]
src = next((c for c in cands if os.path.exists(c)), None)
if src is None:
    print("没有可用的本地视频，跳过（data/ 不入库，CI 上属于正常）")
    sys.exit(0)

out = tempfile.mkdtemp(prefix="cmci_smoke_")
try:
    res = run(src, None, cfg=EngineConfig(max_frames=90))
    assert res.n_frames == 90, res.n_frames
    assert res.score.dtype == np.float64
    w = res.save(out, formats=("dataset", "html", "jsonl"))
    for fmt, p in w.items():
        assert os.path.exists(p), f"{fmt} 未生成: {p}"
    # HTML 必须自包含（离线可开）
    h = open(os.path.join(out, "index.html"), encoding="utf-8").read()
    for bad in ("http://", "https://", "cdn."):
        assert bad not in h, f"HTML 引了外部资源 {bad}"
    # 诚实性声明必须在**任何**片段数下都存在（短片段常选到 0 个，
    # 此时字面量 motion_dedup 不会出现——所以断言放在披露文案上，
    # 而不是放在取值上）。
    assert "第 1 层" in h and "不等于" in h, "HTML 未如实标注选择依据"
    print(f"帧 {res.n_frames} / 片段 {len(res.segments)} / 导出 {sorted(w)}")
    print("HTML 自包含且如实标注了选择依据（第 1 层 / 不等于语义重要）")
finally:
    shutil.rmtree(out, ignore_errors=True)
'''


def step_generate_pdf() -> Step:
    return Step("重建 GUIDE.pdf", [PY, "scripts/build_guide_pdf.py"],
                note="纯派生物，变了会自动重建")


def step_verify_pdf() -> Step:
    return Step("PDF 自检", [PY, "scripts/verify_guide_pdf.py"])


def step_ui_smoke() -> Step:
    """前端冒烟：起真服务、打真 HTTP、验证三条纪律。

    单测直接调 act_* 函数（快、无端口依赖），所以**碰不到** HTTP 层、
    页面本身、以及"前端只调引擎"这条纪律。这一步起真服务跑一遍——
    和引擎那条冒烟一个道理：单测结构上覆盖不到的地方才需要端到端。
    """
    return Step("前端端到端冒烟", [PY, "-c", UI_SMOKE])


UI_SMOKE = r'''
import json, os, subprocess, sys, time, urllib.request, socket

ROOT = os.getcwd()
VIDEO = "data/webcam_rec/video.mp4"
if not os.path.exists(os.path.join(ROOT, VIDEO)):
    print("没有测试视频，跳过"); sys.exit(0)

# 找一个空闲端口，避免和用户正在跑的服务撞车
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()

env = {**os.environ, "PYTHONPATH": ROOT}
proc = subprocess.Popen(
    [sys.executable, "scripts/cmci_ui.py", "--port", str(port)],
    cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
try:
    # 等服务起来（最多 15s）
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/", timeout=1); break
        except Exception:
            time.sleep(0.3)
    else:
        raise RuntimeError("前端没起来")

    def post(path, body):
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=180))

    # 1) 页面必须能打开，且是自包含的（无外部资源）
    html = urllib.request.urlopen(base + "/", timeout=10).read().decode()
    for bad in ("http://", "https://", "cdn."):
        assert bad not in html, f"页面引了外部资源 {bad}（离线打不开）"
    print(f"页面 {len(html)} 字节，自包含")

    # 2) run。**断言要拿引擎当基准**，不能只比对 UI 自己的两个字段——
    #    它们出自同一个 dict，一起被改就一起"自洽"了（实测：把
    #    n_segments 硬编码后，弱断言放过了一个真 bug）。
    d = post("/api/run", {"video": VIDEO, "max_frames": 150})
    assert d["ok"], d.get("error")

    from cerebellum_cancel.engine import EngineConfig, run as engine_run
    ref = engine_run(VIDEO, None, cfg=EngineConfig(max_frames=150))
    ref_s = ref.summary()
    for k in ("n_frames", "n_segments"):
        assert d["summary"][k] == ref_s[k], (
            f"UI 的 {k}={d['summary'][k]} 与引擎的 {ref_s[k]} 不一致"
            "——前端是不是自己算了一遍？")
    assert len(d["segments"]) == ref_s["n_segments"], "片段表与摘要不符"
    print(f"run: {d['summary']['n_frames']} 帧 / {d['summary']['n_segments']} 片段"
          "（与引擎逐项一致）")

    # 3) replay 调参：必须即时（否则说明又重跑了前端），且结果与引擎 replay 一致
    t0 = time.time()
    r = post("/api/replay", {"threshold_k": 20.0, "off_k": 2.5})
    dt = time.time() - t0
    assert r["ok"], r.get("error")
    assert dt < 5.0, f"replay 太慢（{dt:.2f}s）——是不是重跑了前端？"

    from cerebellum_cancel.core import TriggerConfig
    from cerebellum_cancel.engine import replay as engine_replay
    ref_r = engine_replay(ref, trigger_cfg=TriggerConfig(threshold_k=20.0,
                                                        off_k=2.5))
    assert r["summary"]["n_segments"] == ref_r.summary()["n_segments"], (
        "replay 结果与引擎不一致——调参路径被改坏了")
    print(f"replay: {dt*1000:.0f} ms（阈值 20 -> {r['summary']['n_segments']} 片段"
          "，与引擎一致）")

    # 4) 参数非法要明确拒绝
    bad = post("/api/replay", {"threshold_k": 5.0, "off_k": 9.0})
    assert bad["ok"] is False and "off_k" in bad["error"]
    print("非法参数被明确拒绝")

    # 5) 错误要给人话
    miss = post("/api/run", {"video": "/nope/x.mp4"})
    assert miss["ok"] is False and "找不到视频文件" in miss["error"]
    print("缺失文件给出可读提示")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
'''


def step_doc_refs() -> Step:
    """文档里引用的文件必须存在——防止"文档承诺了不存在的东西"。

    这条是实测逼出来的：GUIDE §4.2 曾引用 `scripts/eval_weight_value.py`
    作为 weight 字段的判据脚本，但那个文件从没被创建。
    """
    return Step("文档引用完整性", [PY, "-c", DOC_REF_CHECK])


DOC_REF_CHECK = r'''
import os, re, sys
ROOT = os.getcwd()
DOCS = ["GUIDE.md", "MEMORY_FORMAT.md", "README.md", "README.zh.md"]
# 只在 disk 上找得到的路径才有意义；runs/ 与 data/ 是 gitignore 的产物，
# 不参与检查（它们的缺失是正常的，见 .gitignore）。
SKIP_PREFIX = ("runs/", "data/", "third_party/", "http")
missing = []
checked = 0
for doc in DOCS:
    p = os.path.join(ROOT, doc)
    if not os.path.exists(p):
        continue
    text = open(p, encoding="utf-8").read()
    for ref in re.findall(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|md|tex|pdf|npz|sh|yaml|toml))`", text):
        if ref.startswith(SKIP_PREFIX) or "/" not in ref:
            # 裸文件名（如 `integrity.py`）无法唯一解析，跳过
            continue
        checked += 1
        if not os.path.exists(os.path.join(ROOT, ref)):
            missing.append((doc, ref))
print(f"检查 {checked} 处带路径的文件引用")
if missing:
    print("以下引用指向不存在的文件：")
    for doc, ref in missing:
        print(f"  {doc}: {ref}")
    sys.exit(1)
print("全部存在")
'''


def step_manifest() -> Step:
    """manifest.json 必须合法，且其引用的实现/测试文件必须存在。"""
    return Step("manifest 一致性", [PY, "-c", MANIFEST_CHECK])


MANIFEST_CHECK = r'''
import json, os, sys
ROOT = os.getcwd()
m = json.load(open(os.path.join(ROOT, "manifest.json"), encoding="utf-8"))
bad = []
# 顶层登记的结构化条目：其中列出的文件必须存在
for key in ("format", "dataset_format", "input_contracts"):
    entry = m.get(key)
    if not isinstance(entry, dict):
        continue
    for field in ("doc", "implementation", "integrity", "builder"):
        v = entry.get(field)
        if isinstance(v, str) and not v.startswith(("http", "GUIDE.md §")):
            if not os.path.exists(os.path.join(ROOT, v)):
                bad.append(f"{key}.{field} -> {v}")
    for field in ("enforced_in", "tests"):
        for v in entry.get(field, []) or []:
            if not os.path.exists(os.path.join(ROOT, v)):
                bad.append(f"{key}.{field} -> {v}")
print(f"顶层条目: {[k for k in ('format','dataset_format','input_contracts') if k in m]}")
if bad:
    print("以上路径不存在：")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("manifest 引用的文件全部存在")
'''


def step_public_edition() -> Step:
    """公开子集（public_edition/）必须：(a) 自包含可跑，(b) 与主版共享的
    文件不漂移。

    这是实测逼出来的：public_edition 曾经悄悄落后主版几个月——缺 6 个模块、
    4 个共享模块已漂移、其中 1 个测试因 API 变化而失败。**它自己的测试能过**，
    所以没有任何信号提示它过期了。这类"沉默的过期"只能靠对比检查。
    """
    return Step("公开子集一致性", [PY, "-c", PUBLIC_EDITION_CHECK])


PUBLIC_EDITION_CHECK = r'''
import filecmp, os, subprocess, sys
ROOT = os.getcwd()
PE = os.path.join(ROOT, "public_edition")
if not os.path.isdir(PE):
    print("没有 public_edition/，跳过")
    sys.exit(0)

problems = []

# (a) 共享文件必须逐字节一致（公开版是**子集**，不是分支）
shared_mods = ["__init__", "core", "featmap_cancel", "flow_cancel", "io",
               "adapter", "pipeline", "eval", "acquisition", "contracts",
               "rotwarp"]
for m in shared_mods:
    a = os.path.join(ROOT, "cerebellum_cancel", m + ".py")
    b = os.path.join(PE, "cerebellum_cancel", m + ".py")
    if not os.path.exists(b):
        problems.append(f"public_edition 缺少 cerebellum_cancel/{m}.py（主版有）")
    elif not filecmp.cmp(a, b, shallow=False):
        problems.append(f"cerebellum_cancel/{m}.py 已漂移（主版与公开版不一致）")

shared_tests = ["test_core", "test_flow_cancel", "test_featmap_cancel",
                "test_adapter", "test_eval", "test_acquisition",
                "test_rotwarp", "test_gate_stats"]
for t in shared_tests:
    a = os.path.join(ROOT, "tests", t + ".py")
    b = os.path.join(PE, "tests", t + ".py")
    if not os.path.exists(a):
        continue
    if not os.path.exists(b):
        problems.append(f"public_edition 缺少 tests/{t}.py（主版有）")
    elif not filecmp.cmp(a, b, shallow=False):
        problems.append(f"tests/{t}.py 已漂移")

# (b) 公开版的测试必须真的能过（自包含）
p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                   cwd=PE, capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": "."})
tail = (p.stdout or "").strip().splitlines()[-1] if p.stdout else ""
print(f"public_edition 测试: {tail}")
if p.returncode != 0:
    problems.append(f"public_edition 测试失败（退出码 {p.returncode}）")

# (c) 不得硬编码主版绝对路径（否则不是自包含）
for sub in ("cerebellum_cancel", "tests", "scripts"):
    d = os.path.join(PE, sub)
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        if not fn.endswith(".py"):
            continue
        txt = open(os.path.join(d, fn), encoding="utf-8").read()
        if "/Users/" in txt or "2026-09/CMCI" in txt:
            problems.append(f"public_edition/{sub}/{fn} 含主版硬编码路径")

# (d) 公开版 README 声明的模块必须都在
declared = ["io", "core", "featmap_cancel", "flow_cancel", "adapter",
            "pipeline", "eval", "acquisition", "rotwarp", "contracts"]
missing = [m for m in declared
           if not os.path.exists(os.path.join(PE, "cerebellum_cancel", m + ".py"))]
if missing:
    problems.append(f"公开版缺少 README/布局声明的模块: {missing}")

if problems:
    for x in problems:
        print("  ✗ " + x)
    sys.exit(1)
print(f"  ✓ {len(shared_mods)} 个共享模块 + {len(shared_tests)} 个共享测试逐字节一致")
print("  ✓ 自包含（无主版绝对路径）")
'''


def step_tests_baseline(step: Step) -> list[str]:
    """把 pytest 输出与既存失败基线比对。返回问题列表（空 = OK）。

    注意：pytest 的 `short test summary info` 里每个 ERROR 只出现**一行**
    （形如 ``ERROR tests/x.py::test_y - ...``），而模块名在输出别处也会出现。
    所以要按 ``ERROR <path>::`` 这个模式数，不能直接 substring 计数——否则
    会数成大致的两倍（实测踩过：基线 6 被数成 12）。
    """
    problems: list[str] = []
    tail = step.out.strip().splitlines()[-1] if step.out.strip() else ""

    m = re.search(r"(\d+) passed", tail)
    if not m:
        problems.append(f"无法解析 pytest 汇总行: {tail!r}")
        return problems
    n_pass = int(m.group(1))

    # 按 "ERROR <module>::" 数，避免重复计数
    for mod, expect in KNOWN_TEST_ERRORS.items():
        n = len(re.findall(r"^ERROR\s+" + re.escape(mod) + r"::",
                           step.out, flags=re.M))
        if n != expect:
            problems.append(
                f"{mod}: ERROR 数 {n} != 基线 {expect}"
                "（既存问题变了，或引入了新问题）")

    # 任何**基线之外**的失败都要报
    m2 = re.search(r"(\d+) failed", tail)
    if m2 and int(m2.group(1)) > 0:
        problems.append(f"有 {m2.group(1)} 个测试失败（基线为 0）")
    if n_pass < 190:
        problems.append(f"通过数 {n_pass} 明显低于预期（基线 ~201）")
    return problems


# --------------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="一键验收")
    ap.add_argument("--all", action="store_true",
                    help="跑完所有步骤再汇总（默认第一个失败即停）")
    ap.add_argument("--fix", action="store_true",
                    help="允许重建派生物（GUIDE.pdf）")
    ap.add_argument("--skip-pdf", action="store_true",
                    help="跳过 PDF 相关步骤（不需要 LaTeX 时）")
    args = ap.parse_args()

    if not os.path.exists(PY):
        print(f"找不到 venv python: {PY}", file=sys.stderr)
        return 2

    steps: list[Step] = [step_tests(), step_pyright(), step_engine_smoke(),
                         step_ui_smoke(), step_doc_refs(), step_manifest(),
                         step_public_edition()]
    if not args.skip_pdf:
        steps += [step_generate_pdf(), step_verify_pdf()]

    results: list[tuple[Step, list[str]]] = []
    t_all = time.time()

    for st in steps:
        print(f"\n=== {st.name} ===")
        if st.note:
            print(f"    ({st.note})")
        run(st)
        problems: list[str] = []

        if st.name == "单元测试":
            problems = step_tests_baseline(st)
            # 打印汇总行即可，测试明细太长
            tail = st.out.strip().splitlines()[-1] if st.out.strip() else ""
            print(f"    {tail}")
        elif st.name == "类型检查":
            # pyright 的逐条错误形如 "/abs/path.py:LINE:COL - error: ..."
            err_lines = [l for l in st.out.splitlines() if " - error:" in l]
            strict = [l for l in err_lines
                      if any(f"/{d}" in l for d in STRICT_PYRIGHT)]
            loose = [l for l in err_lines
                     if any(f"/{d}" in l for d in LOOSE_PYRIGHT)]
            print(f"    交付物({'+'.join(STRICT_PYRIGHT)}): "
                  f"{len(strict)} 错误")
            print(f"    探查脚本({'+'.join(LOOSE_PYRIGHT)}): "
                  f"{len(loose)} 错误（基线 {KNOWN_PYRIGHT_ERRORS_SCRIPTS}）")
            for l in strict[:5]:
                print("      ✗ " + l.strip())
            for l in loose[:3]:
                print("      · " + l.strip())
            if strict:
                problems.append(
                    f"交付物有 {len(strict)} 个类型错误（要求 0）")
            if len(loose) != KNOWN_PYRIGHT_ERRORS_SCRIPTS:
                problems.append(
                    f"scripts/ 错误数 {len(loose)} != 基线 "
                    f"{KNOWN_PYRIGHT_ERRORS_SCRIPTS}"
                    "（若是有意改动，请更新基线）")
        else:
            # 其余步骤：rc != 0 即失败，打印输出尾部
            tail = "\n".join(
                l for l in st.out.strip().splitlines()[-12:])
            print("    " + tail.replace("\n", "\n    "))
            if st.rc != 0:
                problems.append(f"退出码 {st.rc}")

        for p in problems:
            print(f"    ✗ {p}")
        if not problems:
            print(f"    ✓ 通过 ({st.secs:.1f}s)")
        results.append((st, problems))

        if problems and not args.all:
            print(f"\n[快速失败] 在「{st.name}」停下。"
                  f"用 --all 可跑完全部再汇总。", file=sys.stderr)
            return 1

    # ---- 汇总
    print("\n" + "=" * 58)
    failed = [(s, p) for s, p in results if p]
    for s, p in results:
        mark = "✗" if p else "✓"
        print(f"  {mark} {s.name:<18} {s.secs:5.1f}s")
    print("=" * 58)
    if failed:
        print(f"结论: ✗ {len(failed)} 个步骤有问题", file=sys.stderr)
        return 1
    print(f"结论: ✓ 全部通过（总 {time.time() - t_all:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

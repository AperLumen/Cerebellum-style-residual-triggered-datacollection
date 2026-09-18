"""协作闸门的单元测试：配置漂移 + 需求溯源 + 文档基线数字对账。"""
import json
import os
import re
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from check_all import doc_baseline_problems  # noqa: E402
from check_drift import diff_defaults, extract_defaults  # noqa: E402
from check_requirements import (  # noqa: E402
    covering_entries, uncovered_files, validate_manifest)

_SRC = """
from dataclasses import dataclass, field

@dataclass
class TriggerConfig:
    threshold_k: float = 6.0
    off_k: float = 2.5
    grid_dims: tuple[int, int] = (8, 6)
    robust: bool = False
    state: dict = field(default_factory=dict)   # 运行时构造，不参与漂移
"""


def _pkg(tmp_path, body):
    d = tmp_path / "pkg"
    d.mkdir(exist_ok=True)
    (d / "mod.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return str(d)


def test_extract_defaults_skips_field_and_normalizes_tuples(tmp_path):
    got = extract_defaults(_pkg(tmp_path, _SRC))
    assert got == {"mod.py:TriggerConfig.threshold_k": 6.0,
                   "mod.py:TriggerConfig.off_k": 2.5,
                   "mod.py:TriggerConfig.grid_dims": [8, 6],
                   "mod.py:TriggerConfig.robust": False}


def test_diff_defaults_clean_when_matching(tmp_path):
    got = extract_defaults(_pkg(tmp_path, _SRC))
    assert diff_defaults(got, got) == []
    # JSON 往返后 tuple 变 list，也必须判等
    assert diff_defaults(got, json.loads(json.dumps(got))) == []


def test_diff_defaults_flags_value_add_remove(tmp_path):
    base = extract_defaults(_pkg(tmp_path, _SRC))
    changed = dict(base)
    changed["mod.py:TriggerConfig.threshold_k"] = 4.0
    msgs = diff_defaults(changed, base)
    assert any("threshold_k" in m and "6.0" in m and "4.0" in m for m in msgs)
    assert diff_defaults({**base, "mod.py:New.x": 1}, base) != []
    assert diff_defaults({k: v for k, v in base.items()
                          if "off_k" not in k}, base) != []


def test_checked_in_baseline_matches_package():
    """入库的 config_baseline.json 必须与当前包一致（防基线/代码脱钩）。"""
    with open(os.path.join(ROOT, "config_baseline.json"),
              encoding="utf-8") as f:
        baseline = json.load(f)["values"]
    got = extract_defaults(os.path.join(ROOT, "cerebellum_cancel"))
    assert diff_defaults(got, baseline) == []


def _entry(**kw):
    e = {"id": "r1", "date": "2026-09-14", "source": "用户原话：改这个",
         "summary": "s", "files": ["scripts/a.py"]}
    e.update(kw)
    return e


def test_uncovered_files():
    entries = [_entry()]
    assert uncovered_files(["scripts/a.py"], entries) == []
    assert uncovered_files(["scripts/b.py"], entries) == ["scripts/b.py"]
    assert covering_entries(["scripts/a.py"], entries)[0]["id"] == "r1"


def test_validate_manifest_rejects_bad_entries():
    assert validate_manifest([_entry()]) == []
    assert validate_manifest([_entry(source="")]) != []
    assert validate_manifest([_entry(files=[])]) != []
    assert validate_manifest([{"id": "x"}]) != []


# ------------------------------------------------- 文档声称的基线数字必须对账
#
# 病灶（2026-09-18 实测）：`AGENTS.md` 长期写着"pytest 537+ passed /
# `make check` 13 步"，而实现早已是 **542 passed / 15 步**。两边脱钩，
# 且**没有任何检查发现**——这正是本项目守则第 6 条的镜像形态：一个总在报
# "有变化"的门会被训练成没人看，而一个**从不报错的文档数字**没人会去维护。
#
# 守法的要点：断言必须锚定**独立真值**。这里真值有两个来源——
#   ① 步数：`build_steps()` 的实现长度（不是文档里抄一遍的数字）；
#   ② 通过数：调用方传入的实测值（`make check` 里就是刚跑完的 pytest）。
# 所以下面**不自己写正则**，全部走生产函数 `doc_baseline_problems`，
# 否则就是"测试里复制了一份实现"（第 5 类静默失效）。

def test_doc_baseline_step_count_matches_implementation():
    """步数真值来自 build_steps()，文档必须与它完全相等。"""
    from check_all import build_steps
    n_steps = len(build_steps(skip_pdf=False))
    text = f"基线：pytest 999+ passed；`make check` {n_steps} 步全绿。"
    assert doc_baseline_problems(999, agents_text=text,
                                 n_steps=n_steps) == []


def test_doc_baseline_flags_stale_step_count():
    """文档步数与实现差 1 就要报——且报错信息里要给出正确值。"""
    from check_all import build_steps
    n_steps = len(build_steps(skip_pdf=False))
    text = f"`make check` {n_steps - 2} 步全绿"
    problems = doc_baseline_problems(999, agents_text=text,
                                     n_steps=n_steps)
    assert any("步" in p and str(n_steps) in p for p in problems), problems


def test_doc_baseline_flags_pass_count_above_reality():
    """文档声称的通过数**高于**实测 = 文档在吹牛，必须报。"""
    text = "基线：pytest 900+ passed；`make check` 1 步全绿。"
    problems = doc_baseline_problems(542, agents_text=text, n_steps=1)
    assert any("高于现实" in p for p in problems), problems


def test_doc_baseline_flags_pass_count_far_below_reality():
    """文档落后太多也要报——阈值失去约束力本身就是缺陷。"""
    text = "基线：pytest 100+ passed；`make check` 1 步全绿。"
    problems = doc_baseline_problems(542, agents_text=text, n_steps=1)
    assert any("落后" in p for p in problems), problems


def test_doc_baseline_tolerance_boundary_is_pinned():
    """**把容忍带的边界钉死在具体数字上**，免得被悄悄放宽。

    为什么是 2% 而不是 5%：一次典型改动只加个位数测试。实测 549 时，
    5% 的带宽是约 27 个测试——要连续多次改动都不更新文档才会触发，等于
    名存实亡（守则 6 的反面：一个实际永不报错的守卫）。2% 约 11 个测试，
    容得下一次中等规模的测试增补，又不至于让数字烂掉。

    用 n_pass=1000 这种圆整数，边界（979 报 / 980 不报）与真实套件规模
    解耦，这条测试不会因为别人加测试而改变含义。
    """
    report = "基线：pytest 979+ passed；`make check` 1 步全绿。"
    assert any("落后" in p
               for p in doc_baseline_problems(1000, agents_text=report,
                                              n_steps=1))
    keep = "基线：pytest 980+ passed；`make check` 1 步全绿。"
    assert doc_baseline_problems(1000, agents_text=keep, n_steps=1) == []


def test_doc_baseline_tolerates_pending_headroom():
    """**预留余量是设计意图，不该报**：写 542 而实测 545 属于正常，
    因为"NNN+ passed"本就是下限语义。这条防的是守卫过严——过严的守卫
    会让人习惯性放宽它，比没有守卫更坏。"""
    text = "基线：pytest 542+ passed；`make check` 1 步全绿。"
    assert doc_baseline_problems(545, agents_text=text, n_steps=1) == []


def test_doc_baseline_missing_claim_is_flagged():
    """文档里根本没有这个声明时要报——**读不到数 = 恒真 = 没有守卫**。"""
    problems = doc_baseline_problems(542, agents_text="什么都没有",
                                     n_steps=1)
    assert len(problems) == 2, problems


def test_real_agents_md_is_consistent_with_reality():
    """对**真文件**跑一遍：AGENTS.md 与当前实现/实测必须自洽。

    通过数怎么来：不能直接数 `--collect-only` 的结果——那是 **collected**
    （实测 556），而 `passed` 是 550，差额正好是已知的 6 个 ERROR
    （实测 550 + 6 == 556，精确成立）。所以 `passed == collected − 已知
    ERROR 数`。

    为什么非要精确到 `passed`：`make check` 的判据用的就是这个量。若这里
    传 556 去比、那边传 550 去比，两边容错不同 → 同一个文档数字会出现
    "pytest 过、make check 不过"的分裂，而分裂的守卫没人信。

    （若将来出现**未知模块**的收集错误，本推导会高估 passed，但那种情况
    `make check` 的 `KNOWN_TEST_ERRORS` 对账会先报错，覆盖住了。）
    """
    from check_all import KNOWN_TEST_ERRORS, build_steps
    n_steps = len(build_steps(skip_pdf=False))
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": ROOT})
    m = re.search(r"(\d+)\s+tests?\s+collected", p.stdout)
    assert m, f"无法从 --collect-only 解析用例数:\n{p.stdout[-800:]}"
    collected = int(m.group(1))
    n_pass = collected - sum(KNOWN_TEST_ERRORS.values())
    problems = doc_baseline_problems(n_pass, agents_text=None,
                                     n_steps=n_steps)
    assert problems == [], (
        "AGENTS.md 的基线数字与实现/实测不符：\n  " + "\n  ".join(problems))

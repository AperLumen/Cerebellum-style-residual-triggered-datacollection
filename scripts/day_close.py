#!/usr/bin/env python3
"""每日收尾：把当天全部改动收成一个"验收全绿"的提交 + 当日报告。

## 为什么（用户定调的协作纪律）

> "每一次修改之后都应该提交一个整体的方案。比如今天是 9 月 9 号，那我
>  9 月 9 号所有的更改都应该……全部合并到一个本地提交里面。"

隐形 bug 的温床是"改动"与"验收通过"脱钩：改了东西、当时能跑、测试也过，
但没有任何记录证明"这一天的整体状态是绿的"。本脚本把绑定做成机制：

  1. **先验收**：跑完整 make check，红了拒绝收尾——当天不许带病下班；
  2. **出报告**：daily/YYYY-MM-DD.md——变更清单、验收结论、当天命中的
     台账条目、人类备注；
  3. **单提交**：message 带日期、验收结论、报告路径；
  4. **移基线**：day-close-baseline tag 移到新 HEAD——"需求溯源检查"
     从此以今天为界。

用法：
    make day-close                      # 正常收尾
    make day-close ARGS="--note '备注'"  # 报告里加一句人话
    make day-close ARGS="--skip-pdf"    # 无 LaTeX 环境
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
DAILY = os.path.join(ROOT, "daily")
MANIFEST = os.path.join(ROOT, "requirement_manifest.json")
BASELINE_TAG = "day-close-baseline"

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from check_requirements import covering_entries  # noqa: E402


def _sh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, text=True,
                          capture_output=True, check=check)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="报告与提交信息里的人类可读备注")
    ap.add_argument("--skip-pdf", action="store_true", help="无 LaTeX 环境")
    args = ap.parse_args()

    # 1) 全量验收，红了拒绝（不先看工作区——验收本身可能重建派生物）
    print("=== 每日收尾：先跑全量验收（make check）===")
    cmd = [PY, os.path.join(ROOT, "scripts", "check_all.py")]
    if args.skip_pdf:
        cmd.append("--skip-pdf")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print("\n✗ 验收未过，拒绝收尾。先修到 make check 全绿，"
              "再跑 make day-close。")
        return 1
    print("✓ 验收全绿\n")

    # 2) 验收后再取变更清单（check 可能重建 GUIDE.pdf 等派生物）
    status = _sh("status", "--porcelain").stdout
    if not status.strip():
        print("验收后工作区干净，无改动需要收尾。")
        return 0
    changed = []
    for ln in status.splitlines():
        if not ln.strip():
            continue
        path = ln[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.append((ln[:2], path))

    today = datetime.date.today().isoformat()

    # 3) 当日报告
    with open(MANIFEST, encoding="utf-8") as f:
        entries = json.load(f).get("requirements", [])
    paths = [p for _, p in changed]
    applied = covering_entries([p for p in paths if p.endswith(".py")], entries)

    lines = [
        f"# 每日收尾 — {today}", "",
        f"- 文件变动：{len(changed)} 个", "",
        "## 变更文件", ""]
    lines += [f"- `{p}`（{xy.strip()}）" for xy, p in changed]
    lines += ["", "## 验收", "",
              "- `make check` 全绿（提交前刚跑过；本报告由 day_close.py 生成）",
              "", "## 需求溯源（本日改动命中的台账条目）", ""]
    if applied:
        lines += [f"- **{e['id']}**：{e['summary']}" for e in applied]
    else:
        lines.append("- （无 .py 变更，或无命中条目）")
    if args.note:
        lines += ["", "## 备注", "", args.note]
    lines.append("")

    os.makedirs(DAILY, exist_ok=True)
    report_rel = os.path.join("daily", f"{today}.md")
    with open(os.path.join(ROOT, report_rel), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 4) 单提交
    if args.note:
        summary = args.note
    elif applied:
        summary = "；".join(e["summary"] for e in applied[:2])
    else:
        summary = "每日收尾"
    subject = f"day-close({today}): {summary}"[:72]
    msg = (f"{subject}\n\n"
           f"- 变更：{len(changed)} 个文件（清单见 {report_rel}）\n"
           f"- 验收：make check 全绿\n")
    if applied:
        msg += f"- 需求溯源：{', '.join(e['id'] for e in applied)}\n"
    if args.note:
        msg += f"- 备注：{args.note}\n"

    _sh("add", "-A")
    if _sh("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("只有被 gitignore 的产物变动，无可提交内容。")
        return 0
    c = _sh("commit", "-m", msg, check=False)
    if c.returncode != 0:
        print(f"✗ 提交失败：{c.stderr.strip()}")
        return 1

    # 5) 移基线 + 日期 tag
    _sh("tag", "-f", BASELINE_TAG)
    _sh("tag", "-f", f"day-close/{today}")
    short = _sh("rev-parse", "--short", "HEAD").stdout.strip()
    print(f"✓ 收尾完成：{short}  {subject}")
    print(f"  报告：{report_rel}    基线 tag：{BASELINE_TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

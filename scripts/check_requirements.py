#!/usr/bin/env python3
"""需求溯源检查——防止 AI 幻觉与伪需求。

## 为什么

工程全绿 ≠ 改动该发生。AI 最容易犯的不是"改坏代码"（那会被测试抓住），
而是"做了用户没要的事"：改行为、调参数、加抽象——测试照样 100% 通过，
但方向错了。能抓它的只有一条：**每个改动的 .py 文件，必须能回溯到
requirement_manifest.json 里一条带用户原话的需求**。

## 判据

  变更集合 = 相对基线 tag（day-close-baseline，每日收尾时移动）的已提交
             改动 + 工作区未提交改动（含未跟踪文件）；
  覆盖     = 每个变更的 .py 至少命中一条 manifest 条目的 files 模式
             （fnmatch）。

基线 tag 不存在（首次运行）时以 HEAD 为界并提示——历史不算账，从今天起算。

## 立条目格式

    {
      "id": "req-2026-09-15-xxx",
      "date": "2026-09-15",
      "source": "用户原话：……（逐字引用，不接受 AI 自拟）",
      "summary": "一句话：做什么、为什么",
      "files": ["scripts/xxx.py", "tests/xxx.py"]
    }
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "requirement_manifest.json")
BASELINE_TAG = "day-close-baseline"


def validate_manifest(entries: list) -> list:
    """条目结构校验。返回错误列表（空 = 通过）。"""
    errs: list[str] = []
    seen: set = set()
    for e in entries:
        rid = e.get("id", "?")
        for field in ("id", "date", "source", "summary", "files"):
            if field not in e:
                errs.append(f"{rid}: 缺字段 {field}")
        if rid in seen:
            errs.append(f"{rid}: id 重复")
        seen.add(rid)
        if len(str(e.get("source", ""))) < 8:
            errs.append(f"{rid}: source 必须引用用户原话（≥8 字符），"
                        "不接受 AI 自拟需求")
        files = e.get("files")
        if (not isinstance(files, list) or not files
                or not all(isinstance(x, str) for x in files)):
            errs.append(f"{rid}: files 必须是非空字符串列表")
    return errs


def uncovered_files(changed: list, entries: list) -> list:
    """返回未被任何 manifest 条目覆盖的变更文件。"""
    patterns = [p for e in entries for p in e.get("files", [])]
    return [c for c in changed
            if not any(fnmatch.fnmatch(c, p) for p in patterns)]


def covering_entries(changed: list, entries: list) -> list:
    """返回覆盖了变更文件的条目（day_close 生成报告时复用）。"""
    return [e for e in entries
            if any(fnmatch.fnmatch(c, p)
                   for c in changed for p in e.get("files", []))]


def _sh(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout


def git_changed_files() -> list:
    """相对基线的已提交改动 + 工作区全部改动（含未跟踪、含重命名新路径）。"""
    tag = _sh("rev-parse", "--verify", "--quiet", BASELINE_TAG).strip()
    base = BASELINE_TAG if tag else "HEAD"
    if not tag:
        print(f"    （首次运行：基线 tag {BASELINE_TAG} 不存在，以 HEAD 为界；"
              "day-close 后建立）")
    committed = _sh("diff", "--name-only", f"{base}..HEAD").split()
    work = []
    for ln in _sh("status", "--porcelain").splitlines():
        if not ln.strip():
            continue
        path = ln[3:]
        if " -> " in path:            # 重命名 "old -> new"，取新路径
            path = path.split(" -> ", 1)[1]
        work.append(path)
    return sorted(set(committed) | set(work))


def main() -> int:
    if not os.path.exists(MANIFEST):
        print(f"✗ 找不到 {os.path.relpath(MANIFEST, ROOT)}——需求溯源的台账"
              "文件。请先建立（格式见本脚本 docstring）。")
        return 1
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("requirements", [])

    errs = validate_manifest(entries)
    changed_py = [c for c in git_changed_files() if c.endswith(".py")]
    uncovered = uncovered_files(changed_py, entries)

    if errs:
        print("✗ requirement_manifest.json 结构问题：")
        for e in errs:
            print(f"    {e}")
    if uncovered:
        print("✗ 以下 .py 改动没有对应的需求条目（幻觉/伪需求的红线）：")
        for c in uncovered:
            print(f"    {c}")
        print("\n处理：在 requirement_manifest.json 里立条目"
              "（id/date/source=用户原话/summary/files），"
              "或确认这些改动不该发生。")
    if errs or uncovered:
        return 1
    print(f"✓ {len(changed_py)} 个变更 .py 全部被需求条目覆盖"
          f"（台账 {len(entries)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

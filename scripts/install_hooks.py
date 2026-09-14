#!/usr/bin/env python3
"""安装 git pre-commit 钩子（可选加固）——提交前快速闸门。

钩子只跑快速检查（配置漂移 + 需求溯源，<1s）；全量 make check 由
`make day-close` 在每个工作日收尾时强制——45s 的全量不该卡在每次提交上。

    make hooks                  # 安装
    rm .git/hooks/pre-commit    # 卸载
"""
from __future__ import annotations

import os
import stat
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOOK = """#!/bin/sh
# pre-commit 快速闸门（scripts/install_hooks.py 安装）
# 提交前拦截两类隐形改动：配置默认值漂移、无需求条目的 .py 改动。
# 全量验收（make check 13 步）由 make day-close 在每个工作日收尾强制。
ROOT="$(git rev-parse --show-toplevel)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" "$ROOT/scripts/check_drift.py" || {
    echo "✗ 配置漂移检查未过——有意调参需：改代码 + 立 manifest 条目 + make drift-update"
    exit 1
}
"$PY" "$ROOT/scripts/check_requirements.py" || {
    echo "✗ 需求溯源检查未过——改动的 .py 需要 requirement_manifest.json 条目（含用户原话）"
    exit 1
}
exit 0
"""


def main() -> int:
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-dir"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip()
    hook_path = os.path.join(git_dir, "hooks", "pre-commit")
    if not os.path.isabs(hook_path):
        hook_path = os.path.join(ROOT, hook_path)
    with open(hook_path, "w", encoding="utf-8") as f:
        f.write(HOOK)
    os.chmod(hook_path, os.stat(hook_path).st_mode | stat.S_IEXEC)
    print(f"✓ 已安装 pre-commit 钩子：{hook_path}")
    print(f"  卸载：rm {hook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

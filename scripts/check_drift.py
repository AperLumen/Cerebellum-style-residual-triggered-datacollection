#!/usr/bin/env python3
"""配置默认值漂移检查——防止"能跑、测试全过、但参数被顺手调了"的隐形改动。

## 为什么

`make check` 全绿只能证明"现有测试没被改坏"，证明不了"默认值没被改"。
触发阈值（threshold_k=6.0）、学习率、网格这类参数是**测量基础设施**——
GUIDE 里所有数字都建立在它们之上。人机协作中"策略性/适应性调参"是最典型
的幻觉产物：测试照样全过，但整篇文档的科学结论悄悄失真。

## 做法

对 cerebellum_cancel/ 里所有 @dataclass 的**字面量默认值**做 AST 提取
（只解析、不 import，无副作用），与 config_baseline.json 逐字节比对：

  * 改了值     → 失败（这就是"隐形 bug"的定义）；
  * 删了参数   → 失败；
  * 新增参数   → 失败（确认有意后 --update）；
  * field(...) 之类的运行时构造不参与——它们不是"裸参数"。

确需调参 = 改代码 + 在 requirement_manifest.json 立条目（含用户原话）
+ make drift-update，三者缺一不可。

用法：
    make drift-check     # 检查（make check 会调用）
    make drift-update    # 有意调参后更新基线
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "cerebellum_cancel")
BASELINE = os.path.join(ROOT, "config_baseline.json")


def _norm(v):
    """tuple→list 归一化，保证与 JSON 往返后的值可比。"""
    if isinstance(v, (tuple, list)):
        return [_norm(x) for x in v]
    return v


def extract_defaults(pkg_dir: str) -> dict:
    """AST 提取包内所有 @dataclass 的字面量默认值（不 import，无副作用）。"""
    out: dict = {}
    for fn in sorted(os.listdir(pkg_dir)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(pkg_dir, fn)
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_dc = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                for d in node.decorator_list)
            if not is_dc:
                continue
            for stmt in node.body:
                if (isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                        and stmt.value is not None):
                    try:
                        val = ast.literal_eval(stmt.value)
                    except Exception:
                        continue  # field(...) 等运行时构造，不是裸参数
                    out[f"{fn}:{node.name}.{stmt.target.id}"] = _norm(val)
    return out


def diff_defaults(detected: dict, baseline: dict) -> list:
    """返回人话差异列表（空 = 一致）。两侧都做 tuple→list 归一化。"""
    det = {k: _norm(v) for k, v in detected.items()}
    base = {k: _norm(v) for k, v in baseline.items()}
    msgs: list[str] = []
    for k in sorted(set(det) | set(base)):
        if k not in base:
            msgs.append(
                f"新增参数 {k} = {det[k]!r}——确认有意后 make drift-update")
        elif k not in det:
            msgs.append(
                f"参数被删 {k}（基线 {base[k]!r}）——确认有意后 make drift-update")
        elif det[k] != base[k]:
            msgs.append(f"{k}: {base[k]!r} → {det[k]!r}")
    return msgs


def _write_baseline(detected: dict) -> None:
    payload = {
        "_comment": "配置默认值基线（scripts/check_drift.py 维护）。任何值变动都会"
                    "让 make check 失败；有意调参需同时满足：改代码 + "
                    "requirement_manifest.json 立条目 + make drift-update。",
        "values": detected,
    }
    with open(BASELINE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="把当前值写入基线（有意调参后用，需先立 manifest 条目）")
    args = ap.parse_args()

    detected = extract_defaults(PKG)

    if not os.path.exists(BASELINE):
        _write_baseline(detected)
        print(f"✓ 首次运行：已生成基线 {os.path.relpath(BASELINE, ROOT)}"
              f"（{len(detected)} 个参数）")
        return 0

    if args.update:
        _write_baseline(detected)
        print(f"✓ 基线已更新（{len(detected)} 个参数）——请确认 "
              "requirement_manifest.json 里已有对应条目")
        return 0

    with open(BASELINE, encoding="utf-8") as f:
        baseline = json.load(f)["values"]
    diffs = diff_defaults(detected, baseline)
    if diffs:
        print("✗ 配置默认值发生漂移（隐形改动）：")
        for m in diffs:
            print(f"    {m}")
        print("\n若这是有意调参：改代码 + 在 requirement_manifest.json 立条目"
              "（含用户原话）+ make drift-update。三者缺一不可。")
        return 1
    print(f"✓ {len(detected)} 个配置默认值与基线一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

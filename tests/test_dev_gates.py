"""协作闸门的单元测试：配置漂移 + 需求溯源。"""
import json
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

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

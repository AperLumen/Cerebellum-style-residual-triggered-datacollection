"""Select 4 high-motion Open-AoE segments for the walking-scale probe.

Scans data/aoe_index (probed segments: camera_traj.npz + annotation, no
video), differentiates each camera trajectory with prep_aoe.py's
trajectory_gyro to get the physical rotation rate (deg/s), and picks the
4 segments with the highest p95 rotation magnitude subject to:

  1. motion: p95 >= 40 deg/s (walking-scale; the desk pool is <10)
  2. scene:  prefer outdoor/street/mall-style scenes, avoid 4x indoor
  3. duration: 60-420 s (full-frame decode in prep_aoe stays in RAM)

Sources already used by other experiments are excluded: every "source"
in data/aoe{,_event,_motion}/val/*/events.json (matched against
aoe_index dir names).

Writes the selection in fetch_aoe.py --selected format (list of {"seg":
...} dicts) to data/aoe_index/selected_walk.json and prints the table.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/select_walk_segments.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_aoe import trajectory_gyro  # noqa: E402

INDEX = "data/aoe_index"
OUT = f"{INDEX}/selected_walk.json"
P95_MIN = 40.0          # deg/s, walking-scale floor
SECONDS_MIN, SECONDS_MAX = 60.0, 420.0
NEW_SCENE_BONUS = 25.0  # deg/s-equivalent weight for scene diversity


def used_sources() -> set[str]:
    """Segment ids already consumed by the desk/event/motion pools."""
    used = set()
    for pool in ("data/aoe", "data/aoe_event", "data/aoe_motion"):
        val = os.path.join(pool, "val")
        if not os.path.isdir(val):
            continue
        for seq in os.listdir(val):
            p = os.path.join(val, seq, "events.json")
            if os.path.isfile(p):
                used.add(json.load(open(p))["source"])
    return used


def seg_stats(seg: str) -> dict | None:
    """scene, duration and trajectory_gyro p95 for one indexed segment."""
    try:
        ann = json.load(open(f"{INDEX}/{seg}/ego_action_annotation.json"))
        traj = np.load(f"{INDEX}/{seg}/camera_traj.npz")["cam_c2w"]
    except Exception:
        return None
    if not ann or len(traj) < 2:
        return None
    seconds = max(float(s["end_ts"]) for s in ann)
    scenes = sorted({s.get("scene", "?") for s in ann})
    # no undistorted_video_info.json at probe time; recover fps from the
    # trajectory span over the annotated duration (fetch_aoe.seg_stats
    # does the same)
    fps = (len(traj) - 1) / seconds if seconds > 0 else 30.0
    gyro = trajectory_gyro(traj, fps)
    p95 = float(np.percentile(np.linalg.norm(gyro, axis=1), 95))
    return {"seg": seg, "seconds": round(seconds, 1), "scenes": scenes,
            "p95": round(p95, 1)}


def main() -> None:
    used = used_sources()
    stats = []
    for seg in sorted(os.listdir(INDEX)):
        if seg in used or not os.path.isdir(f"{INDEX}/{seg}"):
            continue
        s = seg_stats(seg)
        if s:
            stats.append(s)
    print(f"{len(stats)} indexed segments after excluding "
          f"{len(used)} used sources")

    pool = [s for s in stats
            if SECONDS_MIN <= s["seconds"] <= SECONDS_MAX
            and s["p95"] >= P95_MIN]
    pool.sort(key=lambda s: -s["p95"])
    print(f"{len(pool)} pass {P95_MIN:.0f} deg/s & "
          f"{SECONDS_MIN:.0f}-{SECONDS_MAX:.0f}s "
          f"(pool p50/p95: {np.percentile([s['p95'] for s in stats], 50):.1f}/"
          f"{np.percentile([s['p95'] for s in stats], 95):.1f})")
    for s in pool[:15]:
        print(f"  cand {s['seg']}: {s['seconds']:6.0f}s p95={s['p95']:6.1f} "
              f"scenes={s['scenes']}")
    if len(pool) < 4:
        raise SystemExit(f"only {len(pool)} candidates pass the floor")

    # greedy: highest p95, with a bonus for covering a new scene so the
    # 4 segments are not all indoor
    picked, covered = [], set()
    rest = list(pool)
    while rest and len(picked) < 4:
        best = max(rest, key=lambda s: s["p95"]
                   + (NEW_SCENE_BONUS if not covered.issuperset(s["scenes"])
                      else 0.0))
        rest.remove(best)
        picked.append(best)
        covered |= set(best["scenes"])

    with open(OUT, "w") as f:
        json.dump(picked, f, indent=2)
    print(f"\nselected {len(picked)} -> {OUT}")
    print(f"{'id':38s} {'scene':28s} {'p95':>6s} {'dur':>6s}")
    for s in picked:
        print(f"{s['seg']:38s} {','.join(s['scenes']):28s} "
              f"{s['p95']:6.1f} {s['seconds']:5.0f}s")


if __name__ == "__main__":
    main()

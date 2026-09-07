"""One-off selector for 5 additional sparse-event segments (seq_005..seq_009)
in data/aoe_event/val/, from the probed pool in data/aoe_index/.

Selection criteria (priority order):
  1. annotation contains an event verb (drop/throw/toss/break) and the
     event-frame ratio (total event-verb interval duration / segment
     duration) is within 3-30%;
  2. camera rotation rate p95 (deg/s) from the cam_c2w finite difference —
     same math as prep_aoe.trajectory_gyro / fetch_aoe.seg_stats — as high
     as possible among candidates;
  3. scene diversity: greedy pick at most one segment per scene first, then
     fill by p95.

Segments already used in data/aoe_event or data/aoe are excluded. A
duration cap keeps prep_aoe's all-frames-in-RAM decoding safe and the new
segments comparable to the existing 5 (38-308 s).

Writes the pick to data/aoe_index/selected_event.json for fetch_aoe.py.

Usage: PYTHONPATH=. .venv/bin/python scripts/select_event_segments.py
"""

from __future__ import annotations

import json
import os

import numpy as np

INDEX = "data/aoe_index"
OUT = f"{INDEX}/selected_event.json"
EVENT_VERBS = {"drop", "throw", "toss", "break"}
RATIO_MIN, RATIO_MAX = 0.03, 0.30
MAX_SECONDS = 420.0        # prep_aoe holds all decoded frames in RAM
TOP_PRINT = 15


def used_sources() -> set[str]:
    sources = set()
    for root in ("data/aoe_event/val", "data/aoe/val"):
        if not os.path.isdir(root):
            continue
        for seq in sorted(os.listdir(root)):
            p = f"{root}/{seq}/events.json"
            if os.path.exists(p):
                sources.add(json.load(open(p))["source"])
    return sources


def rotation_p95(cam_c2w: np.ndarray, seconds: float) -> float:
    """p95 of the per-frame physical camera rotation rate (deg/s).
    Same quantity as prep_aoe.trajectory_gyro, magnitude form:
    angle(R_rel) per frame * fps, vectorized over frames."""
    a = cam_c2w[1:, :3, :3]
    b = cam_c2w[:-1, :3, :3]
    r = np.einsum("tij,tjk->tik", a.transpose(0, 2, 1), b)
    c = np.clip((np.einsum("tii->t", r) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    fps = (len(cam_c2w) - 1) / seconds if seconds > 0 else 30.0
    return float(np.percentile(ang, 95) * fps) if len(ang) else 0.0


def seg_stats(seg: str) -> dict | None:
    try:
        ann = json.load(open(f"{INDEX}/{seg}/ego_action_annotation.json"))
        traj = np.load(f"{INDEX}/{seg}/camera_traj.npz")["cam_c2w"]
    except Exception:
        return None
    if not ann:
        return None
    seconds = max(float(s["end_ts"]) for s in ann)
    scenes, verbs, ev_dur = set(), set(), 0.0
    for s in ann:
        scenes.add(s.get("scene", "?"))
        for a in s.get("atomic_action") or []:
            v = a.get("verb", "?")
            verbs.add(v)
            if v in EVENT_VERBS:
                ev_dur += float(s["end_ts"]) - float(s["start_ts"])
                break
    ratio = ev_dur / seconds if seconds > 0 else 0.0
    return {"seg": seg, "seconds": round(seconds, 1), "scenes": sorted(scenes),
            "verbs": sorted(verbs), "ev_ratio": round(ratio, 4),
            "motion_p95": round(rotation_p95(traj, seconds), 2)}


def main() -> None:
    used = used_sources()
    print(f"{len(used)} sources already used "
          f"(aoe_event + aoe)")
    stats = []
    for seg in sorted(os.listdir(INDEX)):
        if not os.path.isdir(f"{INDEX}/{seg}") or seg in used:
            continue
        s = seg_stats(seg)
        if s:
            stats.append(s)
    cands = [s for s in stats
             if s["ev_ratio"] >= RATIO_MIN and s["ev_ratio"] <= RATIO_MAX
             and s["seconds"] <= MAX_SECONDS]
    print(f"{len(stats)} probed & unused, {len(cands)} pass "
          f"{RATIO_MIN:.0%}<=ev_ratio<={RATIO_MAX:.0%} & "
          f"seconds<={MAX_SECONDS:.0f}")
    cands.sort(key=lambda s: -s["motion_p95"])

    print(f"\ntop {TOP_PRINT} candidates (by gyro p95):")
    print(f"  {'seg':44s} {'sec':>6s} {'ev%':>6s} {'p95':>7s}  scenes")
    for s in cands[:TOP_PRINT]:
        print(f"  {s['seg']:44s} {s['seconds']:6.0f} "
              f"{s['ev_ratio'] * 100:5.1f}% {s['motion_p95']:7.2f}  "
              f"{','.join(s['scenes'])}  [{','.join(v for v in s['verbs'])}]")

    picked, scene_count = [], {}
    for s in cands:                     # max 2 per scene tag, highest p95
        if len(picked) == 5:            # first — a motion floor by
            break                       # construction
        tags = set(s["scenes"])
        if all(scene_count.get(t, 0) < 2 for t in tags):
            picked.append(s)
            for t in tags:
                scene_count[t] = scene_count.get(t, 0) + 1
    for s in cands:                     # fill remainder by p95
        if len(picked) == 5:
            break
        if s not in picked:
            picked.append(s)

    print(f"\npicked {len(picked)} -> {OUT}")
    print(f"  {'seg':44s} {'sec':>6s} {'ev%':>6s} {'p95':>7s}  scenes")
    for s in picked:
        print(f"  {s['seg']:44s} {s['seconds']:6.0f} "
              f"{s['ev_ratio'] * 100:5.1f}% {s['motion_p95']:7.2f}  "
              f"{','.join(s['scenes'])}")
    with open(OUT, "w") as f:
        json.dump(picked, f, indent=2)


if __name__ == "__main__":
    main()

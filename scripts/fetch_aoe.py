"""Fetch Open-AoE segments from the hf-mirror copy of inclusionAI/OpenAoE-2000h
(~700h of egocentric phone video with SLAM camera trajectories and atomic
action annotations; see GUIDE.md for why this corpus).

Three phases, each cached (safe to re-run, skips existing files):

  probe   download ego_action_annotation.json + camera_traj.npz for N segments
          sampled across the repo listing -> data/aoe_index/<seg>/
  select  pick a diverse subset (scene/verb coverage, duration, head motion
          estimated from the trajectory) -> data/aoe_index/selected.json
  fetch   download the undistorted videos of the selected segments into
          data/aoe_src/<seg>/ in the flat layout prep_aoe.py expects

Only the files prep_aoe.py needs are fetched (no MANO hands.npz).

Usage:
  .venv/bin/python scripts/fetch_aoe.py probe --n 60
  .venv/bin/python scripts/fetch_aoe.py select --k 8
  .venv/bin/python scripts/fetch_aoe.py fetch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

BASE = "https://hf-mirror.com/datasets/inclusionAI/OpenAoE-2000h"
API = "https://hf-mirror.com/api/datasets/inclusionAI/OpenAoE-2000h"
INDEX = "data/aoe_index"
SRC = "data/aoe_src"

ANN = "ego_annotation/ego_action_annotation.json"
TRAJ = "ego_process/ego_hands_reconstruction/camera_traj.npz"
VIDEO = "ego_process/ego_undistorted_video/raw_video_undistorted.mp4"
VINFO = "ego_process/ego_undistorted_video/undistorted_video_info.json"

PASSIVE_VERBS = {"hold"}

# prep_aoe.py expects this flat layout
FLAT = {
    ANN: "ego_action_annotation.json",
    TRAJ: "camera_traj.npz",
    VIDEO: "raw_video_undistorted.mp4",
    VINFO: "undistorted_video_info.json",
}

# hf-mirror 403s the default Python-urllib UA
HEADERS = {"User-Agent": "Mozilla/5.0 (fetch_aoe.py)"}


def open_url(url: str, timeout: int = 300):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def download(rel: str, dest: str, retries: int = 3) -> int:
    """Fetch repo file `rel` to `dest`; returns bytes. Skips if present."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return os.path.getsize(dest)
    url = f"{BASE}/resolve/main/{rel}"
    for attempt in range(retries):
        try:
            t0 = time.time()
            with open_url(url) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            dt = time.time() - t0
            print(f"    {rel.split('/')[-1]}: "
                  f"{os.path.getsize(dest) / 1e6:.1f} MB in {dt:.0f}s")
            return os.path.getsize(dest)
        except Exception as e:
            print(f"    retry {attempt + 1}/{retries} {rel}: {e}")
            time.sleep(5 * (attempt + 1))
    raise IOError(f"failed to download {url}")


def list_segments() -> list[str]:
    """All segment dirs in the repo (paginated tree listing)."""
    segs, cursor = [], None
    while True:
        url = f"{API}/tree/main?recursive=false&limit=1000"
        if cursor:
            url += f"&cursor={cursor}"
        with open_url(url, timeout=60) as r:
            page = json.load(r)
            link = r.headers.get("Link", "")
        segs += [e["path"] for e in page if e.get("type") == "directory"]
        nxt = [s for s in link.split(",") if 'rel="next"' in s]
        if len(page) < 1000 or not nxt:
            return segs
        cursor = nxt[0].split("cursor=")[1].split("&")[0].split(">")[0]


def cmd_probe(args) -> None:
    segs = list_segments()
    print(f"repo lists {len(segs)} segments")
    step = max(1, len(segs) // args.n)
    picked = segs[::step][:args.n]
    print(f"probing {len(picked)} evenly sampled segments -> {INDEX}/")
    for seg in picked:
        for rel in (ANN, TRAJ):
            download(f"{seg}/{rel}", f"{INDEX}/{seg}/{os.path.basename(rel)}")


def seg_stats(seg: str) -> dict | None:
    """Duration, active intervals, scene/verb sets, head-motion p95 — all
    from the probed annotation + trajectory (no video needed)."""
    try:
        ann = json.load(open(f"{INDEX}/{seg}/ego_action_annotation.json"))
        traj = np.load(f"{INDEX}/{seg}/camera_traj.npz")["cam_c2w"]
    except Exception:
        return None
    seconds = max(float(s["end_ts"]) for s in ann) if ann else 0.0
    scenes = {s.get("scene", "?") for s in ann}
    verbs, active = set(), []
    for s in ann:
        acts = s.get("atomic_action") or []
        verbs |= {a.get("verb", "?") for a in acts}
        if any(a.get("verb") not in PASSIVE_VERBS and a.get("hand") != "none"
               for a in acts):
            active.append((float(s["start_ts"]), float(s["end_ts"])))
    # mean physical rotation rate (deg/s) from consecutive cam_c2w poses;
    # fps recovered from traj length over annotated duration
    rates = []
    for t in range(1, len(traj)):
        R = traj[t][:3, :3].T @ traj[t - 1][:3, :3]
        c = min(max((np.trace(R) - 1.0) / 2.0, -1.0), 1.0)
        rates.append(np.degrees(np.arccos(c)))
    fps = (len(traj) - 1) / seconds if seconds > 0 else 30.0
    motion_p95 = float(np.percentile(rates, 95)) * fps if rates else 0.0
    return {"seg": seg, "seconds": round(seconds, 1),
            "scenes": sorted(scenes), "verbs": sorted(verbs),
            "n_active": len(active),
            "motion_p95": round(motion_p95, 2)}


def cmd_select(args) -> None:
    stats = []
    for f in sorted(os.listdir(INDEX)):
        if os.path.isdir(f"{INDEX}/{f}"):
            s = seg_stats(f)
            if s:
                stats.append(s)
    pool = [s for s in stats if s["seconds"] >= 90 and s["n_active"] >= 3]
    print(f"{len(stats)} probed, {len(pool)} pass the >=90s & >=3-active "
          f"filter")
    if args.motion_min > 0:
        hi = [s for s in pool if s["motion_p95"] >= args.motion_min]
        print(f"{len(hi)} pass motion_p95 >= {args.motion_min} deg/s "
              f"(pool p50/p95 motion: "
              f"{np.percentile([s['motion_p95'] for s in pool], 50):.2f}/"
              f"{np.percentile([s['motion_p95'] for s in pool], 95):.2f})")
        pool = hi
    # greedy max-coverage: new scenes dominate, then new verbs, then head
    # motion closest to 30 deg/s (enough ego-rotation to matter, not a rig
    # shake-out)
    picked, covered_scenes, covered_verbs = [], set(), set()
    while pool and len(picked) < args.k:
        def score(s):
            return (10 * len(set(s["scenes"]) - covered_scenes)
                    + len(set(s["verbs"]) - covered_verbs)
                    - 0.02 * abs(s["motion_p95"] - 30.0))
        best = max(pool, key=score)
        pool.remove(best)
        picked.append(best)
        covered_scenes |= set(best["scenes"])
        covered_verbs |= set(best["verbs"])
    with open(f"{INDEX}/selected.json", "w") as f:
        json.dump(picked, f, indent=2)
    print(f"selected {len(picked)} -> {INDEX}/selected.json")
    for s in picked:
        print(f"  {s['seg']}: {s['seconds']:6.0f}s  scenes={s['scenes']}  "
              f"verbs={len(s['verbs'])}  active={s['n_active']}  "
              f"motion_p95={s['motion_p95']} deg/s")


def cmd_fetch(args) -> None:
    selected = json.load(open(f"{INDEX}/selected.json"))
    print(f"fetching {len(selected)} segments -> {SRC}/")
    for s in selected:
        seg = s["seg"]
        for rel, flat in FLAT.items():
            download(f"{seg}/{rel}", f"{SRC}/{seg}/{flat}")
    print("done. Next: scripts/prep_aoe.py per segment, then "
          "scripts/extract_featmaps.py data/aoe")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="download annotations+trajectories")
    p.add_argument("--n", type=int, default=60)
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("select", help="pick diverse subset")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--motion-min", type=float, default=0.0,
                   help="keep only segments with traj-derived motion_p95 "
                        ">= this (deg/s) — for the walking-scale regime")
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("fetch", help="download selected videos")
    p.set_defaults(fn=cmd_fetch)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

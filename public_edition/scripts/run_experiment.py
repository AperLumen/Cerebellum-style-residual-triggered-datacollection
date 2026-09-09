"""Run the cerebellum-style pipeline + baselines on a recording and report.

Usage:
  .venv/bin/python scripts/run_experiment.py demo_scene [--imu-offset 0.0]
"""

from __future__ import annotations

import argparse
import json

from cerebellum_cancel.core import CerebellumCancel, TriggerConfig
from cerebellum_cancel.eval import (evaluate, naive_diff_baseline,
                                    stream_duration_and_fps, uniform_baseline)
from cerebellum_cancel.pipeline import run_pipeline


def resolve_video_path(rec_dir: str) -> str:
    import json as _json, os
    meta_p = f"{rec_dir}/events.json"
    if os.path.exists(meta_p):
        with open(meta_p) as fp:
            v = _json.load(fp).get("video")
        if v:
            return os.path.join(rec_dir, v)
    if os.path.isdir(f"{rec_dir}/frames"):
        return f"{rec_dir}/frames/%05d.png"
    return f"{rec_dir}/video.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="recording dir with video.mp4, imu.csv, events.json")
    ap.add_argument("--imu-offset", type=float, default=0.0,
                    help="seconds added to IMU clock to align with video")
    args = ap.parse_args()

    with open(f"{args.dir}/events.json") as fp:
        meta = json.load(fp)
    events = [tuple(e) for e in meta["events"]]
    if args.imu_offset == 0.0 and meta.get("imu_offset"):
        args.imu_offset = meta["imu_offset"]
    video_path = resolve_video_path(args.dir)
    duration, fps = stream_duration_and_fps(video_path, meta.get("fps"))

    cfg = TriggerConfig()
    result = run_pipeline(video_path, f"{args.dir}/imu.csv",
                          cancel=CerebellumCancel(),
                          trigger_cfg=cfg, imu_offset=args.imu_offset,
                          fps=meta.get("fps"), mirror=meta.get("mirror", False),
                          crop=tuple(meta["crop"]) if meta.get("crop") else None)

    rows = [evaluate("cerebellum", result.clips, events, duration, fps)]

    naive_clips = naive_diff_baseline(video_path, cfg)
    rows.append(evaluate("naive_diff", naive_clips, events, duration, fps))

    # uniform baseline matched to the cerebellum method's own storage cost
    rows.append(evaluate("uniform(matched)",
                         uniform_baseline(duration, rows[0].stored_fraction),
                         events, duration, fps))
    # generous uniform baseline at 5% of stream
    rows.append(evaluate("uniform(5%)",
                         uniform_baseline(duration, 0.05),
                         events, duration, fps))

    print(f"\nstream: {duration:.1f}s @ {fps:.0f}fps, {len(events)} ground-truth events")
    print(f"{'method':<20}{'recall':>8}{'precision':>11}{'stored':>9}{'clips':>7}")
    for r in rows:
        print(f"{r.method:<20}{r.recall:>8.2f}{r.precision:>11.2f}"
              f"{r.stored_fraction:>9.3f}{r.n_clips:>7}")

    import numpy as np
    assert result.surprise is not None   # run_pipeline always fills the trace
    np.save(f"{args.dir}/surprise.npy", result.surprise)
    with open(f"{args.dir}/clips_cerebellum.json", "w") as fp:
        json.dump(result.clips, fp)
    print(f"\nsurprise trace -> {args.dir}/surprise.npy, clips -> clips_cerebellum.json")


if __name__ == "__main__":
    main()

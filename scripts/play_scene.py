"""Play a synthetic scene fullscreen — the Mac's screen becomes the rig's
"world", moving exactly according to the motion that generated the scene.
The known motion stream doubles as the corollary discharge (use the same
imu.csv the scene was rendered from).

A white flash is shown at the start and end; the capture script detects the
first flash to compute video-clock offset (shake-to-sync equivalent).

Usage: .venv/bin/python scripts/play_scene.py data/synthetic/frames
"""

from __future__ import annotations

import glob
import sys
import time

import cv2
import numpy as np

FPS = 30


def sync_digit(d: str) -> np.ndarray:
    """The countdown digit shown before playback — doubles as the sync
    template (far more distinctive than noise texture for matching)."""
    img = np.zeros((240, 320), np.uint8)
    cv2.putText(img, d, (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 8, 255, 25)
    return img


def main():
    frames_dir = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/frames"
    paths = sorted(glob.glob(f"{frames_dir}/*.png"))
    if not paths:
        raise SystemExit(f"no frames in {frames_dir}")
    # fill the screen: Qt's "fullscreen property" is unreliable on macOS, so
    # size the window to the display explicitly and also try real fullscreen
    try:
        import tkinter
        root = tkinter.Tk()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        sw = sh = None
    cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
    if sw:
        cv2.resizeWindow("scene", sw, sh)
        cv2.moveWindow("scene", 0, 0)
    cv2.setWindowProperty("scene", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # countdown: 3-2-1, one second each. Scene frame 0 starts one second
    # after the '1' appears — sync_offset.py uses this to compute the lag.
    for d in "321":
        img = sync_digit(d)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            cv2.imshow("scene", img)
            if cv2.waitKey(30) == 27:
                cv2.destroyAllWindows()
                return

    t0 = time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(p, 0)
        cv2.imshow("scene", img)
        # schedule against wall clock so playback runs at true 30 fps
        delay = max(1, int((i + 1) / FPS * 1000 - (time.time() - t0) * 1000))
        if cv2.waitKey(delay) == 27:
            break
    cv2.destroyAllWindows()
    print(f"played {len(paths)} frames")


if __name__ == "__main__":
    main()

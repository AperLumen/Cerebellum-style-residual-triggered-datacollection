"""Two follow-up analyses for the negative results of the review.

N1 (Section 6.2, "residual spikes anti-correlated with action annotations"):
    quantify WHY. Compare gyro energy (and, where cached featmaps exist, the
    actual prediction residual) inside vs outside annotated event intervals,
    and report the AUC of each quantity as an event predictor.

N2 (Section 6.3 item 5, "walking-level gyro warp is worse than naive"):
    decompose the model-class error. Rebuild the warp from a full 3-axis
    rotation homography H = K R K^-1 and ablate which axes are compensated
    (naive / yaw only / pitch only / roll only / yaw+pitch / all three).
    The shipped 2D-shift model is the "yaw+pitch (shift)" row.

Everything is pure calibration + rollout: no training, no fitting per frame.

Usage: .venv/bin/python scripts/analyze_negatives.py [--dataset DIR ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from cerebellum_cancel.io import load_imu_csv

FPS = 30.0
BORDER = 2
MAX_FRAMES_ABLATION = 1500


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    r = np.arange(1, len(scores) + 1, dtype=np.float64)
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks[order] = r
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def rodrigues(rv: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector (radians)."""
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rot_homography(rv: np.ndarray, f_map: float, h: int, w: int) -> np.ndarray:
    """Homography of the plane at infinity for a camera rotation: the content
    of frame t-1 seen at frame t is F_{t-1}(K R K^-1 x)."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    K = np.array([[f_map, 0.0, cx], [0.0, f_map, cy], [0.0, 0.0, 1.0]])
    Kinv = np.linalg.inv(K)
    R = rodrigues(rv)
    return K @ R @ Kinv


def warp_channels(F: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply H to every channel of a (C, H, W) map with dst(x)=src(H x)."""
    c, h, w = F.shape
    out = np.empty_like(F)
    for ch in range(c):
        out[ch] = cv2.warpPerspective(F[ch], H, (w, h),
                                      flags=cv2.INTER_LINEAR
                                      | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REFLECT)
    return out


def mean_abs_err(pred: np.ndarray, cur: np.ndarray) -> float:
    """Mean |err| over interior cells (border excluded, matching readout)."""
    d = np.abs(cur - pred)[:, BORDER:-BORDER, BORDER:-BORDER]
    return float(d.mean())


def gyro_mag(gyro: np.ndarray) -> np.ndarray:
    return np.linalg.norm(gyro, axis=1)


# --------------------------------------------------------------------------
# N1: event vs non-event energy
# --------------------------------------------------------------------------
def analyse_energy(data_dir: str, cached: bool):
    rows = []
    for s in sorted(glob.glob(f"{data_dir}/val/seq_*")):
        lab_path = f"{s}/labels.npy"
        if not os.path.exists(lab_path):
            continue
        lab = np.load(lab_path).astype(np.int8)
        gy = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                      dtype=np.float32)
        n = min(len(lab), len(gy))
        lab, gy = lab[:n], gy[:n]
        gm = gyro_mag(gy)
        # 5-frame centred smoothing: the trigger sees a clip, not a frame
        k = np.ones(5) / 5.0
        gm_s = np.convolve(gm, k, mode="same")
        ev, non = lab.astype(bool), ~lab.astype(bool)
        row = {
            "seq": os.path.basename(s),
            "n": n,
            "event_frac": float(ev.mean()),
            "gyro_med_event": float(np.median(gm[ev])) if ev.any() else None,
            "gyro_med_non": float(np.median(gm[non])) if non.any() else None,
            "gyro_p95_event": float(np.percentile(gm[ev], 95)) if ev.any() else None,
            "gyro_p95_non": float(np.percentile(gm[non], 95)) if non.any() else None,
            "auc_gyro": auc(lab, gm),
            "auc_gyro_smooth": auc(lab, gm_s),
        }
        if cached and os.path.exists(f"{s}/featmaps.npy"):
            F = np.load(f"{s}/featmaps.npy", mmap_mode="r")
            m = min(len(F), n)
            lab_c = lab[:m]
            ev_c, non_c = lab_c.astype(bool), ~lab_c.astype(bool)
            fm = json.load(open(f"{s}/events.json"))["f_map"]
            # raw prediction residual under the shipped 2D-shift model
            err = np.zeros(m, dtype=np.float32)
            for t in range(1, m):
                Ft = np.asarray(F[t], dtype=np.float32)
                Fp = np.asarray(F[t - 1], dtype=np.float32)
                rs = shipped_shift(Fp, gy[t], fm)
                err[t] = mean_abs_err(rs, Ft)
            # first-difference (no predictor at all), the naive frame-diff baseline
            raw = np.zeros(m, dtype=np.float32)
            for t in range(1, m):
                Ft = np.asarray(F[t], dtype=np.float32)
                Fp = np.asarray(F[t - 1], dtype=np.float32)
                raw[t] = mean_abs_err(Fp, Ft)
            row["resid_med_event"] = float(np.median(err[ev_c])) if ev_c.any() else None
            row["resid_med_non"] = float(np.median(err[non_c])) if non_c.any() else None
            row["auc_resid"] = auc(lab_c, err)
            row["auc_rawdiff"] = auc(lab_c, raw)
            row["resid_ratio"] = (row["resid_med_event"]
                                  / max(row["resid_med_non"] or 1e-6, 1e-6))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# N2: axis ablation on the warp model
# --------------------------------------------------------------------------
AXIS_MODELS = {
    "naive (no warp)": None,
    "shipped 2D shift": "shift",
    "yaw only": (1,),
    "pitch only": (0,),
    "roll only": (2,),
    "yaw+pitch": (0, 1),
    "all three": (0, 1, 2),
}


def shipped_shift(Fp: np.ndarray, g: np.ndarray, f_map: float) -> np.ndarray:
    """Bit-faithful replication of the shipped model
    (featmap_cancel.expected_shift + shift_map): a pure 2D translation from
    yaw (x) and pitch (y). Roll is not expressible in this model.

    Convention verified against shift_map on the interior (border excluded,
    since shift_map wraps with np.roll while this reflects): max |diff| = 0.0."""
    dx = f_map * np.tan(np.radians(g[1]) / FPS)
    dy = f_map * np.tan(np.radians(g[0]) / FPS)
    h, w = Fp.shape[1], Fp.shape[2]
    out = np.empty_like(Fp)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    for ch in range(Fp.shape[0]):
        out[ch] = cv2.warpAffine(Fp[ch], M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    return out


def calibrate_axis_signs(F, gy, idx, f_map, h, w, cal_idx):
    """Fit one sign per gyro axis by minimising the 3-axis homography residual
    on `cal_idx` only (coordinate descent over 2^3 combinations)."""
    best, best_err = (1.0, 1.0, 1.0), np.inf
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                sg = np.array([sx, sy, sz], dtype=np.float32)
                e = 0.0
                for t in cal_idx:
                    Ft = np.asarray(F[idx[t]], dtype=np.float32)
                    Fp = np.asarray(F[idx[t - 1]], dtype=np.float32)
                    rv = np.radians(gy[idx[t]] * sg) / FPS
                    H = rot_homography(rv, f_map, h, w)
                    e += mean_abs_err(warp_channels(Fp, H), Ft)
                if e < best_err:
                    best, best_err = (sx, sy, sz), e
    return np.array(best, dtype=np.float32)


def analyse_axis(data_dir: str):
    rows = []
    for s in sorted(glob.glob(f"{data_dir}/val/seq_*")):
        if not os.path.exists(f"{s}/featmaps.npy"):
            continue
        F = np.load(f"{s}/featmaps.npy", mmap_mode="r")
        gy = np.array([g.gyro for g in load_imu_csv(f"{s}/imu.csv")],
                      dtype=np.float32)
        meta = json.load(open(f"{s}/events.json"))
        f_map = float(meta["f_map"])
        n = min(len(F), len(gy), MAX_FRAMES_ABLATION)
        idx = np.linspace(0, n - 1, n).astype(int)
        _, h, w = np.asarray(F[0]).shape

        # calibrate per-axis signs on the first half only, report on the second
        split = n // 2
        cal_idx = np.linspace(1, split - 1, min(split - 1, 60)).astype(int)
        ev_idx = np.linspace(split, n - 1, n - split).astype(int)
        signs = calibrate_axis_signs(F, gy, idx, f_map, h, w, cal_idx)

        res = {}
        for name, axes in AXIS_MODELS.items():
            errs, mags = [], []
            for t in ev_idx:
                Ft = np.asarray(F[idx[t]], dtype=np.float32)
                Fp = np.asarray(F[idx[t - 1]], dtype=np.float32)
                g = gy[idx[t]]
                if axes is None:
                    pred = Fp
                elif axes == "shift":
                    pred = shipped_shift(Fp, g, f_map)
                else:
                    keep = np.zeros(3, dtype=np.float32)
                    for a in axes:
                        keep[a] = g[a]
                    rv = np.radians(keep * signs) / FPS
                    pred = warp_channels(Fp, rot_homography(rv, f_map, h, w))
                errs.append(mean_abs_err(pred, Ft))
                mags.append(float(np.linalg.norm(g)))
            errs = np.array(errs, dtype=np.float64)
            mags = np.array(mags, dtype=np.float64)
            q = np.percentile(mags, 75)
            hi = mags >= q
            res[name] = {
                "mean": float(errs.mean()),
                "med": float(np.median(errs)),
                "hi_motion_mean": float(errs[hi].mean()) if hi.any() else None,
                "lo_motion_mean": float(errs[~hi].mean()) if (~hi).any() else None,
            }
        base = res["naive (no warp)"]["mean"]
        for name in res:
            res[name]["gain_vs_naive"] = 1.0 - res[name]["mean"] / base
        gmag = np.abs(gy[:n])
        rows.append({
            "seq": os.path.basename(s) + "  [" + os.path.basename(os.path.dirname(
                os.path.dirname(s))) + "]",
            "f_map": f_map,
            "n_frames": int(n),
            "signs": [float(x) for x in signs],
            "gyro_p50": float(np.median(np.linalg.norm(gmag, axis=1))),
            "gyro_p95": float(np.percentile(np.linalg.norm(gmag, axis=1), 95)),
            "roll_p95": float(np.percentile(np.abs(gy[:, 2]), 95)),
            "models": res,
        })
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()

    out = {}

    print("=" * 90)
    print("N1  gyro energy and raw residual: event vs non-event frames")
    print("=" * 90)
    e_rows = []
    for d in ("data/aoe", "data/aoe_event"):
        if os.path.isdir(d):
            e_rows += analyse_energy(d, cached=os.path.isdir(f"{d}/val/seq_000")
                                     and os.path.exists(
                                         f"{d}/val/seq_000/featmaps.npy"))
    hdr = (f"{'seq':<30}{'ev%':>6}{'gyro_ev':>9}{'gyro_non':>10}{'ratio':>7}"
           f"{'AUC_g':>8}{'AUC_gs':>8}{'AUC_r':>8}{'AUC_rd':>8}")
    print(hdr)
    usable = [r for r in e_rows
              if r["gyro_med_event"] is not None and r["gyro_med_non"] is not None]
    for r in e_rows:
        tag = r["seq"].split("/")[0][-4:] + "/" + r["seq"].split("/")[-1]
        if r["gyro_med_event"] is None or r["gyro_med_non"] is None:
            print(f"{tag:<30}{r['event_frac']*100:>5.1f}%   degenerate "
                  f"(single class) — excluded")
            continue
        auc_r = r.get("auc_resid")
        auc_rd = r.get("auc_rawdiff")
        print(f"{tag:<30}{r['event_frac']*100:>5.1f}%"
              f"{r['gyro_med_event']:>9.1f}{r['gyro_med_non']:>10.1f}"
              f"{r['gyro_med_event']/max(r['gyro_med_non'],1e-6):>7.2f}"
              f"{r['auc_gyro']:>8.3f}{r['auc_gyro_smooth']:>8.3f}"
              f"{(f'{auc_r:>8.3f}' if auc_r is not None else '       -')}"
              f"{(f'{auc_rd:>8.3f}' if auc_rd is not None else '       -')}")
    if usable:
        print(f"\n  {'mean over ' + str(len(usable)) + ' segments':<30}"
              f"{np.mean([r['event_frac'] for r in usable])*100:>5.1f}%"
              f"{np.mean([r['gyro_med_event'] for r in usable]):>9.1f}"
              f"{np.mean([r['gyro_med_non'] for r in usable]):>10.1f}"
              f"{np.mean([r['gyro_med_event']/r['gyro_med_non'] for r in usable]):>7.2f}"
              f"{np.mean([r['auc_gyro'] for r in usable]):>8.3f}"
              f"{np.mean([r['auc_gyro_smooth'] for r in usable]):>8.3f}")
    out["energy"] = e_rows

    print()
    print("=" * 78)
    print("N2  axis ablation (mean |err| on interior cells, feature space)")
    print("=" * 78)
    a_rows = []
    for d in ("data/aoe_event", "data/aoe_motion"):
        if os.path.isdir(d) and glob.glob(f"{d}/val/seq_*/featmaps.npy"):
            a_rows += analyse_axis(d)
    for r in a_rows:
        print(f"\n{r['seq']}  (f_map={r['f_map']:.2f}, n={r['n_frames']}, "
              f"gyro p50={r['gyro_p50']:.1f} p95={r['gyro_p95']:.1f} deg/s)")
        print(f"  {'warp model':<20}{'mean |err|':>12}{'vs naive':>10}"
              f"{'hi-motion':>11}{'lo-motion':>11}")
        for name, m in r["models"].items():
            print(f"  {name:<20}{m['mean']:>12.4f}{m['gain_vs_naive']*100:>9.1f}%"
                  f"{m['hi_motion_mean']:>11.4f}{m['lo_motion_mean']:>11.4f}")
    out["axis"] = a_rows

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "negative_followups.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""Prepare a REAL-IMU consumer-phone recording (Galaxy S21, held by hand) for
the cerebellum-cancel pipeline — the first sequence in this project whose
corollary discharge is a **real gyroscope** rather than a SLAM-trajectory
derivative or a synthetic rig signal (GUIDE §4.2 boundary 5).

Source: simorautiainen/moving-in-messy-apartment-video-with-imu (HF),
samsung S21, hand-held, 1088x1088 RGBA frames + per-frame gyro/accel from the
phone's own IMU, nanosecond capture timestamps. Layout:

    data_<ts>/image<i>.bin     20-byte header (w,h,c,data_size,?=0) + raw RGBA
    data_<ts>/gyro<i>.bin      3 x f32
    data_<ts>/accel<i>.bin     3 x f32
    data_<ts>/timestamp<i>.bin i64 nanoseconds

Why this matters for the project. Every real-data number so far came from
Open-AoE, whose gyro is *derived* by differentiating a visual-SLAM trajectory.
That derivative inherits SLAM's failure modes (low texture / fast motion), and
the walking-level probe (§0.4 row 13) named exactly that as one confound for
the "gyro warp does not pay off on real data" result. A phone IMU has none of
those failure modes, so this sequence separates "the corollary discharge is
bad" from "the warping model is bad".

Calibration. The dataset documents neither the gyro unit (deg/s vs rad/s) nor
the axis order/signs nor the focal length, so all three are measured here
against DIS optical flow — the same auto-calibration the AoE prep uses:

  1. unit: fit the scalar that best predicts the DIS-measured content shift
     (rad/s and deg/s differ by a factor of 57.3, so this is not ambiguous
     for any real motion);
  2. per-axis sign: flip each axis so the predicted shift correlates
     positively with the measured one;
  3. f_map: least-squares scale of the content shift (map px per unit rate),
     which folds in the unknown focal length.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/prep_real_imu.py \
      data/real_imu_raw/data_1714767956 --out data/real_imu/val/seq_000
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import cv2
import numpy as np

from cerebellum_cancel.flow_cancel import dis_calc

MAP_W = 28                      # featmap width (MobileNetV3 features[:3])
FRAME_SIZE = (1088, 1088)       # see image header in the source


def read_frame(path: str) -> np.ndarray:
    """Read one image<i>.bin: 5 x int32 header then raw w*h*c bytes."""
    with open(path, "rb") as f:
        w, h, c, data_size, _pad = struct.unpack("<iiiii", f.read(20))
        raw = f.read(data_size)
    img = np.frombuffer(raw, dtype=np.uint8)
    if img.size != w * h * c:
        raise ValueError(f"{path}: payload {img.size} != {w}x{h}x{c}")
    img = img.reshape(h, w, c)
    if c == 4:
        img = img[:, :, :3]                     # drop alpha
    return img


def read_vec3(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.array(struct.unpack("<fff", f.read(12)), dtype=np.float32)


def read_time_ns(path: str) -> int:
    with open(path, "rb") as f:
        return int(struct.unpack("<q", f.read(8))[0])


def load_session(src: str, limit: int | None):
    n = len([f for f in os.listdir(src) if f.startswith("image")])
    if limit:
        n = min(n, limit)
    gray, rgb, gyro, accel, t_ns = [], [], [], [], []
    for i in range(1, n + 1):
        rgb.append(read_frame(f"{src}/image{i}.bin"))
        gyro.append(read_vec3(f"{src}/gyro{i}.bin"))
        accel.append(read_vec3(f"{src}/accel{i}.bin"))
        t_ns.append(read_time_ns(f"{src}/timestamp{i}.bin"))
    for im in rgb:
        gray.append(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
    return (np.stack(rgb), gray, np.stack(gyro), np.stack(accel),
            np.array(t_ns, dtype=np.int64))


def fit_calibration(gray: list, gyro: np.ndarray, t_ns: np.ndarray,
                    calib_size: int, max_frames: int) -> dict:
    """标定 (轴映射, 符号, 焦距)。

    方法（两步，都不依赖任何自由缩放）：

    1. **轴映射 × 符号**：用逐帧**相关系数**判定——预测的 yaw/pitch 位移与
       DIS 实测位移逐帧相关，映射对了 r 接近 1（实测 +0.93），映射错了接近 0。
       相关系数对整体尺度不敏感，所以这一步与焦距解耦。
    2. **焦距**：过原点最小二乘斜率 ``f = Σ(ω·d) / Σ(ω²)``，其中 ω 是
       rad/frame、d 是实测位移——斜率就是该帧尺寸下的像素焦距。

    为什么不用"平均光流 + 拟合"一把梭（初版的做法）：本段数据**平移显著**
    （实测位移 p95 ≈ 26 px @256），旋转与平移在一阶位移上不可分辨，直接拟合
    会把平移吸收进焦距——初版就这样得到 1194 px（FOV 12°，物理上不可能）。
    分离的做法是先靠**时序相关**锁定轴映射（不受平移影响，因为平移与旋转时序
    不同源），再在该映射下取斜率。

    单位：Android 陀螺仪 API 标准返回 **rad/s**，这里由拟合直接确认（若单位
    假设错 57.3 倍，第二步的斜率会差 57.3 倍、FOV 荒谬，一眼可辨）。
    """
    n_use = min(len(gray), max_frames)
    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dx, dy = [], []
    for k in range(1, n_use):
        a = cv2.resize(gray[k - 1], (calib_size, calib_size),
                       interpolation=cv2.INTER_AREA)
        b = cv2.resize(gray[k], (calib_size, calib_size),
                       interpolation=cv2.INTER_AREA)
        fl = dis_calc(dis, a, b)
        mag = np.linalg.norm(fl, axis=2)
        # 取位移最大的 10% 像素：真实运动区，避开平滑区的噪声地板
        conf = mag > np.percentile(mag, 90)
        dx.append(float(np.median(fl[:, :, 0][conf])))
        dy.append(float(np.median(fl[:, :, 1][conf])))
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    g = gyro[:n_use - 1]
    dt = float(np.median(np.diff(t_ns[:n_use]) / 1e9))

    # --- 1) 轴映射 × 符号：用逐帧相关（尺度无关）---
    # 注意配对：位移 flow(t-1 -> t) 由这段间隔内的角速度产生，用 gyro[t-1]。
    # （用 gyro[t] 也能得到同号的高相关，但物理上对应下一段间隔。）
    rx_all = gyro[:n_use - 1]                       # (T,3) raw rad/s
    best_x: tuple | None = None
    best_y: tuple | None = None
    for ax in range(3):
        for s in (1, -1):
            r = float(np.corrcoef(s * rx_all[:, ax], dx)[0, 1])
            if not np.isfinite(r):
                continue
            if best_x is None or abs(r) > abs(best_x[0]):
                best_x = (r, ax, s)
    for ax in range(3):
        for s in (1, -1):
            r = float(np.corrcoef(s * rx_all[:, ax], dy)[0, 1])
            if not np.isfinite(r):
                continue
            if best_y is None or abs(r) > abs(best_y[0]):
                best_y = (r, ax, s)
    assert best_x is not None and best_y is not None
    r_x, ay, sy = best_x
    r_y, ap, sp = best_y
    if ay == ap:
        # 同一轴不能同时驱动两个方向：退化为只有水平可控（pitch 以平移为主
        # 是这类手持数据的常见情形），pitch 取次优的不同轴，符号沿用相关符号。
        for ax in range(3):
            if ax == ay:
                continue
            for s in (1, -1):
                r = float(np.corrcoef(s * rx_all[:, ax], dy)[0, 1])
                if np.isfinite(r) and abs(r) > abs(r_y):
                    r_y, ap, sp = r, ax, s
        if ay == ap:                            # 仍然撞车：让 pitch 用剩下那轴
            ap = next(a for a in range(3) if a != ay)

    # --- 2) 焦距：过原点斜率（rad/frame -> px）---
    # ！！raw 值已经是 rad/s（Android 陀螺仪 API 标准），这里**只乘 dt**；
    # 再套一层 np.radians() 会多除 57.3，把焦距放大到 FOV≈0.7° 的荒谬值
    # （这个坑在标定调试中实际踩到过，两次）。
    wx = sy * rx_all[:len(dx), ay] * dt
    wy = sp * rx_all[:len(dy), ap] * dt
    f_x = float((wx * dx).sum() / max((wx * wx).sum(), 1e-12))
    f_y = float((wy * dy).sum() / max((wy * wy).sum(), 1e-12))
    # 水平轴最可信（垂直方向常被平移污染）：以 |f_x| 为准，除非它明显异常
    f_px = abs(f_x)
    if not (30.0 <= 2 * np.degrees(np.arctan(calib_size / 2 / f_px)) <= 120.0):
        f_px = abs(f_y)
    fov = float(2 * np.degrees(np.arctan(calib_size / 2 / f_px)))

    # 解释度（供报告）
    res_x = dx - f_x * wx
    res_y = dy - f_y * wy
    r2_x = 1 - float(res_x.var() / max(dx.var(), 1e-12))
    r2_y = 1 - float(res_y.var() / max(dy.var(), 1e-12))

    return {"yaw_axis": ay, "yaw_sign": sy, "pitch_axis": ap,
            "pitch_sign": sp, "f_px_calib": f_px,
            "f_x_slope": f_x, "f_y_slope": f_y,
            "calib_size": calib_size, "dt": dt, "unit": 1.0,
            "unit_tag": "rad/s", "r_x": r_x, "r_y": r_y,
            "r2_x": r2_x, "r2_y": r2_y, "fov_deg": fov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=256,
                    help="frame side saved to disk (square source)")
    ap.add_argument("--calib-frames", type=int, default=200)
    ap.add_argument("--fov-deg", type=float, default=0.0,
                    help=">0 时用给定的对角 FOV 直接算 f_map（跳过光流标定）")
    args = ap.parse_args()

    rgb, gray, gyro, accel, t_ns = load_session(
        args.src, args.limit or None)
    n = len(rgb)
    t = (t_ns - t_ns[0]) / 1e9
    dt = np.diff(t)
    fps = float(1.0 / np.median(dt))
    print(f"读取 {n} 帧，跨度 {t[-1]:.2f}s，帧率中位 {fps:.2f} fps"
          f"（间隔 {np.percentile(dt, 5) * 1e3:.1f}–"
          f"{np.percentile(dt, 95) * 1e3:.1f} ms，真机不等间隔）")

    # ---- 标定：单位 / 轴映射 / 符号 / 焦距 ----
    calib = fit_calibration(gray, gyro, t_ns, 256, min(args.calib_frames, n))
    print(f"标定（相关法定轴映射, 斜率法定焦距）:")
    print(f"  单位 {calib['unit_tag']}（倍率 {calib['unit']:.5f}）"
          f"  FOV≈{calib['fov_deg']:.0f}°（f={calib['f_px_calib']:.0f}px "
          f"@{calib['calib_size']}）")
    print(f"  yaw   <- raw[{calib['yaw_axis']}] x{calib['yaw_sign']:+d}"
          f"   pitch <- raw[{calib['pitch_axis']}] x{calib['pitch_sign']:+d}")

    # 组装管线期望的 gyro 向量 [pitch, yaw, roll]，单位 deg/s。
    # raw 是 rad/s（Android 标准）-> deg/s 只需乘 180/π。
    # （CancelConfig.expected_shift 里 gyro[0]->pitch->dy、gyro[1]->yaw->dx）
    to_deg = 180.0 / np.pi
    pitch = calib["pitch_sign"] * gyro[:, calib["pitch_axis"]] * to_deg
    yaw = calib["yaw_sign"] * gyro[:, calib["yaw_axis"]] * to_deg
    used = {calib["yaw_axis"], calib["pitch_axis"]}
    roll_axis = next(a for a in range(3) if a not in used)
    roll = gyro[:, roll_axis] * to_deg
    gd = np.stack([pitch, yaw, roll], axis=1).astype(np.float32)

    # f_map：焦距换算到**特征图**像素。
    # 管线把帧缩到 224 输入、取 stride-8 的 features[:3] -> 28 网格，所以
    #   f_map = f_224px / 8 = f_src * (224 / W_src) / 8 = f_src * 28 / W_src
    # （与 prep_aoe.py 的 f_map = fx * 28 / 1920 是同一个式子，那里 W_src=1920）
    f_src = calib["f_px_calib"] * (FRAME_SIZE[0] / calib["calib_size"])
    f_map = f_src * MAP_W / FRAME_SIZE[0]
    print(f"  焦距 {f_src:.0f}px（源 {FRAME_SIZE[0]}px 方帧）"
          f" -> f_map {f_map:.2f}（224 输入 / 28 网格）")
    print(f"gyro p95 |rate| = "
          f"{np.percentile(np.abs(gd), 95, axis=0).round(2)} deg/s"
          f"（[pitch, yaw, roll]），模长 p95 "
          f"{np.percentile(np.linalg.norm(gd, axis=1), 95):.1f}")

    # ---- 落盘：标准序列布局 ----
    os.makedirs(f"{args.out}/frames", exist_ok=True)
    for i in range(n):
        im = cv2.resize(rgb[i], (args.size, args.size),
                        interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{args.out}/frames/{i:05d}.png",
                    cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    meta = {
        "fps": fps,
        "size": args.size,
        "source": os.path.basename(args.src.rstrip("/")),
        "imu": "real_gyro (Galaxy S21, hand-held)",
        "gyro_unit": "deg/s",
        "gyro_order": "pitch,yaw,roll",
        "f_map": float(f_map),
        "focal_px_src": float(f_src),
        "calib": {k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in calib.items()},
        "frame_dt_median": float(np.median(dt)),
        "frame_dt_jitter_p95": float(np.percentile(dt, 95)
                                     - np.percentile(dt, 5)),
        "events": [],
    }
    with open(f"{args.out}/imu.csv", "w") as f:
        f.write("time,gyroscope.x,gyroscope.y,gyroscope.z\n")
        for i in range(n):
            f.write(f"{t[i]:.6f},{gd[i, 0]:.6f},{gd[i, 1]:.6f},{gd[i, 2]:.6f}\n")
    with open(f"{args.out}/events.json", "w") as f:
        json.dump(meta, f, indent=1)
    np.save(f"{args.out}/accel.npy", accel)

    print(f"\n-> {args.out}: {n} 帧 PNG @ {args.size}px, imu.csv (真 gyro), "
          f"events.json (f_map={f_map:.2f})")
    print("   注意: events 为空——真机数据没有人工事件标注。本段用于"
          "\n   **残差质量/成本**对照（naive vs gyro vs flow），不用于触发 recall。")


if __name__ == "__main__":
    main()

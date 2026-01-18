# evaluation.py
# ------------------------------------------------------------
# Lane Tracking ADAS v4.0 Evaluation Script (Baseline/Ablation)
# Imports your project by FILE PATH (no MODULE_NAME problems)
# Outputs:
#   - metrics_results.csv
#   - metrics_results.json
# ------------------------------------------------------------

import os
import json
import csv
import numpy as np
import importlib.util


# ===================== USER SETTINGS =========================
MAIN_FILE = "yol_takibi_v4.py"   # your project file (must contain class YolTakip)

VIDEO_RURAL = "road (2).mp4"     # rural video
VIDEO_HIGHWAY = "road (3).mp4"   # highway video

# Evaluate first N frames:
#   None => evaluate until video loops once (heuristic)
MAX_FRAMES = None   # e.g., 2000

# Jitter threshold in pixels (|center_t - center_{t-1}| > threshold => jitter)
JITTER_THRESH_PX = 20
# =============================================================


def safe_std(arr):
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr))


def center_x_from_points(sol, sag):
    """
    Robust center_x estimate from detected left/right points.
    Returns None if not enough points.
    """
    if sol is None or sag is None:
        return None
    if len(sol) < 3 or len(sag) < 3:
        return None

    sol_arr = np.array(sol, dtype=np.int32)
    sag_arr = np.array(sag, dtype=np.int32)

    centers = []
    ytol = 15

    # for each left point, find closest right point by y
    for sp in sol_arr:
        y = int(sp[1])
        diffs = np.abs(sag_arr[:, 1] - y)
        j = int(np.argmin(diffs))
        if diffs[j] <= ytol:
            cx = (int(sp[0]) + int(sag_arr[j, 0])) / 2.0
            centers.append(cx)

    if len(centers) == 0:
        return None
    return float(np.median(centers))


def width_from_points(sol, sag):
    """
    Median lane width estimate from matched y-level pairs.
    Returns None if not enough points.
    """
    if sol is None or sag is None:
        return None
    if len(sol) < 3 or len(sag) < 3:
        return None

    sol_arr = np.array(sol, dtype=np.int32)
    sag_arr = np.array(sag, dtype=np.int32)

    widths = []
    ytol = 15

    for sp in sol_arr:
        y = int(sp[1])
        diffs = np.abs(sag_arr[:, 1] - y)
        j = int(np.argmin(diffs))
        if diffs[j] <= ytol:
            w = abs(int(sag_arr[j, 0]) - int(sp[0]))
            widths.append(w)

    if len(widths) == 0:
        return None
    return float(np.median(widths))


def patch_tracker_for_cfg(tracker, cfg):
    """
    Non-invasive patching:
    We override methods on the tracker instance so we can switch modules ON/OFF
    without editing your main project file.
    """
    import types

    # ---- Patch kalman_uygula ----
    orig_kalman_uygula = tracker.kalman_uygula

    def kalman_uygula_patched(self, noktalar, kalman_dict):
        if not cfg["use_kalman"]:
            return noktalar
        return orig_kalman_uygula(noktalar, kalman_dict)

    tracker.kalman_uygula = types.MethodType(kalman_uygula_patched, tracker)

    # ---- Patch alan_koru_gelismis (perspective/width preservation) ----
    orig_alan_koru = tracker.alan_koru_gelismis

    def alan_koru_patched(self, sol_n, sag_n, maske=None):
        if not cfg["use_perspective_width"]:
            return sol_n, sag_n
        return orig_alan_koru(sol_n, sag_n, maske)

    tracker.alan_koru_gelismis = types.MethodType(alan_koru_patched, tracker)

    # ---- Patch _polyfit_noktalar (MAD on/off inside) ----
    orig_polyfit = tracker._polyfit_noktalar

    def polyfit_patched(self, noktalar, w):
        if not cfg["use_mad"]:
            # Simplified polyfit without MAD filtering
            if noktalar is None or len(noktalar) < 4:
                return noktalar
            try:
                arr = np.array(noktalar)
                xs = arr[:, 0]
                ys = arr[:, 1]
                deg = 2 if len(xs) >= 8 else 1
                coeffs = np.polyfit(ys, xs, deg)
                poly = np.poly1d(coeffs)
                min_y = int(ys.min())
                max_y = int(ys.max())

                sonuc = []
                for y in range(min_y, max_y, 5):
                    x = int(poly(y))
                    if 0 <= x < w:
                        sonuc.append([x, y])
                return sonuc
            except Exception:
                return noktalar

        return orig_polyfit(noktalar, w)

    tracker._polyfit_noktalar = types.MethodType(polyfit_patched, tracker)


def evaluate_one(video_path, tracker_factory, cfg, max_frames=None):
    """
    Evaluate one video with a given cfg.
    tracker_factory(video_path) -> YolTakip instance
    """
    tracker = tracker_factory(video_path)
    patch_tracker_for_cfg(tracker, cfg)

    # reset state
    if hasattr(tracker, "sifirla"):
        tracker.sifirla()

    total = 0
    success = 0

    centers = []
    widths = []
    jitter_count = 0
    prev_center = None

    # Heuristic: stop after first loop
    loop_detected = False

    while True:
        if max_frames is not None and total >= max_frames:
            break

        frame = tracker.frame_al()
        sol, sag = tracker.isle(frame)

        # Optional time-window smoothing (toggle)
        if cfg["use_time_window"]:
            sol = tracker.yumusat(sol, tracker.sol_gecmis, tracker.son_sol)
            sag = tracker.yumusat(sag, tracker.sag_gecmis, tracker.son_sag)

        tracker.son_sol, tracker.son_sag = sol, sag

        total += 1

        ok = (sol is not None and sag is not None and len(sol) >= 3 and len(sag) >= 3)
        if ok:
            success += 1

        cx = center_x_from_points(sol, sag)
        if cx is not None:
            centers.append(cx)
            if prev_center is not None and abs(cx - prev_center) > JITTER_THRESH_PX:
                jitter_count += 1
            prev_center = cx

        ww = width_from_points(sol, sag)
        if ww is not None:
            widths.append(ww)

        # Loop detection: your frame_al() restarts video when it ends.
        # We detect restart by checking CAP_PROP_POS_FRAMES.
        try:
            import cv2
            pos = int(tracker.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos <= 2 and total > 30:
                loop_detected = True
        except Exception:
            pass

        if loop_detected:
            break

    success_rate = 100.0 * success / max(total, 1)
    stability_sigma = safe_std(centers)
    jitter_rate = 100.0 * jitter_count / max(len(centers) - 1, 1)

    mean_width = float(np.mean(widths)) if len(widths) > 0 else 0.0
    width_sigma = safe_std(widths)

    return {
        "total_frames": int(total),
        "success_frames": int(success),
        "success_rate_%": round(success_rate, 3),
        "stability_sigma_px": round(stability_sigma, 3),
        "jitter_rate_%": round(jitter_rate, 3),
        "mean_lane_width_px": round(mean_width, 3),
        "lane_width_sigma_px": round(width_sigma, 3),
    }


def load_project_module(base_dir):
    """
    Import yol_takibi_v4.py by file path.
    """
    main_path = os.path.join(base_dir, MAIN_FILE)
    if not os.path.exists(main_path):
        raise FileNotFoundError(
            f"Ana kod bulunamadı: {main_path}\n"
            f"Kontrol et: evaluation.py ile {MAIN_FILE} aynı klasörde mi?"
        )

    spec = importlib.util.spec_from_file_location("lane_module", main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Validate videos
    v_rural = os.path.join(base_dir, VIDEO_RURAL)
    v_highway = os.path.join(base_dir, VIDEO_HIGHWAY)

    if not os.path.exists(v_rural):
        raise FileNotFoundError(f"Video bulunamadı: {v_rural}")
    if not os.path.exists(v_highway):
        raise FileNotFoundError(f"Video bulunamadı: {v_highway}")

    # Load module
    mod = load_project_module(base_dir)

    if not hasattr(mod, "YolTakip"):
        raise AttributeError(
            f"{MAIN_FILE} içinde 'YolTakip' sınıfı bulunamadı.\n"
            f"Dosyanda class YolTakip: ... var mı?"
        )

    YolTakip = getattr(mod, "YolTakip")

    # Factories (match your scenario settings)
    def tracker_factory_rural(video_path):
        # You used saturation for rural
        return YolTakip(video_path, "RURAL_EVAL", "saturation")

    def tracker_factory_highway(video_path):
        # You used beyaz for highway
        return YolTakip(video_path, "HIGHWAY_EVAL", "beyaz")

    # Baseline/Ablation configs
    CONFIGS = [
        ("B0_baseline", {
            "use_kalman": False,
            "use_time_window": False,
            "use_perspective_width": False,
            "use_mad": False
        }),
        ("B1_kalman", {
            "use_kalman": True,
            "use_time_window": False,
            "use_perspective_width": False,
            "use_mad": False
        }),
        ("B2_proposed", {
            "use_kalman": True,
            "use_time_window": True,
            "use_perspective_width": True,
            "use_mad": True
        }),
    ]

    results = []

    for cfg_name, cfg in CONFIGS:
        print(f"\n=== {cfg_name} | RURAL ({VIDEO_RURAL}) ===")
        r1 = evaluate_one(v_rural, tracker_factory_rural, cfg, max_frames=MAX_FRAMES)
        r1.update({"video": VIDEO_RURAL, "scenario": "rural", "config": cfg_name})
        results.append(r1)
        print(r1)

        print(f"\n=== {cfg_name} | HIGHWAY ({VIDEO_HIGHWAY}) ===")
        r2 = evaluate_one(v_highway, tracker_factory_highway, cfg, max_frames=MAX_FRAMES)
        r2.update({"video": VIDEO_HIGHWAY, "scenario": "highway", "config": cfg_name})
        results.append(r2)
        print(r2)

    # Save CSV
    csv_path = os.path.join(base_dir, "metrics_results.csv")
    fieldnames = [
        "video", "scenario", "config",
        "total_frames", "success_frames", "success_rate_%",
        "stability_sigma_px", "jitter_rate_%",
        "mean_lane_width_px", "lane_width_sigma_px"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)

    # Save JSON
    json_path = os.path.join(base_dir, "metrics_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nDONE ✅")
    print(f"CSV saved:  {csv_path}")
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()

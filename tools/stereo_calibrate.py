"""
Stereo Camera Calibration Script
=================================
Run this on your PC (not on the Android device).

Prerequisites:
    pip install opencv-python numpy

Workflow:
    1. Print the checkerboard (tools/checkerboard_9x6.png) on A4
    2. Glue it flat onto rigid cardboard
    3. Use the Android app's "Capture Calibration Pair" button to save
       stereo image pairs into a folder (or use adb pull)
    4. Place left images in  calibration_images/left/
       Place right images in calibration_images/right/
       (filenames must match: 01.jpg, 02.jpg, ...)
    5. Run:  python stereo_calibrate.py
    6. Copy the output stereo_calib.json into:
       app/src/main/assets/stereo_calib.json

Checkerboard spec (must match your printed board):
    - 9x6 inner corners
    - 25mm square size (measure after printing!)
"""

import cv2
import numpy as np
import json
import glob
import os
import sys

# ── Configuration (MUST match your printed checkerboard) ──────
INNER_COLS = 9          # inner corners horizontally
INNER_ROWS = 6          # inner corners vertically
SQUARE_SIZE_MM = 25.0   # measure with ruler after printing!

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEFT_DIR = os.path.join(SCRIPT_DIR, "calibration_images", "left")
RIGHT_DIR = os.path.join(SCRIPT_DIR, "calibration_images", "right")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "stereo_calib.json")

# Minimum number of valid image pairs needed
MIN_PAIRS = 10


def find_corners(img_gray, board_size):
    """Find checkerboard corners with sub-pixel refinement."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(img_gray, board_size, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def main():
    board_size = (INNER_COLS, INNER_ROWS)

    # Prepare 3D object points (z=0 plane)
    # e.g. (0,0,0), (25,0,0), (50,0,0), ..., (0,25,0), (0,50,0), ...
    objp = np.zeros((INNER_COLS * INNER_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:INNER_COLS, 0:INNER_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # scale to real-world mm

    # Gather image pairs
    left_files = sorted(glob.glob(os.path.join(LEFT_DIR, "*")))
    right_files = sorted(glob.glob(os.path.join(RIGHT_DIR, "*")))

    if not left_files:
        print(f"❌ No images found in {LEFT_DIR}")
        print("   Create the folder and add checkerboard images (01.jpg, 02.jpg, ...)")
        sys.exit(1)

    if len(left_files) != len(right_files):
        print(f"❌ Mismatch: {len(left_files)} left vs {len(right_files)} right images")
        sys.exit(1)

    print(f"Found {len(left_files)} image pairs")

    obj_points = []   # 3D points (same for every valid pair)
    img_points_L = [] # 2D corners in left images
    img_points_R = [] # 2D corners in right images
    img_size = None

    for i, (lf, rf) in enumerate(zip(left_files, right_files)):
        img_L = cv2.imread(lf)
        img_R = cv2.imread(rf)
        if img_L is None or img_R is None:
            print(f"  [{i+1}] ⚠️  Skipping (failed to read): {os.path.basename(lf)}")
            continue

        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = (gray_L.shape[1], gray_L.shape[0])  # (width, height)

        found_L, corners_L = find_corners(gray_L, board_size)
        found_R, corners_R = find_corners(gray_R, board_size)

        if found_L and found_R:
            obj_points.append(objp)
            img_points_L.append(corners_L)
            img_points_R.append(corners_R)
            print(f"  [{i+1}] ✅ {os.path.basename(lf)}")
        else:
            status = "L" if not found_L else ""
            status += "R" if not found_R else ""
            print(f"  [{i+1}] ❌ Corners not found ({status}): {os.path.basename(lf)}")

    valid = len(obj_points)
    print(f"\nValid pairs: {valid}/{len(left_files)}")

    if valid < MIN_PAIRS:
        print(f"❌ Need at least {MIN_PAIRS} valid pairs. Got {valid}.")
        print("   Tips: vary angle, distance, position. Avoid motion blur.")
        sys.exit(1)

    # ── Step 1: Individual camera calibration ──────────────────
    print("\n🔧 Calibrating left camera...")
    ret_L, K1, D1, _, _ = cv2.calibrateCamera(
        obj_points, img_points_L, img_size, None, None
    )
    print(f"   Reprojection error: {ret_L:.4f} px")

    print("🔧 Calibrating right camera...")
    ret_R, K2, D2, _, _ = cv2.calibrateCamera(
        obj_points, img_points_R, img_size, None, None
    )
    print(f"   Reprojection error: {ret_R:.4f} px")

    # ── Step 2: Stereo calibration ─────────────────────────────
    print("\n🔧 Stereo calibration...")
    flags = (
        cv2.CALIB_FIX_INTRINSIC  # use the intrinsics we already found
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_L, img_points_R,
        K1, D1, K2, D2, img_size,
        criteria=criteria, flags=flags
    )
    print(f"   Stereo reprojection error: {ret_stereo:.4f} px")

    baseline_mm = np.linalg.norm(T)
    print(f"   Baseline: {baseline_mm:.1f} mm")

    if ret_stereo > 1.0:
        print("⚠️  Warning: reprojection error > 1.0 px. Consider re-capturing images.")

    # ── Step 3: Stereo rectification ───────────────────────────
    print("🔧 Computing rectification maps...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T,
        alpha=0,  # crop black borders
        flags=cv2.CALIB_ZERO_DISPARITY
    )

    # Compute remap lookup tables (these are what the Android app needs)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    # ── Step 4: Save everything to JSON ────────────────────────
    def mat_to_list(m):
        """Convert numpy array to nested Python list for JSON."""
        return m.tolist()

    calib_data = {
        "image_size": list(img_size),  # [width, height]
        "square_size_mm": SQUARE_SIZE_MM,
        "board_size": [INNER_COLS, INNER_ROWS],
        "num_valid_pairs": valid,
        "stereo_reprojection_error": float(ret_stereo),
        "baseline_mm": float(baseline_mm),

        # Intrinsics
        "K1": mat_to_list(K1),  # 3x3 left camera matrix
        "D1": mat_to_list(D1),  # left distortion coefficients
        "K2": mat_to_list(K2),  # 3x3 right camera matrix
        "D2": mat_to_list(D2),  # right distortion coefficients

        # Extrinsics
        "R": mat_to_list(R),    # 3x3 rotation between cameras
        "T": mat_to_list(T),    # 3x1 translation between cameras

        # Rectification
        "R1": mat_to_list(R1),  # 3x3 left rectification rotation
        "R2": mat_to_list(R2),  # 3x3 right rectification rotation
        "P1": mat_to_list(P1),  # 3x4 left projection matrix
        "P2": mat_to_list(P2),  # 3x4 right projection matrix
        "Q": mat_to_list(Q),    # 4x4 disparity-to-depth mapping matrix

        # Remap lookup tables (large, but needed for fast runtime rectification)
        "map1x": mat_to_list(map1x),
        "map1y": mat_to_list(map1y),
        "map2x": mat_to_list(map2x),
        "map2y": mat_to_list(map2y),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(calib_data, f)

    file_size_mb = os.path.getsize(OUTPUT_JSON) / (1024 * 1024)
    print(f"\n✅ Saved: {OUTPUT_JSON} ({file_size_mb:.1f} MB)")
    print(f"\n📋 Next steps:")
    print(f"   1. Copy stereo_calib.json → app/src/main/assets/stereo_calib.json")
    print(f"   2. Rebuild the Android app")
    print(f"   3. The stereo depth anti-spoofing will activate automatically")

    # ── Optional: Quick visual check ───────────────────────────
    if valid > 0:
        print(f"\n🖼️  Generating rectification preview...")
        sample_L = cv2.imread(left_files[0])
        sample_R = cv2.imread(right_files[0])
        rect_L = cv2.remap(sample_L, map1x, map1y, cv2.INTER_LINEAR)
        rect_R = cv2.remap(sample_R, map2x, map2y, cv2.INTER_LINEAR)

        # Draw horizontal lines to verify alignment
        combo = np.hstack([rect_L, rect_R])
        for y in range(0, combo.shape[0], 40):
            cv2.line(combo, (0, y), (combo.shape[1], y), (0, 255, 0), 1)

        preview_path = os.path.join(SCRIPT_DIR, "rectification_preview.jpg")
        cv2.imwrite(preview_path, combo)
        print(f"   Saved: {preview_path}")
        print(f"   Check that horizontal green lines pass through matching features")


if __name__ == "__main__":
    main()

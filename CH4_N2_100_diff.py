#!/usr/bin/env python3

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/CH4_N2_diff/20260123_095342_CH4_CH4_N2_diff_1_4"
OUTPUT_EXCEL = "CH4_N2_100_diff_results.xlsx"

SAVE_PNG = True
PNG_DPI = 200

# -------- AXIAL ROI 1 (SIGNAL) --------
CIRC1_CENTER_X_MM = 24.0
CIRC1_CENTER_Y_MM = 17.5
CIRC1_RADIUS_MM   = 5.0

# -------- AXIAL ROI 2 (COMPARISON / BACKGROUND) --------
CIRC2_CENTER_X_MM = 71.0
CIRC2_CENTER_Y_MM = 18.0
CIRC2_RADIUS_MM   = 5.0

# -------- SAGITTAL ROI 1 --------
RECT1_TOP_MM = 17.0
RECT1_HEIGHT_MM = 50.0
RECT1_WIDTH_MM = 7.0
RECT1_CENTER_X_MM = 14.0

# -------- SAGITTAL ROI 2 --------
RECT2_TOP_MM = 17.0
RECT2_HEIGHT_MM = 50.0
RECT2_WIDTH_MM = 7.0
RECT2_CENTER_X_MM = 59.0

# =====================================================
# HJELPEFUNKSJONER
# =====================================================

def get_scan_time(ds):
    for tag in ["AcquisitionTime", "SeriesTime", "StudyTime"]:
        t = getattr(ds, tag, None)
        if t:
            t = str(t).split(".")[0]
            if len(t) >= 6:
                hh = int(t[0:2])
                mm = int(t[2:4])
                ss = int(t[4:6])
                return hh * 3600 + mm * 60 + ss, f"{hh:02d}:{mm:02d}:{ss:02d}"
    return None, "Unknown"


def rect_bbox_from_mm(shape, pixel_spacing, top_mm, height_mm, width_mm, centre_x_mm):
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    y0 = int(np.floor(top_mm / dy_mm))
    y1 = int(np.ceil((top_mm + height_mm) / dy_mm))

    centre_x_p = centre_x_mm / dx_mm
    half_w_p = (width_mm / dx_mm) / 2.0

    x0 = int(np.floor(centre_x_p - half_w_p))
    x1 = int(np.ceil(centre_x_p + half_w_p))

    y0 = max(0, y0)
    y1 = min(rows, y1)
    x0 = max(0, x0)
    x1 = min(cols, x1)

    return (x0, y0, x1, y1)


def circular_mask_from_mm(shape, pixel_spacing, center_x_mm, center_y_mm, radius_mm):
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    center_x_px = center_x_mm / dx_mm
    center_y_px = center_y_mm / dy_mm
    radius_x_px = radius_mm / dx_mm
    radius_y_px = radius_mm / dy_mm

    yy, xx = np.ogrid[:rows, :cols]

    mask = (
        ((xx - center_x_px) / radius_x_px) ** 2 +
        ((yy - center_y_px) / radius_y_px) ** 2
    ) <= 1

    return mask


def normalise_for_display(img):
    img = np.asarray(img, dtype=float)
    denom = img.max() - img.min()
    if denom == 0:
        return np.zeros_like(img)
    return (img - img.min()) / denom


def save_png_with_rois(img, roi1_info, roi2_info, path, title):
    fig, ax = plt.subplots()
    ax.imshow(normalise_for_display(img), cmap="gray")

    if roi1_info["shape"] == "rect":
        bbox1 = roi1_info["bbox"]
        rect1 = plt.Rectangle(
            (bbox1[0], bbox1[1]),
            bbox1[2] - bbox1[0],
            bbox1[3] - bbox1[1],
            fill=False,
            linewidth=2,
            edgecolor="red"
        )
        ax.add_patch(rect1)

    elif roi1_info["shape"] == "circle":
        circ1 = plt.Circle(
            (roi1_info["center_x_px"], roi1_info["center_y_px"]),
            roi1_info["radius_px"],
            fill=False,
            linewidth=2,
            edgecolor="red"
        )
        ax.add_patch(circ1)

    if roi2_info["shape"] == "rect":
        bbox2 = roi2_info["bbox"]
        rect2 = plt.Rectangle(
            (bbox2[0], bbox2[1]),
            bbox2[2] - bbox2[0],
            bbox2[3] - bbox2[1],
            fill=False,
            linewidth=2,
            edgecolor="blue"
        )
        ax.add_patch(rect2)

    elif roi2_info["shape"] == "circle":
        circ2 = plt.Circle(
            (roi2_info["center_x_px"], roi2_info["center_y_px"]),
            roi2_info["radius_px"],
            fill=False,
            linewidth=2,
            edgecolor="blue"
        )
        ax.add_patch(circ2)

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)

# =====================================================
# SCAN -> SLICE
# =====================================================

def choose_slice(scan_no):
    scan_map = {
        7: 7,
        8: 35,
        9: 35,
        10: 35,
        11: 35,
        12: 35,
        13: 35,
        14: None,
        15: 7,
        16: 35,
        17: 35,
        18: 35,
        19: 35,
        20: 35,
        21: 7,
        22: 35,
        23: 35,
        24: 35,
        25: 35,
        26: 35,
        27: 7,
        28: 35,
        29: 35,
        30: 35,
        31: 35,
        32: 35,
        33: 7,
        34: 35,
        35: 35,
        36: 35,
        37: 35,
        38: 35,
        39: 7,
        40: 35,
        41: 35,
        42: 35,
        43: 35,
        44: 35,
        45: 7,
        46: 35,
        47: 35,
        48: 35,
        49: 35,
        50: 35,
        51: 7,
        52: 35,
        53: 35,
        54: 35,
        55: 35,
        56: 35,
        57: 7,
        58: 35,
        59: 35,
        60: 35,
        61: 35,
        62: 35,
        63: 7,
        64: 35,
        65: 35,
        66: 35,
        67: 35,
        68: 35,
        69: 7,
        70: 35,
        71: 35,
        72: 35,
        73: 35,
        74: 35,
        75: 7,
        76: 35,
        77: 35,
        78: 35,
        79: 35,
        80: 35,
        81: 7,
        82: 35,
        83: 35,
        84: 35,
        85: 35,
        86: 35,
        87: 7,
        88: 35,
        89: 35,
        90: 35,
        91: 35,
        92: 35,
        93: 7,
        94: 35,
        95: 35,
        96: 35,
        97: 35,
        98: 35,
        99: 7,
        100: 35,
        101: 35,
        102: 35,
        103: 35,
        104: 35,
        105: 7,
        106: 35,
        107: 35,
        108: 35,
        109: 35,
        110: 35,
        111: 7,
        112: 35,
        113: 35,
        114: 35,
        115: 35,
        116: 35,
        117: 7,
        118: 35,
        119: 35,
        120: 35,
        121: 35,
        122: 35,
        123: 7,
        124: 35,
        125: 35,
        126: 35,
        127: 35,
        128: 35,
        129: 7,
        130: 35,
        131: 35,
        132: 35,
        133: 35,
        134: 35,
        135: 7,
        136: 35,
        137: 35,
        138: 35,
        139: 35,
        140: 35,
        141: 7,
        142: 35,
        143: 35,
        144: 35,
        145: 35,
        146: 35,
        147: 7,
        148: 35,
        149: 35,
        150: 35,
        151: 35,
        152: 35,
        153: 7,
        154: 35,
        155: 35,
        156: 35,
        157: 35,
        158: 35,
        159: 7,
        160: 35,
        161: 7,
        162: 35,
        163: 7,
        164: 35,
        165: 7,
        166: 35,
        167: 7,
        168: 35,
        169: 7,
        170: 35,
        171: 7,
    }
    return scan_map.get(scan_no, None)


def choose_roi_parameters(slice_no):
    if slice_no == 35:
        return {
            "roi1": {
                "shape": "circle",
                "center_x_mm": CIRC1_CENTER_X_MM,
                "center_y_mm": CIRC1_CENTER_Y_MM,
                "radius_mm": CIRC1_RADIUS_MM
            },
            "roi2": {
                "shape": "circle",
                "center_x_mm": CIRC2_CENTER_X_MM,
                "center_y_mm": CIRC2_CENTER_Y_MM,
                "radius_mm": CIRC2_RADIUS_MM
            }
        }

    elif slice_no == 7:
        return {
            "roi1": {
                "shape": "rect",
                "top_mm": RECT1_TOP_MM,
                "height_mm": RECT1_HEIGHT_MM,
                "width_mm": RECT1_WIDTH_MM,
                "center_x_mm": RECT1_CENTER_X_MM
            },
            "roi2": {
                "shape": "rect",
                "top_mm": RECT2_TOP_MM,
                "height_mm": RECT2_HEIGHT_MM,
                "width_mm": RECT2_WIDTH_MM,
                "center_x_mm": RECT2_CENTER_X_MM
            }
        }

    return None

# =====================================================
# MAIN
# =====================================================

def main():
    results = []

    first_time = None
    previous_time = None
    day_offset = 0
    SECONDS_PER_DAY = 86400

    script_dir = os.path.dirname(os.path.abspath(__file__))

    scan_folders = sorted(
        [int(f) for f in os.listdir(BASE_FOLDER) if f.isdigit() and 7 <= int(f) <= 171]
    )

    print("Fant scan-mapper:", scan_folders)

    for scan_no in scan_folders:
        target_slice = choose_slice(scan_no)

        if target_slice is None:
            print(f"[SCAN {scan_no}] skipped")
            continue

        roi_params = choose_roi_parameters(target_slice)
        if roi_params is None:
            print(f"[SCAN {scan_no}] no ROI parameters for slice {target_slice}")
            continue

        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        dicom_file = os.path.join(dicom_folder, f"MRIm{target_slice:02d}.dcm")

        if not os.path.exists(dicom_file):
            print(f"[SCAN {scan_no}] file missing: {dicom_file}")
            continue

        print(f"[SCAN {scan_no}] using slice {target_slice}")

        ds = pydicom.dcmread(dicom_file)

        img = ds.pixel_array.astype(float)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        img = img * slope + intercept

        pixel_spacing = getattr(ds, "PixelSpacing", [1, 1])

        # ROI 1
        if roi_params["roi1"]["shape"] == "circle":
            mask1 = circular_mask_from_mm(
                img.shape,
                pixel_spacing,
                roi_params["roi1"]["center_x_mm"],
                roi_params["roi1"]["center_y_mm"],
                roi_params["roi1"]["radius_mm"]
            )
            roi1_values = img[mask1]

            dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
            roi1_info = {
                "shape": "circle",
                "center_x_px": roi_params["roi1"]["center_x_mm"] / dx_mm,
                "center_y_px": roi_params["roi1"]["center_y_mm"] / dy_mm,
                "radius_px": roi_params["roi1"]["radius_mm"] / dx_mm
            }

        else:
            bbox1 = rect_bbox_from_mm(
                img.shape,
                pixel_spacing,
                roi_params["roi1"]["top_mm"],
                roi_params["roi1"]["height_mm"],
                roi_params["roi1"]["width_mm"],
                roi_params["roi1"]["center_x_mm"]
            )
            roi1_values = img[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
            roi1_info = {"shape": "rect", "bbox": bbox1}

        # ROI 2
        if roi_params["roi2"]["shape"] == "circle":
            mask2 = circular_mask_from_mm(
                img.shape,
                pixel_spacing,
                roi_params["roi2"]["center_x_mm"],
                roi_params["roi2"]["center_y_mm"],
                roi_params["roi2"]["radius_mm"]
            )
            roi2_values = img[mask2]

            dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
            roi2_info = {
                "shape": "circle",
                "center_x_px": roi_params["roi2"]["center_x_mm"] / dx_mm,
                "center_y_px": roi_params["roi2"]["center_y_mm"] / dy_mm,
                "radius_px": roi_params["roi2"]["radius_mm"] / dx_mm
            }

        else:
            bbox2 = rect_bbox_from_mm(
                img.shape,
                pixel_spacing,
                roi_params["roi2"]["top_mm"],
                roi_params["roi2"]["height_mm"],
                roi_params["roi2"]["width_mm"],
                roi_params["roi2"]["center_x_mm"]
            )
            roi2_values = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
            roi2_info = {"shape": "rect", "bbox": bbox2}

        if roi1_values.size == 0:
            print(f"[SCAN {scan_no}] ROI1 empty")
            continue

        if roi2_values.size == 0:
            print(f"[SCAN {scan_no}] ROI2 empty")
            continue

        roi1_mean = float(np.mean(roi1_values))
        roi1_median = float(np.median(roi1_values))
        roi1_std = float(np.std(roi1_values))

        roi2_mean = float(np.mean(roi2_values))
        roi2_median = float(np.median(roi2_values))
        roi2_std = float(np.std(roi2_values))

        time_sec, time_str = get_scan_time(ds)
        rel_time = None

        if time_sec is not None:
            if previous_time is not None and time_sec < previous_time:
                day_offset += SECONDS_PER_DAY

            absolute_time = time_sec + day_offset

            if first_time is None:
                first_time = absolute_time

            rel_time = absolute_time - first_time
            previous_time = time_sec

        results.append({
            "Scan": scan_no,
            "Slice": target_slice,
            "Time": time_str,
            "Time_seconds": rel_time,

            "ROI1_mean": roi1_mean,
            "ROI1_median": roi1_median,
            "ROI1_std": roi1_std,

            "ROI2_mean": roi2_mean,
            "ROI2_median": roi2_median,
            "ROI2_std": roi2_std,

            "EchoTime": getattr(ds, "EchoTime", None),
            "RepetitionTime": getattr(ds, "RepetitionTime", None),
            "FlipAngle": getattr(ds, "FlipAngle", None),

            "PixelSpacingY": pixel_spacing[0],
            "PixelSpacingX": pixel_spacing[1],
            "SliceThickness": getattr(ds, "SliceThickness", None),
            "Rows": getattr(ds, "Rows", None),
            "Columns": getattr(ds, "Columns", None)
        })

        if SAVE_PNG:
            out_png = os.path.join(script_dir, f"scan{scan_no}_ROI.png")
            title = f"Scan {scan_no} – {time_str}"
            save_png_with_rois(img, roi1_info, roi2_info, out_png, title)

    df = pd.DataFrame(results)

    if df.empty:
        print("Ingen resultater ble samlet inn.")
        return

    df["Time_minutes"] = df["Time_seconds"] / 60

    excel_path = os.path.join(script_dir, OUTPUT_EXCEL)
    df.to_excel(excel_path, index=False)

    print("\nExcel lagret:", excel_path)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
 
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
 
# =====================================================
# KONFIGURASJON
# =====================================================
 
BASE_FOLDER = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/20260209_130301_CH4_CH4_N2_120_diff45V_1_7"
 
OUTPUT_EXCEL = "diffusion_signal_results.xlsx"
 
# ROI 1
RECT1_TOP_MM = 13.0
RECT1_HEIGHT_MM = 89.0
RECT1_WIDTH_MM = 8.0
RECT1_CENTER_X_MM = 18.5
 
# ROI 2
RECT2_TOP_MM = 13.0
RECT2_HEIGHT_MM = 89.0
RECT2_WIDTH_MM = 8.0
RECT2_CENTER_X_MM = 60.0
 
SAVE_PNG = True
PNG_DPI = 200
 
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
 
 
# -----------------------------------------------------
 
 
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
 
 
# -----------------------------------------------------
 
 
def normalise_for_display(img):
 
    img = np.asarray(img, dtype=float)
 
    denom = img.max() - img.min()
 
    if denom == 0:
        return np.zeros_like(img)
 
    return (img - img.min()) / denom
 
 
# -----------------------------------------------------
 
 
def save_png_with_rects(img, bbox1, bbox2, path, title):
 
    fig, ax = plt.subplots()
 
    ax.imshow(normalise_for_display(img), cmap="gray")
 
    rect1 = plt.Rectangle((bbox1[0], bbox1[1]),
                          bbox1[2] - bbox1[0],
                          bbox1[3] - bbox1[1],
                          fill=False,
                          linewidth=2,
                          edgecolor="red")
 
    rect2 = plt.Rectangle((bbox2[0], bbox2[1]),
                          bbox2[2] - bbox2[0],
                          bbox2[3] - bbox2[1],
                          fill=False,
                          linewidth=2,
                          edgecolor="blue")
 
    ax.add_patch(rect1)
    ax.add_patch(rect2)
 
    ax.set_title(title)
    ax.axis("off")
 
    plt.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)
 
 
# =====================================================
# MAIN
# =====================================================
 
 
def main():
 
    results = []
 
    first_time = None
 
    scan_folders = sorted([
        int(f) for f in os.listdir(BASE_FOLDER)
        if f.isdigit()
    ])
 
    for scan_no in scan_folders:
 
        if 10 <= scan_no <= 262:
            target_slice = 8
        elif 266 <= scan_no <= 274:
            target_slice = 6
        else:
            print(f"[SCAN {scan_no}] skipped")
            continue
 
        dicom_folder = os.path.join(
            BASE_FOLDER,
            str(scan_no),
            "pdata",
            "1",
            "dicom"
        )
 
        dicom_file = os.path.join(
            dicom_folder,
            f"MRIm{target_slice:02d}.dcm"
        )
 
        if not os.path.exists(dicom_file):
 
            print(f"[SCAN {scan_no}] file missing:", dicom_file)
            continue
 
        print(f"[SCAN {scan_no}] using slice {target_slice}")
 
        ds = pydicom.dcmread(dicom_file)
 
        img = ds.pixel_array.astype(float)
 
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
 
        img = img * slope + intercept
 
        pixel_spacing = getattr(ds, "PixelSpacing", [1, 1])
 
        bbox1 = rect_bbox_from_mm(
            img.shape,
            pixel_spacing,
            RECT1_TOP_MM,
            RECT1_HEIGHT_MM,
            RECT1_WIDTH_MM,
            RECT1_CENTER_X_MM
        )
 
        bbox2 = rect_bbox_from_mm(
            img.shape,
            pixel_spacing,
            RECT2_TOP_MM,
            RECT2_HEIGHT_MM,
            RECT2_WIDTH_MM,
            RECT2_CENTER_X_MM
        )
 
        roi1 = img[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
        roi2 = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
 
        roi1_mean = float(np.mean(roi1))
        roi1_median = float(np.median(roi1))
        roi1_std = float(np.std(roi1))
 
        roi2_mean = float(np.mean(roi2))
        roi2_median = float(np.median(roi2))
        roi2_std = float(np.std(roi2))
 
        time_sec, time_str = get_scan_time(ds)
 
        if first_time is None and time_sec is not None:
            first_time = time_sec
 
        rel_time = None
        if first_time is not None and time_sec is not None:
            rel_time = time_sec - first_time
 
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
 
            out_png = os.path.join(
                dicom_folder,
                f"scan{scan_no}_ROI.png"
            )
 
            title = f"Scan {scan_no} – {time_str}"
 
            save_png_with_rects(img, bbox1, bbox2, out_png, title)
 
    df = pd.DataFrame(results)
 
    excel_path = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
 
    df.to_excel(excel_path, index=False)
 
    print("\nExcel lagret:", excel_path)
 
 
if __name__ == "__main__":
    main()
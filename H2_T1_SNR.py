#!/usr/bin/env python3

"""
H2_T1_ax_with_SNR.py

T1 analysis for H2 axial measurements, with additional SNR calculation.

SNR is calculated from:
    SNR_simple    = mean(signal ROI) / std(noise ROI)
    SNR_corrected = 0.655 * mean(signal ROI) / std(noise ROI)

The signal ROI is the existing circular ROI in the sample.
The noise ROI is a rectangular ROI placed in the image background.

For T1, the signal is lowest at short TR and highest at long TR.
Therefore, the script saves:
    - SNR for all TR values in separate Excel sheets
    - one representative SNR value per scan, taken at the TR with the highest signal
"""

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

# =====================================================
# CONFIGURATION
# =====================================================

BASE_FOLDER = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/H2"

SCAN_NUMBERS = [23, 30, 35, 40, 44, 48, 52, 56, 59, 62, 66]

# Pressure mapping for H2 T1 axial scans
PRESSURE_BAR = {
    23: 105.0,
    30: 98.9,
    35: 90.1,
    40: 80.1,
    44: 69.6,
    48: 59.9,
    52: 49.6,
    56: 40.6,
    59: 30.2,
    62: 20.1,
    66: 10.0,
}

# Signal ROI in mm
CIRCLE_CENTERS_MM = [(24.5, 22)]
CIRCLE_RADIUS_MM = 4.5

# Noise ROI in mm.
# Move this if the rectangle overlaps with the sample, cell, ghosting, or artefacts.
NOISE_RECT_LEFT_MM = 2.0
NOISE_RECT_TOP_MM = 2.0
NOISE_RECT_WIDTH_MM = 8.0
NOISE_RECT_HEIGHT_MM = 8.0

OUTPUT_EXCEL = "T1_H2_ax_scans_results_with_SNR.xlsx"

# =====================================================


def t1_model(tr, s0, t1, c):
    return s0 * (1.0 - np.exp(-tr / t1)) + c


def load_dicom_series(dicom_folder):
    files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]

    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_folder}")

    datasets = [pydicom.dcmread(f) for f in files]

    datasets.sort(
        key=lambda ds: getattr(ds, "RepetitionTime", getattr(ds, "InstanceNumber", 0))
    )

    images = []
    repetition_times = []

    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(
            getattr(ds, "RescaleIntercept", 0.0)
        )

        images.append(arr)
        repetition_times.append(float(getattr(ds, "RepetitionTime", np.nan)))

    return images, repetition_times, datasets[0]


def create_circle_mask_mm(shape, pixel_spacing, centers_mm, radius_mm):
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    Y, X = np.ogrid[:rows, :cols]
    mask_total = np.zeros(shape, dtype=bool)

    for (cx_mm, cy_mm) in centers_mm:
        cx_p = cx_mm / dx_mm
        cy_p = cy_mm / dy_mm
        radius_p = radius_mm / dx_mm

        dist2 = (X - cx_p) ** 2 + (Y - cy_p) ** 2
        mask_total |= dist2 <= radius_p**2

    return mask_total


def create_rectangle_mask_mm(shape, pixel_spacing, left_mm, top_mm, width_mm, height_mm):
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    left_px = int(round(left_mm / dx_mm))
    right_px = int(round((left_mm + width_mm) / dx_mm))
    top_px = int(round(top_mm / dy_mm))
    bottom_px = int(round((top_mm + height_mm) / dy_mm))

    left_px = max(0, min(cols, left_px))
    right_px = max(0, min(cols, right_px))
    top_px = max(0, min(rows, top_px))
    bottom_px = max(0, min(rows, bottom_px))

    mask = np.zeros(shape, dtype=bool)
    mask[top_px:bottom_px, left_px:right_px] = True

    return mask


def compute_snr(img, signal_mask, noise_mask):
    signal_mean = float(np.mean(img[signal_mask])) if np.any(signal_mask) else np.nan
    noise_mean = float(np.mean(img[noise_mask])) if np.any(noise_mask) else np.nan
    noise_std = (
        float(np.std(img[noise_mask], ddof=1)) if np.sum(noise_mask) > 1 else np.nan
    )

    if np.isfinite(noise_std) and noise_std > 0:
        snr_simple = signal_mean / noise_std
        snr_corrected = 0.655 * signal_mean / noise_std
    else:
        snr_simple = np.nan
        snr_corrected = np.nan

    return signal_mean, noise_mean, noise_std, snr_simple, snr_corrected


def fit_t1_and_r2(tr_list, signal_list):
    tr = np.array(tr_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    threshold = 0.05 * np.nanmax(sig)
    valid = np.isfinite(tr) & np.isfinite(sig) & (sig > threshold)

    tr = tr[valid]
    sig = sig[valid]

    if len(tr) < 3:
        raise RuntimeError("Too few data points to fit T1.")

    s0_init = sig.max()
    t1_init = (tr.max() - tr.min()) / 2 if tr.max() > tr.min() else 200.0

    popt, pcov = curve_fit(
        t1_model,
        tr,
        sig,
        p0=[s0_init, t1_init, 0],
        maxfev=10000,
    )

    s0, t1, c = popt
    s0_err, t1_err, c_err = np.sqrt(np.diag(pcov))

    fit_vals = t1_model(tr, s0, t1, c)
    residuals = sig - fit_vals

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t1, c, s0_err, t1_err, c_err, r2


def add_signal_circle_patch(ax, pixel_spacing):
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    for (cx_mm, cy_mm) in CIRCLE_CENTERS_MM:
        cx_p = cx_mm / dx_mm
        cy_p = cy_mm / dy_mm
        radius_p = CIRCLE_RADIUS_MM / dx_mm

        circle = plt.Circle(
            (cx_p, cy_p),
            radius_p,
            fill=False,
            linewidth=2,
            edgecolor="yellow",
            label="Signal ROI",
        )

        ax.add_patch(circle)


def add_noise_rect_patch(ax, pixel_spacing):
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    left_px = NOISE_RECT_LEFT_MM / dx_mm
    top_px = NOISE_RECT_TOP_MM / dy_mm
    width_px = NOISE_RECT_WIDTH_MM / dx_mm
    height_px = NOISE_RECT_HEIGHT_MM / dy_mm

    rect = Rectangle(
        (left_px, top_px),
        width_px,
        height_px,
        fill=False,
        linewidth=2,
        edgecolor="cyan",
        label="Noise ROI",
    )

    ax.add_patch(rect)


def main():
    t1_results_rows = []
    scan_signal_map = {}
    snr_simple_map = {}
    snr_corrected_map = {}
    noise_std_map = {}
    noise_mean_map = {}
    metadata_rows = []
    all_tr_values = set()

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        print(f"\n[SCAN {scan_no}] {dicom_folder}")

        images, repetition_times, ds0 = load_dicom_series(dicom_folder)
        img0 = images[0]

        pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])

        signal_mask = create_circle_mask_mm(
            img0.shape,
            pixel_spacing,
            CIRCLE_CENTERS_MM,
            CIRCLE_RADIUS_MM,
        )

        noise_mask = create_rectangle_mask_mm(
            img0.shape,
            pixel_spacing,
            NOISE_RECT_LEFT_MM,
            NOISE_RECT_TOP_MM,
            NOISE_RECT_WIDTH_MM,
            NOISE_RECT_HEIGHT_MM,
        )

        dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

        signal_roi_pixels = int(signal_mask.sum())
        signal_roi_area_mm2 = signal_roi_pixels * dx_mm * dy_mm

        noise_roi_pixels = int(noise_mask.sum())
        noise_roi_area_mm2 = noise_roi_pixels * dx_mm * dy_mm

        print(f"[SCAN {scan_no}] Signal ROI pixels: {signal_roi_pixels}")
        print(f"[SCAN {scan_no}] Signal ROI area: {signal_roi_area_mm2:.2f} mm^2")
        print(f"[SCAN {scan_no}] Noise ROI pixels: {noise_roi_pixels}")
        print(f"[SCAN {scan_no}] Noise ROI area: {noise_roi_area_mm2:.2f} mm^2")

        mean_signals = []
        tr_to_sig = {}
        tr_to_snr_simple = {}
        tr_to_snr_corrected = {}
        tr_to_noise_std = {}
        tr_to_noise_mean = {}
        snr_rows_this_scan = []

        for tr, img in zip(repetition_times, images):
            signal_mean, noise_mean, noise_std, snr_simple, snr_corrected = compute_snr(
                img,
                signal_mask,
                noise_mask,
            )

            tr_float = float(tr)

            mean_signals.append(signal_mean)
            tr_to_sig[tr_float] = signal_mean
            tr_to_snr_simple[tr_float] = snr_simple
            tr_to_snr_corrected[tr_float] = snr_corrected
            tr_to_noise_std[tr_float] = noise_std
            tr_to_noise_mean[tr_float] = noise_mean
            all_tr_values.add(tr_float)

            snr_rows_this_scan.append(
                {
                    "TR_ms": tr_float,
                    "Signal_mean": signal_mean,
                    "Noise_mean": noise_mean,
                    "Noise_std": noise_std,
                    "SNR_simple": snr_simple,
                    "SNR_corrected": snr_corrected,
                }
            )

            print(
                f"[SCAN {scan_no}] TR = {tr_float:.1f} ms | "
                f"signal = {signal_mean:.2f} | "
                f"noise_std = {noise_std:.2f} | "
                f"SNR_corrected = {snr_corrected:.2f}"
            )

        scan_signal_map[scan_no] = tr_to_sig
        snr_simple_map[scan_no] = tr_to_snr_simple
        snr_corrected_map[scan_no] = tr_to_snr_corrected
        noise_std_map[scan_no] = tr_to_noise_std
        noise_mean_map[scan_no] = tr_to_noise_mean

        try:
            s0, t1, c, s0_err, t1_err, c_err, r2 = fit_t1_and_r2(
                repetition_times,
                mean_signals,
            )
            print(f"T1 = {t1:.2f} ± {t1_err:.2f} ms | R² = {r2:.4f}")

        except Exception as e:
            print("FIT FAIL:", e)
            s0 = np.nan
            t1 = np.nan
            c = np.nan
            s0_err = np.nan
            t1_err = np.nan
            c_err = np.nan
            r2 = np.nan

        # Representative SNR value:
        # use the TR point with the highest signal for each scan.
        signal_values = np.array(
            [row["Signal_mean"] for row in snr_rows_this_scan],
            dtype=float,
        )

        if signal_values.size and np.any(np.isfinite(signal_values)):
            idx_max_signal = int(np.nanargmax(signal_values))
            max_signal_snr = snr_rows_this_scan[idx_max_signal]
        else:
            max_signal_snr = {}

        snr_simple_values = np.array(
            [row["SNR_simple"] for row in snr_rows_this_scan],
            dtype=float,
        )

        snr_corrected_values = np.array(
            [row["SNR_corrected"] for row in snr_rows_this_scan],
            dtype=float,
        )

        # Save PNG showing signal ROI and noise ROI
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")
        add_signal_circle_patch(ax, pixel_spacing)
        add_noise_rect_patch(ax, pixel_spacing)

        ax.set_title(f"Scan {scan_no} - signal ROI and noise ROI")
        ax.axis("off")
        ax.legend(loc="lower right")

        png_name = f"T1_H2_SNR_ROI_scan_{scan_no}.png"
        out_png = os.path.join(BASE_FOLDER, png_name)

        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        print(f"[SCAN {scan_no}] Saved ROI/SNR PNG: {out_png}")

        pressure = PRESSURE_BAR.get(scan_no, np.nan)

        t1_results_rows.append(
            {
                "Scan_no": scan_no,
                "Pressure_bar": pressure,
                "T1_ms": t1,
                "T1_error_ms": t1_err,
                "R2": r2,
                "S0": s0,
                "S0_error": s0_err,
                "c": c,
                "c_error": c_err,
                "TR_max_signal_ms": max_signal_snr.get("TR_ms", np.nan),
                "Signal_mean_max_signal_TR": max_signal_snr.get("Signal_mean", np.nan),
                "Noise_std_max_signal_TR": max_signal_snr.get("Noise_std", np.nan),
                "SNR_simple_max_signal_TR": max_signal_snr.get("SNR_simple", np.nan),
                "SNR_corrected_max_signal_TR": max_signal_snr.get(
                    "SNR_corrected",
                    np.nan,
                ),
                "SNR_simple_mean_all_TR": (
                    np.nanmean(snr_simple_values) if snr_simple_values.size else np.nan
                ),
                "SNR_corrected_mean_all_TR": (
                    np.nanmean(snr_corrected_values)
                    if snr_corrected_values.size
                    else np.nan
                ),
                "Signal_ROI_pixels": signal_roi_pixels,
                "Signal_ROI_area_mm2": signal_roi_area_mm2,
                "Noise_ROI_pixels": noise_roi_pixels,
                "Noise_ROI_area_mm2": noise_roi_area_mm2,
            }
        )

        valid_tr = [float(tr) for tr in repetition_times if np.isfinite(tr)]
        tr_list_str = ",".join(str(tr) for tr in valid_tr)

        metadata_rows.append(
            {
                "Scan_no": scan_no,
                "Pressure_bar": pressure,
                "RepetitionTime": getattr(ds0, "RepetitionTime", np.nan),
                "EchoTime": getattr(ds0, "EchoTime", np.nan),
                "FlipAngle": getattr(ds0, "FlipAngle", np.nan),
                "ImagingFrequency": getattr(ds0, "ImagingFrequency", np.nan),
                "NumberOfAverages": getattr(ds0, "NumberOfAverages", np.nan),
                "PixelSpacing": str(pixel_spacing),
                "SliceThickness": getattr(ds0, "SliceThickness", np.nan),
                "SpacingBetweenSlices": getattr(ds0, "SpacingBetweenSlices", np.nan),
                "Manufacturer": getattr(ds0, "Manufacturer", ""),
                "SeriesDescription": getattr(ds0, "SeriesDescription", ""),
                "ProtocolName": getattr(ds0, "ProtocolName", ""),
                "MagneticFieldStrength": getattr(ds0, "MagneticFieldStrength", np.nan),
                "TR_list_ms": tr_list_str,
                "Signal_ROI_centers_mm": str(CIRCLE_CENTERS_MM),
                "Signal_ROI_radius_mm": CIRCLE_RADIUS_MM,
                "Noise_ROI_left_mm": NOISE_RECT_LEFT_MM,
                "Noise_ROI_top_mm": NOISE_RECT_TOP_MM,
                "Noise_ROI_width_mm": NOISE_RECT_WIDTH_MM,
                "Noise_ROI_height_mm": NOISE_RECT_HEIGHT_MM,
            }
        )

    # =====================================================
    # Save all data points
    # =====================================================

    all_tr_sorted = sorted(all_tr_values)

    dp_raw = {"TR_ms": all_tr_sorted}
    dp_norm = {"TR_ms": all_tr_sorted}
    dp_snr_simple = {"TR_ms": all_tr_sorted}
    dp_snr_corrected = {"TR_ms": all_tr_sorted}
    dp_noise_std = {"TR_ms": all_tr_sorted}
    dp_noise_mean = {"TR_ms": all_tr_sorted}

    for scan_no, tr_to_sig in scan_signal_map.items():
        pressure = PRESSURE_BAR.get(scan_no, np.nan)
        column_name = f"Scan_{scan_no}_{pressure}_bar"

        raw_vals = [tr_to_sig.get(tr, np.nan) for tr in all_tr_sorted]
        dp_raw[column_name] = raw_vals

        raw_arr = np.array(raw_vals, dtype=float)

        if np.nanmax(raw_arr) > 0:
            norm_arr = raw_arr / np.nanmax(raw_arr)
        else:
            norm_arr = np.full_like(raw_arr, np.nan)

        dp_norm[f"{column_name}_norm"] = norm_arr

        dp_snr_simple[column_name] = [
            snr_simple_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted
        ]

        dp_snr_corrected[column_name] = [
            snr_corrected_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted
        ]

        dp_noise_std[column_name] = [
            noise_std_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted
        ]

        dp_noise_mean[column_name] = [
            noise_mean_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted
        ]

    df_results = pd.DataFrame(t1_results_rows)
    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)
    df_snr_simple = pd.DataFrame(dp_snr_simple)
    df_snr_corrected = pd.DataFrame(dp_snr_corrected)
    df_noise_std = pd.DataFrame(dp_noise_std)
    df_noise_mean = pd.DataFrame(dp_noise_mean)
    df_meta = pd.DataFrame(metadata_rows)

    # Sort result table by pressure from high to low
    df_results = df_results.sort_values("Pressure_bar", ascending=False)

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, index=False, sheet_name="T1_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")
        df_snr_simple.to_excel(writer, index=False, sheet_name="SNR_simple")
        df_snr_corrected.to_excel(writer, index=False, sheet_name="SNR_corrected")
        df_noise_std.to_excel(writer, index=False, sheet_name="Noise_std")
        df_noise_mean.to_excel(writer, index=False, sheet_name="Noise_mean")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
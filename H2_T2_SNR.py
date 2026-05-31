#!/usr/bin/env python3

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

SCAN_NUMBERS = [22, 29, 34, 39, 43, 47, 51, 55, 58, 61, 65]

# Signal ROI in mm
CIRCLE_CENTERS_MM = [(24, 22.5)]
CIRCLE_RADIUS_MM = 4.5

# Noise ROI in mm
# Move this box if it overlaps the sample, cell wall, ghosting, or visible artefacts.
NOISE_RECT_LEFT_MM = 2.0
NOISE_RECT_TOP_MM = 2.0
NOISE_RECT_WIDTH_MM = 8.0
NOISE_RECT_HEIGHT_MM = 8.0

OUTPUT_EXCEL = "T2_H2_ax_scans_results_with_SNR.xlsx"

SAVE_ROI_IMAGES = True
ROI_COLOR = "tab:green"
NOISE_ROI_COLOR = "cyan"

# Rician noise correction often used for magnitude MR images
SNR_CORRECTION_FACTOR = 0.655

# Signal threshold used for T2 fitting
FIT_THRESHOLD_FRACTION = 0.10

# =====================================================
# MODEL
# =====================================================

def t2_model(te, s0, t2, c):
    return s0 * np.exp(-te / t2) + c


# =====================================================
# LOAD DICOM
# =====================================================

def load_dicom_series(dicom_folder):
    files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]

    if not files:
        raise FileNotFoundError(f"No DICOM files found in: {dicom_folder}")

    datasets = [pydicom.dcmread(f) for f in files]

    datasets.sort(
        key=lambda ds: getattr(ds, "EchoTime", getattr(ds, "InstanceNumber", 0))
    )

    images = []
    echo_times = []

    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(
            getattr(ds, "RescaleIntercept", 0.0)
        )

        images.append(arr)
        echo_times.append(float(getattr(ds, "EchoTime", np.nan)))

    return images, echo_times, datasets[0]


# =====================================================
# ROI MASKS
# =====================================================

def create_circle_mask_mm(shape, pixel_spacing, centers_mm, radius_mm):
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    y_grid, x_grid = np.ogrid[:rows, :cols]
    mask_total = np.zeros(shape, dtype=bool)

    for cx_mm, cy_mm in centers_mm:
        cx_px = cx_mm / dx_mm
        cy_px = cy_mm / dy_mm
        radius_px = radius_mm / dx_mm

        dist2 = (x_grid - cx_px) ** 2 + (y_grid - cy_px) ** 2
        mask_total |= dist2 <= radius_px ** 2

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


# =====================================================
# SNR
# =====================================================

def compute_snr(img, signal_mask, noise_mask):
    signal_mean = float(np.mean(img[signal_mask])) if np.any(signal_mask) else np.nan
    noise_mean = float(np.mean(img[noise_mask])) if np.any(noise_mask) else np.nan
    noise_std = (
        float(np.std(img[noise_mask], ddof=1))
        if np.sum(noise_mask) > 1
        else np.nan
    )

    if np.isfinite(noise_std) and noise_std > 0:
        snr_simple = signal_mean / noise_std
        snr_corrected = SNR_CORRECTION_FACTOR * signal_mean / noise_std
    else:
        snr_simple = np.nan
        snr_corrected = np.nan

    return signal_mean, noise_mean, noise_std, snr_simple, snr_corrected


# =====================================================
# FITTING
# =====================================================

def fit_t2_and_r2(te_list, signal_list):
    te = np.array(te_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    threshold = FIT_THRESHOLD_FRACTION * np.nanmax(sig)
    valid = np.isfinite(te) & np.isfinite(sig) & (sig > threshold)

    te_fit = te[valid]
    sig_fit = sig[valid]

    if len(te_fit) < 3:
        raise RuntimeError("Too few data points to fit T2.")

    s0_init = sig_fit.max()
    t2_init = 10.0
    c_init = 0.0

    popt, pcov = curve_fit(
        t2_model,
        te_fit,
        sig_fit,
        p0=[s0_init, t2_init, c_init],
        maxfev=10000,
    )

    s0, t2, c = popt
    s0_err, t2_err, c_err = np.sqrt(np.diag(pcov))

    fit_vals = t2_model(te_fit, s0, t2, c)
    residuals = sig_fit - fit_vals

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig_fit - np.mean(sig_fit)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "S0": s0,
        "T2_ms": t2,
        "Offset_c": c,
        "S0_error": s0_err,
        "T2_error_ms": t2_err,
        "Offset_c_error": c_err,
        "R2": r2,
        "N_fit_points": len(te_fit),
        "First_TE_used_ms": np.nanmin(te_fit),
        "Last_TE_used_ms": np.nanmax(te_fit),
    }


# =====================================================
# ROI FIGURE
# =====================================================

def add_circle_patch(ax, pixel_spacing):
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    for cx_mm, cy_mm in CIRCLE_CENTERS_MM:
        cx_px = cx_mm / dx_mm
        cy_px = cy_mm / dy_mm
        radius_px = CIRCLE_RADIUS_MM / dx_mm

        circle = plt.Circle(
            (cx_px, cy_px),
            radius_px,
            color=ROI_COLOR,
            fill=False,
            linewidth=1.5,
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
        linewidth=1.5,
        edgecolor=NOISE_ROI_COLOR,
        label="Noise ROI",
    )
    ax.add_patch(rect)


# =====================================================
# MAIN
# =====================================================

def main():
    t2_results_rows = []
    metadata_rows = []

    scan_signal_map = {}
    snr_simple_map = {}
    snr_corrected_map = {}
    noise_std_map = {}
    noise_mean_map = {}

    all_te_values = set()

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        print(f"\n[SCAN {scan_no}] {dicom_folder}")

        images, echo_times, ds0 = load_dicom_series(dicom_folder)
        img0 = images[0]

        pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])
        dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

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

        overlap_px = int(np.sum(signal_mask & noise_mask))
        if overlap_px > 0:
            print(
                f"[SCAN {scan_no}] WARNING: Signal ROI and noise ROI overlap by "
                f"{overlap_px} pixels."
            )

        signal_roi_pixels = int(signal_mask.sum())
        noise_roi_pixels = int(noise_mask.sum())
        signal_roi_area_mm2 = signal_roi_pixels * dx_mm * dy_mm
        noise_roi_area_mm2 = noise_roi_pixels * dx_mm * dy_mm

        print(f"[SCAN {scan_no}] Signal ROI pixels: {signal_roi_pixels}")
        print(f"[SCAN {scan_no}] Signal ROI area: {signal_roi_area_mm2:.2f} mm²")
        print(f"[SCAN {scan_no}] Noise ROI pixels: {noise_roi_pixels}")
        print(f"[SCAN {scan_no}] Noise ROI area: {noise_roi_area_mm2:.2f} mm²")

        mean_signals = []
        te_to_sig = {}
        te_to_snr_simple = {}
        te_to_snr_corrected = {}
        te_to_noise_std = {}
        te_to_noise_mean = {}

        snr_rows_this_scan = []

        for te, img in zip(echo_times, images):
            (
                signal_mean,
                noise_mean,
                noise_std,
                snr_simple,
                snr_corrected,
            ) = compute_snr(img, signal_mask, noise_mask)

            te_float = float(te)

            mean_signals.append(signal_mean)
            te_to_sig[te_float] = signal_mean
            te_to_snr_simple[te_float] = snr_simple
            te_to_snr_corrected[te_float] = snr_corrected
            te_to_noise_std[te_float] = noise_std
            te_to_noise_mean[te_float] = noise_mean

            all_te_values.add(te_float)

            snr_rows_this_scan.append(
                {
                    "TE_ms": te_float,
                    "Signal_mean": signal_mean,
                    "Noise_mean": noise_mean,
                    "Noise_std": noise_std,
                    "SNR_simple": snr_simple,
                    "SNR_corrected": snr_corrected,
                }
            )

            print(
                f"[SCAN {scan_no}] TE={te_float:.2f} ms, "
                f"signal={signal_mean:.2f}, noise_std={noise_std:.2f}, "
                f"SNR={snr_simple:.2f}, SNRcorr={snr_corrected:.2f}"
            )

        scan_signal_map[scan_no] = te_to_sig
        snr_simple_map[scan_no] = te_to_snr_simple
        snr_corrected_map[scan_no] = te_to_snr_corrected
        noise_std_map[scan_no] = te_to_noise_std
        noise_mean_map[scan_no] = te_to_noise_mean

        try:
            fit_result = fit_t2_and_r2(echo_times, mean_signals)
            print(
                f"T2 = {fit_result['T2_ms']:.2f} ± "
                f"{fit_result['T2_error_ms']:.2f} ms | "
                f"R² = {fit_result['R2']:.3f}"
            )
        except Exception as e:
            print("FIT FAIL:", e)
            fit_result = {
                "S0": np.nan,
                "T2_ms": np.nan,
                "Offset_c": np.nan,
                "S0_error": np.nan,
                "T2_error_ms": np.nan,
                "Offset_c_error": np.nan,
                "R2": np.nan,
                "N_fit_points": 0,
                "First_TE_used_ms": np.nan,
                "Last_TE_used_ms": np.nan,
            }

        snr_df_this_scan = pd.DataFrame(snr_rows_this_scan).sort_values("TE_ms")

        first_te_row = (
            snr_df_this_scan.iloc[0].to_dict()
            if not snr_df_this_scan.empty
            else {}
        )

        snr_simple_values = snr_df_this_scan["SNR_simple"].to_numpy(dtype=float)
        snr_corrected_values = snr_df_this_scan["SNR_corrected"].to_numpy(dtype=float)

        t2_results_rows.append(
            {
                "Scan_no": scan_no,
                "T2_ms": fit_result["T2_ms"],
                "T2_error_ms": fit_result["T2_error_ms"],
                "R2": fit_result["R2"],
                "S0": fit_result["S0"],
                "S0_error": fit_result["S0_error"],
                "Offset_c": fit_result["Offset_c"],
                "Offset_c_error": fit_result["Offset_c_error"],
                "N_fit_points": fit_result["N_fit_points"],
                "First_TE_used_ms": fit_result["First_TE_used_ms"],
                "Last_TE_used_ms": fit_result["Last_TE_used_ms"],
                "Signal_mean_first_TE": first_te_row.get("Signal_mean", np.nan),
                "Noise_mean_first_TE": first_te_row.get("Noise_mean", np.nan),
                "Noise_std_first_TE": first_te_row.get("Noise_std", np.nan),
                "SNR_simple_first_TE": first_te_row.get("SNR_simple", np.nan),
                "SNR_corrected_first_TE": first_te_row.get("SNR_corrected", np.nan),
                "SNR_simple_mean_all_TE": np.nanmean(snr_simple_values)
                if snr_simple_values.size
                else np.nan,
                "SNR_corrected_mean_all_TE": np.nanmean(snr_corrected_values)
                if snr_corrected_values.size
                else np.nan,
                "Signal_ROI_pixels": signal_roi_pixels,
                "Signal_ROI_area_mm2": signal_roi_area_mm2,
                "Noise_ROI_pixels": noise_roi_pixels,
                "Noise_ROI_area_mm2": noise_roi_area_mm2,
            }
        )

        metadata_rows.append(
            {
                "Scan_no": scan_no,
                "EchoTime_first_ms": getattr(ds0, "EchoTime", np.nan),
                "RepetitionTime_ms": getattr(ds0, "RepetitionTime", np.nan),
                "FlipAngle": getattr(ds0, "FlipAngle", np.nan),
                "PixelSpacing": str(pixel_spacing),
                "SliceThickness": getattr(ds0, "SliceThickness", np.nan),
                "SpacingBetweenSlices": getattr(ds0, "SpacingBetweenSlices", np.nan),
                "Manufacturer": getattr(ds0, "Manufacturer", ""),
                "SeriesDescription": getattr(ds0, "SeriesDescription", ""),
                "ProtocolName": getattr(ds0, "ProtocolName", ""),
                "MagneticFieldStrength": getattr(ds0, "MagneticFieldStrength", np.nan),
                "Signal_ROI_centers_mm": str(CIRCLE_CENTERS_MM),
                "Signal_ROI_radius_mm": CIRCLE_RADIUS_MM,
                "Noise_ROI_left_mm": NOISE_RECT_LEFT_MM,
                "Noise_ROI_top_mm": NOISE_RECT_TOP_MM,
                "Noise_ROI_width_mm": NOISE_RECT_WIDTH_MM,
                "Noise_ROI_height_mm": NOISE_RECT_HEIGHT_MM,
            }
        )

        if SAVE_ROI_IMAGES:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img0, cmap="gray")
            add_circle_patch(ax, pixel_spacing)
            add_noise_rect_patch(ax, pixel_spacing)
            ax.set_title(f"Scan {scan_no} - signal ROI and noise ROI")
            ax.axis("off")
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=2,
                frameon=True,
            )

            png_path = os.path.join(BASE_FOLDER, f"T2_H2_ROI_SNR_scan_{scan_no}.png")
            plt.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"[SCAN {scan_no}] ROI/SNR image saved: {png_path}")

    # =====================================================
    # DATAPOINT TABLES
    # =====================================================

    all_te_sorted = sorted(all_te_values)

    dp_raw = {"TE_ms": all_te_sorted}
    dp_norm = {"TE_ms": all_te_sorted}
    dp_snr_simple = {"TE_ms": all_te_sorted}
    dp_snr_corrected = {"TE_ms": all_te_sorted}
    dp_noise_std = {"TE_ms": all_te_sorted}
    dp_noise_mean = {"TE_ms": all_te_sorted}

    for scan_no, te_to_sig in scan_signal_map.items():
        raw_vals = [te_to_sig.get(te, np.nan) for te in all_te_sorted]
        raw_arr = np.array(raw_vals, dtype=float)

        dp_raw[f"Scan_{scan_no}"] = raw_vals

        if np.nanmax(raw_arr) > 0:
            norm_arr = raw_arr / np.nanmax(raw_arr)
        else:
            norm_arr = np.full_like(raw_arr, np.nan)

        dp_norm[f"Scan_{scan_no}_norm"] = norm_arr

        dp_snr_simple[f"Scan_{scan_no}"] = [
            snr_simple_map[scan_no].get(te, np.nan) for te in all_te_sorted
        ]

        dp_snr_corrected[f"Scan_{scan_no}"] = [
            snr_corrected_map[scan_no].get(te, np.nan) for te in all_te_sorted
        ]

        dp_noise_std[f"Scan_{scan_no}"] = [
            noise_std_map[scan_no].get(te, np.nan) for te in all_te_sorted
        ]

        dp_noise_mean[f"Scan_{scan_no}"] = [
            noise_mean_map[scan_no].get(te, np.nan) for te in all_te_sorted
        ]

    df_results = pd.DataFrame(t2_results_rows)
    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)
    df_snr_simple = pd.DataFrame(dp_snr_simple)
    df_snr_corrected = pd.DataFrame(dp_snr_corrected)
    df_noise_std = pd.DataFrame(dp_noise_std)
    df_noise_mean = pd.DataFrame(dp_noise_mean)
    df_metadata = pd.DataFrame(metadata_rows)

    # =====================================================
    # SAVE EXCEL
    # =====================================================

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, index=False, sheet_name="T2_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")
        df_snr_simple.to_excel(writer, index=False, sheet_name="SNR_simple")
        df_snr_corrected.to_excel(writer, index=False, sheet_name="SNR_corrected")
        df_noise_std.to_excel(writer, index=False, sheet_name="Noise_std")
        df_noise_mean.to_excel(writer, index=False, sheet_name="Noise_mean")
        df_metadata.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
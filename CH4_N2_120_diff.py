#!/usr/bin/env python3

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/CH4_N2_120_diff/20260126_155118_CH4_CH4_N2_120_diff_1_5"

OUTPUT_EXCEL = "diffusion_signal_results_axial_circularROI.xlsx"

# -------- ROI 1 (SIGNAL) --------
CIRC1_CENTER_X_MM = 14.0
CIRC1_CENTER_Y_MM = 18.5
CIRC1_RADIUS_MM   = 5.0

# -------- ROI 2 (COMPARISON / BACKGROUND) --------
CIRC2_CENTER_X_MM = 60.0
CIRC2_CENTER_Y_MM = 21.0
CIRC2_RADIUS_MM   = 5.0

# -------- Output --------
SAVE_PNG = True
PNG_DPI = 200
SAVE_PLOTS = True

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


def load_dicom_series(dicom_folder):

    files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]

    if not files:
        raise FileNotFoundError(f"Ingen DICOM-filer i {dicom_folder}")

    datasets = [pydicom.dcmread(f) for f in files]

    def sort_key(ds, fpath):

        te = getattr(ds, "EchoTime", None)
        inst = getattr(ds, "InstanceNumber", None)

        te_key = float(te) if te is not None else 1e18
        inst_key = int(inst) if inst is not None else 1e18

        name_key = os.path.basename(fpath)

        return (te_key, inst_key, name_key)

    paired = list(zip(datasets, files))
    paired.sort(key=lambda x: sort_key(x[0], x[1]))

    datasets_sorted, files_sorted = zip(*paired)

    images = []

    for ds in datasets_sorted:

        img = ds.pixel_array.astype(float)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))

        img = img * slope + intercept

        images.append(img)

    return images, list(datasets_sorted), list(files_sorted)


# -----------------------------------------------------


def circle_mask_from_mm(shape, pixel_spacing, centre_x_mm, centre_y_mm, radius_mm):

    rows, cols = shape

    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    centre_x_p = centre_x_mm / dx_mm
    centre_y_p = centre_y_mm / dy_mm

    radius_x_p = radius_mm / dx_mm
    radius_y_p = radius_mm / dy_mm

    yy, xx = np.ogrid[:rows, :cols]

    mask = (((xx - centre_x_p) / radius_x_p) ** 2 +
            ((yy - centre_y_p) / radius_y_p) ** 2) <= 1.0

    return mask, centre_x_p, centre_y_p, radius_x_p, radius_y_p


# -----------------------------------------------------


def normalise_for_display(img):

    img = np.asarray(img, dtype=float)

    denom = img.max() - img.min()

    if denom == 0:
        return np.zeros_like(img)

    return (img - img.min()) / denom


# -----------------------------------------------------


def save_png_with_circles(img, circle1, circle2, path, title):

    cx1, cy1, rx1, ry1 = circle1
    cx2, cy2, rx2, ry2 = circle2

    fig, ax = plt.subplots()

    ax.imshow(normalise_for_display(img), cmap="gray")

    circ1 = plt.Circle((cx1, cy1), rx1,
                       fill=False, linewidth=2, edgecolor="red")

    circ2 = plt.Circle((cx2, cy2), rx2,
                       fill=False, linewidth=2, edgecolor="blue")

    ax.add_patch(circ1)
    ax.add_patch(circ2)

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

    scan_folders = sorted([
        int(f) for f in os.listdir(BASE_FOLDER)
        if f.isdigit()
    ])

    first_time = None
    previous_time = None
    day_offset = 0
    SECONDS_PER_DAY = 86400

    for scan_no in scan_folders:

        dicom_folder = os.path.join(
            BASE_FOLDER,
            str(scan_no),
            "pdata",
            "1",
            "dicom"
        )

        print(f"\n[SCAN {scan_no}] {dicom_folder}")

        if not os.path.isdir(dicom_folder):
            print("Fant ikke dicom mappe.")
            continue

        images, datasets, files_sorted = load_dicom_series(dicom_folder)

        n = len(images)

        if n == 0:
            continue

        # -------------------------------------------------
        # ALLTID BRUK FØRSTE SLICE
        # -------------------------------------------------

        idx = 0

        img = images[idx]
        ds = datasets[idx]

        chosen_file = os.path.basename(files_sorted[idx])

        pixel_spacing = getattr(ds, "PixelSpacing", [1, 1])

        print(f"Valgt slice: {idx} | fil: {chosen_file}")
        print("PixelSpacing:", pixel_spacing)

        rows, cols = img.shape
        dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

        print("Image shape (px):", img.shape)
        print(f"Image size (mm): width={cols * dx_mm:.2f}, height={rows * dy_mm:.2f}")

        # ROI 1
        mask1, cx1, cy1, rx1, ry1 = circle_mask_from_mm(
            img.shape,
            pixel_spacing,
            CIRC1_CENTER_X_MM,
            CIRC1_CENTER_Y_MM,
            CIRC1_RADIUS_MM
        )

        # ROI 2
        mask2, cx2, cy2, rx2, ry2 = circle_mask_from_mm(
            img.shape,
            pixel_spacing,
            CIRC2_CENTER_X_MM,
            CIRC2_CENTER_Y_MM,
            CIRC2_RADIUS_MM
        )

        roi1 = img[mask1]
        roi2 = img[mask2]

        if roi1.size == 0:
            print("ROI1 er tom. Sjekk sentrum/radius.")
            continue

        if roi2.size == 0:
            print("ROI2 er tom. Sjekk sentrum/radius.")
            continue

        roi1_mean = float(np.mean(roi1))
        roi1_median = float(np.median(roi1))
        roi1_std = float(np.std(roi1))

        roi2_mean = float(np.mean(roi2))
        roi2_median = float(np.median(roi2))
        roi2_std = float(np.std(roi2))

        print("Mean signal ROI1:", roi1_mean)
        print("Mean signal ROI2:", roi2_mean)

        time_sec, time_str = get_scan_time(ds)

        if time_sec is not None:

            if previous_time is not None and time_sec < previous_time:
                day_offset += SECONDS_PER_DAY

            absolute_time = time_sec + day_offset

            if first_time is None:
                first_time = absolute_time

            rel_time = absolute_time - first_time
            previous_time = time_sec

        else:
            rel_time = None

        results.append({
            "Scan": scan_no,
            "Slice_index": idx,
            "Chosen_file": chosen_file,
            "Time": time_str,
            "Time_seconds": rel_time,

            "ROI1_mean": roi1_mean,
            "ROI1_median": roi1_median,
            "ROI1_std": roi1_std,

            "ROI2_mean": roi2_mean,
            "ROI2_median": roi2_median,
            "ROI2_std": roi2_std
        })

        if SAVE_PNG:

            out_png = os.path.join(
                dicom_folder,
                f"scan{scan_no}_slice{idx}_twoCircularROI.png"
            )

            title = f"Scan {scan_no} – {time_str}"

            save_png_with_circles(
                img,
                (cx1, cy1, rx1, ry1),
                (cx2, cy2, rx2, ry2),
                out_png,
                title
            )

            print("Lagret PNG:", out_png)

    if not results:
        print("\nIngen resultater å lagre.")
        return

    df = pd.DataFrame(results)

    excel_path = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    df.to_excel(excel_path, index=False)

    print("\nExcel lagret:", excel_path)

    # =====================================================
    # PLOT
    # =====================================================

    df = df.dropna(subset=["Time_seconds", "ROI1_mean", "ROI2_mean"])
    df = df.sort_values("Time_seconds").reset_index(drop=True)

    if len(df) == 0:
        print("Ingen gyldige data å plotte.")
        return

    df["ROI1_norm"] = df["ROI1_mean"] / df["ROI1_mean"].iloc[0]
    df["ROI2_norm"] = df["ROI2_mean"] / df["ROI2_mean"].iloc[0]
    df["ROI_difference"] = df["ROI1_mean"] - df["ROI2_mean"]

    # -------- Plot 1: råsignal --------
    plt.figure(figsize=(8, 5))
    plt.plot(df["Time_seconds"], df["ROI1_mean"], "o-", label="ROI1")
    plt.plot(df["Time_seconds"], df["ROI2_mean"], "o-", label="ROI2")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("MRI signal vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if SAVE_PLOTS:
        plot1_path = os.path.join(BASE_FOLDER, "plot_signal_vs_time.png")
        plt.savefig(plot1_path, dpi=300)
        print("Plot lagret:", plot1_path)

    plt.show()

    # -------- Plot 2: normalisert signal --------
    plt.figure(figsize=(8, 5))
    plt.plot(df["Time_seconds"], df["ROI1_norm"], "o-", label="ROI1 normalised")
    plt.plot(df["Time_seconds"], df["ROI2_norm"], "o-", label="ROI2 normalised")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalised signal")
    plt.title("Normalised MRI signal vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if SAVE_PLOTS:
        plot2_path = os.path.join(BASE_FOLDER, "plot_normalised_signal_vs_time.png")
        plt.savefig(plot2_path, dpi=300)
        print("Plot lagret:", plot2_path)

    plt.show()

    # -------- Plot 3: differanse --------
    plt.figure(figsize=(8, 5))
    plt.plot(df["Time_seconds"], df["ROI_difference"], "o-")
    plt.xlabel("Time (s)")
    plt.ylabel("ROI1 - ROI2")
    plt.title("Signal difference vs time")
    plt.grid(True)
    plt.tight_layout()

    if SAVE_PLOTS:
        plot3_path = os.path.join(BASE_FOLDER, "plot_signal_difference_vs_time.png")
        plt.savefig(plot3_path, dpi=300)
        print("Plot lagret:", plot3_path)

    plt.show()


if __name__ == "__main__":
    main()
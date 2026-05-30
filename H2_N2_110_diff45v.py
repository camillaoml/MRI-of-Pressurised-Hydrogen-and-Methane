import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45V"

OUTPUT_EXCEL = "H2_N2_diffusion_results.xlsx"

SAVE_PNG = True
PNG_DPI = 200

# Samme ROI-bredde som de andre H2-script-ene
ROI_WIDTH_PX = 3

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


def rect_bbox_fixed_width_px(shape, pixel_spacing, top_mm, height_mm, width_px, centre_x_mm):
    rows, cols = shape

    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    y0_raw = int(np.floor(top_mm / dy_mm))
    y1_raw = int(np.ceil((top_mm + height_mm) / dy_mm))

    centre_x_px = int(round(centre_x_mm / dx_mm))
    half_w = width_px // 2

    if width_px % 2 == 0:
        x0_raw = centre_x_px - half_w
        x1_raw = centre_x_px + half_w
    else:
        x0_raw = centre_x_px - half_w
        x1_raw = centre_x_px + half_w + 1

    y0 = max(0, y0_raw)
    y1 = min(rows, y1_raw)
    x0 = max(0, x0_raw)
    x1 = min(cols, x1_raw)

    clipped = (
        x0 != x0_raw or
        x1 != x1_raw or
        y0 != y0_raw or
        y1 != y1_raw
    )

    return (x0, y0, x1, y1), clipped


def normalise_for_display(img):
    img = np.asarray(img, dtype=float)
    denom = img.max() - img.min()

    if denom == 0:
        return np.zeros_like(img)

    return (img - img.min()) / denom


def plot_rois(img, bbox1, bbox2, title, save_path=None, show=False):
    fig, ax = plt.subplots()

    ax.imshow(normalise_for_display(img), cmap="gray")

    rect1 = plt.Rectangle(
        (bbox1[0], bbox1[1]),
        bbox1[2] - bbox1[0],
        bbox1[3] - bbox1[1],
        fill=False,
        linewidth=2,
        edgecolor="red"
    )

    rect2 = plt.Rectangle(
        (bbox2[0], bbox2[1]),
        bbox2[2] - bbox2[0],
        bbox2[3] - bbox2[1],
        fill=False,
        linewidth=2,
        edgecolor="blue"
    )

    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=PNG_DPI)

    if show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================
# FUNKSJON SOM BESTEMMER SLICE
# =====================================================

def choose_slice(scan_no):
    if 16 <= scan_no <= 153:
        return 9

    elif 154 <= scan_no <= 168:
        return 13

    elif scan_no == 169:
        return 9

    elif scan_no == 170:
        return 13

    elif scan_no == 171:
        return None

    elif scan_no in [172, 173]:
        return 16

    elif scan_no == 174:
        return 11

    elif scan_no in [175, 176]:
        return 15

    else:
        return None


def choose_roi_parameters(slice_no):
    # Samme ROI-dimensjoner for alle:
    # 70 mm høyde og 3 px bredde

    if slice_no in [9, 11]:
        RECT1_TOP_MM = 20.0
        RECT1_HEIGHT_MM = 70.0
        RECT1_CENTER_X_MM = 14.5

        RECT2_TOP_MM = 20.0
        RECT2_HEIGHT_MM = 70.0
        RECT2_CENTER_X_MM = 53.0

    else:
        RECT1_TOP_MM = 20.0
        RECT1_HEIGHT_MM = 70.0
        RECT1_CENTER_X_MM = 12.0

        RECT2_TOP_MM = 20.0
        RECT2_HEIGHT_MM = 70.0
        RECT2_CENTER_X_MM = 51.0

    return (
        RECT1_TOP_MM, RECT1_HEIGHT_MM, RECT1_CENTER_X_MM,
        RECT2_TOP_MM, RECT2_HEIGHT_MM, RECT2_CENTER_X_MM
    )


# =====================================================
# MAIN
# =====================================================

def main():
    results = []

    first_time = None
    previous_time = None
    day_offset = 0
    SECONDS_PER_DAY = 86400

    last_img = None
    last_bbox1 = None
    last_bbox2 = None
    last_title = None

    scan_folders = sorted([
        int(f) for f in os.listdir(BASE_FOLDER)
        if f.isdigit()
    ])

    for scan_no in scan_folders:
        target_slice = choose_slice(scan_no)

        if target_slice is None:
            print(f"[SCAN {scan_no}] skipped")
            continue

        (
            RECT1_TOP_MM, RECT1_HEIGHT_MM, RECT1_CENTER_X_MM,
            RECT2_TOP_MM, RECT2_HEIGHT_MM, RECT2_CENTER_X_MM
        ) = choose_roi_parameters(target_slice)

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
            print(f"[SCAN {scan_no}] file missing")
            continue

        print(f"[SCAN {scan_no}] using slice {target_slice}")

        ds = pydicom.dcmread(dicom_file)

        img = ds.pixel_array.astype(float)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        img = img * slope + intercept

        pixel_spacing = getattr(ds, "PixelSpacing", [1, 1])

        bbox1, roi1_clipped = rect_bbox_fixed_width_px(
            img.shape,
            pixel_spacing,
            RECT1_TOP_MM,
            RECT1_HEIGHT_MM,
            ROI_WIDTH_PX,
            RECT1_CENTER_X_MM
        )

        bbox2, roi2_clipped = rect_bbox_fixed_width_px(
            img.shape,
            pixel_spacing,
            RECT2_TOP_MM,
            RECT2_HEIGHT_MM,
            ROI_WIDTH_PX,
            RECT2_CENTER_X_MM
        )

        roi1_width_px = bbox1[2] - bbox1[0]
        roi1_height_px = bbox1[3] - bbox1[1]

        roi2_width_px = bbox2[2] - bbox2[0]
        roi2_height_px = bbox2[3] - bbox2[1]

        print("PixelSpacing:", pixel_spacing)
        print("ROI1 size px:", roi1_width_px, "x", roi1_height_px)
        print("ROI2 size px:", roi2_width_px, "x", roi2_height_px)

        if roi1_clipped or roi2_clipped:
            print("WARNING: One ROI has been clipped by the image boundary.")

        roi1 = img[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
        roi2 = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]

        roi1_mean = float(np.mean(roi1))
        roi1_median = float(np.median(roi1))
        roi1_std = float(np.std(roi1))

        roi2_mean = float(np.mean(roi2))
        roi2_median = float(np.median(roi2))
        roi2_std = float(np.std(roi2))

        # Korriger for at systemet ble snudd
        if scan_no >= 170:
            roi1_mean, roi2_mean = roi2_mean, roi1_mean
            roi1_median, roi2_median = roi2_median, roi1_median
            roi1_std, roi2_std = roi2_std, roi1_std

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
            "Slice": target_slice,
            "Time": time_str,
            "Time_seconds": rel_time,

            "ROI1_mean": roi1_mean,
            "ROI1_median": roi1_median,
            "ROI1_std": roi1_std,

            "ROI2_mean": roi2_mean,
            "ROI2_median": roi2_median,
            "ROI2_std": roi2_std,

            "ROI1_width_px": roi1_width_px,
            "ROI1_height_px": roi1_height_px,
            "ROI1_clipped": roi1_clipped,

            "ROI2_width_px": roi2_width_px,
            "ROI2_height_px": roi2_height_px,
            "ROI2_clipped": roi2_clipped,

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

            title = f"Scan {scan_no} – {time_str} – slice {target_slice}"
            plot_rois(img, bbox1, bbox2, title, save_path=out_png, show=False)

        # Lagre siste gyldige ROI-bilde til visning etterpå
        last_img = img
        last_bbox1 = bbox1
        last_bbox2 = bbox2
        last_title = f"Last scan shown: scan {scan_no} – {time_str} – slice {target_slice}"

    df = pd.DataFrame(results)

    if df.empty:
        print("Ingen resultater ble samlet inn.")
        return

    df["Time_minutes"] = df["Time_seconds"] / 60

    excel_path = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    df.to_excel(excel_path, index=False)

    print("\nExcel lagret:", excel_path)

    # Vis siste ROI-bilde
    if last_img is not None:
        plot_rois(
            last_img,
            last_bbox1,
            last_bbox2,
            last_title,
            save_path=None,
            show=True
        )


if __name__ == "__main__":
    main()
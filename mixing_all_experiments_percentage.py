#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# KONFIGURASJON
# =====================================================

OUTPUT_FOLDER = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion"

OUTPUT_FIGURE_40H = os.path.join(
    OUTPUT_FOLDER,
    "all_mixing_degree_comparison_0_40h.png"
)

OUTPUT_FIGURE_FULL = os.path.join(
    OUTPUT_FOLDER,
    "all_mixing_degree_comparison_full_time.png"
)

# -----------------------------------------------------
# Excel-filer med mixing degree
# -----------------------------------------------------

MIXING_FILES = [
    {
        "label": r"H$_2$--N$_2$, horizontal",
        "path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_sidelengs/H2_N2_horisontal_mixing_percent.xlsx",
        "marker": "o",
        "linestyle": "--",
        "sheet_name": "Mixing_percent",
    },
    {
        "label": r"H$_2$--N$_2$, N$_2$ on top",
        "path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45_N2top/H2_N2_N2_on_top_mixing_percent.xlsx",
        "marker": "s",
        "linestyle": "--",
        "sheet_name": "Mixing_percent",
    },
    {
        "label": r"H$_2$--N$_2$, H$_2$ on top + inversion",
        "path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45V/H2_N2_H2_on_top_inversion_mixing_percent.xlsx",
        "marker": "^",
        "linestyle": "--",
        "sheet_name": "Mixing_percent",
    },
    {
        "label": r"CH$_4$--N$_2$, horizontal",
        "path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion/CH4_diff_experiments/CH4_N2_100_diff45_2_mixing_percent.xlsx",
        "marker": "D",
        "linestyle": "--",
        "sheet_name": "Sagittal",
    },
    {
        "label": r"CH$_4$--N$_2$, CH$_4$ on top + inversion I",
        "path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion/CH4_diff_experiments/CH4_N2_120_vertical_flipped_mixing_percent.xlsx",
        "marker": "v",
        "linestyle": "--",
        "sheet_name": "Vertical_flipped",
    },
    {
        "label": r"CH$_4$--N$_2$, CH$_4$ on top + inversion II",
        "path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion/CH4_N2_120_45V2_mixing_percent.xlsx",
        "marker": "v",
        "linestyle": "--",
        "sheet_name": "Mixing_percent",
    },
]


# =====================================================
# FIGURSTIL
# =====================================================

FIGSIZE = (10, 6.5)

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 13

LINEWIDTH = 1.3
MARKER_SIZE = 5.8
END_MARKER_SIZE = 9.0
MARKER_EVERY = 5

ALPHA = 0.9
GRID_ALPHA = 0.3

Y_LIMIT = (-5, 105)


# =====================================================
# HJELPEFUNKSJONER
# =====================================================

def read_mixing_excel(path, sheet_name_requested=None, x_limit=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fant ikke filen:\n{path}")

    xls = pd.ExcelFile(path)

    if sheet_name_requested is not None:
        if sheet_name_requested not in xls.sheet_names:
            raise ValueError(
                f"Fant ikke arket '{sheet_name_requested}' i filen:\n{path}\n"
                f"Ark som finnes: {xls.sheet_names}"
            )
        sheet_name = sheet_name_requested

    elif "Mixing_percent" in xls.sheet_names:
        sheet_name = "Mixing_percent"

    else:
        sheet_name = xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet_name)

    required_cols = ["Time_hours", "Mixing_degree_percent_raw"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Mangler kolonnen '{col}' i filen:\n{path}\n"
                f"Ark brukt: {sheet_name}\n"
                f"Kolonner funnet: {list(df.columns)}"
            )

    df = df[required_cols].copy()

    for col in required_cols:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols)
    df = df.sort_values("Time_hours").reset_index(drop=True)

    if x_limit is not None:
        df = df[
            (df["Time_hours"] >= x_limit[0]) &
            (df["Time_hours"] <= x_limit[1])
        ].copy()

    return df


def make_plot(output_figure, x_limit=None, legend_y=0.52):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for item in MIXING_FILES:
        df = read_mixing_excel(
            item["path"],
            item.get("sheet_name"),
            x_limit=x_limit
        )

        if df.empty:
            continue

        line = ax.plot(
            df["Time_hours"],
            df["Mixing_degree_percent_raw"],
            label=item["label"],
            linestyle=item["linestyle"],
            marker=item["marker"],
            markevery=MARKER_EVERY,
            linewidth=LINEWIDTH,
            markersize=MARKER_SIZE,
            alpha=ALPHA,
        )

        # Marker siste punkt tydelig, slik at kurven ikke ser ut
        # som den fortsetter etter siste måling.
        ax.plot(
            df["Time_hours"].iloc[-1],
            df["Mixing_degree_percent_raw"].iloc[-1],
            marker=item["marker"],
            markersize=END_MARKER_SIZE,
            linestyle="None",
            color=line[0].get_color(),
            alpha=ALPHA,
        )

    ax.set_xlabel("Time [h]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mixing degree [%]", fontsize=AXIS_LABEL_SIZE)

    if x_limit is not None:
        ax.set_xlim(x_limit)
    else:
        ax.set_xlim(left=0)

    ax.set_ylim(Y_LIMIT)

    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=GRID_ALPHA)

    ax.legend(
        fontsize=LEGEND_SIZE,
        loc="lower right",
        bbox_to_anchor=(0.99, legend_y),
        frameon=True,
    )

    fig.tight_layout()

    plt.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.show()

    print("Figur lagret som:")
    print(output_figure)


# =====================================================
# MAIN
# =====================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    make_plot(
        output_figure=OUTPUT_FIGURE_40H,
        x_limit=(0, 40),
        legend_y=0.57
    )

    make_plot(
        output_figure=OUTPUT_FIGURE_FULL,
        x_limit=None,
        legend_y=0.4
    )


if __name__ == "__main__":
    main()
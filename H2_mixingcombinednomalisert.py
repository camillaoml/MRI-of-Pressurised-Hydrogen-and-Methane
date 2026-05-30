#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =====================================================
# PLOT STYLE
# =====================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.facecolor": "white",
    "figure.facecolor": "white"
})

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

FIGSIZE = (6.5, 4.8)
MAX_TIME_HOURS = 12


# =====================================================
# COLOURS AND MARKERS
# =====================================================

SETUP_STYLE = {
    "Horizontal": {
        "color": "#2CA25F",
        "marker": "o",
        "linestyle": "-",
    },
    "N$_2$ on top": {
        "color": "#8856A7",
        "marker": "s",
        "linestyle": "-",
    },
    "H$_2$ on top": {
        "color": "#D95F0E",
        "marker": "^",
        "linestyle": "-",
    },
}


# =====================================================
# EXPERIMENTS
# =====================================================

experiments = [
    {
        "name": "H$_2$ on top",
        "file_path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45V/H2_N2_diffusion_results.xlsx",
        "time_mode": "Auto",
        "h2_col": "ROI1_mean",
    },
    {
        "name": "N$_2$ on top",
        "file_path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45_N2top/H2_N2_diffusion_N2top_results.xlsx",
        "time_mode": "ClockTime",
        "h2_col": "ROI2_bottom_H2_mean",
    },
    {
        "name": "Horizontal",
        "file_path": r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45_sidelengs/H2_N2_horisontal.xlsx",
        "time_mode": "Time_seconds",
        "h2_col": "ROI1_left_H2_mean",
    },
]


# =====================================================
# OUTPUT
# =====================================================

output_folder = os.path.dirname(experiments[0]["file_path"])

output_png = os.path.join(
    output_folder,
    "H2_N2_normalised_1H_concentration_three_configurations_12h_zoomed.png"
)


# =====================================================
# TIME HANDLING
# =====================================================

def add_time_hours(df, time_mode):
    if time_mode == "Time_seconds":
        if "Time_seconds" not in df.columns:
            raise ValueError("Missing column 'Time_seconds'.")
        df["Time_hours"] = pd.to_numeric(df["Time_seconds"], errors="coerce") / 3600
        return df

    if time_mode == "Auto":
        if "Time_seconds" in df.columns:
            df["Time_hours"] = pd.to_numeric(df["Time_seconds"], errors="coerce") / 3600
            return df
        time_mode = "ClockTime"

    if time_mode == "ClockTime":
        if "Time" not in df.columns:
            raise ValueError("Missing column 'Time'.")

        df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")

        clock_seconds = (
            df["Time_dt"].dt.hour * 3600 +
            df["Time_dt"].dt.minute * 60 +
            df["Time_dt"].dt.second
        )

        corrected_seconds = []
        day_offset = 0
        previous_time = None

        for t in clock_seconds:
            if pd.isna(t):
                corrected_seconds.append(np.nan)
                continue

            if previous_time is not None:
                if t < previous_time and (previous_time - t) > 12 * 3600:
                    day_offset += 86400

            corrected_seconds.append(t + day_offset)
            previous_time = t

        df["Time_seconds_plot"] = np.array(corrected_seconds, dtype=float)
        df["Time_seconds_plot"] = (
            df["Time_seconds_plot"] - df["Time_seconds_plot"].dropna().iloc[0]
        )
        df["Time_hours"] = df["Time_seconds_plot"] / 3600
        return df

    raise ValueError(f"Unknown time_mode: {time_mode}")


# =====================================================
# PLOT
# =====================================================

fig, ax = plt.subplots(figsize=FIGSIZE)

for exp in experiments:
    df = pd.read_excel(exp["file_path"])
    df.columns = df.columns.astype(str).str.strip()

    if "Scan" in df.columns:
        df = df.sort_values("Scan").reset_index(drop=True)

    df = add_time_hours(df, exp["time_mode"])
    df = df.sort_values("Time_hours").reset_index(drop=True)
    df = df[df["Time_hours"] <= MAX_TIME_HOURS].copy()

    df[exp["h2_col"]] = pd.to_numeric(df[exp["h2_col"]], errors="coerce")
    df = df.dropna(subset=["Time_hours", exp["h2_col"]])

    x = df["Time_hours"].to_numpy()
    h2_raw = df[exp["h2_col"]].to_numpy()

    # Normalise by the first valid value in each experiment
    h2_initial = h2_raw[0]

    if h2_initial == 0:
        raise ValueError(f"Initial signal is zero for {exp['name']}, cannot normalise.")

    h2_norm = h2_raw / h2_initial

    style = SETUP_STYLE[exp["name"]]

    ax.plot(
        x,
        h2_norm,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=4.2,
        linewidth=1.6,
        color=style["color"],
        label=exp["name"],
    )


# =====================================================
# AXES AND LEGEND
# =====================================================

ax.set_xlabel("Time [h]")
ax.set_ylabel("$^1$H concentration [-]", labelpad=6)

ax.set_xlim(0, MAX_TIME_HOURS)
ax.set_ylim(0.55, 1.02)

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

ax.grid(which="major", alpha=0.30)
ax.grid(which="minor", alpha=0.15)

# Small text inside plot
ax.text(
    0.42, 0.50,
    "H$_2$--N$_2$ mixing experiments",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=11
)

# Legend below the plot
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=True,
    framealpha=0.92,
    edgecolor="0.75",
    fontsize=9,
)

fig.tight_layout(pad=1.2)

fig.savefig(
    output_png,
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("Ferdig!")
print(f"Plot lagret her: {output_png}")
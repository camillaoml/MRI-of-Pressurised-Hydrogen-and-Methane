import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -------------------------------------------------
# Plot style
# -------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 8.5,
    "figure.dpi": 300
})

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

figsize_small = (5.8, 3.9)

# Data curve colours
COLOR_H2 = "seagreen"
COLOR_N2 = "cornflowerblue"
COLOR_MARKER = "0.20"

# Background colours with clearly different lightness
# Chosen so they remain distinguishable in grayscale printing
PRE_CHANGE_COLOR = "#d9e1ef"
POST_CHANGE_COLOR = "#f5efe6"
REGION_ALPHA = 0.65

CENTER_LABEL_COLOR = "0.28"

plot_kwargs = dict(
    marker="o",
    markersize=1.5,
    linestyle="--",
    linewidth=0.8,
    capsize=0
)

# -------------------------------------------------
# Settings
# -------------------------------------------------
SCAN_BEFORE_FLIP = 169
SCAN_AFTER_FLIP = 172
EXCLUDE_SCANS = []

TIME_TICK_STEP = 4

# -------------------------------------------------
# 1. Read Excel
# -------------------------------------------------
file_path = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45V/H2_N2_diffusion_results.xlsx"

output_folder = os.path.dirname(os.path.abspath(file_path))

df = pd.read_excel(file_path)
df = df.sort_values("Scan").reset_index(drop=True)

# -------------------------------------------------
# 2. Clean data
# -------------------------------------------------
df = df[~df["Scan"].isin(EXCLUDE_SCANS)].copy()

numeric_cols = [
    "Scan",
    "ROI1_mean",
    "ROI2_mean",
    "ROI1_std",
    "ROI2_std"
]

if "Time_seconds" in df.columns:
    numeric_cols.append("Time_seconds")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=["Scan", "ROI1_mean", "ROI2_mean", "ROI1_std", "ROI2_std"]
).copy()

df = df.sort_values("Scan").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid data left after removing NaN/Inf values.")

# -------------------------------------------------
# 3. Time in hours
# -------------------------------------------------
if "Time_seconds" in df.columns:
    df = df.dropna(subset=["Time_seconds"]).copy()
    df["Time_hours"] = df["Time_seconds"] / 3600

else:
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

    seconds = (
        df["Time"].dt.hour * 3600 +
        df["Time"].dt.minute * 60 +
        df["Time"].dt.second
    )

    day_jump = seconds.diff() < 0
    day_count = day_jump.cumsum()

    time_seconds = seconds + day_count * 86400
    time_seconds = time_seconds - time_seconds.iloc[0]

    df["Time_seconds"] = time_seconds
    df["Time_hours"] = df["Time_seconds"] / 3600

df = df.sort_values("Scan").reset_index(drop=True)
x = df["Time_hours"].to_numpy()

# -------------------------------------------------
# 4. ROI signals
#
# ROI1 = initially H2-filled chamber
# ROI2 = initially N2-filled chamber
#
# The curves are kept continuous through the full experiment.
# -------------------------------------------------
h2 = df["ROI1_mean"].to_numpy(dtype=float)
n2 = df["ROI2_mean"].to_numpy(dtype=float)

h2_std = df["ROI1_std"].to_numpy(dtype=float)
n2_std = df["ROI2_std"].to_numpy(dtype=float)

# -------------------------------------------------
# 5. 1H concentration from conserved total signal
# Values are scaled from 0 to 1 using the total signal.
# -------------------------------------------------
total_signal = h2 + n2

h2_concentration = h2 / total_signal
n2_concentration = n2 / total_signal

h2_concentration_std = h2_std / total_signal
n2_concentration_std = n2_std / total_signal

# -------------------------------------------------
# 6. Find time where vertical gas arrangement changed
# Midpoint between scan 169 and scan 172
# -------------------------------------------------
time_before = df.loc[df["Scan"] == SCAN_BEFORE_FLIP, "Time_hours"]
time_after = df.loc[df["Scan"] == SCAN_AFTER_FLIP, "Time_hours"]

if len(time_before) > 0 and len(time_after) > 0:
    flip_start = float(time_before.values[0])
    flip_end = float(time_after.values[0])
    arrangement_change_time = (flip_start + flip_end) / 2
else:
    flip_start = None
    flip_end = None
    arrangement_change_time = None

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def format_time_axis(ax):
    ax.set_xlim(-0.5, x.max() + 1.0)
    ax.set_xticks(np.arange(0, x.max() + TIME_TICK_STEP, TIME_TICK_STEP))


def format_concentration_axis(ax):
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))


def add_configuration_regions(ax):
    """
    Adds coloured background regions:
    - before change: H2 above N2
    - after change: N2 above H2
    """
    if arrangement_change_time is None:
        return

    xmin, xmax = ax.get_xlim()

    ax.axvspan(
        xmin,
        arrangement_change_time,
        facecolor=PRE_CHANGE_COLOR,
        alpha=REGION_ALPHA,
        edgecolor="none",
        zorder=0
    )

    ax.axvspan(
        arrangement_change_time,
        xmax,
        facecolor=POST_CHANGE_COLOR,
        alpha=REGION_ALPHA,
        edgecolor="none",
        zorder=0
    )

    ax.axvline(
        arrangement_change_time,
        linestyle="--",
        color=COLOR_MARKER,
        linewidth=1.3,
        zorder=3
    )

def add_center_label(ax):
    """
    Adds a short descriptive label in the upper-right empty part of the plot.
    """
    ax.text(
        0.96,
        0.82,
        r"Vertical H$_2$--N$_2$ mixing",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=10.5,
        color=CENTER_LABEL_COLOR,
        zorder=4
    )
def add_combined_legend_below(fig, include_h2=True, include_n2=True):
    """
    Adds one combined legend below the figure.
    Includes both data curves and background region meanings.
    """
    handles = []

    if include_h2:
        handles.append(
            Line2D(
                [0], [0],
                color=COLOR_H2,
                linestyle="--",
                marker="o",
                markersize=3,
                linewidth=0.8,
                label=r"$^{1}$H signal, initial H$_2$ chamber"
            )
        )

    if include_n2:
        handles.append(
            Line2D(
                [0], [0],
                color=COLOR_N2,
                linestyle="--",
                marker="o",
                markersize=3,
                linewidth=0.8,
                label=r"$^{1}$H signal, initial N$_2$ chamber"
            )
        )

    handles.extend([
        Patch(
            facecolor=PRE_CHANGE_COLOR,
            edgecolor="none",
            alpha=REGION_ALPHA,
            label=r"H$_2$ above N$_2$"
        ),
        Patch(
            facecolor=POST_CHANGE_COLOR,
            edgecolor="none",
            alpha=REGION_ALPHA,
            label=r"N$_2$ above H$_2$"
        )
    ])

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.0,
        handletextpad=0.6
    )


def save_plot(fig, filename):
    fig.subplots_adjust(bottom=0.34)
    fig.savefig(
        os.path.join(output_folder, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08
    )
    plt.show()


# -------------------------------------------------
# 7. Main 1H concentration plot: H2 + N2
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.errorbar(
    x,
    h2_concentration,
    yerr=h2_concentration_std,
    color=COLOR_H2,
    **plot_kwargs
)

ax.errorbar(
    x,
    n2_concentration,
    yerr=n2_concentration_std,
    color=COLOR_N2,
    **plot_kwargs
)

ax.set_xlabel("Time [h]")
ax.set_ylabel(r"$^{1}$H concentration")

format_time_axis(ax)
format_concentration_axis(ax)
add_configuration_regions(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=True)
save_plot(fig, "H1_concentration_H2_N2_110_configuration_regions.png")

# -------------------------------------------------
# 8. Raw signal plot: H2 + N2
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.errorbar(
    x,
    h2,
    yerr=h2_std,
    color=COLOR_H2,
    **plot_kwargs
)

ax.errorbar(
    x,
    n2,
    yerr=n2_std,
    color=COLOR_N2,
    **plot_kwargs
)

ax.set_xlabel("Time [h]")
ax.set_ylabel("Signal intensity")

format_time_axis(ax)
add_configuration_regions(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=True)
save_plot(fig, "signal_H2_N2_110_configuration_regions.png")

# -------------------------------------------------
# 9. Initial N2 chamber, 1H concentration
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.errorbar(
    x,
    n2_concentration,
    yerr=n2_concentration_std,
    color=COLOR_N2,
    **plot_kwargs
)

ax.set_xlabel("Time [h]")
ax.set_ylabel(r"$^{1}$H concentration [-]")

format_time_axis(ax)
format_concentration_axis(ax)
add_configuration_regions(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=False, include_n2=True)
save_plot(fig, "initial_N2_H1_concentration_H2_N2_110_configuration_regions.png")

# -------------------------------------------------
# 10. Initial H2 chamber, 1H concentration
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.errorbar(
    x,
    h2_concentration,
    yerr=h2_concentration_std,
    color=COLOR_H2,
    **plot_kwargs
)

ax.set_xlabel("Time [h]")
ax.set_ylabel(r"$^{1}$H concentration [-]")

format_time_axis(ax)
format_concentration_axis(ax)
add_configuration_regions(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=False)
save_plot(fig, "initial_H2_H1_concentration_H2_N2_110_configuration_regions.png")

# -------------------------------------------------
# 11. Save corrected plotting data
# -------------------------------------------------
df["H2_signal"] = h2
df["N2_signal"] = n2
df["H2_std"] = h2_std
df["N2_std"] = n2_std

df["H2_H1_concentration"] = h2_concentration
df["N2_H1_concentration"] = n2_concentration
df["H2_H1_concentration_std"] = h2_concentration_std
df["N2_H1_concentration_std"] = n2_concentration_std

corrected_excel_path = os.path.join(
    output_folder,
    "H2_N2_110_plot_data_H1_concentration_configuration_regions.xlsx"
)

df.to_excel(corrected_excel_path, index=False)

print("Plots saved to:")
print(output_folder)

print("\nCorrected plotting data saved to:")
print(corrected_excel_path)

print("\nRemoved scans:")
print(EXCLUDE_SCANS)

if arrangement_change_time is not None:
    print(f"\nVertical gas arrangement changed at approximately {arrangement_change_time:.2f} h.")
    print(f"Flip interval: scan {SCAN_BEFORE_FLIP} to scan {SCAN_AFTER_FLIP}")
    print(f"Scan {SCAN_BEFORE_FLIP}: {flip_start:.2f} h")
    print(f"Scan {SCAN_AFTER_FLIP}: {flip_end:.2f} h")
else:
    print("\nNo arrangement-change scans were found. Background regions were not added.")

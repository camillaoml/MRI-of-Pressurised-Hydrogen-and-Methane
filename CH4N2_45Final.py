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
COLOR_CH4 = "indianred"
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
ARRANGEMENT_CHANGE_SCAN = 262
EXCLUDE_SCANS = [260]

# -------------------------------------------------
# 1. Read Excel
# -------------------------------------------------
file_path = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion/CH4_diff_experiments/CH4_N2_120_diff45V_diffusion_results.xlsx"

output_folder = os.path.dirname(os.path.abspath(file_path))

df = pd.read_excel(file_path)
df = df.sort_values("Scan").reset_index(drop=True)

# -------------------------------------------------
# 2. Clean data
# -------------------------------------------------
df = df[~df["Scan"].isin(EXCLUDE_SCANS)].copy()

numeric_cols = [
    "Scan",
    "Time_seconds",
    "ROI1_mean",
    "ROI2_mean",
    "ROI1_std",
    "ROI2_std"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=["Scan", "Time_seconds", "ROI1_mean", "ROI2_mean", "ROI1_std", "ROI2_std"]
).copy()

df = df.sort_values("Scan").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid data left after removing NaN/Inf values.")

# -------------------------------------------------
# 3. Time in hours
# -------------------------------------------------
df["Time_hours"] = df["Time_seconds"] / 3600
x = df["Time_hours"].to_numpy()

# -------------------------------------------------
# 4. ROI signals
#
# ROI1 = initially CH4-filled chamber
# ROI2 = initially N2-filled chamber
#
# The curves are kept continuous through the full experiment.
# -------------------------------------------------
ch4 = df["ROI1_mean"].to_numpy(dtype=float)
n2 = df["ROI2_mean"].to_numpy(dtype=float)

ch4_std = df["ROI1_std"].to_numpy(dtype=float)
n2_std = df["ROI2_std"].to_numpy(dtype=float)

# -------------------------------------------------
# 5. 1H concentration from conserved total signal
# Values are scaled from 0 to 1 using the total signal.
# -------------------------------------------------
total_signal = ch4 + n2

ch4_concentration = ch4 / total_signal
n2_concentration = n2 / total_signal

ch4_concentration_std = ch4_std / total_signal
n2_concentration_std = n2_std / total_signal

# -------------------------------------------------
# 6. Find time where vertical gas arrangement changed
# Midpoint between scan 262 and the next valid scan
# -------------------------------------------------
change_time_series = df.loc[df["Scan"] == ARRANGEMENT_CHANGE_SCAN, "Time_hours"]

if len(change_time_series) > 0:
    change_time = float(change_time_series.values[0])
    change_idx = df.index[df["Scan"] == ARRANGEMENT_CHANGE_SCAN][0]

    if change_idx + 1 < len(df):
        next_time = df["Time_hours"].iloc[change_idx + 1]
    else:
        next_time = change_time

    arrangement_change_time = (change_time + next_time) / 2
else:
    arrangement_change_time = None

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def format_time_axis(ax):
    ax.set_xlim(-0.5, x.max() + 1.0)
    ax.set_xticks(np.arange(0, x.max() + 4, 4))


def format_concentration_axis(ax):
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))


def add_configuration_regions(ax):
    """
    Adds coloured background regions:
    - before change: CH4 above N2
    - after change: N2 above CH4
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
    Adds a short descriptive label in the empty central part of the plot.
    """
    ax.text(
        0.40,
        0.45,
        r"Vertical CH$_4$--N$_2$ mixing",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=CENTER_LABEL_COLOR,
        zorder=4
    )


def add_combined_legend_below(fig, include_ch4=True, include_n2=True):
    """
    Adds one combined legend below the figure.
    Includes both data curves and background region meanings.
    """
    handles = []

    if include_ch4:
        handles.append(
            Line2D(
                [0], [0],
                color=COLOR_CH4,
                linestyle="--",
                marker="o",
                markersize=3,
                linewidth=0.8,
                label=r"$^{1}$H signal, initial CH$_4$ chamber"
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
            label=r"CH$_4$ above N$_2$"
        ),
        Patch(
            facecolor=POST_CHANGE_COLOR,
            edgecolor="none",
            alpha=REGION_ALPHA,
            label=r"N$_2$ above CH$_4$"
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
# 7. Main 1H concentration plot: CH4 + N2
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.errorbar(
    x,
    ch4_concentration,
    yerr=ch4_concentration_std,
    color=COLOR_CH4,
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

add_combined_legend_below(fig, include_ch4=True, include_n2=True)
save_plot(fig, "H1_concentration_CH4_N2_120_configuration_regions.png")

# -------------------------------------------------
# 8. Raw signal plot: CH4 + N2
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.errorbar(
    x,
    ch4,
    yerr=ch4_std,
    color=COLOR_CH4,
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

add_combined_legend_below(fig, include_ch4=True, include_n2=True)
save_plot(fig, "signal_CH4_N2_120_configuration_regions.png")

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

add_combined_legend_below(fig, include_ch4=False, include_n2=True)
save_plot(fig, "initial_N2_H1_concentration_configuration_regions.png")

# -------------------------------------------------
# 10. Initial CH4 chamber, 1H concentration
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.errorbar(
    x,
    ch4_concentration,
    yerr=ch4_concentration_std,
    color=COLOR_CH4,
    **plot_kwargs
)

ax.set_xlabel("Time [h]")
ax.set_ylabel(r"$^{1}$H concentration [-]")

format_time_axis(ax)
format_concentration_axis(ax)
add_configuration_regions(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_ch4=True, include_n2=False)
save_plot(fig, "initial_CH4_H1_concentration_configuration_regions.png")

# -------------------------------------------------
# 11. Save corrected plotting data
# -------------------------------------------------
df["CH4_signal"] = ch4
df["N2_signal"] = n2
df["CH4_std"] = ch4_std
df["N2_std"] = n2_std

df["CH4_H1_concentration"] = ch4_concentration
df["N2_H1_concentration"] = n2_concentration
df["CH4_H1_concentration_std"] = ch4_concentration_std
df["N2_H1_concentration_std"] = n2_concentration_std

corrected_excel_path = os.path.join(
    output_folder,
    "CH4_N2_120_plot_data_H1_concentration_configuration_regions_without_scan260.xlsx"
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
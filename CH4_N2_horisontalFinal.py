import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

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

CENTER_LABEL_COLOR = "0.28"

plot_kwargs = dict(
    marker="o",
    markersize=1.5,
    linestyle="--",
    linewidth=0.8,
    capsize=0
)

plot_kwargs_no_error = dict(
    marker="o",
    markersize=1.5,
    linestyle="--",
    linewidth=0.8
)

# -------------------------------------------------
# Settings
# -------------------------------------------------
SLICE_TO_PLOT = 6
EXCLUDE_SCANS = []

# Zoom plot settings
ZOOM_XMIN = 0
ZOOM_XMAX = 16
ZOOM_TICK_STEP = 4

# Full-time plot settings
FULL_TICK_STEP = 8
# Change to 12 if the full-time plot still looks too crowded.

# -------------------------------------------------
# 1. Read Excel
# -------------------------------------------------
file_path = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion/CH4_N2_100_diff45_2_diffusion_results.xlsx"

output_folder = os.path.dirname(os.path.abspath(file_path))

df = pd.read_excel(file_path)
df = df.sort_values("Scan").reset_index(drop=True)

# -------------------------------------------------
# 2. Keep only Slice 6
# -------------------------------------------------
if "Slice" not in df.columns:
    raise ValueError("The Excel file does not contain a 'Slice' column.")

df = df[df["Slice"] == SLICE_TO_PLOT].copy()
df = df[~df["Scan"].isin(EXCLUDE_SCANS)].copy()

if df.empty:
    raise ValueError(f"No rows found for Slice = {SLICE_TO_PLOT}.")

# -------------------------------------------------
# 3. Clean data
# -------------------------------------------------
numeric_cols = [
    "Scan",
    "Slice",
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
    subset=[
        "Scan",
        "Slice",
        "ROI1_mean",
        "ROI2_mean",
        "ROI1_std",
        "ROI2_std"
    ]
).copy()

df = df.sort_values("Scan").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid data left after removing NaN/Inf values.")

# -------------------------------------------------
# 4. Time in hours
# -------------------------------------------------
if "Time_seconds" in df.columns:
    df = df.dropna(subset=["Time_seconds"]).copy()
    df["Time_seconds_plot"] = df["Time_seconds"] - df["Time_seconds"].iloc[0]
    df["Time_hours"] = df["Time_seconds_plot"] / 3600

else:
    if "Time" not in df.columns:
        raise ValueError("Found neither 'Time_seconds' nor 'Time' in the Excel file.")

    df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

    seconds = (
        df["Time_dt"].dt.hour * 3600
        + df["Time_dt"].dt.minute * 60
        + df["Time_dt"].dt.second
    )

    day_jump = seconds.diff() < 0
    day_count = day_jump.cumsum()

    time_seconds = seconds + day_count * 86400
    time_seconds = time_seconds - time_seconds.iloc[0]

    df["Time_seconds_plot"] = time_seconds
    df["Time_hours"] = df["Time_seconds_plot"] / 3600

df = df.sort_values("Time_hours").reset_index(drop=True)
x = df["Time_hours"].to_numpy()

# -------------------------------------------------
# 5. ROI signals
#
# ROI1 = initially CH4-filled chamber
# ROI2 = initially N2-filled chamber
# -------------------------------------------------
ch4 = df["ROI1_mean"].to_numpy(dtype=float)
n2 = df["ROI2_mean"].to_numpy(dtype=float)

ch4_std = df["ROI1_std"].to_numpy(dtype=float)
n2_std = df["ROI2_std"].to_numpy(dtype=float)

# -------------------------------------------------
# 6. 1H concentration from conserved total signal
# Values are scaled from 0 to 1 using the total signal.
# -------------------------------------------------
total_signal = ch4 + n2

ch4_concentration = ch4 / total_signal
n2_concentration = n2 / total_signal

ch4_concentration_std = ch4_std / total_signal
n2_concentration_std = n2_std / total_signal

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def format_time_axis(ax, mode):
    """
    mode = "zoom" or "full"
    """
    if mode == "zoom":
        ax.set_xlim(ZOOM_XMIN, ZOOM_XMAX)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(ZOOM_TICK_STEP))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    elif mode == "full":
        max_time = x.max()
        ax.set_xlim(-0.5, max_time + 1.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(FULL_TICK_STEP))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(FULL_TICK_STEP / 2))

    else:
        raise ValueError("mode must be 'zoom' or 'full'.")


def format_concentration_axis(ax):
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))


def add_horizontal_label(ax):
    ax.text(
        0.96,
        0.82,
        r"Horizontal CH$_4$--N$_2$ mixing",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=10.5,
        color=CENTER_LABEL_COLOR,
        zorder=4
    )


def add_combined_legend_below(fig, include_ch4=True, include_n2=True):
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

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.0,
        handletextpad=0.6
    )


def add_grid(ax):
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.15)


def save_plot(fig, filename, bottom=0.24):
    fig.subplots_adjust(bottom=bottom)
    fig.savefig(
        os.path.join(output_folder, filename),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08
    )
    plt.show()


def plot_h1_concentration(mode, suffix):
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

    format_time_axis(ax, mode)
    format_concentration_axis(ax)
    add_grid(ax)
    add_horizontal_label(ax)

    add_combined_legend_below(fig, include_ch4=True, include_n2=True)
    save_plot(fig, f"H1_concentration_CH4_N2_100_horizontal_slice6_{suffix}.png")


def plot_raw_signal(mode, suffix):
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

    format_time_axis(ax, mode)
    add_grid(ax)
    add_horizontal_label(ax)

    add_combined_legend_below(fig, include_ch4=True, include_n2=True)
    save_plot(fig, f"signal_CH4_N2_100_horizontal_slice6_{suffix}.png")


def plot_initial_n2(mode, suffix):
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

    format_time_axis(ax, mode)
    format_concentration_axis(ax)
    add_grid(ax)
    add_horizontal_label(ax)

    add_combined_legend_below(fig, include_ch4=False, include_n2=True)
    save_plot(fig, f"initial_N2_H1_concentration_CH4_N2_100_horizontal_slice6_{suffix}.png")


def plot_initial_ch4(mode, suffix):
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

    format_time_axis(ax, mode)
    format_concentration_axis(ax)
    add_grid(ax)
    add_horizontal_label(ax)

    add_combined_legend_below(fig, include_ch4=True, include_n2=False)
    save_plot(fig, f"initial_CH4_H1_concentration_CH4_N2_100_horizontal_slice6_{suffix}.png")


# -------------------------------------------------
# 7. Make zoom plots: 0-16 h, tick every 4 h
# -------------------------------------------------
plot_h1_concentration(mode="zoom", suffix="0_16h")
plot_raw_signal(mode="zoom", suffix="0_16h")
plot_initial_n2(mode="zoom", suffix="0_16h")
plot_initial_ch4(mode="zoom", suffix="0_16h")

# -------------------------------------------------
# 8. Make full-time plots: full experiment, larger tick spacing
# -------------------------------------------------
plot_h1_concentration(mode="full", suffix="full_time")
plot_raw_signal(mode="full", suffix="full_time")
plot_initial_n2(mode="full", suffix="full_time")
plot_initial_ch4(mode="full", suffix="full_time")

# -------------------------------------------------
# 9. Save plotting data
# -------------------------------------------------
df["CH4_signal"] = ch4
df["N2_signal"] = n2
df["CH4_std"] = ch4_std
df["N2_std"] = n2_std

df["CH4_H1_concentration"] = ch4_concentration
df["N2_H1_concentration"] = n2_concentration
df["CH4_H1_concentration_std"] = ch4_concentration_std
df["N2_H1_concentration_std"] = n2_concentration_std

plot_data_path = os.path.join(
    output_folder,
    "CH4_N2_100_horizontal_slice6_plot_data_H1_concentration.xlsx"
)

df.to_excel(plot_data_path, index=False)

# -------------------------------------------------
# 10. Print summary
# -------------------------------------------------
print("Plots saved to:")
print(output_folder)

print("\nPlotting data saved to:")
print(plot_data_path)

print("\nConfiguration:")
print("Horizontal CH4--N2 mixing. Only Slice 6 was plotted.")

print("\nTwo time ranges were plotted:")
print(f"1. Zoom: {ZOOM_XMIN}-{ZOOM_XMAX} h, ticks every {ZOOM_TICK_STEP} h")
print(f"2. Full time: 0-{x.max():.2f} h, ticks every {FULL_TICK_STEP} h")

print("\nNumber of plotted rows:")
print(len(df))

print("\nRemoved scans:")
print(EXCLUDE_SCANS)
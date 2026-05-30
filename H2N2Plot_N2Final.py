import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

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

# Background colour for N2 above H2
N2_TOP_COLOR = "#f5efe6"
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
EXCLUDE_SCANS = []
TIME_TICK_STEP = 2

# Artificial jump correction
MAX_REASONABLE_GAP_SECONDS = 20 * 60
# Any scan-to-scan gap larger than this is treated as an artificial time jump.

# -------------------------------------------------
# 1. Read Excel
# -------------------------------------------------
file_path = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45_N2top/H2_N2_diffusion_N2top_results.xlsx"

output_folder = os.path.dirname(os.path.abspath(file_path))

df = pd.read_excel(file_path)
df = df.sort_values("Scan").reset_index(drop=True)

# -------------------------------------------------
# 2. Clean data
# -------------------------------------------------
df = df[~df["Scan"].isin(EXCLUDE_SCANS)].copy()

numeric_cols = [
    "Scan",
    "ROI1_top_N2_mean",
    "ROI2_bottom_H2_mean",
    "ROI1_top_N2_std",
    "ROI2_bottom_H2_std"
]

if "Time_seconds" in df.columns:
    numeric_cols.append("Time_seconds")

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[
        "Scan",
        "ROI1_top_N2_mean",
        "ROI2_bottom_H2_mean",
        "ROI1_top_N2_std",
        "ROI2_bottom_H2_std"
    ]
).copy()

df = df.sort_values("Scan").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid data left after removing NaN/Inf values.")

# -------------------------------------------------
# 3. Correct time and convert to hours
#
# IMPORTANT:
# Do not use the existing Time_seconds column here.
# In this file, Time_seconds already contains false day jumps.
# Time is rebuilt from the clock-time column instead.
# -------------------------------------------------

ROLLOVER_THRESHOLD_SECONDS = 12 * 3600  # only treat backwards jump > 12 h as midnight

# Keep original scan order for diagnostics
df["Original_row_order"] = np.arange(len(df))

# Parse clock time from the Time column
df["Time_dt"] = pd.to_datetime(
    df["Time"].astype(str),
    format="%H:%M:%S",
    errors="coerce"
)

if df["Time_dt"].isna().any():
    print("\nWarning: Some Time values could not be parsed and will be removed:")
    print(df.loc[df["Time_dt"].isna(), ["Scan", "Time"]].to_string(index=False))
    df = df.dropna(subset=["Time_dt"]).copy().reset_index(drop=True)

clock_seconds = (
    df["Time_dt"].dt.hour * 3600
    + df["Time_dt"].dt.minute * 60
    + df["Time_dt"].dt.second
).to_numpy(dtype=float)

# Unwrap only true midnight rollover.
# Small backwards jumps are not treated as new days.
unwrapped_seconds = []
day_offset = 0
previous_clock_time = None

for i, t in enumerate(clock_seconds):
    scan_no = df["Scan"].iloc[i]

    if previous_clock_time is not None:
        backwards_jump = previous_clock_time - t

        if backwards_jump > ROLLOVER_THRESHOLD_SECONDS:
            day_offset += 86400
            print(
                f"Detected midnight rollover before scan {scan_no}. "
                f"Added 24 h."
            )

        elif backwards_jump > 0:
            print(
                f"Small backwards time jump before scan {scan_no}: "
                f"{backwards_jump / 60:.2f} min. "
                f"Not treated as a new day."
            )

    unwrapped_seconds.append(t + day_offset)
    previous_clock_time = t

unwrapped_seconds = np.array(unwrapped_seconds, dtype=float)

# Start at zero
df["Time_seconds_plot"] = unwrapped_seconds - unwrapped_seconds[0]
df["Time_hours"] = df["Time_seconds_plot"] / 3600

# Sort by corrected chronological time, not scan number
df = df.sort_values("Time_hours").reset_index(drop=True)

x = df["Time_hours"].to_numpy()

# Keep this variable so the rest of your script does not break
total_removed_time = 0.0

# -------------------------------------------------
# Time diagnostics
# -------------------------------------------------
time_check = df[["Scan", "Time", "Time_hours", "Original_row_order"]].copy()
time_check["dt_minutes"] = np.r_[np.nan, np.diff(df["Time_seconds_plot"]) / 60]

print("\nTime check, first 40 chronological points:")
print(time_check.head(40).to_string(index=False))

print("\nLargest time steps:")
print(
    time_check.sort_values("dt_minutes", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\nNon-positive time steps after sorting:")
print(
    time_check[time_check["dt_minutes"] <= 0]
    .to_string(index=False)
)
# -------------------------------------------------
# 4. ROI signals
#
# ROI1 = initially N2-filled chamber, placed on top
# ROI2 = initially H2-filled chamber, placed at bottom
# -------------------------------------------------
n2 = df["ROI1_top_N2_mean"].to_numpy(dtype=float)
h2 = df["ROI2_bottom_H2_mean"].to_numpy(dtype=float)

n2_std = df["ROI1_top_N2_std"].to_numpy(dtype=float)
h2_std = df["ROI2_bottom_H2_std"].to_numpy(dtype=float)

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
# 6. Optional exponential fit for N2 signal
# -------------------------------------------------
n2_norm = (n2 - n2.min()) / (n2.max() - n2.min())


def exp_model(t, A, k, C):
    return A * (1 - np.exp(-k * t)) + C


initial_guess = [1.0, 0.2, 0.0]

try:
    popt, pcov = curve_fit(
        exp_model,
        x,
        n2_norm,
        p0=initial_guess,
        maxfev=10000
    )

    A_fit, k_fit, C_fit = popt

    x_fit = np.linspace(0, x.max(), 300)
    n2_fit = exp_model(x_fit, A_fit, k_fit, C_fit)

    tau = 1 / k_fit
    half_time = np.log(2) / k_fit

    slope_norm, intercept_norm = np.polyfit(x, n2_norm, 1)
    linear_fit_norm = slope_norm * x_fit + intercept_norm

    fit_successful = True

except Exception as e:
    print("\nExponential fit failed:")
    print(e)

    A_fit = np.nan
    k_fit = np.nan
    C_fit = np.nan
    tau = np.nan
    half_time = np.nan
    slope_norm = np.nan
    intercept_norm = np.nan

    x_fit = np.linspace(0, x.max(), 300)
    n2_fit = np.full_like(x_fit, np.nan)
    linear_fit_norm = np.full_like(x_fit, np.nan)

    fit_successful = False

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def format_time_axis(ax):
    ax.set_xlim(-0.2, x.max() + 0.5)
    ax.set_xticks(np.arange(0, x.max() + TIME_TICK_STEP, TIME_TICK_STEP))


def format_concentration_axis(ax):
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))


def add_single_configuration_region(ax):
    """
    Adds one background region over the whole plot:
    N2 above H2.
    """
    xmin, xmax = ax.get_xlim()

    ax.axvspan(
        xmin,
        xmax,
        facecolor=N2_TOP_COLOR,
        alpha=REGION_ALPHA,
        edgecolor="none",
        zorder=0
    )


def add_center_label(ax):
    """
    Adds a short descriptive label in the upper-right part of the plot.
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
    Includes data curves and background region meaning.
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

    handles.append(
        Patch(
            facecolor=N2_TOP_COLOR,
            edgecolor="none",
            alpha=REGION_ALPHA,
            label=r"N$_2$ above H$_2$"
        )
    )

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
    fig.subplots_adjust(bottom=0.30)
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
add_single_configuration_region(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=True)
save_plot(fig, "H1_concentration_H2_N2_110_N2top_single_region.png")

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
add_single_configuration_region(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=True)
save_plot(fig, "signal_H2_N2_110_N2top_single_region.png")

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
add_single_configuration_region(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=False, include_n2=True)
save_plot(fig, "initial_N2_H1_concentration_H2_N2_110_N2top_single_region.png")

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
add_single_configuration_region(ax)
ax.grid(alpha=0.3)
add_center_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=False)
save_plot(fig, "initial_H2_H1_concentration_H2_N2_110_N2top_single_region.png")

# -------------------------------------------------
# 11. Optional N2 normalised fit plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.scatter(
    x,
    n2_norm,
    color=COLOR_N2,
    s=5,
    label=r"N$_2$ data",
    zorder=3
)

if fit_successful:
    ax.plot(
        x_fit,
        n2_fit,
        color="tab:red",
        linewidth=1.3,
        label=fr"Exp. fit: k = {k_fit:.3f} h$^{{-1}}$",
        zorder=3
    )

    ax.plot(
        x_fit,
        linear_fit_norm,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label=fr"Linear rate = {slope_norm:.3f} h$^{{-1}}$",
        zorder=3
    )

ax.set_xlabel("Time [h]")
ax.set_ylabel(r"Normalised N$_2$ signal [-]")

format_time_axis(ax)
ax.set_ylim(-0.05, 1.05)
add_single_configuration_region(ax)
ax.grid(alpha=0.3)

ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    os.path.join(output_folder, "N2_exponential_fit_curvefit_H2_N2_110_N2top.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08
)

plt.show()

# -------------------------------------------------
# 12. Save corrected plotting data
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
    "H2_N2_110_N2top_plot_data_H1_concentration_single_region.xlsx"
)

df.to_excel(corrected_excel_path, index=False)

# -------------------------------------------------
# 13. Save fit metrics
# -------------------------------------------------
metrics = {
    "model": "y = A * (1 - exp(-k*t)) + C",
    "A": A_fit,
    "k_per_hour": k_fit,
    "C": C_fit,
    "tau_hours": tau,
    "half_time_hours": half_time,
    "linear_rate_normalised_signal_per_hour": slope_norm,
    "linear_intercept": intercept_norm,
    "fit_successful": fit_successful,
    "total_removed_time_hours": total_removed_time / 3600
}

metrics_df = pd.DataFrame([metrics])

metrics_path = os.path.join(
    output_folder,
    "N2_curve_fit_metrics_H2_N2_110_N2top.xlsx"
)

metrics_df.to_excel(metrics_path, index=False)

# -------------------------------------------------
# 14. Print summary
# -------------------------------------------------
print("Plots saved to:")
print(output_folder)

print("\nCorrected plotting data saved to:")
print(corrected_excel_path)

print("\nMetrics saved to:")
print(metrics_path)

print("\nRemoved scans:")
print(EXCLUDE_SCANS)

print("\nConfiguration:")
print("N2 above H2 for the full experiment.")

print("\nTime handling:")
print(f"Total removed artificial time: {total_removed_time / 3600:.2f} h")

if fit_successful:
    print("\nExponential fit for N2")
    print("----------------------")
    print("Model: y = A * (1 - exp(-k*t)) + C")
    print(f"A = {A_fit:.4f}")
    print(f"k = {k_fit:.4f} 1/h")
    print(f"C = {C_fit:.4f}")
    print(f"tau = {tau:.3f} h")
    print(f"half-time = {half_time:.3f} h")

    print("\nLinear rate for N2")
    print("------------------")
    print(f"linear rate = {slope_norm:.4f} normalised signal/h")
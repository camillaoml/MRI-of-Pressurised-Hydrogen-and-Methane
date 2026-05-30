import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
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
COLOR_H2 = "seagreen"
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
EXCLUDE_SCANS = []

# -------------------------------------------------
# 1. Read Excel
# -------------------------------------------------
file_path = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/H2_N2_110_diff45_sidelengs/H2_N2_horisontal.xlsx"

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
    "ROI1_left_H2_mean",
    "ROI2_right_N2_mean",
    "ROI1_left_H2_std",
    "ROI2_right_N2_std"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[
        "Scan",
        "Time_seconds",
        "ROI1_left_H2_mean",
        "ROI2_right_N2_mean",
        "ROI1_left_H2_std",
        "ROI2_right_N2_std"
    ]
).copy()

df = df.sort_values("Scan").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid data left after removing NaN/Inf values.")

# -------------------------------------------------
# 3. Time in hours
# -------------------------------------------------
df["Time_seconds_plot"] = df["Time_seconds"] - df["Time_seconds"].iloc[0]
df["Time_hours"] = df["Time_seconds_plot"] / 3600

df = df.sort_values("Time_hours").reset_index(drop=True)
x = df["Time_hours"].to_numpy()

# -------------------------------------------------
# 4. ROI signals
#
# ROI1 = initially H2-filled chamber, left side
# ROI2 = initially N2-filled chamber, right side
# -------------------------------------------------
h2 = df["ROI1_left_H2_mean"].to_numpy(dtype=float)
n2 = df["ROI2_right_N2_mean"].to_numpy(dtype=float)

h2_std = df["ROI1_left_H2_std"].to_numpy(dtype=float)
n2_std = df["ROI2_right_N2_std"].to_numpy(dtype=float)

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


try:
    popt, pcov = curve_fit(
        exp_model,
        x,
        n2_norm,
        p0=[1.0, 0.2, 0.0],
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
    max_time = x.max()

    ax.set_xlim(-0.2, max_time + 0.3)

    if max_time <= 5:
        major = 1
        minor = 0.25
    elif max_time <= 10:
        major = 2
        minor = 0.5
    else:
        major = 2
        minor = 1

    ax.xaxis.set_major_locator(ticker.MultipleLocator(major))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor))


def format_concentration_axis(ax):
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))


def add_horizontal_label(ax):
    """
    Adds a short descriptive label in the upper-right part of the plot.
    """
    ax.text(
        0.96,
        0.82,
        r"Horizontal H$_2$--N$_2$ mixing",
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
    No background-region legend is included for horizontal experiments.
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
add_grid(ax)
add_horizontal_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=True)
save_plot(fig, "H1_concentration_H2_N2_110_horizontal.png")

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
add_grid(ax)
add_horizontal_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=True)
save_plot(fig, "signal_H2_N2_110_horizontal.png")

# -------------------------------------------------
# 9. Raw signal plot without error bars
# -------------------------------------------------
fig, ax = plt.subplots(figsize=figsize_small)

ax.plot(
    x,
    h2,
    color=COLOR_H2,
    **plot_kwargs_no_error
)

ax.plot(
    x,
    n2,
    color=COLOR_N2,
    **plot_kwargs_no_error
)

ax.set_xlabel("Time [h]")
ax.set_ylabel("Signal intensity")

format_time_axis(ax)
add_grid(ax)
add_horizontal_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=True)
save_plot(fig, "signal_H2_N2_110_horizontal_no_error.png")

# -------------------------------------------------
# 10. Initial N2 chamber, 1H concentration
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
add_grid(ax)
add_horizontal_label(ax)

add_combined_legend_below(fig, include_h2=False, include_n2=True)
save_plot(fig, "initial_N2_H1_concentration_H2_N2_110_horizontal.png")

# -------------------------------------------------
# 11. Initial H2 chamber, 1H concentration
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
add_grid(ax)
add_horizontal_label(ax)

add_combined_legend_below(fig, include_h2=True, include_n2=False)
save_plot(fig, "initial_H2_H1_concentration_H2_N2_110_horizontal.png")

# -------------------------------------------------
# 12. Optional N2 normalised fit plot
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
add_grid(ax)

ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    os.path.join(output_folder, "N2_exponential_fit_curvefit_H2_N2_110_horizontal.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08
)

plt.show()

# -------------------------------------------------
# 13. Save corrected plotting data
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
    "H2_N2_110_horizontal_plot_data_H1_concentration.xlsx"
)

df.to_excel(corrected_excel_path, index=False)

# -------------------------------------------------
# 14. Save fit metrics
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
    "fit_successful": fit_successful
}

metrics_df = pd.DataFrame([metrics])

metrics_path = os.path.join(
    output_folder,
    "N2_curve_fit_metrics_H2_N2_110_horizontal.xlsx"
)

metrics_df.to_excel(metrics_path, index=False)

# -------------------------------------------------
# 15. Print summary
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
print("Horizontal H2--N2 experiment. No background colour was added.")

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
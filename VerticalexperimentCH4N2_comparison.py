
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# =================================================
# Plot style
# =================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 8.5,
    "figure.dpi": 300
})

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

figsize = (5.8, 3.9)

# =================================================
# File paths
# =================================================
FILE_EXP_I = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion/CH4_diff_experiments/CH4_N2_120_diff45V_diffusion_results.xlsx"

FILE_EXP_II = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/Code/diffusion/CH4_diff_experiments/CH4_N2_120_diff45V2_diffusion_results_lang.xlsx"

OUTPUT_FOLDER = Path(FILE_EXP_II).parent / "CH4_N2_combined_redistribution_rate_percent"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =================================================
# Experiment settings
# =================================================
EXPERIMENTS = {
    "CH$_4$--N$_2$ I": {
        "file_path": FILE_EXP_I,
        "flip_scan": 262,
        "exclude_scans": [260],
        "roi_already_corrected": False,
        "color": "indianred",
        "linestyle": "-",
        "marker": "o"
    },
    "CH$_4$--N$_2$ II": {
        "file_path": FILE_EXP_II,
        "flip_scan": 127,
        "exclude_scans": [],
        "roi_already_corrected": True,
        "color": "darkslateblue",
        "linestyle": "--",
        "marker": "s"
    }
}

# =================================================
# Plot settings
# =================================================
TIME_TICK_STEP = 4

# "zoom" makes the early-time trend easier to see.
# "full" uses y-axis from 0 to 100%.
Y_AXIS_MODE = "zoom"

BACKGROUND_COLOR = "#d9e1ef"
BACKGROUND_ALPHA = 0.65

# =================================================
# Helper functions
# =================================================
def find_turning_time(df, flip_scan):
    """
    Finds the approximate set-up turning time as the midpoint between
    the flip scan and the next valid scan.
    """
    flip_time_series = df.loc[df["Scan"] == flip_scan, "Time_hours"]

    if len(flip_time_series) == 0:
        return None

    flip_time = float(flip_time_series.values[0])
    flip_idx = df.index[df["Scan"] == flip_scan][0]

    if flip_idx + 1 < len(df):
        next_time = float(df["Time_hours"].iloc[flip_idx + 1])
    else:
        next_time = flip_time

    return (flip_time + next_time) / 2


def load_and_process_experiment(name, settings):
    """
    Loads one experiment and calculates corrected CH4 and N2 signals,
    normalised 1H concentration, and one combined redistribution curve.
    """
    file_path = settings["file_path"]
    flip_scan = settings["flip_scan"]
    exclude_scans = settings["exclude_scans"]
    roi_already_corrected = settings["roi_already_corrected"]

    df = pd.read_excel(file_path)
    df = df.sort_values("Scan").reset_index(drop=True)

    df = df[~df["Scan"].isin(exclude_scans)].copy()

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
        subset=[
            "Scan",
            "Time_seconds",
            "ROI1_mean",
            "ROI2_mean",
            "ROI1_std",
            "ROI2_std"
        ]
    ).copy()

    df = df.sort_values("Scan").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid data left for {name}.")

    df["Time_seconds_plot"] = df["Time_seconds"] - df["Time_seconds"].iloc[0]
    df["Time_hours"] = df["Time_seconds_plot"] / 3600

    turning_time = find_turning_time(df, flip_scan)

    roi1 = df["ROI1_mean"].to_numpy(dtype=float)
    roi2 = df["ROI2_mean"].to_numpy(dtype=float)
    roi1_std = df["ROI1_std"].to_numpy(dtype=float)
    roi2_std = df["ROI2_std"].to_numpy(dtype=float)
    scans = df["Scan"].to_numpy()

    if roi_already_corrected:
        ch4 = roi1
        n2 = roi2
        ch4_std = roi1_std
        n2_std = roi2_std

    else:
        post_flip_mask = scans > flip_scan

        ch4 = np.where(post_flip_mask, roi2, roi1)
        n2 = np.where(post_flip_mask, roi1, roi2)

        ch4_std = np.where(post_flip_mask, roi2_std, roi1_std)
        n2_std = np.where(post_flip_mask, roi1_std, roi2_std)

    total_signal = ch4 + n2

    df["CH4_signal"] = ch4
    df["N2_signal"] = n2
    df["CH4_std"] = ch4_std
    df["N2_std"] = n2_std

    df["CH4_H1_concentration"] = ch4 / total_signal
    df["N2_H1_concentration"] = n2 / total_signal

    ch4_initial = df["CH4_H1_concentration"].iloc[0]
    n2_initial = df["N2_H1_concentration"].iloc[0]

    df["CH4_decrease"] = ch4_initial - df["CH4_H1_concentration"]
    df["N2_increase"] = df["N2_H1_concentration"] - n2_initial

    df["Combined_H1_redistribution"] = 0.5 * (
        df["CH4_decrease"] + df["N2_increase"]
    )

    df["Combined_H1_redistribution_percent"] = (
        df["Combined_H1_redistribution"] * 100
    )

    df["Experiment"] = name
    df["Turning_time_hours"] = turning_time

    return df, turning_time


def fit_linear_rate(df, y_col):
    """
    Fits y = a*t + b.
    If y is in percent, slope is in %-points per hour.
    """
    x = df["Time_hours"].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("At least two data points are needed for a linear fit.")

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, intercept, r2


def format_time_axis(ax, x_max):
    ax.set_xlim(-0.3, x_max + 0.5)
    ax.set_xticks(np.arange(0, x_max + TIME_TICK_STEP, TIME_TICK_STEP))


def format_y_axis(ax, y_values):
    if Y_AXIS_MODE == "full":
        ax.set_ylim(-5, 105)
        ax.set_yticks(np.arange(0, 101, 20))

    elif Y_AXIS_MODE == "zoom":
        y_min = np.nanmin(y_values)
        y_max = np.nanmax(y_values)

        padding = 0.3
        lower = min(0, y_min - padding)
        upper = y_max + padding

        if upper - lower < 1.5:
            mid = 0.5 * (upper + lower)
            lower = mid - 0.75
            upper = mid + 0.75

        ax.set_ylim(lower, upper)

    else:
        raise ValueError("Y_AXIS_MODE must be 'full' or 'zoom'.")


def add_background(ax, x_max):
    ax.axvspan(
        -0.3,
        x_max + 0.5,
        facecolor=BACKGROUND_COLOR,
        alpha=BACKGROUND_ALPHA,
        edgecolor="none",
        zorder=0
    )


def add_grid(ax):
    ax.grid(alpha=0.3)


def add_legend_below(fig, rate_results):
    handles = []

    for row in rate_results:
        name = row["Experiment"]
        settings = EXPERIMENTS[name]

        label = (
            fr"{name}: rate = {row['Rate_percent_points_per_h']:.2f} "
            fr"%-points h$^{{-1}}$"
        )

        handles.append(
            Line2D(
                [0], [0],
                color=settings["color"],
                linestyle=settings["linestyle"],
                marker=settings["marker"],
                markersize=3,
                linewidth=1.4,
                label=label
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=1,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6
    )


def save_plot(fig, filename, bottom=0.27):
    fig.subplots_adjust(bottom=bottom)
    fig.savefig(
        OUTPUT_FOLDER / filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08
    )
    plt.show()


# =================================================
# 1. Load both experiments
# =================================================
processed = {}
turning_times = {}

for name, settings in EXPERIMENTS.items():
    df_exp, turning_time = load_and_process_experiment(name, settings)
    processed[name] = df_exp
    turning_times[name] = turning_time

print("\nApproximate turning times:")
for name, turning_time in turning_times.items():
    if turning_time is not None:
        print(f"{name}: {turning_time:.2f} h")
    else:
        print(f"{name}: turning time not found")

first_turning_time = min(t for t in turning_times.values() if t is not None)

print(f"\nCommon comparison window: 0 to {first_turning_time:.2f} h")
print("Only data before the first set-up turning are included.\n")

# =================================================
# 2. Cut both experiments before first turning
# =================================================
processed_pre_turning = {}

for name, df_exp in processed.items():
    df_cut = df_exp[df_exp["Time_hours"] <= first_turning_time].copy()
    processed_pre_turning[name] = df_cut

    if len(df_cut) < 2:
        raise ValueError(
            f"{name} has fewer than two points before the first turning time."
        )

# =================================================
# 3. Linear rates based on both chambers
# =================================================
rate_results = []

for name, df_cut in processed_pre_turning.items():
    slope_percent, intercept_percent, r2 = fit_linear_rate(
        df_cut,
        "Combined_H1_redistribution_percent"
    )

    slope_fraction, intercept_fraction, r2_fraction = fit_linear_rate(
        df_cut,
        "Combined_H1_redistribution"
    )

    rate_results.append({
        "Experiment": name,
        "Start_time_h": df_cut["Time_hours"].min(),
        "End_time_h": df_cut["Time_hours"].max(),
        "Number_of_points": len(df_cut),

        "Rate_percent_points_per_h": slope_percent,
        "Intercept_percent": intercept_percent,
        "R2": r2,

        "Rate_h_minus_1": slope_fraction,
        "Rate_1e_minus_3_h_minus_1": slope_fraction * 1e3,
        "Intercept_fraction": intercept_fraction,
        "R2_fraction": r2_fraction,
    })

rate_results_df = pd.DataFrame(rate_results)

rates_path = OUTPUT_FOLDER / "CH4_N2_pre_turning_combined_redistribution_rates_percent.xlsx"
rate_results_df.to_excel(rates_path, index=False)

print("Linear rates based on both chambers")
print("-----------------------------------")
print(rate_results_df[[
    "Experiment",
    "End_time_h",
    "Number_of_points",
    "Rate_percent_points_per_h",
    "Rate_h_minus_1",
    "Rate_1e_minus_3_h_minus_1",
    "R2"
]].to_string(index=False))

if len(rate_results_df) == 2:
    r1 = rate_results_df["Rate_percent_points_per_h"].iloc[0]
    r2 = rate_results_df["Rate_percent_points_per_h"].iloc[1]

    diff = abs(r1 - r2)
    mean_rate = np.mean([r1, r2])
    rel_diff = diff / mean_rate * 100 if mean_rate > 0 else np.nan

    print("\nRate comparison between repeated experiments")
    print("--------------------------------------------")
    print(f"Rate difference: {diff:.3f} %-points h^-1")
    print(f"Relative difference: {rel_diff:.1f}%")

print(f"\nRates saved to:\n{rates_path}")

# =================================================
# 4. Plot: points + linear trendlines
# =================================================
fig, ax = plt.subplots(figsize=figsize)

all_y_values = []

for name, df_cut in processed_pre_turning.items():
    settings = EXPERIMENTS[name]

    x_data = df_cut["Time_hours"].to_numpy(dtype=float)
    y_data = df_cut["Combined_H1_redistribution_percent"].to_numpy(dtype=float)

    slope, intercept, r2 = fit_linear_rate(
        df_cut,
        "Combined_H1_redistribution_percent"
    )

    x_fit = np.linspace(x_data.min(), x_data.max(), 300)
    y_fit = slope * x_fit + intercept

    all_y_values.extend(y_data)
    all_y_values.extend(y_fit)

    # Data points
    ax.plot(
        x_data,
        y_data,
        color=settings["color"],
        marker=settings["marker"],
        linestyle="None",
        markersize=2.8,
        alpha=0.85,
        zorder=3
    )

    # Linear fit line
    ax.plot(
        x_fit,
        y_fit,
        color=settings["color"],
        linestyle=settings["linestyle"],
        linewidth=1.4,
        alpha=0.95,
        zorder=4
    )

ax.set_xlabel("Time [h]")
ax.set_ylabel(r"$^{1}$H signal redistribution [%]")

format_time_axis(ax, first_turning_time)
format_y_axis(ax, np.array(all_y_values))
add_background(ax, first_turning_time)
add_grid(ax)
add_legend_below(fig, rate_results)

save_plot(
    fig,
    "CH4_N2_pre_turning_combined_redistribution_percent_points_and_rates.png",
    bottom=0.27
)

# =================================================
# 5. Save processed pre-turning data
# =================================================
combined_pre_turning = pd.concat(
    processed_pre_turning.values(),
    ignore_index=True
)

processed_data_path = OUTPUT_FOLDER / "CH4_N2_pre_turning_combined_redistribution_processed_data_percent.xlsx"
combined_pre_turning.to_excel(processed_data_path, index=False)

print(f"\nProcessed pre-turning data saved to:\n{processed_data_path}")
print(f"\nPlot saved to:\n{OUTPUT_FOLDER}")

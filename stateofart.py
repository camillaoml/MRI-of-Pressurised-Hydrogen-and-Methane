import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Style settings
# ============================================================

plt.rcParams.update({
    "figure.figsize": (6.5, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,
    "mathtext.default": "regular",
})

output_dir = Path.cwd()

# ============================================================
# Data
# All diffusion coefficients are given in mm^2/s
# ============================================================

# This work
p_our = np.array([109.4, 101.0, 89.4, 80.9, 70.8, 61.1, 50.8, 40.2, 30.5, 20.2]) #9.9
D_our = np.array([0.164, 0.186, 0.209, 0.237, 0.268, 0.319, 0.387, 0.498, 0.69, 1.2 ]) #2.0
sigma_our = np.array([0.002, 0.001, 0.001, 0.002, 0.004, 0.006, 0.004, 0.008, 0.02, 0.2]) #0.3

# Dawson et al.
# Values converted from cm^2/s to mm^2/s where needed
p_dawson = np.array([53.158137, 67.705954, 95.560542, 127.276162, 163.404390])
D_dawson = np.array([0.384168865, 0.295307443, 0.205328938, 0.147557328, 0.112420])

# Takahashi
# Original D values were in cm^2/s, converted to mm^2/s by multiplying by 100
p_takahashi = np.array([25.027275, 50.865150, 66.874500, 104.567400, 153.507375])
D_takahashi = np.array([0.941, 0.458, 0.346, 0.203, 0.100])

# Harris
p_harris = np.array([
    45.3, 85.7, 104.9, 123.8, 162.2, 196.9, 204.8, 204.9,
    205.4, 205.5, 258.0, 258.1, 272.9, 328.9, 329.2,
    421.4, 567.1, 767.3, 772.7, 772.9, 773.2, 1076.3,
    1407.5, 1636.8
])
D_harris = np.array([
    0.438, 0.217, 0.169, 0.141, 0.105, 0.0856, 0.0825, 0.0830,
    0.0848, 0.0842, 0.0669, 0.0681, 0.0662, 0.0555, 0.0547,
    0.0464, 0.0389, 0.0314, 0.0317, 0.0313, 0.0324, 0.0269,
    0.0229, 0.0208
])

# ============================================================
# Plot function
# ============================================================

def make_plot(filename, xlim=None):
    fig, ax = plt.subplots()

    ax.errorbar(
        p_our, D_our, yerr=sigma_our,
        fmt="o", markersize=5,
        color="#1F4E79", ecolor="#1F4E79",
        elinewidth=0.8, capsize=2,
        label="This work"
    )

    ax.scatter(
        p_dawson, D_dawson,
        marker="s", s=42,
        facecolors="none", edgecolors="#D55E00",
        linewidths=1.2,
        label="Dawson et al. (1970)"
    )

    ax.scatter(
        p_harris, D_harris,
        marker="^", s=42,
        facecolors="none", edgecolors="#009E73",
        linewidths=1.2,
        label="Harris (1978)"
    )

    ax.scatter(
        p_takahashi, D_takahashi,
        marker="D", s=38,
        facecolors="none", edgecolors="#CC79A7",
        linewidths=1.2,
        label="Takahashi (1972)"
    )

    ax.set_xlabel("Pressure [bar]")
    ax.set_ylabel(r"$D$ [mm$^2$ s$^{-1}$]")

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_ylim(bottom=0)

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(output_dir / f"{filename}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")

    plt.show()


# ============================================================
# Make figures
# ============================================================

# Full plot including all Harris high-pressure points
make_plot("CH4_self_diffusion_comparison_full")

# Zoomed plot, probably best for thesis comparison
make_plot("CH4_self_diffusion_comparison_zoom", xlim=(0, 180))
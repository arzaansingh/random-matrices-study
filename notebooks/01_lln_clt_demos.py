"""01_lln_clt_demos.py

Generates the three figures used in Section 2 of the report:

  figures/lln_convergence.png       Weak/strong law of large numbers,
                                    sample mean vs n for normal and shifted-
                                    exponential samples sharing mean and var.

  figures/clt_exponential.png       Central limit theorem for exponentials,
                                    histograms of standardized sample means
                                    at n = 5, 30, 200 versus the standard
                                    normal density.

  figures/bivariate_normal.png      Bivariate normal density contours for
                                    correlations rho = 0, 0.5, 0.9.

Run:
    python3 01_lln_clt_demos.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from thesis_style import setup_plot, PALETTE

setup_plot()

# ============================================================================
# CONFIG
# ============================================================================

SEED = 2026
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================================
# 1. LLN convergence
# ============================================================================

def make_lln_figure() -> Path:
    """Sample-mean convergence for two distributions with the same mean and
    variance but different shapes."""
    n_max = 10_000

    # Normal(5, sigma^2 = 4): mean 5, var 4
    normal_samples = rng.normal(loc=5.0, scale=2.0, size=n_max)
    # Exponential with rate 1/2 has mean 2, var 4. Shift by 3 to get mean 5, var 4.
    exp_samples = rng.exponential(scale=2.0, size=n_max) + 3.0

    cum_mean_normal = np.cumsum(normal_samples) / np.arange(1, n_max + 1)
    cum_mean_exp = np.cumsum(exp_samples) / np.arange(1, n_max + 1)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    n_axis = np.arange(1, n_max + 1)
    ax.plot(n_axis, cum_mean_normal, color=PALETTE["def"], linewidth=1.0,
            label=r"Normal $\mathcal{N}(5,\,4)$ samples")
    ax.plot(n_axis, cum_mean_exp, color=PALETTE["accent"], linewidth=1.0,
            label=r"Shifted Exp samples (mean 5, var 4)")
    ax.axhline(5.0, color=PALETTE["thm"], linestyle="--", linewidth=1.0,
               label=r"Population mean $\mu = 5$")
    ax.set_xscale("log")
    ax.set_xlim(1, n_max)
    ax.set_ylim(3.5, 7.0)
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"Sample mean $\bar X_n$")
    ax.set_title("Sample mean converges to the population mean")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    path = FIGURES_DIR / "lln_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 2. CLT for exponentials
# ============================================================================

def make_clt_figure() -> Path:
    """Histograms of standardized sample means for iid Exp(1), at increasing n."""
    n_values = (5, 30, 200)
    R = 5000  # Monte Carlo replicates per panel

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)

    x_grid = np.linspace(-4.0, 4.0, 400)
    std_normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x_grid ** 2)

    for ax, n in zip(axes, n_values):
        # Exponential(1) has mean 1, variance 1.
        samples = rng.exponential(scale=1.0, size=(R, n))
        sample_means = samples.mean(axis=1)
        z = np.sqrt(n) * (sample_means - 1.0)

        ax.hist(z, bins=40, density=True, color=PALETTE["def"], alpha=0.75,
                edgecolor="white", linewidth=0.5,
                label=rf"Standardized $Z_{{n}}$, $R = {R}$")
        ax.plot(x_grid, std_normal_pdf, color=PALETTE["thm"], linewidth=1.4,
                label=r"$\mathcal{N}(0, 1)$ density")
        ax.set_xlim(-4.0, 4.0)
        ax.set_xlabel(r"$Z_n = \sqrt{n}(\bar X_n - 1)$")
        ax.set_title(rf"$n = {n}$")
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Empirical density")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()

    path = FIGURES_DIR / "clt_exponential.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 3. Bivariate normal density contours
# ============================================================================

def make_bvn_figure() -> Path:
    """Density contours of centered bivariate normals with three correlations."""
    rhos = (0.0, 0.5, 0.9)

    x = np.linspace(-3.5, 3.5, 200)
    y = np.linspace(-3.5, 3.5, 200)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)

    for ax, rho in zip(axes, rhos):
        det_sigma = 1.0 - rho ** 2
        z_quadratic = (X ** 2 - 2.0 * rho * X * Y + Y ** 2) / det_sigma
        density = np.exp(-0.5 * z_quadratic) / (2.0 * np.pi * np.sqrt(det_sigma))

        # Contour levels chosen by density quantiles for visual clarity.
        levels = np.linspace(density.max() * 0.02, density.max() * 0.95, 7)
        cs = ax.contour(X, Y, density, levels=levels,
                        colors=PALETTE["def"], linewidths=1.0)
        ax.contourf(X, Y, density, levels=levels,
                    cmap=mcolors.LinearSegmentedColormap.from_list(
                        "bvn", ["white", PALETTE["def"]]
                    ),
                    alpha=0.45)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_title(rf"$\rho = {rho}$")
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout()

    path = FIGURES_DIR / "bivariate_normal.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    print("Generating Section 2 figures...")
    for fn in (make_lln_figure, make_clt_figure, make_bvn_figure):
        path = fn()
        print(f"  Saved: {path.name}")
    print("Done.")

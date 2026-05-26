"""02_random_vectors_and_linalg.py

Generates two figures for Section 2.6-2.7 of the report:

  figures/joint_distribution_supports.png   Three joint distributions on R^2
                                            with the same marginals: independent
                                            exponentials, X_1 = X_2 a.s., and a
                                            bivariate normal with rho = 0.7.

  figures/quadratic_form_level_sets.png     Level sets of x^T A x for three
                                            real symmetric 2x2 matrices:
                                            positive definite (ellipses),
                                            positive semidefinite with one
                                            zero eigenvalue (parallel strips),
                                            indefinite (hyperbolas).

Run:
    python3 02_random_vectors_and_linalg.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from thesis_style import setup_plot, PALETTE

setup_plot()

SEED = 2026
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Joint distribution supports
# ============================================================================

def make_supports_figure() -> Path:
    """Three panels: independent exponentials, X_1 = X_2 a.s., bivariate normal."""
    rng = np.random.default_rng(SEED)
    n = 2000

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

    # Panel A: independent exponentials, both rate 1.
    x1 = rng.exponential(scale=1.0, size=n)
    x2 = rng.exponential(scale=1.0, size=n)
    axes[0].scatter(x1, x2, s=6, alpha=0.4, color=PALETTE["def"])
    axes[0].set_xlim(0, 6)
    axes[0].set_ylim(0, 6)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].set_title(r"Independent Exp$(1)$")

    # Panel B: X_2 = X_1 a.s.
    x = rng.exponential(scale=1.0, size=n)
    axes[1].scatter(x, x, s=6, alpha=0.4, color=PALETTE["def"])
    axes[1].plot([0, 6], [0, 6], color=PALETTE["accent"], linestyle="--",
                 linewidth=0.8, label=r"diagonal $x_2 = x_1$")
    axes[1].set_xlim(0, 6)
    axes[1].set_ylim(0, 6)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    axes[1].set_title(r"$X_2 = X_1$ a.s.")
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)

    # Panel C: bivariate normal with rho = 0.7.
    rho = 0.7
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mean = np.zeros(2)
    samples = rng.multivariate_normal(mean, cov, size=n)
    axes[2].scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.4, color=PALETTE["def"])
    axes[2].axhline(0, color=PALETTE["accent"], linewidth=0.4)
    axes[2].axvline(0, color=PALETTE["accent"], linewidth=0.4)
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")
    axes[2].set_title(rf"Bivariate normal, $\rho = {rho}$")

    fig.tight_layout()
    path = FIGURES_DIR / "joint_distribution_supports.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 2. Quadratic form level sets
# ============================================================================

def make_quadratic_forms_figure() -> Path:
    """Three panels: PD ellipses, PSD parallel strips, indefinite hyperbolas."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

    x = np.linspace(-3.0, 3.0, 400)
    y = np.linspace(-3.0, 3.0, 400)
    X, Y = np.meshgrid(x, y)

    cmap_pos = mcolors.LinearSegmentedColormap.from_list(
        "pos", ["white", PALETTE["def"]]
    )

    # Panel A: positive definite. A1 = [[1, -1/2], [-1/2, 1]]; eigs = 3/2, 1/2.
    Q_pd = X**2 - X * Y + Y**2
    levels_pd = [0.25, 0.75, 1.5, 3.0, 5.0]
    axes[0].contour(X, Y, Q_pd, levels=levels_pd, colors=PALETTE["def"], linewidths=1.0)
    axes[0].contourf(X, Y, Q_pd, levels=levels_pd, cmap=cmap_pos, alpha=0.35)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].set_title(r"PD: $\lambda_+ = 3/2,\ \lambda_- = 1/2$")
    axes[0].grid(True, linestyle=":", alpha=0.4)

    # Panel B: PSD with one zero eigenvalue. A2 = [[1, 1], [1, 1]]; eigs = 2, 0.
    Q_psd = (X + Y) ** 2  # = x^T A2 x for A2 above (with factor 1)
    levels_psd = [0.5, 1.5, 3.0, 5.0, 8.0]
    axes[1].contour(X, Y, Q_psd, levels=levels_psd, colors=PALETTE["def"], linewidths=1.0)
    axes[1].contourf(X, Y, Q_psd, levels=levels_psd, cmap=cmap_pos, alpha=0.35)
    # The null eigenvector is along (1, -1)/sqrt(2): plot the null line.
    axes[1].plot([-3, 3], [3, -3], color=PALETTE["accent"], linestyle="--",
                 linewidth=0.8, label=r"null eigenvector")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    axes[1].set_title(r"PSD: $\lambda_+ = 2, \lambda_- = 0$")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.4)

    # Panel C: indefinite. A3 = [[1, -2], [-2, 1]]; eigs = 3, -1.
    Q_ind = X**2 - 4.0 * X * Y + Y**2
    # Plot both positive and negative level sets.
    levels_pos = [0.5, 1.5, 3.0, 5.0]
    levels_neg = [-5.0, -3.0, -1.5, -0.5]
    cmap_neg = mcolors.LinearSegmentedColormap.from_list(
        "neg", ["white", PALETTE["accent"]]
    )
    axes[2].contour(X, Y, Q_ind, levels=levels_pos, colors=PALETTE["def"], linewidths=1.0)
    axes[2].contour(X, Y, Q_ind, levels=levels_neg, colors=PALETTE["accent"], linewidths=1.0)
    axes[2].contourf(X, Y, Q_ind, levels=levels_pos, cmap=cmap_pos, alpha=0.3)
    axes[2].contourf(X, Y, Q_ind, levels=levels_neg, cmap=cmap_neg, alpha=0.3)
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")
    axes[2].set_title(r"Indefinite: $\lambda_+ = 3, \lambda_- = -1$")
    axes[2].grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    path = FIGURES_DIR / "quadratic_form_level_sets.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating §2.6-§2.7 figures...")
    for fn in (make_supports_figure, make_quadratic_forms_figure):
        path = fn()
        print(f"  Saved: {path.name}")
    print("Done.")

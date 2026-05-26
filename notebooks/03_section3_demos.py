"""03_section3_demos.py

Generates four figures for Section 3 of the report:

  figures/wishart_2d_eigenvalues.png    Scatter + gap histogram for 2D
                                        Wishart eigenvalues, showing
                                        repulsion.

  figures/low_dim_qq_plots.png          Normal Q-Q plots for the top
                                        sample eigenvalue in the
                                        repeated (Sigma = I_2) and
                                        distinct (Sigma = diag(1, 2))
                                        cases.

  figures/mp_bulk_with_density.png      Histogram of sample eigenvalues
                                        at p=500, n=1000 (y=0.5) with
                                        the Marchenko-Pastur density
                                        overlaid.

  figures/tw_edge_simulation.png        Histogram of standardized top
                                        sample eigenvalue (Johnstone
                                        centering / scaling) at three
                                        n values, showing convergence
                                        to TW_1 shape.

Run:
    python3 03_section3_demos.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from thesis_style import setup_plot, PALETTE

setup_plot()

SEED = 2026
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Wishart 2D eigenvalues: scatter + gap histogram
# ============================================================================

def make_wishart_2d_figure() -> Path:
    rng = np.random.default_rng(SEED)
    n = 100
    trials = 5000

    eigs = np.zeros((trials, 2))
    for t in range(trials):
        X = rng.normal(size=(2, n))
        S = X @ X.T  # unnormalized: gives W_2(n, I_2)
        eigs[t] = np.linalg.eigvalsh(S)  # ascending order

    lam1 = eigs[:, 0]
    lam2 = eigs[:, 1]
    gap = lam2 - lam1

    fig, (ax_scat, ax_hist) = plt.subplots(1, 2, figsize=(11, 4))

    # Scatter
    ax_scat.scatter(lam1, lam2, s=4, alpha=0.3, color=PALETTE["def"])
    lo, hi = float(lam1.min()), float(lam2.max())
    ax_scat.plot([lo, hi], [lo, hi], "--", color=PALETTE["accent"],
                 linewidth=0.9, label=r"diagonal $\lambda_1 = \lambda_2$")
    ax_scat.set_xlabel(r"$\lambda_1$ (smaller)")
    ax_scat.set_ylabel(r"$\lambda_2$ (larger)")
    ax_scat.set_title(rf"$W_2(n, I_2)$ eigenvalues, $n = {n}$, {trials} trials")
    ax_scat.legend(loc="upper left", frameon=False, fontsize=9)
    ax_scat.set_aspect("equal", adjustable="box")

    # Gap histogram
    ax_hist.hist(gap, bins=60, density=True, color=PALETTE["def"], alpha=0.75,
                 edgecolor="white", linewidth=0.4)
    ax_hist.set_xlabel(r"Gap $\lambda_2 - \lambda_1$")
    ax_hist.set_ylabel("Empirical density")
    ax_hist.set_title("Gap distribution: density vanishes at 0")

    fig.tight_layout()
    path = FIGURES_DIR / "wishart_2d_eigenvalues.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 2. Low-dim Q-Q plots: repeated vs distinct top eigenvalue
# ============================================================================

def _qqdata(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (theoretical_quantiles, ordered_sample) for a Normal Q-Q plot."""
    n = len(sample)
    sample_sorted = np.sort(sample)
    probs = (np.arange(1, n + 1) - 0.5) / n  # mid-rank plotting positions
    theoretical = stats.norm.ppf(probs)
    return theoretical, sample_sorted


def make_lowdim_qq_figure() -> Path:
    rng = np.random.default_rng(SEED)
    n = 500
    trials = 5000

    # Repeated case: Sigma = I_2
    z_rep = np.empty(trials)
    for t in range(trials):
        X = rng.normal(size=(2, n))
        S = (X @ X.T) / n
        z_rep[t] = np.sqrt(n) * (np.linalg.eigvalsh(S)[-1] - 1.0)
    z_rep = (z_rep - z_rep.mean()) / z_rep.std(ddof=1)

    # Distinct case: Sigma = diag(1, 2)
    z_dist = np.empty(trials)
    for t in range(trials):
        Z = rng.normal(size=(2, n))
        Z[1, :] *= np.sqrt(2.0)
        S = (Z @ Z.T) / n
        z_dist[t] = np.sqrt(n) * (np.linalg.eigvalsh(S)[-1] - 2.0)
    z_dist = (z_dist - z_dist.mean()) / z_dist.std(ddof=1)

    fig, (ax_rep, ax_dist) = plt.subplots(1, 2, figsize=(11, 4))

    for ax, sample, title in (
        (ax_rep,  z_rep,  r"$\Sigma = I_2$ (repeated eigenvalue)"),
        (ax_dist, z_dist, r"$\Sigma = \mathrm{diag}(1, 2)$ (distinct eigenvalues)"),
    ):
        tq, sq = _qqdata(sample)
        ax.scatter(tq, sq, s=4, alpha=0.5, color=PALETTE["def"])
        lim_lo, lim_hi = -4.0, 4.0
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--",
                color=PALETTE["accent"], linewidth=0.9,
                label="reference: $y = x$")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Standard normal quantiles")
        ax.set_ylabel("Standardized sample quantiles")
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / "low_dim_qq_plots.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 3. Marchenko-Pastur bulk
# ============================================================================

def _mp_density(x: np.ndarray, y: float) -> np.ndarray:
    lam_minus = (1.0 - np.sqrt(y)) ** 2
    lam_plus = (1.0 + np.sqrt(y)) ** 2
    out = np.zeros_like(x, dtype=float)
    inside = (x >= lam_minus) & (x <= lam_plus)
    xi = x[inside]
    out[inside] = np.sqrt((lam_plus - xi) * (xi - lam_minus)) / (2.0 * np.pi * y * xi)
    return out


def make_mp_bulk_figure() -> Path:
    rng = np.random.default_rng(SEED)
    n, y_target = 1000, 0.5
    p = int(round(y_target * n))
    X = rng.normal(size=(p, n))
    S = (X @ X.T) / n
    eigs = np.linalg.eigvalsh(S)

    y = p / n
    lam_minus = (1 - np.sqrt(y)) ** 2
    lam_plus = (1 + np.sqrt(y)) ** 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(eigs, bins=50, density=True, color=PALETTE["def"], alpha=0.7,
            edgecolor="white", linewidth=0.4,
            label=f"Empirical histogram ({p} eigenvalues)")

    x_grid = np.linspace(0.001, lam_plus * 1.05, 600)
    ax.plot(x_grid, _mp_density(x_grid, y), color=PALETTE["thm"], linewidth=1.7,
            label=r"Marchenko-Pastur density")
    ax.axvline(lam_minus, color=PALETTE["accent"], linestyle=":", linewidth=0.9)
    ax.axvline(lam_plus, color=PALETTE["accent"], linestyle=":", linewidth=0.9,
               label=r"MP edges $\lambda_\pm$")

    ax.set_xlim(0, lam_plus * 1.1)
    ax.set_xlabel(r"Eigenvalue $\lambda$")
    ax.set_ylabel("Empirical density")
    ax.set_title(rf"Marchenko-Pastur bulk at $p = {p}$, $n = {n}$, $y = {y:g}$")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / "mp_bulk_with_density.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 4. Tracy-Widom edge: standardized top eigenvalue at three n's
# ============================================================================

def _johnstone_constants(n: int, p: int) -> tuple[float, float]:
    """Johnstone (2001) mu_np and sigma_np for the null Wishart edge."""
    a = np.sqrt(n - 1) + np.sqrt(p)
    mu = a ** 2 / n
    sigma = (a / n) * (1.0 / np.sqrt(n - 1) + 1.0 / np.sqrt(p)) ** (1.0 / 3.0)
    return float(mu), float(sigma)


def make_tw_edge_figure() -> Path:
    rng = np.random.default_rng(SEED)
    n_values = (100, 400, 1600)
    y_target = 0.5
    trials = 1500

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    x_grid = np.linspace(-5, 5, 400)
    gauss_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid ** 2)

    for ax, n in zip(axes, n_values):
        p = int(round(y_target * n))
        mu, sigma = _johnstone_constants(n, p)
        z = np.empty(trials)
        for t in range(trials):
            X = rng.normal(size=(p, n))
            S = (X @ X.T) / n
            lam_top = np.linalg.eigvalsh(S)[-1]
            z[t] = (lam_top - mu) / sigma

        ax.hist(z, bins=40, density=True, color=PALETTE["def"], alpha=0.75,
                edgecolor="white", linewidth=0.4,
                label=fr"Standardized $\lambda_{{\max}}$, $n = {n}$")
        ax.plot(x_grid, gauss_pdf, color=PALETTE["accent"], linewidth=1.0,
                linestyle="--", label=r"$\mathcal{N}(0,1)$ density (for comparison)")
        ax.axvline(-1.21, color=PALETTE["thm"], linestyle=":", linewidth=0.8,
                   label=r"TW$_1$ mean $\approx -1.21$")
        ax.set_xlim(-5, 5)
        ax.set_xlabel(r"$(\lambda_{\max}(\Sigmahat_n) - \mu_{np}) / \sigma_{np}$".replace(r"\Sigmahat", r"\widehat{\Sigma}"))
        ax.set_title(rf"$n = {n}$, $p = {p}$ ($y = {p/n:g}$)")
        if n == n_values[0]:
            ax.set_ylabel("Empirical density")
        ax.legend(loc="upper left", frameon=False, fontsize=7)

    fig.tight_layout()
    path = FIGURES_DIR / "tw_edge_simulation.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating Section 3 figures...")
    for fn in (make_wishart_2d_figure, make_lowdim_qq_figure,
               make_mp_bulk_figure, make_tw_edge_figure):
        path = fn()
        print(f"  Saved: {path.name}")
    print("Done.")

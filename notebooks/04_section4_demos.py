"""04_section4_demos.py

Generates figures for Section 4 (Spiked covariance and the BBP transition).

  figures/spiked_spectrum.png    Two-panel histogram of the ESD of Sigmahat_n
                                 under a STRONG spike (alpha = 2.5) and a
                                 WEAK spike (alpha = 1.4), at y = 1/2. The
                                 strong panel shows an outlier separating
                                 from the MP bulk; the weak panel shows the
                                 spike absorbed at the upper edge.

  figures/spike_location_map.png Plot of the spike-location map
                                 Psi(alpha) = alpha + y * alpha / (alpha - 1)
                                 on (1, infty) for y = 1/2, with the
                                 minimum at alpha = 1 + sqrt(y) marked.
                                 The minimum value Psi(1 + sqrt(y)) equals
                                 the MP upper edge (1 + sqrt(y))^2.

Run:
    python3 04_section4_demos.py

These follow the spiked-model setup of Exercise 4 in the project repo
(MATH_Spectral_Clustering_Exercise_4.ipynb), which in turn implements
Corollary 11.4 of Yao-Zheng-Bai (2015).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thesis_style import setup_plot, PALETTE

setup_plot()

SEED = 4040
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Spiked-model helpers (mirroring Exercise 4)
# ============================================================================

def psi(alpha: float, y: float) -> float:
    """Spike-location map Psi(alpha) = alpha + y * alpha / (alpha - 1)."""
    return alpha + y * alpha / (alpha - 1.0)


def mp_upper_edge(y: float) -> float:
    return (1.0 + np.sqrt(y)) ** 2


def mp_lower_edge(y: float) -> float:
    return (1.0 - np.sqrt(y)) ** 2


def mp_density(lam: np.ndarray, y: float) -> np.ndarray:
    """Marchenko-Pastur density at points lam, for aspect ratio y in (0, 1)."""
    lam_m = mp_lower_edge(y)
    lam_p = mp_upper_edge(y)
    out = np.zeros_like(lam, dtype=float)
    inside = (lam >= lam_m) & (lam <= lam_p)
    li = lam[inside]
    out[inside] = np.sqrt((lam_p - li) * (li - lam_m)) / (2.0 * np.pi * y * li)
    return out


def simulate_spiked_eigenvalues(spikes, p, n, rng):
    """One realization of Sigmahat_n = (1/n) X X^T for X = Sigma^{1/2} Z,
    Z standard Gaussian, with Sigma = diag(spikes_1,...,spikes_m, 1, ..., 1).

    Returns eigenvalues in DESCENDING order.
    """
    spikes = np.asarray(spikes, dtype=float)
    m = len(spikes)
    if m >= p:
        raise ValueError("Need m < p.")
    sigma_diag = np.ones(p)
    sigma_diag[:m] = spikes
    Z = rng.standard_normal((p, n))
    X = np.sqrt(sigma_diag)[:, None] * Z
    S = (X @ X.T) / n
    return np.linalg.eigvalsh(S)[::-1]


# ============================================================================
# Figure 1: spectrum under a strong spike vs a weak spike
# ============================================================================

def make_spiked_spectrum_figure() -> Path:
    """Two-panel histogram of the empirical spectrum of Sigmahat_n with
    alpha > 1 + sqrt(y) (strong, outlier) and 1 < alpha <= 1 + sqrt(y)
    (weak, absorbed)."""
    rng = np.random.default_rng(SEED)
    n, y = 1000, 0.5
    p = int(round(y * n))
    lam_plus = mp_upper_edge(y)
    lam_minus = mp_lower_edge(y)
    alpha_strong = 2.5
    alpha_weak = 1.4
    thr = 1.0 + np.sqrt(y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, alpha, label, color_outlier in (
        (axes[0], alpha_strong, "strong", PALETTE["thm"]),
        (axes[1], alpha_weak,   "weak",   PALETTE["accent"]),
    ):
        eigs = simulate_spiked_eigenvalues([alpha], p=p, n=n, rng=rng)
        lam_max = eigs[0]
        bulk = eigs[1:]  # everything except the (potential) outlier

        # Histogram of the non-top eigenvalues (the bulk) for clarity.
        bins = np.linspace(0.0, max(lam_plus * 1.1, lam_max * 1.05), 60)
        ax.hist(bulk, bins=bins, density=True, color=PALETTE["def"],
                alpha=0.65, edgecolor="white", linewidth=0.3,
                label=fr"Bulk: $\lambda_2,\dots,\lambda_p$ ({p - 1} values)")

        # Theoretical MP density overlay (the null bulk shape).
        x_grid = np.linspace(max(lam_minus * 0.5, 0.001),
                             lam_plus * 1.0, 600)
        ax.plot(x_grid, mp_density(x_grid, y),
                color=PALETTE["thm"], linewidth=1.5,
                label=r"Marchenko-Pastur density")

        # MP edge marker.
        ax.axvline(lam_plus, color=PALETTE["accent"], linestyle=":",
                   linewidth=1.0,
                   label=fr"MP edge $\lambda_+ \approx {lam_plus:.2f}$")

        # The top eigenvalue: vertical line at lam_max, styled by regime.
        ax.axvline(lam_max, color=color_outlier, linewidth=1.6,
                   label=fr"$\lambda_{{\max}} = {lam_max:.2f}$")

        # Optionally annotate where Psi(alpha) predicts the outlier (only
        # interpretable for the strong case; for the weak case the
        # prediction lies in the supercritical branch and is wrong).
        if alpha > thr:
            ax.axvline(psi(alpha, y), color=PALETTE["link"], linestyle="--",
                       linewidth=1.0,
                       label=fr"$\Psi(\alpha) \approx {psi(alpha, y):.2f}$")

        ax.set_xlim(0.0, max(lam_plus * 1.1, lam_max * 1.07))
        ax.set_xlabel(r"Eigenvalue $\lambda$")
        if label == "strong":
            ax.set_ylabel("Empirical / theoretical density")
        ax.set_title(
            (rf"\textbf{{{label.capitalize()} spike}} "
             rf"$\alpha = {alpha}$, threshold $1 + \sqrt{{y}} \approx "
             rf"{thr:.2f}$").replace(r"\textbf", "")
        )
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(
        rf"Spectrum of $\widehat{{\Sigma}}_n$ at $n = {n}$, $p = {p}$ "
        rf"($y = {y:g}$) under a single spike $\alpha$",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "spiked_spectrum.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Figure 2: the spike-location map Psi(alpha)
# ============================================================================

def make_spike_location_map_figure() -> Path:
    y = 0.5
    thr = 1.0 + np.sqrt(y)
    lam_plus = mp_upper_edge(y)

    # Two pieces of the curve: subcritical (1, thr] (Psi decreasing) and
    # supercritical [thr, infty) (Psi increasing).
    a_sub = np.linspace(1.02, thr, 200)
    a_sup = np.linspace(thr, 5.0, 400)
    psi_sub = psi(a_sub, y)
    psi_sup = psi(a_sup, y)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(a_sup, psi_sup, color=PALETTE["def"], linewidth=1.8,
            label=r"$\Psi(\alpha) = \alpha + y\alpha/(\alpha-1)$")
    ax.plot(a_sub, psi_sub, color=PALETTE["def"], linewidth=1.4,
            linestyle="--",
            label=r"$\Psi(\alpha)$ on the subcritical branch (does not realize $\lambda_{\max}$)")

    # The minimum at alpha = 1 + sqrt(y) and the value (1 + sqrt(y))^2.
    ax.axhline(lam_plus, color=PALETTE["accent"], linestyle=":",
               linewidth=1.0,
               label=fr"MP upper edge $\lambda_+ = (1+\sqrt{{y}})^2 \approx {lam_plus:.2f}$")
    ax.axvline(thr, color=PALETTE["thm"], linestyle=":", linewidth=1.0)
    ax.scatter([thr], [lam_plus], s=60, color=PALETTE["thm"], zorder=5,
               label=fr"Threshold $\alpha = 1+\sqrt{{y}} \approx {thr:.2f}$")

    # 45-degree line y = alpha for context (Psi is approximately alpha for
    # large alpha; departs near the threshold).
    a_ref = np.linspace(0.5, 5.0, 100)
    ax.plot(a_ref, a_ref, color="lightgray", linestyle="-",
            linewidth=0.7, label=r"reference: $\Psi = \alpha$")

    # Shade the supercritical region (alpha > threshold).
    ax.axvspan(thr, 5.0, color=PALETTE["vocab"], alpha=0.35,
               label="Supercritical: outlier above the bulk")

    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(2.0, 6.5)
    ax.set_xlabel(r"Population spike $\alpha$")
    ax.set_ylabel(r"$\Psi(\alpha)$")
    ax.set_title(rf"Spike-location map at $y = {y}$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    path = FIGURES_DIR / "spike_location_map.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating Section 4 figures...")
    for fn in (make_spiked_spectrum_figure, make_spike_location_map_figure):
        path = fn()
        print(f"  Saved: {path.name}")
    print("Done.")

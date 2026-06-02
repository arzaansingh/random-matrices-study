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
from scipy import stats

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

    # Shade the fundamental-spike region (alpha > threshold).
    ax.axvspan(thr, 5.0, color=PALETTE["vocab"], alpha=0.35,
               label="Fundamental spikes (outlier above the bulk)")

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


# ============================================================================
# Figure 3: m=1 phase transition -- almost-sure convergence to Psi(alpha)
#           (fundamental) or lambda_+ (non-fundamental) as n grows.
# ============================================================================

def make_phase_transition_figure() -> Path:
    """Smoothed m=1 simulation illustrating Yao-Zheng-Bai Cor 11.4 specialised
    to Johnstone. For each n in a geometric sequence we run R replications
    and plot the mean top sample eigenvalue with a 10-90 percentile band,
    for a fundamental spike (alpha=2.5, lambda_max should approach Psi(alpha))
    and a non-fundamental spike (alpha=1.4, lambda_max should approach
    (1+sqrt(y))^2).
    """
    rng = np.random.default_rng(SEED + 1)
    y = 0.5
    n_values = [100, 200, 400, 800, 1600]
    R = 100
    alpha_strong = 2.5
    alpha_weak = 1.4

    psi_strong = psi(alpha_strong, y)
    edge = mp_upper_edge(y)

    strong_mean, strong_q10, strong_q90 = [], [], []
    weak_mean, weak_q10, weak_q90 = [], [], []

    for n in n_values:
        p = int(round(y * n))
        s_vals = np.empty(R)
        w_vals = np.empty(R)
        for r in range(R):
            eigs_s = simulate_spiked_eigenvalues([alpha_strong], p=p, n=n, rng=rng)
            eigs_w = simulate_spiked_eigenvalues([alpha_weak], p=p, n=n, rng=rng)
            s_vals[r] = eigs_s[0]
            w_vals[r] = eigs_w[0]
        strong_mean.append(s_vals.mean())
        strong_q10.append(np.quantile(s_vals, 0.10))
        strong_q90.append(np.quantile(s_vals, 0.90))
        weak_mean.append(w_vals.mean())
        weak_q10.append(np.quantile(w_vals, 0.10))
        weak_q90.append(np.quantile(w_vals, 0.90))

    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Fundamental spike (alpha=2.5).
    ax.plot(n_values, strong_mean, marker="o", color=PALETTE["def"], linewidth=1.4,
            label=rf"Fundamental spike $\alpha = {alpha_strong}$ (mean over $R = {R}$)")
    ax.fill_between(n_values, strong_q10, strong_q90, color=PALETTE["def"],
                    alpha=0.18, label=r"10--90\% band")
    ax.axhline(psi_strong, color=PALETTE["def"], linestyle="--", linewidth=1.0,
               label=rf"prediction $\Psi(\alpha) \approx {psi_strong:.2f}$")

    # Non-fundamental spike (alpha=1.4).
    ax.plot(n_values, weak_mean, marker="s", color=PALETTE["accent"], linewidth=1.4,
            label=rf"Non-fundamental spike $\alpha = {alpha_weak}$")
    ax.fill_between(n_values, weak_q10, weak_q90, color=PALETTE["accent"], alpha=0.20)
    ax.axhline(edge, color=PALETTE["accent"], linestyle=":", linewidth=1.0,
               label=rf"MP edge $\lambda_+ \approx {edge:.2f}$")

    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel(r"Sample size $n$ (log scale, $p = n/2$)")
    ax.set_ylabel(r"$\lambda_{\max}(\widehat{\Sigma}_n)$")
    ax.set_title(rf"Convergence of $\lambda_{{\max}}$ at $y = 1/2$, $m = 1$")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    path = FIGURES_DIR / "spike_phase_transition.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Figure 4: top-eigenvalue fluctuations for fundamental and non-fundamental spikes
# (Yao-Zheng-Bai Theorem 11.11 / Feral-Peche Theorem 1.6 vs Tracy-Widom edge).
# ============================================================================

def _johnstone_constants(n: int, p: int) -> tuple[float, float]:
    """Johnstone (2001) finite-sample TW edge centering/scaling, also used
    in Exercise 5; matches the constants in section 3 of the thesis."""
    a = np.sqrt(n - 1) + np.sqrt(p)
    mu = a ** 2 / n
    sigma = (a / n) * (1.0 / np.sqrt(n - 1) + 1.0 / np.sqrt(p)) ** (1.0 / 3.0)
    return float(mu), float(sigma)


def _yzb_variance_simple_fundamental(alpha: float, y: float) -> float:
    """Asymptotic variance sigma_alpha^2 from YZB Thm 11.11 / Eq. (11.36),
    Gaussian case (beta_y = 0), m_k = 1:

        sigma_alpha^2 = 2 alpha^2 psi'(alpha)
                      = 2 alpha^2 [(alpha - 1)^2 - y] / (alpha - 1)^2.
    """
    return 2.0 * alpha ** 2 * (1.0 - y / (alpha - 1.0) ** 2)


def make_top_eigenvalue_distributions_figure() -> Path:
    """Side-by-side histogram of the raw top sample eigenvalue at
    y = 1/2, n = 800, R = 1000 trials, for a fundamental spike (alpha=2.5)
    and a non-fundamental spike (alpha=1.4). Vertical lines mark the
    asymptotic outlier location Psi(alpha_strong), the asymptotic
    Marchenko-Pastur edge lambda_+, and Johnstone's finite-sample edge
    mu_{n,p}. This mirrors the final figure of Exercise 5 of the project
    notebook (raw_top_eigenvalue_strong_vs_weak_with_finite_edge.png) and
    is the qualitative entry point to the fluctuation discussion.
    """
    rng = np.random.default_rng(SEED + 4)
    y = 0.5
    n, p = 800, 400
    R = 1000

    alpha_strong = 2.5
    alpha_weak = 1.4
    psi_strong = psi(alpha_strong, y)
    lam_plus = mp_upper_edge(y)
    mu_np, _ = _johnstone_constants(n, p)

    lam_strong = np.empty(R)
    lam_weak = np.empty(R)
    for r in range(R):
        eigs_s = simulate_spiked_eigenvalues([alpha_strong], p=p, n=n, rng=rng)
        eigs_w = simulate_spiked_eigenvalues([alpha_weak], p=p, n=n, rng=rng)
        lam_strong[r] = eigs_s[0]
        lam_weak[r] = eigs_w[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(lam_strong, bins=35, density=True, color=PALETTE["def"],
            alpha=0.55, edgecolor="white", linewidth=0.3,
            label=rf"Fundamental spike $\alpha = {alpha_strong}$")
    ax.hist(lam_weak, bins=35, density=True, color=PALETTE["accent"],
            alpha=0.55, edgecolor="white", linewidth=0.3,
            label=rf"Non-fundamental spike $\alpha = {alpha_weak}$")

    ax.axvline(psi_strong, color=PALETTE["def"], linestyle="--", linewidth=1.6,
               label=rf"$\Psi(\alpha_{{\mathrm{{strong}}}}) \approx {psi_strong:.2f}$")
    ax.axvline(lam_plus, color=PALETTE["accent"], linestyle=":", linewidth=1.6,
               label=rf"Asymptotic edge $\lambda_+ \approx {lam_plus:.2f}$")
    ax.axvline(mu_np, color=PALETTE["thm"], linestyle="-.", linewidth=1.6,
               label=rf"Finite-sample edge $\mu_{{n,p}} \approx {mu_np:.2f}$")

    ax.set_xlabel(r"Largest sample eigenvalue $\lambda_{\max}(\widehat{\Sigma}_n)$")
    ax.set_ylabel("Empirical density")
    ax.set_title(
        rf"Top-eigenvalue distributions at $y = 1/2$, $n = {n}$, $p = {p}$, "
        rf"$R = {R}$ trials per spike"
    )
    ax.legend(loc="upper center", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    path = FIGURES_DIR / "top_eigenvalue_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_spike_fluctuations_figure() -> Path:
    """Two-row, two-column comparison of top-eigenvalue fluctuations for one
    fundamental (alpha=2.5) and one non-fundamental (alpha=1.4) spike at
    y = 1/2, n = 1500, R = 2000 trials.

    Top row (fundamental spike): histogram of sqrt(n)(lambda_max - Psi(alpha))
    against the predicted Gaussian limit N(0, sigma_alpha^2) from
    YZB Thm 11.11 / Feral-Peche Thm 1.6; Q-Q plot against N(0, 1).

    Bottom row (non-fundamental spike): histogram of the Johnstone-standardised
    statistic (lambda_max - mu_np) / sigma_np against the TW1 density; Q-Q
    plot against N(0, 1) showing the non-Gaussian shape.
    """
    try:
        from TracyWidom import TracyWidom
        tw1 = TracyWidom(beta=1)
        have_tw = True
    except ImportError:
        have_tw = False

    rng = np.random.default_rng(SEED + 3)
    y = 0.5
    n, p = 1500, 750
    R = 2000

    alpha_strong = 2.5
    alpha_weak = 1.4
    psi_strong = psi(alpha_strong, y)
    lam_plus = mp_upper_edge(y)
    mu_np, sigma_np = _johnstone_constants(n, p)
    sigma2_alpha = _yzb_variance_simple_fundamental(alpha_strong, y)
    sigma_alpha = np.sqrt(sigma2_alpha)

    # Simulate top sample eigenvalues for both spike values.
    lam_strong = np.empty(R)
    lam_weak = np.empty(R)
    for r in range(R):
        eigs_s = simulate_spiked_eigenvalues([alpha_strong], p=p, n=n, rng=rng)
        eigs_w = simulate_spiked_eigenvalues([alpha_weak], p=p, n=n, rng=rng)
        lam_strong[r] = eigs_s[0]
        lam_weak[r] = eigs_w[0]

    # Standardised statistics.
    z_strong = np.sqrt(n) * (lam_strong - psi_strong)         # ~ N(0, sigma_alpha^2)
    z_weak = (lam_weak - mu_np) / sigma_np                    # ~ TW1 at the edge

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- TOP-LEFT: histogram of z_strong against N(0, sigma_alpha^2) ---
    ax = axes[0, 0]
    bins = np.linspace(z_strong.min() - 0.5, z_strong.max() + 0.5, 50)
    ax.hist(z_strong, bins=bins, density=True, color=PALETTE["def"], alpha=0.7,
            edgecolor="white", linewidth=0.3,
            label=fr"$\sqrt{{n}}(\lambda_{{\max}} - \Psi(\alpha))$, $R = {R}$")
    grid = np.linspace(z_strong.min() - 0.5, z_strong.max() + 0.5, 400)
    gauss = (1.0 / (sigma_alpha * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (grid / sigma_alpha) ** 2)
    ax.plot(grid, gauss, color=PALETTE["thm"], linewidth=1.6,
            label=fr"$\mathcal{{N}}(0,\, \sigma_\alpha^2)$, $\sigma_\alpha^2 \approx {sigma2_alpha:.2f}$")
    ax.set_xlabel(r"$\sqrt{n}\,(\lambda_{\max} - \Psi(\alpha))$")
    ax.set_ylabel("Empirical density")
    ax.set_title(rf"Fundamental spike $\alpha = {alpha_strong}$: histogram vs predicted Gaussian")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # --- TOP-RIGHT: Q-Q of z_strong / sigma_alpha against N(0,1) ---
    ax = axes[0, 1]
    sample = np.sort(z_strong / sigma_alpha)
    probs = (np.arange(1, R + 1) - 0.5) / R
    theo_q = stats.norm.ppf(probs)
    ax.scatter(theo_q, sample, s=4, alpha=0.5, color=PALETTE["def"])
    lim_lo = min(theo_q.min(), sample.min()) - 0.3
    lim_hi = max(theo_q.max(), sample.max()) + 0.3
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=PALETTE["accent"],
            linestyle="--", linewidth=0.9, label=r"reference: $y = x$")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"Standard normal quantiles")
    ax.set_ylabel(r"Standardised $\sqrt{n}(\lambda_{\max} - \Psi(\alpha))/\sigma_\alpha$")
    ax.set_title(r"Fundamental spike: Q-Q vs $\mathcal{N}(0,1)$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # --- BOTTOM-LEFT: histogram of z_weak against TW1 ---
    ax = axes[1, 0]
    bins = np.linspace(-6, 4, 50)
    ax.hist(z_weak, bins=bins, density=True, color=PALETTE["def"], alpha=0.7,
            edgecolor="white", linewidth=0.3,
            label=fr"$(\lambda_{{\max}} - \mu_{{np}})/\sigma_{{np}}$, $R = {R}$")
    grid = np.linspace(-6, 4, 600)
    gauss = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * grid ** 2)
    if have_tw:
        ax.plot(grid, tw1.pdf(grid), color=PALETTE["thm"], linewidth=1.6,
                label=r"$\mathrm{TW}_1$ density")
    ax.plot(grid, gauss, color=PALETTE["accent"], linewidth=1.0, linestyle="--",
            label=r"$\mathcal{N}(0, 1)$ (for contrast)")
    ax.set_xlabel(r"$(\lambda_{\max} - \mu_{np}) / \sigma_{np}$")
    ax.set_ylabel("Empirical density")
    ax.set_title(rf"Non-fundamental spike $\alpha = {alpha_weak}$: edge-scaled vs $\mathrm{{TW}}_1$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # --- BOTTOM-RIGHT: Q-Q of z_weak against N(0,1) ---
    ax = axes[1, 1]
    sample = np.sort(z_weak)
    probs = (np.arange(1, R + 1) - 0.5) / R
    theo_q = stats.norm.ppf(probs)
    ax.scatter(theo_q, sample, s=4, alpha=0.5, color=PALETTE["def"])
    lim_lo = min(theo_q.min(), sample.min()) - 0.3
    lim_hi = max(theo_q.max(), sample.max()) + 0.3
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=PALETTE["accent"],
            linestyle="--", linewidth=0.9, label=r"reference: $y = x$")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"Standard normal quantiles")
    ax.set_ylabel(r"$(\lambda_{\max} - \mu_{np})/\sigma_{np}$ quantiles")
    ax.set_title(r"Non-fundamental spike: Q-Q vs $\mathcal{N}(0,1)$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / "spike_fluctuations.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Figure 5: fluctuation phase transition swept across alpha. Mirrors the
# Section 4.2 alpha-sweep but on the sqrt(n) fluctuation scale: shows the
# theoretical band ±1.282 sigma_alpha (Theorem 4.3) opening up only above the
# threshold while the non-fundamental side stays pinned near zero on the
# Gaussian scale (the true edge fluctuations live on the smaller p^{-2/3}
# scale).
# ============================================================================

def make_fluctuation_phase_transition_figure() -> Path:
    """Sweep alpha across the threshold and plot the centered, sqrt(n)-scaled
    top sample eigenvalue, with the predicted ±1.282 sigma_alpha band on the
    fundamental side and the empirical 10-90% band overlaid. The visual
    content of the phase transition for fluctuations: the predicted
    Gaussian band collapses to width zero at alpha = 1+sqrt(y), and stays
    closed below it where the relevant scale is no longer sqrt(n)."""
    rng = np.random.default_rng(SEED + 2)
    y = 0.5
    n, p = 800, 400
    R = 80
    thr = 1.0 + np.sqrt(y)
    lam_plus = mp_upper_edge(y)

    alpha_grid = np.concatenate([
        np.linspace(1.05, thr, 14, endpoint=False),
        np.linspace(thr, 3.5, 24),
    ])
    alpha_grid = np.unique(np.round(alpha_grid, 4))

    all_alpha, all_z = [], []
    means_z, q10_z, q90_z = [], [], []
    for alpha in alpha_grid:
        center = psi(alpha, y) if alpha > thr else lam_plus
        lam_vals = np.empty(R)
        for r in range(R):
            eigs = simulate_spiked_eigenvalues([alpha], p=p, n=n, rng=rng)
            lam_vals[r] = eigs[0]
        z_vals = np.sqrt(n) * (lam_vals - center)
        all_alpha.extend([alpha] * R)
        all_z.extend(z_vals.tolist())
        means_z.append(z_vals.mean())
        q10_z.append(np.quantile(z_vals, 0.10))
        q90_z.append(np.quantile(z_vals, 0.90))
    means_z = np.asarray(means_z)
    q10_z = np.asarray(q10_z)
    q90_z = np.asarray(q90_z)

    # Theoretical 1-sigma and 10-90% bands above threshold.
    a_dense = np.linspace(thr * 1.0005, 3.6, 400)
    sigma_theory = np.sqrt(2.0 * a_dense ** 2 * (1.0 - y / (a_dense - 1.0) ** 2))
    q90_gauss = 1.2816  # Phi^{-1}(0.9)
    band_hi = +q90_gauss * sigma_theory
    band_lo = -q90_gauss * sigma_theory

    fig, ax = plt.subplots(figsize=(9, 5.6))

    # Translucent scatter of all trials.
    ax.scatter(all_alpha, all_z, s=6, color=PALETTE["def"], alpha=0.15,
               label=rf"Individual trials ($R = {R}$ per $\alpha$)")

    # Empirical 10-90% band.
    ax.fill_between(alpha_grid, q10_z, q90_z, color=PALETTE["def"], alpha=0.25,
                    label=r"10--90\% empirical band")

    # Empirical mean curve (should be approximately zero by construction).
    ax.plot(alpha_grid, means_z, marker="o", markersize=3.0,
            color=PALETTE["def"], linewidth=1.2,
            label=r"Empirical mean of $\sqrt{n}(\lambda_{\max} - c_\alpha)$")

    # Theoretical 10-90% Gaussian band on the fundamental side.
    ax.plot(a_dense, band_hi, color=PALETTE["thm"], linewidth=1.8,
            label=r"Predicted 10--90\% Gaussian band $\pm 1.28\,\sigma_\alpha$")
    ax.plot(a_dense, band_lo, color=PALETTE["thm"], linewidth=1.8)
    ax.fill_between(a_dense, band_lo, band_hi, color=PALETTE["thm"], alpha=0.08)

    # Zero reference.
    ax.axhline(0, color="black", linewidth=0.6)

    # Threshold marker.
    ax.axvline(thr, color=PALETTE["accent"], linestyle="--", linewidth=1.0,
               label=rf"Threshold $\alpha = 1+\sqrt{{y}} \approx {thr:.2f}$")

    # Shade the fundamental region for consistency with Figure 4.2.
    ax.axvspan(thr, 3.6, color=PALETTE["vocab"], alpha=0.15)

    ax.set_xlim(1.0, 3.55)
    y_band = max(abs(q90_z).max(), abs(q10_z).max(), abs(band_hi).max()) * 1.1
    ax.set_ylim(-y_band, y_band)
    ax.set_xlabel(r"Population spike $\alpha$")
    ax.set_ylabel(
        r"$\sqrt{n}\,(\lambda_{\max} - c_\alpha)$, "
        r"$c_\alpha = \Psi(\alpha)$ or $\lambda_+$"
    )
    ax.set_title(
        rf"Fluctuation phase transition at $y = 1/2$, "
        rf"$n = {n}$, $p = {p}$, $m = 1$"
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    path = FIGURES_DIR / "spike_fluctuation_phase_transition.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Figure 6: the critical window. Two figures supporting the §4.3 discussion of
# what happens to the top eigenvalue as the spike approaches alpha_c = 1+sqrt(y)
# (ported from Exercise 5, restyled, with our notation: c -> y, S_n -> Sigmahat;
# raw eigenvalues throughout, no edge rescaling).
# ============================================================================

def make_critical_violin_sweep_figure() -> Path:
    """Violin sweep of the RAW top eigenvalue across the threshold.

    Shows how the whole distribution of lambda_max(Sigmahat_n) changes with the
    spike. Below alpha_c it is tight, pinned at the MP edge, and asymmetric (the
    Tracy-Widom edge law); above alpha_c it lifts along Psi(alpha), broadens, and
    becomes symmetric (Gaussian). Near alpha_c the violins overlap because the
    location barely moves (Psi'(alpha_c) = 0); they separate only well away from
    the threshold. One picture for location, spread, and shape at once.
    """
    rng = np.random.default_rng(SEED + 5)
    y = 0.5
    n, p = 800, 400
    R = 800
    thr = 1.0 + np.sqrt(y)
    lam_plus = mp_upper_edge(y)

    offsets = np.array([-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30])
    alpha_grid = thr + offsets

    data = []
    for alpha in alpha_grid:
        lam_vals = np.empty(R)
        for r in range(R):
            lam_vals[r] = simulate_spiked_eigenvalues([alpha], p=p, n=n, rng=rng)[0]
        data.append(lam_vals)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    parts = ax.violinplot(data, positions=alpha_grid, widths=0.035,
                          showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set(facecolor=PALETTE["def"], edgecolor=PALETTE["def"], alpha=0.45)
    parts["cmedians"].set(color=PALETTE["accent"], linewidth=1.2)

    # Theoretical location: lambda_+ below the threshold, Psi(alpha) above.
    a_sub = np.linspace(alpha_grid.min() - 0.02, thr, 50)
    a_sup = np.linspace(thr, alpha_grid.max() + 0.02, 120)
    ax.plot(a_sub, np.full_like(a_sub, lam_plus), color=PALETTE["thm"], linewidth=1.6)
    ax.plot(a_sup, psi(a_sup, y), color=PALETTE["thm"], linewidth=1.6,
            label=r"Predicted location ($\lambda_+$, then $\Psi(\alpha)$)")
    ax.axhline(lam_plus, color=PALETTE["accent"], linestyle="--", linewidth=1.0,
               label=rf"MP edge $\lambda_+ \approx {lam_plus:.2f}$")
    ax.axvline(thr, color=PALETTE["thm"], linestyle=":", linewidth=1.0,
               label=rf"Threshold $\alpha_c = 1+\sqrt{{y}} \approx {thr:.2f}$")

    ax.set_xlabel(r"Population spike $\alpha$")
    ax.set_ylabel(r"Top sample eigenvalue $\lambda_{\max}(\widehat{\Sigma}_n)$")
    ax.set_title(
        rf"Distribution of $\lambda_{{\max}}$ across the threshold "
        rf"($y = 1/2$, $n = {n}$, $p = {p}$, $R = {R}$)"
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    path = FIGURES_DIR / "critical_violin_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_critical_merge_densities_figure() -> Path:
    """Raw top-eigenvalue densities just below vs just above the threshold, at a
    SMALL and a LARGE distance delta = |alpha - alpha_c|.

    Near the threshold (small delta) the below- and above-distributions nearly
    coincide, because the outlier location has barely left the edge. Far from the
    threshold (large delta) the supercritical side has lifted away and the two
    are clearly distinct. No edge rescaling: the horizontal axis is the
    eigenvalue itself, so the merge is read directly off the raw distributions.
    """
    rng = np.random.default_rng(SEED + 6)
    y = 0.5
    n, p = 800, 400
    R = 800
    thr = 1.0 + np.sqrt(y)
    lam_plus = mp_upper_edge(y)

    def lam_sample(alpha):
        v = np.empty(R)
        for r in range(R):
            v[r] = simulate_spiked_eigenvalues([alpha], p=p, n=n, rng=rng)[0]
        return v

    near, far = 0.05, 0.30
    style = {
        ("near", "below"): (thr - near, PALETTE["def"], "-",
                            rf"$\alpha_c - {near:g}$ (below)"),
        ("near", "above"): (thr + near, PALETTE["def"], "--",
                            rf"$\alpha_c + {near:g}$ (above)"),
        ("far", "below"): (thr - far, PALETTE["accent"], "-",
                           rf"$\alpha_c - {far:g}$ (below)"),
        ("far", "above"): (thr + far, PALETTE["accent"], "--",
                           rf"$\alpha_c + {far:g}$ (above)"),
    }

    grid = np.linspace(2.70, 3.15, 400)
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Exactly-critical distribution, drawn first (underneath) as the reference.
    kde_c = stats.gaussian_kde(lam_sample(thr))
    ax.plot(grid, kde_c(grid), color=PALETTE["thm"], linestyle="-", linewidth=2.6,
            alpha=0.9, label=r"$\alpha_c$ (critical)")

    for (alpha, color, ls, lab) in style.values():
        kde = stats.gaussian_kde(lam_sample(alpha))
        ax.plot(grid, kde(grid), color=color, linestyle=ls, linewidth=1.8, label=lab)

    ax.axvline(lam_plus, color=PALETTE["thm"], linestyle=":", linewidth=1.0,
               label=rf"MP edge $\lambda_+ \approx {lam_plus:.2f}$")
    ax.set_xlabel(r"Top sample eigenvalue $\lambda_{\max}(\widehat{\Sigma}_n)$")
    ax.set_ylabel("Estimated density")
    ax.set_title(
        rf"Just below vs just above $\alpha_c$, near ($\delta = {near:g}$) "
        rf"and far ($\delta = {far:g}$) "
        rf"($y = 1/2$, $n = {n}$, $p = {p}$, $R = {R}$)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    path = FIGURES_DIR / "critical_merge_densities.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Figure 7: eigenvector alignment (Section 4.4). Yao-Zheng-Bai Thm 11.5 /
# Cor 11.6: the squared overlap |<u_1, e_1>|^2 of the top sample eigenvector
# with the population spike direction converges to rho(alpha, y), zero below
# the threshold. Two figures: (a) alignment turning on as alpha grows at fixed
# y; (b) the effect of the aspect ratio y (larger y needs a larger alpha to
# reach the same alignment).
# ============================================================================

def rho_overlap(alpha, y):
    """Limiting squared overlap |<u_1, e_1>|^2 = d^2 = alpha*Psi'(alpha)/Psi(alpha)
    for a simple Johnstone spike (YZB Cor. 11.6):

        rho = (1 - y/(alpha-1)^2) / (1 + y/(alpha-1)),   alpha > 1 + sqrt(y),

    and 0 at or below the threshold.
    """
    alpha = np.asarray(alpha, dtype=float)
    thr = 1.0 + np.sqrt(y)
    r = (1.0 - y / (alpha - 1.0) ** 2) / (1.0 + y / (alpha - 1.0))
    return np.where(alpha > thr, np.maximum(r, 0.0), 0.0)


def simulate_top_overlap(alpha, p, n, rng, R):
    """R replications of |<u_1, e_1>|^2 for Sigma = diag(alpha, 1, ..., 1).
    Since e_1 is the first coordinate axis, the overlap is the square of the
    first entry of the top sample eigenvector."""
    sqrt_diag = np.ones(p)
    sqrt_diag[0] = np.sqrt(alpha)
    out = np.empty(R)
    for r in range(R):
        Z = rng.standard_normal((p, n))
        X = sqrt_diag[:, None] * Z
        S = (X @ X.T) / n
        w, V = np.linalg.eigh(S)
        out[r] = V[0, np.argmax(w)] ** 2
    return out


def make_eigenvector_alignment_figure() -> Path:
    """Mean squared overlap |<u_1, e_1>|^2 against the spike alpha at y = 1/2,
    with the theoretical rho(alpha, y) curve and the threshold. Below the
    threshold the overlap is ~0 (the top eigenvector is noise); above it the
    overlap turns on and climbs toward 1."""
    rng = np.random.default_rng(SEED + 7)
    y = 0.5
    n, p = 800, 400
    R = 250
    thr = 1.0 + np.sqrt(y)

    alpha_grid = np.unique(np.round(np.concatenate([
        np.linspace(1.10, thr - 0.02, 6),
        np.linspace(thr + 0.02, 3.5, 11),
    ]), 4))

    means = np.empty(len(alpha_grid))
    sds = np.empty(len(alpha_grid))
    for i, alpha in enumerate(alpha_grid):
        ov = simulate_top_overlap(float(alpha), p=p, n=n, rng=rng, R=R)
        means[i] = ov.mean()
        sds[i] = ov.std(ddof=1)

    alpha_fine = np.linspace(1.02, 3.6, 500)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(alpha_fine, rho_overlap(alpha_fine, y), color=PALETTE["thm"],
            linewidth=1.8, label=r"Theory $\rho(\alpha, y) = \alpha\Psi'(\alpha)/\Psi(\alpha)$")
    ax.errorbar(alpha_grid, means, yerr=sds, fmt="o", markersize=4,
                color=PALETTE["def"], capsize=3, linewidth=1.0,
                label=rf"Simulation mean $\pm$ sd ($R = {R}$)")
    ax.axvline(thr, color=PALETTE["accent"], linestyle="--", linewidth=1.0,
               label=rf"Threshold $\alpha_c = 1+\sqrt{{y}} \approx {thr:.2f}$")
    ax.axhline(1.0, color=PALETTE["accent"], linestyle=":", linewidth=0.9)

    ax.set_xlim(1.0, 3.55)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"Population spike $\alpha$")
    ax.set_ylabel(r"Squared alignment $|\langle \widehat{u}_1, e_1\rangle|^2$")
    ax.set_title(rf"Eigenvector alignment turns on at the threshold "
                 rf"($y = 1/2$, $n = {n}$, $p = {p}$)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    path = FIGURES_DIR / "eigenvector_alignment.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_eigenvector_dimension_effect_figure() -> Path:
    """Squared overlap against alpha for several aspect ratios y. Theory curves
    (dense) with simulation means (sparse points) overlaid. A horizontal target
    line shows the compensation question: to reach a fixed alignment, a larger y
    requires a larger alpha."""
    rng = np.random.default_rng(SEED + 8)
    n = 700
    R = 150
    y_values = [0.25, 0.5, 0.75, 1.0]
    colors = [PALETTE["def"], PALETTE["link"], PALETTE["accent"], PALETTE["thm"]]
    markers = ["o", "s", "^", "D"]

    alpha_fine = np.linspace(1.05, 4.0, 500)
    alpha_pts = np.linspace(1.2, 3.9, 8)
    target = 0.5

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for y, color, mk in zip(y_values, colors, markers):
        p = int(round(y * n))
        ax.plot(alpha_fine, rho_overlap(alpha_fine, y), color=color,
                linewidth=1.7, label=rf"$y = {y}$")
        pts = np.empty(len(alpha_pts))
        for i, alpha in enumerate(alpha_pts):
            pts[i] = simulate_top_overlap(float(alpha), p=p, n=n, rng=rng, R=R).mean()
        ax.scatter(alpha_pts, pts, color=color, marker=mk, s=22, zorder=5)

    ax.axhline(target, color="black", linestyle=":", linewidth=1.0,
               label=rf"target alignment $= {target}$")

    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"Population spike $\alpha$")
    ax.set_ylabel(r"Squared alignment $|\langle \widehat{u}_1, e_1\rangle|^2$")
    ax.set_title(rf"Larger aspect ratio $y$ needs a larger spike "
                 rf"for the same alignment ($n = {n}$, $R = {R}$)")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    path = FIGURES_DIR / "eigenvector_dimension_effect.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating Section 4 figures...")
    # NOTE: make_fluctuation_phase_transition_figure() is defined above but
    # not generated here. It will be re-enabled once §4.3 is extended with
    # a dedicated phase-transition discussion for the fluctuation regime.
    for fn in (make_spiked_spectrum_figure,
               make_spike_location_map_figure,
               make_phase_transition_figure,
               make_top_eigenvalue_distributions_figure,
               make_spike_fluctuations_figure,
               make_critical_violin_sweep_figure,
               make_critical_merge_densities_figure,
               make_eigenvector_alignment_figure,
               make_eigenvector_dimension_effect_figure):
        path = fn()
        print(f"  Saved: {path.name}")
    print("Done.")

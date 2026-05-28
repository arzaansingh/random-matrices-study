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
# 0z. Sample covariance geometry: data cloud + covariance ellipse + arrows
# ============================================================================

def make_geometry_figure() -> Path:
    """One-panel figure for §3.1: 2D data cloud from N(0, Sigma) with the
    population covariance ellipse and eigenvector arrows overlaid, plus
    numerical eigenvalue labels."""
    from matplotlib.patches import Ellipse

    rng = np.random.default_rng(SEED)
    n = 500

    Sigma = np.array([[4.0, 1.5], [1.5, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Z = rng.normal(size=(n, 2))
    X = Z @ L.T

    evals, evecs = np.linalg.eigh(Sigma)        # ascending: evals[0] < evals[1]
    lam_max, lam_min = float(evals[-1]), float(evals[0])
    v_max, v_min = evecs[:, -1], evecs[:, 0]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Data scatter
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.35, color=PALETTE["def"],
               label=fr"Sample of $n = {n}$ data points")

    # Population 2-sigma covariance ellipse: axes 2*sqrt(lambda_i)
    angle_deg = np.degrees(np.arctan2(v_max[1], v_max[0]))
    width = 2 * 2 * np.sqrt(lam_max)
    height = 2 * 2 * np.sqrt(lam_min)
    ellipse = Ellipse((0, 0), width, height, angle=angle_deg,
                      fill=False, edgecolor=PALETTE["thm"], linewidth=2.0,
                      label=r"Population covariance ellipse ($2\sigma$)")
    ax.add_patch(ellipse)

    # Eigenvectors as arrows scaled by 2*sqrt(eigenvalue)
    for v, lam in ((v_max, lam_max), (v_min, lam_min)):
        endpoint = v * 2 * np.sqrt(lam)
        ax.annotate("", xy=tuple(endpoint), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["link"], lw=2.5))
    ax.plot([], [], color=PALETTE["link"], lw=2.5,
            label=r"Eigenvectors of $\Sigma$ (length $2\sqrt{\lambda_i}$)")

    # Numerical eigenvalue labels near arrowheads
    end_max = v_max * 2 * np.sqrt(lam_max)
    end_min = v_min * 2 * np.sqrt(lam_min)
    ax.text(end_max[0] * 1.05 + 0.2, end_max[1] * 1.05 + 0.2,
            rf"$\lambda_1 = {lam_max:.2f}$",
            color=PALETTE["link"], fontsize=11)
    ax.text(end_min[0] * 1.7 - 0.2, end_min[1] * 1.7 + 0.5,
            rf"$\lambda_2 = {lam_min:.2f}$",
            color=PALETTE["link"], fontsize=11)

    ax.axhline(0, color=PALETTE["accent"], linewidth=0.4)
    ax.axvline(0, color=PALETTE["accent"], linewidth=0.4)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Geometry of the covariance matrix $\Sigma$")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / "sample_covariance_geometry.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 0a. Scatter with principal axes (motivates §3.1)
# ============================================================================

def make_scatter_axes_figure() -> Path:
    """Two panels of n=400 samples in R^2 with population and sample
    principal axes overlaid. Left: Sigma = I (circular). Right:
    Sigma = diag(4, 1) (elongated). Illustrates what the SCM
    eigendecomposition captures geometrically."""
    rng = np.random.default_rng(SEED)
    n = 400

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    cases = [
        (np.eye(2), r"$\Sigma = I_2$"),
        (np.diag([4.0, 1.0]), r"$\Sigma = \mathrm{diag}(4, 1)$"),
    ]

    for ax, (Sigma, label) in zip(axes, cases):
        L = np.linalg.cholesky(Sigma)
        Z = rng.normal(size=(n, 2))
        X = Z @ L.T

        # Sample covariance (mean-zero formula since population mean is 0)
        S = (X.T @ X) / n

        # Eigendecompositions
        evals_pop, evecs_pop = np.linalg.eigh(Sigma)
        evals_sam, evecs_sam = np.linalg.eigh(S)

        # Plot data
        ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.35, color=PALETTE["def"],
                   label="Data")

        # Plot population principal axes (scaled by sqrt eigenvalue)
        for i in range(2):
            v = evecs_pop[:, i] * np.sqrt(evals_pop[i]) * 2.0
            ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["thm"],
                                        lw=1.8))
        ax.plot([], [], color=PALETTE["thm"], lw=1.8,
                label=r"Population axes (eigenvectors of $\Sigma$)")

        # Plot sample principal axes (scaled by sqrt sample eigenvalue), dashed
        for i in range(2):
            v = evecs_sam[:, i] * np.sqrt(evals_sam[i]) * 2.0
            ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=PALETTE["link"],
                                        lw=1.5, linestyle="--"))
        ax.plot([], [], color=PALETTE["link"], lw=1.5, linestyle="--",
                label=r"Sample axes (eigenvectors of $\widehat{\Sigma}_n$)")

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(label)
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / "scatter_with_principal_axes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 0b. SCM consistency (motivates §3.2 Theorem 3.2)
# ============================================================================

def make_scm_consistency_figure() -> Path:
    """Plot ||Sigmahat_n - Sigma||_F vs n on a log-log axis for several p,
    with the theoretical 1/sqrt(n) reference rate overlaid. Demonstrates
    entrywise SLLN convergence visually."""
    rng = np.random.default_rng(SEED)

    n_values = np.unique(np.round(np.logspace(1.5, 4.0, 20)).astype(int))
    trials = 40
    p_values = (2, 5, 20)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette_cycle = [PALETTE["def"], PALETTE["link"], PALETTE["accent"]]

    for p, col in zip(p_values, palette_cycle):
        Sigma = np.eye(p)
        errs = np.zeros(len(n_values))
        for j, n in enumerate(n_values):
            errs_trial = np.empty(trials)
            for t in range(trials):
                X = rng.normal(size=(p, n))
                S = (X @ X.T) / n
                errs_trial[t] = np.linalg.norm(S - Sigma, ord="fro")
            errs[j] = errs_trial.mean()
        ax.loglog(n_values, errs, marker="o", color=col, linewidth=1.2,
                  markersize=4, label=rf"$p = {p}$ (mean over {trials} trials)")

    # Reference 1/sqrt(n) line
    ref_n = n_values
    ref_err = ref_n.astype(float) ** (-0.5)
    # Anchor the reference at the median observed value to make the slope visible
    ax.loglog(ref_n, ref_err * 5.0, "--", color=PALETTE["thm"], linewidth=1.0,
              label=r"Reference slope $n^{-1/2}$")

    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"$\|\widehat{\Sigma}_n - \Sigma\|_F$ (Frobenius error)")
    ax.set_title(r"Sample covariance consistency: $\widehat{\Sigma}_n \to I_p$")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    path = FIGURES_DIR / "scm_consistency.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 0c. Wishart 2D joint eigenvalue density heatmap (theoretical)
# ============================================================================

def make_wishart_density_heatmap_figure() -> Path:
    """Heatmap of the theoretical joint density of (lambda_1, lambda_2)
    for W_2(n, I_2), showing the (lambda_2 - lambda_1) repulsion factor
    that creates a 'dead zone' near the diagonal."""
    from scipy.special import gammaln
    n = 100

    # Joint density: f(l1, l2) = C * (l1 l2)^((n-3)/2) * exp(-(l1+l2)/2)
    #                            * (l2 - l1) on l1 <= l2.
    # Normalizing constant (Yao-Zheng-Bai or direct calculation):
    # For W_p(n, I_p) eigenvalue joint density,
    # f(l1,...,lp) = (1 / Z) * prod l_i^((n-p-1)/2) * exp(-sum l_i / 2)
    #                       * prod_{i<j} (l_j - l_i).
    # We just plot the density up to constant scaling (the heatmap shape
    # is what matters).

    grid = np.linspace(50, 200, 250)
    L1, L2 = np.meshgrid(grid, grid)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_dens = ((n - 3) / 2.0) * (np.log(L1) + np.log(L2)) \
                   - 0.5 * (L1 + L2) + np.log(np.maximum(L2 - L1, 0))
    dens = np.exp(log_dens - np.nanmax(log_dens))
    dens[L1 > L2] = np.nan  # mask the half-plane lambda_1 > lambda_2

    fig, ax = plt.subplots(figsize=(6.5, 5))
    pcm = ax.pcolormesh(L1, L2, dens, shading="auto", cmap="Blues", vmin=0)
    ax.plot([grid.min(), grid.max()], [grid.min(), grid.max()], "--",
            color=PALETTE["accent"], linewidth=1.0, label=r"diagonal $\lambda_2 = \lambda_1$")
    ax.set_xlabel(r"$\lambda_1$ (smaller eigenvalue)")
    ax.set_ylabel(r"$\lambda_2$ (larger eigenvalue)")
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(grid.min(), grid.max())
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(rf"Joint eigenvalue density for $W_2({n}, I_2)$ (theoretical, normalized)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("density (relative)")

    fig.tight_layout()
    path = FIGURES_DIR / "wishart_density_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


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
    # Pool enough realizations to accumulate ~10,000 eigenvalues.
    n_real = 10000 // p + 1
    eigs_list = []
    for _ in range(n_real):
        X = rng.normal(size=(p, n))
        S = (X @ X.T) / n
        eigs_list.append(np.linalg.eigvalsh(S))
    eigs = np.concatenate(eigs_list)

    y = p / n
    lam_minus = (1 - np.sqrt(y)) ** 2
    lam_plus = (1 + np.sqrt(y)) ** 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(eigs, bins=80, density=True, color=PALETTE["def"], alpha=0.7,
            edgecolor="white", linewidth=0.4,
            label=f"Empirical histogram ({eigs.size:,} eigenvalues)")

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


# ============================================================================
# 5. MP law across varying aspect ratios c = p/n
# ============================================================================

def make_mp_varying_c_figure() -> Path:
    """Four-panel grid showing MP density for c in {0.1, 0.5, 1, 2}.

    Each panel pools eigenvalues from multiple independent realizations to
    accumulate at least 10,000 eigenvalues, giving a clean histogram match
    against the theoretical MP density.
    """
    rng = np.random.default_rng(SEED)
    n = 2000
    target_eigs = 10000  # minimum non-zero eigenvalues per panel
    c_values = (0.1, 0.5, 1.0, 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for ax, c in zip(axes.flat, c_values):
        p = int(round(c * n))
        y = p / n
        lam_minus = (1.0 - np.sqrt(y)) ** 2
        lam_plus = (1.0 + np.sqrt(y)) ** 2

        # Pool eigenvalues across enough realizations to get target_eigs total.
        # For y > 1 we keep only the nonzero eigenvalues (rank deficiency).
        per_real = p if y <= 1 else min(p, n)
        n_real = max(1, target_eigs // per_real + 1)
        pooled = []
        for _ in range(n_real):
            X = rng.normal(size=(p, n))
            S = (X @ X.T) / n
            eigs = np.linalg.eigvalsh(S)
            if y > 1:
                eigs = eigs[eigs > 1e-8]
            pooled.append(eigs)
        pooled = np.concatenate(pooled)

        bin_lo = max(0.0, lam_minus * 0.5) if y <= 1 else max(0.0, lam_minus)
        bins = np.linspace(bin_lo, lam_plus * 1.05, 80)
        ax.hist(pooled, bins=bins, density=True, color=PALETTE["def"],
                alpha=0.7, edgecolor="white", linewidth=0.3,
                label=f"Empirical ({pooled.size:,} eigenvalues)")

        x_grid = np.linspace(max(lam_minus, 0.001) if y <= 1 else lam_minus * 1.001,
                             lam_plus * 1.0, 600)
        density = _mp_density(x_grid, y)
        ax.plot(x_grid, density, color=PALETTE["thm"], linewidth=1.5,
                label=rf"$\mathrm{{MP}}_{{y={y:g}}}$ density")
        ax.axvline(lam_minus, color=PALETTE["accent"], linestyle=":",
                   linewidth=0.7)
        ax.axvline(lam_plus, color=PALETTE["accent"], linestyle=":",
                   linewidth=0.7)
        if y > 1:
            ax.axvline(0.0, color=PALETTE["accent"], linewidth=1.6,
                       alpha=0.85,
                       label=fr"Atom at 0, mass $= 1 - 1/y = {1 - 1/y:.2f}$")

        ax.set_xlim(bin_lo - 0.05, lam_plus * 1.08)
        ax.set_xlabel(r"Eigenvalue $\lambda$")
        ax.set_ylabel("Empirical density")
        ax.set_title(rf"$y = p/n = {c:g}$ ($p = {p}$, $n = {n}$)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / "mp_varying_c.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 6. TW1 universality: large-trial histogram + theoretical density + Q-Q plot
# ============================================================================

def make_tw_universality_figure() -> Path:
    """Two-panel figure showing the raw Johnstone-standardized top sample
    eigenvalue (TW1-distributed: mean ~ -1.21, std ~ 1.27), without any
    further mean-0/var-1 rescaling.

    The Johnstone standardization z = (lambda_max - mu_np) / sigma_np is the
    natural object: it is the unscaled TW1 random variable, so the histogram
    shows the TW1 distribution directly and the Q-Q plot against N(0, 1)
    reveals the full picture (location offset, scale, and shape) in one go.
    """
    try:
        from TracyWidom import TracyWidom
        tw1 = TracyWidom(beta=1)
        have_tw = True
    except ImportError:
        have_tw = False

    rng = np.random.default_rng(SEED + 1)
    n, p = 1000, 500
    trials = 5000
    mu_np, sigma_np = _johnstone_constants(n, p)

    # Collect Johnstone-standardized samples z = (lambda_max - mu_np)/sigma_np.
    # These are TW1-distributed: mean ~ -1.21, std ~ 1.27, left-skewed.
    z = np.empty(trials)
    for t in range(trials):
        X = rng.normal(size=(p, n))
        S = (X @ X.T) / n
        lam_top = float(np.linalg.eigvalsh(S)[-1])
        z[t] = (lam_top - mu_np) / sigma_np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram (raw TW1-distributed samples) with the TW1 density
    # and N(0,1) density overlaid for direct comparison.
    ax_h = axes[0]
    bins = np.linspace(-6, 4, 60)
    ax_h.hist(z, bins=bins, density=True, color=PALETTE["def"], alpha=0.7,
              edgecolor="white", linewidth=0.3,
              label=fr"$z = (\lambda_{{\max}} - \mu_{{np}})/\sigma_{{np}}$ ({trials:,} trials)")
    x_grid = np.linspace(-6, 4, 800)
    if have_tw:
        ax_h.plot(x_grid, tw1.pdf(x_grid), color=PALETTE["thm"], linewidth=1.8,
                  label=r"$\mathrm{TW}_1$ density")
    gauss = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid ** 2)
    ax_h.plot(x_grid, gauss, color=PALETTE["accent"], linewidth=1.0,
              linestyle="--", label=r"$\mathcal{N}(0,1)$ density")
    # Vertical markers for TW1 mean and N(0,1) mean to highlight the offset.
    ax_h.axvline(-1.2065, color=PALETTE["thm"], linewidth=0.7, linestyle=":")
    ax_h.text(-1.2065, ax_h.get_ylim()[1] * 0.92, r"  $\mu_{\mathrm{TW}_1} \approx -1.21$",
              color=PALETTE["thm"], fontsize=8, ha="left", va="top")
    ax_h.set_xlim(-6, 4)
    ax_h.set_xlabel(r"$z = (\lambda_{\max}(\widehat{\Sigma}_n) - \mu_{np}) / \sigma_{np}$")
    ax_h.set_ylabel("Empirical density")
    ax_h.set_title(rf"Top sample eigenvalue at $n = {n}$, $p = {p}$ ($y = {p/n:g}$)")
    ax_h.legend(loc="upper left", frameon=False, fontsize=9)

    # Right: Q-Q plot of the raw TW1-distributed z against N(0, 1) quantiles.
    # Because z has mean ~ -1.21 and std ~ 1.27, the Q-Q is offset and
    # tilted relative to y = x; the curvature on top of that affine shift
    # is the visible shape difference between TW1 and Gaussian.
    ax_q = axes[1]
    sample = np.sort(z)
    k = len(sample)
    probs = (np.arange(1, k + 1) - 0.5) / k
    theo_q = stats.norm.ppf(probs)
    ax_q.scatter(theo_q, sample, s=4, alpha=0.45, color=PALETTE["def"])
    qmin = min(theo_q.min(), sample.min()) - 0.3
    qmax = max(theo_q.max(), sample.max()) + 0.3
    ref = np.linspace(qmin, qmax, 50)
    ax_q.plot(ref, ref, color=PALETTE["accent"], linestyle="--",
              linewidth=0.9, label=r"reference: $y = x$")
    ax_q.set_xlim(qmin, qmax)
    ax_q.set_ylim(qmin, qmax)
    ax_q.set_xlabel(r"Standard normal quantiles")
    ax_q.set_ylabel(r"$z$ quantiles (TW$_1$ scale)")
    ax_q.set_title(r"Q-Q plot: Johnstone-standardized $\lambda_{\max}$ vs $\mathcal{N}(0,1)$")
    ax_q.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / "tw_universality.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# 7. Edge repulsion: visual evidence of the asymmetric MP edge geometry
# ============================================================================

def make_edge_repulsion_figure() -> Path:
    """Visualize WHY the TW1 limit is left-skewed.

    For each of R realizations of Sigmahat_n we record the top K
    eigenvalues. The picture is a strip plot: rank on the y-axis
    (lambda_max on top, lambda_{max-1} below, ...), eigenvalue on the
    x-axis, with the MP upper edge lambda_+ drawn as a reference. The
    two-sided geometry that produces TW1's left-skew becomes visible:
    - the rows below lambda_max all sit packed against lambda_+ on the
      left, none of them ever escaping past the bulk;
    - lambda_max itself scatters substantially, with realizations both
      above and below lambda_+.

    The increasing width of the rows from the bulk inward toward
    lambda_max is the Vandermonde "pressure" that pushes neighbouring
    eigenvalues apart, in action.
    """
    rng = np.random.default_rng(SEED + 2)
    n, p = 600, 300
    y = p / n
    lam_plus = (1.0 + np.sqrt(y)) ** 2
    R = 500
    K = 6

    top_eigs = np.empty((R, K))
    for r in range(R):
        X = rng.normal(size=(p, n))
        S = (X @ X.T) / n
        eigs = np.linalg.eigvalsh(S)  # ascending
        top_eigs[r] = eigs[-K:]       # K largest, in ascending order

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Strip plot: each rank gets a y-position; eigenvalues are scattered
    # horizontally with a small vertical jitter to make the cloud readable.
    rank_labels = [
        r"$\lambda_{\max}$",
        r"$\lambda_{\max-1}$",
        r"$\lambda_{\max-2}$",
        r"$\lambda_{\max-3}$",
        r"$\lambda_{\max-4}$",
        r"$\lambda_{\max-5}$",
    ]
    cmap = plt.cm.viridis_r
    for k in range(K):
        # Rank k=0 -> lambda_max (top of plot). top_eigs[:, K-1] is lambda_max.
        x_vals = top_eigs[:, K - 1 - k]
        y_vals = k + rng.uniform(-0.32, 0.32, size=R)
        ax.scatter(
            x_vals,
            y_vals,
            s=8,
            color=cmap(k / max(K - 1, 1)),
            alpha=0.55,
            edgecolor="none",
        )

    ax.axvline(
        lam_plus,
        color=PALETTE["accent"],
        linestyle="--",
        linewidth=1.2,
        label=rf"MP upper edge $\lambda_+ = (1+\sqrt{{y}})^2 \approx {lam_plus:.2f}$",
    )

    ax.set_yticks(range(K))
    ax.set_yticklabels(rank_labels, fontsize=11)
    ax.invert_yaxis()  # so lambda_max appears at top
    ax.set_xlabel(r"Eigenvalue value")
    ax.set_ylabel(r"Rank")
    ax.set_title(
        rf"Top {K} eigenvalues across {R} realizations of $\widehat{{\Sigma}}_n$ "
        rf"at $n = {n}$, $p = {p}$ ($y = {y:g}$)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    path = FIGURES_DIR / "edge_repulsion.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating Section 3 figures...")
    for fn in (make_geometry_figure,
               make_scatter_axes_figure, make_scm_consistency_figure,
               make_wishart_density_heatmap_figure,
               make_wishart_2d_figure, make_lowdim_qq_figure,
               make_mp_bulk_figure, make_tw_edge_figure,
               make_mp_varying_c_figure, make_tw_universality_figure,
               make_edge_repulsion_figure):
        path = fn()
        print(f"  Saved: {path.name}")
    print("Done.")

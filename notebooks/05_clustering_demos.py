"""05_clustering_demos.py

Generates figures for Section 4.6 (spectral clustering as a spiked model,
after Lebeau, Chatelain and Couillet 2024), Section 4.7 (MNIST case study),
and the conclusion.

Synthetic figures (model X = M J^T + W, Gram matrix G = X^T X / p):

  figures/clustering_one_spike.png     Study (a): one fundamental spike.
                                       ESD of G with the rescaled MP bulk and
                                       the predicted outlier xi; eigenvector
                                       entries vs sqrt(zeta) v with the
                                       Theorem-2 one-sigma band; residual
                                       histogram vs N(0, (1-zeta)/n).

  figures/clustering_two_spikes.png    Study (b): two fundamental spikes.
                                       Entries of v-hat_1 and v-hat_2 with
                                       bands; joint scatter of
                                       ([v-hat_1]_i, [v-hat_2]_i) showing the
                                       two class clouds.

  figures/clustering_accuracy_transition.png
                                       Clustering accuracy against the SNR ell
                                       sweeping through the threshold
                                       sqrt(y): coin-flip below, the
                                       Phi(sqrt(zeta/(1-zeta))) law above.

MNIST figures (real data, LeCun et al. 1998; downloads via fetch_openml on
first run and caches under ~/scikit_learn_data):

  figures/mnist_zero_one.png           Digits 0 vs 1: top eigenvector of the
                                       Gram matrix against the Theorem-2
                                       prediction, residual histogram.

  figures/mnist_pairs_heatmap.png      All 45 digit pairs: observed (upper
                                       triangle) vs predicted (lower triangle)
                                       clustering accuracy.

  figures/mnist_ten_class.png          Ten classes: k-means on the top nine
                                       eigenvectors; confusion matrix and the
                                       ten cluster-mean images.

Conventions follow Lebeau-Chatelain-Couillet (2024), with their aspect
ratio c = p/n written y to match the rest of the report:

  SNRs        ell_k  = eigenvalues of (1/n) L^T L
  threshold   ell_k > sqrt(y)
  outlier     xi_k   = (ell_k + y)(ell_k + 1) / (y ell_k)
  alignment   zeta_k = 1 - (ell_k + y) / (ell_k (ell_k + 1))
  Gram bulk   edges (1 +/- 1/sqrt(y))^2, atom [1 - y]_+ at zero
  Theorem 2   entries of v-hat_k ~ N(sqrt(zeta_k) [v_k]_i, (1 - zeta_k)/n)

Run:
    python3 05_clustering_demos.py            # synthetic only
    python3 05_clustering_demos.py --mnist    # everything (downloads MNIST)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from thesis_style import setup_plot, PALETTE

setup_plot()

SEED = 5050
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Theory helpers (Lebeau-Chatelain-Couillet 2024, Theorems 1 and 2)
# ============================================================================

def zeta_theory(ell, y):
    """Limiting squared alignment |<v_k, vhat_k>|^2 (LCC Thm 1)."""
    ell = np.asarray(ell, dtype=float)
    z = 1.0 - (ell + y) / (ell * (ell + 1.0))
    return np.where(ell > np.sqrt(y), np.maximum(z, 0.0), 0.0)


def xi_theory(ell, y):
    """Limiting location of the isolated Gram eigenvalue (LCC Thm 1)."""
    ell = np.asarray(ell, dtype=float)
    edge = (1.0 + 1.0 / np.sqrt(y)) ** 2
    return np.where(ell > np.sqrt(y), (ell + y) * (ell + 1.0) / (y * ell), edge)


def gram_bulk_density(x, y):
    """Continuous part of the limiting ESD of G = X^T X / p (no signal):
    the Marchenko-Pastur bulk of Section 3 rescaled by 1/y, with edges
    (1 +/- 1/sqrt(y))^2 and total mass min(1, y)."""
    x = np.asarray(x, dtype=float)
    e_minus = (1.0 - 1.0 / np.sqrt(y)) ** 2
    e_plus = (1.0 + 1.0 / np.sqrt(y)) ** 2
    out = np.zeros_like(x)
    inside = (x > e_minus) & (x < e_plus)
    xi = x[inside]
    out[inside] = y * np.sqrt((xi - e_minus) * (e_plus - xi)) / (2.0 * np.pi * xi)
    return out


def accuracy_theory(zeta):
    """Two-class sign-rule accuracy Phi(sqrt(zeta/(1-zeta))) (LCC Sec. IV)."""
    zeta = np.asarray(zeta, dtype=float)
    return stats.norm.cdf(np.sqrt(zeta / np.maximum(1.0 - zeta, 1e-12)))


# ============================================================================
# Simulation helpers
# ============================================================================

def simulate_two_class_symmetric(ell, p, n, rng):
    """One draw of X = mu s^T + W with means +/- mu, equal classes.
    mu is along the first coordinate axis with ||mu||^2 = ell, so the single
    SNR is exactly ell and the population direction is v = s/sqrt(n) with
    s_i = +1 (first half) / -1 (second half).

    Returns (eigenvalues ascending, top eigenvector sign-fixed, v)."""
    s = np.ones(n)
    s[n // 2:] = -1.0
    v = s / np.sqrt(n)
    mu = np.zeros(p)
    mu[0] = np.sqrt(ell)
    X = np.outer(mu, s) + rng.standard_normal((p, n))
    G = (X.T @ X) / p
    w, U = np.linalg.eigh(G)
    vhat = U[:, -1]
    if vhat @ v < 0:
        vhat = -vhat
    return w, vhat, v


def simulate_two_class_two_spikes(ell1, ell2, p, n, rng):
    """One draw of X = M J^T + W with two ORTHOGONAL means mu_1 = a e_1,
    mu_2 = b e_2 and equal classes, so that (1/n) L^T L = diag(a^2/2, b^2/2)
    exactly and the population directions are the normalized class
    indicators v_k = j_k / sqrt(n/2).

    ell_k = a^2/2 resp. b^2/2; pass the desired ell's directly.
    Returns (top-two eigenvalues desc, [vhat_1 vhat_2] sign-fixed, V)."""
    half = n // 2
    a = np.sqrt(2.0 * ell1)
    b = np.sqrt(2.0 * ell2)
    V = np.zeros((n, 2))
    V[:half, 0] = 1.0 / np.sqrt(half)
    V[half:, 1] = 1.0 / np.sqrt(half)
    P = np.zeros((p, n))
    P[0, :half] = a
    P[1, half:] = b
    X = P + rng.standard_normal((p, n))
    G = (X.T @ X) / p
    w, U = np.linalg.eigh(G)
    lams = w[[-1, -2]]
    Uhat = U[:, [-1, -2]]
    for k in range(2):
        if Uhat[:, k] @ V[:, k] < 0:
            Uhat[:, k] = -Uhat[:, k]
    return lams, Uhat, V


# ============================================================================
# Figure 1 (Section 4.6, study a): one fundamental spike
# ============================================================================

def make_clustering_one_spike_figure() -> Path:
    rng = np.random.default_rng(SEED)
    y = 0.5
    n, p = 800, 400
    ell = 4.0
    zeta = float(zeta_theory(ell, y))
    xi = float(xi_theory(ell, y))
    e_plus = (1.0 + 1.0 / np.sqrt(y)) ** 2

    w, vhat, v = simulate_two_class_symmetric(ell, p=p, n=n, rng=rng)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))

    # --- Panel 1: ESD of the Gram matrix ---
    pos = w[w > 1e-8]
    bins = np.linspace(0.0, max(xi * 1.08, e_plus * 1.1), 60)
    ax1.hist(pos, bins=bins, weights=np.full(pos.size, 1.0 / n), density=False,
             color=PALETTE["def"], alpha=0.65, edgecolor="white", linewidth=0.3,
             label=r"positive eigenvalues of $G$")
    # convert the weighted histogram to density units (divide by bin width)
    # by plotting the theory curve scaled the same way:
    grid = np.linspace(0.01, e_plus * 1.05, 500)
    binw = bins[1] - bins[0]
    ax1.plot(grid, gram_bulk_density(grid, y) * binw, color=PALETTE["thm"],
             linewidth=1.6, label="rescaled MP bulk")
    ax1.axvline(xi, color=PALETTE["accent"], linestyle="--", linewidth=1.3,
                label=rf"predicted outlier $\xi \approx {xi:.1f}$")
    ax1.axvline(e_plus, color=PALETTE["accent"], linestyle=":", linewidth=1.0,
                label=rf"bulk edge $\approx {e_plus:.2f}$")
    ax1.set_xlabel("Eigenvalue of the Gram matrix")
    ax1.set_ylabel("Fraction per bin")
    ax1.set_title("Spectrum: bulk plus one spike")
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.35)

    # --- Panel 2: eigenvector entries vs the Theorem-2 prediction ---
    idx = np.arange(n)
    half = n // 2
    sd = np.sqrt((1.0 - zeta) / n)
    mean_vec = np.sqrt(zeta) * v
    ax2.scatter(idx[:half], vhat[:half], s=5, color=PALETTE["def"], alpha=0.5,
                label="class 1 entries")
    ax2.scatter(idx[half:], vhat[half:], s=5, color=PALETTE["accent"], alpha=0.5,
                label="class 2 entries")
    ax2.plot(idx, mean_vec, color=PALETTE["thm"], linewidth=1.6,
             label=r"prediction $\sqrt{\zeta}\, v$")
    ax2.plot(idx, mean_vec + sd, color=PALETTE["thm"], linewidth=0.9,
             linestyle=":")
    ax2.plot(idx, mean_vec - sd, color=PALETTE["thm"], linewidth=0.9,
             linestyle=":", label=r"$\pm$ one sd $\sqrt{(1-\zeta)/n}$")
    ax2.axhline(0.0, color="black", linewidth=0.6)
    ax2.set_xlabel(r"Sample index $i$ (sorted by class)")
    ax2.set_ylabel(r"$[\widehat{v}]_i$")
    ax2.set_title("Top eigenvector: two bands of entries")
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.35)

    # --- Panel 3: residuals vs the predicted normal ---
    res = vhat - mean_vec
    ax3.hist(res, bins=40, density=True, color=PALETTE["def"], alpha=0.65,
             edgecolor="white", linewidth=0.3, label="residuals")
    grid = np.linspace(res.min() * 1.3, res.max() * 1.3, 400)
    ax3.plot(grid, stats.norm.pdf(grid, scale=sd), color=PALETTE["thm"],
             linewidth=1.6, label=r"$\mathcal{N}(0,\, (1-\zeta)/n)$")
    ax3.set_xlabel(r"$[\widehat{v}]_i - \sqrt{\zeta}\, [v]_i$")
    ax3.set_ylabel("Density")
    ax3.set_title("Residuals match the predicted Gaussian")
    ax3.legend(loc="upper right", frameon=False, fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.35)

    fig.suptitle(
        rf"One fundamental spike: $y = 1/2$, $n = {n}$, $p = {p}$, "
        rf"$\ell = {ell:g}$, $\zeta \approx {zeta:.3f}$",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "clustering_one_spike.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Figure 2 (Section 4.6, study b): two fundamental spikes
# ============================================================================

def make_clustering_two_spikes_figure() -> Path:
    rng = np.random.default_rng(SEED + 1)
    y = 0.5
    n, p = 800, 400
    ell1, ell2 = 5.0, 2.0
    z1 = float(zeta_theory(ell1, y))
    z2 = float(zeta_theory(ell2, y))
    half = n // 2

    lams, Uhat, V = simulate_two_class_two_spikes(ell1, ell2, p=p, n=n, rng=rng)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.4))
    idx = np.arange(n)

    for ax, k, z in ((ax1, 0, z1), (ax2, 1, z2)):
        sd = np.sqrt((1.0 - z) / n)
        mean_vec = np.sqrt(z) * V[:, k]
        ax.scatter(idx[:half], Uhat[:half, k], s=5, color=PALETTE["def"],
                   alpha=0.5, label="class 1 entries")
        ax.scatter(idx[half:], Uhat[half:, k], s=5, color=PALETTE["accent"],
                   alpha=0.5, label="class 2 entries")
        ax.plot(idx, mean_vec, color=PALETTE["thm"], linewidth=1.6,
                label=rf"$\sqrt{{\zeta_{k+1}}}\, v_{k+1}$")
        ax.plot(idx, mean_vec + sd, color=PALETTE["thm"], linewidth=0.9,
                linestyle=":")
        ax.plot(idx, mean_vec - sd, color=PALETTE["thm"], linewidth=0.9,
                linestyle=":", label=r"$\pm$ one sd")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xlabel(r"Sample index $i$ (sorted by class)")
        ax.set_ylabel(rf"$[\widehat{{v}}_{k+1}]_i$")
        ax.set_title(rf"Eigenvector {k+1}: lights up on class {k+1}")
        ax.legend(loc="center right", frameon=False, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)

    # --- Panel 3: the embedding ---
    c1 = (np.sqrt(z1 / half), 0.0)
    c2 = (0.0, np.sqrt(z2 / half))
    ax3.scatter(Uhat[:half, 0], Uhat[:half, 1], s=6, color=PALETTE["def"],
                alpha=0.5, label="class 1")
    ax3.scatter(Uhat[half:, 0], Uhat[half:, 1], s=6, color=PALETTE["accent"],
                alpha=0.5, label="class 2")
    ax3.scatter(*zip(c1, c2), marker="D", s=55, color=PALETTE["thm"], zorder=5,
                label="predicted centers")
    ax3.set_xlabel(r"$[\widehat{v}_1]_i$")
    ax3.set_ylabel(r"$[\widehat{v}_2]_i$")
    ax3.set_title("The embedding: one cloud per class")
    ax3.legend(loc="upper right", frameon=False, fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.35)

    fig.suptitle(
        rf"Two fundamental spikes: $y = 1/2$, $n = {n}$, $p = {p}$, "
        rf"$\ell_1 = {ell1:g}$, $\ell_2 = {ell2:g}$, "
        rf"$\zeta_1 \approx {z1:.2f}$, $\zeta_2 \approx {z2:.2f}$",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "clustering_two_spikes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Figure 3 (Section 4.6): clustering accuracy through the threshold
# ============================================================================

def make_accuracy_transition_figure() -> Path:
    rng = np.random.default_rng(SEED + 2)
    y = 0.5
    n, p = 800, 400
    R = 60
    thr = np.sqrt(y)

    ell_grid = np.unique(np.round(np.concatenate([
        np.linspace(0.1, thr, 8, endpoint=False),
        np.linspace(thr, 4.0, 14),
    ]), 4))

    acc_mean = np.empty(len(ell_grid))
    acc_sd = np.empty(len(ell_grid))
    for i, ell in enumerate(ell_grid):
        accs = np.empty(R)
        for r in range(R):
            _, vhat, v = simulate_two_class_symmetric(float(ell), p=p, n=n,
                                                      rng=rng)
            accs[r] = np.mean(np.sign(vhat) == np.sign(v))
        acc_mean[i] = accs.mean()
        acc_sd[i] = accs.std(ddof=1)

    ell_fine = np.linspace(0.05, 4.0, 500)
    acc_th = accuracy_theory(zeta_theory(ell_fine, y))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(ell_fine, acc_th, color=PALETTE["thm"], linewidth=1.8,
            label=r"theory $\Phi(\sqrt{\zeta/(1-\zeta)})$")
    ax.errorbar(ell_grid, acc_mean, yerr=acc_sd, fmt="o", markersize=4,
                color=PALETTE["def"], capsize=3, linewidth=1.0,
                label=rf"simulation mean $\pm$ sd ($R = {R}$)")
    ax.axvline(thr, color=PALETTE["accent"], linestyle="--", linewidth=1.1,
               label=rf"threshold $\sqrt{{y}} \approx {thr:.2f}$")
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle=":")

    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel(r"Signal-to-noise ratio $\ell$")
    ax.set_ylabel("Fraction of points classified correctly")
    ax.set_title(rf"Clustering accuracy through the threshold "
                 rf"($y = 1/2$, $n = {n}$, $p = {p}$)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    path = FIGURES_DIR / "clustering_accuracy_transition.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# MNIST helpers (Section 4.7)
# ============================================================================

def load_mnist():
    """Load MNIST (70k digits, 784 pixels), sorted by class, scaled to [0,1].
    Returns (X_full p x N, labels N)."""
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = np.asarray(mnist.data, dtype=float).T / 255.0
    yl = np.asarray(mnist.target, dtype=int)
    order = np.argsort(yl, kind="stable")
    return X[:, order], yl[order]


def pair_spike(X_full, labels, k1, k2, per_class, rng):
    """Top Gram eigenvector for the two-digit subproblem (k1 vs k2), with
    per_class samples of each digit. Mirrors the LCC pipeline: top right
    singular vector of X/sqrt(p), centered, renormalized, sign-fixed by the
    true labels. Returns (vhat, ytrue) with ytrue = +/- 1/sqrt(n)."""
    from scipy.sparse.linalg import svds
    cols1 = np.where(labels == k1)[0]
    cols2 = np.where(labels == k2)[0]
    cols1 = rng.choice(cols1, size=per_class, replace=False)
    cols2 = rng.choice(cols2, size=per_class, replace=False)
    X = X_full[:, np.concatenate([np.sort(cols1), np.sort(cols2)])]
    p, n = X.shape
    ytrue = np.ones(n) / np.sqrt(n)
    ytrue[:per_class] *= -1.0
    _, _, Vt = svds(X / np.sqrt(p), k=1, which="LM")
    v = Vt[0] - Vt[0].mean()
    v /= np.linalg.norm(v)
    if v @ ytrue < 0:
        v = -v
    return v, ytrue


def make_mnist_zero_one_figure() -> Path:
    rng = np.random.default_rng(SEED + 3)
    X_full, labels = load_mnist()
    per_class = 6000
    v, ytrue = pair_spike(X_full, labels, 0, 1, per_class, rng)
    n = v.size
    zeta = float((ytrue @ v) ** 2)
    sd = np.sqrt((1.0 - zeta) / n)
    acc_obs = float(np.mean(np.sign(v) == np.sign(ytrue)))
    acc_th = float(accuracy_theory(zeta))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    idx = np.arange(n)
    mean_vec = np.sqrt(zeta) * ytrue
    step = 4  # plot every 4th entry to keep the scatter light
    ax1.scatter(idx[:per_class:step], v[:per_class:step], s=4,
                color=PALETTE["def"], alpha=0.4, label="digit 0 entries")
    ax1.scatter(idx[per_class::step], v[per_class::step], s=4,
                color=PALETTE["accent"], alpha=0.4, label="digit 1 entries")
    ax1.plot(idx, mean_vec, color=PALETTE["thm"], linewidth=1.6,
             label=r"$\sqrt{\widehat{\zeta}}\, v$")
    ax1.plot(idx, mean_vec + sd, color=PALETTE["thm"], linewidth=0.9,
             linestyle=":")
    ax1.plot(idx, mean_vec - sd, color=PALETTE["thm"], linewidth=0.9,
             linestyle=":", label=r"$\pm$ one sd")
    ax1.axhline(0.0, color="black", linewidth=0.6)
    ax1.set_xlabel(r"Image index $i$ (sorted by digit)")
    ax1.set_ylabel(r"$[\widehat{v}]_i$")
    ax1.set_title("Top Gram eigenvector of the 0/1 data")
    ax1.legend(loc="center right", frameon=False, fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.35)

    res = v - mean_vec
    ax2.hist(res, bins=60, density=True, color=PALETTE["def"], alpha=0.65,
             edgecolor="white", linewidth=0.3, label="residuals")
    grid = np.linspace(res.min() * 1.2, res.max() * 1.2, 400)
    ax2.plot(grid, stats.norm.pdf(grid, scale=sd), color=PALETTE["thm"],
             linewidth=1.6, label=r"$\mathcal{N}(0,\, (1-\widehat{\zeta})/n)$")
    ax2.set_xlabel(r"$[\widehat{v}]_i - \sqrt{\widehat{\zeta}}\, [v]_i$")
    ax2.set_ylabel("Density")
    ax2.set_title("Residuals on real data")
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.35)

    fig.suptitle(
        rf"MNIST digits 0 vs 1: $n = {n}$, $p = 784$, "
        rf"$\widehat{{\zeta}} \approx {zeta:.3f}$",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "mnist_zero_one.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [0 vs 1] zeta-hat = {zeta:.4f}, observed accuracy = {acc_obs:.4f}, "
          f"predicted = {acc_th:.4f}")
    return path


def make_mnist_pairs_heatmap_figure() -> Path:
    from itertools import combinations
    rng = np.random.default_rng(SEED + 4)
    X_full, labels = load_mnist()
    per_class = 5400

    acc_obs = np.full((10, 10), np.nan)
    acc_th = np.full((10, 10), np.nan)
    for k1, k2 in combinations(range(10), 2):
        v, ytrue = pair_spike(X_full, labels, k1, k2, per_class, rng)
        zeta = float((ytrue @ v) ** 2)
        acc_th[k1, k2] = float(accuracy_theory(zeta))
        acc_obs[k1, k2] = float(np.mean(np.sign(v) == np.sign(ytrue)))

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    M = np.full((10, 10), np.nan)
    for k1 in range(10):
        for k2 in range(k1 + 1, 10):
            M[k1, k2] = acc_obs[k1, k2]   # upper triangle: observed
            M[k2, k1] = acc_th[k1, k2]    # lower triangle: predicted
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="Greens")
    for k1 in range(10):
        for k2 in range(10):
            if k1 != k2:
                ax.text(k2, k1, f"{M[k1, k2]:.2f}", ha="center", va="center",
                        fontsize=7.5)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("digit")
    ax.set_ylabel("digit")
    ax.set_title("Pairwise clustering accuracy on MNIST: observed (upper "
                 "triangle) vs predicted (lower triangle)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIGURES_DIR / "mnist_pairs_heatmap.png"
    fig.savefig(path)
    plt.close(fig)

    diff = np.abs(acc_obs - acc_th)
    print(f"  [pairs] mean |observed - predicted| = {np.nanmean(diff):.4f}, "
          f"max = {np.nanmax(diff):.4f}")
    flat = [(acc_obs[a, b], a, b) for a, b in combinations(range(10), 2)]
    flat.sort()
    print(f"  [pairs] hardest: {flat[:3]}")
    print(f"  [pairs] easiest: {flat[-3:]}")
    return path


def make_mnist_ten_class_figure() -> Path:
    from scipy.sparse.linalg import svds
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(SEED + 5)
    X_full, labels = load_mnist()
    per_class = 5400
    cols = np.concatenate([
        rng.choice(np.where(labels == k)[0], size=per_class, replace=False)
        for k in range(10)
    ])
    cols.sort()
    X = X_full[:, cols]
    ytrue = labels[cols]
    p, n = X.shape

    # Centre the data (remove the grand-mean image) and take the top nine
    # right singular vectors: the spike embedding.
    Xc = X - X.mean(axis=1, keepdims=True)
    _, _, Vt = svds(Xc / np.sqrt(p), k=9, which="LM")
    emb = Vt.T  # n x 9

    km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(emb)
    pred = km.labels_

    # Hungarian matching of clusters to digits.
    C = np.zeros((10, 10), dtype=int)
    for t, q in zip(ytrue, pred):
        C[t, q] += 1
    row, col = linear_sum_assignment(-C)
    mapping = {q: t for t, q in zip(row, col)}
    pred_mapped = np.array([mapping[q] for q in pred])
    acc = float(np.mean(pred_mapped == ytrue))
    Cm = np.zeros((10, 10), dtype=int)
    for t, q in zip(ytrue, pred_mapped):
        Cm[t, q] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    im = ax1.imshow(Cm / per_class, vmin=0, vmax=1, cmap="Greens")
    for t in range(10):
        for q in range(10):
            val = Cm[t, q] / per_class
            if val >= 0.01:
                ax1.text(q, t, f"{100*val:.0f}", ha="center", va="center",
                         fontsize=7,
                         color="white" if val > 0.6 else "black")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xlabel("cluster (matched digit)")
    ax1.set_ylabel("true digit")
    ax1.set_title(f"Confusion matrix (percent of true class), "
                  f"accuracy about {100*acc:.0f} percent")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Cluster representatives: mean image of each matched cluster, 2 x 5.
    montage = np.ones((2 * 28 + 6, 5 * 28 + 24))
    for d in range(10):
        members = cols[pred_mapped == d]
        mean_img = X_full[:, np.searchsorted(cols, members)].mean(axis=1)
        img = mean_img.reshape(28, 28)
        r, c = divmod(d, 5)
        r0 = r * (28 + 6)
        c0 = c * (28 + 6)
        montage[r0:r0 + 28, c0:c0 + 28] = 1.0 - img
    ax2.imshow(montage, cmap="gray", vmin=0, vmax=1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Cluster representatives (mean image per cluster)")

    fig.suptitle(rf"Ten-class spectral clustering of MNIST: $k$-means on the "
                 rf"top nine eigenvectors ($n = {n}$, $p = {p}$)", fontsize=11)
    fig.tight_layout()
    path = FIGURES_DIR / "mnist_ten_class.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [10-class] n = {n}, accuracy after matching = {acc:.4f}")
    per_digit = {d: Cm[d, d] / per_class for d in range(10)}
    print(f"  [10-class] per-digit accuracy: "
          f"{ {d: round(a, 3) for d, a in per_digit.items()} }")
    return path


if __name__ == "__main__":
    print("Generating clustering figures...")
    for fn in (make_clustering_one_spike_figure,
               make_clustering_two_spikes_figure,
               make_accuracy_transition_figure):
        path = fn()
        print(f"  Saved: {path.name}")
    if "--mnist" in sys.argv:
        print("MNIST figures (downloads ~15 MB on first run)...")
        for fn in (make_mnist_zero_one_figure,
                   make_mnist_pairs_heatmap_figure,
                   make_mnist_ten_class_figure):
            path = fn()
            print(f"  Saved: {path.name}")
    print("Done.")

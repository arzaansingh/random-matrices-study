"""06_titlepage_art.py

Generates the cover composition for the title page: ONE simulation, four
views, connected by a dashed path that walks the reader through the
report's pipeline, in the house palette.

The single data set is the clustering model of the report,

    X = M J^T + W,   p = 450, n = 900, y = 1/2, K = 3 classes,

with one draw of the noise matrix W threading all four panels:

  1 (top-left):     eigenvalues of the sample covariance matrix of the
                    noise alone, W W^T / n: the Marchenko-Pastur bulk
  2 (top-right):    eigenvalues of X X^T / n for the SAME W: the same bulk
                    plus three detached spikes (drawn as stems at their
                    simulated locations)
  3 (mid-left):     entries of the top eigenvector, which lights up on a
                    class
  4 (bottom-right): the spectral embedding by the top two eigenvectors,
                    where the classes appear as clouds

Clear zones are reserved for the LaTeX title block (top) and the author
block (the void left of panel 4); frontmatter/titlepage.tex overlays the
type.

Run:
    python3 06_titlepage_art.py

Output:
    Thesis/figures/titlepage_background.pdf   (vector, used by the cover)
    Thesis/figures/titlepage_background.png   (raster preview)
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from thesis_style import PALETTE

SEED = 6060
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

CREAM = "#FBFAF5"
INK = PALETTE["thm"]        # charcoal
SLATE = PALETTE["def"]      # slate navy
BLUE = PALETTE["link"]      # muted blue
GRAY = PALETTE["accent"]    # warm gray
SAGE = PALETTE["vocab"]     # sage green

Y_ASPECT = 0.5


def gram_bulk_density(x, y):
    """Continuous part of the limiting ESD of W^T W / p: the MP bulk of the
    report rescaled by 1/y, edges (1 +/- 1/sqrt(y))^2."""
    e_minus = (1.0 - 1.0 / np.sqrt(y)) ** 2
    e_plus = (1.0 + 1.0 / np.sqrt(y)) ** 2
    out = np.zeros_like(x)
    inside = (x > e_minus) & (x < e_plus)
    xi = x[inside]
    out[inside] = y * np.sqrt((xi - e_minus) * (e_plus - xi)) / (2.0 * np.pi * xi)
    return out


def simulate_story(rng, K=3, per=300, p=450):
    """One draw. Returns everything all four panels need."""
    n = K * per
    ells = np.array([7.0, 4.0, 2.5])
    M = np.zeros((p, K))
    for k in range(K):
        M[k, k] = np.sqrt(ells[k] * n / per)
    J = np.zeros((n, K))
    for k in range(K):
        J[k * per:(k + 1) * per, k] = 1.0
    W = rng.standard_normal((p, n))           # the one noise matrix
    X = M @ J.T + W                           # the one data set

    noise_eigs = np.linalg.eigvalsh(W @ W.T / n)
    data_eigs = np.linalg.eigvalsh(X @ X.T / n)
    bulk, spikes = data_eigs[:-K], data_eigs[-K:]

    _, _, Vt = np.linalg.svd(X / np.sqrt(p), full_matrices=False)
    v1, v2 = Vt[0], Vt[1]
    if v1[:per].mean() < 0:
        v1 = -v1
    if v2[per:2 * per].mean() < 0:
        v2 = -v2
    return noise_eigs, bulk, spikes, v1, v2, n, per


def make_cover() -> Path:
    rng = np.random.default_rng(SEED)
    noise_eigs, bulk, spikes, v1, v2, n, per = simulate_story(rng)
    class_colors = [SLATE, BLUE, GRAY]

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(CREAM)

    def bare_axes(rect):
        ax = fig.add_axes(rect)
        ax.set_facecolor("none")
        ax.axis("off")
        return ax

    def caption(xc, y, s):
        fig.text(xc, y, s, style="italic", fontsize=9, color=GRAY, ha="center")

    def badge(x, y, num):
        fig.text(x, y, num, fontsize=10, color=INK, ha="center", va="center",
                 fontweight="bold",
                 bbox=dict(boxstyle="circle,pad=0.32", facecolor=SAGE,
                           edgecolor=INK, linewidth=0.9))

    def arrow(p0, p1, rad):
        fig.add_artist(FancyArrowPatch(
            p0, p1, transform=fig.transFigure,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=15,
            linestyle=(0, (4, 3)), linewidth=1.25, color=GRAY, alpha=0.95))

    xlim = (-0.2, 9.4)
    xg = np.linspace(0.01, 9.3, 900)

    def kde(x, grid, bw=0.12):
        z = (grid[:, None] - x[None, :]) / bw
        return np.exp(-0.5 * z * z).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))

    noise_dens = kde(noise_eigs, xg)
    bulk_dens = kde(bulk, xg)
    ymax = 1.12 * max(noise_dens.max(), bulk_dens.max())

    # ---- panel 1 (top-left): the noise alone -------------------------------
    ax = bare_axes([0.075, 0.615, 0.355, 0.185])
    ax.fill_between(xg, 0, noise_dens, color=SLATE, alpha=0.30, linewidth=0)
    ax.plot(xg, noise_dens, color=INK, lw=1.4)
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(0, ymax)
    badge(0.068, 0.812, "1")
    caption(0.2525, 0.594, "noise alone: the Marchenko-Pastur bulk")

    # ---- panel 2 (top-right): same noise + signal --------------------------
    ax = bare_axes([0.565, 0.545, 0.36, 0.185])
    ax.fill_between(xg, 0, bulk_dens, color=SLATE, alpha=0.30, linewidth=0)
    ax.plot(xg, bulk_dens, color=INK, lw=1.4)
    for sp in spikes:
        ax.plot([sp, sp], [0, 0.32 * ymax], color=INK, lw=1.4)
        ax.plot([sp], [0.32 * ymax], "o", ms=4.5, mfc=SAGE, mec=INK, mew=0.9)
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(0, ymax)
    badge(0.558, 0.742, "2")
    caption(0.745, 0.524, "the same noise plus signal: three spikes detach")

    # ---- panel 3 (mid-left): the top eigenvector ---------------------------
    ax = bare_axes([0.085, 0.30, 0.36, 0.16])
    idx = np.arange(n)
    for k, col in enumerate(class_colors):
        sl = slice(k * per, (k + 1) * per)
        ax.scatter(idx[sl], v1[sl], s=3.2, color=col, alpha=0.55, linewidths=0)
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlim(-12, n + 12)
    lvl = np.abs(v1).max()
    ax.set_ylim(-0.55 * lvl, 1.25 * lvl)
    badge(0.078, 0.472, "3")
    caption(0.265, 0.279, "the top eigenvector lights up on a class")

    # ---- panel 4 (bottom-right): the embedding -----------------------------
    ax = bare_axes([0.575, 0.13, 0.33, 0.205])
    for k, col in enumerate(class_colors):
        sl = slice(k * per, (k + 1) * per)
        ax.scatter(v1[sl] * np.sqrt(n), v2[sl] * np.sqrt(n),
                   s=5.5, color=col, alpha=0.45, linewidths=0)
    ax.set_aspect("equal")
    badge(0.566, 0.348, "4")
    caption(0.74, 0.112, "the embedding: the classes appear")

    # ---- the path -----------------------------------------------------------
    arrow((0.443, 0.690), (0.555, 0.660), rad=-0.30)
    arrow((0.715, 0.512), (0.475, 0.388), rad=-0.32)
    arrow((0.455, 0.352), (0.563, 0.260), rad=-0.30)

    out_pdf = FIGURES_DIR / "titlepage_background.pdf"
    fig.savefig(out_pdf, facecolor=CREAM, dpi=300)
    fig.savefig(FIGURES_DIR / "titlepage_background.png", facecolor=CREAM, dpi=140)
    plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    print("Composing the title-page background...")
    path = make_cover()
    print(f"  Saved: {path.name} (+ .png preview)")
    print("Done.")

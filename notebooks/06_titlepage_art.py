"""06_titlepage_art.py

Cover composition: ONE simulation, four views, one continuous flight path.

The single data set is the clustering model of the report,

    X = M J^T + W,   p = 450, n = 900, y = 1/2, K = 3 classes,

with one draw of the noise matrix W threading every panel:

  1 (top-left):     the sample covariance matrix itself; its top corner is
                    typeset as a numeric matrix by frontmatter/titlepage.tex
                    (the numbers quoted there are this simulation's values)
  2 (top-right):    a 3D waterfall of spectra of (t M J^T + W) as the signal
                    dial t turns from 0 (the noise of panel 1) to 1 (the
                    data of panels 3 and 4); the outlier peaks are drawn at
                    their simulated locations with stylized height
  3 (mid-left):     entries of the top eigenvector, which lights up on a
                    class
  4 (bottom-right): the spectral embedding by the top two eigenvectors

A single dashed path flies 1 -> 2 -> 3 -> 4. Clear zones are reserved for
the LaTeX title block (top) and the author block (the void beside panel 4);
frontmatter/titlepage.tex overlays the type.

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
K, PER, P_DIM = 3, 300, 450
N_DIM = K * PER


def kde(x, grid, bw):
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def simulate_story(rng):
    """One draw. Returns everything all four panels need."""
    ells = np.array([7.0, 4.0, 2.5])
    M = np.zeros((P_DIM, K))
    for k in range(K):
        M[k, k] = np.sqrt(ells[k] * N_DIM / PER)
    J = np.zeros((N_DIM, K))
    for k in range(K):
        J[k * PER:(k + 1) * PER, k] = 1.0
    W = rng.standard_normal((P_DIM, N_DIM))   # the one noise matrix
    signal = M @ J.T
    X = signal + W                            # the one data set

    scm = X @ X.T / N_DIM                     # panel 1: the SCM itself

    # panel 2: spectra of (t * signal + W) for the dial t in [0, 1]
    dial = np.linspace(0.0, 1.0, 13)
    rows = []
    for t in dial:
        ev = np.linalg.eigvalsh((t * signal + W) @ (t * signal + W).T / N_DIM)
        rows.append((ev[:-K], ev[-K:]))       # (bulk, top-3)

    _, _, Vt = np.linalg.svd(X / np.sqrt(P_DIM), full_matrices=False)
    v1, v2 = Vt[0], Vt[1]
    if v1[:PER].mean() < 0:
        v1 = -v1
    if v2[PER:2 * PER].mean() < 0:
        v2 = -v2
    return scm, rows, v1, v2


def make_cover() -> Path:
    rng = np.random.default_rng(SEED)
    scm, rows, v1, v2 = simulate_story(rng)
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

    # ---- panel 1 (top-left): zone reserved for the LaTeX-typeset matrix ----
    badge(0.052, 0.812, "1")
    caption(0.235, 0.625, "the sample covariance matrix")

    # ---- panel 2 (top-right): the waterfall, signal dialed up --------------
    ax = bare_axes([0.475, 0.475, 0.48, 0.33])
    xg = np.linspace(0.01, 9.4, 700)
    edge_plus = (1 + np.sqrt(Y_ASPECT)) ** 2
    n_rows = len(rows)
    step, amp, xshift = 0.58, 1.42, 0.42
    dens_all = [kde(b, xg, 0.12) for b, _ in rows]
    zmax = max(d.max() for d in dens_all)
    for i in range(n_rows - 1, -1, -1):
        bulk_i, spikes_i = rows[i]
        base = i * step
        xs = xg + i * xshift
        z = dens_all[i] / zmax * amp
        for sp in spikes_i:
            if sp > edge_plus + 0.25:
                h = 0.32 + 0.85 * min(1.0, (sp - edge_plus) / 5.0)
                z = z + h * np.exp(-0.5 * ((xg - sp) / 0.085) ** 2)
        zo = 2 * (n_rows - i)
        ax.fill_between(xs, base, base + z, color=CREAM, zorder=zo)
        ax.plot(xs, base + z, color=INK, lw=1.2, zorder=zo + 1)
    ax.set_xlim(-0.3, 9.4 + n_rows * xshift + 0.3)
    ax.set_ylim(-0.3, n_rows * step + 2.2)
    badge(0.468, 0.792, "2")
    caption(0.715, 0.455, "turning the signal up: the spikes detach")

    # ---- panel 3 (mid-left): the top eigenvector ---------------------------
    ax = bare_axes([0.085, 0.275, 0.36, 0.155])
    idx = np.arange(N_DIM)
    for k, col in enumerate(class_colors):
        sl = slice(k * PER, (k + 1) * PER)
        ax.scatter(idx[sl], v1[sl], s=3.2, color=col, alpha=0.55, linewidths=0)
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlim(-12, N_DIM + 12)
    lvl = np.abs(v1).max()
    ax.set_ylim(-0.55 * lvl, 1.25 * lvl)
    badge(0.078, 0.443, "3")
    caption(0.265, 0.254, "the top eigenvector lights up on a class")

    # ---- panel 4 (bottom-right): the embedding -----------------------------
    ax = bare_axes([0.575, 0.115, 0.33, 0.205])
    for k, col in enumerate(class_colors):
        sl = slice(k * PER, (k + 1) * PER)
        ax.scatter(v1[sl] * np.sqrt(N_DIM), v2[sl] * np.sqrt(N_DIM),
                   s=5.5, color=col, alpha=0.45, linewidths=0)
    ax.set_aspect("equal")
    badge(0.566, 0.333, "4")
    caption(0.74, 0.097, "the embedding: the classes appear")

    # (the dotted flow line 1 -> 2 -> 3 -> 4 is drawn by frontmatter/titlepage.tex)

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

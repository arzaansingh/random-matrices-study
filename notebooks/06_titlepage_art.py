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

    # ---- panel 1 (top-left): zone reserved for the LaTeX-typeset matrix ----

    # ---- panel 2 (right): the waterfall, signal dialed up -------------------
    ax = bare_axes([0.50, 0.435, 0.465, 0.32])
    xg = np.linspace(0.01, 9.4, 700)
    edge_plus = (1 + np.sqrt(Y_ASPECT)) ** 2
    n_rows = len(rows)
    step, amp, xshift = 0.58, 1.42, 0.40
    dens_all = [kde(b, xg, 0.12) for b, _ in rows]
    zmax = max(d.max() for d in dens_all)
    track = []
    for i in range(n_rows - 1, -1, -1):
        bulk_i, spikes_i = rows[i]
        base = i * step
        xs = xg + i * xshift
        z = dens_all[i] / zmax * amp
        top_spike = max(spikes_i)
        for sp in spikes_i:
            if sp > edge_plus + 0.25:
                h = 0.32 + 0.85 * min(1.0, (sp - edge_plus) / 5.0)
                z = z + h * np.exp(-0.5 * ((xg - sp) / 0.085) ** 2)
        if top_spike > edge_plus + 0.25:
            j = np.argmin(np.abs(xg - top_spike))
            track.append((top_spike + i * xshift, base + z[j]))
        zo = 2 * (n_rows - i)
        ax.fill_between(xs, base, base + z, color=CREAM, zorder=zo)
        ax.plot(xs, base + z, color=INK, lw=1.2, zorder=zo + 1)
    # the comet track through the detaching peaks
    track = np.array(sorted(track))
    ax.plot(track[:, 0], track[:, 1], ls="-", color=BLUE,
            lw=0.9, alpha=0.45, zorder=2 * n_rows + 5)
    ax.set_xlim(-0.3, 9.4 + n_rows * xshift + 0.3)
    ax.set_ylim(-0.3, n_rows * step + 2.2)

    # ---- panel 3 (mid-left): the top TWO eigenvectors -----------------------
    idx = np.arange(N_DIM)
    lvl1, lvl2 = np.abs(v1).max(), np.abs(v2).max()
    for rect, vec, lvl, lab in [
        ([0.10, 0.315, 0.34, 0.085], v1, lvl1, r"$\widehat{v}_1$"),
        ([0.10, 0.215, 0.34, 0.085], v2, lvl2, r"$\widehat{v}_2$"),
    ]:
        ax = bare_axes(rect)
        for k, col in enumerate(class_colors):
            sl = slice(k * PER, (k + 1) * PER)
            ax.scatter(idx[sl], vec[sl], s=2.8, color=col, alpha=0.55,
                       linewidths=0)
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_xlim(-12, N_DIM + 12)
        ax.set_ylim(-1.3 * lvl, 1.3 * lvl)
        fig.text(rect[0] - 0.022, rect[1] + rect[3] / 2, lab, color=SLATE,
                 fontsize=11, ha="center", va="center")

    # ---- panel 4 (bottom-right): the embedding finale ------------------------
    from matplotlib.patches import Ellipse
    ax = bare_axes([0.50, 0.015, 0.48, 0.37])
    emb = np.c_[v1, v2] * np.sqrt(N_DIM)
    for k, col in enumerate(class_colors):
        pts = emb[k * PER:(k + 1) * PER]
        ax.scatter(pts[:, 0], pts[:, 1], s=9.0, color=col, alpha=0.55,
                   linewidths=0)
        mu = pts.mean(0)
        C = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(C)
        ang = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))
        ax.add_patch(Ellipse(mu, 4 * np.sqrt(evals[-1]), 4 * np.sqrt(evals[0]),
                             angle=ang, fill=False, ls=(0, (5, 3)),
                             edgecolor=col, lw=1.35, alpha=0.95))
        ax.plot(*mu, marker="+", ms=7, mew=1.4, color=INK)
    ax.set_aspect("equal")
    ax.margins(0.08)

    # ---- the flow: one continuous line, always behind the figures ----------
    # The route: from the matrix, over the waterfall's shoulder, down the
    # spike diagonal (the ridges occlude it), a wide loop wrapping the
    # eigenvector strips, then out along the bottom toward the clouds.
    # Rendered as a single tapered line: thin warm gray at the start,
    # swelling to slate at the destination.
    waypoints = np.array([
        (0.435, 0.690),   # leave the matrix
        (0.585, 0.740),   # rise over the waterfall's shoulder
        (0.760, 0.748),   # crest above the ridges
        (0.905, 0.718),   # turn at the summit spike
        (0.840, 0.598),   # descend the spike diagonal, behind the ridges
        (0.745, 0.488),
        (0.640, 0.428),   # exit at the waterfall's foot
        (0.470, 0.452),   # sweep left, a low shoulder
        (0.255, 0.430),   # crossing toward the margin
        (0.115, 0.345),   # down the left edge, wrapping the strips
        (0.105, 0.235),
        (0.150, 0.165),   # around the bottom corner
        (0.330, 0.142),   # along the bottom, under the strips
        (0.480, 0.150),
        (0.545, 0.160),   # stop at the clouds' doorstep
    ])
    n_wp = len(waypoints)
    tang = np.zeros_like(waypoints)
    for i in range(n_wp):
        tang[i] = 0.55 * (waypoints[min(i + 1, n_wp - 1)]
                          - waypoints[max(i - 1, 0)]) / 2.0
    dense = []
    for i in range(n_wp - 1):
        p0, p1 = waypoints[i], waypoints[i + 1]
        c0, c1 = p0 + tang[i] / 3.0, p1 - tang[i + 1] / 3.0
        t = np.linspace(0, 1, 60)[:, None]
        seg = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0
               + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * p1)
        dense.append(seg[:-1])
    dense = np.vstack(dense + [waypoints[-1][None, :]])
    from matplotlib.collections import LineCollection
    pts = dense.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    m = len(segs)
    frac = np.linspace(0, 1, m)
    g, sl = np.array([0x8B, 0x83, 0x78]) / 255, np.array([0x3A, 0x5A, 0x78]) / 255
    colors = (1 - frac[:, None]) * g + frac[:, None] * sl
    widths = 1.1 + 2.3 * frac ** 1.4
    ax_flow = fig.add_axes([0, 0, 1, 1])
    ax_flow.set_zorder(-5)
    ax_flow.set_facecolor("none")
    ax_flow.axis("off")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    lc = LineCollection(segs, colors=colors, linewidths=widths,
                        capstyle="round", alpha=0.85)
    ax_flow.add_collection(lc)
    tip = dense[-1]
    direction = dense[-1] - dense[-6]
    direction = direction / np.hypot(*direction)
    from matplotlib.patches import FancyArrowPatch
    ax_flow.add_patch(FancyArrowPatch(
        tip, tip + direction * 0.018, arrowstyle="-|>",
        mutation_scale=20, color=sl, linewidth=2.0))

    # (the header and the numeric matrix are typeset by frontmatter/titlepage.tex)

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

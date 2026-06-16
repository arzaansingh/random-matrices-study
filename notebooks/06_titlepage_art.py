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
  3 (mid-left):     entries of the top three eigenvectors, each of which
                    lights up on its own class
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
    dial = np.linspace(0.0, 1.0, 13) ** 0.75
    rows = []
    for t in dial:
        ev = np.linalg.eigvalsh((t * signal + W) @ (t * signal + W).T / N_DIM)
        rows.append((ev[:-K], ev[-K:]))       # (bulk, top-3)

    _, _, Vt = np.linalg.svd(X / np.sqrt(P_DIM), full_matrices=False)
    v1, v2, v3 = Vt[0], Vt[1], Vt[2]
    if v1[:PER].mean() < 0:
        v1 = -v1
    if v2[PER:2 * PER].mean() < 0:
        v2 = -v2
    if v3[2 * PER:3 * PER].mean() < 0:
        v3 = -v3
    return scm, rows, v1, v2, v3


def make_cover() -> Path:
    rng = np.random.default_rng(SEED)
    scm, rows, v1, v2, v3 = simulate_story(rng)
    class_colors = [SLATE, BLUE, GRAY]

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(CREAM)

    def bare_axes(rect):
        ax = fig.add_axes(rect)
        ax.set_facecolor("none")
        ax.axis("off")
        return ax

    # ---- the four-panel comic strip ----------------------------------------
    # Four torn-paper panels with ink borders and soft shadows, read in comic
    # order: the matrix, the waterfall, the eigenvectors, and the embedding
    # as the oversized final splash panel. No arrows: the form carries the
    # direction.
    tear_rng = np.random.default_rng(77)

    def torn_card(x0, y0, x1, y1, ax_cards):
        """A torn-paper rectangle: jittered, lightly smoothed edges."""
        def side(p, q, n=42):
            t = np.linspace(0, 1, n)[:, None]
            base = (1 - t) * np.array(p) + t * np.array(q)
            normal = np.array([-(q[1] - p[1]), q[0] - p[0]], dtype=float)
            normal /= np.hypot(*normal)
            raw = tear_rng.normal(0, 1, n)
            kern = np.ones(5) / 5
            smooth = np.convolve(raw, kern, mode="same")
            amp = 0.0035
            jit = smooth * amp
            jit[0] = jit[-1] = 0.0
            return base + jit[:, None] * normal[None, :]
        verts = np.vstack([
            side((x0, y0), (x1, y0))[:-1],
            side((x1, y0), (x1, y1))[:-1],
            side((x1, y1), (x0, y1))[:-1],
            side((x0, y1), (x0, y0)),
        ])
        from matplotlib.patches import Polygon
        ax_cards.add_patch(Polygon(verts + np.array([0.0045, -0.0055]),
                                   closed=True, facecolor="#54504A",
                                   alpha=0.16, edgecolor="none", zorder=1))
        ax_cards.add_patch(Polygon(verts, closed=True, facecolor="#F9F4E7",
                                   edgecolor=INK, linewidth=1.15, zorder=2))

    ax_cards = fig.add_axes([0, 0, 1, 1])
    ax_cards.set_zorder(-3)
    ax_cards.set_facecolor("none")
    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    torn_card(0.065, 0.475, 0.445, 0.725, ax_cards)   # 1: the matrix
    torn_card(0.475, 0.475, 0.935, 0.725, ax_cards)   # 2: the waterfall
    torn_card(0.065, 0.045, 0.385, 0.445, ax_cards)   # 3: the eigenvectors
    torn_card(0.415, 0.045, 0.935, 0.445, ax_cards)   # 4: the splash panel

    def narrate(xc, y, text):
        fig.text(xc, y, text, style="italic", fontsize=9.5, color=INK,
                 alpha=0.82, ha="center", va="center")

    narrate(0.255, 0.700, "the data arrive looking like noise,")
    narrate(0.255, 0.681, "but three numbers stand out")
    narrate(0.705, 0.700, "turn the signal up: three peaks")
    narrate(0.705, 0.681, "break away from the noise")
    narrate(0.225, 0.422, "each peak carries a clue:")
    narrate(0.225, 0.403, "it singles out one group")
    narrate(0.675, 0.422, "put two clues together, point by point,")
    narrate(0.675, 0.403, "and the groups sort themselves out")

    # ---- panel 1 content: the LaTeX-typeset matrix (titlepage.tex) ----------

    # ---- panel 2 content: the waterfall, signal dialed up -------------------
    ax = bare_axes([0.493, 0.483, 0.425, 0.207])
    xg = np.linspace(0.01, 9.4, 700)
    edge_plus = (1 + np.sqrt(Y_ASPECT)) ** 2
    n_rows = len(rows)
    step, amp, xshift = 0.58, 1.42, 0.40
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
        ax.fill_between(xs, base, base + z, color="#F9F4E7", zorder=zo)
        ax.plot(xs, base + z, color=INK, lw=1.1, zorder=zo + 1)
    ax.set_xlim(-0.3, 9.4 + n_rows * xshift + 0.3)
    ax.set_ylim(-0.3, n_rows * step + 2.2)

    # ---- panel 3 content: the top three eigenvectors ------------------------
    idx = np.arange(N_DIM)
    lvl1, lvl2, lvl3 = np.abs(v1).max(), np.abs(v2).max(), np.abs(v3).max()
    for rect, vec, lvl, lab in [
        ([0.112, 0.290, 0.255, 0.088], v1, lvl1, r"$\widehat{v}_1$"),
        ([0.112, 0.176, 0.255, 0.088], v2, lvl2, r"$\widehat{v}_2$"),
        ([0.112, 0.062, 0.255, 0.088], v3, lvl3, r"$\widehat{v}_3$"),
    ]:
        ax = bare_axes(rect)
        for k, col in enumerate(class_colors):
            sl = slice(k * PER, (k + 1) * PER)
            ax.scatter(idx[sl], vec[sl], s=2.4, color=col, alpha=0.55,
                       linewidths=0)
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_xlim(-12, N_DIM + 12)
        ax.set_ylim(-1.3 * lvl, 1.3 * lvl)
        fig.text(rect[0] - 0.020, rect[1] + rect[3] / 2, lab, color=SLATE,
                 fontsize=11, ha="center", va="center")

    # ---- panel 4 content: the embedding, the splash -------------------------
    ax = bare_axes([0.445, 0.056, 0.455, 0.345])
    emb = np.c_[v1, v2] * np.sqrt(N_DIM)
    for k, col in enumerate(class_colors):
        pts = emb[k * PER:(k + 1) * PER]
        ax.scatter(pts[:, 0], pts[:, 1], s=9.0, color=col, alpha=0.55,
                   linewidths=0)
        mu = pts.mean(0)
        ax.plot(*mu, marker="+", ms=7, mew=1.4, color=INK)
    ax.set_aspect("equal")
    ax.margins(0.07)

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

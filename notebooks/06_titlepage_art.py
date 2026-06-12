"""06_titlepage_art.py

Generates the cover composition for the title page: three motifs from the
report's own simulations, set formally with generous whitespace.

  - top:          ridgeline of spiked-model eigenvalue densities as the spike
                  strength sweeps through the threshold (the outlier peak is
                  drawn at its simulated location with stylized height, since
                  one eigenvalue among p carries negligible density mass)
  - bottom-left:  spectral embedding of three clusters (the clustering model)
  - bottom-right: entries of the top Gram eigenvector for two classes

The middle of the page is left clear for the LaTeX title block, and a strip
at the foot for the one-line colophon; frontmatter/titlepage.tex overlays
the type.

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

Y_ASPECT = 0.5


def kde(x, grid, bw):
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


# ----------------------------------------------------------------------------
# Simulations
# ----------------------------------------------------------------------------

def ridgeline_rows(rng, n=600, p=300, reps=8, n_rows=15):
    """Bulk densities plus mean outlier location, alpha sweeping 1 -> 3.6."""
    alphas = np.linspace(1.0, 3.6, n_rows)
    grid = np.linspace(0.0, 5.4, 760)
    alpha_c = 1.0 + np.sqrt(Y_ASPECT)
    rows = []
    for a in alphas:
        bulk, top = [], []
        for _ in range(reps):
            d = np.ones(p)
            d[0] = a
            X = np.sqrt(d)[:, None] * rng.standard_normal((p, n))
            ev = np.linalg.eigvalsh(X @ X.T / n)
            bulk.append(ev[:-1])
            top.append(ev[-1])
        dens = kde(np.concatenate(bulk), grid, 0.09)
        dens = dens / dens.max()
        h = 0.18 + 0.85 * max(0.0, a - alpha_c) / (3.6 - alpha_c)
        dens = dens + h * np.exp(-0.5 * ((grid - np.mean(top)) / 0.055) ** 2)
        rows.append(dens)
    return grid, rows


def embedding_clouds(rng, K=3, per=500, p=500):
    """Top-two eigenvector embedding of three clusters (centered data)."""
    n = K * per
    ells = np.array([7.0, 4.0, 2.5])
    M = np.zeros((p, K))
    for k in range(K):
        M[k, k] = np.sqrt(ells[k] * n / per)
    J = np.zeros((n, K))
    for k in range(K):
        J[k * per:(k + 1) * per, k] = 1.0
    X = M @ J.T + rng.standard_normal((p, n))
    Xc = X - X.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc / np.sqrt(p), full_matrices=False)
    return Vt[:2].T * np.sqrt(n), per


def eigvec_bands(rng, ell=4.0, n=700, p=350):
    """Entries of the top Gram eigenvector, two balanced classes (+-mu)."""
    mu = rng.standard_normal(p)
    mu *= np.sqrt(ell) / np.linalg.norm(mu)
    s = np.repeat([1.0, -1.0], n // 2)
    X = np.outer(mu, s) + rng.standard_normal((p, n))
    _, _, Vt = np.linalg.svd(X / np.sqrt(p), full_matrices=False)
    v = Vt[0]
    if v[: n // 2].mean() < 0:
        v = -v
    zeta = 1.0 - (ell + Y_ASPECT) / (ell * (ell + 1.0))
    return v, n, zeta


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------

def make_cover() -> Path:
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(CREAM)

    def bare_axes(rect):
        ax = fig.add_axes(rect)
        ax.set_facecolor("none")
        ax.axis("off")
        return ax

    # ---------------- top: the ridgeline -----------------------------------
    grid, rows = ridgeline_rows(rng)
    ax = bare_axes([0.09, 0.685, 0.82, 0.255])
    step, amp = 0.50, 1.9
    N = len(rows)
    zmax = max(r.max() for r in rows)
    for i in range(N - 1, -1, -1):
        base = i * step
        z = rows[i] / zmax * amp * 1.9
        zo = 2 * (N - i)
        ax.fill_between(grid, base, base + z, color=CREAM, zorder=zo)
        ax.plot(grid, base + z, color=INK, lw=1.25, zorder=zo + 1)
    ax.set_xlim(-0.15, 5.5)
    ax.set_ylim(-0.35, N * step + 3.85)
    fig.text(0.155, 0.694, "the bulk", style="italic", fontsize=8.5, color=GRAY)
    fig.text(0.545, 0.910, "the spike detaches", style="italic", fontsize=8.5, color=GRAY)

    # ---------------- bottom-left: the embedding ---------------------------
    emb, per = embedding_clouds(rng)
    ax = bare_axes([0.10, 0.075, 0.20, 0.15])
    for k, col in enumerate([SLATE, BLUE, GRAY]):
        ax.scatter(emb[k * per:(k + 1) * per, 0], emb[k * per:(k + 1) * per, 1],
                   s=5, color=col, alpha=0.40, linewidths=0)
    ax.set_aspect("equal")
    fig.text(0.20, 0.062, "the spectral embedding", style="italic",
             fontsize=8, color=GRAY, ha="center")

    # ---------------- bottom-right: eigenvector bands ----------------------
    v, n, zeta = eigvec_bands(rng)
    ax = bare_axes([0.70, 0.085, 0.20, 0.125])
    idx = np.arange(n)
    half = n // 2
    ax.scatter(idx[:half], v[:half], s=3.5, color=SLATE, alpha=0.5, linewidths=0)
    ax.scatter(idx[half:], v[half:], s=3.5, color=GRAY, alpha=0.6, linewidths=0)
    lvl = np.sqrt(zeta / n)
    ax.plot([0, half], [lvl, lvl], color=INK, lw=1.3)
    ax.plot([half, n], [-lvl, -lvl], color=INK, lw=1.3)
    ax.set_xlim(-10, n + 10)
    ax.set_ylim(-3.4 * lvl, 3.4 * lvl)
    fig.text(0.80, 0.062, "the eigenvector knows the classes", style="italic",
             fontsize=8, color=GRAY, ha="center")

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

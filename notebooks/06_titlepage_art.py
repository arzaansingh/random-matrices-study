"""06_titlepage_art.py

Generates the full-page cover composition for the title page: a whiteboard
style amalgamation of the report's own simulations, in the house palette.

Motifs (all real simulations or exact formulas from the report):
  - top:          ridgeline of spiked-model eigenvalue densities as the spike
                  strength sweeps through the threshold (the outlier peak is
                  drawn at its simulated location with stylized height, since
                  one eigenvalue among p carries negligible density mass)
  - mid-left:     simulated Tracy-Widom edge fluctuations (null Wishart,
                  Johnstone standardization) with a standard normal ghost
  - mid-right:    the spike-location map Psi with the tangency at the
                  threshold
  - bottom-left:  spectral embedding of three clusters (the model of the
                  clustering section)
  - bottom-right: entries of the top Gram eigenvector for two classes, the
                  two-bands picture
  - scattered:    the report's signature formulas as faint annotations

Clear bands are reserved for the LaTeX title block (center) and the
colophon (bottom center); frontmatter/titlepage.tex overlays the type.

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


def kde(x, grid, bw):
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


# ----------------------------------------------------------------------------
# Simulations
# ----------------------------------------------------------------------------

def ridgeline_rows(rng, n=600, p=300, reps=8, n_rows=16):
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


def tw_sample(rng, p=200, n=400, reps=300):
    """Standardized null top eigenvalues (Johnstone centering and scaling)."""
    a = np.sqrt(n - 1) + np.sqrt(p)
    mu = a ** 2 / n
    sigma = (a / n) * (1.0 / np.sqrt(n - 1) + 1.0 / np.sqrt(p)) ** (1.0 / 3.0)
    tops = np.empty(reps)
    for r in range(reps):
        X = rng.standard_normal((p, n))
        tops[r] = np.linalg.eigvalsh(X @ X.T / n)[-1]
    return (tops - mu) / sigma


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
    ax = bare_axes([0.045, 0.665, 0.91, 0.30])
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
    ax.set_ylim(-0.35, N * step + 2.5)
    fig.text(0.115, 0.677, "the bulk", style="italic", fontsize=8.5, color=GRAY)
    fig.text(0.535, 0.958, "the spike detaches", style="italic", fontsize=8.5, color=GRAY)

    # ---------------- mid-left: Tracy-Widom edge ---------------------------
    tw = tw_sample(rng)
    ax = bare_axes([0.055, 0.305, 0.235, 0.145])
    xg = np.linspace(-4.6, 2.6, 300)
    dens = kde(tw, xg, 0.34)
    gauss = np.exp(-0.5 * xg ** 2) / np.sqrt(2 * np.pi)
    ax.plot(xg, gauss, color=GRAY, lw=1.0, ls=(0, (3, 2)))
    ax.fill_between(xg, 0, dens, color=SAGE, alpha=0.55)
    ax.plot(xg, dens, color=SLATE, lw=1.5)
    ax.set_xlim(-4.8, 2.8)
    ax.set_ylim(0, 0.52)
    fig.text(0.21, 0.428, r"$\mathrm{TW}_1$", fontsize=11, color=SLATE)
    fig.text(0.062, 0.292, "the edge law, simulated", style="italic", fontsize=8, color=GRAY)

    # ---------------- mid-right: the spike-location map --------------------
    ax = bare_axes([0.715, 0.30, 0.235, 0.155])
    al = np.linspace(1.15, 3.4, 300)
    psi = al + Y_ASPECT * al / (al - 1.0)
    lam_plus = (1 + np.sqrt(Y_ASPECT)) ** 2
    ac = 1 + np.sqrt(Y_ASPECT)
    ax.plot(al, psi, color=SLATE, lw=1.6)
    ax.axhline(lam_plus, color=GRAY, lw=0.9, ls=(0, (3, 2)))
    ax.plot([ac], [lam_plus], "o", ms=5.5, mfc=SAGE, mec=INK, mew=1.0, zorder=5)
    ax.set_xlim(1.05, 3.45)
    ax.set_ylim(2.4, 5.5)
    fig.text(0.905, 0.426, r"$\Psi(\alpha)$", fontsize=11, color=SLATE)
    fig.text(0.872, 0.336, r"$\lambda_{+}$", fontsize=9.5, color=GRAY)
    fig.text(0.722, 0.287, "the spike-location map", style="italic", fontsize=8, color=GRAY)

    # ---------------- bottom-left: the embedding ---------------------------
    emb, per = embedding_clouds(rng)
    ax = bare_axes([0.06, 0.075, 0.225, 0.165])
    for k, col in enumerate([SLATE, BLUE, GRAY]):
        ax.scatter(emb[k * per:(k + 1) * per, 0], emb[k * per:(k + 1) * per, 1],
                   s=6, color=col, alpha=0.40, linewidths=0)
    ax.set_aspect("equal")
    fig.text(0.067, 0.062, "the spectral embedding", style="italic", fontsize=8, color=GRAY)

    # ---------------- bottom-right: eigenvector bands ----------------------
    v, n, zeta = eigvec_bands(rng)
    ax = bare_axes([0.70, 0.08, 0.245, 0.14])
    idx = np.arange(n)
    half = n // 2
    ax.scatter(idx[:half], v[:half], s=4, color=SLATE, alpha=0.5, linewidths=0)
    ax.scatter(idx[half:], v[half:], s=4, color=GRAY, alpha=0.6, linewidths=0)
    lvl = np.sqrt(zeta / n)
    ax.plot([0, half], [lvl, lvl], color=INK, lw=1.4)
    ax.plot([half, n], [-lvl, -lvl], color=INK, lw=1.4)
    ax.set_xlim(-10, n + 10)
    ax.set_ylim(-3.4 * lvl, 3.4 * lvl)
    fig.text(0.707, 0.067, "the eigenvector knows the classes", style="italic",
             fontsize=8, color=GRAY)

    # ---------------- scattered equations ----------------------------------
    def eq(x, y, s, size, color=INK, alpha=0.55, rot=0.0):
        fig.text(x, y, s, fontsize=size, color=color, alpha=alpha,
                 rotation=rot, ha="center", va="center")

    # flanking the ridgeline base
    eq(0.10, 0.655, r"$\widehat{\Sigma}_n = \frac{1}{n} X X^{\top}$", 11.5, rot=1.5)
    eq(0.90, 0.655, r"$p/n \to y$", 10.5, color=SLATE, alpha=0.6, rot=-1.5)
    # the central cluster between TW and Psi
    eq(0.50, 0.415, r"$\alpha_c = 1 + \sqrt{y}$", 13.5, color=SLATE, alpha=0.65)
    eq(0.50, 0.355, r"$\rho(\alpha, y) = \frac{\alpha\,\Psi'(\alpha)}{\Psi(\alpha)}$",
       12, alpha=0.5, rot=-1.5)
    eq(0.50, 0.300, r"$\mathrm{accuracy} \approx \Phi\left(\sqrt{\zeta/(1-\zeta)}\,\right)$",
       11.5, alpha=0.5, rot=1.0)
    # the thin row above the bottom motifs
    eq(0.175, 0.262, r"$\lambda_{\pm} = (1 \pm \sqrt{y})^2$", 10, alpha=0.45, rot=-1.0)
    eq(0.50, 0.262, r"$X = M J^{\top} + W$", 10.5, alpha=0.5)
    eq(0.825, 0.262, r"$G = \frac{1}{p} X^{\top} X$", 10, alpha=0.45, rot=1.0)
    # beside the embedding
    eq(0.50, 0.2355, r"$\langle v_k, \widehat{v}_k \rangle^2 \to \zeta_k$", 9.5, alpha=0.45)

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

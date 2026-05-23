"""00_intro_lowdim_vs_highdim.py

Generates the motivating figure used in Section 1 of the report. Shows the
empirical eigenvalue density of a Gaussian sample covariance matrix with
population covariance the identity, in two regimes:

    Left  panel: p = 10,  n = 1000   (aspect ratio y = 0.01, low-dimensional)
    Right panel: p = 500, n = 1000   (aspect ratio y = 0.5,  high-dimensional)

The low-dimensional eigenvalues concentrate at the population value 1;
the high-dimensional eigenvalues spread into the Marchenko-Pastur bulk
on [(1 - sqrt(y))^2, (1 + sqrt(y))^2].

Run:
    python3 00_intro_lowdim_vs_highdim.py

Output:
    Thesis/figures/intro_lowdim_vs_highdim.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thesis_style import setup_plot, PALETTE

setup_plot()

# ============================================================================
# CONFIG
# ============================================================================

SEED = 2026

# Low-dimensional regime
P_LOW, N_LOW = 10, 1000

# High-dimensional regime
P_HIGH, N_HIGH = 500, 1000

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "figures" / "intro_lowdim_vs_highdim.png"


# ============================================================================
# HELPERS
# ============================================================================

def sample_covariance_eigenvalues(p: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return the p eigenvalues of (1/n) X X^T for X ~ N(0, I_p)^{otimes n}."""
    X = rng.normal(size=(p, n))
    S = (X @ X.T) / n
    return np.linalg.eigvalsh(S)


def marchenko_pastur_density(x: np.ndarray, y: float) -> np.ndarray:
    """Marchenko-Pastur density at x for aspect ratio y in (0, 1], assuming
    population covariance Sigma = I (so the variance scale is 1)."""
    lam_minus = (1.0 - np.sqrt(y)) ** 2
    lam_plus = (1.0 + np.sqrt(y)) ** 2
    out = np.zeros_like(x, dtype=float)
    inside = (x >= lam_minus) & (x <= lam_plus)
    xi = x[inside]
    out[inside] = np.sqrt((lam_plus - xi) * (xi - lam_minus)) / (2.0 * np.pi * y * xi)
    return out


# ============================================================================
# SIMULATE
# ============================================================================

rng = np.random.default_rng(SEED)

eigs_low = sample_covariance_eigenvalues(P_LOW, N_LOW, rng)
eigs_high = sample_covariance_eigenvalues(P_HIGH, N_HIGH, rng)

y_low = P_LOW / N_LOW
y_high = P_HIGH / N_HIGH


# ============================================================================
# PLOT
# ============================================================================

fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(10, 3.6))

# ---- Left panel: low-dimensional regime ----------------------------------
ax_low.hist(
    eigs_low,
    bins=8,
    density=True,
    color=PALETTE["def"],
    alpha=0.75,
    edgecolor="white",
    linewidth=0.6,
    label=f"Empirical eigenvalues ({P_LOW} values)",
)
ax_low.axvline(
    1.0,
    color=PALETTE["thm"],
    linestyle="--",
    linewidth=1.2,
    label=r"Population eigenvalue $\lambda = 1$",
)
ax_low.set_xlim(0, 3)
ax_low.set_xlabel(r"Eigenvalue $\lambda$")
ax_low.set_ylabel("Empirical density")
ax_low.set_title(rf"Low-dimensional: $p = {P_LOW}$, $n = {N_LOW}$ ($y = {y_low:g}$)")
ax_low.legend(loc="upper right", frameon=False)

# ---- Right panel: high-dimensional regime --------------------------------
ax_high.hist(
    eigs_high,
    bins=40,
    density=True,
    color=PALETTE["def"],
    alpha=0.75,
    edgecolor="white",
    linewidth=0.6,
    label=f"Empirical eigenvalues ({P_HIGH} values)",
)

x_grid = np.linspace(0.0, 3.0, 600)
ax_high.plot(
    x_grid,
    marchenko_pastur_density(x_grid, y_high),
    color=PALETTE["thm"],
    linewidth=1.6,
    label=r"Marchenko-Pastur density",
)
edge_minus = (1.0 - np.sqrt(y_high)) ** 2
edge_plus = (1.0 + np.sqrt(y_high)) ** 2
for edge in (edge_minus, edge_plus):
    ax_high.axvline(edge, color=PALETTE["accent"], linestyle=":", linewidth=1.0)

ax_high.set_xlim(0, 3)
ax_high.set_xlabel(r"Eigenvalue $\lambda$")
ax_high.set_ylabel("Empirical density")
ax_high.set_title(rf"High-dimensional: $p = {P_HIGH}$, $n = {N_HIGH}$ ($y = {y_high:g}$)")
ax_high.legend(loc="upper right", frameon=False)

fig.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH)
plt.close(fig)

print(f"Saved: {OUTPUT_PATH}")
print(f"  low-dim:  y = {y_low:.4f}, eigenvalues in [{eigs_low.min():.4f}, {eigs_low.max():.4f}]")
print(f"  high-dim: y = {y_high:.4f}, MP support [{edge_minus:.4f}, {edge_plus:.4f}]")
print(f"  high-dim eigenvalues in [{eigs_high.min():.4f}, {eigs_high.max():.4f}]")

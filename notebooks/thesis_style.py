"""thesis_style.py

Shared matplotlib style helper for every notebook in this repository.
Locks the six-color palette defined in style/colors.tex and applies
consistent typography, grid, and figure dimensions.

Usage at the top of every notebook:

    from thesis_style import setup_plot, PALETTE, PLOT_COLORS
    setup_plot()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Six-color palette, mirrored from Thesis/style/colors.tex
PALETTE = {
    "vocab":     "#D4E8C8",   # sage green: vocab highlight (light accents in plots)
    "def":       "#3A5A78",   # slate navy: primary data color
    "thm":       "#2D2D2D",   # charcoal: lines, text, primary contrast
    "intuition": "#FBF3D9",   # warm cream: subtle backgrounds
    "link":      "#3D7BAA",   # muted blue: secondary data color
    "accent":    "#8B8378",   # warm gray: tertiary, axes, low-emphasis
}

# Cycler for matplotlib: defaults to slate navy first, then warm gray, then
# muted blue, then sage. Other accent colors are pulled by name when needed.
PLOT_COLORS = [
    PALETTE["def"],
    PALETTE["accent"],
    PALETTE["link"],
    "#50A076",                # mint green (used sparingly for "success" lines)
]


def setup_plot():
    """Apply the thesis plot style. Idempotent."""
    mpl.rcParams.update({
        # Typography: serif body to match Computer Modern in the LaTeX doc.
        "font.family":       "serif",
        "font.serif":        ["Times", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,

        # Color cycle
        "axes.prop_cycle":   mpl.cycler("color", PLOT_COLORS),

        # Light grid, no top/right spines (clean academic plot look).
        "axes.grid":         True,
        "grid.linestyle":    ":",
        "grid.alpha":        0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,

        # Figure defaults: width matches a 6.5in text column at 12pt.
        "figure.figsize":    (6.5, 3.5),
        "figure.dpi":        100,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    })

"""BI10 colour palette and matplotlib style shared by every notebook, chart and dashboard of the team.

Usage in a notebook or script:

    import sys; sys.path.append("..")          # path to the repo root if needed
    from style.bi10_colors import BI_COLORS, HEALTH_BAND_COLORS, apply_style, style_axes, highlight_colors

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.0), layout="constrained")
    ax.bar(labels, values, color=highlight_colors(labels, focus="TW"))
    style_axes(ax)

Colour rules (see README.md in this folder):
- Grey (`context`) for everything that is only context; one accent marks the point of the chart.
- Orange = risk or the group of interest; teal = healthy / engaged; blue = neutral focus; navy = text and benchmarks.
- Segment colours are fixed: the same segment always has the same colour in every chart.
"""

from __future__ import annotations

import numpy as np

BI_COLORS = {
    "navy": "#16324F",     # titles, benchmark lines, "Very high" engagement
    "blue": "#0072B2",     # neutral focus, "Stable" / "Moderate"
    "teal": "#009E73",     # healthy / engaged / good outcome
    "orange": "#D55E00",   # risk / group of interest / bad outcome
    "gold": "#E69F00",     # watch / warning
    "slate": "#667085",    # neutral data series, no-history group
    "grid": "#D8E1E8",     # gridlines and axis lines
    "text": "#1F2933",     # body text and value labels
    "muted": "#667085",    # tick labels, secondary text
    "surface": "#FFFFFF",  # figure and axes background
    "context": "#C3CAD4",  # bars that are only context
}

# Fixed segment colours (Challenge 1: Financial Health bands).
HEALTH_BAND_COLORS = {
    "Stressed": BI_COLORS["orange"],
    "Watchlist": BI_COLORS["gold"],
    "Stable": BI_COLORS["blue"],
    "Healthy": BI_COLORS["teal"],
    "No credit history": BI_COLORS["slate"],
}

# Fixed segment colours (Challenge 2: Engagement bands, from orange to navy).
ENGAGEMENT_BAND_COLORS = {
    "Dormant": BI_COLORS["orange"],
    "Low": BI_COLORS["gold"],
    "Moderate": BI_COLORS["blue"],
    "High": BI_COLORS["teal"],
    "Very high": BI_COLORS["navy"],
}

# Order to use when a chart needs several categorical colours with no special meaning.
CATEGORICAL = [BI_COLORS[c] for c in ("navy", "blue", "teal", "gold", "orange", "slate", "context")]


def apply_style():
    """Apply the BI10 matplotlib / seaborn defaults. Call once at the top of a notebook."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="white", context="notebook", font_scale=1.0)
    except ImportError:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "figure.facecolor": BI_COLORS["surface"],
            "axes.facecolor": BI_COLORS["surface"],
            "axes.edgecolor": BI_COLORS["grid"],
            "axes.labelcolor": BI_COLORS["text"],
            "axes.titlecolor": BI_COLORS["navy"],
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlepad": 8,
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
            "xtick.color": BI_COLORS["muted"],
            "ytick.color": BI_COLORS["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
            "text.color": BI_COLORS["text"],
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def style_axes(axes, grid_axis="y"):
    """Light gridlines on the value axis only, no top/right spines, no tick marks."""
    for axis in np.atleast_1d(axes).ravel():
        axis.set_facecolor(BI_COLORS["surface"])
        axis.set_axisbelow(True)
        axis.grid(axis=grid_axis, color=BI_COLORS["grid"], linewidth=0.8)
        axis.grid(axis="x" if grid_axis == "y" else "y", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(BI_COLORS["grid"])
        axis.spines["bottom"].set_color(BI_COLORS["grid"])
        axis.tick_params(length=0, pad=5)
    return axes


def highlight_colors(labels, focus, focus_color=None, base=None):
    """Grey for context, one accent colour (orange by default) for the label(s) the chart is about."""
    focus = {focus} if isinstance(focus, str) else set(focus)
    return [(focus_color or BI_COLORS["orange"]) if label in focus else (base or BI_COLORS["context"]) for label in labels]

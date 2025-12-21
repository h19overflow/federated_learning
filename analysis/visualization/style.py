"""
Publication-quality matplotlib style configuration.

Sets up consistent styling for all analysis figures with
proper sizing, fonts, and resolution for academic papers.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


PUBLICATION_STYLE = {
    "font.size": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.framealpha": 0.9,
    "figure.titlesize": 18,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "errorbar.capsize": 3,
}

COLORS = {
    "centralized": "#2ecc71",
    "federated": "#3498db",
    "centralized_dark": "#27ae60",
    "federated_dark": "#2980b9",
    "highlight": "#e74c3c",
    "neutral": "#95a5a6",
    "background": "#ecf0f1",
}

FIGURE_SIZES = {
    "single_column": (3.5, 2.5),
    "double_column": (7.0, 3.5),
    "full_page": (7.0, 9.0),
    "square": (5.0, 5.0),
    "wide": (10.0, 4.0),
}


def set_publication_style() -> None:
    """Apply publication-quality style to matplotlib."""
    plt.rcParams.update(PUBLICATION_STYLE)


def get_color(name: str) -> str:
    """Get color by name."""
    return COLORS.get(name, COLORS["neutral"])


def get_figure_size(name: str) -> tuple:
    """Get figure size by name."""
    return FIGURE_SIZES.get(name, FIGURE_SIZES["double_column"])


def create_figure(
    size_name: str = "double_column",
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> tuple:
    """
    Create figure with publication styling.

    Args:
        size_name: Size preset name
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        **kwargs: Additional arguments for subplots

    Returns:
        Tuple of (figure, axes)
    """
    set_publication_style()
    figsize = get_figure_size(size_name)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def add_significance_stars(
    ax,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.02,
) -> None:
    """
    Add significance stars between two bars.

    Args:
        ax: Matplotlib axes
        x1: Left position
        x2: Right position
        y: Y position (top of bars)
        p_value: P-value for significance
        height: Height of bracket
    """
    if p_value >= 0.05:
        return

    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    else:
        stars = "*"

    bar_height = y * height
    bar_tips = y * 0.01

    ax.plot([x1, x1, x2, x2], [y + bar_tips, y + bar_height, y + bar_height, y + bar_tips],
            color="black", linewidth=1)
    ax.text((x1 + x2) / 2, y + bar_height, stars, ha="center", va="bottom", fontsize=14)

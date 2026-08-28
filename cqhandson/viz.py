"""Figure styling.

The palette encodes **role**, not author identity. Every chart already names the
author on its axis, so colour is free to carry the thing a reader must not
forget: three of these five authors *built* the benchmark, one is the human
reference, one is the model under test.

Slots come from a validated categorical palette (all-pairs CVD ΔE 9.2, normal
vision 24.0, both well clear of the floors). The aqua slot sits below 3:1
contrast on a white surface, so every chart that uses it ships direct value
labels — the documented relief.
"""
from __future__ import annotations

#: The three roles. Fixed order, never cycled.
ROLE_COLORS = {
    "reference":    "#1baf7a",   # aqua — the human reference
    "construction": "#2a78d6",   # blue — the three models that built the benchmark
    "under test":   "#eb6834",   # orange — the submission
}

#: A sequential ramp *within* the construction category, for the few charts that
#: must tell those three apart. Monotonic in lightness, always shipped with a
#: legend or direct labels.
CONSTRUCTION_RAMP = ["#89b6ee", "#2a78d6", "#17457f"]

#: Status colours, reserved: never reused as a series colour.
STATUS = {
    "good":     "#1baf7a",
    "warning":  "#eda100",
    "critical": "#e34948",
    "neutral":  "#9a9a93",
}

#: Sequential hue for magnitude heatmaps: one hue, light to dark.
SEQUENTIAL = "Blues"

INK = "#1a1a19"
MUTED = "#6b6b64"
GRID = "#e4e4de"


def author_color(author: str) -> str:
    """The colour for one author, by role."""
    from .loading import AUTHOR_ROLES
    return ROLE_COLORS[AUTHOR_ROLES[author]]


def author_colors(authors) -> list[str]:
    return [author_color(a) for a in authors]


def series_colors(authors) -> list[str]:
    """Colours for a chart that must tell the three construction models apart.

    Role still governs: the reference keeps aqua, the submission keeps orange,
    and the three construction models take three steps of one blue ramp — a
    sequential sub-encoding inside a single category, always shipped with a
    legend. Use this only where a reader must identify individual series;
    everywhere else `author_color` keeps the group flat, which is the message.
    """
    from .loading import AUTHOR_ROLES
    out, step = [], 0
    for author in authors:
        role = AUTHOR_ROLES[author]
        if role == "construction":
            out.append(CONSTRUCTION_RAMP[min(step, len(CONSTRUCTION_RAMP) - 1)])
            step += 1
        else:
            out.append(ROLE_COLORS[role])
    return out


def label_colors(labels) -> list[str]:
    """Same, but keyed by display label (what most tables are indexed by)."""
    from .loading import AUTHOR_LABELS, AUTHOR_ROLES
    reverse = {v: k for k, v in AUTHOR_LABELS.items()}
    return [ROLE_COLORS[AUTHOR_ROLES[reverse[str(l)]]] for l in labels]


def style() -> None:
    """Apply the session's matplotlib defaults. Call once per notebook."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "font.size": 10,
    })
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(ROLE_COLORS.values()))


def role_legend(axis, loc: str = "best") -> None:
    """One legend explaining the three colours. Identity is never colour-alone."""
    from matplotlib.patches import Patch
    axis.legend(handles=[Patch(facecolor=c, label=r) for r, c in ROLE_COLORS.items()],
                loc=loc, title=None)


# Backwards-compatible aliases used by earlier notebooks.
PALETTE = list(ROLE_COLORS.values())
AUTHOR_COLORS = ROLE_COLORS

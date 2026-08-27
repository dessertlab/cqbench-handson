"""Figure styling. The palette is the one used in the CQBench paper."""
from __future__ import annotations

PALETTE = ["#008080", "#40e0d0", "#afeeee", "#4169e1", "#87cefa", "#4682b4"]
HEATMAP_CMAP = "Blues"

#: Fixed colour per author so every figure in the session is comparable.
AUTHOR_COLORS = {
    "Human":           "#008080",
    "ChatGPT":         "#40e0d0",
    "DeepSeek-Coder":  "#4169e1",
    "Qwen2.5-Coder":   "#87cefa",
    "Claude Opus 4.8": "#4682b4",
}


def style() -> None:
    """Apply the session's matplotlib defaults. Call once per notebook."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    })
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE)


def author_palette(labels) -> list[str]:
    """Colours in the order of the labels you pass, falling back to the palette."""
    out = []
    for i, label in enumerate(labels):
        out.append(AUTHOR_COLORS.get(str(label), PALETTE[i % len(PALETTE)]))
    return out

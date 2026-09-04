"""Every chart the session shows, as one call each.

Why this file exists
--------------------
A notebook cell that reads

    frame.pivot_table(index="author", columns="odc", values="n", observed=True)

teaches nothing about code quality. It is plumbing. The plumbing lives here so
the notebooks can stay one line per idea; the functions are short and the file
is meant to be opened if you want to see how a number is made.

Design rules, applied throughout:

* **Colour carries role, not identity.** The author's name is already on the
  axis. Three of the five built the benchmark; colour says so in every chart.
* **One axis, ever.** No dual scales.
* **Direct labels on bars**, which is also the documented relief for the aqua
  slot's contrast on a white surface.
* **Sequential = one hue** (Blues) for magnitude grids; the categorical role
  palette is never used for magnitude.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import paths
from .loading import AUTHOR_LABELS, AUTHOR_ORDER, AUTHOR_ROLES, ROLE_SHORT
from .metrics import ODC_COLUMNS, ODC_LABELS, RATE_METRICS
from .viz import (CONSTRUCTION_RAMP, GRID, INK, MUTED, ROLE_COLORS, SEQUENTIAL,
                  STATUS, author_color, label_colors, role_legend, series_colors)


def _order(frame: pd.DataFrame) -> list[str]:
    present = [a for a in AUTHOR_ORDER if a in set(frame["author"])]
    return [AUTHOR_LABELS[a] for a in present]


def _save(fig, name: str | None):
    if name:
        paths.FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.FIGURES / f"{name}.png")
    return fig


def _bar_labels(axis, bars, fmt="{:.1f}", pad=0.8):
    for bar in bars:
        h = bar.get_height()
        axis.text(bar.get_x() + bar.get_width() / 2, h + pad, fmt.format(h),
                  ha="center", va="bottom", fontsize=9, color=INK)


# --------------------------------------------------------------------------- #
# 1. Structural validity — did the author answer the question at all?
# --------------------------------------------------------------------------- #
STATUS_ORDER = ["nontrivial", "target_missing", "arity_mismatch",
                "explicit_stub", "complexity_degenerate", "parse_error", "empty"]
STATUS_COLORS = {
    "nontrivial": STATUS["good"],
    "target_missing": "#4a4a45",
    "arity_mismatch": "#77776e",
    "explicit_stub": "#a3a399",
    "complexity_degenerate": "#c4c4bb",
    "parse_error": "#deded6",
    "empty": "#efefe9",
}
STATUS_TEXT = {
    "nontrivial": "answered the question",
    "target_missing": "wrote a different function",
    "arity_mismatch": "right name, wrong parameters",
    "explicit_stub": "a stub",
    "complexity_degenerate": "too small to count",
    "parse_error": "does not parse",
    "empty": "empty",
}


def plot_validity(results: pd.DataFrame, save: str | None = "validity"):
    """Where each author's 200 answers end up in the structural gate.

    One horizontal bar per author, split by outcome. Only the green segment
    reaches the quality analysis at all.
    """
    order = _order(results)
    table = pd.crosstab(results["author_label"], results["status"]).reindex(order)
    present = [s for s in STATUS_ORDER if s in table.columns]
    table = table[present] / 2.0                      # 200 tasks -> percent

    fig, axis = plt.subplots(figsize=(10, 3.4))
    left = np.zeros(len(table))
    for status in present:
        values = table[status].to_numpy()
        axis.barh(range(len(table)), values, left=left, height=0.62,
                  color=STATUS_COLORS[status], edgecolor="white", linewidth=1.6,
                  label=STATUS_TEXT[status])
        for y, (v, l) in enumerate(zip(values, left)):
            if v >= 7:
                axis.text(l + v / 2, y, f"{v:.0f}", ha="center", va="center",
                          fontsize=9, color="white", fontweight="bold")
        left += values

    axis.set_yticks(range(len(table)), table.index)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("% of the 200 tasks")
    axis.set_title("Structural validity: what happened to each author's answers")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=3)
    fig.tight_layout()
    return _save(fig, save)


# --------------------------------------------------------------------------- #
# 2. The scoreboard
# --------------------------------------------------------------------------- #
SCOREBOARD = [
    ("defective", "Has at least one defect", False),
    ("vulnerable", "Has at least one security finding", False),
    ("high_severity", "Has a high-severity finding", False),
    ("clean_strict", "Clean: passed all four layers", True),
]


def plot_scoreboard(results: pd.DataFrame, save: str | None = "scoreboard"):
    """The four headline rates, one small multiple each, coloured by role.

    A dashed line marks the human reference in every panel. It is a *reference*,
    not a target: three of these panels count problems, so being under the line
    is the good side there, and only the fourth rewards being above it. Every
    title says which way it runs, because a reader scanning four panels will
    otherwise carry one panel's direction into the next.
    """
    order = _order(results)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))

    for axis, (column, title, higher_is_better) in zip(axes.ravel(), SCOREBOARD):
        values = (results.groupby("author_label", observed=True)[column]
                  .mean().reindex(order) * 100)
        bars = axis.bar(range(len(values)), values.to_numpy(), width=0.62,
                        color=label_colors(values.index))
        _bar_labels(axis, bars, pad=max(values) * 0.03)
        axis.axhline(values["Human"], color=ROLE_COLORS["reference"],
                     ls="--", lw=1.5, zorder=0)
        axis.set_xticks(range(len(values)),
                        [l.replace(" Opus", "\nOpus").replace("2.5-", "2.5-\n")
                         for l in values.index], fontsize=8)
        axis.set_ylim(0, max(values) * 1.22)
        axis.set_ylabel("% of tasks")
        axis.set_title(title + ("   (higher is better)" if higher_is_better
                                else "   (lower is better)"))
        axis.grid(axis="x", visible=False)

    role_legend(axes[0][0], loc="upper left")
    fig.suptitle("")
    fig.tight_layout()
    return _save(fig, save)


# --------------------------------------------------------------------------- #
# 3. The forest plot — the session's most important picture
# --------------------------------------------------------------------------- #
def plot_forest(comparison: pd.DataFrame, title: str,
                subject: str = "Claude Opus 4.8", save: str | None = None,
                xlabel: str | None = None):
    """Paired differences with 95% intervals, and a line at zero.

    `comparison` is the output of `metrics.compare_submission`. A bar that
    crosses the zero line is a difference the data does not support.
    """
    data = comparison.copy()
    order = ["reference", "built it", "UNDER TEST"]
    data["_k"] = data["role"].map({r: i for i, r in enumerate(order)})
    data = data.sort_values(["_k", "delta"]).reset_index(drop=True)

    fig, axis = plt.subplots(figsize=(10, 0.52 * len(data) + 1.7))
    axis.axvline(0, color=INK, lw=1.4, zorder=1)

    # Fix the scale first, so every verdict can sit in one right-hand column
    # instead of trailing its own interval across the zero line.
    low, high = data["ci_lo"].min(), data["ci_hi"].max()
    span = max(high - low, 1e-6)
    axis.set_xlim(low - span * 0.08, low + span * 1.62)

    for y, row in data.iterrows():
        is_reference = row["role"] == "reference"
        color = (ROLE_COLORS["reference"] if is_reference
                 else ROLE_COLORS["construction"])
        axis.plot([row["ci_lo"], row["ci_hi"]], [y, y], color=color,
                  lw=4 if is_reference else 2.5, solid_capstyle="round", zorder=2)
        axis.plot(row["delta"], y, "o", color=color, markersize=11 if is_reference else 9,
                  markeredgecolor="white", markeredgewidth=1.6, zorder=3)
        verdict = "difference is real" if row["significant"] else "crosses zero"
        axis.annotate(f"{row['delta']:+.3f}    {verdict}",
                      xy=(0.72, y), xycoords=("axes fraction", "data"),
                      va="center", ha="left", fontsize=9,
                      color=INK if is_reference else MUTED,
                      fontweight="bold" if is_reference else "normal")

    labels = [f"− {row['baseline']}" for _, row in data.iterrows()]
    axis.set_yticks(range(len(data)), labels)
    for tick, role in zip(axis.get_yticklabels(), data["role"]):
        if role == "reference":
            tick.set_color(INK); tick.set_fontweight("bold")
    axis.invert_yaxis()
    axis.set_xlabel(xlabel or f"{subject} minus the baseline, paired over 200 tasks")
    axis.set_title(title)
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, save)


# --------------------------------------------------------------------------- #
# 4. RQ1 — structure and style
# --------------------------------------------------------------------------- #
def _paired(results: pd.DataFrame, column: str):
    """Per-task values and a strict-nontrivial mask, wide by author."""
    values = results.pivot_table(index="task_id", columns="author",
                                 values=column, observed=True)
    strict = results.pivot_table(index="task_id", columns="author",
                                 values="strict_nontrivial", observed=True,
                                 aggfunc="first").astype(bool)
    return values, strict


def plot_size_ratio(results: pd.DataFrame, save: str | None = "size_ratio"):
    """How big each model's answer is next to the human answer to the same task.

    Left: per task, model size against human size, with the identity line.
    Right: the distribution of the ratio in log2, so "half" and "double" sit
    symmetrically around zero.
    """
    values, strict = _paired(results, "complexity.nloc_mean")
    models = [a for a in ("chatgpt", "dsc", "qwen", "claude") if a in values.columns]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for author, colour in zip(models, series_colors(models)):
        both = strict["human"] & strict[author]
        axes[0].scatter(values.loc[both, "human"], values.loc[both, author],
                        s=22, alpha=0.5, color=colour, edgecolors="none",
                        label=AUTHOR_LABELS[author])
        axes[1].hist(np.log2(values.loc[both, author] / values.loc[both, "human"]),
                     bins=np.arange(-4, 4.25, 0.4), histtype="step", lw=2,
                     color=colour, label=AUTHOR_LABELS[author])

    limit = 70
    axes[0].plot([0, limit], [0, limit], "--", lw=1.4, color=INK, zorder=0)
    axes[0].annotate("same size as the human", (limit * 0.97, limit * 0.97),
                     ha="right", va="bottom", rotation=45, fontsize=8, color=MUTED)
    axes[0].set(xlim=(0, limit), ylim=(0, limit),
                xlabel="lines of code, human", ylabel="lines of code, model")
    axes[0].set_title("Every task, model size against human size")

    axes[1].axvline(0, color=INK, lw=1.4, zorder=0)
    axes[1].set(xlabel="← more compact      log2(model / human)      more verbose →",
                ylabel="tasks")
    axes[1].set_title("Distribution of the size ratio")
    axes[1].legend()
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, save)


STRUCTURAL_METRICS = {
    "complexity.nloc_mean": "lines of code",
    "complexity.ccn_mean": "cyclomatic complexity",
    "complexity.max_nesting_depth_mean": "nesting depth",
    "complexity.halstead_volume_mean": "Halstead volume",
    "complexity.halstead_difficulty_mean": "Halstead difficulty",
}


def plot_structural_profile(results: pd.DataFrame, save: str | None = "structural_profile"):
    """Each model's structural metrics as a fraction of the human's, same tasks.

    A dot at 1.0 means "structurally like the human implementation". Left of the
    line is compression, right is verbosity. Only tasks both authors completed
    are used, so every dot is a like-for-like comparison.
    """
    models = [a for a in ("chatgpt", "dsc", "qwen", "claude")
              if a in set(results["author"])]
    rows = []
    for column, label in STRUCTURAL_METRICS.items():
        values, strict = _paired(results, column)
        for author in models:
            both = strict["human"] & strict[author]
            rows.append({"metric": label, "author": author,
                         "ratio": values.loc[both, author].mean()
                                  / values.loc[both, "human"].mean(),
                         "n": int(both.sum())})
    data = pd.DataFrame(rows)

    fig, axis = plt.subplots(figsize=(9.5, 4.9))
    axis.axvline(1.0, color=ROLE_COLORS["reference"], lw=2, zorder=1)
    axis.annotate("the human implementation", (1.0, len(STRUCTURAL_METRICS) - 0.42),
                  xytext=(6, 0), textcoords="offset points", fontsize=8,
                  color=ROLE_COLORS["reference"], va="center")

    labels = list(STRUCTURAL_METRICS.values())
    offsets = np.linspace(-0.24, 0.24, len(models))
    for author, offset, colour in zip(models, offsets, series_colors(models)):
        subset = data[data["author"] == author].set_index("metric").reindex(labels)
        axis.plot(subset["ratio"], np.arange(len(labels)) + offset, "o",
                  color=colour, markersize=9,
                  markeredgecolor="white", markeredgewidth=1.4,
                  label=f"{AUTHOR_LABELS[author]}  (n={subset['n'].iloc[0]})",
                  linestyle="none", zorder=3)

    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("model value ÷ human value, on the tasks both completed")
    axis.set_title("Structural profile, relative to the human implementation")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=4, fontsize=8)
    fig.tight_layout()
    return _save(fig, save)


def plot_lexical(results: pd.DataFrame, authors=("human", "dsc", "qwen", "claude"),
                 save: str | None = "lexical"):
    """Distinct tokens against tokens written — vocabulary at matched volume.

    Comparing total vocabularies is unfair to whoever wrote less code. This
    curve removes that: read it vertically at the dashed line.
    """
    from .loading import load_results

    fig, axis = plt.subplots(figsize=(9, 4.4))
    curves, shades = {}, dict(zip(authors, series_colors(authors)))
    for author in authors:
        rows = sorted([r for r in load_results(author) if r["strict_nontrivial"]],
                      key=lambda r: r["task_id"])
        seen, volume, xs, ys = set(), 0, [], []
        for row in rows:
            seen |= set(row["lexical_tokens"])
            volume += row["token_count"]
            xs.append(volume); ys.append(len(seen))
        curves[author] = (np.array(xs), np.array(ys))
        axis.plot(xs, ys, color=shades[author], label=AUTHOR_LABELS[author])

    matched = min(xs[-1] for xs, _ in curves.values())
    axis.axvline(matched, color=INK, ls="--", lw=1.2, zorder=0)
    # Several authors land within a few tokens of each other at the matched
    # volume, so the value labels are stacked in a column to the right of the
    # marker rather than beside it, in descending order, each on its own row.
    landing = {a: float(np.interp(matched, xs, ys)) for a, (xs, ys) in curves.items()}
    span = axis.get_ylim()[1] - axis.get_ylim()[0]
    for rank, (author, at) in enumerate(sorted(landing.items(),
                                               key=lambda kv: -kv[1])):
        axis.plot(matched, at, "o", color=shades[author], markersize=9,
                  markeredgecolor="white", markeredgewidth=1.5, zorder=4)
        short = AUTHOR_LABELS[author].split()[0].replace("2.5-Coder", "2.5")
        axis.annotate(f"{short}  {at:.0f}",
                      (matched, axis.get_ylim()[1] * 0.95 - rank * span * 0.055),
                      xytext=(10, 0), textcoords="offset points",
                      fontsize=9, color=shades[author], va="center", zorder=5)
    axis.annotate("read the vocabularies here,\nat equal volume",
                  (matched, axis.get_ylim()[1] * 0.18), xytext=(-10, 0),
                  textcoords="offset points", ha="right", fontsize=8, color=MUTED)

    axis.set(xlabel="tokens written", ylabel="distinct tokens used")
    axis.set_title("Vocabulary, at matched volume")
    axis.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


# --------------------------------------------------------------------------- #
# 5. RQ2 — defects
# --------------------------------------------------------------------------- #
BURDEN_BINS = [(0, 0, "none"), (1, 1, "1"), (2, 2, "2"),
               (3, 4, "3–4"), (5, 99, "5 or more")]
BURDEN_COLORS = [STATUS["good"], "#bcd6f2", "#7fb0e4", "#3d84d8", "#17457f"]


def plot_defect_burden(results: pd.DataFrame, save: str | None = "defect_burden"):
    """How many defects a task carries, as a composition rather than a histogram."""
    order = _order(results)
    fig, axis = plt.subplots(figsize=(10, 3.4))
    left = np.zeros(len(order))
    for (lo, hi, label), colour in zip(BURDEN_BINS, BURDEN_COLORS):
        share = []
        for author_label in order:
            group = results[results["author_label"] == author_label]["defects_total"]
            share.append(100 * group.between(lo, hi).mean())
        share = np.array(share)
        axis.barh(range(len(order)), share, left=left, height=0.62, color=colour,
                  edgecolor="white", linewidth=1.6, label=label)
        for y, (v, l) in enumerate(zip(share, left)):
            if v >= 7:
                axis.text(l + v / 2, y, f"{v:.0f}", ha="center", va="center",
                          fontsize=9, color="white", fontweight="bold")
        left += share

    axis.set_yticks(range(len(order)), order)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("% of the 200 tasks")
    axis.set_title("Defects per task")
    axis.grid(axis="y", visible=False)
    axis.legend(title="defects on the task", loc="upper center",
                bbox_to_anchor=(0.5, -0.3), ncols=5)
    fig.tight_layout()
    return _save(fig, save)


def plot_odc_profile(results: pd.DataFrame, save: str | None = "odc_profile"):
    """What *kind* of mistake each author makes, as a magnitude grid.

    Sequential single hue: this is magnitude, not identity.
    """
    order = _order(results)
    table = pd.DataFrame({
        label: {ODC_LABELS[c]: 100 * (group[c] > 0).mean() for c in ODC_COLUMNS}
        for label, group in results.groupby("author_label", observed=True)
    }).T.reindex(order)

    fig, axis = plt.subplots(figsize=(9.5, 3.8))
    image = axis.imshow(table.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=65,
                        aspect="auto")
    axis.set_xticks(range(table.shape[1]), table.columns, rotation=18, ha="right")
    axis.set_yticks(range(table.shape[0]), table.index)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            v = table.iloc[r, c]
            axis.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=9,
                      color="white" if v > 36 else INK)
    axis.set_title("Defect types: % of tasks with at least one of this kind")
    axis.grid(visible=False)
    fig.colorbar(image, ax=axis, label="% of tasks", fraction=0.03, pad=0.02)
    fig.tight_layout()
    return _save(fig, save)


def plot_top_symbols(results: pd.DataFrame, top: int = 10,
                     save: str | None = "top_symbols"):
    """The actual Pylint checks behind the counts, per author."""
    from .loading import load_results

    counters = {a: collections.Counter(
        f["symbol"] for r in load_results(a) for f in r["defect_findings"])
        for a in AUTHOR_ORDER if a in set(results["author"])}
    names = sorted({s for c in counters.values() for s in c},
                   key=lambda s: -sum(c[s] for c in counters.values()))[:top]

    fig, axis = plt.subplots(figsize=(9.5, 0.44 * len(names) + 2.2))
    offsets = np.linspace(-0.28, 0.28, len(counters))
    shades = series_colors(list(counters))
    for (author, counter), offset, colour in zip(counters.items(), offsets, shades):
        axis.plot([counter[s] for s in names], np.arange(len(names)) + offset, "o",
                  color=colour, markersize=8, linestyle="none",
                  markeredgecolor="white", markeredgewidth=1.2,
                  label=AUTHOR_LABELS[author], zorder=3)
    axis.set_yticks(range(len(names)), names, fontfamily="monospace")
    axis.invert_yaxis()
    axis.set_xlabel("findings across the 200 tasks")
    axis.set_title("Which checks actually fire")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=5, fontsize=8)
    fig.tight_layout()
    return _save(fig, save)


# --------------------------------------------------------------------------- #
# 6. RQ3 — security
# --------------------------------------------------------------------------- #
CWE_NAMES = {
    "CWE-78": "OS command injection", "CWE-400": "resource exhaustion",
    "CWE-611": "XML external entity", "CWE-319": "cleartext transmission",
    "CWE-95": "eval injection", "CWE-330": "weak randomness",
    "CWE-89": "SQL injection", "CWE-939": "URL authorization",
    "CWE-79": "cross-site scripting", "CWE-116": "improper escaping",
}


def plot_cwe(results: pd.DataFrame, top: int = 8, save: str | None = "cwe"):
    """Where the security findings actually are, by weakness class."""
    records = [{"author": row["author"], "cwe": cwe}
               for _, row in results.iterrows() for cwe in (row["cwes"] or [])]
    counts = pd.DataFrame(records).value_counts(["cwe", "author"]).unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index].head(top)
    authors = [a for a in AUTHOR_ORDER if a in counts.columns]

    fig, axis = plt.subplots(figsize=(9.5, 0.52 * len(counts) + 2.3))
    offsets = np.linspace(-0.28, 0.28, len(authors))
    for author, offset, colour in zip(authors, offsets, series_colors(authors)):
        axis.plot(counts[author].to_numpy(), np.arange(len(counts)) + offset, "o",
                  color=colour, markersize=8, linestyle="none",
                  markeredgecolor="white", markeredgewidth=1.2,
                  label=AUTHOR_LABELS[author], zorder=3)
    axis.set_yticks(range(len(counts)),
                    [f"{c}\n{CWE_NAMES.get(c, '')}" for c in counts.index], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("findings across the 200 tasks")
    axis.set_title("Which weaknesses, by class")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=5, fontsize=8)
    fig.tight_layout()
    return _save(fig, save)


def _consensus_bars(series: pd.Series, xlabel: str, title: str, save: str | None):
    """One bar per author, the human's rate marked as the floor."""
    fig, axis = plt.subplots(figsize=(9.5, 3.6))
    bars = axis.barh(range(len(series)), series.to_numpy(), height=0.6,
                     color=label_colors(series.index))
    for bar, value in zip(bars, series):
        axis.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%",
                  va="center", fontsize=9, color=INK)
    axis.axvline(series["Human"], color=ROLE_COLORS["reference"], ls="--", lw=1.5,
                 zorder=0)
    axis.set_yticks(range(len(series)), series.index)
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="y", visible=False)
    role_legend(axis, loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def _consensus_rates(results: pd.DataFrame, tasks: dict, key: str, hit) -> tuple:
    """Per-author % of the gated tasks where `hit` fires, and the gate size."""
    from .loading import load_results

    consensus = {t: set(task["difficulty"].get(key) or ()) for t, task in tasks.items()}
    gate = [t for t, classes in consensus.items() if classes]
    values = {}
    for author in [a for a in AUTHOR_ORDER if a in set(results["author"])]:
        scored = {r["task_id"]: r for r in load_results(author)}
        values[AUTHOR_LABELS[author]] = 100 * sum(
            bool(hit(scored[t]) & consensus[t]) for t in gate) / len(gate)
    return pd.Series(values), len(gate)


def plot_consensus(results: pd.DataFrame, tasks: dict,
                   save: str | None = "consensus"):
    """Does the selection generalise beyond the models that made it?

    Each task carries the weakness class two 2023-24 models agreed on. This asks
    how often each author trips *that* class — the three that built the
    benchmark are the ceiling, the human is the floor.
    """
    series, n = _consensus_rates(
        results, tasks, "consensus_cwes", lambda row: set(row["cwes"]))
    return _consensus_bars(
        series,
        f"% of the {n} tasks that carry a weakness-class consensus",
        "Does the selection generalise? Tripping the task's own weakness class",
        save)


def plot_consensus_odc(results: pd.DataFrame, tasks: dict,
                       save: str | None = "consensus_odc"):
    """The same question on the defect side, using the agreed ODC class."""
    series, n = _consensus_rates(
        results, tasks, "consensus_odc",
        lambda row: {ODC_LABELS[c] for c in ODC_COLUMNS if (row.get(c) or 0) > 0})
    return _consensus_bars(
        series,
        f"% of the {n} tasks that carry a defect-class consensus",
        "Does the selection generalise? Tripping the task's own defect class",
        save)

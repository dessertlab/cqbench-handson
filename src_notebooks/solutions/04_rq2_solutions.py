# %% [markdown]
# # 04 — RQ2 exercises, worked

# %%
import sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cqhandson as ch
from cqhandson import paths
from cqhandson.metrics import ODC_COLUMNS, ODC_LABELS, paired_bootstrap_ci

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

results = ch.results_frame()
tasks = ch.load_tasks()
order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]

# %% [markdown]
# ## Exercise 1 — Defects per 100 NLOC

# %%
strict = results[results["strict_nontrivial"]]
density = (strict.groupby("author_label", observed=True)
           .agg(outputs=("task_id", "size"),
                total_nloc=("complexity.nloc_mean", "sum"),
                defects=("defects_total", "sum"))
           .reindex(order))
density["defects per 100 NLOC"] = 100 * density["defects"] / density["total_nloc"]
density["defects per task"] = density["defects"] / density["outputs"]
display(density.round(2))

# %% [markdown]
# **Answer — the ranking inverts at the top.**
#
# | | per task | per 100 NLOC |
# |---|---:|---:|
# | Human | 1.42 | **10.6** |
# | Claude Opus 4.8 | 1.20 | **9.1** |
# | Qwen2.5-Coder | 2.99 | 28.5 |
# | DeepSeek-Coder | 2.48 | 31.9 |
# | ChatGPT | 2.58 | 51.0 *(n = 12)* |
#
# Per task, Claude and the human are level. **Per line of code, Claude is
# cleaner than the human** — 9.1 findings per 100 NLOC against 10.6 — because it
# writes marginally more code for the same defect count.
#
# Two cautions before quoting that.
#
# 1. Per-task and per-line answer different questions. A reviewer integrating a
#    generated function cares about *per task* — will this one need fixing? A
#    maintainer inheriting a codebase cares about *per line*. Neither is the
#    right normalisation in general.
# 2. ChatGPT's 51.0 is computed on **12 surviving outputs totalling 61 lines**.
#    It is not a measurement, it is a rounding artifact with a decimal point.
#    Show the `n`.
#
# Note also that DeepSeek and Qwen swap places between the two columns: Qwen
# looks worse per task, DeepSeek worse per line. Any "which model is buggier"
# headline has to pick one and say so.

# %% [markdown]
# ## Exercise 2 — Remove the format artifacts

# %%
ARTIFACT_SYMBOLS = {"unused-argument", "too-few-public-methods",
                    "too-many-arguments", "too-many-positional-arguments"}

ODC_COLUMN_OF = {
    "Assignment": "def_assignment", "Algorithm": "def_algorithm",
    "Algorithm/Method": "def_algorithm", "Interface": "def_interface",
    "Checking": "def_checking", "Timing": "def_timing",
    "Timing/Serialization": "def_timing",
    "Function/Class/Object": "def_function_class_object",
}


def recount(author: str, excluded: set[str]) -> pd.DataFrame:
    """Re-derive per-task defect counts and ODC columns from the raw findings."""
    rows = []
    for row in ch.load_results(author):
        kept = [f for f in row["defect_findings"] if f["symbol"] not in excluded]
        counts = collections.Counter(ODC_COLUMN_OF[f["odc_category"]] for f in kept)
        rows.append({"task_id": row["task_id"], "defects_total": len(kept),
                     **{column: counts.get(column, 0) for column in ODC_COLUMNS}})
    return pd.DataFrame(rows).set_index("task_id").sort_index()

official, cleaned = {}, {}
for author in ch.AUTHOR_ORDER:
    official[author] = recount(author, set())
    cleaned[author] = recount(author, ARTIFACT_SYMBOLS)

comparison = pd.DataFrame({
    ch.AUTHOR_LABELS[a]: {
        "incidence, official": 100 * (official[a]["defects_total"] > 0).mean(),
        "incidence, de-artifacted": 100 * (cleaned[a]["defects_total"] > 0).mean(),
        "total, official": int(official[a]["defects_total"].sum()),
        "total, de-artifacted": int(cleaned[a]["defects_total"].sum()),
        "% of findings removed": 100 * (1 - cleaned[a]["defects_total"].sum()
                                        / official[a]["defects_total"].sum()),
    } for a in ch.AUTHOR_ORDER}).T.reindex(order)
display(comparison.round(1))

# %%
before = (pd.DataFrame({ch.AUTHOR_LABELS[a]: {ODC_LABELS[c]: 100 * (official[a][c] > 0).mean()
                                              for c in ODC_COLUMNS}
                        for a in ch.AUTHOR_ORDER}).T.reindex(order))
after = (pd.DataFrame({ch.AUTHOR_LABELS[a]: {ODC_LABELS[c]: 100 * (cleaned[a][c] > 0).mean()
                                             for c in ODC_COLUMNS}
                       for a in ch.AUTHOR_ORDER}).T.reindex(order))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for axis, table, title in zip(axes, (before, after),
                              ("official", "artifact symbols removed")):
    image = axis.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=65, aspect="auto")
    axis.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right", fontsize=8)
    axis.set_yticks(range(table.shape[0]), table.index)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            axis.text(c, r, f"{table.iloc[r, c]:.0f}", ha="center", va="center",
                      fontsize=8, color="white" if table.iloc[r, c] > 36 else "black")
    axis.set_title(f"ODC incidence — {title}")
fig.colorbar(image, ax=axes, label="% of tasks")
fig.savefig(paths.FIGURES / "04_solution_deartifacted.png", bbox_inches="tight")
plt.show()

# %%
for author in ("chatgpt", "dsc", "qwen", "claude"):
    a = (cleaned[author]["defects_total"] == 0).astype(float)
    b = (cleaned["human"]["defects_total"] == 0).astype(float)
    stats = paired_bootstrap_ci(a, b, label=f"clean:{author}")
    print(f"{ch.AUTHOR_LABELS[author]:16s} defect-free {stats['a']:.3f} "
          f"vs human {stats['b']:.3f}  delta {stats['delta']:+.3f} "
          f"CI [{stats['ci_lo']:+.3f}, {stats['ci_hi']:+.3f}]  "
          f"significant={stats['significant']}")

# %% [markdown]
# **Answer — the conclusions survive; the effect sizes shrink.**
#
# | | incidence official | incidence de-artifacted | findings removed |
# |---|---:|---:|---:|
# | Human | 62.0 | 41.0 | 35% |
# | ChatGPT | 68.0 | 58.5 | 25% |
# | DeepSeek-Coder | 89.5 | 64.5 | 49% |
# | Qwen2.5-Coder | 85.5 | 66.0 | 41% |
# | Claude Opus 4.8 | 63.0 | 42.0 | 47% |
#
# Four symbols account for **a third to a half of every author's findings**. That
# alone should make you cautious about any absolute defect rate from this
# pipeline.
#
# But the *comparisons* hold:
#
# * Human ≈ Claude before (62.0 / 63.0) and after (41.0 / 42.0), and the paired
#   test still finds no significant difference.
# * DeepSeek and Qwen remain significantly worse than the human, though the gap
#   narrows from ~27 points to ~24.
# * In the ODC heatmap, the Function/Class/Object column collapses for DeepSeek
#   (25.5% → ~1%) — confirming it was `too-few-public-methods` from synthetic
#   class wrappers, exactly as diagnosed. The **Assignment** gap shrinks but does
#   **not** disappear: `unused-variable` survives the filter, and Qwen still
#   assigns values it never reads far more often than the human.
#
# **The methodological point.** Absolute rates from a static-analysis benchmark
# are fragile — a defensible change to an exclusion list moved every one of them
# by 25–49%. The *paired differences between authors* barely moved, because every
# author meets the same artifacts on the same tasks. Design your benchmark so
# your claims live in the differences.

# %% [markdown]
# ## Exercise 3 — Does the ODC consensus generalise?

# %%
consensus = {t: task["difficulty"]["consensus_odc"] for t, task in tasks.items()}
gate_tasks = [t for t, classes in consensus.items() if classes]

rows = []
for author in ch.AUTHOR_ORDER:
    scored = {row["task_id"]: row for row in ch.load_results(author)}
    hits = sum(any(scored[t].get(ODC_COLUMN_OF[c], 0) > 0 for c in consensus[t])
               for t in gate_tasks)
    rows.append({"author": ch.AUTHOR_LABELS[author],
                 "trips the task's own ODC class %": 100 * hits / len(gate_tasks)})
print(f"{len(gate_tasks)} of 200 tasks carry an ODC consensus class\n")
display(pd.DataFrame(rows).set_index("author").round(1))

# %% [markdown]
# **Answer — it generalises, but weakly for defects.**
#
# | author | trips the task's own ODC class |
# |---|---:|
# | DeepSeek-Coder | 90.4% |
# | Qwen2.5-Coder | 86.4% |
# | ChatGPT | 67.2% |
# | Claude Opus 4.8 | **59.3%** |
# | Human | **50.8%** |
#
# DeepSeek and Qwen are near-ceiling because they *defined* these classes.
# Claude, which took no part in selection, is at 59.3% — above the human's 50.8%,
# but only by 8.5 points.
#
# Compare that with the **CWE** gate in notebook 05, where Claude trips the task's
# own class on 49% against the human's 22% — more than double. So the ODC
# consensus generalises much less strongly than the CWE consensus.
#
# A plausible reading: ODC defect classes are broad (six buckets, and "Interface"
# or "Assignment" catches a great deal), so a 50% human baseline is close to what
# chance would give on issue-prone code. CWE classes are narrow, so tripping the
# *same* one is far more informative. **A consensus gate is only as sharp as the
# taxonomy it is defined over** — worth knowing before you build one.

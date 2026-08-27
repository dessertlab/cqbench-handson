# %% [markdown]
# # 03 — RQ1 exercises, worked
#
# Run `03_rq1_structure_and_style.ipynb` first; this notebook only contains the
# answers, with the reasoning that matters.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cqhandson as ch
from cqhandson import paths
from cqhandson.metrics import compare_authors

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

results = ch.results_frame()
order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]

values = results.pivot_table(index="task_id", columns="author",
                             values="complexity.nloc_mean", observed=True)
strict = results.pivot_table(index="task_id", columns="author",
                             values="strict_nontrivial", observed=True,
                             aggfunc="first").astype(bool)

# %% [markdown]
# ## Exercise 1 — Is the structural-validity gap statistically real?

# %%
for metric in ("target_present_rate", "arity_ok_rate", "strict_nontrivial_rate"):
    table = compare_authors(results, metric, reference="human")
    table["author"] = table["author"].map(ch.AUTHOR_LABELS)
    print(f"\n{metric}")
    print(table.round(3).to_string(index=False))

# %% [markdown]
# **Answer.** Every model except Claude is significantly below the human
# reference on all three, and the intervals are nowhere near zero — ChatGPT's
# `target_present` deficit is −0.86 with a CI of roughly [−0.91, −0.81].
#
# Claude's `strict_nontrivial` delta is exactly **0.000 with a CI of [0, 0]**:
# it passed the structural gate on all 200 tasks, as did the human reference, so
# every paired difference is zero and the bootstrap has nothing to resample. A
# degenerate interval like that is a signal to check for a ceiling effect, not a
# spuriously precise result — with 200/200 on both sides there is simply no
# variation left to quantify.

# %% [markdown]
# ## Exercise 2 — Does verbosity depend on task size?

# %%
bins = pd.cut(values["human"], [0, 5, 10, 20, np.inf], right=False,
              labels=["<5", "5-9", "10-19", "20+"])

ratios = {}
for author in ("dsc", "qwen", "claude"):
    both = strict["human"] & strict[author]
    ratios[ch.AUTHOR_LABELS[author]] = (
        (values.loc[both, author] / values.loc[both, "human"])
        .groupby(bins[both], observed=True).mean())

table = pd.DataFrame(ratios)
table.insert(0, "tasks", bins.value_counts().sort_index())
display(table.round(2))

# %%
fig, ax = plt.subplots(figsize=(7.5, 4))
for label in ratios:
    ax.plot(range(len(table)), table[label], marker="o", lw=2, label=label,
            color=ch.AUTHOR_COLORS[label])
ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
ax.set_xticks(range(len(table)), table.index)
ax.set(xlabel="human NLOC on the task", ylabel="model NLOC / human NLOC",
       title="Verbosity is a small-task phenomenon; compression is a large-task one")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Answer — and it resolves the aggregation puzzle from section 3.**
#
# | human NLOC | tasks | DeepSeek | Qwen | Claude |
# |---|---:|---:|---:|---:|
# | `<5` | 31 | 2.30 | 2.64 | 2.78 |
# | `5–9` | 61 | 1.03 | 1.30 | 1.44 |
# | `10–19` | 72 | 0.57 | 0.81 | 1.09 |
# | `20+` | 36 | 0.36 | 0.47 | 0.64 |
#
# The ratio falls monotonically with task size for **every** model. On short
# functions all three write two to three times more than the human — docstrings,
# type checks, try/except scaffolding around a two-line job. On the longest
# functions all three write substantially *less* than the job needs.
#
# So "structural compression" and "LLM verbosity", usually reported as competing
# findings, are **the same phenomenon measured on different task mixes**. Models
# regress toward a preferred output length.
#
# That is exactly why `mean of ratios` (1.38 for Claude) and `ratio of means`
# (0.99) disagreed: the first is dominated by the 31 tiny tasks where the ratio
# is ~2.8; the second is dominated by the 36 large ones where it is 0.64. Report
# the *conditional* table and the paradox dissolves.

# %% [markdown]
# ## Exercise 3 — Is low lexical diversity just low volume?

# %%
def accumulation(author: str):
    """Distinct tokens seen as a function of cumulative tokens emitted."""
    rows = sorted([r for r in ch.load_results(author) if r["strict_nontrivial"]],
                  key=lambda r: r["task_id"])
    seen, volume, xs, ys = set(), 0, [], []
    for row in rows:
        seen |= set(row["lexical_tokens"])
        volume += row["token_count"]
        xs.append(volume)
        ys.append(len(seen))
    return np.array(xs), np.array(ys)

curves = {a: accumulation(a) for a in ("human", "dsc", "qwen", "claude")}
matched = min(xs[-1] for xs, _ in curves.values())

summary = pd.DataFrame({
    ch.AUTHOR_LABELS[a]: {
        "total volume (tokens)": int(xs[-1]),
        "distinct tokens (all)": int(ys[-1]),
        f"distinct at matched volume ({matched:,})": int(np.interp(matched, xs, ys)),
    } for a, (xs, ys) in curves.items()}).T
display(summary)

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
for author, (xs, ys) in curves.items():
    ax.plot(xs, ys, lw=2, label=ch.AUTHOR_LABELS[author],
            color=ch.AUTHOR_COLORS[ch.AUTHOR_LABELS[author]])
ax.axvline(matched, color="k", ls="--", lw=1, alpha=0.6)
ax.annotate("matched volume", (matched, 400), rotation=90,
            va="bottom", ha="right", fontsize=8, alpha=0.7)
ax.set(xlabel="cumulative tokens emitted", ylabel="distinct tokens seen",
       title="Type–token accumulation (ChatGPT omitted: only 12 surviving outputs)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Answer — the raw metric was measuring two different things.**
#
# At a matched volume of 12,134 tokens:
#
# | | distinct tokens, all output | distinct tokens, matched volume |
# |---|---:|---:|
# | Human | 3,451 | 2,151 |
# | DeepSeek-Coder | 2,149 | **2,149** |
# | Qwen2.5-Coder | 2,525 | **1,759** |
# | Claude Opus 4.8 | 3,227 | **2,211** |
#
# DeepSeek's 38% vocabulary deficit **vanishes** once volume is held constant
# (2,149 against the human's 2,151). It was never repetitive — it just wrote half
# as much code.
#
# Qwen's does **not** vanish: at equal volume it uses 18% fewer distinct tokens
# than the human. Qwen is genuinely more repetitive, reusing the same identifiers
# and constructs. Claude sits marginally *above* the human curve.
#
# One uncontrolled confounder turned a real finding about one model into a false
# finding about two. Corpus-level lexical diversity should not be reported
# without a volume control.

# %% [markdown]
# ## Exercise 4 — A claim and its objection
#
# One worked example of the shape the answer should take.
#
# > **Claim.** On 200 issue-prone Python tasks, Claude Opus 4.8 produces
# > structurally human-like code: on tasks both completed, its mean NLOC is 98.7%
# > and its mean Halstead volume 86.4% of the human implementation's, and at
# > matched volume its lexical diversity is indistinguishable from the human
# > corpus. The 2023–24 models do not: DeepSeek writes 61% of the human's lines
# > and Qwen is measurably more repetitive at equal volume.
#
# > **The objection that lands.** *"Both of your headline ratios are conditioned
# > on the output passing the structural gate, and the gate pass rate differs by
# > author — 200/200 for Claude, 165/200 for DeepSeek, 12/200 for ChatGPT. You are
# > comparing Claude's full distribution against DeepSeek's easiest 82% and
# > ChatGPT's easiest 6%. The comparison is only clean between Claude and the
# > human, where both pass everything."*
#
# That objection is correct and it is why the claim above names Claude and the
# human as the primary comparison and treats the rest as directional. Other
# objections worth having ready: failure-derived sampling (§4 of notebook 00),
# n = 200, and the task format (bare functions extracted from their classes).

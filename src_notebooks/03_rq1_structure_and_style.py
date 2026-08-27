# %% [markdown]
# # 03 — RQ1: Do models write structurally different code?
#
# **Time:** ~35 minutes · **Format:** worked analysis + `TODO` exercises
#
# > **RQ1.** Do code structural properties and style differ between human-written
# > and AI-generated code?
#
# Notebook 02 gave us verdicts. This one asks what the code actually *looks
# like*. Three passes:
#
# 1. **Structural validity** — did the author answer the question at all?
# 2. **Structural complexity** — size, control flow, Halstead, maintainability
# 3. **Style** — lexical diversity and naming
#
# Along the way you will meet the two traps that make this RQ harder than it
# looks: **survivorship** (you can only measure code that exists) and
# **aggregation** (two defensible averages, opposite conclusions).

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

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
tasks = ch.load_tasks()
print("source:", paths.results_dir().relative_to(paths.REPO), "|", len(results), "rows")

# %% [markdown]
# ## 1. Structural validity: did you answer the question?
#
# Before complexity, the mechanical check from notebook 01, now over 200 tasks.

# %%
status = pd.crosstab(results["author_label"], results["status"])
status = status.reindex([ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER])
display(status.fillna(0).astype(int))

rates = (results.groupby("author_label", observed=True)[
    ["parseable", "target_present", "target_matches_arity", "nonstub", "strict_nontrivial"]]
    .mean().reindex([ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]) * 100)
display(rates.round(1))

# %% [markdown]
# ### The ChatGPT result
#
# ChatGPT's `strict_nontrivial` rate is **6%**. That is not a code-quality
# result — its outputs parse 99% of the time. Look at the `target_present`
# column: on **86.5% of tasks it wrote a function with a different name**.
#
# Why so systematically? These tasks were mined from real repositories, so many
# of the requested signatures are **methods** — `def install_key(self, key_data)`,
# `def attach(self, engine, start=..., ...)`. `gpt-3.5-turbo` reliably rewrites
# them as free-standing functions with names it prefers.
#
# That single behaviour propagates into every downstream number: ChatGPT's
# `clean_strict@1` of 0.5% mostly measures **instruction-following**, not code
# quality. When a composite metric collapses, decompose it before you rank
# anything.

# %%
worst = results[results["author"] == "chatgpt"]
example = worst[worst["status"] == "target_missing"]["task_id"].iloc[3]
print("requested:", tasks[example]["signature"]["text"])
ch.show_code("chatgpt", example)

# %% [markdown]
# ### Trap 1 — survivorship
#
# Every complexity metric below is computed **only on outputs that passed the
# structural gate**, because you cannot measure the cyclomatic complexity of a
# function that does not exist. But the surviving sets are different sizes:

# %%
survivors = results[results["strict_nontrivial"]]
display(survivors.groupby("author_label", observed=True).size()
        .reindex([ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER])
        .rename("outputs that reach the complexity analysis").to_frame())

# %% [markdown]
# **12 outputs for ChatGPT against 200 for Claude.** And they are not a random
# 12: they are the tasks where the requested function happened to be a plain
# function with a memorable name — i.e. the *easy* ones. Any ChatGPT number
# below is measured on a biased survivor set and is reported for completeness
# only. This is why the paper reports structural metrics *and* validity rates
# together, and why we keep the `n` column visible.

# %% [markdown]
# ## 2. Structural complexity
#
# `lizard` gives, per function: NLOC, cyclomatic complexity (CCN), parameter
# count, max nesting depth, the Halstead family (volume ≈ information content,
# difficulty ≈ effort to understand), and the Maintainability Index (0–100,
# higher is nominally better).

# %%
METRICS = {
    "complexity.nloc_mean": "NLOC",
    "complexity.ccn_mean": "Cyclomatic",
    "complexity.parameter_count_mean": "Parameters",
    "complexity.max_nesting_depth_mean": "Max nesting",
    "complexity.halstead_volume_mean": "Halstead V",
    "complexity.halstead_difficulty_mean": "Halstead D",
    "complexity.maintainability_index_mean": "Maintainability",
}

order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]
profile = (survivors.groupby("author_label", observed=True)[list(METRICS)]
           .mean().rename(columns=METRICS).reindex(order))
profile.insert(0, "n", survivors.groupby("author_label", observed=True).size().reindex(order))
display(profile.round(2))

# %% [markdown]
# Read the Maintainability column with suspicion. ChatGPT scores **70.9**, the
# best of the five, and the human reference scores **59.1**, the worst. MI is
# largely a decreasing function of size — a five-line function is "more
# maintainable" than a thirteen-line one almost by definition. ChatGPT's high MI
# is a restatement of its low NLOC, on the twelve easy tasks it survived.
#
# A metric that rewards writing less is not a quality metric unless you hold the
# job constant. Which is what the next section does.

# %% [markdown]
# ## 3. Trap 2 — how you average changes the answer
#
# Every author faced the same tasks, so we can measure size **relative to the
# human implementation of the same task**. There are two obvious ways to do it,
# and they disagree:
#
# * **ratio of means** — average NLOC of the author ÷ average NLOC of the human,
#   over tasks both completed. This is what the paper reports.
# * **mean of ratios** — average, over tasks, of `author_NLOC / human_NLOC`.

# %%
def paired(metric: str) -> pd.DataFrame:
    """Both aggregations, on tasks where the author *and* the human are strict."""
    values = results.pivot_table(index="task_id", columns="author",
                                 values=metric, observed=True)
    strict = results.pivot_table(index="task_id", columns="author",
                                 values="strict_nontrivial", observed=True,
                                 aggfunc="first").astype(bool)
    rows = []
    for author in ("chatgpt", "dsc", "qwen", "claude"):
        both = strict["human"] & strict[author]
        mine, human = values.loc[both, author], values.loc[both, "human"]
        rows.append({
            "author": ch.AUTHOR_LABELS[author],
            "n tasks": int(both.sum()),
            "ratio of means": mine.mean() / human.mean(),
            "mean of ratios": float(np.nanmean(mine / human)),
            "median ratio": float(np.nanmedian(mine / human)),
        })
    return pd.DataFrame(rows).set_index("author")

print("NLOC, relative to the human implementation of the same task")
display(paired("complexity.nloc_mean").round(3))
print("Halstead volume, same tasks")
display(paired("complexity.halstead_volume_mean").round(3))

# %% [markdown]
# Look at Claude: **ratio of means 0.99, mean of ratios 1.38.** One says "Claude
# writes code the same size as the human"; the other says "Claude writes 38%
# more". Both are arithmetically correct.
#
# The reconciliation: `mean of ratios` weights every task equally, so a task
# where the human wrote 3 lines and Claude wrote 9 contributes a ratio of 3.0 —
# as much as a 60-line task where Claude matched exactly. `ratio of means` weights
# by size, so large tasks dominate. Claude *is* systematically more verbose on
# small functions (docstrings, input validation, error handling) while matching
# on large ones.
#
# The paper's 98.7% is the ratio of means. Neither aggregation is wrong; the
# claim you make has to name which one it is.

# %%
values = results.pivot_table(index="task_id", columns="author",
                             values="complexity.nloc_mean", observed=True)
strict = results.pivot_table(index="task_id", columns="author",
                             values="strict_nontrivial", observed=True,
                             aggfunc="first").astype(bool)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for author in ("dsc", "qwen", "claude"):
    both = strict["human"] & strict[author]
    axes[0].scatter(values.loc[both, "human"], values.loc[both, author], s=14, alpha=0.55,
                    label=ch.AUTHOR_LABELS[author],
                    color=ch.AUTHOR_COLORS[ch.AUTHOR_LABELS[author]])
    axes[1].hist(np.log2(values.loc[both, author] / values.loc[both, "human"]),
                 bins=np.arange(-4, 4.25, 0.4), histtype="step", lw=2,
                 label=ch.AUTHOR_LABELS[author],
                 color=ch.AUTHOR_COLORS[ch.AUTHOR_LABELS[author]])

limit = 80
axes[0].plot([0, limit], [0, limit], "k--", lw=1, alpha=0.5, label="same size as human")
axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel="human NLOC", ylabel="model NLOC",
            title="Per-task size against the human implementation")
axes[0].legend()

axes[1].axvline(0, color="k", ls="--", lw=1, alpha=0.5)
axes[1].set(xlabel="log2(model NLOC / human NLOC)", ylabel="tasks",
            title="Size ratio: left = more compact, right = more verbose")
axes[1].legend()

fig.tight_layout()
fig.savefig(paths.FIGURES / "03_size_ratio.png")
plt.show()

# %% [markdown]
# The left panel is the honest picture: DeepSeek sits mostly **below** the
# diagonal (structural compression — it writes less than the job needs), Claude
# straddles it, and the spread grows with task size.

# %% [markdown]
# ## 4. Style: vocabulary and naming
#
# Two cheap proxies for "does this code read like human code":
#
# * **corpus lexical diversity** — how many distinct identifiers, literals and
#   operators an author uses across all 200 answers. A model that reaches for the
#   same constructs every time has a smaller vocabulary.
# * **function name length** — templated names (`process_data`, `main`) are
#   shorter than the domain-specific names real codebases accumulate.

# %%
style_rows = []
for author in ch.AUTHOR_ORDER:
    rows = ch.load_results(author)
    vocabulary = {token for row in rows for token in row["lexical_tokens"]}
    names = [length for row in rows if row["strict_nontrivial"]
             for length in row["complexity"].get("function_name_lengths", [])]
    style_rows.append({
        "author": ch.AUTHOR_LABELS[author],
        "unique tokens (corpus)": len(vocabulary),
        "mean function-name length": float(np.mean(names)) if names else np.nan,
        "functions named": len(names),
    })
style = pd.DataFrame(style_rows).set_index("author")
display(style.round(2))

# %%
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(range(len(style)), style["unique tokens (corpus)"],
       color=[ch.AUTHOR_COLORS[a] for a in style.index])
ax.set_xticks(range(len(style)), style.index, rotation=20, ha="right")
ax.set(ylabel="distinct tokens", title="Lexical diversity over the same 200 tasks")
for x, value in enumerate(style["unique tokens (corpus)"]):
    ax.text(x, value + 40, f"{value:,}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(paths.FIGURES / "03_lexical_diversity.png")
plt.show()

# %% [markdown]
# The human corpus is the richest (3,451 distinct tokens); Claude comes closest
# (3,227); the 2023–24 models sit 20–29% below. Some of that gap is confounded
# with volume — writing fewer lines means emitting fewer tokens — which is
# exactly what exercise 3 asks you to check.

# %% [markdown]
# ---
# # Exercises
#
# Worked answers: `notebooks/solutions/03_rq1_solutions.ipynb`.

# %% [markdown]
# ### Exercise 1 — Is the structural-validity gap statistically real? (5 min)
#
# The `target_present` rates differ enormously across authors, but every author
# faced the same 200 tasks, so we can compare **paired**.
#
# **TODO:** use `compare_authors(results, metric, reference="human")` to get
# task-paired bootstrap intervals for `target_present_rate` and
# `strict_nontrivial_rate`. Which gaps exclude zero?
#
# *(Available metric names: `cqhandson.metrics.RATE_METRICS.keys()`)*

# %%
from cqhandson.metrics import RATE_METRICS
print(list(RATE_METRICS))

# TODO
# for metric in (...):
#     print(compare_authors(results, metric, reference="human"))

# %% [markdown]
# ### Exercise 2 — Does verbosity depend on task size? (10 min)
#
# Section 3 claimed Claude is verbose on small tasks and matches on large ones.
# Check it.
#
# **TODO:** bin tasks by human NLOC (say `<5`, `5–9`, `10–19`, `20+`), and for
# each bin report the mean `claude_NLOC / human_NLOC`. Then do the same for
# DeepSeek. Does either author's compression depend on task size?

# %%
# TODO
# sizes = values["human"]
# bins  = pd.cut(sizes, [0, 5, 10, 20, np.inf], right=False,
#                labels=["<5", "5-9", "10-19", "20+"])
# ...

# %% [markdown]
# ### Exercise 3 — Is low lexical diversity just low volume? (10 min)
#
# A model that writes less code necessarily emits fewer distinct tokens. Separate
# the two.
#
# **TODO:** compute, per author, the total number of token *occurrences*
# (`token_count` summed over strict-nontrivial outputs) and the number of
# *distinct* tokens. Then compare authors at **matched volume** — e.g. plot
# distinct tokens against cumulative token count as you add answers one by one
# (a type–token accumulation curve). Does the human curve stay above the models'
# at equal volume?
#
# *Hint: `row["lexical_tokens"]` is the per-task sorted set; accumulate with a
# running `set` while summing `row["token_count"]`.*

# %%
# TODO

# %% [markdown]
# ### Exercise 4 — Pick your own claim (10 min, discuss)
#
# **TODO:** choose one metric from `METRICS` and make the strongest *defensible*
# claim you can about human vs model code from these 200 tasks — then write the
# sentence that a reviewer would use to reject it. Candidate reviewer objections
# to defend against: survivorship, aggregation choice, failure-derived sampling,
# task format (bare functions out of their class), n = 200.

# %%
# TODO — your claim, and its strongest objection

# %% [markdown]
# ---
# ## Takeaways
#
# 1. **Structural validity is a separate axis from code quality.** ChatGPT's
#    collapse is an instruction-following result wearing a quality result's
#    clothes.
# 2. **Survivorship is unavoidable here** — you can only measure code that
#    exists — so report the surviving `n` next to every conditional metric.
# 3. **Ratio of means ≠ mean of ratios.** Name your aggregation, or your reader
#    will assume the one that favours your conclusion.
# 4. Claude closes the *structural* gap to human code that the 2023–24 models
#    left open — in size, control flow, and vocabulary. Whether it closes the
#    *quality* gap is notebooks 04 and 05.

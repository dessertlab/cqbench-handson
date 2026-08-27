# %% [markdown]
# # 04 — RQ2: Do defect types and frequencies differ?
#
# **Time:** ~30 minutes · **Format:** worked analysis + `TODO` exercises
#
# > **RQ2.** Do defect types and frequencies differ between human-written and
# > AI-generated code?
#
# Counting defects is easy and nearly useless on its own. The interesting
# question is *what kind of mistake* each author makes — which is why every
# Pylint finding is mapped to an **Orthogonal Defect Classification** category.
#
# ODC (Chillarege et al., IBM, 1992) classifies a defect by the nature of the fix
# it needs:
#
# | Category | The defect is about |
# |---|---|
# | **Assignment** | a value: wrong initialisation, unused or misassigned variable |
# | **Checking** | a missing or wrong validation, guard, or error check |
# | **Algorithm** | logic or efficiency inside a correct structure |
# | **Interface** | interaction with callers, parameters, or other modules |
# | **Function/Class/Object** | the structure of the unit itself is wrong |
# | **Timing/Serialization** | ordering, concurrency, resource lifetime |

# %%
import sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cqhandson as ch
from cqhandson import paths
from cqhandson.metrics import ODC_COLUMNS, ODC_LABELS, compare_authors, odc_profile

ch.style()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

results = ch.results_frame()
tasks = ch.load_tasks()
order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]

# %% [markdown]
# ## 1. How many, and how concentrated
#
# Incidence (share of tasks with ≥1 defect) and burden (how many, when there are
# any) answer different questions. Report both.

# %%
burden = results.groupby("author_label", observed=True).agg(
    incidence=("defective", lambda s: 100 * s.mean()),
    total=("defects_total", "sum"),
    mean_per_task=("defects_total", "mean"),
    median_when_defective=("defects_total", lambda s: s[s > 0].median()),
    worst_task=("defects_total", "max"),
).reindex(order)
display(burden.round(2))

# %% [markdown]
# Human and Claude are close on every column. DeepSeek and Qwen are not: Qwen
# accumulates 649 findings against the human's 284 on the same 200 tasks, and its
# worst single answer carries 46.
#
# ChatGPT's low total is an artifact you now recognise: 86% of its answers were
# a *different* function, usually a short one, so there was less code to find
# defects in. **A benchmark cannot distinguish "clean" from "absent" by counting.**

# %%
fig, ax = plt.subplots(figsize=(9, 4))
for author in ch.AUTHOR_ORDER:
    values = results.loc[results["author"] == author, "defects_total"]
    ax.hist(values, bins=np.arange(-0.5, 12.5, 1), histtype="step", lw=2,
            label=ch.AUTHOR_LABELS[author],
            color=ch.AUTHOR_COLORS[ch.AUTHOR_LABELS[author]])
ax.set(xlabel="defects on a task", ylabel="tasks",
       title="Defect burden per task (clipped at 12)")
ax.legend()
fig.tight_layout()
fig.savefig(paths.FIGURES / "04_defect_burden.png")
plt.show()

# %% [markdown]
# ## 2. The ODC profile: what *kind* of mistake
#
# Incidence per category — the share of tasks on which an author produced at
# least one defect of that type.

# %%
incidence = (odc_profile(results, "incidence").reindex(order) * 100)
display(incidence.round(1))

fig, ax = plt.subplots(figsize=(8.5, 4.5))
image = ax.imshow(incidence.to_numpy(), cmap="Blues", vmin=0, vmax=65, aspect="auto")
ax.set_xticks(range(incidence.shape[1]), incidence.columns, rotation=25, ha="right")
ax.set_yticks(range(incidence.shape[0]), incidence.index)
for row in range(incidence.shape[0]):
    for column in range(incidence.shape[1]):
        value = incidence.iloc[row, column]
        ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=9,
                color="white" if value > 36 else "black")
ax.set_title("ODC defect-type incidence (% of the 200 tasks)")
fig.colorbar(image, ax=ax, label="% of tasks")
fig.tight_layout()
fig.savefig(paths.FIGURES / "04_odc_heatmap.png")
plt.show()

# %% [markdown]
# Two patterns jump out.
#
# **Function/Class/Object.** Human 1.5%, Claude 1.0% — but DeepSeek **25.5%** and
# Qwen **20.5%**. That column is the signature of *templated* output. Asked for a
# method, these models wrap it in a synthetic class:
#
# ```python
# class MyClass:
#     def install_key(self, key_data):
#         ...
# ```
#
# which trips `too-few-public-methods` every time. It is scaffolding the model
# emitted to make its answer look complete.
#
# **Assignment.** Qwen 62%, DeepSeek 54.5%, against 25.5% for the human. Driven
# almost entirely by `unused-argument` and `unused-variable`: parameters accepted
# and never read, values computed and never used. Both are symptoms of code that
# is shaped like a solution without being one.

# %% [markdown]
# ## 3. Down to the actual findings
#
# Category counts hide their causes. `defect_findings` keeps every raw Pylint
# message, so we can look.

# %%
symbols = {}
for author in ch.AUTHOR_ORDER:
    counter = collections.Counter(
        finding["symbol"] for row in ch.load_results(author)
        for finding in row["defect_findings"])
    symbols[ch.AUTHOR_LABELS[author]] = counter

top = sorted({s for counter in symbols.values() for s in counter},
             key=lambda s: -sum(c[s] for c in symbols.values()))[:14]
table = pd.DataFrame({label: {s: counter[s] for s in top}
                      for label, counter in symbols.items()})[order]
display(table)

# %% [markdown]
# ### Three things in that table are worth stopping for
#
# **(a) Some defects are inherited from the task, not authored.**
# `too-many-arguments` and `too-many-positional-arguments`: Human 39/38, Claude
# 39/38 — *identical*. These fire because the **requested signature** has too many
# parameters. Every author who follows the contract inherits the finding; the
# only way to avoid it is to change the signature, which is what ChatGPT (14/14)
# does. The benchmark penalises obedience here.
#
# **(b) Some defects are artifacts of the task format.** `unused-argument` is the
# single largest category for Qwen (165) and DeepSeek (102), and it is inflated
# for everyone because methods are scored *outside their class*, so `self` is
# always unused. Real, but partly manufactured by the evaluation setup.
#
# **(c) And some are exactly the real-world risk the study is about.**
# `unspecified-encoding` (`open()` without an encoding — a portability bug that
# only shows up on someone else's machine) and `missing-timeout`
# (`requests.get()` with no timeout — a hang waiting to happen) are elevated for
# the models: ChatGPT 39 and 31, Qwen 49, against the human's 24 and near-zero.
# Nobody's test suite catches either one.

# %%
for symbol in ("unspecified-encoding", "missing-timeout", "too-few-public-methods"):
    example = next((row, finding) for author in ("qwen", "dsc", "chatgpt")
                   for row in ch.load_results(author)
                   for finding in row["defect_findings"] if finding["symbol"] == symbol)
    row, finding = example
    print(f"{symbol}  ({finding['odc_category']})  —  {finding['message']}")
    print(f"    task {row['task_id']}, line {finding['line']}\n")

# %% [markdown]
# ## 4. Is any of this significant?
#
# Task-paired bootstrap against the human reference.

# %%
paired = compare_authors(results, "defect_free_rate", reference="human")
paired["author"] = paired["author"].map(ch.AUTHOR_LABELS)
display(paired.round(3))

# %% [markdown]
# DeepSeek and Qwen are significantly worse than the human reference at producing
# defect-free code. ChatGPT and Claude are not distinguishable from it — for
# opposite reasons. Claude genuinely matches the human profile; ChatGPT mostly
# wrote something else.

# %% [markdown]
# ---
# # Exercises
#
# Worked answers: `notebooks/solutions/04_rq2_solutions.ipynb`.

# %% [markdown]
# ### Exercise 1 — Normalise for volume (10 min)
#
# Section 1 showed Qwen with 649 defects and ChatGPT with 296 — but they wrote
# very different amounts of code. A defect count is not comparable across authors
# unless you divide by opportunity.
#
# **TODO:** compute **defects per 100 NLOC** for each author, over
# strict-nontrivial outputs only (use `complexity.nloc_mean`). Does the ranking
# change? Which author looks worst per line rather than per task?

# %%
# TODO
# strict = results[results["strict_nontrivial"]]
# density = strict.groupby("author_label", observed=True).apply(...)

# %% [markdown]
# ### Exercise 2 — Remove the format artifacts (10 min)
#
# Section 3 identified three findings that are arguably manufactured by the
# evaluation setup rather than by the author: `unused-argument` (methods scored
# without their class), `too-few-public-methods` (synthetic class wrappers), and
# `too-many-arguments` / `too-many-positional-arguments` (inherited from the
# requested signature).
#
# **TODO:** recompute `defects_total` per author with those symbols excluded,
# starting from `defect_findings`. Report the before/after incidence and the
# before/after ODC profile. Which conclusions from section 2 survive?
#
# *This is the exercise that matters most. You are deciding what the benchmark
# should have counted, and finding out whether the paper's story depends on it.*

# %%
ARTIFACT_SYMBOLS = {"unused-argument", "too-few-public-methods",
                    "too-many-arguments", "too-many-positional-arguments"}

# TODO
# def recount(author, excluded):
#     ...
# compare the "official" and "de-artifacted" tables

# %% [markdown]
# ### Exercise 3 — Does the consensus hold? (10 min)
#
# Each task was selected partly because two 2023–24 models produced findings of a
# **shared ODC type**, recorded in `tasks[task_id]["difficulty"]["consensus_odc"]`.
# The paper's claim is that this selection *generalises*: a model that took no
# part in the selection still trips the same class.
#
# **TODO:** for the tasks that have a non-empty `consensus_odc`, compute the share
# on which **Claude** produced a defect of that same ODC category. Compare it with
# the human reference on the same tasks. Does selection generalise, or did the
# benchmark simply memorise its own construction set?
#
# *Hint: the per-task ODC counts are the `def_*` columns; map a consensus label
# like `"Checking"` to `"def_checking"`.*

# %%
consensus = {t: tasks[t]["difficulty"]["consensus_odc"] for t in tasks}
print("tasks with an ODC consensus class:",
      sum(1 for v in consensus.values() if v))
print("example:", [(k, v) for k, v in consensus.items() if v][:3])

# TODO

# %% [markdown]
# ---
# ## Takeaways
#
# 1. **Types beat counts.** The ODC profile separates "wrapped it in a pointless
#    class" from "forgot to validate input"; a defect total does not.
# 2. **Not every counted defect was authored.** Some are inherited from the
#    requested signature; some are manufactured by scoring methods outside their
#    class. Check which of your findings are which before you interpret them.
# 3. **A low defect count can mean "wrote less" or "wrote something else."**
#    Always read it next to the structural-validity rates.
# 4. On defect behaviour, the frontier model has converged on the human profile.
#    The 2023–24 models had a distinct, recognisable defect signature.

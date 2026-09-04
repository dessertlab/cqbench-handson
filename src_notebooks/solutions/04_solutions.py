# %% [markdown]
# # 04 — The exercise, worked
#
# Run `04_where_the_differences_are.ipynb` first. This notebook holds the worked
# answer to its exercise — and, more importantly, what that answer is evidence
# *for*.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))

import pandas as pd
import cqhandson as ch
from cqhandson import whatif

ch.style()
pd.set_option("display.width", 200)

results = ch.results_frame()
order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]

# %% [markdown]
# ## The exercise — where should the "too small to count" floor be?

# %%
clean = pd.DataFrame({
    f"gate {g:.2f}": 100 * whatif.with_complexity_gate(results, g)
        .groupby("author_label", observed=True)["clean_strict"].mean().reindex(order)
    for g in (0.02, 0.10, 0.30, 0.50)})
display(clean.round(1))

disqualified = pd.DataFrame({
    f"gate {g:.2f}": whatif.with_complexity_gate(results, g)
        .query("status == 'complexity_degenerate'")
        .groupby("author_label", observed=True).size().reindex(order).fillna(0).astype(int)
    for g in (0.10, 0.30, 0.50)})
print("answers disqualified as too small:")
display(disqualified)

# %% [markdown]
# **Answer — the gate is far less powerful than it looks, and for a good reason.**
#
# At the shipped 0.10, **nobody is disqualified**. That is not a coincidence: the
# benchmark's own construction already required the models to be
# complexity-qualified, so the tasks that survived selection are tasks where the
# models wrote something substantial.
#
# Raise it and the disqualifications appear, very unevenly:
#
# | | gate 0.30 | gate 0.50 |
# |---|---:|---:|
# | Human | 0 | 0 |
# | DeepSeek-Coder | 26 | 49 |
# | Qwen2.5-Coder | 11 | 36 |
# | Claude Opus 4.8 | 5 | 16 |
#
# The human is at 0 by construction — it *is* the denominator, so its ratio is
# always 1.0. DeepSeek is hit ten times harder than Claude, which is the same
# compression you saw in RQ1, now expressed as a pass/fail.
#
# **But look at the clean rates.** Claude goes 27.5 → 26.5 → 24.5. Almost
# nothing. Why? Because an answer small enough to be disqualified had usually
# already failed for another reason — it carried a defect or a finding. The gate
# and the analyzers are catching **the same bad answers twice**.
#
# The lesson generalises: when you add a filter to a composite metric, measure
# how much of it is *new* rejection rather than assuming it earns its place. This
# one is doing much less work than its prominence in the table suggests.

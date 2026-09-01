# %% [markdown]
# # 04 — Exercises, worked
#
# Run `04_where_the_differences_are.ipynb` first. This notebook holds the worked
# answer to its exercise and to the demo that follows it — and, more importantly,
# what each one is evidence *for*.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))

import pandas as pd
import cqhandson as ch
from cqhandson import figures, whatif
from cqhandson.metrics import compare_submission

ch.style()
pd.set_option("display.width", 200)

results = ch.results_frame()
order = [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER]

# %% [markdown]
# ## Exercise 1 — where should the "too small to count" floor be?

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

# %% [markdown]
# ## The demo — is the security gap carried by one rule?

# %%
filtered = whatif.without_rules(results, ("B404",))
display(pd.DataFrame({
    "vulnerable %, official": 100 * results.groupby("author_label", observed=True)["vulnerable"].mean(),
    "vulnerable %, no B404": 100 * filtered.groupby("author_label", observed=True)["vulnerable"].mean(),
}).reindex(order).round(1))

for metric, title in [("vulnerability_free_rate", "Free of security findings — B404 removed"),
                      ("high_severity_free_rate", "Free of high-severity findings — B404 removed")]:
    display(compare_submission(filtered, metric).round(3))
    figures.plot_forest(compare_submission(filtered, metric), title, save=None)

# %% [markdown]
# **Answer — and the answer is no, which is the more useful outcome.**
#
# **(1) Removing `B404` moves the levels, and almost only for the models.**
#
# | | official | without `B404` |
# |---|---:|---:|
# | Human | 15.5 | 15.0 |
# | ChatGPT | 48.5 | 46.0 |
# | DeepSeek-Coder | 54.0 | 50.0 |
# | Qwen2.5-Coder | 41.5 | 34.5 |
# | Claude Opus 4.8 | **28.0** | **23.0** |
#
# The human barely moves — one task. Every model drops, because models import
# `subprocess` far more often than humans do: `B404` fires 110 times across the
# five authors and **once** on the human.
#
# **(2) The conclusion narrows but survives.** The submission's gap against the
# human reference on overall vulnerability incidence goes from **−0.125,
# interval [−0.190, −0.060]** to **−0.080, interval [−0.140, −0.020]**. A third
# of the gap was the import rule. Two thirds were not, and the interval still
# clears zero.
#
# **(3) High severity does not move at all** — **−0.060, interval [−0.095,
# −0.025]**, identical before and after. `B404` is an advisory finding; it never
# enters the high-severity count. Those findings are `B410` (unsafe XML
# parsing) and `B602` (`shell=True`), rules that require actual misuse rather
# than an import.
#
# **Push harder and it still holds.** Removing the next-loudest rules as well
# does not close the gap — and can widen it, because the human trips those rules
# too, so the human benefits from their removal:
#
# | removed | delta | interval |
# |---|---:|---|
# | `B404` | −0.080 | [−0.140, −0.020] |
# | `B404` + `B113` | −0.095 | [−0.155, −0.040] |
# | `B404` + `B310` | −0.070 | [−0.130, −0.010] |
#
# **The lesson is that the loudest rule is not the load-bearing one.** `B404` is
# the most-fired rule against our submission and the most obviously permissive
# one in the set — an import is not a vulnerability. It is exactly the rule an
# author would remove if they wanted to make the result go away, and removing it
# leaves the conclusion standing.
#
# **What you are allowed to say afterwards** is more precise than the headline:
#
# > The frontier model's security gap against human code is not an artefact of
# > one permissive import-level rule: it narrows by about a third when that rule
# > is removed and remains significant, and the high-severity gap — unsafe XML
# > parsing and shell execution — does not move at all.
#
# You did not break the claim, and that is a result too. The transferable skill
# is the attempt: **the last step of an evaluation is attacking your own
# headline.** A claim that has survived a serious attempt to break it is worth
# more than one that was never tested — and had the attack succeeded, you would
# have wanted to know before a reviewer did.

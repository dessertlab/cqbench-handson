# %% [markdown]
# # 02 — Meet the cast, and score them
#
# **Time:** ~9 minutes · **You run cells and read charts. No code to write.**
#
# Two things happen here, and the first one governs how you read every chart for
# the rest of the day:
#
# 1. **who the five authors are** — and why they are not five peers;
# 2. **the benchmark runs** — 800 evaluations on your laptop, in about two
#    minutes.
#
# We score the **four authors the benchmark ships with**: the human reference
# and the three models that built it. Claude does not appear here — it arrives
# in notebook 03, as a submission the benchmark has never seen.

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import cqhandson as ch
from cqhandson import runner

ch.style()
pd.set_option("display.width", 200)
print(runner.check_analyzers())

# %% [markdown]
# ## 1. Who is in this benchmark, and in what capacity
#
# **The five authors are not five peers.** CQBench kept a task only when at
# least two of ChatGPT, DeepSeek-Coder and Qwen2.5-Coder produced three or more
# findings of a shared class. Those three *defined* the selection: their failure
# rates here are inflated by construction. They are a ceiling.
#
# The human reference never entered the consensus gate, and Claude Opus 4.8 was
# released after the benchmark was built. Both are outside the construction —
# which is the only reason the benchmark can test anything.

# %%
display(pd.DataFrame({
    "Author": [ch.AUTHOR_LABELS[a] for a in ch.AUTHOR_ORDER],
    "Role": [ch.ROLE_LABELS[ch.AUTHOR_ROLES[a]] for a in ch.AUTHOR_ORDER],
}).set_index("Author"))

# %% [markdown]
# Every chart in this session colours by that role. Aqua is the reference, blue
# is "built the benchmark", orange is the model under test. You will not have to
# remember which is which.

# %% [markdown]
# ## 2. Score the four authors the benchmark ships with
#
# One `pylint` run and one `semgrep` run per author, over 200 files each.
# About **two minutes** for all four.

# %%
started = time.time()
counts = runner.evaluate_all(ch.BASELINES)
print(f"\n{sum(counts.values())} evaluations in {time.time() - started:.0f}s")

# %% [markdown]
# > **If that failed**, nothing downstream breaks: every chart falls back to
# > `results/precomputed/`, produced by exactly this code.

# %%
results = ch.results_frame(ch.BASELINES)
display(ch.results_source(ch.BASELINES).to_frame())
display(ch.headline_table(results).round(1))

# %% [markdown]
# ### Reading the table
#
# Six numbers per author, and they do not all point the same way. Four of them
# are **percentages of the 200 tasks** — how *often* something happened. Two are
# **raw totals** — how *many* times in all.
#
# | column | what it counts | high means |
# |---|---|---|
# | `Defective %` | tasks where Pylint found **≥ 1** defect | **worse** |
# | `Vulnerable %` | tasks where Semgrep found **≥ 1** weakness | **worse** |
# | `High sev. %` | tasks with ≥ 1 weakness rated `CRITICAL` or `ERROR` | **worse** |
# | `Clean %` | tasks that passed **all four** layers | **better** |
# | `Total defects` | defects summed over all 200 tasks | **worse** |
# | `Total vulns` | weaknesses summed over all 200 tasks | **worse** |
#
# **`Clean %` is the one that runs the other way**, and it is the metric the
# benchmark leads with. It is a **conjunction**: a task counts as clean only if
# the answer is the requested function, reaches 10% of the human implementation's
# size, has zero defects **and** zero weaknesses. One failure anywhere and the
# task scores zero — which is why a low `Clean %` never tells you *which* layer
# said no.
#
# The four `%` columns and the two `Total` columns answer different questions,
# and the difference matters more than it looks. *"On what fraction of tasks did
# this happen"* is bounded between 0 and 100 and survives a reasonable
# disagreement about how findings are counted. *"How many in total"* has no upper
# bound, and moves whenever anyone changes their mind about when two findings are
# the same finding. **Prefer the bounded ones when you report.**

# %% [markdown]
# ---
# ## Takeaways
#
# 1. **Three of these five authors built the benchmark.** A task is here
#    precisely because at least two of them failed it in a shared way, so their
#    rates are a ceiling, not a measurement. Beating them is the weakest
#    possible result.
# 2. **The human reference is the number that means something.** It took no part
#    in the selection, and it is real code that shipped in real repositories.
# 3. **`Clean %` is a conjunction, and its first layer can veto.** ChatGPT's
#    0.5% is not a statement about its code quality — on 86% of tasks it wrote a
#    function with a different name, and a conjunction reports "did not follow
#    the format" and "wrote bad code" with the same number. Decompose before you
#    rank.
#
# Next: `03_evaluate_a_new_model.ipynb` — a model the benchmark has never seen.

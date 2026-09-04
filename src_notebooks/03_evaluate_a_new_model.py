# %% [markdown]
# # 03 — Evaluating a model the benchmark has never seen
#
# **This is the hands-on.**
#
# Notebook 02 scored the four authors the benchmark ships with, and showed how
# much of a count can hang on one line of the evaluator. Now we use the
# benchmark for what it was built for: **testing a model that took no part in
# its construction.**
#
# Our submission is **Claude Opus 4.8**, released after CQBench was built. Its
# 200 answers already sit in `data/predictions/claude.jsonl` — treat them as the
# output of a model you ran this morning. We follow the flow any CQBench user
# follows:
#
# > **validate** → **evaluate** → **compare**

# %%
import sys, pathlib, time, os, subprocess
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import pandas as pd
import cqhandson as ch
from cqhandson import paths, runner, figures
from cqhandson.metrics import compare_submission, paired_bootstrap_ci
from cqhandson.metrics import ODC_COLUMNS, ODC_LABELS

ch.style()
pd.set_option("display.width", 200)

ENVIRONMENT = dict(os.environ) | {"SEMGREP_ENABLE_VERSION_CHECK": "0"}

# %% [markdown]
# ## Step 1 — validate the submission
#
# Before any analyzer runs, CQBench checks the file itself: one JSON object per
# task, `task_id` as the only join key, no unknown ids, no duplicates. This is
# the stock CLI.

# %%
print(subprocess.run(
    [sys.executable, "-m", "cqbench", "validate-submission",
     "--tasks", str(paths.TASKS),
     "--predictions", str(paths.PREDICTIONS / "claude.jsonl")],
    cwd=paths.VENDOR, capture_output=True, text=True, env=ENVIRONMENT
).stdout.strip())

# %% [markdown]
# `{"tasks": 200, "predictions": 200}` — every task answered.
#
# Notice what validation does **not** do: it never looks at the code. A file of
# 200 empty strings validates fine. Missing answers are legal too — they stay in
# the denominator and score as empty output, so you cannot quietly shrink your
# own test set.

# %% [markdown]
# ## Step 2 — evaluate
#
# 200 tasks, one author, about **25 seconds**.

# %%
rows = runner.evaluate(paths.PREDICTIONS / "claude.jsonl",
                       output=paths.LIVE / "claude.jsonl")
print(f"\n{len(rows)} tasks scored")

# %% [markdown]
# That is the entire submission flow. If you had generated 200 completions from
# your own model this morning, you would now be done.

# %%
results = ch.results_frame()
display(ch.results_source().to_frame())

# %% [markdown]
# ## Step 3 — the first question: did it even answer?
#
# Before any quality judgement, a mechanical one. Every task names a function
# and its parameters; the benchmark checks that the answer contains *that*
# function, and that it is more than a stub.

# %%
figures.plot_validity(results);

# %% [markdown]
# **Read the ChatGPT row before anything else.** On 86% of tasks it wrote a
# function with a *different name*. Its code parses and is often reasonable — it
# simply is not the function that was asked for.
#
# Why so systematically? Many of these signatures are **methods** taken from
# real classes — `def install_key(self, key_data)`. `gpt-3.5-turbo` reliably
# rewrites them as free-standing functions it prefers.
#
# That single behaviour will drag every later ChatGPT number down. When a
# composite metric collapses, look here first: **you may be measuring
# instruction-following, not code quality.**
#
# Claude clears this gate on all 200 — and so does the human reference, but for
# a different reason. **The human's 100% is guaranteed by construction**: a task
# only entered CQBench if its human implementation parses, exposes the requested
# signature, and is structurally non-trivial. That bar is an entry requirement
# for the task, not a measurement of the author. Claude's 100% is a result; the
# human's is a tautology. Any chart that shows both has to be read that way.

# %% [markdown]
# ## Step 4 — the scoreboard
#
# Four rates, one panel each. The dashed line is the human reference in every
# panel: that is the bar a submission is trying to reach.

# %%
figures.plot_scoreboard(results);

# %% [markdown]
# The submission is clean on **27.5%** of these tasks. On its own that number
# says nothing — it needs the scale the other bars give it:
#
# * against the three models that **built** the benchmark (0.5%, 2.5%, 5.5%) it
#   looks transformative, and that comparison is close to worthless: these tasks
#   exist *because* those models failed them;
# * against the **human reference** (31.5%) it is slightly behind, and that is
#   the comparison that carries information.

# %% [markdown]
# ## Step 5 — compare, and ask whether the differences are real
#
# CQBench ships `cqbench compare`. It merges the submission with each baseline
# **task by task**, then bootstraps the paired difference 10,000 times.
#
# Pairing is the point: every author answered the *same* 200 tasks, so we
# subtract per task instead of comparing two independent percentages. That
# removes task difficulty from the comparison entirely.

# %%
print(subprocess.run(
    [sys.executable, "-m", "cqbench", "compare",
     "--submission", str(paths.LIVE / "claude.jsonl"),
     *sum([["--baseline", str(paths.results_dir() / f"{b}.jsonl")]
           for b in ch.BASELINES], []),
     "--output", str(paths.RESULTS / "comparison.csv")],
    cwd=paths.VENDOR, capture_output=True, text=True, env=ENVIRONMENT
).stdout.strip())

# %% [markdown]
# The numbers are easier to read as a picture. Each row is the submission minus
# one baseline; the dot is the difference, the bar is the 95% interval.
#
# **A bar that touches the vertical line at zero is a difference the data does
# not support.** That is the whole idea — you do not need to know what a
# confidence interval is to read it.

# %%
for metric, title in [
    ("clean_strict_at_1", "Clean code rate"),
    ("defect_free_rate", "Free of defects"),
    ("vulnerability_free_rate", "Free of security findings"),
    ("high_severity_free_rate", "Free of high-severity findings"),
]:
    figures.plot_forest(compare_submission(results, metric), title,
                        save=f"forest_{metric}")

# %% [markdown]
# Ignore the three blue rows in every chart — the submission beats the models
# that built the benchmark, as it must. **The aqua row is the result:**
#
# | | vs the human reference |
# |---|---|
# | clean code rate | **level** — the interval crosses zero |
# | free of defects | **level** — crosses zero |
# | free of security findings | **worse**, clearly |
# | free of high-severity findings | **worse**, clearly |
#
# A frontier model has closed the maintainability gap to human code on these
# tasks and has **not** closed the security gap. Whatever changed between 2023
# and 2026 fixed one and not the other.

# %% [markdown]
# ## Step 6 — does the benchmark still bite?
#
# The sharpest objection to a failure-derived benchmark: *you built it from
# three models' failures, so of course those three fail it. Does it have any
# force against a model that took no part?*
#
# Each task was kept because two 2023–24 models failed it **in the same way**,
# and that shared way was recorded at selection time — years before Claude
# existed. It was recorded twice, once per side of the pipeline:
#
# * a **consensus weakness class**, a specific CWE — 100 of our tasks carry one;
# * a **consensus defect class**, a specific ODC category — 177 carry one.
#
# So the question is not "does Claude produce findings". It is: **does Claude
# trip the class the task itself predicted?**

# %%
tasks = ch.load_tasks()
figures.plot_consensus(results, tasks);

# %% [markdown]
# The three construction models sit near the ceiling, as they must — they
# defined these classes. The **human reference**, also outside the construction,
# trips them on 22%.
#
# **Claude: 49%.** More than double the human rate on the same tasks. The tasks
# elicit not merely findings but the *same class* of finding from a model
# outside the selection.
#
# Now the same question on the defect side.

# %%
figures.plot_consensus_odc(results, tasks);

# %% [markdown]
# A different picture. The construction models are still near the ceiling — 90%
# and 86% — but the **human is at 51%**, not 22%, and **Claude is at 59%**.
#
# Do not read those two bars by eye. Both figures compare authors on the same
# tasks, so pair them and put an interval on the difference:

# %%
def consensus_gap(key, hit, submission="claude", reference="human"):
    """Claude minus human on the tasks that carry a class of this kind."""
    consensus = {t: set(task["difficulty"].get(key) or ()) for t, task in tasks.items()}
    gate = sorted(t for t, classes in consensus.items() if classes)

    def vector(author):
        scored = {r["task_id"]: r for r in ch.load_results(author)}
        return [float(bool(hit(scored[t]) & consensus[t])) for t in gate]

    return paired_bootstrap_ci(vector(submission), vector(reference),
                               label=f"consensus:{key}")


ODC_HIT = lambda row: {ODC_LABELS[c] for c in ODC_COLUMNS if (row.get(c) or 0) > 0}

for key, hit, label in [("consensus_cwes", lambda row: set(row["cwes"]), "weakness class (CWE)"),
                        ("consensus_odc", ODC_HIT, "defect class (ODC)")]:
    stats = consensus_gap(key, hit)
    print(f"{label:<22} n={stats['n']:>4}   Claude {100 * stats['a']:>3.0f}%   "
          f"human {100 * stats['b']:>3.0f}%   gap {stats['delta']:+.3f}  "
          f"[{stats['ci_lo']:+.3f}; {stats['ci_hi']:+.3f}]")

# %% [markdown]
# Both gaps are real — neither interval touches zero — but they are nowhere near
# the same size, and that difference is the finding.
#
# **The security side of the selection transfers; the defect side barely does.**
# On CWEs the task predicts a frontier model's weakness more than twice as often
# as it predicts the human's. On ODC classes the same tasks separate Claude from
# the human by eight points, and only just.
#
# That is not a failure of the benchmark — it is a finding about *which half of
# it generalises*. Two honest readings, and they are worth arguing about:
#
# * the ODC classes are coarse (five buckets in practice) and the commonest one,
#   Assignment, is easy to trip by accident, so agreement on it is cheap;
# * or a defect-prone task really is less transferable than a security-prone
#   one, because a security weakness follows from *how you solve* the problem
#   while a defect follows from how carefully you wrote it.
#
# Either way the answer to the objection is: **yes, but unevenly.** The tasks
# retain predictive force against a model that took no part in building them,
# clearly on security and weakly on defects. State it that way and nobody can
# take it apart.
#
# ---
# ## Takeaways
#
# 1. **Validate, evaluate, compare.** Three commands, and the third is the one
#    that turns a number into a claim.
# 2. **Beating the models that built the benchmark is not a result.** Only the
#    human row is news.
# 3. **A bar that crosses zero is not a difference.** The forest plot says it
#    without any statistics vocabulary.
# 4. **A composite metric can collapse for the wrong reason** — check structural
#    validity before you rank anything.
# 5. **A failure-derived benchmark has to prove it still bites.** Ours does, on
#    security clearly and on defects only weakly — and saying which is which is
#    part of the result.
#
# Next: `04_where_the_differences_are.ipynb` — what the code actually looks like.
